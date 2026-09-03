"""User labels on events — the ``key=value`` vocabulary carried in the frozen ``flag`` column.

The design bet is that labels belong in ``flag`` rather than in a new column, because tokens
outside DERIVED_FLAGS are already inherited verbatim through merges, resizes and hand-adds.
These tests exist to hold that bet: if the inheritance rule ever changes, the merge test here
fails before a user's molecule assignment quietly disappears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import (
    FLAG_CLIPPED, FLAG_CURATED, FLAG_NON_FINITE, combine_flags, inherited_curation,
)
from nanoscope.analysis.labels import (
    LabelEntry, LabelVocabulary, apply_label, clear_label, clear_labels, format_labels,
    label_column, label_keys, parse_labels, set_label, validate_label,
    validate_label_hotkey,
)


# ================================================================================================
# the token form
# ================================================================================================

def test_a_label_never_disturbs_the_flags_already_on_the_row():
    """Writing over the cell instead of combining into it is the exact defect FLAG_CURATED's
    docstring records. A label must add a token, not replace the contents."""
    start = combine_flags(FLAG_NON_FINITE, FLAG_CURATED)
    out = set_label(start, "molecule", "apo-ferritin")
    assert "non_finite" in out and "flagged" in out
    assert parse_labels(out) == {"molecule": "apo-ferritin"}


def test_setting_a_key_twice_replaces_rather_than_appends():
    """One key holds one value: a row must never claim two molecules at once."""
    out = set_label(set_label("", "molecule", "apo"), "molecule", "holo")
    assert parse_labels(out) == {"molecule": "holo"}
    assert out.count("molecule") == 1


def test_many_keys_coexist_on_one_event():
    out = ""
    for k, v in (("molecule", "apo-ferritin"), ("type", "translocation"), ("levels", "3")):
        out = set_label(out, k, v)
    assert parse_labels(out) == {"molecule": "apo-ferritin", "type": "translocation", "levels": "3"}


def test_bare_flags_are_not_labels_and_labels_are_not_flags():
    """The two vocabularies share a column and must not read as each other. Nothing in
    DERIVED_FLAGS contains '=' — ``dwell>max_dwell_s`` is the closest and uses '>'."""
    assert parse_labels("non_finite;dwell>max_dwell_s;flagged") == {}
    assert parse_labels("molecule=apo") == {"molecule": "apo"}


def test_a_pre_label_project_reads_back_unchanged():
    for legacy in ("", "flagged", "non_finite;clipped_by_boundary", None):
        assert parse_labels(legacy) == {}


@pytest.mark.parametrize("key, value", [
    ("mol;ecule", "apo"), ("molecule", "a;b"), ("mol=ecule", "apo"),
    ("molecule", "a=b"), ("", "apo"), ("molecule", ""), ("  ", "apo"),
])
def test_a_separator_in_a_label_is_refused_not_sanitised(key, value):
    """A label quietly rewritten will not match the one typed yesterday, and matching is the
    entire point of a vocabulary."""
    with pytest.raises(ValueError):
        validate_label(key, value)
    with pytest.raises(ValueError):
        set_label("", key, value)


def test_clearing_a_key_leaves_everything_else():
    f = set_label(set_label(FLAG_CURATED, "molecule", "apo"), "type", "collision")
    out = clear_label(f, "molecule")
    assert parse_labels(out) == {"type": "collision"}
    assert "flagged" in out
    assert clear_label(out, "nosuchkey") == out       # unknown key is a no-op


# ================================================================================================
# THE BET: labels survive the operations that rebuild a row
# ================================================================================================

def test_labels_survive_a_merge_because_they_are_not_derived_flags():
    """The reason this feature needed no merge code at all.

    A merge destroys rows and builds one. ``inherited_curation`` unions every token outside
    DERIVED_FLAGS, so both fragments' labels come through, while ``clipped_by_boundary`` is
    correctly dropped to be re-derived from the new span.
    """
    a = set_label(FLAG_CURATED, "molecule", "apo-ferritin")
    b = set_label(FLAG_CLIPPED, "type", "collision")
    flag, accepted = inherited_curation([a, b], [True, True])

    assert parse_labels(flag) == {"molecule": "apo-ferritin", "type": "collision"}
    assert "flagged" in flag
    assert FLAG_CLIPPED not in flag, "a derived flag must be re-derived, not inherited"
    assert accepted is True


def test_a_hand_rejection_still_survives_alongside_a_label():
    """Labelling must not weaken the curation guarantee: one rejected fragment rejects the
    merge, label or no label."""
    a = set_label(FLAG_CURATED, "molecule", "apo")
    flag, accepted = inherited_curation([a], [False])
    assert accepted is False
    assert parse_labels(flag) == {"molecule": "apo"}


# ================================================================================================
# reading labels back off a frame
# ================================================================================================

def _frame(flags):
    return pd.DataFrame({"event_id": np.arange(len(flags)), "flag": list(flags)})


def test_label_column_treats_unlabelled_as_a_category_not_a_hole():
    """Empty string, not NaN — the table, the exports and groupby all need "unlabelled" to
    be a value you can count."""
    got = label_column(["molecule=apo", "", "flagged", "molecule=holo"], "molecule")
    assert list(got) == ["apo", "", "", "holo"]
    assert not any(isinstance(v, float) for v in got)


def test_label_keys_lists_what_is_actually_in_use():
    assert label_keys(["molecule=apo;type=x", "levels=3", "flagged"]) == ["molecule", "type", "levels"]


def test_apply_label_writes_the_shared_frame_in_place_and_counts_changes():
    df = _frame(["", FLAG_CURATED, "molecule=apo"])
    changed = apply_label(df, [0, 1, 2], "molecule", "apo")
    assert changed == 2, "the row already carrying that exact label must not count as changed"
    assert list(label_column(df["flag"], "molecule")) == ["apo", "apo", "apo"]
    assert "flagged" in df.iat[1, df.columns.get_loc("flag")]


def test_clear_labels_on_a_frame():
    df = _frame(["molecule=apo;flagged", "molecule=holo"])
    assert clear_labels(df, [0, 1], "molecule") == 2
    assert list(label_column(df["flag"], "molecule")) == ["", ""]
    assert "flagged" in df.iat[0, df.columns.get_loc("flag")]


def test_apply_label_refuses_a_frame_that_is_not_an_events_table():
    with pytest.raises(KeyError, match="flag"):
        apply_label(pd.DataFrame({"a": [1]}), [0], "molecule", "apo")


# ================================================================================================
# the palette
# ================================================================================================

def test_the_palette_maps_1_to_9_by_position_and_honours_explicit_hotkeys():
    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin")
    v.add("molecule", "holo-ferritin")
    v.add("type", "collision", hotkey="c")
    assert v.for_hotkey("1").token == "molecule=apo-ferritin"
    assert v.for_hotkey("2").token == "molecule=holo-ferritin"
    assert v.for_hotkey("c").token == "type=collision"
    assert v.for_hotkey("9") is None
    assert v.for_hotkey("0") is None, "0 is not a palette slot"


def test_custom_hotkeys_are_case_insensitive_and_reserved_keys_are_refused():
    assert validate_label_hotkey("C") == "c"
    assert validate_label_hotkey("0") == "0"
    assert validate_label_hotkey("") == ""
    for key in ("a", "A", "r", "R", "t", "T", " "):
        with pytest.raises(ValueError, match="reserved"):
            validate_label_hotkey(key)
    with pytest.raises(ValueError, match="one printable character"):
        validate_label_hotkey("Ctrl+Z")


def test_legacy_unsafe_and_duplicate_hotkeys_load_but_stay_unarmed():
    v = LabelVocabulary.from_json([
        {"key": "old", "value": "reserved", "hotkey": "a", "note": "legacy"},
        {"key": "type", "value": "first", "hotkey": "c"},
        {"key": "type", "value": "second", "hotkey": "C"},
        {"key": "old", "value": "sequence", "hotkey": "Ctrl+Z"},
    ])

    assert len(v.entries) == 4, "unsafe legacy shortcuts must not delete label definitions"
    assert v.for_hotkey("a") is None
    assert v.for_hotkey("Ctrl+Z") is None
    assert v.for_hotkey("C").token == "type=first"
    assert v.hotkey_for(v.entries[2]) == "", "the duplicate must not appear armed"


def test_adding_the_same_label_twice_updates_it_rather_than_duplicating():
    v = LabelVocabulary()
    v.add("type", "collision")
    v.add("type", "collision", hotkey="c", note="glancing hit")
    assert len(v.entries) == 1
    assert v.entries[0].hotkey == "c" and v.entries[0].note == "glancing hit"


def test_the_palette_round_trips_through_json():
    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin", hotkey="a", note="500 pM")
    v.add("levels", "3")
    back = LabelVocabulary.from_json(v.to_json())
    assert [e.token for e in back.entries] == [e.token for e in v.entries]
    assert back.entries[0].hotkey == "a" and back.entries[0].note == "500 pM"


def test_a_broken_palette_entry_is_skipped_rather_than_stopping_a_project_opening():
    back = LabelVocabulary.from_json([
        {"key": "molecule", "value": "apo"},
        {"key": "bad;key", "value": "x"},      # invalid
        {"nope": 1},                            # malformed
        {"key": "type", "value": "collision"},
    ])
    assert [e.token for e in back.entries] == ["molecule=apo", "type=collision"]


def test_an_invalid_palette_entry_is_refused_at_construction():
    with pytest.raises(ValueError):
        LabelEntry(key="mol;ecule", value="apo")


def test_format_labels_is_readable():
    f = set_label(set_label(FLAG_CURATED, "molecule", "apo"), "levels", "3")
    assert format_labels(f) == "molecule=apo, levels=3"


# ================================================================================================
# a label is not a rejection REASON
# ================================================================================================

def test_a_label_does_not_freeze_an_automatic_rejection():
    """MEASURED BEFORE, and the worst defect labelling introduced.

    ``inherited_curation`` decides whether a rejection was automatic (re-derivable, so
    re-judge it) or a person's (so honour it) by asking whether every token is a DERIVED
    flag. Labels are not derived flags — so one label on a row rejected solely for
    ``dwell>max_dwell_s`` made the rejection look like a curator's, froze it forever, and
    dropped the derived reason as usual. The row came back rejected with NO REASON: invisible
    to every filter and count, which is exactly the C21 failure this module documents.

    A label says what an event IS, never why it was rejected.
    """
    from nanoscope.analysis.events import FLAG_MAX_DWELL

    plain, plain_ok = inherited_curation([FLAG_MAX_DWELL], [False])
    labelled, labelled_ok = inherited_curation(
        [set_label(FLAG_MAX_DWELL, "molecule", "apo-ferritin")], [False])

    assert plain_ok is True, "an automatic rejection is re-judged on a rebuild"
    assert labelled_ok is True, "the label froze an automatic rejection"
    assert parse_labels(labelled) == {"molecule": "apo-ferritin"}, "and the label must survive"


def test_a_hand_rejection_is_still_honoured_when_the_row_carries_a_label():
    """The other half. Ignoring labels when judging must not make every rejection automatic."""
    flag, accepted = inherited_curation([set_label("", "molecule", "apo")], [False])
    assert accepted is False, "a bare hand rejection was silently reversed"
    assert parse_labels(flag) == {"molecule": "apo"}

    flag2, accepted2 = inherited_curation(
        [set_label(FLAG_CURATED, "molecule", "apo")], [False])
    assert accepted2 is False
    assert "flagged" in flag2


def test_a_label_alone_never_makes_an_accepted_row_rejected():
    flag, accepted = inherited_curation([set_label("", "type", "collision")], [True])
    assert accepted is True
    assert parse_labels(flag) == {"type": "collision"}


# ================================================================================================
# defects an adversarial review found after the first pass
# ================================================================================================

def test_the_palette_display_and_the_key_handler_can_never_disagree():
    """MEASURED BEFORE: with one explicit hotkey in the list, the combo printed
    "molecule=apo-ferritin  [2]" while pressing 2 wrote *holo*-ferritin — the display
    numbered ALL entries by position, the handler numbered only those WITHOUT an explicit
    key. Wrong labels, written in bulk, silently. One assignment now serves both.
    """
    v = LabelVocabulary()
    v.add("type", "collision", hotkey="c")
    v.add("molecule", "apo-ferritin")
    v.add("molecule", "holo-ferritin")

    for e in v.entries:
        ch = v.hotkey_for(e)
        assert ch, f"{e.token} was printed with no key"
        assert v.for_hotkey(ch).token == e.token, (
            f"the palette prints {e.token} as [{ch}] but that key writes "
            f"{v.for_hotkey(ch).token}")


def test_an_explicit_hotkey_does_not_consume_a_digit_twice():
    v = LabelVocabulary()
    v.add("a", "1", hotkey="2")          # explicitly claims '2'
    v.add("b", "2")
    v.add("c", "3")
    keys = v.hotkeys()
    assert keys["2"].token == "a=1"
    assert sorted(k for k in keys if k.isdigit()) == ["1", "2", "3"]
    assert len({e.token for e in keys.values()}) == 3, "two keys wrote the same entry"


def test_an_entry_beyond_the_ninth_says_it_has_no_key_rather_than_pretending():
    v = LabelVocabulary()
    for i in range(12):
        v.add("k", f"v{i}")
    assert v.hotkey_for(v.entries[9]) == ""
    assert len([c for c in v.hotkeys() if c.isdigit()]) == 9


def test_a_merge_that_produces_two_values_for_one_key_shows_both():
    """MEASURED BEFORE: merging fragments labelled apo- and holo-ferritin unioned both
    tokens (inherited_curation is generic and cannot know one key holds one value) and every
    reader silently returned the first — so the merged row claimed one molecule while
    carrying evidence of two, with nobody told. The app cannot know which fragment was
    right, so it says so."""
    from nanoscope.analysis.labels import CONFLICT_SEP, label_conflicts

    a = set_label("", "molecule", "apo-ferritin")
    b = set_label("", "molecule", "holo-ferritin")
    flag, _ = inherited_curation([a, b], [True, True])

    assert parse_labels(flag)["molecule"] == "apo-ferritin" + CONFLICT_SEP + "holo-ferritin"
    assert label_conflicts(flag) == {"molecule": ["apo-ferritin", "holo-ferritin"]}
    assert CONFLICT_SEP in format_labels(flag), "the readout hides the conflict"


def test_merging_the_same_value_twice_is_not_a_conflict():
    from nanoscope.analysis.labels import label_conflicts

    a = set_label("", "molecule", "apo")
    flag, _ = inherited_curation([a, a], [True, True])
    assert parse_labels(flag) == {"molecule": "apo"}
    assert label_conflicts(flag) == {}


def test_the_conflict_separator_cannot_be_typed_into_a_value():
    """Otherwise a hand-typed value would be indistinguishable from a merge conflict."""
    from nanoscope.analysis.labels import CONFLICT_SEP

    with pytest.raises(ValueError):
        validate_label("molecule", f"apo{CONFLICT_SEP}holo")


def test_a_padded_key_normalises_instead_of_making_an_undeletable_duplicate():
    """MEASURED BEFORE: validate_label returned the cleaned pair and LabelEntry threw it
    away, so " molecule " sat beside "molecule" as a second entry that remove("molecule")
    could never reach."""
    v = LabelVocabulary()
    v.add("molecule", "apo")
    v.add("  molecule  ", "  apo  ")
    assert [e.token for e in v.entries] == ["molecule=apo"]
    assert v.remove("molecule", "apo") is True
    assert v.entries == []
