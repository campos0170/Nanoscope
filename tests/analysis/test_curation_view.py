"""analysis.curation_view — the rules both curation windows share.

The Event inspector and the Event gallery may not import each other, so what "Unreviewed"
means, which rows a label filter holds, how far a curation pass has got and what the end
of a filter says are decided here, once, on plain pandas frames.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import curation_view as cv
from nanoscope.analysis.events import (
    FLAG_CLIPPED, FLAG_CURATED, REVIEW_COL, is_reviewed, reviewed_mask,
)
from nanoscope.analysis.labels import LabelVocabulary


def _frame(n: int = 6, *, flags=None, review=None, accepted=None) -> pd.DataFrame:
    df = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "start_idx": np.arange(n, dtype=np.int64) * 100,
        "accepted": np.ones(n, dtype=bool) if accepted is None else np.asarray(accepted, bool),
        "flag": [""] * n if flags is None else list(flags),
    })
    if review is not None:
        df[REVIEW_COL] = pd.array(list(review), dtype="string")
    return df


def _vocab(*tokens: str) -> LabelVocabulary:
    v = LabelVocabulary()
    for t in tokens:
        k, _, val = t.partition("=")
        v.add(k, val)
    return v


# -- filter names ------------------------------------------------------------------------------
def test_parse_filter_round_trips_and_rejects_malformed():
    assert cv.parse_filter("All") == ("all", "", "")
    assert cv.parse_filter("Unreviewed") == ("unreviewed", "", "")
    assert cv.parse_filter(cv.label_filter("type=collision")) == ("label", "type", "collision")
    assert cv.parse_filter("Labelled: a=b=c") == ("label", "a", "b=c")
    for bad in ("Labelled: =x", "Labelled: x=", "Labelled: nokey", "Rejected", "", None):
        assert cv.parse_filter(bad) is None
        assert not cv.is_filter(bad)
    # A label value that merely contains "reject" is still a label filter.
    assert cv.parse_filter("Labelled: type=rejected-noise")[0] == "label"


def test_only_label_and_review_filters_depend_on_label_or_review_writes():
    assert not any(cv.filter_depends_on_labels_or_review(m)
                   for m in ("All", "Accepted only", "Rejected only", "nonsense"))
    assert all(cv.filter_depends_on_labels_or_review(m)
               for m in ("Unreviewed", "Unlabelled", "Labelled: k=v"))


# -- masks -------------------------------------------------------------------------------------
def test_unreviewed_includes_missing_blank_and_unrecognised_review():
    assert cv.filter_mask(_frame(3), "Unreviewed").tolist() == [True, True, True]
    df = _frame(5, review=["reviewed", " reviewed ", "unreviewed", pd.NA, "bogus"])
    assert cv.filter_mask(df, "Unreviewed").tolist() == [False, False, True, True, True]


def test_reviewed_mask_agrees_with_is_reviewed_row_by_row():
    rng = np.random.default_rng(3)
    pool = ["reviewed", " reviewed", "reviewed ", "unreviewed", "REVIEWED", "", "  ", None,
            "x", "reviewed\t"]
    values = [pool[i] for i in rng.integers(0, len(pool), 50)]
    df = _frame(50)
    df[REVIEW_COL] = pd.array([pd.NA if v is None else v for v in values], dtype="string")
    mask = reviewed_mask(df)
    for i in range(len(df)):
        assert bool(mask[i]) == (is_reviewed(df.iloc[i]) is True), (i, values[i])


def test_a_label_filter_matches_a_merge_conflict_under_each_value():
    df = _frame(4, flags=["molecule=apo|holo", f"{FLAG_CURATED}", FLAG_CLIPPED,
                          f"{FLAG_CURATED};molecule=apo"])
    assert cv.filter_mask(df, "Labelled: molecule=apo").tolist() == [True, False, False, True]
    assert cv.filter_mask(df, "Labelled: molecule=holo").tolist() == [True, False, False, False]
    # Bare tokens -- the curator's own mark, a derived flag -- are not labels.
    assert cv.filter_mask(df, "Unlabelled").tolist() == [False, True, True, False]
    assert cv.labelled_mask(df).tolist() == [True, False, False, True]
    # token_mask itself: a padded key / value still matches, a missing cell never does.
    df.loc[2, "flag"] = None
    assert cv.token_mask(df, " molecule ", "holo ").tolist() == [True, False, False, False]
    assert cv.token_mask(df.iloc[:0], "molecule", "apo").tolist() == []
    assert cv.token_mask(df.drop(columns="flag"), "molecule", "apo").tolist() == [False] * 4


def test_accept_and_reject_filters_read_accepted_and_unknown_modes_raise():
    df = _frame(3, accepted=[True, False, True])
    assert cv.filter_mask(df, "Accepted only").tolist() == [True, False, True]
    assert cv.filter_mask(df, "Rejected only").tolist() == [False, True, False]
    assert cv.filter_mask(df, "All").tolist() == [True, True, True]
    with pytest.raises(ValueError):
        cv.filter_mask(df, "Rejected")


def test_masks_survive_frames_without_flag_or_review():
    df = pd.DataFrame({"event_id": [0, 1], "accepted": [True, False]})
    assert cv.labelled_mask(df).tolist() == [False, False]
    assert cv.filter_mask(df, "Unlabelled").tolist() == [True, True]
    assert cv.filter_mask(df, "Unreviewed").tolist() == [True, True]
    p = cv.curation_progress(df, _vocab("type=a"))
    assert (p.n_total, p.n_reviewed, p.n_accepted, p.n_unknown_review) == (2, 0, 1, 2)
    assert cv.format_progress(p) == "Reviewed 0 / 2 · accepted 1 · labels: none yet"
    assert cv.curation_progress(pd.DataFrame(), None).n_total == 0
    assert cv.curation_progress(None).n_total == 0


# -- navigation core ---------------------------------------------------------------------------
def test_the_navigation_core_steps_over_a_mask_and_never_wraps():
    mask = np.array([False, True, False, True, True, False])
    assert cv.first_member(mask) == 1
    assert cv.next_member_after(mask, 1) == 3
    assert cv.next_member_after(mask, 4) is None
    assert cv.next_member_after(mask, 0) == 1
    assert cv.prev_member_before(mask, 3) == 1
    assert cv.prev_member_before(mask, 1) is None
    assert cv.count_before(mask, 4) == 2
    assert cv.count_before(mask, 0) == 0
    assert cv.first_member(np.zeros(3, bool)) is None


def test_same_events_needs_identical_ids_and_starts():
    a = _frame(4)
    assert cv.same_events(a, a.copy())
    b = a.copy(); b["event_id"] = [0, 1, 2, 9]
    assert not cv.same_events(a, b)
    c = a.copy(); c.loc[2, "start_idx"] = 201
    assert not cv.same_events(a, c)
    assert not cv.same_events(a, a.iloc[:3])
    assert not cv.same_events(None, a)


# -- progress ----------------------------------------------------------------------------------
def test_the_progress_line_is_the_requested_sentence():
    n = 870
    flags = [""] * n
    for i in range(40):
        flags[i] = "type=collision"
    for i in range(40, 300):
        flags[i] = "type=translocation"
    review = ["reviewed"] * 312 + [pd.NA] * (n - 312)
    accepted = np.ones(n, bool); accepted[:69] = False
    df = _frame(n, flags=flags, review=review, accepted=accepted)
    p = cv.curation_progress(df, _vocab("type=collision", "type=translocation"))
    assert cv.format_progress(p) == (
        "Reviewed 312 / 870 · accepted 801 · labels: type=collision 40, type=translocation 260")
    assert p.n_unreviewed == 558 and p.n_unknown_review == 558
    detail = cv.describe_progress(p)
    assert "558 of them have no review record" in detail
    assert "whole detection" in detail

    plain = cv.curation_progress(_frame(3), None)
    assert cv.format_progress(plain) == "Reviewed 0 / 3 · accepted 3"
    palette_only = cv.curation_progress(_frame(3), _vocab("type=a"))
    assert cv.format_progress(palette_only) == "Reviewed 0 / 3 · accepted 3 · labels: none yet"


def test_progress_counts_conflicts_once_per_value_and_names_orphans():
    df = _frame(4, flags=["molecule=apo|holo", "type=x", "type=x;molecule=apo", ""])
    p = cv.curation_progress(df, _vocab("type=x", "type=y"))
    assert p.label_counts == (("type=x", 2), ("type=y", 0),
                              ("molecule=apo", 2), ("molecule=holo", 1))
    assert p.orphan_tokens == ("molecule=apo", "molecule=holo")
    assert (p.n_labelled, p.n_conflicted) == (3, 1)
    assert "Labels not in the palette: molecule=apo 2" in cv.describe_progress(p)


def test_progress_is_counted_per_distinct_flag():
    def build(n):
        flags = np.array(["", "type=a", "type=b|c", FLAG_CURATED], dtype=object)[np.arange(n) % 4]
        review = np.array(["reviewed", None, "unreviewed"], dtype=object)[np.arange(n) % 3]
        return _frame(n, flags=list(flags), review=[pd.NA if r is None else r for r in review])
    vocab = _vocab("type=a", "type=b")
    for n, budget in ((6_000, 0.020), (100_000, 0.150)):
        df = build(n)
        cv.curation_progress(df, vocab)            # warm up imports / pandas caches
        t0 = time.perf_counter()
        cv.curation_progress(df, vocab)
        assert time.perf_counter() - t0 < budget, n
    # ...and by construction: neither the counter nor the helper loops over rows.
    tree = ast.parse(Path(cv.__file__).read_text())
    funcs = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
    for name in ("curation_progress", "_per_distinct"):
        src = ast.unparse(funcs[name])
        assert "factorize" in src or "value_counts" in src, name
        assert "iterrows" not in src and "range(len" not in src, name


# -- filter options and notes ------------------------------------------------------------------
def test_the_filter_options_are_the_fixed_four_then_unlabelled_palette_and_orphans():
    assert cv.filter_options(cv.curation_progress(_frame(3)), LabelVocabulary()) == (
        "All", "Accepted only", "Rejected only", "Unreviewed")
    df = _frame(3, flags=["molecule=apo", "", ""])
    vocab = _vocab("type=collision", "type=junk")
    assert cv.filter_options(cv.curation_progress(df, vocab), vocab) == (
        "All", "Accepted only", "Rejected only", "Unreviewed", "Unlabelled",
        "Labelled: type=collision", "Labelled: type=junk", "Labelled: molecule=apo")
    # The filter being browsed is never dropped, even when nothing offers it any more.
    opts = cv.filter_options(cv.curation_progress(_frame(3)), None, "Labelled: gone=away")
    assert opts[-1] == "Labelled: gone=away"
    assert cv.filter_options(None, None, "not a filter") == cv.FIXED_FILTERS


def test_end_and_no_unreviewed_notes_say_what_remains():
    short, full = cv.end_of_filter_note("Unreviewed", noun="event", n_earlier=12,
                                        n_unreviewed=40, first_key="⌘U")
    assert short == "End of Unreviewed · 12 earlier (⌘U)" and len(short) <= 40
    assert "12 earlier remain" in full and "⌘U" in full
    assert cv.end_of_filter_note("Unreviewed", noun="event", n_unreviewed=0)[0] == \
        "Every event reviewed"
    assert cv.end_of_filter_note("Unreviewed", noun="tile", n_unreviewed=5)[0] == \
        "End of Unreviewed"
    short, full = cv.end_of_filter_note("Accepted only", noun="event")
    assert short == "End of Accepted only"
    assert "last event" in full and "decision was written" in full

    short, full = cv.no_unreviewed_note("Accepted only", noun="event", n_unreviewed=3,
                                        chooser="Filter")
    assert short == "None unreviewed here · 3 elsewhere" and len(short) <= 40
    assert "Filter ▸ Unreviewed" in full
    assert cv.no_unreviewed_note("All", noun="tile", n_unreviewed=0, chooser="Show")[0] == \
        "Every event reviewed"


def test_label_writes_stamp_review_and_refuse_a_frame_without_flag():
    df = _frame(3)
    assert cv.label_rows(df, [1], "type", "a") == (1, 1)
    assert df["flag"].tolist()[1] == "type=a" and bool(reviewed_mask(df)[1])
    assert cv.label_rows(df, [1], "type", "a") == (0, 0)       # already there and reviewed
    assert cv.unlabel_rows(df, [1], "type") == (1, 0)
    flagless = pd.DataFrame({"event_id": [0], "accepted": [True]})
    assert cv.label_rows(flagless, [0], "type", "a") == (0, 0)
    assert cv.unlabel_rows(flagless, [0], "type") == (0, 0)
    assert REVIEW_COL not in flagless.columns


def test_the_module_imports_no_qt_and_no_gui():
    tree = ast.parse(Path(cv.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [("." * node.level) + (node.module or "")]
        else:
            continue
        for name in names:
            assert "PySide6" not in name and "Qt" not in name and "gui" not in name, name


def test_row_unchanged_asks_whether_an_undo_touched_one_event():
    a = _frame(4, review=[pd.NA, "reviewed", pd.NA, pd.NA])
    a["depth"] = [1.0, np.nan, 3.0, 4.0]
    b = a.copy(deep=True)
    assert all(cv.row_unchanged(a, b, r) for r in range(4))       # NA == NA, NaN == NaN
    b.iat[2, b.columns.get_loc("accepted")] = False
    assert not cv.row_unchanged(a, b, 2) and cv.row_unchanged(a, b, 1)
    b.iat[1, b.columns.get_loc(REVIEW_COL)] = pd.NA
    assert not cv.row_unchanged(a, b, 1)
    assert not cv.row_unchanged(a, b.drop(columns="depth"), 0)    # another schema
    assert not cv.row_unchanged(a, b, 9)


def test_display_candidates_keep_a_label_filters_value_and_a_hotkey():
    assert cv.filter_display_candidates("Labelled: molecule=holo-ferritin") == (
        "Labelled: molecule=holo-ferritin", "molecule=holo-ferritin", "holo-ferritin")
    assert cv.filter_display_candidates("Unreviewed") == ("Unreviewed",)
    # Updated (review B-CLIP-2): the last form keeps the hotkey; a too-narrow face cuts it
    # in the middle, so the digit stays in view.
    assert cv.label_display_candidates("molecule=holo-ferritin", "2") == (
        "molecule=holo-ferritin   [2]", "holo-ferritin   [2]", "holo-ferritin [2]")
    assert cv.label_display_candidates("type=a") == ("type=a", "a")
