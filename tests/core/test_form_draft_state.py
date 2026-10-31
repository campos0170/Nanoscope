"""The analysis forms' un-run edits, saved with the project (design §8, the owner's decision 4):
``nanoscope/core/form_draft_state.py``, pure.

What is pinned here:

* a draft goes into a project and comes back, keyed by the recording's PERSISTENT key and
  matched by recording identity, not by path spelling and never by a per-load ``doc_id``;
* a draft is put back only on the channel it was made for: a channel renamed or gone since is
  left alone and said;
* an unknown, damaged or future record is never applied and survives the next save verbatim;
* the live session replaces every record about a channel it holds -- a draft its own run has
  since settled is not resurrected by the next reopen -- and keeps every record it cannot
  speak for;
* an unreadable section is quarantined, never overwritten.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

import pytest

from nanoscope.core.form_draft_state import (
    UNREADABLE_KEY, WORKSPACE_KEY, capture_form_drafts, prepare_form_draft_workspace,
    restore_form_drafts,
)
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project

SCHMITT = "detect.schmitt"


def _state(k_in: float = 6.0, rough_s: float = 2.0) -> dict:
    """A form state in AnalysisPanel.form_state()'s shape (only the values matter here)."""
    return {"detection": {"k_in": k_in, "k_out": 1.5, "filter_cutoff_hz": 5000.0,
                          "notch_freqs": [60.0, 120.0]},
            "intents": {"filter_cutoff_hz": 5000.0},
            "ml": {"enter": 0.5, "ml_decode": {"smoothing_kernel": 9}},
            "baseline": {"rough_s": rough_s, "base_s": 0.1, "source": "user",
                         "accepted_condemned": [], "reference": {"mode": "automatic"}}}


def _record(key: str, *, channel: int = 0, name: str | None = "current",
            module_id: str | None = SCHMITT, state: dict | None = None) -> dict:
    return {"doc_id": key, "channel": channel, "channel_name": name, "module_id": module_id,
            "state": _state() if state is None else state}


@pytest.fixture
def recording(tmp_path):
    """A recording on disk: its persistent key, and that key spelled through a symlink."""
    path = tmp_path / "pore.csv"
    path.write_text("time,current\n0,1\n")
    link = tmp_path / "link.csv"
    os.symlink(path, link)
    return str(path.resolve()), str(link)


# -- the round trip ----------------------------------------------------------------------------

def test_a_draft_goes_into_a_project_and_comes_back_on_its_own_table(tmp_path, recording):
    key, _link = recording
    live = [_record(key), _record(key, module_id=None, state=_state(k_in=7.0))]
    records = capture_form_drafts(live, None, [(key, {0: "current"})])
    assert records == live
    state = ProjectState(workspace=AnalyticalWorkspace(extra={WORKSPACE_KEY: records}))
    saved = load_project(save_project(tmp_path / "drafts.nanoscope", state))

    drafts, said = restore_form_drafts(saved.workspace.extra[WORKSPACE_KEY], key,
                                       {0: "current"})
    assert said == ()
    assert drafts == {(0, SCHMITT): _state(), (0, None): _state(k_in=7.0)}
    assert json.loads(json.dumps(drafts[(0, SCHMITT)])) == _state(), "JSON-safe both ways"


def test_records_are_matched_by_recording_identity_not_by_path_spelling(recording):
    """The project stores the path as it resolved at save time; the reopened file can be
    reached through a symlink left behind by a moved folder. Same recording, same drafts."""
    key, link = recording
    drafts, said = restore_form_drafts([_record(key)], link, {0: "current"})
    assert said == () and drafts == {(0, SCHMITT): _state()}

    other = restore_form_drafts([_record(key)], key + ".other", {0: "current"})
    assert other == ({}, ()), "a record about another recording is not this one's business"


def test_a_restored_state_is_a_copy_the_panel_can_keep(recording):
    key, _link = recording
    records = [_record(key)]
    drafts, _said = restore_form_drafts(records, key, {0: "current"})
    drafts[(0, SCHMITT)]["detection"]["k_in"] = 99.0
    assert records[0]["state"]["detection"]["k_in"] == 6.0, (
        "the panel's draft and the workspace's record are one object")


# -- a draft lands only on the channel it was made for ----------------------------------------

def test_a_renamed_or_missing_channel_is_not_given_the_draft(recording):
    key, _link = recording
    drafts, said = restore_form_drafts([_record(key)], key, {0: "voltage"})
    assert drafts == {} and len(said) == 1 and "channel" in said[0]
    drafts, said = restore_form_drafts([_record(key, channel=3)], key, {0: "current"})
    assert drafts == {} and len(said) == 1, "a channel this load does not have"
    drafts, said = restore_form_drafts([_record(key, name=None)], key, {0: "voltage"})
    assert drafts == {(0, SCHMITT): _state()} and said == (), (
        "a record that names no channel goes by its index")


# -- unknown and damaged records: kept, never applied -------------------------------------------

def test_unknown_damaged_and_future_records_are_kept_verbatim_and_never_applied(recording):
    key, _link = recording
    junk = [None, "not a record", 7, ["a", "list"]]
    future_field = {**_record(key), "future_field": 1}
    missing_field = {k: v for k, v in _record(key).items() if k != "channel_name"}
    future_part = _record(key, state={**_state(), "stage_marks": {"notch": True}})
    bad_values = [_record(key, state={"detection": "k_in=6"}),
                  _record(key, state={}),
                  {**_record(key), "channel": "0"},
                  {**_record(key), "channel": True},
                  {**_record(key), "channel": -1},
                  {**_record(key), "module_id": ""},
                  {**_record(key), "channel_name": 3}]
    damaged = [future_field, missing_field, future_part, *bad_values]

    drafts, said = restore_form_drafts(junk + damaged, key, {0: "current"})
    assert drafts == {}, "an unreadable record was applied"
    assert said.count("A malformed analysis-form draft was not applied.") == len(junk), said
    assert sum("not ones this version can read" in m for m in said) == len(damaged), said

    held = [(key, {0: "current"})]
    assert capture_form_drafts([], junk + damaged, held) == junk + damaged, (
        "a record this build cannot read must survive the save verbatim")


def test_the_first_of_two_drafts_for_one_table_wins(recording):
    key, _link = recording
    first, second = _record(key), _record(key, state=_state(k_in=9.0))
    drafts, said = restore_form_drafts([first, second], key, {0: "current"})
    assert drafts == {(0, SCHMITT): _state()}
    assert len(said) == 1 and "duplicate" in said[0]


# -- the save: the live session speaks for the channels it holds --------------------------------

def test_a_draft_the_session_has_since_settled_is_not_saved_again(recording, tmp_path):
    """Reopen a project with a draft, run the form it holds (the run drops the draft), save:
    the old record must go, or the next reopen puts back a form its own run settled."""
    key, _link = recording
    previous = [_record(key)]
    assert capture_form_drafts([], previous, [(key, {0: "current"})]) == []
    live = [_record(key, state=_state(k_in=8.0))]
    assert capture_form_drafts(live, previous, [(key, {0: "current"})]) == live


def test_records_the_session_cannot_speak_for_are_kept(recording, tmp_path):
    """A recording that is not open (its file was missing on reopen), and a channel that did
    not come back as itself, keep their drafts for the session that can apply them."""
    key, _link = recording
    absent = _record(str(tmp_path / "missing.csv"))
    renamed = _record(key, name="old name")
    other_channel = _record(key, channel=1, name="current 2")
    previous = [absent, renamed, other_channel]
    live = [_record(key, state=_state(k_in=8.0))]
    assert capture_form_drafts(live, previous, [(key, {0: "current"})]) == (
        [absent, renamed, other_channel] + live)


def test_capture_hands_out_copies(recording):
    key, _link = recording
    live, previous = [_record(key)], [_record(key, channel=1)]
    out = capture_form_drafts(live, previous, [])
    out[0]["state"]["detection"]["k_in"] = 0.0
    out[1]["state"]["detection"]["k_in"] = 0.0
    assert previous[0]["state"]["detection"]["k_in"] == 6.0
    assert live[0]["state"]["detection"]["k_in"] == 6.0


# -- an unreadable section ---------------------------------------------------------------------

@pytest.mark.parametrize("unreadable", [None, {}, "unknown", 3, {"doc_id": "x"}])
def test_an_unreadable_section_is_quarantined_never_overwritten(recording, unreadable):
    key, _link = recording
    extra = {WORKSPACE_KEY: deepcopy(unreadable), UNREADABLE_KEY: {"older": True}}
    note = prepare_form_draft_workspace(extra)
    assert note and extra[WORKSPACE_KEY] == []
    assert extra[UNREADABLE_KEY] == {"older": True}, "an earlier quarantine was overwritten"
    assert extra[UNREADABLE_KEY + "_1"] == unreadable
    if unreadable is not None:              # None is also how an absent section reads
        with pytest.raises(ValueError, match="unreadable"):
            capture_form_drafts([_record(key)], unreadable, [])
    drafts, said = restore_form_drafts(unreadable, key, {0: "current"})
    assert drafts == {} and len(said) == 1, "an unreadable section was applied"
    assert prepare_form_draft_workspace({}) == () and prepare_form_draft_workspace(extra) == ()
