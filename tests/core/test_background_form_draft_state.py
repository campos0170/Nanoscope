"""Custom background recipes survive projects without losing unfinished edits or identity."""
from copy import deepcopy

import pytest

from nanoscope.core.form_draft_state import (
    WORKSPACE_KEY, background_state_readable, capture_form_drafts, restore_form_drafts,
)
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project


def background(key):
    return {"output_mode": "zero_centered_display", "draft": {
        "version": 1, "source": "anchors", "units": "pA",
        "binding": {"doc_key": key, "channel": 0, "units": "pA",
                    "calibration": [0.5, -2.0], "sample_rate_hz": 50_000.0},
        "regions": [["before jump", "0", "10"], ["after jump", "10", "20"]],
        "anchors": [["before jump", "0", "1e-"], ["after jump", "10", "85.5"]],
        "interpolation": "constant", "edge_policy": "refuse", "segment_policy": "separate",
        "reference": {"doc_key": key + ".blank", "channel": 1},
        "shift": "-", "scale": "1", "offset": "0", "resampling": "none",
    }}


def record(key, state):
    return {"doc_id": key, "channel": 0, "channel_name": "current", "module_id": None,
            "state": {"background": state}}


def test_custom_background_project_roundtrip_preserves_regions_binding_and_unfinished_text(tmp_path):
    source = tmp_path / "signal.csv"
    source.write_text("time,current\n0,100\n")
    key = str(source.resolve())
    state = background(key)
    assert background_state_readable(state)
    records = capture_form_drafts([record(key, state)], held=[(key, {0: "current"})])
    project = ProjectState(workspace=AnalyticalWorkspace(extra={WORKSPACE_KEY: records}))
    reopened = load_project(save_project(tmp_path / "background.nanoscope", project))
    drafts, warnings = restore_form_drafts(reopened.workspace.extra[WORKSPACE_KEY], key, {0: "current"})
    assert not warnings
    assert drafts == {(0, None): {"background": state}}
    # Editing a restored form cannot change the saved recipe.
    drafts[(0, None)]["background"]["draft"]["anchors"][0][2] = "100"
    assert reopened.workspace.extra[WORKSPACE_KEY][0]["state"]["background"] == state
    assert restore_form_drafts(records, key + ".different", {0: "current"}) == ({}, ())
    assert restore_form_drafts(records, key, {0: "voltage"})[0] == {}


@pytest.mark.parametrize("change", [
    lambda d: d.update(version=2),
    lambda d: d.update(version=True),
    lambda d: d.update(future_algorithm={"order": 3}),
    lambda d: d.update(anchors=[["region", "0"]]),
    lambda d: d.update(regions=[["region", 0, "100"]]),
    lambda d: d.update(source="future"),
    lambda d: d.update(reference={"doc_key": "blank", "channel": True}),
    lambda d: d.update(reference={"doc_key": "blank", "channel": 10**500}),
    lambda d: d["binding"].update(calibration=[float("nan"), 0]),
    lambda d: d["binding"].update(sample_rate_hz=-1),
    lambda d: d["binding"].update(sample_rate_hz=10**500),
    lambda d: d["binding"].update(units="nA"),
])
def test_unreadable_background_is_never_half_applied_or_discarded(tmp_path, change):
    key = str(tmp_path / "signal.csv")
    state = background(key)
    change(state["draft"])
    saved = [record(key, state)]
    original = deepcopy(saved)
    assert not background_state_readable(state)
    restored, warnings = restore_form_drafts(saved, key, {0: "current"})
    assert not restored and warnings
    kept = capture_form_drafts([], saved, [(key, {0: "current"})])
    # NaN compares unequal, so inspect the structure without normalizing its payload.
    assert repr(kept) == repr(original)


def test_a_live_background_recipe_replaces_the_same_channel_without_resurrecting_old_edits(tmp_path):
    key = str(tmp_path / "signal.csv")
    previous = [record(key, background(key))]
    current = deepcopy(previous)
    current[0]["state"]["background"]["draft"]["anchors"][0][2] = "100"
    assert capture_form_drafts(current, previous, [(key, {0: "current"})]) == current
    assert capture_form_drafts([], previous, [(key, {0: "current"})]) == []
