"""New summaries persist through the existing opaque workspace container."""
from copy import deepcopy

import pandas as pd

from nanoscope.core.analysis_workspace import (
    DISTRIBUTION_SUMMARIES_ARTIFACT_ID, capture_analysis_workspace,
    capture_distribution_summaries, restore_analysis_workspace,
)
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, WorkspaceArtifact, load_project, save_project


class Summary:
    def __init__(self):
        self.payload = {"draft": {"kind": "Box"}, "applied": {"kind": "ECDF"},
                        "figure_sources": [{"source_key": "source", "channel": 0}]}
        self.tables = {"coordinates": pd.DataFrame({"value": [1., 2.], "probability": [.5, 1.]})}

    def snapshot_state(self):
        return self.payload, self.tables

    def restore_state(self, payload, tables):
        self.payload, self.tables = payload, tables


def test_summary_adapter_detaches_and_roundtrips_without_compute(tmp_path):
    page = Summary()
    captured = capture_distribution_summaries(page)
    original = deepcopy(captured)
    page.payload["draft"]["kind"] = "Violin"
    page.tables["coordinates"].iloc[0, 0] = 999
    assert captured.payload == original.payload
    assert captured.blobs == original.blobs
    unknown = WorkspaceArtifact("future", "future", payload={"keep": True}, blobs={"raw": b"unchanged"})
    workspace = AnalyticalWorkspace(artifacts=[unknown, captured], extra={"view": "saved"})
    carried = capture_analysis_workspace(workspace)
    assert [entry.artifact_id for entry in carried.artifacts] == ["future", DISTRIBUTION_SUMMARIES_ARTIFACT_ID]
    path = tmp_path / "summary.nanoscope"
    save_project(path, ProjectState(workspace=carried))
    restored = Summary()
    report = restore_analysis_workspace(load_project(path).workspace, distribution_summaries=restored)
    assert report.restored == (DISTRIBUTION_SUMMARIES_ARTIFACT_ID,)
    assert restored.payload == {key: value for key, value in original.payload.items() if key != "table_resources"}
    pd.testing.assert_frame_equal(restored.tables["coordinates"], Summary().tables["coordinates"])


def test_newer_summary_schema_is_carried_not_downgraded():
    future = WorkspaceArtifact(DISTRIBUTION_SUMMARIES_ARTIFACT_ID, "nanoscope.distribution-summaries",
                               schema_version=2, payload={"future": True}, blobs={"data": b"untouched"})
    workspace = capture_analysis_workspace(AnalyticalWorkspace(artifacts=[future]),
                                          distribution_summaries=Summary())
    assert workspace.artifacts[0].schema_version == 2
    assert workspace.artifacts[0].payload == future.payload
    assert workspace.artifacts[0].blobs == future.blobs
    report = restore_analysis_workspace(workspace, distribution_summaries=Summary())
    assert report.skipped == (DISTRIBUTION_SUMMARIES_ARTIFACT_ID,)
    assert report.warnings


def test_mixed_raw_observations_and_missing_values_survive_project_roundtrip(tmp_path):
    page = Summary()
    page.tables["raw"] = pd.DataFrame({
        "measurement": pd.Series([1.5, "not a number", None, pd.NA, 2], dtype=object)})
    expected = page.tables["raw"].copy(deep=True)
    captured = capture_distribution_summaries(page)
    assert "raw" in captured.payload["table_resources"]
    path = tmp_path / "mixed.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[captured])))
    restored = Summary()
    report = restore_analysis_workspace(load_project(path).workspace, distribution_summaries=restored)
    assert report.restored == (DISTRIBUTION_SUMMARIES_ARTIFACT_ID,)
    pd.testing.assert_frame_equal(restored.tables["raw"], expected, check_exact=True)
