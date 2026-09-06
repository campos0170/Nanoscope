"""Comparison figures use opaque workspace resources and never run plot generation."""
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import (
    COMPARISON_FIGURES_ARTIFACT_ID as ARTIFACT,
    capture_analysis_workspace, capture_comparison_figures,
    restore_analysis_workspace, restore_comparison_figures,
)
from nanoscope.core.project import (
    AnalyticalWorkspace, ProjectState, WorkspaceArtifact, load_project, save_project,
)


class Comparison:
    def __init__(self):
        self.payload = {
            "draft": {"kind": "pair"}, "applied": {"kind": "facet"},
            "figure_sources": [{"source_key": "saved-recording", "channel": 0}],
        }
        self.tables = {"observations": pd.DataFrame({
            "dwell_s": [.001, .002], "depth_pA": [15., 20.],
        })}
        self.restore_calls = 0

    def snapshot_state(self):
        return self.payload, self.tables

    def restore_state(self, payload, tables):
        self.payload, self.tables = payload, tables
        self.restore_calls += 1

    def analyze(self):
        pytest.fail("Workspace adapters must not generate a comparison")


def test_capture_detaches_and_project_roundtrip_restores_without_generation(tmp_path):
    page = Comparison()
    captured = capture_comparison_figures(page)
    original = deepcopy(captured)
    page.payload["draft"]["kind"] = "overlay"
    page.tables["observations"].iloc[0, 0] = 999.
    assert captured.payload == original.payload
    assert captured.blobs == original.blobs
    assert captured.kind == "nanoscope.comparison-figures"
    assert not captured.tables  # mixed observations use the existing lossless codec
    unknown = WorkspaceArtifact("future", "future", payload={"keep": True}, blobs={"raw": b"unchanged"})
    workspace = capture_analysis_workspace(AnalyticalWorkspace(
        artifacts=[unknown, captured], extra={"view": "saved"}))
    assert [entry.artifact_id for entry in workspace.artifacts] == ["future", ARTIFACT]
    assert workspace.extra == {"view": "saved"}
    path = tmp_path / "comparison.nanoscope"
    save_project(path, ProjectState(workspace=workspace))
    restored = Comparison()
    report = restore_analysis_workspace(load_project(path).workspace, comparison_figures=restored)
    assert report.restored == (ARTIFACT,)
    assert restored.restore_calls == 1
    assert restored.payload == {k: v for k, v in original.payload.items() if k != "table_resources"}
    pd.testing.assert_frame_equal(restored.tables["observations"], Comparison().tables["observations"])


def test_future_schema_is_preserved_not_downgraded():
    future = WorkspaceArtifact(ARTIFACT, "nanoscope.comparison-figures", schema_version=2,
                               payload={"future": True}, blobs={"data": b"untouched"})
    workspace = capture_analysis_workspace(AnalyticalWorkspace(artifacts=[future]),
                                          comparison_figures=Comparison())
    assert workspace.artifacts[0].schema_version == 2
    assert workspace.artifacts[0].payload == future.payload
    assert workspace.artifacts[0].blobs == future.blobs
    page = Comparison()
    report = restore_analysis_workspace(workspace, comparison_figures=page)
    assert report.skipped == (ARTIFACT,) and report.warnings
    assert page.restore_calls == 0


def test_mixed_observation_cells_are_lossless(tmp_path):
    page = Comparison()
    page.tables["raw"] = pd.DataFrame({"measurement": pd.Series(
        [1.5, "not a number", None, pd.NA, 2], dtype=object)})
    expected = page.tables["raw"].copy(deep=True)
    path = tmp_path / "mixed.nanoscope"
    save_project(path, ProjectState(workspace=capture_analysis_workspace(comparison_figures=page)))
    restored = Comparison()
    report = restore_analysis_workspace(load_project(path).workspace, comparison_figures=restored)
    assert report.restored == (ARTIFACT,)
    pd.testing.assert_frame_equal(restored.tables["raw"], expected, check_exact=True)


@pytest.mark.parametrize("damage", ["missing", "malformed", "duplicate"])
def test_corrupt_resource_map_is_reported_without_partial_restore(damage):
    artifact = capture_comparison_figures(Comparison())
    if damage == "missing":
        artifact.blobs.clear()
    elif damage == "malformed":
        artifact.payload["table_resources"] = ["not a resource mapping"]
    else:
        artifact.tables["observations"] = pd.DataFrame({"x": [1.]})
    target = Comparison()
    report = restore_comparison_figures(target, artifact)
    assert report.skipped == (ARTIFACT,) and report.warnings
    assert target.restore_calls == 0


@pytest.mark.parametrize("payload,tables", [([], {}), ({}, []), ({}, {1: pd.DataFrame()}), ({}, {"bad": []})])
def test_invalid_capture_contract_is_rejected(payload, tables):
    page = Comparison()
    page.payload, page.tables = payload, tables
    with pytest.raises(ValueError):
        capture_comparison_figures(page)


def test_owned_comparison_replacement_preserves_unknown_sibling():
    original = capture_comparison_figures(Comparison())
    unknown = WorkspaceArtifact("plugin.custom", "plugin.custom", blobs={"raw": b"bytes"})
    page = Comparison()
    page.payload["applied"]["kind"] = "overlay"
    captured = capture_analysis_workspace(AnalyticalWorkspace(artifacts=[unknown, original]),
                                         comparison_figures=page)
    assert [a.artifact_id for a in captured.artifacts] == [unknown.artifact_id, ARTIFACT]
    assert captured.artifacts[0].blobs == unknown.blobs
    assert captured.artifacts[1].payload["applied"]["kind"] == "overlay"


@pytest.mark.parametrize("error", [ImportError("Seaborn unavailable"), RuntimeError("Rendering unavailable")])
def test_unavailable_renderer_preserves_pending_snapshot(error):
    artifact = capture_comparison_figures(Comparison())
    original = deepcopy(artifact)
    page = Comparison()
    def fail(_payload, _tables):
        raise error
    page.restore_state = fail
    report = restore_comparison_figures(page, artifact)
    assert report.skipped == (ARTIFACT,)
    assert str(error) in report.warnings[0]
    assert artifact.payload == original.payload
    assert artifact.blobs == original.blobs
