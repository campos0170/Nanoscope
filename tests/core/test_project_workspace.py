"""Project format v11: durable analytical-workspace artifacts.

The core contract is intentionally producer-neutral. Built-in plot/PCA/GMM windows and
third-party plugins use the same stable-id envelope; core owns ordering, JSON metadata,
Parquet tables, opaque blobs, forward preservation, and validation.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import nanoscope.core.project as project
from nanoscope.core.project import (
    AnalyticalWorkspace,
    ProjectError,
    ProjectState,
    ProjectVersionError,
    WorkspaceArtifact,
    load_project,
    save_project,
)
from nanoscope.core.recovery import clone_project_state


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _replace_manifest(src: Path, dst: Path, edit) -> Path:
    members = _members(src)
    manifest = json.loads(members[project.MANIFEST_NAME])
    edit(manifest, members)
    members[project.MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return dst


def _scores() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": pd.Series([4, 9, 12], dtype="int64"),
        "pc1": np.asarray([1.25, -0.5, 3.75], dtype=np.float64),
        "cluster": pd.Series([0, pd.NA, 1], dtype="Int64"),
        "accepted": pd.Series([True, False, True], dtype="boolean"),
    }).set_index("event_id")


def test_v11_workspace_round_trips_order_json_tables_and_blobs(tmp_path):
    scores = _scores()
    first = WorkspaceArtifact(
        artifact_id="plots/main",
        kind="nanoscope.plot-window",
        payload={"plot_type": "PCA", "features": ["dwell_s", "depth_pA"]},
        tables={"scores": scores},
        blobs={"thumbnail": b"\x89PNG\r\nopaque\x00bytes"},
    )
    second = WorkspaceArtifact(
        artifact_id="subpop/apo-700",
        kind="nanoscope.subpopulation",
        schema_version=3,
        payload={"method": "gmm", "components": 2},
    )
    state = ProjectState(workspace=AnalyticalWorkspace(artifacts=[first, second]))

    path = save_project(tmp_path / "workspace.nanoscope", state)
    back = load_project(path).workspace

    assert back.version == project.WORKSPACE_VERSION
    assert [a.artifact_id for a in back.artifacts] == ["plots/main", "subpop/apo-700"]
    assert [a.kind for a in back.artifacts] == [
        "nanoscope.plot-window", "nanoscope.subpopulation"]
    assert back.artifacts[0].payload == first.payload
    assert back.artifacts[1].schema_version == 3
    assert_frame_equal(back.artifacts[0].tables["scores"], scores)
    assert back.artifacts[0].blobs["thumbnail"] == first.blobs["thumbnail"]

    members = _members(path)
    manifest = json.loads(members[project.MANIFEST_NAME])
    entry = manifest["workspace"]["artifacts"][0]
    assert entry["payload"]["plot_type"] == "PCA"
    assert entry["tables"]["scores"].endswith(".parquet")
    assert entry["blobs"]["thumbnail"].endswith(".bin")
    # Bulk event-wise values are members, never a giant manifest JSON array.
    assert "pc1" not in entry["payload"]
    assert entry["tables"]["scores"] in members
    assert entry["blobs"]["thumbnail"] in members


def test_unknown_plugin_kind_schema_and_extension_fields_survive_load_save(tmp_path):
    original = save_project(tmp_path / "seed.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(
            extra={"layout_engine": "future-grid", "layout_options": {"gap": 7}},
            artifacts=[WorkspaceArtifact(
                artifact_id="org.example.plugin:embedding-1",
                kind="org.example.mystery-embedding",
                schema_version=41,
                payload={"axes": ["latent_a", "latent_b"], "future": {"mode": None}},
                tables={"embedding": _scores()},
                blobs={"vendor-state": bytes(range(64))},
                extra={
                    "plugin_version": "9.4.1",
                    "future_top_level": {"enabled": True, "weights": [1, 2, 3]},
                },
            )],
        )))

    loaded = load_project(original)
    artifact = loaded.workspace.artifacts[0]
    assert artifact.kind == "org.example.mystery-embedding"  # unknown is not rejected
    assert artifact.schema_version == 41
    assert loaded.workspace.extra == {
        "layout_engine": "future-grid", "layout_options": {"gap": 7}}
    assert artifact.extra == {
        "plugin_version": "9.4.1",
        "future_top_level": {"enabled": True, "weights": [1, 2, 3]},
    }

    resaved = save_project(tmp_path / "resaved.nanoscope", loaded)
    again = load_project(resaved).workspace
    unknown = again.artifacts[0]
    assert again.extra == loaded.workspace.extra
    assert unknown.artifact_id == artifact.artifact_id
    assert unknown.kind == artifact.kind
    assert unknown.schema_version == artifact.schema_version
    assert unknown.payload == artifact.payload
    assert unknown.extra == artifact.extra
    assert_frame_equal(unknown.tables["embedding"], artifact.tables["embedding"])
    assert unknown.blobs["vendor-state"] == bytes(range(64))


def test_recovery_snapshot_detaches_workspace_payload_tables_and_extensions():
    table = pd.DataFrame({"event_id": [1], "meta": [{"label": "original"}]})
    state = ProjectState(workspace=AnalyticalWorkspace(
        extra={"layout": {"columns": 2}},
        artifacts=[WorkspaceArtifact(
            "a", "plugin", payload={"features": ["dwell_s"]}, tables={"scores": table},
            extra={"vendor": {"revision": 1}})]))

    snapshot = clone_project_state(state)
    state.workspace.extra["layout"]["columns"] = 9
    state.workspace.artifacts[0].payload["features"].append("depth_pA")
    state.workspace.artifacts[0].tables["scores"].at[0, "meta"]["label"] = "mutated"
    state.workspace.artifacts[0].extra["vendor"]["revision"] = 2

    saved = snapshot.workspace.artifacts[0]
    assert snapshot.workspace.extra == {"layout": {"columns": 2}}
    assert saved.payload == {"features": ["dwell_s"]}
    assert saved.tables["scores"].at[0, "meta"] == {"label": "original"}
    assert saved.extra == {"vendor": {"revision": 1}}


def test_extension_fields_in_a_hand_authored_manifest_are_preserved(tmp_path):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(artifacts=[WorkspaceArtifact("a", "plugin.unknown")])))

    def add_unknown_fields(manifest, _members):
        workspace = manifest["workspace"]
        workspace["future_workspace_flag"] = ["leave", "me"]
        artifact = workspace["artifacts"][0]
        artifact["vendor"] = {"name": "Example", "opaque": [1, None, False]}

    hand_authored = _replace_manifest(seed, tmp_path / "hand.nanoscope", add_unknown_fields)
    state = load_project(hand_authored)
    resaved = save_project(tmp_path / "again.nanoscope", state)
    manifest = json.loads(_members(resaved)[project.MANIFEST_NAME])

    assert manifest["workspace"]["future_workspace_flag"] == ["leave", "me"]
    assert manifest["workspace"]["artifacts"][0]["vendor"] == {
        "name": "Example", "opaque": [1, None, False]}


def test_v1_through_v10_without_workspace_load_as_empty(tmp_path):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(artifacts=[WorkspaceArtifact("discarded", "test")])))

    def make_v10(manifest, _members):
        manifest["version"] = 10
        manifest.pop("workspace")

    old = _replace_manifest(seed, tmp_path / "v10.nanoscope", make_v10)
    workspace = load_project(old).workspace
    assert workspace == AnalyticalWorkspace()


def test_a_v11_manifest_missing_its_workspace_is_malformed(tmp_path):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState())

    def remove_workspace(manifest, _members):
        assert manifest["version"] == 11
        manifest.pop("workspace")

    bad = _replace_manifest(seed, tmp_path / "missing.nanoscope", remove_workspace)
    with pytest.raises(ProjectError, match="workspace.*missing"):
        load_project(bad)


def test_a_v10_reader_refuses_v11_instead_of_silently_stripping_workspace(
        tmp_path, monkeypatch):
    path = save_project(tmp_path / "v11.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(artifacts=[WorkspaceArtifact("plot", "nanoscope.plot-window")])))

    monkeypatch.setattr(project, "VERSION", 10)
    with pytest.raises(ProjectVersionError, match="needs a newer NanoScope"):
        load_project(path)


@pytest.mark.parametrize("workspace", [
    None,
    [],
    {},
    {"version": 1},
    {"version": True, "artifacts": []},
    {"version": 1, "artifacts": {}},
    {"version": 1, "artifacts": [None]},
    {"version": 1, "artifacts": [{
        "artifact_id": "a", "kind": "k", "schema_version": 1, "tables": []}]},
])
def test_malformed_workspace_sections_raise_project_error(tmp_path, workspace):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState())

    def corrupt(manifest, _members):
        manifest["workspace"] = workspace

    bad = _replace_manifest(seed, tmp_path / "bad.nanoscope", corrupt)
    with pytest.raises(ProjectError, match="workspace"):
        load_project(bad)


def test_missing_workspace_table_or_blob_is_a_project_error(tmp_path):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
            "a", "plugin", tables={"scores": _scores()}, blobs={"state": b"state"})])))

    for resource, logical in (("tables", "scores"), ("blobs", "state")):
        members = _members(seed)
        manifest = json.loads(members[project.MANIFEST_NAME])
        member = manifest["workspace"]["artifacts"][0][resource][logical]
        members.pop(member)
        missing = tmp_path / f"missing-{resource}.nanoscope"
        with zipfile.ZipFile(missing, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        with pytest.raises(ProjectError, match=f"workspace .*{logical!r}.*missing"):
            load_project(missing)


def test_duplicate_ids_and_table_blob_name_collisions_are_refused_on_save(tmp_path):
    duplicate_ids = AnalyticalWorkspace(artifacts=[
        WorkspaceArtifact("same", "one"), WorkspaceArtifact("same", "two")])
    with pytest.raises(ValueError, match="artifact_id .*duplicated"):
        save_project(tmp_path / "duplicate.nanoscope", ProjectState(workspace=duplicate_ids))

    name_collision = AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
        "one", "test", tables={"result": _scores()}, blobs={"result": b"bytes"})])
    with pytest.raises(ValueError, match="resource name .*duplicated"):
        save_project(tmp_path / "collision.nanoscope", ProjectState(workspace=name_collision))


def test_unsafe_logical_names_and_canonical_extra_collisions_are_refused(tmp_path):
    unsafe = AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
        "one", "test", tables={"../scores": _scores()})])
    with pytest.raises(ValueError, match="logical name, not a path"):
        save_project(tmp_path / "unsafe.nanoscope", ProjectState(workspace=unsafe))

    shadow = AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
        "one", "test", extra={"kind": "shadow"})])
    with pytest.raises(ValueError, match="cannot replace canonical key.*kind"):
        save_project(tmp_path / "shadow.nanoscope", ProjectState(workspace=shadow))

    non_json_key = AnalyticalWorkspace(extra={1: "would silently stringify"})
    with pytest.raises(ValueError, match="extra-field names must be strings"):
        save_project(tmp_path / "key.nanoscope", ProjectState(workspace=non_json_key))


def test_duplicate_resource_member_references_are_refused_on_load(tmp_path):
    seed = save_project(tmp_path / "seed.nanoscope", ProjectState(workspace=
        AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
            "a", "plugin", tables={"scores": _scores()}, blobs={"state": b"state"})])))

    def alias_members(manifest, _members):
        artifact = manifest["workspace"]["artifacts"][0]
        artifact["blobs"]["state"] = artifact["tables"]["scores"]

    aliased = _replace_manifest(seed, tmp_path / "aliased.nanoscope", alias_members)
    with pytest.raises(ProjectError, match="referenced twice"):
        load_project(aliased)
