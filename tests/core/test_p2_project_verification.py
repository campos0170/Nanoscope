"""Verified current saves and explicit v12-to-v13 conversion backup gates."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from nanoscope.core import project, project_guard, atomic_write
from nanoscope.core.project import WorkspaceArtifact, AnalyticalWorkspace, ProjectState


def state(tag="A"):
    return ProjectState(workspace=AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
        "opaque.figure", "plugin.figure", payload={"title": tag, "unknown": {"none": None}},
        tables={"values": pd.DataFrame({"value": [1., 2.]}, index=pd.Index([4, 9], name="row"))},
        blobs={"opaque": b"preserved"}, extra={"future": [1, 2]})],
        extra={"figure_workspace_v1": {"future": "kept"}}))


def test_verified_save_reopens_exact_logical_workspace(tmp_path):
    original = state()
    target = tmp_path / "verified.nanoscope"
    project.save_project(target, original, verify_workspace=True)
    project.verify_workspace_equal(original.workspace, project.load_project(target).workspace)


@pytest.mark.parametrize("corruption", ["payload", "table", "blob", "extra", "workspace_extra"])
def test_corrupt_stage_refuses_replacement_and_names_affected_artifact(tmp_path, monkeypatch, corruption):
    target = tmp_path / "existing.nanoscope"
    project.save_project(target, state("Original"))
    original_bytes = target.read_bytes()
    write = project._write_workspace

    def corrupt(zf, workspace):
        changed = deepcopy(workspace)
        artifact = changed.artifacts[0]
        if corruption == "payload":
            artifact.payload["title"] = "Corrupted"
        elif corruption == "table":
            artifact.tables["values"].iloc[0, 0] = 99.
        elif corruption == "blob":
            artifact.blobs["opaque"] = b"changed"
        elif corruption == "extra":
            artifact.extra["future"] = []
        else:
            changed.extra.clear()
        return write(zf, changed)

    monkeypatch.setattr(project, "_write_workspace", corrupt)
    with pytest.raises(project.ProjectError, match="verification failed") as error:
        project.save_project(target, state("New"), verify_workspace=True)
    if corruption != "workspace_extra":
        assert "opaque.figure" in str(error.value)
    assert target.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))


def backup_under_guard(target):
    stamp = project_guard.project_disk_stamp(target)
    return project_guard.guarded_project_write(target, stamp, None,
        lambda: project_guard.retain_verified_project_backup(target))[0]


def save_v12_seed(monkeypatch, target, value):
    # Only fixture creation uses the old gate; conversion runs under the current writer.
    with monkeypatch.context() as historical:
        historical.setattr(project, "VERSION", 12)
        project.save_project(target, value)
    assert project.VERSION == 13
    assert project.read_project_container_version(target) == 12


def test_conversion_backup_is_byte_identical_and_repeated_backup_reuses_it(tmp_path, monkeypatch):
    target = tmp_path / "study.nanoscope"
    save_v12_seed(monkeypatch, target, state())
    original = target.read_bytes()
    first = backup_under_guard(target)
    second = backup_under_guard(target)
    assert first == second == Path(str(target) + ".v12.bak")
    assert first.read_bytes() == target.read_bytes() == original
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_unrelated_or_invalid_backup_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "study.nanoscope"
    save_v12_seed(monkeypatch, target, state())
    collision = Path(str(target) + ".v12.bak")
    save_v12_seed(monkeypatch, collision, state("Other original"))
    before = collision.read_bytes()
    protected = backup_under_guard(target)
    assert protected == Path(str(target) + ".v12.2.bak")
    assert protected.read_bytes() == target.read_bytes()
    assert collision.read_bytes() == before


def test_save_as_new_target_needs_no_backup_and_does_not_touch_source(tmp_path):
    source = tmp_path / "source.nanoscope"
    project.save_project(source, state())
    before = source.read_bytes()
    target = tmp_path / "new.nanoscope"
    assert backup_under_guard(target) is None
    project.save_project(target, state("New"), verify_workspace=True)
    assert source.read_bytes() == before and not list(tmp_path.glob("*.bak"))


def test_backup_publication_failure_preserves_original_and_cleans_stage(tmp_path, monkeypatch):
    target = tmp_path / "study.nanoscope"
    save_v12_seed(monkeypatch, target, state())
    before = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected backup publication failure")

    monkeypatch.setattr(project_guard.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        backup_under_guard(target)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.bak")) and not list(tmp_path.glob("*.tmp"))


def test_stamp_conflict_precedes_backup_creation(tmp_path):
    target = tmp_path / "study.nanoscope"
    project.save_project(target, state("Observed"))
    stamp = project_guard.project_disk_stamp(target)
    project.save_project(target, state("Other writer"))
    before = target.read_bytes()
    with pytest.raises(project_guard.ProjectSaveConflict):
        project_guard.guarded_project_write(target, stamp, None,
            lambda: project_guard.retain_verified_project_backup(target))
    assert target.read_bytes() == before and not list(tmp_path.glob("*.bak"))


def test_writer_identity_describes_current_build_without_changing_version_gate(tmp_path, monkeypatch):
    import json
    import zipfile
    from nanoscope import build_info

    monkeypatch.setattr(build_info, "build_info", lambda: {
        "version": "current", "build_id": "fixture-build", "git_commit": "abc123",
        "source_sha256": "sha", "environment": {"should": "not be copied"}})
    value = state()
    value.app_version = "previous-writer"
    target = tmp_path / "writer.nanoscope"
    project.save_project(target, value, verify_workspace=True)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read(project.MANIFEST_NAME))
    assert manifest["version"] == project.VERSION
    assert manifest["writer"] == {"version": "current", "build_id": "fixture-build", "git_commit": "abc123", "source_sha256": "sha"}
    assert manifest["app_version"] == "previous-writer"


def test_container_version_probe_reads_header_without_loading_tables(tmp_path, monkeypatch):
    target = tmp_path / "probe.nanoscope"
    project.save_project(target, state())

    def forbid(*args, **kwargs):
        raise AssertionError("Header inspection must not read workspace tables")

    monkeypatch.setattr(project, "_read_workspace", forbid)
    assert project.read_project_container_version(target) == project.VERSION
