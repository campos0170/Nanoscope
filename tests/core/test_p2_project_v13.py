"""The activated writer uses v13 and verifies every workspace before replacement."""
from copy import deepcopy
from pathlib import Path
import json
import zipfile

import pytest

from nanoscope.core import project
from nanoscope.core.figure_migration_v12 import migrate_workspace_v12


def test_v13_round_trip_preserves_all_migrated_figures_and_older_read_range(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "phase2" / "all_pages_v12.nanoscope"
    original_bytes = fixture.read_bytes()
    state = project.load_project(fixture)
    migration = migrate_workspace_v12(state.workspace)
    assert not migration.failures and len(migration.records) == 8
    state.workspace = migration.workspace
    target = tmp_path / "converted.nanoscope"
    project.save_project(target, state)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read(project.MANIFEST_NAME))
    assert manifest["version"] == project.VERSION == 13
    assert project.MIN_READ_VERSION == 1 and project.WORKSPACE_VERSION == 1
    assert len([a for a in manifest["workspace"]["artifacts"] if a["kind"] == "nanoscope.figure"]) == 8
    restored = project.load_project(target)
    project.verify_workspace_equal(state.workspace, restored.workspace)
    project.save_project(target, restored)
    project.verify_workspace_equal(restored.workspace, project.load_project(target).workspace)
    assert fixture.read_bytes() == original_bytes


def test_v13_verification_cannot_be_disabled_by_omitting_or_clearing_the_old_opt_in(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "phase2" / "all_pages_v12.nanoscope"
    state = project.load_project(fixture)
    state.workspace = migrate_workspace_v12(state.workspace).workspace
    target = tmp_path / "existing.nanoscope"
    project.save_project(target, state)
    before = target.read_bytes()
    writer = project._write_workspace

    def corrupt(archive, workspace):
        altered = deepcopy(workspace)
        altered.artifacts[0].payload["name"] = "Corrupted name"
        return writer(archive, altered)

    monkeypatch.setattr(project, "_write_workspace", corrupt)
    with pytest.raises(project.ProjectError, match="verification failed"):
        project.save_project(target, state, verify_workspace=False)
    assert target.read_bytes() == before
