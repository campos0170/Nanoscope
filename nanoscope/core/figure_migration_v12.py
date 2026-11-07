"""Pure, opt-in conversion of v12 built-ins; the project writer remains version 12.

Unknown artifacts, future schemas, snapshots and failed native conversions survive
unchanged. Identity collisions are detected before a caller can install any result.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid5

from .figure_codecs import (NATIVE_KINDS, MigrationReport, decode_legacy,
                            record_to_artifact, record_from_artifact)
from .figure_spec import FigureRecord, FigureValidationError, FIGURE_ARTIFACT_KIND

MIGRATION_NAMESPACE = UUID("9dc0a553-e9c5-5c31-8d2e-4a8601c77f89")


def migrated_figure_id(legacy_id: str) -> str:
    return str(uuid5(MIGRATION_NAMESPACE, legacy_id))


@dataclass(frozen=True)
class MigrationFailure:
    artifact_id: str
    reason: str


@dataclass(frozen=True)
class WorkspaceMigration:
    workspace: object
    records: tuple[FigureRecord, ...] = ()
    reports: tuple[MigrationReport, ...] = ()
    failures: tuple[MigrationFailure, ...] = ()


def migrate_workspace_v12(workspace) -> WorkspaceMigration:
    from .project import AnalyticalWorkspace

    if type(workspace.version) is not int or workspace.version != 1:
        return WorkspaceMigration(workspace, failures=(MigrationFailure(
            "workspace", f"Unsupported analytical workspace version {workspace.version!r}; retained unchanged"),))
    identities = [artifact.artifact_id for artifact in workspace.artifacts]
    if len(set(identities)) != len(identities):
        raise FigureValidationError("Workspace has duplicate artifact identities")
    candidates = []
    for artifact in workspace.artifacts:
        entry = NATIVE_KINDS.get(artifact.artifact_id)
        if (entry is not None and artifact.kind == entry[0]
                and type(artifact.schema_version) is int and artifact.schema_version == 1):
            target = "live." + migrated_figure_id(artifact.artifact_id)
            if target in identities:
                raise FigureValidationError(f"Migration identity collides with existing artifact: {target}")
            candidates.append(artifact.artifact_id)
    artifacts, records, reports, failures = [], [], [], []
    for artifact in workspace.artifacts:
        if artifact.artifact_id in candidates:
            try:
                decoded = decode_legacy(artifact, figure_id=migrated_figure_id(artifact.artifact_id),
                                        name=NATIVE_KINDS[artifact.artifact_id][2])
                converted = record_to_artifact(decoded.record)
            except (FigureValidationError, TypeError, ValueError, KeyError) as exc:
                failures.append(MigrationFailure(artifact.artifact_id, str(exc)))
                artifacts.append(artifact)
            else:
                artifacts.append(converted)
                records.append(decoded.record)
                reports.append(decoded.report)
        else:
            artifacts.append(artifact)
            if artifact.kind == FIGURE_ARTIFACT_KIND and artifact.schema_version == 1:
                try:
                    records.append(record_from_artifact(artifact))
                except (FigureValidationError, TypeError, ValueError, KeyError) as exc:
                    failures.append(MigrationFailure(artifact.artifact_id, str(exc)))
    return WorkspaceMigration(AnalyticalWorkspace(workspace.version, artifacts, dict(workspace.extra)),
                              tuple(records), tuple(reports), tuple(failures))
