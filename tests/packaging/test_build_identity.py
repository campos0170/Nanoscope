"""Build identity is immutable, scoped, and excludes user data and machine paths."""
from datetime import datetime, timedelta, timezone
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nanoscope import __version__
from nanoscope.build_info import diagnostic_summary

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nanoscope_packaging_build_identity", ROOT / "packaging/build_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def test_one_marketing_version_source():
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert identity.marketing_version() == __version__ == "0.1.0"
    assert "version" not in project["project"]
    assert project["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "nanoscope._version.__version__"
    for filename in ("NanoScope.spec", "NanoScope-win.spec", "build_app.sh", "build_app.ps1"):
        assert 'APP_VERSION = "0.1.0"' not in (ROOT / "packaging" / filename).read_text()


@pytest.fixture
def source(tmp_path):
    (tmp_path / "nanoscope").mkdir()
    (tmp_path / "packaging").mkdir()
    (tmp_path / "nanoscope/_version.py").write_text('__version__ = "0.1.0"\n')
    (tmp_path / "nanoscope/example.py").write_text("value = 1\n")
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["top>=1"]\n')
    return tmp_path


def test_source_digest_tracks_code_not_prior_builds_or_private_files(source):
    original = identity.source_digest(source)
    (source / "build").mkdir()
    (source / "build/private-recording.csv").write_text("private")
    (source / "nanoscope/_build_info.json").write_text("old generated record")
    assert identity.source_digest(source) == original
    (source / "nanoscope/example.py").write_text("value = 2\n")
    assert identity.source_digest(source) != original


def test_scoped_transitive_inventory_records_incompatibility_without_private_urls(source, monkeypatch):
    monkeypatch.setattr(identity, "OPTIONAL_RUNTIME", ())
    monkeypatch.setattr(identity, "BUILD_TOOLS", ())
    distributions = {
        "top": SimpleNamespace(version="1.0", requires=["bottom>=2", "unused; python_version>'9'"]),
        "bottom": SimpleNamespace(version="1.0", requires=[]),
        "unrelated-anaconda-tool": SimpleNamespace(version="9.0", requires=[]),
    }
    def lookup(name):
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]
    manifest = identity.environment_manifest(source, distribution=lookup)
    assert manifest["packages"] == {"bottom": "1.0", "top": "1.0"}
    assert manifest["compatibility_issues"] == ["top requires bottom>=2; installed 1.0"]
    assert str(source) not in json.dumps(manifest)
    assert "unrelated-anaconda-tool" not in json.dumps(manifest)


def test_build_record_uses_build_time_and_source_not_only_marketing_version(source, monkeypatch):
    monkeypatch.setattr(identity, "_git", lambda root, *args: "a" * 40 if args[0] == "rev-parse" else " M nanoscope/example.py")
    monkeypatch.setattr(identity, "environment_manifest", lambda root: {"packages": {}})
    record = identity.create_build_record(source, now=datetime(2026, 9, 11, 13, 15, 21, 123456, tzinfo=timezone.utc))
    other = identity.create_build_record(source, now=datetime(2026, 9, 11, 13, 16, 22, 123456, tzinfo=timezone.utc))
    assert record["git_dirty"] is True
    assert record["git_commit"] == "a" * 40
    assert record["source_sha256"] == identity.source_digest(source)
    assert record["bundle_version"] != other["bundle_version"]
    assert record["build_id"] != other["build_id"]
    assert len(record["bundle_version"].split(".")) == 3
    assert all(component.isdigit() for component in record["bundle_version"].split("."))
    assert str(source) not in json.dumps(record)
    assert "example.py" not in json.dumps(record)


def test_bundle_version_obeys_apple_component_limits_and_ordering():
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for minute in (0, 1, 99, 100, 9999, 10000, 99989998):
        first = identity.bundle_version(epoch + timedelta(minutes=minute))
        second = identity.bundle_version(epoch + timedelta(minutes=minute + 1))
        components = first.split(".")
        assert all(part.isdigit() for part in components)
        assert all(len(part) <= limit for part, limit in zip(components, (4, 2, 2)))
        assert 1 <= int(components[0]) <= 9999
        assert all(0 <= int(part) <= 99 for part in components[1:])
        assert tuple(map(int, first.split("."))) < tuple(map(int, second.split(".")))
    assert identity.bundle_version(epoch) == "1.0.0"
    with pytest.raises(ValueError, match="epoch"):
        identity.bundle_version(epoch - timedelta(seconds=1))
    with pytest.raises(ValueError, match="epoch"):
        identity.bundle_version(epoch + timedelta(minutes=99990000))


def test_same_minute_builds_have_fine_grained_id_not_unique_bundle_number(source, monkeypatch):
    monkeypatch.setattr(identity, "_git", lambda *args: None)
    monkeypatch.setattr(identity, "environment_manifest", lambda root: {})
    instant = datetime(2026, 9, 11, 13, 15, 21, 123456, tzinfo=timezone.utc)
    first = identity.create_build_record(source, now=instant)
    second = identity.create_build_record(source, now=instant + timedelta(microseconds=1))
    assert first["bundle_version"] == second["bundle_version"]
    assert first["build_id"] != second["build_id"]


def test_unknown_git_is_not_reported_clean(source, monkeypatch):
    monkeypatch.setattr(identity, "_git", lambda *args: None)
    monkeypatch.setattr(identity, "environment_manifest", lambda root: {})
    assert identity.create_build_record(source)["git_dirty"] is None


def test_diagnostics_allowlist_excludes_user_paths_and_logs():
    secret = "/Users/private/science/recording.abf"
    text = diagnostic_summary({"version": "0.1.0", "build_id": "test-build", "git_dirty": False,
                               "project_path": secret, "recordings": [secret], "logs": secret,
                               "environment": {"packages": {}, "installer_path": secret}})
    assert secret not in text
    assert sys.executable not in text
    assert "Source working tree: clean" in text
    assert "Local diagnostic summary only" in text


def test_windows_resource_uses_same_marketing_version_and_build_id(tmp_path):
    resource = identity.write_windows_version_resource({"version": "0.1.0", "build_id": "2026-test-build"}, tmp_path)
    text = resource.read_text()
    compile(text, "version_resource", "eval")
    assert "@APP_VERSION@" not in text
    assert "2026-test-build" in text
    assert "filevers=(0, 1, 0, 0)" in text


def test_both_specs_embed_the_generated_build_record():
    for filename in ("NanoScope.spec", "NanoScope-win.spec"):
        text = (ROOT / "packaging" / filename).read_text()
        assert "write_build_record(ROOT)" in text
        assert '(str(BUILD_RECORD_PATH), "nanoscope")' in text
    assert '"CFBundleVersion": BUILD_RECORD["bundle_version"]' in (ROOT / "packaging/NanoScope.spec").read_text()
