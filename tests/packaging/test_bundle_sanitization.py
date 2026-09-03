"""Release bundles must not retain loader or metadata paths from the build machine."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_bundle_under_test", PACKAGING / "verify_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _otool(*rpaths: str) -> str:
    blocks = []
    for index, rpath in enumerate(rpaths):
        blocks.extend([
            f"Load command {index}",
            "          cmd LC_RPATH",
            "      cmdsize 88",
            f"         path {rpath} (offset 12)",
        ])
    return "\n".join(blocks)


def test_rpath_parser_preserves_spaces_and_only_foreign_absolute_paths_are_dirty(verifier):
    output = _otool(
        "/opt/gfortran cross/lib",
        "@loader_path/../..",
        "/System/Library/Frameworks",
        "/usr/lib/swift",
        "/usr/local/lib",
    )

    paths = verifier._rpaths_from_otool(output)

    assert paths[0] == "/opt/gfortran cross/lib"
    assert [path for path in paths if verifier._is_non_system_absolute_rpath(path)] == [
        "/opt/gfortran cross/lib", "/usr/local/lib"]


def test_bundle_cleaner_deletes_foreign_rpath_and_verifies_the_result(
        verifier, tmp_path, monkeypatch):
    app = tmp_path / "NanoScope.app"
    binary = app / "Contents" / "Frameworks" / "libopenblas.dylib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"mach-o stand-in")
    foreign = "/opt/gfortran-darwin-arm64-cross/lib"
    inspections = iter([
        _otool(foreign, "@loader_path/../.."),
        _otool("@loader_path/../.."),
    ])
    calls: list[list[str]] = []

    monkeypatch.setattr(verifier, "_macho_files", lambda _app: [binary])

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "otool":
            return subprocess.CompletedProcess(argv, 0, stdout=next(inspections), stderr="")
        assert argv == ["install_name_tool", "-delete_rpath", foreign, str(binary)]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)

    verifier.clean_non_system_absolute_rpaths(app)

    assert [call[0] for call in calls] == ["otool", "install_name_tool", "otool"]


def test_bundle_cleaner_fails_closed_when_install_name_tool_cannot_rewrite(
        verifier, tmp_path, monkeypatch):
    app = tmp_path / "NanoScope.app"
    binary = app / "Contents" / "Frameworks" / "bad.dylib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"mach-o stand-in")
    monkeypatch.setattr(verifier, "_macho_files", lambda _app: [binary])

    def fake_run(argv, **kwargs):
        if argv[0] == "otool":
            return subprocess.CompletedProcess(
                argv, 0, stdout=_otool("/Users/builder/private/lib"), stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no room in load commands")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="could not remove LC_RPATH"):
        verifier.clean_non_system_absolute_rpaths(app)


def test_direct_url_cleanup_is_scoped_to_dist_info_and_keeps_other_metadata(verifier, tmp_path):
    app = tmp_path / "NanoScope.app"
    dist_info = app / "Contents" / "Resources" / "nanoscope-0.1.0.dist-info"
    dist_info.mkdir(parents=True)
    direct_url = dist_info / "direct_url.json"
    direct_url.write_text('{"url":"file:///Users/builder/checkout","dir_info":{"editable":true}}')
    useful = dist_info / "METADATA"
    useful.write_text("Name: nanoscope\nVersion: 0.1.0\n")
    unrelated = app / "Contents" / "Resources" / "fixtures" / "direct_url.json"
    unrelated.parent.mkdir()
    unrelated.write_text('{"url":"https://example.test/data"}')

    verifier.clean_direct_url_metadata(app)

    assert not direct_url.exists()
    assert useful.exists()
    assert unrelated.exists()
