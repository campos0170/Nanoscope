"""Capture an immutable, path-free build record and scoped dependency inventory.

Used by both PyInstaller specs. Importing this module does not generate files,
import Qt/TensorFlow, install dependencies, or change the environment.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL_RUNTIME = ("tensorflow", "tensorflow-macos", "keras", "numba")
BUILD_TOOLS = ("pyinstaller", "pyinstaller-hooks-contrib", "setuptools", "wheel", "pillow")


def marketing_version(root=ROOT):
    tree = ast.parse((Path(root) / "nanoscope" / "_version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and len(value.split(".")) == 3 and all(p.isdigit() for p in value.split(".")):
                return value
    raise ValueError("No three-component marketing version in nanoscope/_version.py")


def source_digest(root=ROOT):
    """Hash packaged application/build inputs, including uncommitted source files.

    Excludes caches, prior builds and generated metadata. This identifies source
    inputs, not a bit-for-bit binary reproducibility claim.
    """
    root = Path(root)
    paths = [root / "pyproject.toml"]
    for directory in (root / "nanoscope", root / "packaging"):
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and not any(part in {"__pycache__", ".DS_Store"} for part in p.parts)
                     and p.suffix not in {".pyc", ".pyo"} and p.name != "_build_info.json")
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError("Source inputs must not contain symlinks")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest()


def _git(root, *args):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_manifest(root=ROOT, *, distribution=metadata.distribution):
    """Transitive runtime closure only, plus separately identified build tools.

    Metadata versions/requirements only: never direct_url.json, installer paths,
    the full Conda environment, credentials, or shell environment variables.
    A captured environment is not represented as solver-valid when it is not.
    """
    project = tomllib.loads((Path(root) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    required = list(project["dependencies"])
    optional, missing_optional = [], []
    for name in OPTIONAL_RUNTIME:
        try:
            distribution(name)
        except metadata.PackageNotFoundError:
            missing_optional.append(name)
        else:
            optional.append(name)
    queue = [(requirement, "application") for requirement in required]
    queue += [(name, "optional runtime") for name in optional]
    packages, requirements, issues, seen = {}, {}, set(), set()
    environment = default_environment()
    environment["extra"] = ""
    while queue:
        raw, parent = queue.pop()
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate(environment):
            continue
        name = canonicalize_name(req.name)
        if req.url:
            issues.add(f"{parent} declares a direct-URL dependency for {name}; URL not retained")
        try:
            dist = distribution(req.name)
        except metadata.PackageNotFoundError:
            issues.add(f"{parent} requires {name}, which is not installed")
            continue
        version = dist.version
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            issues.add(f"{parent} requires {name}{req.specifier}; installed {version}")
        packages[name] = version
        if name in seen:
            continue
        seen.add(name)
        children = dist.requires or []
        requirements[name] = sorted({canonicalize_name(Requirement(child).name) for child in children
                                     if Requirement(child).marker is None or Requirement(child).marker.evaluate(environment)})
        queue.extend((child, name) for child in children)
    tools = {}
    for name in BUILD_TOOLS:
        try:
            tools[name] = distribution(name).version
        except metadata.PackageNotFoundError:
            pass
    return {"schema_version": 1,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "python": platform.python_version(),
            "platform": platform.system(), "platform_release": platform.release(),
            "architecture": platform.machine(), "packages": dict(sorted(packages.items())),
            "runtime_roots": sorted(canonicalize_name(Requirement(r).name) for r in required),
            "optional_runtime_present": optional, "optional_runtime_absent": missing_optional,
            "requirements": dict(sorted(requirements.items())), "build_tools": dict(sorted(tools.items())),
            "compatibility_issues": sorted(issues),
            "reproduction_scope": "Captured installed versions for this platform/Python; compatibility notes must be resolved before claiming a clean solver-reproducible environment. No upgrades were performed."}


def bundle_version(now):
    """A UTC-minute ordinal within Apple's conservative 4/2/2 digit limits.

    Separate-minute builds sort monotonically if the clock is correct. Builds
    in one minute share this number; the full build_id provides finer identity.
    Fail rather than wrapping the ordinal if its finite range is exhausted.
    """
    if now.tzinfo is None:
        raise ValueError("Build time must include a time zone")
    elapsed = now.astimezone(timezone.utc) - datetime(2026, 1, 1, tzinfo=timezone.utc)
    minute = elapsed.days * 1440 + elapsed.seconds // 60
    ordinal = minute + 10000  # Apple requires a positive first component.
    if minute < 0 or ordinal >= 100000000:
        raise ValueError("Build time is outside the supported CFBundleVersion epoch")
    return f"{ordinal // 10000}.{ordinal // 100 % 100}.{ordinal % 100}"


def create_build_record(root=ROOT, *, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Build time must include a time zone")
    now = now.astimezone(timezone.utc)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    digest = source_digest(root)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    return {"schema_version": 1, "version": marketing_version(root),
            "build_id": f"{stamp}-{(commit or 'nogit')[:12]}-{'dirty-' if status else ''}{digest[:12]}",
            "bundle_version": bundle_version(now),
            "built_at_utc": now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "git_commit": commit, "git_dirty": None if status is None else bool(status),
            "source_sha256": digest, "environment": environment_manifest(root)}


def write_build_record(root=ROOT, destination=None):
    root = Path(root)
    destination = Path(destination) if destination else root / "build" / "release-metadata"
    destination.mkdir(parents=True, exist_ok=True)
    record = create_build_record(root)
    path = destination / "_build_info.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record, path


def write_windows_version_resource(record, destination, root=ROOT):
    version = str(record["version"])
    components = tuple(int(value) for value in version.split(".")) + (0,)
    if len(components) != 4 or any(value < 0 or value > 65535 for value in components):
        raise ValueError("Windows version components must fit unsigned 16-bit integers")
    template = (Path(root) / "packaging" / "resources" / "win_version_info.txt").read_text(encoding="utf-8")
    rendered = template.replace("@VERSION_TUPLE@", repr(components)).replace(
        "@APP_VERSION@", repr(version)).replace("@BUILD_ID@", repr(record["build_id"]))
    path = Path(destination) / "win_version_info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def write_environment_capture(destination, root=ROOT):
    """Explicit maintenance action; never invoked by importing or opening About."""
    destination = Path(destination)
    manifest = environment_manifest(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pins = ["# Captured app runtime dependency closure; platform/Python and compatibility notes are in the companion JSON.",
            "# Not an assertion that this existing environment is solver-consistent. No upgrades performed."]
    pins.extend(f"{name}=={version}" for name, version in manifest["packages"].items())
    destination.with_suffix(".constraints.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--capture-environment", type=Path)
    args = parser.parse_args()
    if args.version:
        print(marketing_version())
    elif args.capture_environment:
        result = write_environment_capture(args.capture_environment)
        print(json.dumps({"packages": len(result["packages"]), "compatibility_issues": result["compatibility_issues"]}, indent=2))
    else:
        result, path = write_build_record()
        print(result["build_id"])
