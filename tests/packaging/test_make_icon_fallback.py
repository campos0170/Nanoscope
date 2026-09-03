"""Fail-closed coverage for the macOS 26 ``iconutil`` compilation fallback."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(scope="module")
def make_icon_module():
    path = Path(__file__).resolve().parents[2] / "packaging" / "make_icon.py"
    spec = importlib.util.spec_from_file_location("nanoscope_make_icon", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_iconset(module, root: Path) -> None:
    root.mkdir()
    for px in module.ICONSET_POINTS:
        for scale, name in ((1, f"icon_{px}x{px}.png"),
                            (2, f"icon_{px}x{px}@2x.png")):
            size = px * scale
            Image.new("RGBA", (size, size), (size % 251, 50, 100, 255)).save(root / name)


def _write_icns(module, iconset: Path, output: Path) -> None:
    by_width = {}
    for logical_size, name in module.ICNS_FRAMES.items():
        width = logical_size[0] * logical_size[2]
        with Image.open(iconset / name) as image:
            by_width[width] = image.convert("RGBA")
    by_width[1024].save(
        output, format="ICNS",
        append_images=[by_width[width] for width in sorted(by_width) if width != 1024])


def test_pixel_gate_accepts_every_matching_embedded_frame(make_icon_module, tmp_path):
    rendered = tmp_path / "rendered.iconset"
    _write_iconset(make_icon_module, rendered)
    checked_in = tmp_path / "NanoScope.icns"
    _write_icns(make_icon_module, rendered, checked_in)

    make_icon_module._assert_icns_matches_iconset(rendered, checked_in)


@pytest.mark.parametrize("mutation", ["missing", "extra", "pixel"])
def test_pixel_gate_rejects_any_incomplete_or_changed_iconset(
        make_icon_module, tmp_path, mutation):
    rendered = tmp_path / "rendered.iconset"
    _write_iconset(make_icon_module, rendered)
    checked_in = tmp_path / "NanoScope.icns"
    _write_icns(make_icon_module, rendered, checked_in)
    first = make_icon_module.ICONSET_FILES[0]
    if mutation == "missing":
        (rendered / first).unlink()
    elif mutation == "extra":
        Image.new("RGBA", (1, 1)).save(rendered / "unexpected.png")
    else:
        changed = make_icon_module.ICNS_FRAMES[(16, 16, 2)]
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(rendered / changed)

    with pytest.raises(RuntimeError):
        make_icon_module._assert_icns_matches_iconset(rendered, checked_in)


def test_compile_failure_writes_and_validates_a_pillow_fallback(
        make_icon_module, tmp_path, monkeypatch):
    rendered = tmp_path / "rendered.iconset"
    _write_iconset(make_icon_module, rendered)
    checked_in = tmp_path / "NanoScope.icns"
    checked_in.write_bytes(b"old-icon-must-not-be-used")

    def fake_run(command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(make_icon_module.subprocess, "run", fake_run)
    result = make_icon_module._compile_or_fallback(
        "iconutil", rendered, checked_in)

    assert result == "pillow-fallback"
    make_icon_module._assert_icns_matches_iconset(rendered, checked_in)


def test_failed_fallback_validation_preserves_the_existing_icon(
        make_icon_module, tmp_path, monkeypatch):
    rendered = tmp_path / "rendered.iconset"
    _write_iconset(make_icon_module, rendered)
    checked_in = tmp_path / "NanoScope.icns"
    original = b"existing-icon-must-survive"
    checked_in.write_bytes(original)

    def fake_run(command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(make_icon_module.subprocess, "run", fake_run)

    def reject(_iconset, _candidate):
        raise RuntimeError("injected pixel mismatch")

    monkeypatch.setattr(make_icon_module, "_assert_icns_matches_iconset", reject)
    with pytest.raises(RuntimeError, match="validated Pillow fallback failed"):
        make_icon_module._compile_or_fallback("iconutil", rendered, checked_in)
    assert checked_in.read_bytes() == original
