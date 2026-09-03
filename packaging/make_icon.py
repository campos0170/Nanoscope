"""Generate packaging/resources/NanoScope.icns — run by build_app.sh, checked in as source.

The motif: a solid-state nanopore seen head-on (the ring) with a current trace running through it
and one deep translocation blockade at the centre. It has to survive being shrunk to 16 px in the
Dock and the Finder sidebar, so there are exactly two shapes and one accent colour — no fine
detail, no lettering, no gradient inside the mark.

Drawn with Pillow at 4x and downsampled. Pillow's `ImageDraw.line(width=...)` produces visible
seams where thick segments meet, so the trace is stroked by hand into a filled polygon (offset the
path along its normals, then close it) — clean edges at every size.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT_ICNS = HERE / "resources" / "NanoScope.icns"

ICONSET_POINTS = (16, 32, 128, 256, 512)
ICONSET_FILES = tuple(
    name
    for px in ICONSET_POINTS
    for name in (f"icon_{px}x{px}.png", f"icon_{px}x{px}@2x.png")
)

# Pillow exposes the actual raster entries embedded in an ICNS as logical
# (width, height, scale) triples.  ``iconutil -c iconset`` additionally writes
# synthesized 16 px and 32 px 1x PNGs whose straight/premultiplied-alpha
# representation is not byte-equivalent to the embedded raster.  Enumerating
# these eight entries directly is both exhaustive and immune to that conversion.
ICNS_FRAMES = {
    (16, 16, 2): "icon_16x16@2x.png",
    (32, 32, 2): "icon_32x32@2x.png",
    (128, 128, 1): "icon_128x128.png",
    (128, 128, 2): "icon_128x128@2x.png",
    (256, 256, 1): "icon_256x256.png",
    (256, 256, 2): "icon_256x256@2x.png",
    (512, 512, 1): "icon_512x512.png",
    (512, 512, 2): "icon_512x512@2x.png",
}

S = 4  # supersample factor
BASE = 1024

# App theme (nanoscope/gui/app.py::_apply_theme), so the icon and the window agree.
BG_TOP = (30, 35, 43)
BG_BOTTOM = (12, 15, 18)
RING = (48, 61, 76)
APERTURE = (17, 21, 26)
ACCENT = (79, 195, 247)

# macOS icon grid: the artwork sits inside ~80% of the canvas with a squircle-ish radius.
MARGIN = 0.098
RADIUS = 0.185


def _rounded_mask(size: int, box: tuple[float, float, float, float], radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    return mask


def _vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    grad = Image.new("RGB", (1, size))
    px = grad.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        px[0, y] = tuple(int(round(a + (b - a) * t)) for a, b in zip(top, bottom))
    return grad.resize((size, size), Image.BILINEAR)


def _blockade_path(x0: float, x1: float, y_base: float, depth: float, n: int = 400) -> list:
    """Baseline -> one deep, flat-bottomed blockade -> baseline.

    Two back-to-back logistics rather than a literal step: a real translocation has finite rise
    and fall, and the softened corners keep the shape reading as an *event* instead of a glitch
    once the icon is 16 px wide.
    """
    cx = (x0 + x1) / 2
    span = (x1 - x0) / 2
    half = span * 0.185      # half-width of the flat bottom
    edge = span * 0.030      # transition width
    pts = []
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        outside = 1.0 / (1.0 + math.exp(-(abs(x - cx) - half) / edge))
        pts.append((x, y_base + depth * (1.0 - outside)))
    return pts


def _stroke(points: list, width: float) -> list:
    """Return a closed polygon that is `points` stroked to `width`, with real mitre-free joins.

    Offsetting each vertex along the average of its adjacent segment normals; the path here is a
    function graph with bounded slope, so no self-intersection cleanup is needed.
    """
    h = width / 2
    normals = []
    for i, (x, y) in enumerate(points):
        px, py = points[max(i - 1, 0)]
        nx, ny = points[min(i + 1, len(points) - 1)]
        dx, dy = nx - px, ny - py
        length = math.hypot(dx, dy) or 1.0
        normals.append((-dy / length, dx / length))
    upper = [(x + nx * h, y + ny * h) for (x, y), (nx, ny) in zip(points, normals)]
    lower = [(x - nx * h, y - ny * h) for (x, y), (nx, ny) in zip(points, normals)]
    return upper + lower[::-1]


def render(size: int) -> Image.Image:
    n = size * S
    canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))

    m = MARGIN * n
    box = (m, m, n - m, n - m)
    rad = RADIUS * n

    body = _vertical_gradient(n, BG_TOP, BG_BOTTOM).convert("RGBA")
    canvas.paste(body, (0, 0), _rounded_mask(n, box, rad))

    d = ImageDraw.Draw(canvas)
    cx = cy = n / 2

    # The pore, head-on: a thick annulus drawn as two filled discs so the edges are clean.
    r_out = n * 0.300
    ring_w = n * 0.060
    d.ellipse((cx - r_out, cy - r_out, cx + r_out, cy + r_out), fill=RING)
    r_in = r_out - ring_w
    d.ellipse((cx - r_in, cy - r_in, cx + r_in, cy + r_in), fill=APERTURE)

    # The trace: baseline in the upper third, the blockade plunging through the aperture.
    lw = n * 0.050
    y_base = cy - n * 0.115
    path = _blockade_path(m + n * 0.055, n - m - n * 0.055, y_base, depth=n * 0.255)
    d.polygon(_stroke(path, lw), fill=ACCENT)
    # Round caps at the two open ends.
    for x, y in (path[0], path[-1]):
        d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2), fill=ACCENT)

    # A hairline top highlight sells the rounded rect as a physical tile at large sizes.
    hl = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle(box, radius=rad, outline=(255, 255, 255, 28),
                                         width=max(int(n * 0.005), 1))
    canvas = Image.alpha_composite(canvas, hl)

    return canvas.resize((size, size), Image.LANCZOS)


#: Sizes Windows actually pulls out of a .ico: 16 (tree/tray), 32 (taskbar, small tiles), 48
#: (Explorer medium), 256 (large icons + the Alt-Tab / task-view thumbnail). Pillow writes them
#: all into one file, so the .ico is checked in and the Windows build needs no icon toolchain.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT_ICO = OUT_ICNS.parent / "NanoScope.ico"


def write_ico(path: Path = OUT_ICO) -> Path:
    """Render the same mark into a multi-resolution Windows .ico.

    Each size is drawn at its own resolution rather than downscaled from one big bitmap:
    `render()` already draws at 4x and LANCZOS-downsamples internally, and re-scaling a 256 px
    render to 16 px on top of that turns the two-shape mark into grey mush.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(px) for px in ICO_SIZES]
    frames[-1].save(path, format="ICO",
                    sizes=[(px, px) for px in ICO_SIZES], append_images=frames[:-1])
    return path


def _assert_icns_matches_iconset(rendered: Path, candidate: Path) -> None:
    """Require every actual ICNS raster to match the newly rendered iconset.

    This is the fail-closed gate for macOS releases whose ``iconutil`` can extract an
    existing ``.icns`` but rejects the same valid iconset on compilation.  Frame count,
    logical size, dimensions and every decoded RGBA pixel must agree; container and PNG
    metadata are intentionally irrelevant to the icon displayed by macOS.
    """
    required = set(ICONSET_FILES)
    if not rendered.is_dir():
        raise RuntimeError(f"rendered iconset was not created: {rendered}")
    entries = list(rendered.iterdir())
    names = {entry.name for entry in entries}
    if names != required or any(not entry.is_file() or entry.is_symlink() for entry in entries):
        missing = sorted(required - names)
        extra = sorted(names - required)
        raise RuntimeError(
            f"rendered iconset has the wrong frames (missing={missing}, extra={extra})")
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"candidate ICNS does not exist as a regular file: {candidate}")

    try:
        with Image.open(candidate) as container:
            actual_sizes = {tuple(size) for size in container.info.get("sizes", ())}
    except Exception as exc:
        raise RuntimeError(f"could not decode candidate ICNS {candidate}: {exc}") from exc
    expected_sizes = set(ICNS_FRAMES)
    if actual_sizes != expected_sizes:
        raise RuntimeError(
            "candidate ICNS has the wrong embedded frames "
            f"(missing={sorted(expected_sizes - actual_sizes)}, "
            f"extra={sorted(actual_sizes - expected_sizes)})")

    for logical_size, name in ICNS_FRAMES.items():
        try:
            with Image.open(rendered / name) as want_image, Image.open(candidate) as got_image:
                got_image.size = logical_size
                got_image.load()
                want = want_image.convert("RGBA")
                got = got_image.convert("RGBA")
                if want.size != got.size or want.tobytes() != got.tobytes():
                    raise RuntimeError(f"candidate icon frame does not match rendered pixels: {name}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"could not decode icon frame {name}: {exc}") from exc


def _write_pillow_icns(iconset: Path, output: Path) -> None:
    """Write the eight modern ICNS raster roles using Pillow's native encoder."""
    by_width: dict[int, Image.Image] = {}
    for logical_size, name in ICNS_FRAMES.items():
        width = logical_size[0] * logical_size[2]
        with Image.open(iconset / name) as image:
            by_width[width] = image.convert("RGBA")
    by_width[1024].save(
        output, format="ICNS",
        append_images=[by_width[width] for width in sorted(by_width) if width != 1024])


def _compile_or_fallback(iconutil: str, iconset: Path, output: Path) -> str:
    """Compile ``iconset`` with iconutil, or create and verify it with Pillow.

    Compilation targets a scratch file so a broken ``iconutil`` cannot truncate the
    current artifact.  The Pillow fallback is atomically installed only after every
    canonical embedded frame decodes to the exact pixels in the fresh iconset.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nanoscope-icon-", dir=output.parent) as raw:
        scratch = Path(raw)
        generated = scratch / output.name
        try:
            subprocess.run(
                [iconutil, "-c", "icns", str(iconset), "-o", str(generated)], check=True)
        except subprocess.CalledProcessError as compile_error:
            generated.unlink(missing_ok=True)
            try:
                _write_pillow_icns(iconset, generated)
                _assert_icns_matches_iconset(iconset, generated)
            except Exception as fallback_error:
                raise RuntimeError(
                    "iconutil could not compile the iconset and the validated Pillow "
                    f"fallback failed: {fallback_error}"
                ) from compile_error
            generated.replace(output)
            return "pillow-fallback"

        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError("iconutil reported success but did not create a non-empty .icns")
        generated.replace(output)
        return "compiled"


def main() -> int:
    if "--ico" in sys.argv:
        out = write_ico()
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
        return 0
    iconutil = shutil.which("iconutil")
    if iconutil is None:
        print("iconutil not found — this script only runs on macOS", file=sys.stderr)
        return 2
    OUT_ICNS.parent.mkdir(parents=True, exist_ok=True)
    iconset = OUT_ICNS.parent / "NanoScope.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()

    try:
        for px in ICONSET_POINTS:
            render(px).save(iconset / f"icon_{px}x{px}.png")
            render(px * 2).save(iconset / f"icon_{px}x{px}@2x.png")

        result = _compile_or_fallback(iconutil, iconset, OUT_ICNS)
    finally:
        shutil.rmtree(iconset)
    action = "wrote via validated Pillow fallback" if result == "pillow-fallback" else "wrote"
    print(f"{action} {OUT_ICNS} ({OUT_ICNS.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
