"""Conservative print estimates from frozen SVG, never a compliance certificate.

Unknown CSS, glyph outlines and cosmetic strokes are disclosed rather than
assigned invented sizes. This module never alters artwork or scientific data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import xml.etree.ElementTree as ET

_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_LENGTH = re.compile(rf"^\s*({_NUMBER})\s*(px|pt|mm|cm|in)?\s*$")
_IDENTITY = (1.0, 0.0, 0.0, 1.0)
_GRAPHICS = {"path", "line", "polyline", "polygon", "rect", "circle", "ellipse"}


@dataclass(frozen=True)
class PrintMetrics:
    """Smallest measurable sizes in root SVG user-coordinate units."""

    width: float = 0.0
    height: float = 0.0
    text_size: float | None = None
    stroke_size: float | None = None
    text_count: int = 0
    stroke_count: int = 0
    notes: tuple[str, ...] = ()

    def at_size(self, width_mm: float, height_mm: float) -> tuple[float | None, float | None]:
        if self.width <= 0 or self.height <= 0:
            return None, None
        scale = min(width_mm / self.width, height_mm / self.height) * 72 / 25.4
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Print dimensions must be finite and positive.")
        return (None if self.text_size is None else self.text_size * scale,
                None if self.stroke_size is None else self.stroke_size * scale)


def _length(value: str) -> float:
    match = _LENGTH.fullmatch(value)
    if match is None:
        raise ValueError("Unsupported relative or CSS length")
    value, unit = float(match[1]), match[2]
    value *= {None: 1, "px": 1, "pt": 96 / 72, "mm": 96 / 25.4,
              "cm": 96 / 2.54, "in": 96}[unit]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid drawing length")
    return value


def _multiply(a, b):
    return (a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3])


def _transform(value: str):
    result, previous = _IDENTITY, 0
    for match in re.finditer(r"([A-Za-z]+)\s*\(([^)]*)\)", value):
        if value[previous:match.start()].strip(" ,\t\r\n"):
            raise ValueError("Unsupported transform")
        previous = match.end()
        numbers = [float(n) for n in re.findall(_NUMBER, match[2])]
        if re.sub(_NUMBER, "", match[2]).strip(" ,\t\r\n") or not all(map(math.isfinite, numbers)):
            raise ValueError("Invalid transform")
        name = match[1]
        if name == "matrix" and len(numbers) == 6:
            matrix = tuple(numbers[:4])
        elif name == "scale" and len(numbers) in (1, 2):
            matrix = (numbers[0], 0, 0, numbers[-1])
        elif name == "translate" and len(numbers) in (1, 2):
            matrix = _IDENTITY
        elif name == "rotate" and len(numbers) in (1, 3):
            angle = math.radians(numbers[0])
            matrix = (math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle))
        else:
            raise ValueError("Unsupported transform")
        result = _multiply(result, matrix)
    if value[previous:].strip(" ,\t\r\n"):
        raise ValueError("Unsupported transform")
    return result


def _minimum_scale(matrix):
    # Smallest singular value is a conservative bound for rotated text/strokes.
    a, b, c, d = matrix
    total, determinant = a * a + b * b + c * c + d * d, abs(a * d - b * c)
    largest = math.sqrt(max(0.0, (total + math.sqrt(max(0.0, total * total - 4 * determinant ** 2))) / 2))
    return determinant / largest if largest else 0.0


def inspect_print_metrics(svg: bytes | None) -> PrintMetrics:
    """Inspect explicit SVG text/strokes without fetching external resources."""
    if not isinstance(svg, bytes) or not svg:
        return PrintMetrics(notes=("Frozen vector typography is unavailable; inspect the export manually.",))
    if len(svg) > 32 * 1024 * 1024 or b"<!DOCTYPE" in svg.upper() or b"<!ENTITY" in svg.upper():
        return PrintMetrics(notes=("This SVG exceeds the safe preflight scope; inspect it manually.",))
    try:
        root = ET.fromstring(svg)
        viewbox = [float(n) for n in re.split(r"[\s,]+", root.attrib.get("viewBox", "").strip())]
        if len(viewbox) != 4 or not all(map(math.isfinite, viewbox)) or min(viewbox[2:]) <= 0:
            raise ValueError("Invalid SVG viewport")
    except (ET.ParseError, ValueError):
        return PrintMetrics(notes=("The frozen SVG viewport could not be inspected.",))
    notes, text_sizes, stroke_sizes = set(), [], []
    css = " ".join(node.text or "" for node in root.iter() if node.tag.split("}")[-1] == "style")
    if re.search(r"(?:font|stroke|transform|display|visibility)\s*[-:]", css, flags=re.I):
        return PrintMetrics(*viewbox[2:], notes=(
            "Stylesheet-controlled typography or strokes cannot be estimated safely; inspect manually.",))

    def walk(node, inherited, matrix, depth=0):
        if depth > 64:
            notes.add("Deeply nested drawing elements were not inspected.")
            return
        tag = node.tag.split("}")[-1]
        if tag in {"defs", "clipPath", "mask", "metadata", "style", "title", "desc"}:
            return
        attributes = dict(node.attrib)
        for entry in attributes.get("style", "").split(";"):
            if ":" in entry:
                key, value = entry.split(":", 1)
                attributes[key.strip()] = value.strip()
        if "font" in attributes:
            notes.add("Shorthand font styles were not measured.")
            return
        style = dict(inherited)
        style.update({key: value for key, value in attributes.items()
                      if key in {"font-size", "stroke", "stroke-width", "stroke-opacity",
                                 "fill", "fill-opacity", "visibility"}})
        if attributes.get("display") == "none" or style.get("visibility") in {"hidden", "collapse"}:
            return
        if attributes.get("opacity") in {"0", "0.0"}:
            return
        try:
            matrix = _multiply(matrix, _transform(attributes.get("transform", "")))
            scale = _minimum_scale(matrix)
        except (ValueError, OverflowError):
            notes.add("Some transformed elements could not be measured.")
            return
        if not math.isfinite(scale) or scale <= 0:
            return
        if tag == "svg" and node is not root:
            notes.add("Nested SVG viewports were not measured; inspect every composition panel manually.")
            return
        if tag in {"image", "use"}:
            notes.add("Embedded images or outlined/reused glyphs may contain unmeasured text and lines.")
        if tag in {"text", "tspan"} and (node.text or "").strip():
            try:
                size = _length(style["font-size"]) * scale
                if size > 0:
                    text_sizes.append(size)
            except (KeyError, ValueError):
                notes.add("Some text has an unsupported or unspecified font size.")
        if tag in _GRAPHICS and style.get("stroke", "none") != "none" and style.get("stroke-opacity") not in {"0", "0.0"}:
            if attributes.get("vector-effect", "none") != "none":
                notes.add("Cosmetic/non-scaling strokes need manual checking in each exported format.")
            else:
                try:
                    size = _length(style.get("stroke-width", "1")) * scale
                    if size > 0:
                        stroke_sizes.append(size)
                except ValueError:
                    notes.add("Some stroke widths could not be measured.")
        for child in node:
            walk(child, style, matrix, depth + 1)

    walk(root, {}, _IDENTITY)
    if not text_sizes:
        notes.add("No editable text sizes were measurable; glyph outlines cannot certify readability.")
    if not stroke_sizes:
        notes.add("No scalable stroke widths were measurable.")
    notes.add("Only explicit SVG text and scalable strokes are measured; filled glyph outlines are not checked.")
    notes.add("Estimates do not check clipping, overlap, contrast, labels, units or journal compliance.")
    return PrintMetrics(*viewbox[2:], min(text_sizes, default=None), min(stroke_sizes, default=None),
                        len(text_sizes), len(stroke_sizes), tuple(sorted(notes)))


def readability_enlargement(metrics: PrintMetrics, width_mm: float, height_mm: float,
                            text_target_pt: float = 8.0, line_target_pt: float = 0.5) -> float:
    """Enlarge only; preserve all drawing geometry and the chosen aspect ratio."""
    if not all(math.isfinite(v) and v > 0 for v in (text_target_pt, line_target_pt)):
        raise ValueError("Readability targets must be finite and positive.")
    text, stroke = metrics.at_size(width_mm, height_mm)
    factors = [1.0]
    if text is not None and text > 0:
        factors.append(text_target_pt / text)
    if stroke is not None and stroke > 0:
        factors.append(line_target_pt / stroke)
    return max(factors)
