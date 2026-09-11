"""Presentation-only source names and figure-local ink overrides.

Keys are the existing PlotSource keys (recording/segment + channel + detector),
never labels, row hashes or acquisition filenames guessed from display text.
Project colors continue to belong to ColourRegistry. Saved figures may pin their
own names/inks without changing a source table or the project's defaults.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import re

MAX_SOURCE_STYLES = 2_000
MAX_LABEL_LENGTH = 200


def source_label(value: str) -> str:
    """Normalize user display text without interpreting scientific conditions."""
    if not isinstance(value, str) or len(value) > MAX_LABEL_LENGTH:
        raise ValueError(f"A source display label must contain at most {MAX_LABEL_LENGTH} characters.")
    return " ".join(value.split())


def validate_source_appearance(value):
    """Return an independent, bounded map of optional figure-local overrides."""
    if not isinstance(value, Mapping) or len(value) > MAX_SOURCE_STYLES:
        raise ValueError("Source appearance must be a bounded mapping.")
    result = {}
    for key, style in value.items():
        if not isinstance(key, str) or not key or len(key) > 4_096 or not isinstance(style, Mapping):
            raise ValueError("Invalid source appearance identity.")
        if set(style) - {"label", "color"}:
            raise ValueError("Source appearance may contain only label and color overrides.")
        entry = {}
        if "label" in style:
            label = source_label(style["label"])
            if label:
                entry["label"] = label
        if "color" in style:
            color = style["color"]
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("Source appearance colors must be six-digit hex colors.")
            entry["color"] = color.lower()
        if entry:
            result[key] = entry
    return result


@dataclass
class SourceLabelRegistry:
    """Optional project source labels; no effects on identity or measurements."""

    labels: dict[str, str] = field(default_factory=dict)

    def label(self, key: str, fallback: str = "") -> str:
        return self.labels.get(str(key), fallback)

    def set_label(self, key: str, label: str) -> None:
        key = str(key)
        if not key or len(key) > 4_096:
            raise ValueError("Invalid source label identity.")
        label = source_label(label)
        if label:
            if key not in self.labels and len(self.labels) >= MAX_SOURCE_STYLES:
                raise ValueError("This project has reached the source-label limit.")
            self.labels[key] = label
        else:
            self.labels.pop(key, None)

    def to_json(self):
        return dict(self.labels)

    @classmethod
    def from_json(cls, value=None):
        if not isinstance(value, Mapping):
            return cls()
        registry = cls()
        for key, label in list(value.items())[:MAX_SOURCE_STYLES]:
            try:
                if not isinstance(key, str):
                    continue
                registry.set_label(key, label)
            except (TypeError, ValueError):
                continue
        return registry


def display_sources(sources, *, labels=None, colours=None):
    """Copy only presentation fields, keeping frames/provenance by reference.

    Caller supplies the raw discovered sources on each refresh. Project aliases
    are intentionally source-specific, not an implicit alias for other detectors,
    channels, segments or similarly named recordings. Collisions are named rather
    than collapsed into one legend entry.
    """
    from .sources import SOURCE_COLOURS, disambiguate_labels

    sources = list(sources)
    entries = [{"label": labels.label(s.key, s.label) if labels is not None else s.label,
                "detector": s.detector_label, "channel_name": str(s.channel) if s.channel is not None else "",
                "is_segment": s.is_segment} for s in sources]
    disambiguate_labels(entries)
    return [replace(source, label=entry["label"],
                    colour=colours.ink(source.key, SOURCE_COLOURS) if colours is not None else source.colour)
            for source, entry in zip(sources, entries)]


def compact_source_labels(sources, overrides=None):
    """Readable short labels, preserving long names in source metadata/Details.

    Sources are mappings with key and label. Short human labels do not get a
    redundant numeric prefix. Duplicate or truncated labels are disambiguated.
    """
    overrides = validate_source_appearance(overrides or {})
    sources = list(sources)
    labels = [" ".join(str(overrides.get(s["key"], {}).get("label", s["label"])).split()) for s in sources]
    result = {}
    for i, (source, label) in enumerate(zip(sources, labels)):
        if len(label) > 28:
            shown = f"{i + 1} · {label[:14]}…{label[-13:]}"
        elif labels.count(label) > 1:
            shown = f"{i + 1} · {label}"
        else:
            shown = label
        result[source["key"]] = shown
    return result
