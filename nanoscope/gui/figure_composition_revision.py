"""Read one saved composition layer without resolving or reopening live analyses."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .figure_bundle import FigureSnapshot, _FILES


@dataclass(frozen=True)
class FrozenPanel:
    panel_id: str
    label: str
    snapshot: FigureSnapshot


def extract_composition(snapshot: FigureSnapshot) -> tuple[tuple[FrozenPanel, ...], dict]:
    """Validate and extract embedded originals, never today's similarly named figures.

    Nested compositions remain opaque single panels. Every resource must belong
    to its canonical panel directory; hashes alone do not permit cross-panel aliases.
    """
    metadata, blobs = snapshot.to_storage()
    snapshot = FigureSnapshot.from_storage(metadata, blobs)
    config = snapshot.config()
    if (not isinstance(config, dict) or config.get("kind") != "figure-composition"
            or type(config.get("version")) is not int or config["version"] != 1):
        raise ValueError("This figure is not a supported version-1 composition. Its saved image can still be exported.")
    panels = config.get("panels")
    labels = config.get("panel_labels")
    if (not isinstance(panels, list) or not 1 <= len(panels) <= 12
            or not isinstance(labels, list) or len(labels) != len(panels)
            or any(not isinstance(label, str) or len(label) > 32 or "\n" in label or "\r" in label for label in labels)
            or type(config.get("columns")) is not int or not 1 <= config["columns"] <= min(4, len(panels))
            or any(type(config.get(key)) is not bool for key in ("share_x", "share_y"))):
        raise ValueError("The saved composition layout is malformed; nothing has been changed.")
    resources = dict(snapshot.additional_resources)
    extracted = []
    for index, panel in enumerate(panels):
        panel_id = f"panel_{index + 1:03d}"
        if (not isinstance(panel, dict) or panel.get("panel_id") != panel_id
                or panel.get("label") != labels[index]
                or not isinstance(panel.get("snapshot"), dict)
                or not isinstance(panel.get("blob_resources"), dict)):
            raise ValueError("The saved composition panel description is malformed.")
        child_metadata = panel["snapshot"]
        child_resources = child_metadata.get("resources", {})
        if not isinstance(child_resources, dict) or any(
                not isinstance(filename, str) or not isinstance(key, str)
                for filename, key in child_resources.items()):
            raise ValueError("The saved panel resource map is malformed.")
        if set(child_resources.values()) != {f"resource_{n}" for n in range(len(child_resources))}:
            raise ValueError("The saved panel uses noncanonical resource identifiers; it cannot be revised losslessly.")
        expected = {key: f"panels/{panel_id}/{filename}" for key, filename in _FILES.items()}
        expected.update({key: f"panels/{panel_id}/{filename}" for filename, key in child_resources.items()})
        mapping = panel["blob_resources"]
        if mapping != expected or any(filename not in resources for filename in mapping.values()):
            raise ValueError("A saved panel resource is missing or points outside its own panel.")
        original = FigureSnapshot.from_storage(child_metadata, {key: resources[path] for key, path in mapping.items()})
        # JSON sorts resource filenames, whereas to_storage assigns resource_N in
        # insertion order. Restore the declared ordinal order so even a nested
        # panel keeps its original blob keys/checksums, not merely identical bytes.
        original = replace(original, additional_resources=tuple(sorted(
            original.additional_resources, key=lambda item: int(child_resources[item[0]].split("_")[1]))))
        extracted.append(FrozenPanel(panel_id, labels[index], original))
    return tuple(extracted), {key: config[key] for key in ("columns", "share_x", "share_y", "panel_labels")}
