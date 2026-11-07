"""Applied FigureSpec and origin lineage are additive immutable snapshot resources."""
from copy import deepcopy
import json
import zipfile

import pandas as pd
import pytest

from nanoscope.core.figure_library import make_figure
from nanoscope.core.project import WorkspaceArtifact
from nanoscope.gui.figure_bundle import FigureSnapshot, capture_figure_snapshot, write_figure_bundle

from .test_figure_bundle import plots


def test_capture_embeds_detached_applied_spec_in_legacy_compatible_bundle(plots, tmp_path):
    main, _residual, curve, _resid_curve, x, y = plots
    spec = {"spec_version": 1, "figure_id": "origin-id", "name": "Applied view",
            "recipe": {"draft": {"fit": "exponential"}, "applied": {"fit": "exponential"}},
            "extra": {"future": [1, None]}}
    before = deepcopy(spec)
    snapshot = capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"x": x, "y": y}),
                                       applied_spec=spec)
    spec["extra"]["future"].append(2)
    assert snapshot.figure_spec() == before
    copy = snapshot.figure_spec()
    copy["name"] = "Changed"
    assert snapshot.figure_spec()["name"] == "Applied view"
    metadata, blobs = snapshot.to_storage()
    assert metadata["version"] == 1 and "figure-spec.json" in metadata["resources"]
    assert FigureSnapshot.from_storage(metadata, blobs).figure_spec() == before
    path = write_figure_bundle(tmp_path / "snapshot.zip", snapshot)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("figure-spec.json")) == before
    assert len(curve.xData) == len(x)


def test_snapshot_origin_and_spec_survive_independently_of_live_values():
    spec = {"figure_id": "live.origin", "extra": {"retained": [1]}}
    analysis = WorkspaceArtifact("original", "test", payload={"value": 4})
    saved = make_figure(name="Capture", page_id="iv", source_label="Synthetic", analysis=analysis,
                        snapshot_blobs={"png": b"unchanged"}, snapshot_metadata={"version": 1},
                        live_state=None, applied_spec=spec)
    assert saved.schema_version == 1
    assert saved.payload["origin_figure_id"] == "live.origin"
    assert saved.payload["lineage"]["origin_figure_id"] == "live.origin"
    spec["extra"]["retained"].append(2)
    analysis.payload["value"] = 9
    assert saved.payload["figure_spec"]["extra"]["retained"] == [1]
    assert saved.payload["analysis"]["payload"] == {"value": 4}
    assert saved.blobs["png"] == b"unchanged"


def test_applied_spec_resource_cannot_be_shadowed(plots):
    main, _residual, _curve, _resid_curve, x, y = plots
    with pytest.raises(ValueError, match="reserved"):
        capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"x": x, "y": y}),
                                additional_resources={"FIGURE-SPEC.JSON": b"other"}, applied_spec={})
