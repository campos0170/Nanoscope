"""Historical writers, not a changed version integer, establish dialect coverage."""
from pathlib import Path
import hashlib
import json
import zipfile

import pandas as pd
import pytest

from nanoscope.core.project import load_project
from nanoscope.core.figure_codecs import TAG, decode_legacy, encode_legacy, materialize_v12_defaults
from nanoscope.core.figure_migration_v12 import migrated_figure_id

FIXTURES = Path(__file__).parent / "fixtures" / "phase2" / "legacy"
REVISIONS = {
    "d1": ("54c14838834aec453a9ebc052611604ad04ffbb5", 12),
    "d2": ("56bdf1e60a8122b242c5e83cf983f2fddc8c56d6", 12),
    "v11": ("5eae42a9d74326980553848f3d934c921b00a9cb", 11),
}


@pytest.mark.parametrize("dialect", tuple(REVISIONS))
@pytest.mark.parametrize("variant", ("defaults", "legacy-look"))
def test_actual_historical_writer_inverse_and_frozen_restore_semantics(dialect, variant):
    provenance = json.loads((FIXTURES / f"{dialect}-provenance.json").read_text())
    revision, version = REVISIONS[dialect]
    assert provenance["revision"] == revision and provenance["writer_project_version"] == version
    name = f"{dialect}-{variant}.nanoscope"
    fixture = FIXTURES / name
    witness = provenance["files"][name]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == witness["sha256"]
    with zipfile.ZipFile(fixture) as archive:
        assert json.loads(archive.read("manifest.json"))["version"] == version
    native = load_project(fixture).workspace.artifacts[0]
    assert hashlib.sha256(json.dumps(native.payload, sort_keys=True).encode()).hexdigest() == witness["payload_sha256"]
    decoded = decode_legacy(native, figure_id=migrated_figure_id(native.artifact_id), name="Historical")
    inverse = encode_legacy(decoded.spec, decoded.resources)
    assert inverse.payload == native.payload and inverse.extra == native.extra
    assert inverse.blobs == native.blobs and set(inverse.tables) == set(native.tables)
    for key in native.tables:
        pd.testing.assert_frame_equal(inverse.tables[key], native.tables[key])
    assert len(inverse.tables["seeded.events"]) == witness["rows"] == 9

    restored = encode_legacy(decoded.spec, decoded.resources, materialize_defaults=True)
    fields = restored.payload["config"]["fields"]
    original_style = native.payload["config"]["fields"]["style"]["fields"]
    style = fields["style"]["fields"]
    for key in ("title", "x_label", "y_label"):
        assert style[key] == original_style[key]
    assert "text_styles" in style and "fit_label" in style
    assert not {"title_bold", "title_italic", "show_r2", "show_gof", "fit_values_on_plot"} & set(style)
    if variant == "legacy-look":
        assert style["text_styles"]["title"]["bold"] is True
        assert style["text_styles"]["title"]["italic"] is True
        fit = style["fit_label"]
        assert fit["fields"][TAG] == "tuple"
        assert fit["fields"]["items"] == (["equation", "n"] if original_style.get("fit_label") else ["equation", "n", "bic"])
        if original_style.get("fit_label"):
            assert fit["template"] == "Own r2=—; ks=—"
    # Materialization is a detached restore view, never a rewrite of the original.
    fields["style"]["fields"]["title"] = "Unrelated resident page"
    assert materialize_v12_defaults(inverse).payload["config"]["fields"]["style"]["fields"]["title"] == original_style["title"]
    assert encode_legacy(decoded.spec, decoded.resources).payload == native.payload
