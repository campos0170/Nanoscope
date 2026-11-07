"""Synthetic dialects exercise exact leaf accounting; these are not historical files."""
from dataclasses import replace
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.core.project import WorkspaceArtifact, AnalyticalWorkspace
from nanoscope.core.figure_spec import FigureValidationError, TextWords, FigureRecord
from nanoscope.core.figure_codecs import (TAG, NATIVE_KINDS, decode_legacy, encode_legacy,
    record_to_artifact, record_from_artifact, leaf_paths)
from nanoscope.core.figure_migration_v12 import migrate_workspace_v12, migrated_figure_id

ID = "64d9be44-d980-43ab-8da5-bcc95a827e45"


def tagged(name, values):
    return {TAG: "dataclass", "type": "nanoscope.gui.plots_panel:" + name, "fields": values}


def native(identity):
    appearance = {"title": "Own title", "x_label": " ", "y_label": "", "future": {"kept": None},
        "text": {"title": {"bold": True}, "annotation.limit": {"text": "Limit", "color": "#abcdef"}},
        "axes": {"density.bottom": {"min": 1., "max": 4., "future_pin": "kept"}},
        "fit_label": {"template": "Own {equation}", "fields": ["r2"], "unknown": 3},
        "annotations": {"annotation.caption": {"offset": [2., 3.], "future": 4}}}
    payload = {"future_root": {"empty": {}, "null": None}, "appearance": appearance,
               "fit": {TAG: "ndarray", "table": "fit", "shape": [2], "dtype": "<f8"}}
    if identity == "builtin.plots":
        style = {"title": "Own title", "x_label": " ", "y_label": "", "font_family": "",
                 "export_w": 0, "export_h": 500, "show_ci_band": True, "show_residuals": True,
                 "show_provenance": True, "heatmap_colormap": "plasma", "legend_counts": False,
                 "label_overrides": {"stable#0": "Pore A"}, "legend_pos": None,
                 "text_styles": appearance["text"], "fit_label": appearance["fit_label"], "future": 12}
        payload.pop("appearance")
        payload["config"] = tagged("PlotConfig", {"plot_type": "Histogram", "future_config": 7,
            "style": tagged("StyleSettings", style), "axes": tagged("AxesConfig", {"time_unit": "ms", "x_min": "nan", "x_max": "4", "manual_x": None}),
            "fit": tagged("FitConfig", {"kind": "gaussian", "applied": "none", "range_lo": "0."}),
            "histogram": tagged("HistogramConfig", {"column": "dwell_s", "bins": 30})})
        payload["selected_sources"] = [{"source_key": "stable.csv", "channel": 0, "unknown": None}]
    elif identity in ("builtin.distribution-summaries", "builtin.comparison-figures"):
        payload.update(draft={"metric": "dwell_s", "kind": "ecdf"}, applied={"metric": "depth_pA"},
            source_recipes=[{"source_key": "stable.csv", "channel": 0}],
            source_appearance={"stable#0": {"label": "Pore A", "color": "#123456", "future": 2}},
            snapshot={"title": "Automatic result title", "future_geometry": [1, 2]},
            table_resources={"points": "points.parquet"}, stale=True, canvas_view={"x": [1., 2.]})
    elif identity in ("builtin.event-plots", "builtin.psd", "builtin.voltage-dependence"):
        payload["config"] = {"appearance": payload.pop("appearance"), "view": "density", "future_config": 2}
        if identity == "builtin.event-plots":
            payload["config"].update(density_title="Density own", ensemble_title="Ensemble own", density_x_label="Time")
            payload["source"] = {"source_key": "stable.csv", "channel": 0}
        elif identity == "builtin.psd":
            payload["config"].update(style={"title": "Old PSD title", "data_color": "#123456", "export_w": 300}, target_source={"source_key": "stable.csv"}, compare_sources=[])
        else:
            payload["config"].update(selected_fit="quadratic", applied_fit="linear")
            payload["point_sources"] = [{"source_key": "stable.csv"}]
    elif identity == "builtin.iv":
        payload["points"] = [{"source_key": "stable.csv", "channel": 0, "v_volts": .1, "i_open": 30.}]
        payload["exclusions"] = []
    else:
        payload.update(source={"source_key": "stable.csv"}, accepted_event_ids=[1, 3],
                       view={"embedding": "PCA", "splitter_sizes": [100, 200]}, assignment_state="stale")
    return WorkspaceArtifact(identity, NATIVE_KINDS[identity][0], payload=payload,
                             tables={"fit": pd.DataFrame({"v": [1., 2.]})},
                             blobs={"points.parquet": b"opaque bytes"}, extra={"plugin": {"keep": 4}})


@pytest.mark.parametrize("identity", tuple(NATIVE_KINDS))
def test_eight_native_kinds_have_exact_inverse_and_complete_leaf_accounting(identity):
    original = native(identity)
    before = deepcopy(original.payload)
    decoded = decode_legacy(original, figure_id=ID, name="Own name")
    restored = encode_legacy(decoded.spec, decoded.resources)
    assert original.payload == before == restored.payload
    assert restored.extra == original.extra
    assert restored.blobs == original.blobs
    pd.testing.assert_frame_equal(restored.tables["fit"], original.tables["fit"])
    claimed, carried = set(decoded.report.claimed_paths), set(decoded.report.carried_paths)
    assert claimed and carried and not claimed.intersection(carried)
    assert claimed | carried == set(leaf_paths(before))
    assert record_from_artifact(record_to_artifact(decoded.record)).spec == decoded.spec
    assert "future_root" in decoded.spec.result.payload


@pytest.mark.parametrize("identity", tuple(NATIVE_KINDS))
def test_canonical_words_edit_the_native_value_without_changing_results(identity):
    decoded = decode_legacy(native(identity), figure_id=ID, name="A")
    title = next(text for text in decoded.spec.texts if text.element_id == "title")
    edited = replace(title, words=TextWords("literal", "Edited title"))
    spec = replace(decoded.spec, texts=tuple(edited if text == title else text for text in decoded.spec.texts))
    restored = encode_legacy(spec, decoded.resources)
    if identity == "builtin.plots":
        assert restored.payload["config"]["fields"]["style"]["fields"]["title"] == "Edited title"
    elif identity in ("builtin.event-plots", "builtin.psd", "builtin.voltage-dependence"):
        assert restored.payload["config"]["appearance"]["title"] == "Edited title"
    else:
        assert restored.payload["appearance"]["title"] == "Edited title"
    assert restored.payload["fit"] == native(identity).payload["fit"]


def test_draft_and_applied_fit_and_literal_invalid_range_are_independent():
    d = decode_legacy(native("builtin.plots"), figure_id=ID, name="A")
    assert d.spec.recipe.draft["fit.kind"] == "gaussian"
    assert d.spec.recipe.applied["fit.applied"] == "none"
    assert d.spec.axes[0].pin.values["x_min"] == "nan"
    assert d.spec.recipe.draft["fit.range_lo"] == "0."


def test_missing_resources_and_duplicate_native_owners_are_rejected():
    artifact = native("builtin.iv")
    artifact.tables.clear()
    with pytest.raises(FigureValidationError, match="resource"):
        decode_legacy(artifact, figure_id=ID, name="A")
    d = decode_legacy(native("builtin.iv"), figure_id=ID, name="A")
    raw = d.spec.to_dict()
    raw["result"]["payload"]["appearance"]["title"] = "second owner"
    with pytest.raises(FigureValidationError, match="also exists"):
        encode_legacy(type(d.spec).from_dict(raw), d.resources)


def test_migration_preserves_unknown_snapshots_future_schemas_and_failures_and_is_idempotent():
    unknown = WorkspaceArtifact("plugin.1", "plugin.kind", payload={"opaque": [None, {}]})
    snapshot = WorkspaceArtifact("figure.saved", "nanoscope.snapshot", payload={"words": "A"})
    future = native("builtin.psd"); future.schema_version = 2
    broken = native("builtin.plots"); broken.payload["config"] = []
    workspace = AnalyticalWorkspace(1, [unknown, native("builtin.iv"), snapshot, future, broken], {"future_workspace": 4})
    result = migrate_workspace_v12(workspace)
    assert result.workspace.extra == workspace.extra
    assert result.workspace.artifacts[0] is unknown and result.workspace.artifacts[2] is snapshot
    assert result.workspace.artifacts[3] is future and result.workspace.artifacts[4] is broken
    assert [f.artifact_id for f in result.failures] == ["builtin.plots"]
    twice = migrate_workspace_v12(result.workspace)
    assert [a.artifact_id for a in twice.workspace.artifacts] == [a.artifact_id for a in result.workspace.artifacts]
    assert len(twice.records) == 1 and twice.records[0].spec == result.records[0].spec
    assert workspace.artifacts[1].artifact_id == "builtin.iv"


def test_migration_collision_is_preflighted_before_any_conversion():
    old = native("builtin.iv")
    collision = WorkspaceArtifact("live." + migrated_figure_id(old.artifact_id), "plugin.kind")
    workspace = AnalyticalWorkspace(artifacts=[old, collision])
    with pytest.raises(FigureValidationError, match="collides"):
        migrate_workspace_v12(workspace)
    assert workspace.artifacts == [old, collision]


def test_restore_defaults_are_frozen_and_do_not_change_lossless_native_document():
    d = decode_legacy(native("builtin.plots"), figure_id=ID, name="A")
    original = encode_legacy(d.spec, d.resources)
    hydrated = encode_legacy(d.spec, d.resources, materialize_defaults=True)
    fields = hydrated.payload["config"]["fields"]
    assert fields["scatter"]["fields"] == {"x": "", "y": "", "log_x": False, "log_y": False, "colour_by": "None"}
    assert fields["style"]["fields"]["font_family"] == ""
    assert fields["style"]["fields"]["point_size"] == 7
    assert fields["axes"]["fields"]["x_min"] == "nan"
    assert "scatter" not in original.payload["config"]["fields"]
    assert encode_legacy(d.spec, d.resources).payload == original.payload


def test_capture_merge_keeps_unknowns_without_reviving_reset_words_or_old_results():
    from nanoscope.core.figure_codecs import merge_native_capture

    old = native("builtin.iv")
    old.payload["appearance"]["text"]["title"]["future_style"] = {"tracked": 1}
    prior = decode_legacy(old, figure_id=ID, name="Named figure").record
    new = native("builtin.iv")
    new.payload["appearance"] = {}
    del new.payload["future_root"], new.payload["fit"]
    new.tables.clear()
    merged = merge_native_capture(new, prior)
    restored = encode_legacy(merged.spec, merged.resources)
    assert restored.payload["future_root"] == old.payload["future_root"]
    assert restored.payload["appearance"]["future"] == {"kept": None}
    assert restored.payload["appearance"]["text"]["title"] == {"future_style": {"tracked": 1}}
    assert "title" not in restored.payload["appearance"]
    assert "bold" not in restored.payload["appearance"]["text"]["title"]
    assert "fit" not in restored.payload and "fit" not in restored.tables
    assert merged.spec.figure_id == ID and merged.spec.name == "Named figure"


def test_capture_merge_preserves_unknown_tagged_config_but_resets_known_style_fields():
    from nanoscope.core.figure_codecs import merge_native_capture

    old = native("builtin.plots")
    prior = decode_legacy(old, figure_id=ID, name="A").record
    new = native("builtin.plots")
    fields = new.payload["config"]["fields"]
    del fields["future_config"]
    fields["style"]["fields"] = {"title": "New"}
    merged = merge_native_capture(new, prior)
    result = encode_legacy(merged.spec, merged.resources)
    result_fields = result.payload["config"]["fields"]
    assert result_fields["future_config"] == 7
    assert result_fields["style"]["fields"] == {"title": "New", "future": 12, "fit_label": {"unknown": 3}}
    assert "template" not in result_fields["style"]["fields"]["fit_label"]
    assert "fields" not in result_fields["style"]["fields"]["fit_label"]


def test_live_artifact_unknown_envelope_fields_round_trip():
    record = decode_legacy(native("builtin.iv"), figure_id=ID, name="A").record
    artifact = record_to_artifact(record)
    artifact.extra = {"future_envelope": {"revision": 3}}
    restored = record_to_artifact(record_from_artifact(artifact))
    assert restored.extra == artifact.extra and restored.payload == artifact.payload


def test_distributions_mixed_axis_units_preserve_seconds_and_display_text_separately():
    artifact = native("builtin.plots")
    axes = artifact.payload["config"]["fields"]["axes"]["fields"]
    axes.update(time_unit="ms", x_min="1", x_max="4", manual_x={TAG: "tuple", "items": [.001, .004]})
    decoded = decode_legacy(artifact, figure_id=ID, name="A")
    pin = next(axis.pin for axis in decoded.spec.axes if axis.side == "bottom")
    assert pin.unit_basis == "mixed"
    assert pin.extra["value_unit_basis"] == {"x_min": "display", "x_max": "display", "manual_x": "data"}
    assert pin.extra["display_time_unit"] == "ms" and pin.values["x_min"] == "1"
    assert pin.values["manual_x"]["items"] == (.001, .004)
    assert encode_legacy(decoded.spec, decoded.resources).payload == artifact.payload


def test_iv_point_coordinates_have_one_series_owner_and_source_recipe_has_only_identity():
    decoded = decode_legacy(native("builtin.iv"), figure_id=ID, name="A")
    series = next(s for s in decoded.spec.series if s.series_id == "iv.data")
    assert series.data["points"][0]["v_volts"] == .1
    assert series.data["points"][0]["i_open"] == 30.
    source = next(s for s in decoded.spec.sources if s.source_id == series.source_ids[0])
    assert source.recipe == {"source_key": "stable.csv", "channel": 0}
    assert decoded.spec.result.payload["points"] == (type(decoded.spec.result.payload)(),)


def test_sampled_extension_applied_recipe_and_sources_have_single_canonical_owners():
    artifact = native("builtin.plots")
    artifact.payload["phase2_render"] = {"version": 1,
        "applied_config": {"plot_type": "KDE", "kde": {"x": "dwell_s", "y": "depth_pA"}},
        "drawn_sources": [{"source_key": "stable.csv", "channel": 0}],
        "nodes": [{"kind": "curve", "table": "fit", "pen": "#123456"}], "future_render": "kept"}
    decoded = decode_legacy(artifact, figure_id=ID, name="A")
    assert decoded.spec.recipe.applied["plot_config"]["plot_type"] == "KDE"
    assert "applied_config" not in decoded.spec.result.payload["phase2_render"]
    assert "drawn_sources" not in decoded.spec.result.payload["phase2_render"]
    assert any(s.primitive == "sampled-curve" for s in decoded.spec.series)
    assert encode_legacy(decoded.spec, decoded.resources).payload == artifact.payload


def test_authentic_recovered_v12_corpus_all_eight_pages_have_exact_inverse_and_series():
    from pathlib import Path
    import hashlib
    import json
    from nanoscope.core.project import load_project

    folder = Path(__file__).parent / "fixtures" / "phase2"
    manifest = json.loads((folder / "PROVENANCE.json").read_text())
    fixture = folder / manifest["file"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["sha256"]
    state = load_project(fixture)
    seen = set()
    for artifact in state.workspace.artifacts:
        if artifact.artifact_id not in NATIVE_KINDS:
            continue
        decoded = decode_legacy(artifact, figure_id=migrated_figure_id(artifact.artifact_id), name="Recovered")
        restored = encode_legacy(decoded.spec, decoded.resources)
        assert restored.payload == artifact.payload
        assert restored.extra == artifact.extra and restored.blobs == artifact.blobs
        assert set(restored.tables) == set(artifact.tables)
        for key in artifact.tables:
            pd.testing.assert_frame_equal(restored.tables[key], artifact.tables[key])
        assert decoded.spec.series, artifact.artifact_id
        assert set(decoded.report.claimed_paths) | set(decoded.report.carried_paths) == set(leaf_paths(artifact.payload))
        seen.add(decoded.spec.kind)
    assert seen == {entry[1] for entry in NATIVE_KINDS.values()}


def test_frozen_legacy_style_fold_keeps_words_and_only_folds_duplicate_switches():
    artifact = native("builtin.plots")
    style = artifact.payload["config"]["fields"]["style"]["fields"]
    style.update(title_bold=True, title_italic=True, fit_values_on_plot=True, show_r2=False, show_gof=False)
    style["text_styles"] = {"title": {"bold": False}}
    style["fit_label"] = {"fields": {TAG: "tuple", "items": ["equation", "r2", "n", "ks"]},
                          "template": "Own r2={r2}; ks={ks}"}
    decoded = decode_legacy(artifact, figure_id=ID, name="A")
    restored = encode_legacy(decoded.spec, decoded.resources, materialize_defaults=True)
    folded = restored.payload["config"]["fields"]["style"]["fields"]
    assert folded["title"] == style["title"]
    assert folded["text_styles"]["title"] == {"bold": False, "italic": True}
    assert folded["fit_label"]["fields"]["items"] == ["equation", "n"]
    assert folded["fit_label"]["template"] == "Own r2=—; ks=—"
    assert not set(("show_r2", "show_gof", "title_bold", "title_italic", "fit_values_on_plot")) & set(folded)
    assert encode_legacy(decoded.spec, decoded.resources).payload == artifact.payload


def test_future_workspace_envelope_is_retained_without_interpreting_known_artifacts():
    artifact = native("builtin.iv")
    workspace = AnalyticalWorkspace(2, [artifact], {"future_semantics": True})
    result = migrate_workspace_v12(workspace)
    assert result.workspace is workspace and result.workspace.artifacts[0] is artifact
    assert not result.records and not result.reports
    assert result.failures[0].artifact_id == "workspace"
    assert "retained unchanged" in result.failures[0].reason


def test_capture_keeps_future_schema_fields_even_when_known_text_is_reset():
    from nanoscope.core.figure_codecs import merge_native_capture
    from nanoscope.core.figure_spec import FigureSpec, FigureRecord

    prior = decode_legacy(native("builtin.iv"), figure_id=ID, name="A").record
    raw = prior.spec.to_dict()
    raw["future_document"] = {"revision": 7}
    raw["presentation"]["future_presentation"] = {"effect": "kept"}
    raw["recipe"]["future_recipe"] = [None, 2]
    title = next(node for node in raw["texts"] if node["element_id"] == "title")
    title["future_text"] = "kept"
    title["words"]["future_word_mode"] = 9
    prior = FigureRecord(FigureSpec.from_dict(raw), prior.resources)
    captured = native("builtin.iv")
    captured.payload["appearance"] = {}
    merged = merge_native_capture(captured, prior)
    result = merged.spec.to_dict()
    assert result["future_document"] == {"revision": 7}
    assert result["presentation"]["future_presentation"] == {"effect": "kept"}
    assert result["recipe"]["future_recipe"] == [None, 2]
    title = next(node for node in result["texts"] if node["element_id"] == "title")
    assert title["future_text"] == "kept" and title["words"]["future_word_mode"] == 9
    assert title["words"]["mode"] == "automatic" and title["words"]["value"] is None
    assert "title" not in encode_legacy(merged.spec, merged.resources).payload["appearance"]
