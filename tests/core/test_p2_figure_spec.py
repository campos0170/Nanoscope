"""Pure figure values and detached resources; no GUI construction or imports."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from nanoscope.core.figure_spec import (
    FigureSpec, FigureRecipe, FigurePresentation, FigureRecord, ResultState,
    TextSpec, TextWords, SourceRef, SeriesSpec, FrozenMap, ResourceSet,
    FigureValidationError, freeze_json, canonical_json,
)
from nanoscope.core.figure_history_state import HistoryCapsule

ID = "64d9be44-d980-43ab-8da5-bcc95a827e45"


def figure(**changes):
    return FigureSpec(figure_id=ID, kind="iv", name="I–V A", **changes)


def test_nested_values_are_detached_and_canonical_json_is_stable():
    raw = {"a": [1, {"z": None}], "b": False}
    spec = figure(recipe=FigureRecipe(codec="iv", draft=raw))
    raw["a"][1]["z"] = "changed"
    assert spec.recipe.draft["a"][1]["z"] is None
    with pytest.raises(TypeError):
        spec.recipe.draft["a"][1]["z"] = "changed"
    assert FigureSpec.from_json(spec.to_json()) == spec
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_future_fields_survive_at_each_original_node_level():
    raw = figure().to_dict()
    raw["future"] = {"nested": [None, False, 3]}
    raw["presentation"]["future_look"] = {"ink": "x"}
    raw["extra"] = {"plugin": "kept"}
    assert FigureSpec.from_dict(raw).to_dict() == raw


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "not a JSON key"}, object()])
def test_non_json_and_nonfinite_values_are_refused(value):
    with pytest.raises(FigureValidationError):
        freeze_json(value)


def test_duplicate_json_keys_cycles_and_future_versions_are_refused():
    with pytest.raises(FigureValidationError, match="Duplicate"):
        FigureSpec.from_json('{"kind":"iv","kind":"noise"}')
    circular = []
    circular.append(circular)
    with pytest.raises(FigureValidationError, match="cycle"):
        freeze_json(circular)
    with pytest.raises(FigureValidationError, match="version"):
        replace(figure(), spec_version=True)
    with pytest.raises(FigureValidationError, match="version"):
        replace(figure(), spec_version=2)


def test_duplicate_element_ids_and_dangling_series_references_are_refused():
    text = TextSpec("title", TextWords("literal", "Own words"))
    with pytest.raises(FigureValidationError, match="Duplicate"):
        figure(texts=(text, text))
    with pytest.raises(FigureValidationError, match="unknown source"):
        figure(series=(SeriesSpec("line", source_ids=("missing",)),))
    assert figure(sources=(SourceRef("source"),),
                  series=(SeriesSpec("line", source_ids=("source",)),)).series


def test_words_keep_automatic_hidden_literal_and_legacy_values_distinct():
    words = [TextWords("automatic", None), TextWords("automatic", ""),
             TextWords("hidden", " "), TextWords("literal", "A title"),
             TextWords("legacy", {"unknown": 2})]
    assert len(set(words)) == 5
    assert [TextWords.from_dict(word.to_dict()) for word in words] == words


def test_resources_are_detached_and_references_must_match():
    source = pd.DataFrame({"x": [1., 2.]})
    resources = ResourceSet({"points": source}, {"picture": b"original"})
    source.iloc[0, 0] = 99
    copied = resources.tables
    copied["points"].iloc[0, 0] = 88
    assert resources.tables["points"].iloc[0, 0] == 1
    spec = figure(result=ResultState(table_refs=("points",), blob_refs=("picture",)))
    assert FigureRecord(spec, resources).resources.blobs == {"picture": b"original"}
    with pytest.raises(FigureValidationError, match="references"):
        FigureRecord(figure(), resources)


def test_history_capsule_undo_redo_and_branching_preserve_immutable_states():
    history = HistoryCapsule("appearance", {"title": "A"})
    history = history.note({"title": "B"}).note({"title": "C"}, limit=1)
    assert history.dropped == 1 and history.undo().base["title"] == "B"
    assert history.undo().redo() == history
    branch = history.undo().note({"title": "D"})
    assert branch.future == () and branch.base["title"] == "D"
    assert HistoryCapsule.from_dict(history.to_dict()) == history


def test_object_cells_and_table_metadata_are_detached():
    source = pd.DataFrame({"objects": [{"items": [1]}]})
    source.attrs["meta"] = {"items": [2]}
    resources = ResourceSet({"points": source})
    source.iat[0, 0]["items"].append(9)
    source.attrs["meta"]["items"].append(9)
    copied = resources.tables["points"]
    copied.iat[0, 0]["items"].append(8)
    copied.attrs["meta"]["items"].append(8)
    assert resources.tables["points"].iat[0, 0] == {"items": [1]}
    assert resources.tables["points"].attrs == {"meta": {"items": [2]}}


def test_unknown_fields_never_borrow_extension_keys_and_survive_value_replacement():
    raw = figure().to_dict()
    raw["future"] = {"value": 1}
    raw["_unknown_fields"] = "A legitimate future wire field"
    raw["extra"] = {"_unknown_fields": {"plugin_owned": 2}, "other": 3}
    raw["presentation"]["future"] = "kept"
    spec = FigureSpec.from_dict(raw)
    assert spec.to_dict() == raw
    assert replace(spec, name="New").to_dict() == {**raw, "name": "New"}
    assert replace(spec.presentation, style_ref="style.1").to_dict()["future"] == "kept"
