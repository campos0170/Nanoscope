from types import SimpleNamespace

import pytest

from nanoscope.core.analysis_workspace import _resolve_plot_recipe


def test_frozen_figure_source_resolver_receives_exact_recipe_and_row_requirement():
    seen = []
    recipe = {"source_key": "closed-recording.csv", "channel": 2, "event_frame_fingerprint": "saved"}

    def resolver(value, *, require_same_rows):
        seen.append((value, require_same_rows))
        return "frozen.source", "Saved source is closed"

    window = SimpleNamespace(_resolve_frozen_plot_source=resolver)
    assert _resolve_plot_recipe(window, recipe, require_same_rows=True) == ("frozen.source", "Saved source is closed")
    assert seen == [(recipe, True)]


def test_frozen_source_refusal_does_not_retarget_live_catalog():
    window = SimpleNamespace(_resolve_frozen_plot_source=lambda *args, **kwargs: (None, "Fingerprint mismatch"),
                             _sources={"live": object()})
    assert _resolve_plot_recipe(window, {}) == (None, "Fingerprint mismatch")


def test_legacy_source_resolution_without_hook_is_unchanged():
    assert _resolve_plot_recipe(SimpleNamespace(_sources={}), {"source_key": "missing"})[0] is None


def test_malformed_frozen_resolver_result_is_explicitly_refused():
    with pytest.raises(TypeError, match="resolver"):
        _resolve_plot_recipe(SimpleNamespace(_resolve_frozen_plot_source=lambda *a, **k: "bad"), {})


def test_frozen_source_capture_preserves_missing_document_channel_identity_and_copies():
    from nanoscope.core.analysis_workspace import _plot_source_recipe

    recipe = {"source_key": "closed-recording.csv", "channel": 2, "channel_name": "current",
              "channel_kind": "current", "event_frame_fingerprint": "saved", "future": [1]}
    source = object()
    seen = []

    def capture(value):
        seen.append(value)
        return recipe

    copied = _plot_source_recipe(SimpleNamespace(_frozen_plot_source_recipe=capture), source)
    assert copied == recipe and seen == [source]
    copied["future"].append(2)
    assert recipe["future"] == [1]


def test_none_frozen_capture_recipe_uses_normal_manual_identity():
    from nanoscope.core.analysis_workspace import _plot_source_recipe

    window = SimpleNamespace(_frozen_plot_source_recipe=lambda source: None)
    recipe = _plot_source_recipe(window, SimpleNamespace(kind="manual", doc_key="manual-table", df=None))
    assert recipe["kind"] == "manual" and recipe["source_key"] == "manual-table"
