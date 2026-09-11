from dataclasses import replace

import pandas as pd
import pytest

from nanoscope.core.source_style import (SourceLabelRegistry, compact_source_labels,
                                         display_sources, validate_source_appearance)
from nanoscope.core.sources import PlotSource


def test_project_aliases_keep_exact_identity_and_references():
    frame = pd.DataFrame({"depth_pA": [1., 2.]})
    original = PlotSource("file#0", "long recording filename", frame,
                          provenance={"min_dwell_s": .001}, colour="#123456", display_name="raw-file")
    labels = SourceLabelRegistry()
    labels.set_label(original.key, "  Apo   -700 mV  ")
    styled = display_sources([original], labels=labels)[0]
    assert styled.label == "Apo -700 mV"
    assert styled.key == original.key and styled.display_name == original.display_name
    assert styled.df is frame and styled.provenance is original.provenance
    assert styled.colour == "#123456"
    assert original.label == "long recording filename"
    other_detector = replace(original, key="file#0@detect.cusum", module_id="detect.cusum")
    assert display_sources([other_detector], labels=labels)[0].label == original.label


def test_registry_roundtrip_and_clear_do_not_touch_color_registry():
    labels = SourceLabelRegistry()
    labels.set_label("file#0", "Apo")
    restored = SourceLabelRegistry.from_json(labels.to_json())
    restored.set_label("file#0", "")
    assert restored.label("file#0", "original") == "original"
    assert labels.label("file#0") == "Apo"
    assert SourceLabelRegistry.from_json({"ok": "  Name ", "bad": None}).to_json() == {"ok": "Name"}


def test_short_labels_are_clean_and_duplicates_remain_distinct():
    sources = [{"key": "a", "label": "-500 mV"}, {"key": "b", "label": "-700 mV"}]
    assert compact_source_labels(sources) == {"a": "-500 mV", "b": "-700 mV"}
    assert len(set(compact_source_labels(sources, {"a": {"label": "Apo"}, "b": {"label": "Apo"}}).values())) == 2


@pytest.mark.parametrize("value", [{"key": {"color": "invalid"}}, {"key": {"label": 2}},
                                  {"key": {"source_key": "different"}}, {"": {"label": "Apo"}}])
def test_invalid_appearance_is_rejected(value):
    with pytest.raises(ValueError):
        validate_source_appearance(value)


def test_session_style_mutation_emits_no_analysis_signal(qtbot):
    from nanoscope.core.session import Session
    session = Session()
    session.colours.set_ink("file#0", "#123456")
    science, styles = [], []
    session.resultsChanged.connect(lambda *args: science.append(args))
    session.calibrationChanged.connect(lambda *args: science.append(args))
    session.sourceAppearanceChanged.connect(styles.append)
    session.set_source_appearance("file#0", label="Apo")
    assert session.colours.inks["file#0"] == "#123456"
    assert session.source_labels.label("file#0") == "Apo"
    assert not science and styles == ["file#0"]
    session.set_source_appearance("file#0", label="Apo")
    assert styles == ["file#0"]
    session.set_source_appearance("file#0", label="Apo", reset_color=True)
    assert not session.colours.is_user_set("file#0")


def test_source_aliases_roundtrip_project(tmp_path):
    from nanoscope.core.project import ProjectState, load_project, save_project
    state = ProjectState(source_labels={"file#0": "Apo -700 mV"})
    path = tmp_path / "styles.nanoscope"
    save_project(path, state)
    restored = load_project(path)
    assert restored.source_labels == state.source_labels


@pytest.mark.parametrize("malformed", [42, "invalid", ["invalid"], {str(i): "Apo" for i in range(2001)}])
def test_bad_source_label_section_has_a_project_error_boundary(tmp_path, malformed):
    import json
    import zipfile
    from nanoscope.core.project import ProjectError, ProjectState, load_project, save_project
    original = tmp_path / "good.nanoscope"
    corrupt = tmp_path / "bad.nanoscope"
    save_project(original, ProjectState())
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(corrupt, "w") as destination:
        for name in source.namelist():
            data = source.read(name)
            if name == "manifest.json":
                manifest = json.loads(data)
                manifest["source_labels"] = malformed
                data = json.dumps(manifest).encode()
            destination.writestr(name, data)
    with pytest.raises(ProjectError, match="source_labels"):
        load_project(corrupt)


def test_figure_library_source_freshness_ignores_project_names_and_inks(qtbot, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.figure_library import source_state
    from nanoscope.core.session import Session
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    path = tmp_path / "trace.csv"
    path.write_bytes(b"original acquisition")
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=4,
                         sample_rate_hz=1000, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(0, "current", "pA", ChannelKind.CURRENT),))
    session = Session()
    session.add_document(TraceDocument(meta, (np.ones(4, dtype=np.float32),)))
    owner = SimpleNamespace(_session=session)
    recipes = [{"source_key": str(path), "channel": 0,
                "channel_name": "current", "channel_kind": "current"}]
    before = source_state(owner, recipes)
    assert before is not None
    session.set_source_appearance(str(path) + "#0", label="Apo -700 mV", color="#654321")
    assert source_state(owner, recipes) == before
