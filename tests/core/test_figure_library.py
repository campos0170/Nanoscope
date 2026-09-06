"""Project-owned figures preserve frozen data, metadata and future artifacts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import capture_analysis_workspace
from nanoscope.core.figure_library import (
    FIGURE_KIND, duplicate_figure, figure_status, figures, make_figure,
    rename_figure, source_recipes, source_state,
)
from nanoscope.core.project import (
    AnalyticalWorkspace, ProjectState, WorkspaceArtifact, load_project, save_project,
)


def _entry():
    analysis = WorkspaceArtifact("builtin.test", "test", payload={
        "config": {"bins": 40, "appearance": {"title": "Paper figure"}},
        "source": {"source_key": "/trace.abf", "channel": 0, "module_id": "detect.test"}},
        tables={"fit": pd.DataFrame({"value": [np.nextafter(1.0, 2.0), np.nan],
                                      "accepted": pd.array([True, pd.NA], dtype="boolean")})},
        blobs={"fit": b"\x00original fit"})
    return make_figure(name="Figure 1", page_id="distributions", source_label="Trace",
                       analysis=analysis, snapshot_metadata={"version": 1},
                       snapshot_blobs={"png": b"frozen PNG", "data": b"exact values"},
                       live_state={"calibration": 1})


def test_figures_roundtrip_and_unknown_artifacts_survive(tmp_path):
    entry = _entry()
    unknown = WorkspaceArtifact("future", FIGURE_KIND, schema_version=99,
                                payload={"future": True}, blobs={"untouched": b"future"})
    workspace = AnalyticalWorkspace(artifacts=[entry, unknown], extra={"plugin": "keep"})
    path = tmp_path / "figures.nanoscope"
    save_project(path, ProjectState(workspace=workspace))
    loaded = load_project(path).workspace
    assert len(figures(loaded)) == 1
    back = figures(loaded)[0]
    assert back.payload == entry.payload
    assert back.blobs == entry.blobs
    pd.testing.assert_frame_equal(back.tables["fit"], entry.tables["fit"], check_exact=True)
    carried = capture_analysis_workspace(loaded)
    assert carried.artifacts[1].blobs == unknown.blobs
    assert carried.artifacts[1].schema_version == 99
    assert carried.extra == {"plugin": "keep"}


def test_duplicate_and_rename_leave_snapshots_unchanged():
    entry = _entry()
    before = deepcopy(entry)
    duplicate = duplicate_figure(entry)
    assert duplicate.artifact_id != entry.artifact_id
    assert duplicate.payload["export_path"] is None
    assert duplicate.blobs == entry.blobs
    rename_figure(duplicate, "For manuscript")
    duplicate.tables["fit"].iloc[0, 0] = -10
    duplicate.payload["analysis"]["payload"]["config"]["bins"] = 80
    assert entry.payload == before.payload
    pd.testing.assert_frame_equal(entry.tables["fit"], before.tables["fit"])
    assert duplicate.payload["name"] == "For manuscript"
    for invalid in ("", "   ", "x" * 201):
        with pytest.raises(ValueError):
            rename_figure(duplicate, invalid)


def test_freshness_distinguishes_missing_changed_and_already_stale():
    entry = _entry()
    assert figure_status(entry, {"calibration": 1})[0] == "Current"
    assert figure_status(entry, {"calibration": 2})[0] == "Outdated"
    assert figure_status(entry, None)[0] == "Saved snapshot"
    entry.payload["capture_status"] = "stale"
    assert figure_status(entry, None)[0] == "Outdated"


def test_sources_include_all_recordings_and_exclude_launch_ids():
    a = WorkspaceArtifact("a", "b", payload={"points": [
        {"source_key": "/a.abf", "channel": 0, "doc_id": "launch-1", "v": 1},
        {"source_key": "/b.abf", "channel": 1, "doc_id": "launch-2", "v": 2},
        {"source_key": "/a.abf", "channel": 0, "doc_id": "launch-1", "v": 3}]})
    recipes = source_recipes(a)
    assert len(recipes) == 2
    assert all("doc_id" not in recipe for recipe in recipes)
    a.payload["figure_sources"] = [{"source_key": "/committed.abf", "channel": 0}]
    assert [r["source_key"] for r in source_recipes(a)] == ["/committed.abf"]
    stamps = {"event_frame_fingerprint": "saved frame", "result_fingerprint": "saved result",
              "module_version": "2.1"}
    a.payload["figure_sources"][0].update(stamps)
    binding = source_recipes(a)[0]
    assert {key: binding[key] for key in stamps} == stamps
    binding["event_frame_fingerprint"] = "caller mutation"
    assert source_recipes(a)[0]["event_frame_fingerprint"] == "saved frame"


def test_manual_measurement_status_tracks_exact_table_not_style():
    panel = SimpleNamespace(measurements_df=pd.DataFrame({"depth_pA": [20.0]}))
    owner = SimpleNamespace(_analysis_panel=SimpleNamespace(measure_panel=panel))
    recipes = [{"kind": "manual", "source_key": "manual-measurements"}]
    original = source_state(owner, recipes)
    owner.appearance = {"title": "New title", "color": "red"}
    assert source_state(owner, recipes) == original
    panel.measurements_df.iloc[0, 0] = 25
    assert source_state(owner, recipes) != original


def test_source_status_detects_calibration_file_and_voltage_changes(tmp_path):
    from dataclasses import replace
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    path = tmp_path / "trace.csv"
    path.write_bytes(b"trace A")
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=4,
                         sample_rate_hz=1000, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(0, "current", "pA", ChannelKind.CURRENT),))
    document = TraceDocument(meta, (np.ones(4, dtype=np.float32),))
    gain = [1.0]
    active = [document]
    session = SimpleNamespace(
        documents=lambda: tuple(active), document=lambda _id: active[0],
        calibration=lambda _id, ch: SimpleNamespace(fingerprint=lambda: (gain[0], 0.0, "pA")),
        results=SimpleNamespace(for_doc=lambda _id: []))
    owner = SimpleNamespace(_session=session)
    recipes = [{"source_key": str(path), "channel": 0,
                "channel_name": "current", "channel_kind": "current"}]
    original = source_state(owner, recipes)
    assert original is not None
    owner.title = "Appearance-only edit"
    assert source_state(owner, recipes) == original
    gain[0] = 2
    assert source_state(owner, recipes) != original
    gain[0] = 1
    active[0] = replace(document, meta=replace(meta, nominal_voltage_mV=-100))
    assert source_state(owner, recipes) != original
    active[0] = document
    path.write_bytes(b"trace B")
    assert source_state(owner, recipes) != original
    active.clear()
    assert source_state(owner, recipes) is None


@pytest.fixture
def loaded_source(tmp_path):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    path = tmp_path / "same_path.csv"
    path.write_bytes(b"unchanged file at capture and subsequent check")
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=4,
                         sample_rate_hz=1000, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(0, "current", "pA", ChannelKind.CURRENT),))
    active = [TraceDocument(meta, (np.array([1., 2., 3., 4.], dtype=np.float32),))]
    calibration = [1., 0.]
    owner = SimpleNamespace(_session=SimpleNamespace(
        documents=lambda: tuple(active), document=lambda _id: active[0],
        calibration=lambda _id, _ch: SimpleNamespace(
            fingerprint=lambda: (*calibration, "pA")),
        results=SimpleNamespace(for_doc=lambda _id: [])))
    recipes = [{"source_key": str(path), "channel": 0,
                "channel_name": "current", "channel_kind": "current"}]
    return owner, recipes, active, calibration, meta


def test_same_path_reload_is_checked_against_loaded_bytes_not_current_disk(loaded_source):
    from nanoscope.core.document import TraceDocument
    owner, recipes, active, _calibration, meta = loaded_source
    first = source_state(owner, recipes)
    assert first["sources"][0]["loaded_content"]
    entry = _entry()
    entry.payload["source_state"] = first
    # The original load may predate a disk overwrite. At both capture and reload
    # the on-disk fingerprint and sample count are identical; the actual samples
    # in this newly loaded document are the independent evidence that differs.
    active[0] = TraceDocument(meta, (np.array([1., 2., 30., 4.], dtype=np.float32),))
    later = source_state(owner, recipes)
    assert later["sources"][0]["file_content"] == first["sources"][0]["file_content"]
    assert later["sources"][0]["loaded_content"] != first["sources"][0]["loaded_content"]
    assert figure_status(entry, later)[0] == "Outdated"
    # The per-load uid is not persisted as a content difference: a genuine equal
    # reload remains current rather than warning just because the app reopened it.
    active[0] = TraceDocument(meta, (np.array([1., 2., 3., 4.], dtype=np.float32),))
    assert figure_status(entry, source_state(owner, recipes))[0] == "Current"


def test_loaded_content_cache_avoids_rehash_for_style_gain_and_bias(loaded_source, monkeypatch):
    import nanoscope.core.figure_library as library
    owner, recipes, active, calibration, _meta = loaded_source
    original_hash = library._hash_loaded_channel
    hashed = []
    def counted(array):
        hashed.append(array)
        return original_hash(array)
    monkeypatch.setattr(library, "_hash_loaded_channel", counted)
    before = source_state(owner, recipes)
    owner.style = {"title": "Different title", "line_width": 4}
    assert source_state(owner, recipes) == before
    calibration[0] = 2.
    assert source_state(owner, recipes) != before
    calibration[0] = 1.
    calibration[1] = 10.
    assert source_state(owner, recipes) != before
    calibration[1] = 0.
    assert source_state(owner, recipes) == before
    assert len(hashed) == 1 and hashed[0] is active[0].channels[0]


def test_loaded_hash_uses_buffer_without_copying_or_mutating():
    import hashlib
    import struct
    from nanoscope.core.figure_library import _hash_loaded_channel
    class NoCopyArray(np.ndarray):
        def copy(self, *args, **kwargs):
            pytest.fail("Hashing copied the recording")
        def tobytes(self, *args, **kwargs):
            pytest.fail("Hashing materialized recording-sized bytes")
    array = np.array([1., 2.], dtype=np.float32).view(NoCopyArray)
    flags = array.flags.writeable
    assert _hash_loaded_channel(array) == hashlib.sha256(struct.pack("=ff", 1., 2.)).hexdigest()
    np.testing.assert_array_equal(array, [1., 2.])
    assert array.flags.writeable == flags


def test_loaded_content_cache_is_detached_weak_and_bounded(loaded_source, monkeypatch):
    from collections import OrderedDict
    import gc
    import weakref
    import nanoscope.core.figure_library as library
    from nanoscope.core.document import TraceDocument
    _owner, _recipes, _active, _calibration, meta = loaded_source
    cache = OrderedDict()
    monkeypatch.setattr(library, "_LOADED_STAMP_CACHE", cache)
    monkeypatch.setattr(library, "_LOADED_STAMP_CACHE_MAX", 2)
    documents = [TraceDocument(meta, (np.full(4, value, dtype=np.float32),))
                 for value in (1., 2., 3.)]
    for document in documents:
        returned = library.loaded_source_content(document)
        returned[0]["sha256"] = "caller edit"
        assert library.loaded_source_content(document)[0]["sha256"] != "caller edit"
    assert len(cache) == 2 and id(documents[0]) not in cache
    reference = weakref.ref(document)
    del document
    documents.clear()
    gc.collect()
    assert reference() is None and not cache
    assert library.loaded_source_content(SimpleNamespace(channels=[])) is None


def test_loaded_content_includes_voltage_channel_even_for_current_recipe(loaded_source):
    from dataclasses import replace
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind
    owner, recipes, active, _calibration, meta = loaded_source
    meta = replace(meta, channels=(*meta.channels,
                                  ChannelInfo(1, "voltage", "mV", ChannelKind.VOLTAGE)))
    active[0] = TraceDocument(meta, (np.ones(4, dtype=np.float32),
                                    np.full(4, -100., dtype=np.float32)))
    first = source_state(owner, recipes)
    active[0] = TraceDocument(meta, (np.ones(4, dtype=np.float32),
                                    np.full(4, -200., dtype=np.float32)))
    assert source_state(owner, recipes) != first
