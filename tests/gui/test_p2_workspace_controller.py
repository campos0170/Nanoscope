"""Controller transactions independent of analytical calculations and native canvases."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from nanoscope.core.figure_codecs import NATIVE_KINDS, merge_native_capture
from nanoscope.core.figure_history_state import HistoryCapsule
from nanoscope.core.project import AnalyticalWorkspace, WorkspaceArtifact
from nanoscope.gui.figure_workspace import FigureWorkspace
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_workflow import HistoryState


class Page(QtCore.QObject):
    workspaceChanged = QtCore.Signal()
    def __init__(self, kind, key):
        super().__init__()
        self.kind, self.key = kind, key
        self.word = None
        self.visible = True
        self.ready = True
        self.fail_capture = False
        self.restores = []
        self.capsule = HistoryCapsule("nanoscope.appearance-history/1", PlotAppearance().to_dict())
    def hide(self): self.visible = False
    def isVisible(self): return self.visible
    def inspector_adapter(self): return SimpleNamespace(history=lambda: HistoryState())


class Adapter:
    def __init__(self, page, kind): self.page, self.kind = page, kind
    def capture_native(self):
        native_id, (native_kind, _kind, _label) = next((k, v) for k, v in NATIVE_KINDS.items() if v[1] == self.kind)
        appearance = PlotAppearance(title=self.page.word).to_dict()
        payload = {"config": {"appearance": appearance}} if self.kind in ("noise", "event_waveforms", "voltage_series") else {"appearance": appearance}
        return WorkspaceArtifact(native_id, native_kind, payload=payload)
    def capture_record(self, previous):
        if self.page.fail_capture: raise ValueError("capture failed")
        return merge_native_capture(self.capture_native(), previous)
    def capture_transient(self): return self.page.capsule
    def install_transient(self, capsule): self.page.capsule = capsule
    def restore_record(self, record, mode="project"):
        self.page.restores.append(mode)
        self.page.word = next((t.words.value for t in record.spec.texts if t.element_id == "title"), None)
        return SimpleNamespace(skipped=(), warnings=())
    def can_suspend(self): return self.page.ready, "worker busy"
    def suspend(self): self.page.hide()


@pytest.fixture
def rig(qapp):
    built, mounted, unmounted, bindings, dirty = [], [], [], [], []
    def factory(kind, key):
        page = Page(kind, key)
        built.append(page)
        return page
    owner = FigureWorkspace(factory=factory, adapter_factory=Adapter,
        mount=lambda key, page: mounted.append(key), unmount=lambda key, page: unmounted.append(key),
        initialize=lambda page, record, recipe, binding: None,
        bind_sources=lambda page, record, binding, compute: bindings.append((record.spec.figure_id, binding, compute)),
        current_binding=lambda kind: {"source": {"source_key": "synthetic", "channel": 0}, "follow": False},
        dirty=lambda: dirty.append(1), resident_limit=2)
    yield SimpleNamespace(owner=owner, built=built, mounted=mounted, unmounted=unmounted,
                          bindings=bindings, dirty=dirty)
    for page in built:
        page.ready = True
    owner.clear()
    owner.deleteLater()


def test_new_duplicate_rename_and_deleted_restore_have_independent_identity(rig):
    owner = rig.owner
    first = owner.create("iv")
    owner.page(first).word = "A typed title"
    owner.page(first).workspaceChanged.emit()
    second = owner.duplicate(first)
    assert first != second and owner.current_id() == second
    assert owner.page(second).word == "A typed title"
    assert owner.history(second).mark == 0
    owner.page(second).word = "Different title"
    owner.page(second).workspaceChanged.emit()
    assert owner.page(first).word == "A typed title"
    assert owner.rename(second, "New name") and owner.get(second).spec.name == "New name"
    assert owner.delete(second) and owner.get(second) is None
    assert owner.restore_deleted(second) == second and owner.page(second).word == "Different title"


def test_checkpoint_failure_never_discards_a_live_page_or_old_record(rig):
    owner = rig.owner
    key = owner.create("iv")
    page, before = owner.page(key), owner.get(key)
    page.word = "Uncaptured edit"
    page.fail_capture = True
    assert not owner.delete(key)
    assert owner.page(key) is page and owner.get(key) is before
    with pytest.raises(RuntimeError, match="checkpointed"):
        owner.capture_workspace()
    page.fail_capture = False


def test_unknown_figures_and_snapshots_survive_capture_and_supported_deletion(rig):
    owner = rig.owner
    unknown = WorkspaceArtifact("live.future", "nanoscope.figure", schema_version=99,
                                payload={"name": "Future"}, blobs={"opaque": b"keep"})
    saved = WorkspaceArtifact("figure.saved", "nanoscope.saved-figure", payload={"name": "Snapshot"},
                              blobs={"image": b"immutable"})
    plugin = WorkspaceArtifact("plugin", "foreign", payload={"future": None})
    owner.install_workspace(AnalyticalWorkspace(artifacts=[unknown, saved, plugin], extra={"plugin": [1]}))
    assert owner.unsupported_records()[0]["name"] == "Future" and not rig.built
    key = owner.create("iv")
    assert owner.delete(key)
    captured = owner.capture_workspace()
    assert [(a.artifact_id, a.blobs) for a in captured.artifacts] == [
        (unknown.artifact_id, unknown.blobs), (saved.artifact_id, saved.blobs), (plugin.artifact_id, plugin.blobs)]
    assert captured.extra["plugin"] == [1]
    assert owner.delete("live.future")
    assert not any(a.artifact_id == "live.future" for a in owner.capture_workspace().artifacts)
    owner.restore_deleted("live.future")
    assert next(a for a in owner.capture_workspace().artifacts if a.artifact_id == "live.future").blobs == unknown.blobs


def test_opaque_future_payload_and_project_reset_are_lossless(rig):
    owner = rig.owner
    artifact = WorkspaceArtifact("live.opaque", "nanoscope.figure", schema_version=99,
                                 payload=["a future shape", None], blobs={"raw": b"unchanged"})
    owner.install_workspace(AnalyticalWorkspace(artifacts=[artifact], extra={"plugin": [4],
        "figure_workspace_v1": {"default_style": PlotAppearance(title_size=23).to_dict()}}))
    assert owner.unsupported_records()[0]["name"] == "Unsupported figure"
    assert owner.capture_workspace().artifacts[0].payload == artifact.payload
    owner.clear()
    assert "plugin" not in owner.capture_workspace().extra
    assert owner.capture_workspace().extra["figure_workspace_v1"]["default_style"]["title_size"] == 7


def test_checkpoint_is_not_reentered_by_a_commit_signal_and_keeps_source_status(rig, monkeypatch):
    owner = rig.owner
    key = owner.create("iv")
    entry = owner._entries[key]
    entry.record = replace(entry.record, spec=replace(entry.record.spec,
        result=replace(entry.record.spec.result, status="current")))
    capture = entry.adapter.capture_transient
    calls = []
    def commits_editor():
        calls.append(1)
        entry.page.workspaceChanged.emit()
        return capture()
    monkeypatch.setattr(entry.adapter, "capture_transient", commits_editor)
    assert owner.checkpoint(key) and calls == [1]
    assert owner.get(key).spec.result.status == "current"


def test_failed_mount_preserves_retryable_new_record_and_old_foreground(rig, monkeypatch):
    owner = rig.owner
    first = owner.create("iv")
    mount = owner._mount
    monkeypatch.setattr(owner, "_mount", lambda *args: (_ for _ in ()).throw(ValueError("mount failed")))
    second = owner.create("iv")
    assert owner.current_id() == first and owner.page(second) is None
    assert owner.get(second).spec.extra["figure_workspace"]["uninitialized"] is True
    monkeypatch.setattr(owner, "_mount", mount)
    assert owner.activate(second) and owner.page(second) is not None


def test_future_workspace_is_not_interpreted_or_rewritten(rig):
    owner = rig.owner
    future = AnalyticalWorkspace(version=2, artifacts=[WorkspaceArtifact("opaque", "future", payload=[1, None])],
                                 extra={"figure_workspace_v1": {"order": ["future-id"], "unknown": 4}})
    owner.install_workspace(future)
    assert owner.is_read_only() and not owner.records()
    assert owner.create("iv") is None and not owner.delete("opaque")
    assert owner.capture_workspace() is future
    assert future.extra == {"figure_workspace_v1": {"order": ["future-id"], "unknown": 4}}
