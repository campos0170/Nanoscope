"""Project styles own pinned revisions without hydrating or recalculating figures."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nanoscope.core.figure_style import FigureStyle
from nanoscope.core.figure_style_registry import registry_from_workspace
from nanoscope.core.figure_spec import thaw_json
from nanoscope.core.project import AnalyticalWorkspace
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_styles import StyleService
from .test_p2_workspace_controller import rig  # noqa: F401


def dormant(rig, monkeypatch):
    owner = rig.owner
    monkeypatch.setattr(owner, "activate", lambda _key: False)
    return owner


def test_default_revision_is_pinned_and_new_uses_edited_definition(rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    first = owner.create("iv")
    original = owner.get(first)
    style = owner.save_style("Apo figures", look={"title_size_pt": 9.5, "line_width_pt": .625})
    owner.set_default_style(style.ref)
    second = owner.create("iv")
    assert owner.get(first) == original
    assert owner.get(second).spec.presentation.style_ref == style.ref.to_dict()
    revised = owner.save_style("Apo figures", overwrite=True,
                               look={"title_size_pt": 10.25, "line_width_pt": .75})
    assert revised.revision == 2
    assert owner.get(second).spec.presentation.style_ref == style.ref.to_dict()
    third = owner.create("iv")
    assert owner.get(third).spec.presentation.style_ref == revised.ref.to_dict()
    assert not rig.built


def test_project_save_reopen_keeps_style_revisions_and_figure_overrides(rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    key = owner.create("iv")
    style = owner.save_style("Apo", look={"title_size_pt": 9.5, "text_roles": {"title": {"bold": True}}})
    look = PlotAppearance.from_mapping(bind_native_style({"title": "not part of style"}, style))
    assert owner.apply_look_to_figure(key, look, keep_hand=False)
    applied = owner.get(key)
    assert owner.undo(key)
    assert owner.get(key).spec.presentation.style_ref != applied.spec.presentation.style_ref
    assert owner.redo(key)
    snapshot = owner.capture_workspace()
    owner.install_workspace(snapshot)
    assert owner.get(key).spec.presentation == applied.spec.presentation
    restored = next(s for s in owner.available_styles() if s.name == "Apo")
    assert restored == style
    assert owner.page(key) is None and not rig.built


def test_same_name_overwrite_refreshes_service_value_and_deleted_ref_survives(rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    monkeypatch.setattr(PlotAppearanceWidget, "user_preset_states", staticmethod(lambda: {}))
    service = StyleService(SimpleNamespace(figure_workspace=lambda: owner))
    style = owner.save_style("Apo", look={"title_size_pt": 9.25})
    first = dict(service.saved_looks())["Apo"]
    revised = owner.save_style("Apo", overwrite=True, look={"title_size_pt": 10.5})
    next_look = dict(service.saved_looks())["Apo"]
    assert first.title_size == 9.25 and next_look.title_size == 10.5
    assert first.style_binding["ref"]["revision"] == 1
    assert next_look.style_binding["ref"]["revision"] == 2
    key = owner.create("iv")
    owner.apply_look_to_figure(key, first)
    before = owner.get(key)
    owner.delete_style(style.style_id)
    assert "Apo" not in dict(service.saved_looks())
    assert owner.get(key) == before
    assert len([s for s in owner.style_registry().styles if s.style_id == revised.style_id]) == 2


def test_legacy_import_is_once_and_future_registry_remains_opaque(rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    legacy = {"My old look": {"title_size": 11.25, "line_width": .75}}
    owner.import_legacy_styles(legacy)
    count = len(owner.available_styles())
    owner.import_legacy_styles(legacy)
    assert len(owner.available_styles()) == count
    opaque = {"version": 99, "unknown": [1, None, {"raw": "keep"}]}
    owner.install_workspace(AnalyticalWorkspace(extra={"figure_styles_v1": opaque}))
    assert owner.style_registry() is None
    owner.import_legacy_styles(legacy)
    with pytest.raises(ValueError):
        owner.save_style("Replacement", look={"title_size_pt": 9})
    assert owner.create("iv") is None
    assert owner.capture_workspace().extra["figure_styles_v1"] == opaque


def test_unreadable_legacy_default_is_not_replaced_on_an_unrelated_save(rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    raw = ["future", {"preserve": None}]
    owner.install_workspace(AnalyticalWorkspace(extra={"figure_workspace_v1": {"default_style": raw}}))
    assert owner.style_registry() is None
    assert owner.capture_workspace().extra["figure_workspace_v1"]["default_style"] == raw


def test_sparse_imported_default_keeps_native_body_and_exact_definition(rig, monkeypatch):
    from nanoscope.core.figure_codecs import encode_legacy
    from nanoscope.gui.figure_adapter_plots import _native_config
    owner = dormant(rig, monkeypatch)
    owner.install_workspace(AnalyticalWorkspace(extra={"figure_workspace_v1": {
        "default_style": {"title_size": 19.}}}))
    exact = owner.default_style().to_dict()
    key = owner.create("distributions")
    record = owner.get(key)
    native = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    assert _native_config(native).style.font_size_pt == 19.
    assert record.spec.presentation.overrides["body_size_pt"] == 19.
    owner.install_workspace(owner.capture_workspace())
    restored = owner.get(key)
    assert _native_config(encode_legacy(restored.spec, restored.resources,
        materialize_defaults=True)).style.font_size_pt == 19.
    assert owner.default_style().to_dict() == exact
    assert "body_size_pt" not in owner.default_style().look


def test_editing_grid_preserves_untouched_fractional_definition(qtbot):
    from nanoscope.gui.project_style_editor import ProjectStyleEditor
    style = FigureStyle(style_id="project.precise", revision=1, name="Precise",
        look={"line_width_pt": .625, "title_size_pt": 9.125, "grid": False,
              "future": {"values": [1, None]}})
    dialog = ProjectStyleEditor(style)
    qtbot.addWidget(dialog)
    dialog.grid.setCurrentIndex(dialog.grid.findData(True))
    result = dialog.look()
    assert result == {**thaw_json(style.look), "grid": True}


def _bound_explorer(qtbot, owner):
    from .p1_hosting import AnalysisExplorer
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    return explorer


def test_style_menu_rebuilds_for_future_then_readable_project_and_back(qtbot, rig, monkeypatch):
    owner = dormant(rig, monkeypatch)
    monkeypatch.setattr(PlotAppearanceWidget, "user_preset_states", staticmethod(lambda: {}))
    future = AnalyticalWorkspace(extra={"figure_styles_v1": {"version": 99}})
    owner.install_workspace(future)
    explorer = _bound_explorer(qtbot, owner)
    menu = explorer._style_menu
    menu._sync()
    assert not [a for a in menu.scope_menu("page").actions() if a.isEnabled() and not a.isSeparator()]
    owner.install_workspace(AnalyticalWorkspace())
    owner.create("iv")
    menu._sync()
    assert menu._default_menu.menuAction().isVisible()
    assert menu.scope_menu("selected").menuAction().isVisible()
    names = [a.text() for a in menu.scope_menu("page").actions() if not a.isSeparator()]
    assert owner.default_style().name in names
    owner.install_workspace(future)
    menu._sync()
    assert not menu.scope_menu("page").isEnabled()
    assert not menu._default_menu.actions()


def test_modeless_manager_cannot_delete_same_named_style_from_another_project(qtbot, rig, monkeypatch):
    from nanoscope.gui.plot_styles import _ManageLooks
    from PySide6 import QtWidgets
    owner = dormant(rig, monkeypatch)
    monkeypatch.setattr(PlotAppearanceWidget, "user_preset_states", staticmethod(lambda: {}))
    a = owner.save_style("Apo", look={"title_size_pt": 9.})
    explorer = _bound_explorer(qtbot, owner)
    manager = _ManageLooks(explorer, explorer._style)
    manager.refill()
    manager.list.setCurrentRow(next(i for i in range(manager.list.count()) if manager.list.item(i).text() == "Apo"))
    owner.install_workspace(AnalyticalWorkspace())
    b = owner.save_style("Apo", look={"title_size_pt": 11.})
    assert a.style_id != b.style_id
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: pytest.fail("stale delete asked for confirmation"))
    manager._delete()
    assert b in owner.available_styles()
    assert "changed" in manager._note.text()


def test_edit_and_save_dialogs_refuse_a_project_switch(qtbot, rig, monkeypatch):
    from nanoscope.gui.plot_styles import _ManageLooks
    from nanoscope.gui.project_style_editor import ProjectStyleEditor
    from PySide6 import QtWidgets
    owner = dormant(rig, monkeypatch)
    monkeypatch.setattr(PlotAppearanceWidget, "user_preset_states", staticmethod(lambda: {}))
    owner.save_style("Apo", look={"title_size_pt": 9.})
    explorer = _bound_explorer(qtbot, owner)
    manager = _ManageLooks(explorer, explorer._style)
    manager.refill()
    manager.list.setCurrentRow(next(i for i in range(manager.list.count()) if manager.list.item(i).text() == "Apo"))
    def switched_edit(dialog):
        owner.install_workspace(AnalyticalWorkspace())
        owner.save_style("Apo", look={"title_size_pt": 11.})
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(ProjectStyleEditor, "exec", switched_edit)
    manager._edit()
    replacement = next(s for s in owner.available_styles() if s.name == "Apo")
    assert replacement.revision == 1 and replacement.look["title_size_pt"] == 11.
    key = owner.create("iv")
    owner._current = key
    def switched_save(*args):
        owner.install_workspace(AnalyticalWorkspace())
        return "Must not save", True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", switched_save)
    explorer._style_menu._ask_save()
    assert "Must not save" not in {s.name for s in owner.available_styles()}
    assert "changed" in explorer.statusBar().currentMessage()



def test_save_current_sparse_look_keeps_native_inheritance_when_copied(rig, monkeypatch):
    from nanoscope.core.figure_codecs import decode_legacy
    from nanoscope.core.project import WorkspaceArtifact
    from nanoscope.gui.figure_workspace import _Entry
    from nanoscope.core.figure_style_mapping import project_legacy_style
    from nanoscope.core.figure_style import style_binding
    owner = dormant(rig, monkeypatch)
    raw = WorkspaceArtifact("iv", "nanoscope.iv", payload={"appearance": {"title_size": 19.}})
    from nanoscope.core.figure_codecs import NATIVE_KINDS
    native_id, (native_kind, _, _) = next((key, info) for key, info in NATIVE_KINDS.items() if info[1] == "iv")
    raw = replace(raw, artifact_id=native_id, kind=native_kind)
    record = decode_legacy(raw, figure_id="4f98b8f6-1f67-4ada-847d-95bd61427c54", name="Sparse").record
    owner._entries["4f98b8f6-1f67-4ada-847d-95bd61427c54"] = _Entry(record)
    owner._current = "4f98b8f6-1f67-4ada-847d-95bd61427c54"
    saved = owner.save_style("Copied native")
    assert saved.provenance["inherit_native_defaults"] is True
    assert saved.look["title_size_pt"] == 19.
    assert "axis_size_pt" not in saved.look
    assert "axis_size" not in project_legacy_style(style_binding(saved), "plot_appearance", {})
    owner.apply_look_to_figure("4f98b8f6-1f67-4ada-847d-95bd61427c54", PlotAppearance.from_mapping(bind_native_style({}, saved)))
    copied = owner.save_style("Copied again")
    assert copied.provenance["inherit_native_defaults"] is True
    assert "axis_size_pt" not in copied.look
