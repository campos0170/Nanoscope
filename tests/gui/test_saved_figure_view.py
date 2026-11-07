"""Saved selection owns the Explorer canvas/actions without opening another inventory."""
from PySide6 import QtWidgets

from nanoscope.core.project import WorkspaceArtifact
from nanoscope.core.figure_library import FIGURE_KIND, duplicate_figure
from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.saved_figure_view import SavedFigureView

from . import p1_hosting as H
from .p2_shell_rig import Controller, record


def _rig(qtbot):
    live = record("Live figure", kind="iv")
    owner = Controller(live)
    entries = [WorkspaceArtifact(
        artifact_id="figure.saved", kind=FIGURE_KIND, payload={
            "name": "Saved A", "page_id": "iv", "source_label": "Pinned recording A",
            "created_at": "2026-11-04", "snapshot": {}}, blobs={})]
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    status = lambda entry: ("Saved snapshot", "Frozen numerical data and artwork.")
    explorer.set_saved_figure_provider(lambda: entries, status_provider=status)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page(live.spec.figure_id, page, kind="iv")
    view = SavedFigureView(explorer, entries=lambda: entries, status=status, append=entries.append)
    view.changed.connect(explorer.refresh_figures)
    explorer.show()
    return explorer, owner, live, page, view, entries


def test_saved_canvas_exports_target_snapshot_and_live_actions_have_no_target(qtbot, monkeypatch):
    explorer, _owner, live, page, view, entries = _rig(qtbot)
    assert explorer.show_saved_figure(entries[0].artifact_id, view)
    bar = explorer.toolbar()
    assert explorer.current_page_id() == entries[0].artifact_id
    assert explorer.figures_list().current_id("snapshot") == entries[0].artifact_id
    assert explorer.figures_list().selected_ids("saved") == (entries[0].artifact_id,)
    assert explorer.saved_figure_selected()
    assert not view.isWindow()
    assert not view.findChildren(QtWidgets.QListWidget)
    assert "Pinned recording A" in explorer._context_label.text()
    assert bar.adapter() is None
    assert not bar.update_action.isEnabled()
    assert not bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    assert not bar.style_button.isEnabled()
    bar.run_update()
    assert page.updates == 0
    exported = []
    monkeypatch.setattr(view, "export_current", lambda *args, **kwargs: exported.append((view.current_id(), kwargs)))
    monkeypatch.setattr(view, "export_data", lambda: exported.append((view.current_id(), "data")))
    bar._fill_export_menu()
    actions = {action.text(): action for action in bar._export_menu.actions()}
    for name in ("Figure…", "Data…", "Bundle…"):
        assert actions[name].isEnabled()
        actions[name].trigger()
    assert all(item[0] == entries[0].artifact_id for item in exported)
    assert "Snapshot to Figures…" not in actions
    assert explorer.raise_page(live.spec.figure_id)
    assert bar.update_action.isEnabled() and bar.style_button.isEnabled()
    bar.run_update()
    assert page.updates == 1
    assert explorer.raise_page(entries[0].artifact_id)
    assert not bar.update_action.isEnabled()


def test_saved_rename_duplicate_and_delete_dispatch_do_not_mutate_live_figure(qtbot, monkeypatch):
    explorer, owner, live, _page, view, entries = _rig(qtbot)
    saved = entries[0]
    explorer.show_saved_figure(saved.artifact_id, view)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *_args, **_kwargs: ("Named snapshot", True))
    explorer._rename_figure(saved.artifact_id)
    assert saved.payload["name"] == "Named snapshot"
    assert live.spec.name == "Live figure"
    created = []
    view.entryCreated.connect(created.append)
    duplicate = explorer._figure_command("duplicate", saved.artifact_id)
    assert duplicate.artifact_id != saved.artifact_id
    assert duplicate.blobs == saved.blobs
    assert created == [duplicate.artifact_id]
    assert len(owner.records()) == 1
    removed = []
    explorer.savedDeleteRequested.connect(removed.append)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args, **_kwargs: QtWidgets.QMessageBox.Yes)
    explorer._delete_figures((saved.artifact_id,))
    assert removed == [(saved.artifact_id,)]
    assert owner.get(live.spec.figure_id) is live


def test_deleting_displayed_snapshot_returns_home_without_selecting_hidden_live_page(qtbot):
    explorer, _owner, _live, page, view, entries = _rig(qtbot)
    explorer.show_saved_figure(entries[0].artifact_id, view)
    entries.clear()
    explorer.refresh_figures()
    assert explorer.current_page_id() == "analysis_home"
    assert not explorer.saved_figure_selected()
    assert not explorer.toolbar().update_action.isEnabled()
    assert page.updates == 0


def test_saved_user_multiselection_survives_canvas_display_and_new_capture_selects_exact_row(qtbot):
    explorer, _owner, _live, _page, view, entries = _rig(qtbot)
    second = duplicate_figure(entries[0])
    entries.append(second)
    explorer.refresh_figures()
    listing = explorer.figures_list()
    both = (entries[0].artifact_id, second.artifact_id)
    listing.restore_selection(both, current_id=second.artifact_id)
    explorer.show_saved_figure(second.artifact_id, view)
    assert set(listing.selected_ids("saved")) == set(both)
    third = duplicate_figure(second)
    entries.append(third)
    explorer.show_saved_figure(third.artifact_id, view)
    assert listing.current_id("snapshot") == third.artifact_id
    assert listing.selected_ids("saved") == (third.artifact_id,)


def test_reopening_same_live_figure_from_saved_selection_restores_matching_row(qtbot):
    explorer, owner, live, page, view, entries = _rig(qtbot)
    listing = explorer.figures_list()
    explorer.show_saved_figure(entries[0].artifact_id, view)
    assert owner.current_id() == live.spec.figure_id
    owner.activate(live.spec.figure_id)
    explorer.refresh_figures()
    assert explorer.current_page_id() == live.spec.figure_id
    assert listing.current_id() == live.spec.figure_id
    assert listing.selected_ids() == (live.spec.figure_id,)
    assert not listing.selected_ids("saved")
    assert listing.snapshot_button.isEnabled()
    assert explorer.toolbar().update_action.isEnabled()
    assert page.updates == 0

    second = record("Other live figure", kind="iv")
    owner._records[second.spec.figure_id] = second
    explorer.refresh_figures()
    both = (live.spec.figure_id, second.spec.figure_id)
    listing.restore_selection(both, current_id=live.spec.figure_id)
    owner.activate(live.spec.figure_id)
    explorer.refresh_figures()
    assert set(listing.selected_ids()) == set(both)


def test_standalone_library_retains_its_explicit_inventory(qtbot):
    from nanoscope.gui.figure_library import FigureLibrary
    _explorer, _owner, _live, _page, view, entries = _rig(qtbot)
    library = FigureLibrary(entries=lambda: entries, status=view._status, append=entries.append)
    qtbot.addWidget(library)
    library.show()
    assert library._list.count() == 1
    assert library._rename_btn.isVisible()
    assert library.current_entry() is entries[0]
