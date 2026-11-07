"""The Figures list routes durable IDs while displaying user-facing names."""
from types import SimpleNamespace as NS

from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.figure_list import CATEGORY, ID, FigureList

from . import p1_hosting as H
from .p2_shell_rig import Controller, record


def _saved(key, name, kind="iv"):
    return NS(artifact_id=key, payload={"name": name, "page_id": kind,
                                      "source_label": "Synthetic source", "created_at": "2026-11-03"})


def test_list_groups_saved_resources_and_keeps_unsupported_visible_without_activation(qtbot):
    listing = FigureList()
    qtbot.addWidget(listing)
    live = record("My plot")
    saved = (_saved("figure.one", "First capture"), _saved("figure.two", "Panel A", "composition"))
    unsupported = ({"id": "future.id", "name": "Future format", "kind": "new", "reason": "Schema 99 is newer"},)
    listing.set_records((live,), saved, current=live.spec.figure_id, unsupported=unsupported)
    assert [listing.tree.topLevelItem(i).text(0) for i in range(4)] == [
        "Live", "Snapshots", "Compositions", "Unsupported"]
    assert listing._items[live.spec.figure_id].text(0) == "My plot"
    assert listing._items["figure.two"].data(0, CATEGORY) == "composition"
    future = listing._items["future.id"]
    assert future.text(0) == "Future format" and "Schema 99" in future.toolTip(0)
    activated = []
    listing.liveActivated.connect(activated.append)
    listing.savedActivated.connect(activated.append)
    listing.tree.setCurrentItem(future)
    assert not activated and not listing.duplicate_button.isEnabled()
    assert listing.delete_button.isEnabled()


def test_saved_selection_emits_exact_composition_ids_and_remains_on_refresh(qtbot):
    listing = FigureList()
    qtbot.addWidget(listing)
    listing.resize(300, 600)
    listing.show()
    saved = (_saved("figure.one", "One"), _saved("figure.two", "Two"))
    listing.set_records((), saved)
    for item in listing._items.values():
        item.setSelected(True)
    ids = listing.selected_ids("saved")
    calls = []
    listing.compositionRequested.connect(calls.append)
    H.click(listing.compose_button)
    assert calls == [ids] and set(ids) == {"figure.one", "figure.two"}
    listing.set_records((), saved)
    assert set(listing.selected_ids("saved")) == set(ids)


def test_new_duplicate_rename_delete_restore_commands_use_controller_not_page_factories(qtbot, monkeypatch):
    value = record("Independent figure")
    key = value.spec.figure_id
    owner = Controller(value)
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    explorer.show()
    listing = explorer.figures_list()
    listing.new_menu.actions()[0].trigger()
    assert owner.calls[0] == ("create", "distributions", {"plot_type": "Histogram"}, None)
    explorer._figure_command("duplicate", key)
    assert len(owner.records()) == 3
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *_a, **_kw: ("Renamed", True))
    explorer._rename_figure(key)
    assert value.spec.name == "Renamed"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.Yes)
    explorer._delete_figures((key,))
    assert owner.get(key) is None
    explorer._figure_command("restore_deleted")
    assert owner.get(key) is value and explorer.page_count() == 0


def test_live_selection_and_snapshot_requests_are_id_based(qtbot):
    values = (record("A"), record("B"))
    owner = Controller(*values)
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    explorer.show()
    listing = explorer.figures_list()
    for item in listing._items.values():
        item.setSelected(True)
    assert set(owner.selected_ids()) == {v.spec.figure_id for v in values}
    calls = []
    explorer.snapshotRequested.connect(calls.append)
    H.click(listing.snapshot_button)
    assert calls == [owner.current_id()] and explorer.page_count() == 0


def test_a_project_switch_during_delete_confirmation_does_not_delete_either_record(qtbot, monkeypatch):
    original = record("Original")
    owner = Controller(original)
    replacement = Controller(record("New project"))
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)

    def switch_project(*_args, **_kwargs):
        explorer.bind_figure_workspace(replacement)
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", switch_project)
    explorer._delete_figures((original.spec.figure_id,))
    assert owner.get(original.spec.figure_id) is original
    assert len(replacement.records()) == 1
    assert not any(call[0] == "delete" for call in owner.calls + replacement.calls)
