"""Stable figure identities are distinct from kinds and from resident widgets."""
import pytest
from PySide6 import QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer

from . import p1_hosting as H
from .p2_shell_rig import Controller, record


def test_three_same_kind_surfaces_have_independent_panes_and_shared_kind_preferences(qtbot):
    records = [record(f"I-V {i}") for i in range(3)]
    owner = Controller(*records)
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    pages = []
    for rec in records:
        page = H.SyntheticPlotPage("iv")
        pages.append(page)
        explorer.add_page(rec.spec.figure_id, page)
    explorer.show()
    qtbot.waitExposed(explorer)
    assert explorer.page_count() == 3 and explorer._tabs.tabBar().isHidden()
    assert len({id(explorer.inspector().pane(r.spec.figure_id)) for r in records}) == 3
    for rec, page in zip(records, pages):
        adapter = explorer.adapter(rec.spec.figure_id)
        assert adapter.key == rec.spec.figure_id and adapter.kind == "iv" and adapter.page is page
        assert explorer.inspector().pane(rec.spec.figure_id).data_pane._kind == "iv"
        owner.activate(rec.spec.figure_id)
        assert explorer.current_page_id() == rec.spec.figure_id
        assert explorer.inspector().current_key() == rec.spec.figure_id
    replacement = H.SyntheticPlotPage("iv")
    qtbot.addWidget(replacement)
    with pytest.raises(ValueError, match="Remove"):
        explorer.add_page(records[0].spec.figure_id, replacement)
    assert explorer.page(records[0].spec.figure_id) is pages[0]


def test_closing_bound_explorer_hides_without_closing_or_resetting_pages(qtbot, monkeypatch):
    rec = record()
    owner = Controller(rec)
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page(rec.spec.figure_id, page)
    calls = []
    monkeypatch.setattr(page, "close", lambda: calls.append("closed"))
    explorer.close()
    assert not explorer.isVisible() and not calls
    assert explorer.page(rec.spec.figure_id) is page and owner.get(rec.spec.figure_id) is rec


def test_record_only_listing_never_builds_pages_and_unbinding_restores_legacy_tabs(qtbot):
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    owner = Controller(*(record(f"Figure {i}") for i in range(20)))
    explorer.bind_figure_workspace(owner)
    assert explorer.page_count() == 0 and len(explorer.figures_list()._items) == 20
    explorer.bind_figure_workspace(None)
    assert explorer.figures_list().isHidden() and not explorer._tabs.tabBar().isHidden()
    owner.create("iv")
    QtWidgets.QApplication.processEvents()
    assert explorer.page_count() == 0 and not explorer.figures_list()._items
