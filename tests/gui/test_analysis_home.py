"""The analysis catalog advertises capabilities without instantiating analyses."""
import ast
from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui import analysis_catalog as catalog
from nanoscope.gui.analysis_home import AnalysisHome
from nanoscope.gui.analysis_explorer import AnalysisExplorer, PAGE_HOME
from nanoscope.gui.analysis_launcher import AnalysisLauncher, SURFACES


def test_catalog_contains_every_page_and_support_route_with_canonical_names():
    routes = tuple(entry.route for entry in catalog.ANALYSIS_ENTRIES)
    assert len(routes) == len(set(routes)) == 12
    assert set(catalog.PAGE_IDS) <= set(routes)
    assert set(routes) - set(catalog.PAGE_IDS) == {
        "guided_first_analysis", "report_figures", "event_gallery", "figure_library"}
    assert all(entry.label and entry.description and entry.inputs for entry in catalog.ANALYSIS_ENTRIES)
    assert "No event detection is required" in catalog.ENTRY_BY_ROUTE["noise"].inputs
    assert "not certify" in catalog.STATUS_HELP
    mapping = {"populations": "distributions", "event_plots": "event_waveforms", "psd": "noise",
               "iv": "iv", "vdep": "voltage_series", "figures": "report_figures", "gallery": "event_gallery"}
    assert all(surface.label == catalog.ANALYSIS_LABELS[mapping[surface.sid]] for surface in SURFACES)


def test_home_metadata_imports_no_analytical_panel_or_recording_code():
    from nanoscope.gui import analysis_home
    for module in (catalog, analysis_home):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert all(not name.startswith(("nanoscope.analysis", "analysis.", "core.", "nanoscope.core")) for name in imports)
        assert not any("panel" in name or "main_window" in name for name in imports)


def test_home_gates_are_read_only_visible_and_rechecked_at_activation(qtbot):
    gates = {entry.route: (False, "Open a current recording first.") for entry in catalog.ANALYSIS_ENTRIES}
    reads, launches = [], []
    def available(route):
        reads.append(route)
        return gates[route]
    home = AnalysisHome(availability_provider=available)
    qtbot.addWidget(home)
    home.launchRequested.connect(launches.append)
    assert set(reads) == set(home.route_ids())
    assert not launches
    home.set_current_route("noise")
    assert "Needs input" in home._availability.text()
    assert not home._items["noise"].isHidden()
    assert home._open.isEnabled()
    home.request_current()
    assert not launches and "current recording" in home._availability.text()
    gates["noise"] = (True, "Current channel and sampling rate are available.")
    home.request_current()
    assert launches == ["noise"]
    assert "Available" in home._availability.text()


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, KeyError, TypeError])
def test_unavailable_provider_failure_is_explained_without_launch(qtbot, failure):
    def unavailable(_route):
        raise failure("gone")
    home = AnalysisHome(availability_provider=unavailable)
    qtbot.addWidget(home)
    launches = []
    home.launchRequested.connect(launches.append)
    home.request_current()
    assert not launches
    assert "could not be checked" in home._availability.text()


def test_search_selects_matching_capability_without_opening_it_and_enter_launches(qtbot):
    home = AnalysisHome(availability_provider=lambda _route: (True, "Ready"))
    qtbot.addWidget(home)
    home.show()
    launches = []
    home.launchRequested.connect(launches.append)
    home._search.setText("ECDF")
    assert home.current_route() == "distribution_summaries"
    assert not launches
    home._list.setFocus()
    qtbot.keyClick(home._list, QtCore.Qt.Key.Key_Return)
    assert launches == ["distribution_summaries"]
    home._search.setText("there is no such tool")
    assert home.current_route() is None and home._open.isHidden()
    home.request_current()
    assert launches == ["distribution_summaries"]
    home._search.clear()
    assert not any(item.isHidden() for item in home._items.values())


def test_explorer_home_is_persistent_and_never_activates_or_refreshes_analysis(qtbot):
    operations, launches, states = [], [], []
    def status(route, _page):
        states.append(route)
        return "Current", "current"
    shell = AnalysisExplorer(availability_provider=lambda _route: (True, "Ready"), status_provider=status)
    qtbot.addWidget(shell)
    shell.launchRequested.connect(launches.append)
    assert shell.page_ids() == () and shell.page_count() == 0
    assert shell.current_page_id() == PAGE_HOME
    assert shell._tabs.count() == 1
    assert states == []
    assert not shell._save_figure_btn.isEnabled() and not shell._save_bundle_btn.isEnabled()
    shell.show_home()
    shell.refresh_current()
    assert shell.page_count() == 0 and states == []
    page = QtWidgets.QWidget()
    shell.show_page("noise", page, refresh=lambda: operations.append("refresh"),
                     activate=lambda: operations.append("activate"))
    assert shell._tabs.tabText(shell._tabs.indexOf(page)) == "Noise spectrum"
    previous_states = states[:]
    shell.show_home()
    shell.refresh_current()
    assert operations == [] and set(states) == {"noise"} and states == previous_states
    assert shell._status_label.toolTip() == catalog.STATUS_HELP
    shell.home().set_current_route("figure_library")
    shell.home().request_current()
    assert launches == ["figure_library"] and operations == []
    shell.remove_page("noise")
    assert shell.page_count() == 0 and shell.current_page_id() == PAGE_HOME
    assert shell._tabs.count() == 1


def test_launcher_home_is_available_even_when_all_analysis_surfaces_are_locked(qtbot):
    launcher = AnalysisLauncher()
    qtbot.addWidget(launcher)
    launcher.show()
    assert all(not launcher.is_enabled(surface.sid) for surface in SURFACES)
    with qtbot.waitSignal(launcher.homeRequested):
        launcher._home_btn.click()
    launcher.resize(launcher.minimumSizeHint().width(), launcher.height())
    assert not launcher._home_btn.isHidden() and launcher._home_btn.isEnabled()


def test_home_fits_explorer_minimum_size_with_scrollable_details(qtbot, tmp_path):
    shell = AnalysisExplorer(availability_provider=lambda _route: (False, "Select a recording with a current channel."))
    qtbot.addWidget(shell)
    shell.resize(840, 560)
    shell.show_home()
    qtbot.wait(20)
    assert shell.size() == QtCore.QSize(840, 560)
    assert shell.home()._list.width() >= 265
    assert shell.grab().save(str(tmp_path / "analysis-home-minimum.png"))
