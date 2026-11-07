"""Phase 0 final review R4: opening a project does not carry the last project's Subpopulations
words and ranges into the next project's first report.

``MainWindow._reset_analytical_workspace`` (File > Open, Open Recent, the recovery offer) destroys
every analytical page it can rebuild and keeps the Subpopulations window, which it blanks with
``set_report(None)`` -- and that ends the appearance HISTORY only. A project saved with no
subpopulation report therefore drew its first report under the title and the pinned embedding
range typed on the previous project's scatter, and a project saved while that empty page was on
screen stored them in its own file. The reset now hands the window the appearance a restore with
none saved hands it (``{}``), as Distributions' ``reset_state`` already does for its own.

Real routes: the title is typed with **Add title…** and the range pinned in **Axis properties…**
on the scatter, the project is opened with ``_open_project``, the report arrives the way a run's
does (``_open_subpop_window``). Synthetic data only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.core.analysis_workspace import SUBPOPULATION_ARTIFACT_ID
from nanoscope.core.project import ProjectState, save_project
from nanoscope.gui import figure_text

from .test_p0_reset_on_quantity_change import subpop_report  # noqa: F401 -- a fixture
from .test_plot_window_routes import _add_title, _main_window, _pin_through_axis_properties


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _first_report(window, report):
    """What ``_on_subpop_finished`` does with a finished run."""
    window._open_subpop_window(report, source_result=None, source_recipe=None,
                               accepted_event_ids=None)
    QtWidgets.QApplication.processEvents()


def _drawn(page):
    """The scatter's title and X range as drawn."""
    plot = page.embedding_view()._plot.getPlotItem()
    title = plot.titleLabel.text if plot.titleLabel.isVisible() else ""
    lo, hi = plot.getViewBox().viewRange()[0]
    return (title or "").strip(), (round(lo, 3), round(hi, 3))


def _typed_and_pinned(window, report):
    _first_report(window, report)
    page = window._subpop_window
    plot = page.embedding_view()._plot.getPlotItem()
    window._analysis_explorer.resize(1280, 880)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    _add_title(plot, "Pore A in t-SNE")
    _pin_through_axis_properties(plot, "bottom", -3.0, 3.0)
    assert _drawn(page) == ("Pore A in t-SNE", (-3.0, 3.0))
    return page


def test_a_project_without_a_subpopulation_report_does_not_inherit_the_last_ones_words_and_ranges(
        qapp, qtbot, monkeypatch, tmp_path, subpop_report):
    other = tmp_path / "other.nanoscope"
    save_project(str(other), ProjectState())              # a project that holds no report
    window = _main_window(qtbot, monkeypatch)
    page = _typed_and_pinned(window, subpop_report)
    window._dirty = False                                 # the Discard answer

    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._open_project(str(other))

    # Managed figures are retired on project open, rather than keeping a singleton.
    assert window._subpop_window is None and not window._dirty
    assert window._ensure_figure_workspace().most_recent("subpopulations") is None
    _first_report(window, subpop_report)
    fresh = window._subpop_window
    assert fresh is not page
    assert fresh._appearance_widget.title.text() == ""
    assert not fresh._appearance_widget.undo_button.isEnabled()
    title, x_range = _drawn(fresh)
    assert title == "" and x_range != (-3.0, 3.0)
    # New figures receive the incoming project's default style, without old overrides.
    assert not fresh.appearance_state().get("title")
    assert "axes" not in fresh.appearance_state()
    window._dirty = False


def test_the_file_a_project_without_a_report_is_saved_to_holds_none_of_the_last_ones_words(
        qapp, qtbot, monkeypatch, tmp_path, subpop_report):
    """The second way the words travelled: the empty page was still on screen when the next
    project was saved, and the file got a subpopulation artifact holding the old appearance."""
    from nanoscope.core.project import load_project

    other = tmp_path / "other.nanoscope"
    save_project(str(other), ProjectState())
    window = _main_window(qtbot, monkeypatch)
    _typed_and_pinned(window, subpop_report)
    window._dirty = False
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._open_project(str(other))
    saved = tmp_path / "saved.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False

    state = load_project(str(saved))
    artifacts = [a for a in state.workspace.artifacts
                 if a.artifact_id == SUBPOPULATION_ARTIFACT_ID]
    # No old singleton artifact is needed when the incoming project has no figure.
    assert window._ensure_figure_workspace().most_recent("subpopulations") is None
    assert "Pore A in t-SNE" not in repr(state.workspace)
    for artifact in artifacts:
        assert not (artifact.payload or {}).get("appearance"), artifact.payload.get("appearance")


def test_a_project_that_does_hold_a_report_opens_with_its_own_words_and_not_the_last_ones(
        qapp, qtbot, monkeypatch, tmp_path, subpop_report):
    """The project's own words still win: the restore installs them after the blanking."""
    from nanoscope.gui.plot_appearance import PlotAppearance

    window = _main_window(qtbot, monkeypatch)
    _first_report(window, subpop_report)
    page = window._subpop_window
    page.set_appearance_state(PlotAppearance(title="Pore B in UMAP").to_dict())
    mine = tmp_path / "mine.nanoscope"
    assert window._write_project(str(mine))
    window._dirty = False

    other = _main_window(qtbot, monkeypatch)
    page = _typed_and_pinned(other, subpop_report)
    other._dirty = False
    with qtbot.waitSignal(other.projectRestored, timeout=20_000):
        other._open_project(str(mine))

    owner = other._ensure_figure_workspace()
    key = owner.most_recent("subpopulations")
    assert key is not None and owner.activate(key)
    restored = owner.page(key)
    assert restored is not page
    assert restored.appearance_state().get("title") == "Pore B in UMAP"
    assert "axes" not in restored.appearance_state()
    other._dirty = False
