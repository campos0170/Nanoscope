"""Plot UX quick wins outside the plot pages (plotux G5).

Figure Library names and rows, Compose's whole-pixel spacing / colour swatch / font
default, refusals said inside the Analysis Explorer, greyed Home rows, and Ctrl+1–4 for
the plot pages' Section. Synthetic data only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.analysis import iv
from nanoscope.core.project import WorkspaceArtifact
from nanoscope.gui import analysis_catalog as catalog
from nanoscope.gui.analysis_explorer import AnalysisExplorer, PAGE_HOME
from nanoscope.gui.analysis_home import AnalysisHome
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_library import FigureLibrary, default_figure_name
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_workflow import PlotWorkflowBar, SECTIONS
from .test_figure_composer import entries  # noqa: F401 -- real frozen snapshots


# -- Figure Library: default names --------------------------------------------------------------
def test_default_name_uses_the_catalogue_label_not_the_page_id():
    assert default_figure_name("iv") == "I-V / conductance"
    assert default_figure_name("voltage_series") == "Voltage dependence"
    assert default_figure_name("iv") != "Iv"


def test_default_name_adds_the_shown_plot_type(qtbot):
    distributions = SimpleNamespace(figure_applied_config=lambda: {"plot_type": "Histogram"})
    assert default_figure_name("distributions", distributions) == "Distributions · Histogram"
    combo = QtWidgets.QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Welch PSD (averaged noise spectrum)", "psd")
    combo.addItem("FFT (single-shot magnitude)", "fft")
    noise = SimpleNamespace(_view_combo=combo, current_view=lambda: "fft")
    assert default_figure_name("noise", noise) == "Noise spectrum · FFT"
    # The APPLIED kind names the picture, not a pending choice in the controls.
    combo.setCurrentIndex(0)
    summaries = SimpleNamespace(_view_combo=combo, figure_applied_config=lambda: {"kind": "fft"})
    assert default_figure_name("distribution_summaries", summaries) == "Distribution summaries · FFT"
    # Nothing readable: the label alone, never an error.
    assert default_figure_name("distributions", SimpleNamespace()) == "Distributions"


@pytest.fixture
def iv_window(qtbot):
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    win._open_iv()
    panel = win._iv_window
    panel._points = [iv.IVPoint(v, 100 * v + 2, "pA", "measured", str(i), f"run {i}")
                     for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    panel._redraw()
    qtbot.wait(20)
    yield win
    win._dirty = False


def test_save_dialog_proposes_the_catalogue_name(iv_window, monkeypatch):
    proposed = []

    def look(dialog, *_args):
        proposed.append(dialog.findChild(QtWidgets.QLineEdit).text())
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec", look)
    assert iv_window._figure_capture_options("iv", ask_name=True) is None
    assert proposed == ["I-V / conductance"]   # was "Iv"


# -- Figure Library: file names in the list -----------------------------------------------------
def _figure(artifact_id, label):
    return WorkspaceArtifact(artifact_id=artifact_id, kind="nanoscope.figure", schema_version=1,
                             payload={"name": f"Figure {artifact_id}", "source_label": label})


def test_library_rows_show_file_names_and_keep_full_paths_in_the_tooltip(qtbot):
    figures = [
        _figure("a", "/data/pore runs/2026/pore1_80mV.abf · channel 0; "
                     "/data/pore runs/2026/pore2_100mV.abf · channel 1"),
        _figure("b", r"C:\Lab\runs\pore3.abf · channel 0"),
        _figure("c", "3 saved panels · component names and source details in provenance"),
    ]
    library = FigureLibrary(entries=lambda: figures, status=lambda _entry: ("Saved snapshot", "Frozen"),
                            append=figures.append)
    qtbot.addWidget(library)
    rows = [library._list.item(i) for i in range(library._list.count())]
    assert rows[0].text().splitlines()[-1] == "pore1_80mV.abf · channel 0; pore2_100mV.abf · channel 1"
    assert rows[1].text().splitlines()[-1] == "pore3.abf · channel 0"
    assert not any("/" in row.text() or "\\" in row.text() for row in rows[:2])
    assert rows[2].text().splitlines()[-1] == figures[2].payload["source_label"]
    assert "/data/pore runs/2026/pore1_80mV.abf · channel 0" in rows[0].toolTip()
    assert r"C:\Lab\runs\pore3.abf" in rows[1].toolTip()
    assert "Frozen" in rows[0].toolTip()
    # The detail pane still carries the whole label.
    library._list.setCurrentRow(0)
    assert "/data/pore runs/2026/pore2_100mV.abf" in library._summary.text()


# -- Compose -------------------------------------------------------------------------------------
def _layout(spacing, family):
    return {"columns": 2, "share_x": False, "share_y": False, "panel_labels": ["A", "B"],
            "spacing": spacing,
            "label_style": {"family": family, "size_px": 18, "bold": True, "color": "#111111"}}


def test_compose_spacing_is_whole_pixels(qtbot):
    figures = [_figure("a", "x"), _figure("b", "y")]
    dialog = FigureComposerDialog(figures)
    qtbot.addWidget(dialog)
    assert dialog._spacing.decimals() == 0
    assert dialog._spacing.text() == "20 px"
    # A composition saved with a fractional gap keeps it when revised.
    revised = FigureComposerDialog(figures, initial_layout=_layout(20.5, "Arial"))
    qtbot.addWidget(revised)
    assert revised._spacing.value() == pytest.approx(20.5)


def test_compose_label_colour_is_a_swatch(qtbot, monkeypatch):
    dialog = FigureComposerDialog([_figure("a", "x"), _figure("b", "y")])
    qtbot.addWidget(dialog)
    sheet = dialog._label_color.styleSheet()
    assert "background-color: #111111" in sheet and "color: #ffffff" in sheet
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor",
                        staticmethod(lambda *_a, **_k: QtGui.QColor("#f0e442")))
    dialog._choose_label_color()
    assert dialog._label_color_value == "#f0e442"
    assert "background-color: #f0e442" in dialog._label_color.styleSheet()
    assert "color: #000000" in dialog._label_color.styleSheet()   # legible ink on yellow


def test_compose_label_font_defaults_to_arial(entries, qtbot):  # noqa: F811
    """The panel-label font starts at Arial (default) -- the figure default (P0 G2) -- and no
    longer at the application font, which on macOS is a private face no journal has."""
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    assert dialog._label_font.currentText() == "Arial (default)"
    assert dialog.label_family() == "Arial"
    dialog.update_preview()
    assert dialog.snapshot is not None
    assert dialog.snapshot.config()["label_style"]["family"] == "Arial"
    # A family picked by name is used as named, and the preview is invalidated.
    helvetica = dialog._label_font.findData("Helvetica")
    dialog._label_font.setCurrentIndex(helvetica)
    assert dialog.snapshot is None and dialog.label_family() == "Helvetica"


def test_compose_revision_keeps_its_saved_label_family(qtbot):
    figures = [_figure("a", "x"), _figure("b", "y")]
    for family in ("Arial", "No Such Family 4711"):
        dialog = FigureComposerDialog(figures, initial_layout=_layout(20, family))
        qtbot.addWidget(dialog)
        assert dialog.label_family() == family
    # A composition saved before the default moved to Arial carries the application's own
    # family; it is kept as saved, on a row of its own.
    old = QtWidgets.QApplication.font().family()
    same = FigureComposerDialog(figures, initial_layout=_layout(20, old))
    qtbot.addWidget(same)
    assert same.label_family() == old
    assert same._label_font.currentData() == old


# -- Explorer refusals and greyed Home rows -------------------------------------------------------
def test_home_greys_unavailable_tools_with_the_reason_first(qtbot):
    gates = {entry.route: (True, "Ready.") for entry in catalog.ANALYSIS_ENTRIES}
    gates["noise"] = (False, "Open a recording first.")
    home = AnalysisHome(availability_provider=lambda route: gates[route])
    qtbot.addWidget(home)
    dim = home._list.palette().color(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text)
    row = home._items["noise"]
    assert row.foreground().color() == dim
    assert row.toolTip().startswith("Needs input: Open a recording first.")
    assert home._items["iv"].data(QtCore.Qt.ItemDataRole.ForegroundRole) is None
    # Grey, not disabled: the row can still be chosen to read its requirements.
    assert row.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
    home.set_current_route("noise")
    assert home.current_route() == "noise"
    gates["noise"] = (True, "Ready.")
    home.refresh_availability()
    assert row.data(QtCore.Qt.ItemDataRole.ForegroundRole) is None
    assert row.toolTip().endswith("Available: Ready.")


def test_explorer_banner_shows_clears_and_follows_availability(qtbot):
    gates = {entry.route: (False, "Open a recording first.") for entry in catalog.ANALYSIS_ENTRIES}
    shell = AnalysisExplorer(availability_provider=lambda route: gates[route])
    qtbot.addWidget(shell)
    assert shell.refusal_text() == ""
    shell.show_refusal("noise", "Open a recording first.")
    assert shell.refusal_text() == "Noise spectrum cannot open yet. Open a recording first."
    assert shell._refusal.isVisible()
    assert shell.current_page_id() == PAGE_HOME and shell.home().current_route() == "noise"
    shell._refusal.findChild(QtWidgets.QToolButton).click()
    assert shell.refusal_text() == ""
    shell.show_refusal("noise", "Open a recording first.")
    shell.show_page("iv", QtWidgets.QWidget())
    assert shell.refusal_text() == ""          # a page opened: the refusal is old news
    shell.show_refusal("noise", "Open a recording first.")
    gates["noise"] = (True, "Ready.")
    shell.home().refresh_availability()
    assert shell.refusal_text() == ""          # stale once the tool can open


def test_refused_launch_is_said_in_the_explorer_not_the_status_bar(qtbot, monkeypatch):
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    opened = []
    monkeypatch.setattr(win, "_open_plots", lambda: opened.append("plots"))
    win._launch_analysis_route("distributions")
    explorer = win._analysis_explorer
    assert explorer is not None and explorer.isVisible()
    assert explorer.refusal_text().startswith("Distributions cannot open yet. Open a recording first")
    assert explorer._refusal.isVisible()
    assert win.statusBar().currentMessage() == ""
    assert opened == []
    home = explorer.home()
    assert home.current_route() == "distributions"
    row = home._items["distributions"]
    dim = home._list.palette().color(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text)
    assert row.foreground().color() == dim
    assert "Open a recording first" in row.toolTip()
    # The rows refresh on the same signal as the launcher: once the tool can open, the row
    # is no longer grey and the banner goes.
    monkeypatch.setattr(win, "_gate_populations", lambda: (True, ""))
    win._sync_analysis_surfaces()
    assert row.data(QtCore.Qt.ItemDataRole.ForegroundRole) is None
    assert explorer.refusal_text() == ""
    win._launch_analysis_route("distributions")
    assert opened == ["plots"]


# -- Ctrl+1–4 choose the Section ------------------------------------------------------------------
def _page_with_bar():
    page = QtWidgets.QMainWindow()
    bar = PlotWorkflowBar(actions=(("Analyze", lambda: pytest.fail("Section ran the analysis")),))
    sections = {name: [QtWidgets.QLabel(name)] for name in SECTIONS}
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)
    layout.addWidget(bar)
    for widgets in sections.values():
        layout.addWidget(widgets[0])
    edit = QtWidgets.QLineEdit()
    layout.addWidget(edit)
    page.setCentralWidget(central)
    bar.set_sections(sections)
    return page, bar, edit


def _ctrl(qtbot, widget, key):
    widget.setFocus()
    qtbot.wait(20)       # let the focus change land before the shortcut map is asked
    QtTest.QTest.keyClick(widget, key, QtCore.Qt.KeyboardModifier.ControlModifier)


def test_ctrl_digits_pick_the_section_of_the_page_in_front_only(qtbot):
    tabs = QtWidgets.QTabWidget()
    qtbot.addWidget(tabs)
    first, first_bar, first_edit = _page_with_bar()
    second, second_bar, second_edit = _page_with_bar()
    tabs.addTab(first, "First")
    tabs.addTab(second, "Second")
    tabs.show()
    tabs.activateWindow()
    qtbot.waitActive(tabs)
    _ctrl(qtbot, first_edit, QtCore.Qt.Key.Key_2)
    assert first_bar.sections.currentText() == "Analysis"
    _ctrl(qtbot, first_edit, QtCore.Qt.Key.Key_3)
    assert first_bar.sections.currentText() == "Appearance"
    assert second_bar.sections.currentText() == "Data"     # hidden tab: untouched
    tabs.setCurrentWidget(second)
    _ctrl(qtbot, second_edit, QtCore.Qt.Key.Key_4)
    assert second_bar.sections.currentText() == "Export"
    assert first_bar.sections.currentText() == "Appearance"
    _ctrl(qtbot, second_edit, QtCore.Qt.Key.Key_1)
    assert second_bar.sections.currentText() == "Data"
    assert "–" in second_bar.sections.toolTip()


def test_ctrl_2_on_a_real_page_shows_analysis_without_computing(iv_window, qtbot, monkeypatch):
    page = iv_window._iv_window
    explorer = iv_window._analysis_explorer
    for name in ("refresh", "_refresh_from_user", "_redraw"):
        if hasattr(page, name):
            monkeypatch.setattr(page, name, lambda *_a, **_k: pytest.fail("Ctrl+2 computed"))
    explorer.show()
    explorer.activateWindow()
    qtbot.waitActive(explorer)
    _ctrl(qtbot, page._workflow.sections, QtCore.Qt.Key.Key_2)
    assert page._workflow.sections.currentText() == "Analysis"


# -- integration fixes (plotux build 4, 2026-11-01) ------------------------------------------
def test_a_real_double_click_on_an_unavailable_tool_shows_the_banner(qtbot):
    """Home's own gate refuses before the owner's route ever runs, so the banner must be
    raised from Home's refusal: double-click, Enter and Explain requirements alike."""
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    win._open_analysis_home()
    explorer = win._analysis_explorer
    qtbot.waitExposed(explorer)
    home = explorer.home()
    row = home._items["distributions"]
    rect = home._list.visualItemRect(row)
    left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier
    # QTest's double-click is the second click's event only: the first click comes first.
    QtTest.QTest.mouseClick(home._list.viewport(), left, none, rect.center())
    QtTest.QTest.mouseDClick(home._list.viewport(), left, none, rect.center())
    qtbot.waitUntil(lambda: explorer.refusal_text() != "", timeout=2000)
    assert explorer.refusal_text().startswith("Distributions cannot open yet.")
    assert win.statusBar().currentMessage() == ""
    explorer.clear_refusal()
    QtTest.QTest.mouseClick(home._open, left)            # Explain requirements
    assert explorer.refusal_text().startswith("Distributions cannot open yet.")
    explorer.clear_refusal()
    QtTest.QTest.keyClick(home._list, QtCore.Qt.Key.Key_Return)
    assert explorer.refusal_text().startswith("Distributions cannot open yet.")


def test_export_suggests_a_tidy_file_name_for_a_default_figure_name(iv_window, monkeypatch):
    import nanoscope.gui.figure_library as module
    from nanoscope.gui.analysis_explorer import PAGE_IV

    win = iv_window
    win._save_analysis_figure(PAGE_IV, "I-V / conductance")
    asked = []
    monkeypatch.setattr(module, "ask_export_path",
                        lambda *_a, **kw: asked.append(kw.get("suggestion")))
    win._figure_library._list.setCurrentRow(0)
    assert win._figure_library.export_current(bundle=True) is None
    assert asked == ["I-V - conductance.zip"]
