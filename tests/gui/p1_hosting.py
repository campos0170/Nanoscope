"""Test support for the Phase 1 inspector and toolbar: host a page, then drive it for real.

Not a test module (no ``test_`` prefix). Every helper that performs a gesture performs the
user's: a mouse click or key press goes through the QWINDOW of the widget's top-level window
(``widget.window().windowHandle()``), a menu entry is clicked, never triggered. A page group
puts its own page-specific helpers in its own test file and asks the shell group for shared
ones.

``SyntheticPlotPage`` hosts the whole interface with no real page, so the shell and the
Format tab can be built and tested in isolation.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.figure_text import appearance_text_host, figure_text_for
from nanoscope.gui.fit_label import FitLabelItem
from nanoscope.gui.plot_appearance import (PlotAppearance, PlotAppearanceWidget,
                                           apply_plot_appearance)
from nanoscope.gui.plot_interaction import (install_plot_interaction, view_button,
                                            window_readout)
from nanoscope.gui.plot_workflow import (ExportEntry, FormatSpec, InspectorGroup, PageAdapter,
                                         PendingState, control_group, install_plot_workflow)

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier

__all__ = ["SyntheticPlotPage", "child_added_counter", "choose_export", "click", "count_widgets",
           "export_menu", "hosted", "key", "page_builders", "press_redo", "press_reset_figure",
           "press_undo", "press_update", "show_data", "show_format", "window_handle"]


# -- real gestures --------------------------------------------------------------------------------

def window_handle(widget):
    """The QWindow of ``widget``'s top-level window: where a real click or key is sent.

    A wrapper of an object Qt deleted underneath it can be handed back for a window created
    at its address (``test_plot_window_routes.window_handle`` says why); the stale wrapper is
    dropped and the window asked again."""
    top = widget.window()
    handle = top.windowHandle()
    for _ in range(3):
        if handle is None or isinstance(handle, QtGui.QWindow):
            break
        shiboken6.invalidate(handle)
        handle = top.windowHandle()
    return handle


def click(widget, button=L, pos=None, modifiers=NOMOD, double=False):
    """A real mouse click on ``widget`` (at its centre, or ``pos`` in its coordinates)."""
    handle = window_handle(widget)
    spot = pos if pos is not None else widget.rect().center()
    where = widget.mapTo(widget.window(), QtCore.QPoint(int(spot.x()), int(spot.y())))
    QtTest.QTest.mouseMove(handle, where)
    if double:
        QtTest.QTest.mouseDClick(handle, button, modifiers, where)
    else:
        QtTest.QTest.mouseClick(handle, button, modifiers, where)
    QtWidgets.QApplication.processEvents()


def key(widget, keycode, modifiers=NOMOD):
    """A real key press through the window's QWindow, after the focus change has landed."""
    widget.window().activateWindow()
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.keyClick(window_handle(widget), keycode, modifiers)
    QtWidgets.QApplication.processEvents()


# -- hosting ----------------------------------------------------------------------------------------

def hosted(qtbot, page, key, *, size=(1280, 820)) -> AnalysisExplorer:
    """A real Explorer with ``page`` added and in front: adopted, shown, active."""
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(*size)
    explorer.show_page(key, page)
    explorer.show()
    explorer.activateWindow()
    qtbot.waitExposed(explorer)
    QtWidgets.QApplication.processEvents()
    return explorer


def _explorer_of(target) -> AnalysisExplorer:
    if isinstance(target, AnalysisExplorer):
        return target
    top = target.window()
    if not isinstance(top, AnalysisExplorer):
        raise TypeError(f"{target!r} is not in an Analysis Explorer")
    return top


def show_data(target, group: str | None = None):
    """Data & analysis tab, with ``group`` revealed (the first group when None)."""
    explorer = _explorer_of(target)
    explorer.inspector().set_tab(0)
    if group is not None:
        assert explorer.reveal(group), f"no group {group!r}"
    QtWidgets.QApplication.processEvents()


def show_format(target, element: str = "figure"):
    """Format tab, with ``element`` shown (``"@format:<element>"`` through the shell)."""
    explorer = _explorer_of(target)
    explorer.reveal(f"@format:{element}" if element else "@format")
    QtWidgets.QApplication.processEvents()


def _toolbar_button(explorer, action):
    return explorer.toolbar().widgetForAction(action)


def press_update(explorer):
    """A real click on the main half of Update plot (the left of its drop-down button)."""
    button = explorer.toolbar().update_button
    click(button, pos=QtCore.QPoint(6, button.height() // 2))


def press_undo(explorer):
    click(_toolbar_button(explorer, explorer.toolbar().undo_action))


def press_redo(explorer):
    click(_toolbar_button(explorer, explorer.toolbar().redo_action))


def press_reset_figure(explorer):
    click(explorer.toolbar().reset_figure_button)


def export_menu(explorer) -> QtWidgets.QMenu:
    """Open Export ▾ and return the menu that is showing.

    ``showMenu()`` and not a click on the button, on purpose: a press on an ``InstantPopup``
    button runs the menu in its own nested event loop and does not return until the menu has
    closed (measured: a press hangs a test that has no timer to close it), and a test could
    not read the entries or click one. Everything after the menu is open is real: the entry
    is picked by :func:`choose_export` with a click on the menu."""
    toolbar = explorer.toolbar()
    toolbar.export_button.showMenu()
    QtWidgets.QApplication.processEvents()
    return toolbar.export_button.menu()


def choose_export(explorer, text: str):
    """Click the Export ▾ entry whose text is ``text`` (a real click on the open menu)."""
    menu = export_menu(explorer)
    action = next(a for a in menu.actions() if a.text() == text)
    assert action.isEnabled(), f"{text!r} is disabled"
    QtTest.QTest.mouseClick(menu, L, NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()


# -- counting ---------------------------------------------------------------------------------------

def count_widgets(root: QtCore.QObject) -> int:
    """Every ``QWidget`` at or under ``root``."""
    return len(root.findChildren(QtWidgets.QWidget)) + (1 if isinstance(root, QtWidgets.QWidget) else 0)


class _ChildCounter(QtCore.QObject):
    def __init__(self, widget: QtWidgets.QWidget):
        super().__init__(widget)
        self.count = 0
        self._widget = widget

    def eventFilter(self, _watched, event):  # noqa: N802 - Qt override
        if event.type() == QtCore.QEvent.Type.ChildAdded:
            self.count += 1
        return False


def child_added_counter(widget: QtWidgets.QWidget) -> _ChildCounter:
    """Counts ``ChildAdded`` events on ``widget`` from now on (``.count``): a tab switch that
    moves nothing leaves it at zero. TEST-ONLY: it is a filter on a pane, never on a view."""
    counter = _ChildCounter(widget)
    widget.installEventFilter(counter)
    return counter


# -- the eight pages on an empty Session -----------------------------------------------------------

def _frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": np.abs(rng.normal(3e-4, 1e-4, n)), "amplitude": np.abs(rng.normal(30, 5, n)),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)), "snr": np.abs(rng.normal(9, 2, n)),
        "rms": np.abs(rng.normal(1.4, 0.3, n)), "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool)})


def page_builders() -> dict[str, "callable"]:
    """``{page id: build()}`` for the eight pages, each on a fresh empty ``Session`` with
    synthetic data only. Imports are local so a tree without a page still imports this."""
    from nanoscope.core.session import Session

    def distributions():
        from nanoscope.gui.plots_panel import PlotsWindow
        page = PlotsWindow(Session())
        page.set_data({"run A": _frame(seed=1), "run B": _frame(seed=2)})
        return page

    def summaries():
        from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
        return DistributionSummaryWindow(Session())

    def comparison():
        from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
        return ComparisonFiguresWindow(Session())

    def events():
        from nanoscope.gui.event_plot_panel import EventPlotWindow
        return EventPlotWindow(Session())

    def noise():
        from nanoscope.gui.psd_panel import PSDWindow
        return PSDWindow(Session(), defer_initial_refresh=True)

    def iv():
        from nanoscope.gui.iv_panel import IVWindow
        return IVWindow(Session(), defer_initial_refresh=True)

    def vdep():
        from nanoscope.gui.voltage_dep_panel import VDepWindow
        return VDepWindow(Session(), defer_initial_refresh=True)

    def subpop():
        from nanoscope.gui.subpop_window import SubpopReportWindow
        return SubpopReportWindow()

    return {"distributions": distributions, "distribution_summaries": summaries,
            "comparison_figures": comparison, "event_waveforms": events, "noise": noise,
            "iv": iv, "voltage_series": vdep, "subpopulations": subpop}


# -- a page that is nothing but the interface -----------------------------------------------------

class SyntheticPlotPage(QtWidgets.QMainWindow):
    """One ``PlotItem`` with the whole shared interface and no analysis.

    Modelled on ``IVWindow._build_ui``: a ``PlotAppearanceWidget`` with its own history, an
    ``AppearanceTextHost``, ``install_plot_interaction(text=...)``, a ``FitLabelItem``, one
    estimator group, an Update plot verb that counts its calls, an Export list, and an
    ``inspector_adapter()``. ``updates`` counts the verb; ``pending_count`` is what its
    ``pending()`` reports; ``busy`` greys the verb."""

    workspaceChanged = QtCore.Signal()
    pendingChanged = QtCore.Signal()

    def __init__(self, key: str = "synthetic") -> None:
        super().__init__()
        self._key = key
        self.updates = 0
        self.exported: list[str] = []
        self.pending_count = 0
        self.pending_reasons: tuple[str, ...] = ()
        self.busy = False
        self._appearance = PlotAppearance()
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        self._appearance_widget = PlotAppearanceWidget(self._appearance, shortcut_parent=self)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._appearance_section = QtWidgets.QScrollArea()
        self._appearance_section.setWidget(self._appearance_widget)
        self._appearance_section.setWidgetResizable(True)
        root.addWidget(self._appearance_section)
        self._estimator = control_group("Estimator", "Synthetic page: nothing to estimate.")
        root.addWidget(self._estimator)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumSize(420, 300)
        self._plot = self._plot_widget.getPlotItem()
        self._plot.plot([0, 1, 2, 3], [1.0, 2.0, 0.5, 3.0], name="series")
        root.addWidget(self._plot_widget, 1)
        self.setCentralWidget(central)
        self._exports = (("Export current figure…", lambda: self.exported.append("figure")),
                         ("Export data (CSV)…", lambda: self.exported.append("data")))
        self._workflow = install_plot_workflow(
            self, sections={"Data": (self._estimator,), "Analysis": (),
                            "Appearance": (self._appearance_section,), "Export": ()},
            actions=(("Update", self._update),), exports=self._exports)
        self._workflow.adopt_reset_figure(self._appearance_widget)
        self._readout = window_readout(self)
        self._text_host = appearance_text_host(self._plot, self._appearance_widget)
        self._plot_nav = install_plot_interaction(
            self._plot, readout=self._readout, exports=self._exports, text=self._text_host)
        self._workflow.add_view_control(view_button(lambda: self._plot_nav, (self._plot_nav,)))
        self._fit_label = FitLabelItem(self._plot.getViewBox())
        self._text_host.register_fit_label(self._fit_label)
        self._appearance_widget.set_fit_label_available(True, in_legend=False)

    def _update(self) -> None:
        self.updates += 1

    def _appearance_from_user(self, state: PlotAppearance) -> None:
        self._appearance = state
        apply_plot_appearance(self._plot, state, plot_widget=self._plot_widget)
        self.workspaceChanged.emit()

    def set_pending(self, count: int, *reasons: str, busy: bool = False) -> None:
        self.pending_count, self.pending_reasons, self.busy = count, tuple(reasons), busy
        self.pendingChanged.emit()

    def appearance_state(self) -> dict:
        return self._appearance.to_dict()

    def inspector_adapter(self) -> PageAdapter:
        if getattr(self, "_inspector_adapter", None) is None:
            self._inspector_adapter = PageAdapter.from_widget(
                self, self._key, self._appearance_widget,
                groups=[InspectorGroup("estimator", "Estimator", [self._estimator])],
                format=FormatSpec(self._appearance_widget,
                                  controllers=lambda: [c for c in (figure_text_for(self._plot),) if c]),
                update=self._update, update_tooltip="Count one update.",
                pending=lambda: PendingState(self.pending_count, self.pending_reasons, self.busy),
                pending_changed=self.pendingChanged,
                exports=lambda: [ExportEntry(t, fn, kind=k) for (t, fn), k
                                 in zip(self._exports, ("figure", "data"))],
                navigation=lambda: self._plot_nav, navigations=lambda: (self._plot_nav,),
                status_bar=self.statusBar, chrome=[self._workflow],
                on_adopt=lambda: self._appearance_section.hide())
        return self._inspector_adapter
