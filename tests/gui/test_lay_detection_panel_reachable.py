"""The Analysis controls column fits the shipped Inspector — no sideways scrolling.

THE DISPLAY IS THE SPEC. The owner's laptop is 1512x900 and the app ships the Inspector
dock at 460 px, which gives the Analysis tab's controls a **420 px viewport**. Before this
round the column's content was **836 px** wide, so a horizontal scrollbar appeared inside
the dock and everything to the right of x=420 was off-screen. That is not a cosmetic
complaint: seven controls with no menu item and no keyboard shortcut were unreachable, and
a user who never thinks to drag a scrollbar sideways inside a dock could not use them at
all —

* ``Preprocess ▸ Protocol ▸ Save as… / Rename / Delete`` sat at x=561/647/733,
* ``ML detector ▸ preset ▸ Save as… / Delete`` at x=647/733 — pushed off the edge by the
  combo *directly above them*,
* ``Detection ▸ Null control… / Recovery map…`` at x=421/626, the first missing the
  viewport by a single pixel, while ``StatsPanel`` prints "false-positive rate: not
  measured (run Null control)" and points at it.

Three causes, all measured rather than guessed:

1. **A combo sized to its longest item.** ``ml_preset``'s ``SizeAdjustPolicy`` defaulted to
   ``AdjustToContentsOnFirstShow``, so a 115-character preset label made it 706 px wide, and
   that number propagated up through the form, the group box, the page, the tab widget and
   the panel to set the whole column's 836 px. Two more combos did the same in smaller ways.
2. **Un-shrinkable rows of push buttons.** ``QPushButton`` has no shrink flag, so
   ``qSmartMinSize`` propagates the SUM of a horizontal row; four detection buttons floored
   their page at 431 px.
3. **Un-shrinkable check box labels.** Same mechanism: one 387 px label
   ('enable (peak-preserving polynomial smooth — KEEPS the fit)') floored the Preprocess
   page, and a 305 px one floored ``StatsPanel``.

These tests pin the RESULT — every one of those controls entirely inside the viewport, and
no horizontal scrollbar for it to hide behind — plus the three mechanisms, so that widening
any one of them again fails here rather than in the owner's hands. They also pin that the
floor does not MOVE with state: a control that shifts when a preset is picked or a box is
ticked is a control that dodges the pointer aiming at it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import _INSPECTOR_DEFAULT_WIDTH, MainWindow
from nanoscope.gui.panels import DetectionPanel, StatsPanel, _ElidingCheckBox

#: The owner's screen. Every number in this module is measured at exactly this size.
_SCREEN = (1512, 900)

#: Frame + vertical scrollbar: what the dock's width loses before the controls see it.
#: Measured (dock 460 -> viewport 420, dock 600 -> 560, dock 800 -> 760).
_DOCK_CHROME = 40

#: The controls viewport at the Inspector's NARROWEST shipped width. The dock may open
#: wider — ``_apply_default_dock_widths`` sizes it to its content up to a cap — but
#: ``_INSPECTOR_DEFAULT_WIDTH`` is the floor it is never opened below, and it is also the
#: width the owner's 1512x900 screen was measured at. Every position below is taken with
#: the dock dragged to exactly that, i.e. at the worst case the app ships.
_VIEWPORT = _INSPECTOR_DEFAULT_WIDTH - _DOCK_CHROME

#: What the column used to demand, kept as a landmark for the failure message.
_THE_OLD_FLOOR = 836


# ================================================================================================
# fixtures
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so "first launch" really means first launch.

    ``MainWindow._restore_settings`` reads a saved ``windowState``; a developer who has
    dragged the Inspector wider in the real app would otherwise measure THAT layout and
    never see the defect this module exists to prevent.
    """
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("LayDetectionPanelReachable")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


def _settle(win: MainWindow, rounds: int = 3) -> None:
    """Force the layout to finish.

    The offscreen platform does not propagate size hints on its own ("This plugin does not
    support propagateSizeHints()"), so a plain ``processEvents`` can leave a freshly shown
    page holding the geometry it had before. Activating every layout makes the measurement
    the same one the real window makes.
    """
    for _ in range(rounds):
        QtWidgets.QApplication.processEvents()
        for w in win.findChildren(QtWidgets.QWidget):
            lay = w.layout()
            if lay is not None:
                lay.activate()
        if win.layout() is not None:
            win.layout().activate()
        QtWidgets.QApplication.processEvents()


class _Column:
    """The Analysis tab's controls scroll area, and the widgets that live in it."""

    def __init__(self, win: MainWindow) -> None:
        self.win = win
        tabs = win._inspector.widget()
        for i in range(tabs.count()):
            if tabs.tabText(i).startswith("Analysis"):
                tabs.setCurrentIndex(i)
        _settle(win)
        self.analysis: AnalysisPanel = win.findChild(AnalysisPanel)
        self.detection: DetectionPanel = self.analysis._detection
        self.stats: StatsPanel = self.analysis.findChild(StatsPanel)
        self.area = self._owning_scroll(self.detection)
        self.inner = self.area.widget()
        self.mode_tabs = self.detection.findChild(QtWidgets.QTabWidget)

    @staticmethod
    def _owning_scroll(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        p = w.parentWidget()
        while p is not None:
            if isinstance(p, QtWidgets.QScrollArea):
                return p
            p = p.parentWidget()
        raise AssertionError("the controls are not inside a scroll area at all")

    @property
    def viewport_width(self) -> int:
        return self.area.viewport().width()

    def raise_page(self, title: str) -> None:
        for i in range(self.mode_tabs.count()):
            if self.mode_tabs.tabText(i) == title:
                self.mode_tabs.setCurrentIndex(i)
                _settle(self.win)
                return
        raise AssertionError(f"no {title!r} page: {[self.mode_tabs.tabText(i) for i in range(self.mode_tabs.count())]}")

    def span(self, w: QtWidgets.QWidget) -> tuple[int, int]:
        """``(left, right)`` of ``w`` in the scroll area's CONTENT coordinates."""
        x = w.mapTo(self.inner, QtCore.QPoint(0, 0)).x()
        return x, x + w.width()


@pytest.fixture
def column(isolated_settings, qapp, qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*_SCREEN)
    win.show()
    _settle(win)
    # Measure at the NARROWEST width the Inspector ships at, not at whatever
    # _apply_default_dock_widths chose for today's content: that is the case the controls
    # have to survive, and it keeps this module honest if the default is ever widened to
    # paper over a floor instead of removing it.
    win.resizeDocks([win._inspector], [_INSPECTOR_DEFAULT_WIDTH], QtCore.Qt.Horizontal)
    _settle(win)
    col = _Column(win)
    # The premise of every number below. If the dock's chrome changes, this is the line
    # that says so, instead of a dozen confusing off-by-N failures.
    assert col.viewport_width == _VIEWPORT, (
        f"a {_INSPECTOR_DEFAULT_WIDTH} px Inspector no longer gives the controls a "
        f"{_VIEWPORT} px viewport (measured {col.viewport_width}); re-measure _DOCK_CHROME"
    )
    return col


# ================================================================================================
# the column fits
# ================================================================================================

def test_no_horizontal_scrollbar_in_the_analysis_controls(column):
    """The failure this whole module is about, stated in one number.

    ``horizontalScrollBar().maximum() > 0`` means content is cut off horizontally. It used
    to be 416 — of an 836 px column shown through 420 px.
    """
    hbar = column.area.horizontalScrollBar()
    assert hbar.maximum() == 0, (
        f"the Analysis controls scroll sideways again: content "
        f"{column.inner.minimumSizeHint().width()} px through a {column.viewport_width} px "
        f"viewport (was {_THE_OLD_FLOOR}/{_VIEWPORT} before this round)"
    )
    assert not hbar.isVisible()


def test_the_controls_column_demands_no_more_than_the_viewport(column):
    """The structural form of the same thing, independent of the current scroll position."""
    demanded = column.inner.minimumSizeHint().width()
    assert demanded <= column.viewport_width, (
        f"the controls column demands {demanded} px inside a {column.viewport_width} px "
        f"viewport; the widest child is what to look at"
    )


@pytest.mark.parametrize("attr", ["_detection", "_stats"])
def test_no_panel_in_the_column_floors_it_above_the_viewport(column, attr):
    """Each panel stacked in the column, measured on its own.

    ``DetectionPanel`` was 836 and ``StatsPanel`` 463; either one alone re-creates the
    scrollbar, so they are pinned separately rather than only through their sum.
    """
    panel = getattr(column.analysis, attr)
    assert panel.minimumSizeHint().width() <= _VIEWPORT, (
        f"{type(panel).__name__} floors the Analysis column at "
        f"{panel.minimumSizeHint().width()} px, past the {_VIEWPORT} px viewport"
    )


# ================================================================================================
# the seven controls, each one whole and inside
# ================================================================================================

def _assert_inside(column, name: str, w: QtWidgets.QWidget) -> None:
    left, right = column.span(w)
    assert 0 <= left and right <= column.viewport_width, (
        f"{name} is not fully on screen: x={left}..{right} against a "
        f"{column.viewport_width} px viewport"
    )
    assert w.isVisibleTo(column.inner), f"{name} is not visible on its page"


def test_protocol_save_rename_delete_are_reachable(column):
    """The three verbs of the named-protocol store, at x=561/647/733 before this round."""
    column.raise_page("Preprocess")
    det = column.detection
    for name in ("protocol_save_btn", "protocol_rename_btn", "protocol_delete_btn"):
        _assert_inside(column, name, getattr(det, name))


def test_the_protocol_verbs_sit_below_the_combo_not_beside_it(column):
    """The mechanism, not just the outcome.

    Left-aligned on their own row, the buttons' x is independent of the panel's width — the
    reason they cannot be pushed off again by something on another page. If someone puts
    them back on the combo's row this fails even at a width where they happen to fit.
    """
    column.raise_page("Preprocess")
    det = column.detection
    combo_y = det.protocol_combo.mapTo(column.inner, QtCore.QPoint(0, 0)).y()
    save_y = det.protocol_save_btn.mapTo(column.inner, QtCore.QPoint(0, 0)).y()
    assert save_y > combo_y, "the protocol verbs are back on the combo's row"
    assert column.span(det.protocol_save_btn)[0] < _VIEWPORT // 2, (
        "the protocol verbs are right-aligned again; their position now depends on a width "
        "set three pages away"
    )


def test_ml_preset_save_and_delete_are_reachable(column):
    """Pushed off-screen by the widget directly above them — the purest case."""
    column.raise_page("ML detector")
    det = column.detection
    for name in ("ml_preset_save_btn", "ml_preset_delete_btn"):
        _assert_inside(column, name, getattr(det, name))


def test_the_ml_preset_verbs_are_left_aligned(column):
    """A leading ``addStretch(1)`` is what put them at the right edge of an 814 px form."""
    column.raise_page("ML detector")
    left, _ = column.span(column.detection.ml_preset_save_btn)
    assert left < _VIEWPORT // 2, (
        f"ml_preset_save_btn starts at x={left}: the row is right-aligned again"
    )


def test_all_four_detection_buttons_are_reachable(column):
    """``Null control…`` and ``Recovery map…`` — the FDR control and the efficiency map."""
    column.raise_page("Detection")
    det = column.detection
    for name in ("detect_button", "sweep_button", "null_button", "recovery_button"):
        _assert_inside(column, name, getattr(det, name))


def test_the_detection_buttons_are_two_rows_of_two(column):
    """Fixed at two columns on purpose.

    Two rows is what drops the page's floor from 431 to ~260. It must be a FIXED grid and
    not a reflowing one: a row that rearranges as the dock is dragged moves a button out
    from under the pointer aiming at it.
    """
    column.raise_page("Detection")
    det = column.detection
    y = {n: getattr(det, n).mapTo(column.inner, QtCore.QPoint(0, 0)).y()
         for n in ("detect_button", "sweep_button", "null_button", "recovery_button")}
    assert y["detect_button"] == y["sweep_button"], "row 1 broke apart"
    assert y["null_button"] == y["recovery_button"], "row 2 broke apart"
    assert y["null_button"] > y["detect_button"], "the four buttons are one row again"


def test_the_frequency_response_plot_fits_its_column(column):
    """The live |H(f)| curve was drawn 790 px wide inside a 420 px viewport."""
    column.raise_page("Preprocess")
    _assert_inside(column, "response_plot", column.detection.response_plot)


def test_the_stats_fit_controls_are_reachable(column):
    """``use measured efficiency`` is 305 px of label; it floored StatsPanel at 463."""
    for name in ("use_efficiency", "use_filter_limited", "fit_button"):
        _assert_inside(column, name, getattr(column.stats, name))


# ================================================================================================
# the three mechanisms
# ================================================================================================

def test_the_preset_combo_does_not_take_its_width_from_its_longest_item(column):
    """``AdjustToContentsOnFirstShow`` + a 115-character label = a 706 px width floor.

    The cap has to be set BEFORE the items go in, because ``QComboBox`` caches
    ``minimumSizeHint``; a test that only checked the policy would pass on a combo that
    still measures 706. This checks the measurement.
    """
    combo = column.detection.ml_preset
    longest = max((len(combo.itemText(i)) for i in range(combo.count())), default=0)
    assert longest > 80, "the preset labels got short; this test no longer proves anything"
    assert combo.minimumSizeHint().width() <= 260, (
        f"ml_preset demands {combo.minimumSizeHint().width()} px again — it is sized to its "
        f"longest item ({longest} characters). Set the SizeAdjustPolicy before addItem()."
    )


@pytest.mark.parametrize("attr", ["ml_norm_method", "normalize_mode", "protocol_combo",
                                  "detector_combo", "detrend_method"])
def test_no_combo_in_the_panel_is_wider_than_its_column(column, attr):
    """Every combo, not just the one that was caught."""
    combo = getattr(column.detection, attr)
    assert combo.minimumSizeHint().width() <= 260, (
        f"{attr} demands {combo.minimumSizeHint().width()} px"
    )


def test_long_check_box_labels_elide_instead_of_flooring_the_panel(column):
    """The label text is unchanged; only its PAINT gives way.

    A ``QCheckBox`` whose ``minimumSizeHint`` carries a full sentence is a width floor for
    everything above it. These keep their text (so every existing assertion on ``text()``
    still holds) and their meaning (the full sentence is in the tooltip); what they give up
    is the right to dictate the panel's width.
    """
    savgol = column.detection.savgol_enabled
    assert isinstance(savgol, _ElidingCheckBox)
    assert savgol.text() == "enable (peak-preserving polynomial smooth — KEEPS the fit)"
    assert savgol.toolTip(), "an elided label with no tooltip loses the sentence"
    assert savgol.minimumSizeHint().width() <= 200, (
        f"savgol_enabled demands {savgol.minimumSizeHint().width()} px; it used to be 387 "
        f"and that was the Preprocess page's floor"
    )
    assert column.stats.use_efficiency.minimumSizeHint().width() <= 200


# ================================================================================================
# nothing moves under the pointer
# ================================================================================================

def test_the_column_width_does_not_move_with_state(column):
    """A layout that reflows when the user acts is the same defect wearing a disguise.

    Every one of these was a real state change on this panel: picking a preset, changing the
    detector, walking the filter stack, ticking a stage on, gating the ML run. If any of them
    moved the floor, a control could shift out from under a pointer already aiming at it.
    """
    det = column.detection
    baseline = (det.minimumSizeHint().width(), column.inner.minimumSizeHint().width())

    def _floor(tag: str) -> None:
        _settle(column.win)
        now = (det.minimumSizeHint().width(), column.inner.minimumSizeHint().width())
        assert now == baseline, f"{tag} moved the column floor: {baseline} -> {now}"
        assert column.area.horizontalScrollBar().maximum() == 0, f"{tag} brought the scrollbar back"

    det.ml_preset.setCurrentIndex(det.ml_preset.count() - 1)
    _floor("picking the last ML preset")
    det.detector_combo.setCurrentIndex(det.detector_combo.count() - 1)
    _floor("changing the detector")
    for i in range(det.filter_stack.count()):
        det.filter_stack.setCurrentIndex(i)
        _floor(f"filter stack page {i}")
    det.savgol_enabled.setChecked(True)
    _floor("enabling Savitzky-Golay")
    det.set_ml_runnable(False, "no model loaded")
    _floor("gating the ML run off")
    det.set_ml_runnable(True, "")
    _floor("gating the ML run back on")
