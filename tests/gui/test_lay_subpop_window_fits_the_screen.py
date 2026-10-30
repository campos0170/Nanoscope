"""Reachability round: the subpopulation report window opens INSIDE the owner's screen.

THE DEFECT, measured on a fresh QSettings profile with the shipped default
(``subpop_window._DEFAULT_SIZE = (1600, 980)``, applied unconditionally) against the
owner's 1512x900 display:

    control                       window coords          verdict
    ---------------------------   --------------------   -----------------------------
    Copy report                   x 8-96,      y 947-972  72 px BELOW the screen bottom
    Export…                       x 102-182,   y 947-972  72 px below
    Event gallery…                x 188-291,   y 947-972  72 px below
    Colour the trace              x 297-410,   y 947-972  72 px below
    Close                         x 1512-1592, y 947-972  72 px below, 80 px past right
    the standing note (live_note) x 8-1592,    y 892-941  41 px below, 80 px past right

Zero pixels of any of the five buttons were on the screen, and that row is everything
this window can DO: copy the report, export it, open the gallery on it, colour the trace
by it, close it. Before the macOS menu bar and the window's own title bar are subtracted.
The window is resizable, so the report was readable and nothing about it was actionable —
the exact shape the audit's skeptics overturned five "this exists" verdicts on.

THE FIX these tests pin: :func:`~nanoscope.gui.subpop_window.fitted_window_size` shrinks
the preferred size to the screen the window will open on (1496x856 at 1512x900), floored
at the figures' own minimum, and the window is placed inside the available rectangle once,
at construction. The pinning is geometric and would fail again if someone widened or
heightened this window past what the display holds.

WHY THE SCREEN IS INJECTED. The offscreen platform these tests run under reports an
800x800 screen, so the assertions are made against the owner's real 1512x900 through the
module's one seam, ``available_screen_rect`` — patch that, not Qt.

NOT IN SCOPE, and measured so it is not mistaken for a regression: the 4.6 s the audit
timed between the menu item and this window is the mixture search on the worker thread
(``run_subpopulations`` at the shipped ``SubpopConfig``: 23.4 s on 220 synthetic events;
the audit's own 6.58 s on 83). This window's whole share of it is 0.18 s — construct
0.126, ``set_report`` 0.012, ``show`` 0.040 — so nothing here is a build-time defect.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet
from nanoscope.gui import subpop_window as SW

FS = 50_000.0
N_SAMPLES = 200_000
N_EVENTS = 140


def _synthetic_events():
    """Two dwell/depth populations at 50 kHz, float32 -- no file is read."""
    rng = np.random.default_rng(0)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans, pos = [], 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    return (np.ascontiguousarray(x, dtype=np.float32),
            EventSet(pd.DataFrame(rows)[REQUIRED],
                     {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1}))


@pytest.fixture(scope="module")
def report():
    """A real report through the real pipeline, on synthetic events."""
    arr, events = _synthetic_events()
    cfg = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    return SR.run_subpopulations(arr, arr, FS, events, recording="r",
                                 duration_s=N_SAMPLES / FS, config=cfg)

#: The display this whole pass is measured against — the owner's laptop screen.
SCREEN = QtCore.QRect(0, 0, 1512, 900)


@pytest.fixture
def screen_1512x900(monkeypatch):
    """Make the window believe it is opening on the owner's 1512x900 display."""
    monkeypatch.setattr(SW, "available_screen_rect", lambda _widget: SCREEN)
    return SCREEN


@pytest.fixture
def window(qtbot, screen_1512x900):
    """A window built exactly as ``main_window._open_subpop_window`` builds it.

    No report installed: every widget asserted below is present, positioned and
    reachable before one lands, and the geometry of the button row does not depend on
    what the report says.
    """
    win = SW.SubpopReportWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    qtbot.wait(30)
    return win


def _controls(win) -> list[tuple[str, QtWidgets.QWidget]]:
    """The row that is everything this window can do, plus the two lines above it."""
    close = [b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == "Close"]
    assert len(close) == 1, "the Close button was renamed or duplicated"
    return [
        ("Copy report", win._copy_btn),
        ("Export…", win._export_btn),
        ("Event gallery…", win._gallery_btn),
        ("Colour the trace", win._trace_btn),
        ("Close", close[0]),
        ("the standing note", win._live_note),
        ("the header", win._header),
    ]


def _frame(win) -> QtCore.QRect:
    """The window's rectangle ON THE SCREEN, title bar included.

    ``geometry()`` is the client area; the title bar sits above it and is part of what
    has to fit. Budgeted from :data:`subpop_window._FRAME_ALLOWANCE` rather than read
    from Qt, because offscreen has no frame and the size has to be right at the FIRST
    paint — which is the whole defect.
    """
    dw, dh = SW._FRAME_ALLOWANCE
    g = win.geometry()
    return QtCore.QRect(g.x() - dw, g.y() - dh, g.width() + dw, g.height() + dh)


# ================================================================================================
# the arithmetic, on its own
# ================================================================================================

def test_the_fitted_size_leaves_the_title_bar_and_both_margins_on_the_screen():
    size = SW.fitted_window_size(SCREEN)
    assert size == QtCore.QSize(1496, 856), (
        f"the opening size for a 1512x900 screen moved to {size}")
    assert size.width() + 2 * SW._SCREEN_MARGIN <= SCREEN.width()
    assert (size.height() + 2 * SW._SCREEN_MARGIN
            + SW._FRAME_ALLOWANCE[1]) <= SCREEN.height()


def test_a_screen_bigger_than_the_preferred_size_still_gets_the_preferred_size():
    """The clamp only ever shrinks: a 27-inch display keeps the measured 1600x980."""
    assert SW.fitted_window_size(QtCore.QRect(0, 0, 2560, 1440)) == QtCore.QSize(
        *SW._DEFAULT_SIZE)


def test_a_screen_smaller_than_the_figures_floor_still_gets_the_floor():
    """Below :data:`_MIN_SIZE` the scatter is a strip; a window that must be dragged is
    the lesser failure, and Qt enforces this floor anyway."""
    assert SW.fitted_window_size(QtCore.QRect(0, 0, 640, 480)) == QtCore.QSize(
        *SW._MIN_SIZE)


def test_the_offscreen_platform_s_own_screen_is_accepted_as_a_screen(qtbot):
    """The seam is a seam and not a test-only branch: unpatched, the window sizes itself
    to whatever screen it really has, and never past it."""
    win = SW.SubpopReportWindow()
    qtbot.addWidget(win)
    avail = SW.available_screen_rect(win)
    assert avail.width() > 0 and avail.height() > 0
    assert win.size() == SW.fitted_window_size(avail)


# ================================================================================================
# the window, on the owner's screen
# ================================================================================================

def test_the_window_and_its_title_bar_open_inside_the_1512x900_display(window):
    """The headline regression: 1600x980 on a 1512x900 screen, unclamped."""
    frame = _frame(window)
    assert SCREEN.contains(frame), (
        f"the window opens at {frame} on a {SCREEN.width()}x{SCREEN.height()} screen")


def test_every_control_this_window_has_opens_wholly_on_the_screen(window):
    """Each of the five buttons, the standing note and the header — whole, in screen
    coordinates, with nothing relying on the user dragging the window first."""
    off = []
    for name, w in _controls(window):
        top_left = w.mapToGlobal(QtCore.QPoint(0, 0))
        rect = QtCore.QRect(top_left, w.size())
        if not SCREEN.contains(rect):
            off.append(f"{name} at {rect}")
    assert not off, "off the 1512x900 screen: " + "; ".join(off)


def test_no_control_is_clipped_by_the_window_either(window):
    """On the screen is not the same as inside the window: a control can be on the
    display and still be cut off by the window's own edge."""
    inside = QtCore.QRect(QtCore.QPoint(0, 0), window.size())
    clipped = []
    for name, w in _controls(window):
        rect = QtCore.QRect(w.mapTo(window, QtCore.QPoint(0, 0)), w.size())
        if not inside.contains(rect):
            clipped.append(f"{name} at {rect} in a {window.width()}x{window.height()}")
    assert not clipped, "clipped by the window: " + "; ".join(clipped)


def test_the_button_row_cannot_be_widened_past_the_screen_by_adding_to_it(window):
    """THE KNOWN SHAPE, pinned: a QHBoxLayout of QPushButtons has no shrink flag, so the
    sum of their minimums is a hard floor on the window. Five buttons need 488 px of the
    1496 available; this fails the day the row is given more than the display holds."""
    buttons = [w for name, w in _controls(window)
               if isinstance(w, QtWidgets.QPushButton)]
    spacing = 6 * (len(buttons) - 1)
    need = sum(b.minimumSizeHint().width() for b in buttons) + spacing + 2 * 8
    assert need <= SW.fitted_window_size(SCREEN).width(), (
        f"the button row needs {need} px inside a {SW.fitted_window_size(SCREEN).width()} "
        f"px window")


def test_the_window_s_own_minimum_still_fits_the_screen_it_opens_on(window):
    """The guard against someone widening the CONTENT again: whatever the panes demand
    has to fit the size the display allows, or the window opens with a scrollbar or a
    clipped pane instead of a control the user can hit."""
    need, size = window.minimumSizeHint(), SW.fitted_window_size(SCREEN)
    assert need.width() <= size.width() and need.height() <= size.height(), (
        f"the window's contents demand {need.width()}x{need.height()} inside "
        f"{size.width()}x{size.height()}")


def _figures_are_figures(window, when):
    for name, view in (("scatter", window.embedding_view()),
                       ("waveforms", window.waveform_view())):
        rect = view._plot.getPlotItem().getViewBox().screenGeometry()
        assert rect.height() >= 60, (
            f"{when}: {name} data rect is {rect.width()}x{rect.height()}")
        assert rect.width() / max(rect.height(), 1) < 9.0, (
            f"{when}: {name} is drawn at {rect.width() / max(rect.height(), 1):.1f}:1")


def test_both_figures_are_still_figures_at_the_size_that_fits(window, qtbot, report):
    """The height that fits is 124 px shorter than the preferred one, and the pane floors
    are what stop that from being paid out of the pictures. The contract is
    ``test_subpop_views``' own: a data rectangle at least 60 px tall and under 9:1.

    Since b21e8ae the caveat box is the 'Full details' of a standing summary that exists
    only with a report. The figures are measured empty, and again with a report and the
    details open -- the state in which the caveats take their share of the figure column,
    which is the state the original assertion measured."""
    _figures_are_figures(window, "empty")
    window.set_report(report, subtitle="synthetic")
    window._caveat_summary.toggle.setChecked(True)
    qtbot.wait(30)
    _figures_are_figures(window, "report, details open")
    assert window._standing_caveats.isVisible() and window._standing_caveats.height() > 0
    assert window._caveat_box.isVisible() and window._caveat_box.height() > 0


# ================================================================================================
# nothing moves under the pointer
# ================================================================================================

def test_re_opening_the_window_does_not_move_it_out_from_under_the_user(window, qtbot):
    """This window is built once and re-shown for every later report. The placement is
    done at construction and never again, so a window the user has dragged or resized
    stays where they put it — a second run must not make it jump."""
    window.move(300, 120)
    window.resize(1000, 700)
    qtbot.wait(20)
    placed = window.geometry()
    window.hide()
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    assert window.geometry() == placed, (
        f"re-showing moved the window {placed} -> {window.geometry()}")


def test_installing_a_report_moves_no_control(window, qtbot):
    """The report arrives into a window that is already the size it will stay: the
    button row, the note and the splitter are where they were before it landed.

    ``set_report(None)`` is the clearing path, and it is the one that runs when the
    window is re-used; the header text changes, no geometry does.
    """
    before = [(name, w.geometry()) for name, w in _controls(window)]
    window.set_report(None)
    qtbot.wait(30)
    after = [(name, w.geometry()) for name, w in _controls(window)]
    assert after == before


# ================================================================================================
# the figures the window opens with are not cropped by the size that fits
# ================================================================================================
#
# WHY THIS IS IN A REACHABILITY TEST. Fitting the window to a 1512x900 screen takes 124 px
# off its height, and the legend's height is a DEMAND on the plot: with eight classes it
# asked for 202 px inside the 168 px plot the fitted window gives it, so the ViewBox grew
# past its own widget and the bottom of the cloud was cropped while the caption underneath
# still read "220 of 220 event(s) drawn". Two figures and a caveat strip cannot each keep a
# 205 px plot inside a 900 px screen, so the row cap alone could never have held on this
# display -- ``subpop_window._fit_legend`` now asks how much height it has and wraps into
# another column when it does not fit, spending figure WIDTH the way the row cap already
# did. Measured, 8 classes at the fitted size: ViewBox 202 px in a 168 px plot -> 128 in 168.

def test_a_legend_too_tall_for_its_plot_wraps_into_columns_instead_of_cropping_it(qtbot):
    """The mechanism, on a real legend in a real plot at the height the owner's screen
    leaves it. One column of eight rows does not fit 168 px; the fit must find that out
    from the plot rather than from a constant."""
    plot = SW._plot_widget("x", "y")
    qtbot.addWidget(plot)
    legend = SW._legend(plot)
    plot.resize(600, 168)          # the embedding plot's height at 1512x900
    plot.show()
    qtbot.waitExposed(plot)
    qtbot.wait(20)
    for cid in range(8):
        legend.addItem(SW.legend_swatch(cid, size=6), f"subpop {cid}  (n = 12)")

    SW._fit_legend(legend, 8)                              # rows alone: the old cap
    tall = legend.boundingRect().height()
    room = SW._legend_room(plot)
    assert tall > room, (
        f"this test proves nothing: eight rows are {tall:.0f} px and the plot has "
        f"{room:.0f} px, so there was nothing to fit")

    SW._fit_legend(legend, 8, room)                        # ...and with the height
    assert legend.boundingRect().height() <= room + 1, (
        f"the legend still demands {legend.boundingRect().height():.0f} px of a "
        f"{room:.0f} px plot, so the figure it labels is cropped")
    assert legend.columnCount > 1, "it fitted without wrapping, which cannot be"


def test_the_legend_is_refitted_when_the_plot_it_lives_in_is_resized(qtbot):
    """The path the user actually takes: ``main_window`` installs the report BEFORE it
    shows the window, so the first fit runs against a plot with no size yet. Without the
    refit the default colouring — the mixture's own, up to eight classes — opens cropped
    and the caption claims every point is drawn."""
    view = SW.EmbeddingScatterView()
    qtbot.addWidget(view)
    view.resize(560, 520)
    view.show()
    qtbot.waitExposed(view)
    qtbot.wait(20)
    for cid in range(8):
        view._legend.addItem(SW.legend_swatch(cid, size=6), f"subpop {cid}  (n = 12)")
    view._legend_n = 8
    SW._fit_legend(view._legend, 8, SW._legend_room(view._plot))

    view.resize(560, 300)          # the embedding pane's height inside the fitted window
    qtbot.wait(40)
    vb = view._plot.getPlotItem().getViewBox()
    assert vb.boundingRect().height() <= view._plot.height() + 1, (
        f"after the resize the ViewBox is {vb.boundingRect().height():.0f} px inside a "
        f"{view._plot.height()} px plot: the figure is cropped")
    assert view._legend.boundingRect().height() <= SW._legend_room(view._plot) + 1
