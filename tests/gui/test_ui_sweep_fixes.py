"""Beta-sweep UI group: a quit keeps a floating dock, Reset Window Layout forgets every
remembered window, a hand-add leaves the gallery and trace where they were, and a hand-add
over a detected event is refused."""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

import pytest
from PySide6 import QtCore

from nanoscope.gui.main_window import MainWindow


# -- LC-1: quitting with a floating dock ---------------------------------------------------
@pytest.mark.parametrize("attr", ["_results_dock", "_file_dock"])
def test_a_floating_dock_survives_a_quit(private_settings, qtbot, attr):
    window = build_window(qtbot)
    dock = getattr(window, attr)
    dock.setFloating(True)
    dock.show()
    settle()
    assert dock.isFloating() and dock.isVisible()
    close_window(window)
    again = build_window(qtbot, detection=False)
    redock = getattr(again, attr)
    assert redock.isFloating() and not redock.isHidden(), "teardown closed the dock first"
    redock.setFloating(False)
    close_window(again)


# -- X2: Reset Window Layout forgets every remembered window -------------------------------
def _reset(window, monkeypatch):
    monkeypatch.setattr(MainWindow, "_confirm_reset_window_layout", lambda self: True)
    window._reset_window_layout()
    settle()


def _stored_geometry_keys() -> list[str]:
    s = QtCore.QSettings()
    return sorted(k for k in s.allKeys() if k.endswith("/geometry")
                  or k == "signalConditioning/splitter")


def _default_sc_rect(window) -> tuple:
    from nanoscope.gui.signal_conditioning import default_window_rect
    ws = window._signal_conditioning
    return default_window_rect(
        window.frameGeometry(), window.geometry(), window._available_screen_geometry(),
        width=max(ws.minimumWidth(), ws.DEFAULT_WIDTH)).getRect()


def _open_sc_and_move(window) -> None:
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    ws.setGeometry(300, 60, 820, 700)
    ws.splitter.setSizes([300, 520])
    settle()
    ws.hide()
    settle()


def test_reset_forgets_a_hidden_signal_conditioning_window(private_settings, qtbot,
                                                           monkeypatch):
    window = build_window(qtbot, detection=False)
    _open_sc_and_move(window)
    close_window(window)
    assert "signalConditioning/geometry" in QtCore.QSettings().allKeys()

    window = build_window(qtbot, detection=False)      # reset without ever opening it
    _reset(window, monkeypatch)
    close_window(window)
    assert _stored_geometry_keys() == []

    window = build_window(qtbot, detection=False)      # the relaunch: where a first launch puts it
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    assert ws.geometry().getRect() == _default_sc_rect(window)
    assert ws.splitter.sizes()[0] == ws.LEFT_WIDTH
    ws.hide()
    close_window(window)


def test_reset_puts_a_signal_conditioning_window_hidden_this_session_back(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot, detection=False)
    _open_sc_and_move(window)
    _reset(window, monkeypatch)
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    assert ws.geometry().getRect() == _default_sc_rect(window)
    assert ws.splitter.sizes()[0] == ws.LEFT_WIDTH
    ws.hide()
    settle()
    close_window(window)
    assert "signalConditioning/geometry" in QtCore.QSettings().allKeys(), \
        "reopened after the Reset, so it is the user's again"
    assert "windows/signal_conditioning/geometry" not in QtCore.QSettings().allKeys()


def test_reset_forgets_a_child_window_hidden_before_the_reset(private_settings, qtbot,
                                                               monkeypatch):
    window = build_window(qtbot)
    window._open_event_gallery()
    gal = window._event_gallery
    gal.move(100, 100)
    settle()
    gal.hide()
    settle()
    assert "windows/event_gallery/geometry" in QtCore.QSettings().allKeys()
    _reset(window, monkeypatch)
    assert _stored_geometry_keys() == []
    close_window(window)                       # the quit-save used to write it back
    assert _stored_geometry_keys() == []


# -- XI-1 / XI-2: the hand-add (Ctrl+Shift+M) ---------------------------------------------
def _frame(window):
    return window._analysis_panel.current_detection().payload.df


def _bracket(window, start: int, end: int) -> None:
    from ._inspector_hosting import FS
    window._analysis_panel.measure_panel.set_active(True)
    window._trace_view.set_measure_region(start / FS, end / FS)
    settle()


def _x_range(window) -> list[float]:
    return [round(float(v), 6) for v in window._trace_view.anchor_plot().vb.viewRange()[0]]


def test_a_hand_add_with_the_gallery_open_does_not_move_the_trace(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_gallery()
    settle()
    assert window._event_gallery.isVisible()
    window._trace_view.set_x_range(0.40, 0.50)         # an empty stretch, away from tile 0
    settle()
    before = _x_range(window)
    announced: list[int] = []
    window._event_gallery.eventSelected.connect(announced.append)
    _bracket(window, 21_000, 21_400)
    window._mark_selection_as_missed()
    settle()
    assert len(_frame(window)) == 7
    assert announced == [] and _x_range(window) == before
    window._act_undo.trigger()
    settle()
    assert len(_frame(window)) == 6
    assert announced == [] and _x_range(window) == before
    close_window(window)


def test_a_hand_add_over_a_detected_event_is_refused(private_settings, qtbot):
    window = build_window(qtbot)
    df = _frame(window)
    first, last = int(df["start_idx"].iloc[2]), int(df["end_idx"].iloc[2])
    for start, end in ((first, last), (first - 40, first + 40), (last - 40, last + 40),
                       (first + 5, last - 5)):
        _bracket(window, start, end)
        window._mark_selection_as_missed()
        assert len(_frame(window)) == 6, (start, end)
        assert "overlaps event" in window.statusBar().currentMessage()
        assert "could not be added" not in window.statusBar().currentMessage()
    _bracket(window, last, last + 300)                   # touching is not overlapping
    window._mark_selection_as_missed()
    assert len(_frame(window)) == 7
    assert "Added missed event" in window.statusBar().currentMessage()
    close_window(window)
