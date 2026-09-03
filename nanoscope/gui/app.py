from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

# Importing this module imports the `nanoscope` package first, whose __init__ runs
# bootstrap (Qt plugin-path fix) BEFORE PySide6 is imported above. That ordering is load-bearing;
# do not import PySide6 from a module that is not under the package.

__all__ = ["configure_pyqtgraph", "create_app", "main"]

#: A final, bounded chance for a detached worker after ``app.exec()`` returns — on top
#: of ``MainWindow._teardown``'s own budget. Kept short because by this point the user
#: has seen the window disappear and is waiting on a Dock icon.
_FINAL_DRAIN_MS = 1500


def configure_pyqtgraph() -> None:
    """The three settled pyqtgraph options (contract §14). See the contract for why each is fixed:
    antialias off (nothing to gain on ~15k downsampled points), OpenGL off (raster path is already
    1.8–5.6 ms at 15M and the GL curve path has macOS line-width/AA artifacts), row-major images."""
    pg.setConfigOptions(
        antialias=False,
        useOpenGL=False,
        imageAxisOrder="row-major",
    )


def _apply_theme(app: QtWidgets.QApplication) -> None:
    """A calm dark theme so the app reads as an instrument, not a default-grey demo. Applied to
    both Qt widgets (Fusion + palette) and pyqtgraph (global background/foreground) so the docks
    and the plots share one look."""
    app.setStyle("Fusion")

    bg = QtGui.QColor("#16181c")
    base = QtGui.QColor("#1c1f24")
    panel = QtGui.QColor("#22262c")
    text = QtGui.QColor("#d7dbe0")
    dim = QtGui.QColor("#8b9199")
    accent = QtGui.QColor("#4FC3F7")

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, bg)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, base)
    pal.setColor(QtGui.QPalette.AlternateBase, panel)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, panel)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.ToolTipBase, panel)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#0d0f12"))
    pal.setColor(QtGui.QPalette.PlaceholderText, dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, dim)
    app.setPalette(pal)

    pg.setConfigOption("background", "#111316")
    pg.setConfigOption("foreground", "#b9bfc7")


class _WheelGuard(QtCore.QObject):
    """Scrolling the page must never edit a value — clicked or not.

    Qt's default gives spinboxes and combos wheel focus: a two-finger scroll that merely
    PASSES OVER one changes its value, silently. The user reported it twice — "the number
    scroll changes values by simply scrolling on top of them", and "the dropdown menus
    change by scrolling over them. The user needs to click first, then select" — and on
    this app a silently changed number is a silently changed ANALYSIS (a merge window, a
    fit component count, a threshold).

    NO FOCUS EXEMPTION. An earlier version let a FOCUSED spinbox keep the gesture, reading
    the click as consent. It is the opposite: click-then-scroll is precisely the sequence
    the second report describes — the user clicks a control to read it or to open its
    popup, then scrolls the dock, and the click is what ARMS the silent edit. So a value
    widget refuses the wheel whether or not it has focus. Nothing is lost: the arrow keys,
    typing and the spin buttons still change a number, and a combo is changed the way the
    report asks for — click, then pick from the popup. Scrolling INSIDE an open popup is
    untouched, because the popup's list is a QAbstractItemView and not a QComboBox.

    QAbstractSlider is not guarded at all (it was, on focus). Every QAbstractSlider in
    this app is a QScrollBar — the scroll areas' own bars and the pan bar under the trace
    — which must stay scrollable or the docks and the pan gesture break; there is no
    QSlider or QDial here whose value a stray scroll could quietly change.

    THE WHEEL IS REDIRECTED, NOT SWALLOWED. The analysis dock is a QScrollArea holding
    ~140 spinboxes and combos, so a guard that merely returned True would stop the dock
    scrolling whenever the pointer happened to sit over one of them — a bigger complaint
    than the one being fixed. The event is instead handed to the nearest enclosing scroll
    area's viewport, so the page scrolls under the cursor exactly as it does over a label.

    One application-level filter, not per-widget patches: installed on the QApplication in
    :func:`create_app`, it sees events bound for EVERY object, so every such widget —
    present, future, and the ones Qt builds in C++ — is covered by construction.
    """

    _GUARDED = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)

    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if event.type() != QtCore.QEvent.Wheel or not isinstance(obj, self._GUARDED):
            return False

        # The value widget declines the gesture; the scroll area below is entitled to see
        # it as unhandled, so clear the accepted flag before handing it on.
        event.ignore()

        # Walk out to the page that SHOULD have scrolled. A guarded widget is always a
        # DESCENDANT of the scroll area, never an ancestor of it, so the redirect can
        # never arrive back at a widget this filter guards: no recursion is possible.
        area = obj.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
            area = area.parentWidget()
        viewport = area.viewport() if area is not None else None
        if viewport is not None:
            # The same event object, deliberately. QAbstractScrollArea reads angleDelta /
            # pixelDelta / modifiers and never the local position (which still refers to
            # the spinbox), and reusing it keeps the scroll phase and inverted flags a
            # trackpad gesture depends on. If this viewport has nothing left to scroll it
            # ignores the event and Qt walks it out to the next scroll area up, which is
            # what a nested dock should do.
            QtWidgets.QApplication.sendEvent(viewport, event)

        # True either way: with no enclosing scroll area there is simply nothing to
        # scroll, and the one outcome that must not happen is the value widget seeing it.
        return True


def create_app(argv: list[str]) -> QtWidgets.QApplication:
    """Return the singleton QApplication, naming it so QSettings has an org/app to key on."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(argv)
    app.setApplicationName("NanoScope")
    app.setApplicationDisplayName("NanoScope")
    app.setOrganizationName("NanoporeAnalyzer")
    app.setOrganizationDomain("nanoscope.local")
    # Installed once and kept alive on the app object itself (a local would be
    # garbage-collected and the filter silently removed).
    if getattr(app, "_nanoscope_wheel_guard", None) is None:
        app._nanoscope_wheel_guard = _WheelGuard(app)
        app.installEventFilter(app._nanoscope_wheel_guard)
    return app


def main(argv: list[str] | None = None) -> int:
    """Entry point (`nanoscope` / `python -m nanoscope`)."""
    argv = list(sys.argv if argv is None else argv)
    app = create_app(argv)
    configure_pyqtgraph()
    _apply_theme(app)

    # v2 registrations MUST run before any module lookup / overlay dispatch, i.e. before
    # MainWindow builds the AnalysisController + AnalysisPanel: register the ported
    # AnalysisModules into ANALYSIS_REGISTRY and the overlay renderers into the TraceView seam.
    from ..analysis import register_all
    from .overlays import register_all_renderers

    register_all()
    register_all_renderers()

    # Imported lazily so this module stays importable (for create_app/configure_pyqtgraph) without
    # constructing the whole widget tree, and so app.py never pulls a panel in at import time.
    from .main_window import MainWindow

    window = MainWindow()
    window.show()
    rc = int(app.exec())

    # THE LAST WORD ON QUITTING WHILE A WORKER IS STILL RUNNING.
    #
    # ``MainWindow._teardown`` has already aborted every controller, synced QSettings and
    # spent its drain budget; the window is gone and ``app.exec()`` has returned. If a
    # worker is STILL running here it is inside an uninterruptible C call (a TensorFlow
    # import is 4.1 s and polls nothing while it runs), and letting Python finalise would
    # clear ``worker_thread._DETACHED``, destroy the C++ QThread under it, and take the
    # process down with ``qFatal`` — measured exit 134, 4/4 runs, plus a macOS crash
    # report on every one.
    #
    # So: one more bounded chance, then leave without running destructors. Nothing is
    # lost by that — the run was cancelled, its answer discarded, settings already on
    # disk — and it is the only mechanism that actually works on PySide6 6.11.1
    # (``releaseOwnership`` does not exist; ``shiboken6.invalidate`` deletes the C++
    # object and still aborts, 3/3).
    from .worker_thread import drain_detached, exit_without_destroying_live_threads

    if drain_detached(_FINAL_DRAIN_MS):
        exit_without_destroying_live_threads(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
