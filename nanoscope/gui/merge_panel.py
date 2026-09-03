"""Post-detection sub-event merge controls (a pure, signal-only widget).

A single translocation that briefly recrosses the exit threshold is reported as
several adjacent fragments. This little control group lets the user stitch them
back together AFTER detection — over the whole trace by a time window, or over the
on-trace Measure selection — without re-tuning thresholds and dropping real short
events.

Like :class:`~nanoscope.gui.panels.DetectionPanel`, this is a **pure view**: it holds
the window value and emits intent, nothing else. It imports only Qt and never reaches
into another panel — :class:`~nanoscope.gui.main_window.MainWindow` is the sole wirer
that routes these signals onto the active detection's :class:`EventSet` and the
existing curation refresh. The actual merge maths lives in
:mod:`nanoscope.analysis.merge`.

Signals:
    mergeAllRequested(float): "Merge all" clicked, carrying the window in SECONDS.
    mergeRegionRequested(): "Merge events in selection" clicked.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["MergePanel"]

# Sensible default: ~150 µs is a few samples at 50 kHz — long enough to bridge a brief
# threshold recrossing, short enough to leave genuinely separate events alone.
#: Default merge window, in MILLISECONDS. Was 150 µs behind a µs spinbox — and the first
#: hands-on replication of the tutorial typed "15" into that field meaning 15 ms, merged at
#: 15 µs, and reasonably concluded the merge "was not giving good results": a 1000x silent
#: under-merge. Fragment gaps on this bench's own recordings are millisecond-scale (the
#: walkthroughs merge at 10-15 ms), so the field now speaks the unit the physics does.
#:
#: AND THAT 10-15 ms RECOMMENDATION IS STILL RIGHT, even though the reproduced runaway that
#: put an occupancy floor into :mod:`nanoscope.analysis.detect` was a 10 ms merge. The two
#: are not claims about the same quantity. The WINDOW is the NECESSARY condition -- how far
#: apart the pieces of one translocation are allowed to fall, which on this bench is
#: milliseconds. :data:`~nanoscope.analysis.detect.MIN_MERGE_OCCUPANCY` is the SUFFICIENT
#: one: of everything a window that wide sweeps up, only the groups that actually FILL the
#: span they would make are one event. So 10-15 ms goes on stitching the shattered events it
#: was chosen for, and the floor declines the 8 ms-spaced train it also happens to reach.
DEFAULT_WINDOW_MS = 0.15

#: Smallest window the spinbox can hold, in MILLISECONDS. It used to be 0.0, which LOOKED
#: like the "0 = off" sentinel its sibling :class:`~nanoscope.gui.panels.DetectionPanel`
#: spells out on ``merge_gap_ms`` and ``max_dwell_s`` — and was not one: at 0.0 the window
#: is 0 samples, ``gap <= 0`` is true for exactly-touching rows, and "Merge all" still
#: merged them. A control that reads as off and is not off is worse than either.
#:
#: The floor is a MINIMUM rather than a real 0-is-off sentinel because there is nothing
#: for "off" to mean on this panel. ``merge_gap_ms`` is a step INSIDE a detection run that
#: can be skipped; "Merge all" is a button whose only meaning is merge, so a window of
#: "don't" would give the user a click that does nothing — and the codebase's own
#: argument against greying the Detect button applies verbatim: a control that silently
#: refuses is a dead control and a guess. 0.01 ms is a fifth of a sample at 50 kHz, i.e.
#: the smallest window that is still honestly a window: it stitches fragments that touch
#: exactly and nothing else.
MIN_WINDOW_MS = 0.01

# What actually reverses a merge. The old wording — "Merges are undone by re-running
# detection" — pointed at the single most destructive click in the application: a
# re-detection replaces the whole table and takes every accept/reject decision with it.
# Edit ▸ Undo restores the exact table the merge replaced, curation included.
_UNDO_NOTE = "Reversed by Edit ▸ Undo (⌘Z)."

# "Merge all" re-characterizes every surviving span and costs 3.3-8.7 s on the owner's
# real recordings. It used to run on the GUI thread with nothing on screen to say so;
# it now runs on the shared analysis controller, which is what puts the status-bar
# progress and Cancel in front of it. The button says so, because a click whose effect
# arrives seconds later must be a click the user knows is still working.
_BACKGROUND_NOTE = ("Runs in the background on a large table — watch the status bar; "
                    "Cancel there stops it and changes nothing.")

# The brake on transitivity, said out loud on both buttons. A window wide enough to bridge
# real fragments is also wide enough to chain a whole train of separate events into one row
# (at 10 ms, 40 clean events 8 ms apart became one 312 ms event of depth 0.70 pA against a
# 300 pA truth), so a candidate span that is mostly OPEN PORE is refused instead of merged.
# The user has to be told that, or a refusal is just the merge "not working" again.
_OCCUPANCY_NOTE = ("A run that is mostly open pore between its events is not one "
                   "translocation, so it is left alone and the status bar says how many "
                   "were skipped.")


class MergePanel(QtWidgets.QGroupBox):
    """Window spinbox + "Merge all" + "Merge events in selection".

    Public API:
        window_s() -> float                 # the merge window, in seconds
        set_window_s(seconds)               # set it programmatically
        set_detection_present(bool)         # gate the buttons on an active detection
    """

    mergeAllRequested = QtCore.Signal(float)   # window in SECONDS
    mergeRegionRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None, *,
                 table_selector: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Merge sub-events", parent)
        self.setToolTip(
            "Stitch a translocation the detector split into several fragments back "
            "into one event. " + _UNDO_NOTE)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Window"))
        self.window_ms = QtWidgets.QDoubleSpinBox()
        self.window_ms.setRange(MIN_WINDOW_MS, 100.0)
        self.window_ms.setDecimals(2)
        self.window_ms.setSingleStep(0.5)
        self.window_ms.setValue(DEFAULT_WINDOW_MS)
        self.window_ms.setSuffix(" ms")
        self.window_ms.setToolTip(
            "Merge window: detected events whose gap (next start − current end) is "
            "within this many milliseconds are stitched into one (transitively along a "
            f"chain). The smallest window ({MIN_WINDOW_MS:g} ms) still stitches "
            "fragments that touch exactly — there is no 'off' here, because the button "
            "next to it only ever means merge. " + _UNDO_NOTE)
        top.addWidget(self.window_ms)
        top.addStretch(1)
        self.merge_all_btn = QtWidgets.QPushButton("Merge all")
        self._merge_all_tip = (
            "Merge every run of fragments within the window across the whole detection.\n\n"
            + _OCCUPANCY_NOTE + "\n\n" + _BACKGROUND_NOTE + " " + _UNDO_NOTE)
        self.merge_all_btn.setToolTip(self._merge_all_tip)
        self.merge_all_btn.clicked.connect(self._emit_all)
        top.addWidget(self.merge_all_btn)
        root.addLayout(top)

        self.merge_region_btn = QtWidgets.QPushButton("Merge events in selection")
        self._merge_region_tip = (
            "Merge every detected event overlapping the on-trace Measure selection into "
            "one event (an explicit selection overrides the window — not the open-pore "
            "check). Turn on the Measure tool and drag a region first.\n\n"
            + _OCCUPANCY_NOTE + " " + _UNDO_NOTE)
        self.merge_region_btn.setToolTip(self._merge_region_tip)
        self.merge_region_btn.clicked.connect(self.mergeRegionRequested)

        # Keep the result selector in the strip's EXISTING second row. A standalone row
        # above this scroll-wrapped group adds its own width/height minimum to the entire
        # Inspector dock and steals vertical space from the controls column at 1280x800.
        # Here it shares the already-reserved strip height and inherits the strip's narrow-
        # dock horizontal scrolling. Put it first so a restored 180 px dock still exposes
        # the table chooser without scrolling; the longer merge action remains reachable
        # by the strip's scrollbar.
        bottom = QtWidgets.QHBoxLayout()
        if table_selector is not None:
            bottom.addWidget(QtWidgets.QLabel("Table"))
            bottom.addWidget(table_selector, 1)
        bottom.addWidget(self.merge_region_btn)
        root.addLayout(bottom)

        self.set_detection_present(False)

    # -- intent ---------------------------------------------------------------------------------
    def _emit_all(self) -> None:
        self.mergeAllRequested.emit(self.window_s())

    # -- state ----------------------------------------------------------------------------------
    def window_s(self) -> float:
        """The merge window in seconds (the spinbox reads milliseconds)."""
        return float(self.window_ms.value()) * 1e-3

    def set_window_s(self, seconds: float) -> None:
        """Set the merge window from seconds. Clamped to ``[MIN_WINDOW_MS, 100 ms]`` by the
        spinbox, so a restored 0 from an older session comes back as the minimum window
        rather than as a window that reads "off" and is not."""
        self.window_ms.setValue(float(seconds) * 1e3)

    def set_detection_present(self, present: bool, reason: str = "") -> None:
        """Enable the merge buttons only when the active target has a stored detection.

        Pushed by the owner (:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`) on
        every result / selection change. The window spinbox stays live so the user can
        dial it in ahead of a detection."""
        on = bool(present)
        self.merge_all_btn.setEnabled(on)
        self.merge_region_btn.setEnabled(on)
        suffix = ("\n\n" + str(reason)) if not on and reason else ""
        self.merge_all_btn.setToolTip(self._merge_all_tip + suffix)
        self.merge_region_btn.setToolTip(self._merge_region_tip + suffix)
