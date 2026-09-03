"""S6-01 — does the Style panel's Font row name the font the plot is drawn in?

The defect: ``StyleSettings.font_family`` defaults to ``""`` ("the application default
family"), the sync showed that as ``setCurrentFont(QFont())``, and on macOS the application
default is ``.AppleSystemUIFont`` — a PRIVATE system family ``QFontComboBox`` does not list,
so the combo fell back to its first entry ("Academy Engraved LET" on this machine) and the
panel named a decorative engraved face for a figure drawn in the system UI font.

WHAT THIS SCRIPT NOW ASSERTS is the contract the Font row owes the user, not the Qt
primitive's behaviour — see the comment below. Both directions:

  A. with no family chosen, the row must NOT name a concrete installed family;
  B. with a family chosen, the row must name THAT family.

Run:  QT_QPA_PLATFORM=offscreen python docs/sweep6/repro_S6-01_font_combo.py
      python docs/sweep6/repro_S6-01_font_combo.py          # cocoa, no window is shown
Exit: 7 = reproduced, 0 = fixed.
"""
import os, sys
import nanoscope                                    # Qt plugin-path bootstrap
import numpy as np, pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

# DRIVEN, NOT DEMONSTRATED. The first version of this script built a bare
# QFontComboBox and would have gone on exiting 7 forever, because the Qt primitive
# never changes -- the APP is what had to. This one asserts the invariant the Font row
# owes the user, in both directions:
#   A. with no family chosen, the row must NOT name a concrete installed family;
#   B. with a family chosen, the row must name THAT family.
EVENTS = "/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv"

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
def pump(n=20):
    for _ in range(n): app.processEvents(QtCore.QEventLoop.AllEvents, 20)

if os.path.exists(EVENTS):                          # the owner's own 982 events
    frame = pd.read_csv(EVENTS)
else:                                               # ...or a stand-in, so this still runs
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"dwell_s": np.abs(rng.normal(2e-2, 3e-3, 300)),
                          "depth_pA": np.abs(rng.normal(100, 20, 300)),
                          "frac_depth": np.abs(rng.normal(.25, .05, 300)),
                          "accepted": np.ones(300, bool)})

w = PlotsWindow(Session())
w.set_data({"events_nm_0005": frame})
w.set_plot_type("Histogram")
w._run_plot(); pump()

row = w._font_combo
drawn = QtGui.QFontInfo(w._qfont()).family()        # the family the plot is really in
print(f"platform          : {app.platformName()}")
print(f"style.font_family : {w._style.font_family!r}   <- 'the application default family'")
print(f"plot is drawn in  : {drawn!r}")
print(f"Font row says     : {row.currentText()!r}")

bad_blank = QtGui.QFont(row.currentText()).exactMatch()
print(f"  A FAILS — the row names the installed family {row.currentText()!r}, but the "
      f"plot is drawn in {drawn!r}" if bad_blank else
      "  A ok — the row does not pretend a family was chosen")

w.set_font_family("Courier New"); pump()
bad_named = "Courier New" not in row.currentText()
print(f"after set_font_family('Courier New') the row says {row.currentText()!r}")
print("  B FAILS — the row does not name the chosen family" if bad_named else "  B ok")

bad = bool(bad_blank or bad_named)
print("REPRODUCED" if bad else "consistent")
os._exit(7 if bad else 0)
