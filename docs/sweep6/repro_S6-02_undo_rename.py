"""S6-02 — Undo of a source rename reverts the plot but leaves the typed name in the list.

Gesture: Plots ▸ double-click a row in Sources ▸ type a new name ▸ Enter ▸ Undo.
Expected: the row reads its original name again, as the legend does.
Observed:  the row keeps the typed name while the legend, the canvas and `config()` revert.
           The window then shows two different names for the same source, and it survives
           a re-Plot; only a Refresh (set_data) puts the row back.

Run:  QT_QPA_PLATFORM=offscreen python docs/sweep6/repro_S6-02_undo_rename.py
Exit: 7 = reproduced, 0 = fixed.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope                                     # Qt plugin-path bootstrap
import numpy as np, pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import shiboken6
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
def pump(n=20):
    for _ in range(n): app.processEvents(QtCore.QEventLoop.AllEvents, 20)

rng = np.random.default_rng(0)
frame = pd.DataFrame({"dwell_s": np.abs(rng.normal(2e-2, 3e-3, 300)),
                      "depth_pA": np.abs(rng.normal(100, 20, 300)),
                      "frac_depth": np.abs(rng.normal(.25, .05, 300)),
                      "snr": np.abs(rng.normal(9, 2, 300)),
                      "accepted": np.ones(300, bool),
                      "filter_limited": np.zeros(300, bool)})
w = PlotsWindow(Session()); w.set_data({"-700 mV": frame}); w.resize(1200, 800); w.show(); pump()
for b in w.findChildren(QtWidgets.QPushButton):
    if b.text().strip().lower().startswith("plot"): b.click(); break
pump(30)

sl = w._source_list
def rows(): return [sl.item(i).text() for i in range(sl.count())]
def legend(): return [w._legend_entries.item(i).text() for i in range(w._legend_entries.count())]

original = rows()[0]
sl.editItem(sl.item(0)); pump(5)
ed = sl.findChild(QtWidgets.QLineEdit)
ed.selectAll()
for ch in "Apo -700 mV":                              # real typing, through the delegate
    QtWidgets.QApplication.sendEvent(ed, QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_A, QtCore.Qt.NoModifier, ch))
if shiboken6.isValid(ed):
    sl.commitData(ed); sl.closeEditor(ed, QtWidgets.QAbstractItemDelegate.SubmitModelCache)
pump(25)
print(f"renamed : row={rows()[0]!r}  legend={legend()!r}  overrides={dict(w._style.label_overrides)}")

w._undo_btn.click(); pump(30)
print(f"undone  : row={rows()[0]!r}  legend={legend()!r}  overrides={dict(w._style.label_overrides)}")
bad = rows()[0] != original
if bad:
    print(f"REPRODUCED — the Sources list still says {rows()[0]!r}; "
          f"the plot went back to {original!r}")
else:
    print("consistent: Undo put the row back too")
sys.exit(7 if bad else 0)
