"""S6-03b — a PlotConfig carrying the pre-rename "density (area = 1)" drew counts.

S6-03 renamed the histogram's Normalize item, which renamed a PERSISTED VALUE, and put the
migration at the file reader (`plot_config_from_json`). Both file doors — Import JSON and a
stored template — go through it. The two IN-PROCESS doors do not, and those are the ones a
script or a notebook holding a `PlotConfig` uses:

  * `apply_config(cfg)` writes the mode with `QComboBox.setCurrentText`, a silent no-op for
    an item the combo has not got, so the window went on drawing whatever it already showed;
  * `apply_template(cfg)` goes through `_installable`, whose job is to resolve a name this
    window cannot select back to the one on screen — and it cannot tell yesterday's word for
    the mode you are in from a mode this build has never heard of.

Both silently substituted RAW COUNTS for the density the caller asked for: no exception, no
note in Details, no banner. The migration now lives in `HistogramConfig.__post_init__`, so
no live PlotConfig can carry the old spelling and every door is covered by one point.

Run:  QT_QPA_PLATFORM=offscreen python docs/sweep6/repro_S6-03b_config_doors.py
Exit: 7 = reproduced, 0 = fixed.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope                                     # Qt plugin-path bootstrap
import numpy as np, pandas as pd
from dataclasses import replace
from PySide6 import QtCore, QtWidgets
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

OLD = "density (area = 1)"

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
def pump(n=20):
    for _ in range(n): app.processEvents(QtCore.QEventLoop.AllEvents, 20)

rng = np.random.default_rng(0)
frame = pd.DataFrame({"dwell_s": 2e-5 + rng.exponential(2.94e-4, 300),
                      "depth_pA": np.abs(rng.normal(100, 20, 300)),
                      "frac_depth": np.abs(rng.normal(.25, .05, 300)),
                      "snr": np.abs(rng.normal(9, 2, 300)),
                      "accepted": np.ones(300, bool),
                      "filter_limited": np.zeros(300, bool)})
w = PlotsWindow(Session()); w.set_data({"-700 mV": frame}); w.resize(1200, 800); w.show()
pump()
w.set_plot_type("Histogram"); w.set_histogram_column("dwell_s"); w.set_fit("none")
w.set_normalization("count")
for b in w.findChildren(QtWidgets.QPushButton):
    if b.text().strip().lower().startswith("plot"): b.click(); break
pump(30)


def heights():
    for item in w.plot_item().listDataItems():
        y = item.getData()[1]
        if y is not None and len(y):
            return np.asarray(y, float)
    return np.array([])


counts = heights()
print(f"opened on : {w.normalization()!r}  bars {counts[:5].tolist()}")

failed = []
for door, call in (("apply_config", lambda c: w.apply_config(c)),
                   ("apply_template", lambda c: w.apply_template(c))):
    w.set_normalization("count"); pump(10)
    cfg = w.config()
    call(replace(cfg, histogram=replace(cfg.histogram, normalization=OLD)))
    pump(30)
    got, bars = w.normalization(), heights()
    print(f"{door:15s}: asked {OLD!r} -> combo {got!r}, y axis "
          f"{w.plot_item().getAxis('left').labelText!r}, bars {bars[:5].round(3).tolist()}")
    if got != "density":
        failed.append(door)

if failed:
    print(f"REPRODUCED — {', '.join(failed)} drew {w.normalization()!r} under a request "
          f"for a density, silently.")
    sys.exit(7)
print("honoured: the pre-rename spelling reaches the density mode through every door.")
sys.exit(0)
