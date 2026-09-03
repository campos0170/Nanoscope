"""S6-03 — the histogram's y axis did not say what its numbers are in.

The density heights are a probability density PER SECOND, always: ``_normalized``
divides the counts by the bin width in the column's STORED unit. The x axis is rescaled
for DISPLAY only (``_axis_label_for``: ``AxisItem.setScale`` on linear x,
``_DecadeLabelAxis.display_factor`` on log x), and it must stay that way — the
exponential dwell fit takes its detection floor from the detector's ``min_dwell_s`` in
seconds, so scaling the sample would bias tau silently. So the area under the curve AS
DRAWN is

    linear x:  1 (Time unit = s)   1000 (ms, the app's default)   1e6 (µs)
    log x:     23.3011485 — IN ALL THREE UNITS

while the mode was named ``density (area = 1)`` and the y axis said a bare ``density``.

THE FIX IS NOT A RESCALE, AND THIS SCRIPT NO LONGER CHECKS FOR ``area = 1``. On a log
axis the drawn abscissa is log10(seconds): the drawn area is a property of the data's
RANGE, identical in every time unit, and no constant factor can make it 1 — dividing the
heights by the display factor (the obvious repair, prototyped and measured) moves it to
0.0233 at ms and 2.33e-05 at µs, i.e. further from 1. The app's own shipped "Dwell-time
histogram (log-x, exponential)" template selects exactly that case. So the fixed contract
is that the FIGURE states what its numbers are in and the mode stops promising an area it
cannot keep:

  * the y axis reads ``density (1/s)`` — in s, ms AND µs, and on linear AND log x,
    because the heights are per second whatever the ticks say. A label that tracked the
    ticks would be a second, subtler lie;
  * the mode's NAME no longer promises ``area = 1``;
  * Details names the factor wherever the drawn area is not 1 on a linear axis, says
    NOTHING when it is 1 (Time unit = s — a pane that cries wolf is not read), and on
    log x says instead that the drawn area is not the integral of the density at all;
  * and the STORED-unit area is still exactly 1: no drawn number moved.

The measured drawn areas are still printed on every row, so the record stays honest about
what the picture integrates to.

Run:  QT_QPA_PLATFORM=offscreen python docs/sweep6/repro_S6-03_density_unit.py
Exit: 7 = reproduced, 0 = fixed.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope                                     # Qt plugin-path bootstrap
import numpy as np, pandas as pd
from pathlib import Path
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

EV = Path("/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
def pump(n=20):
    for _ in range(n): app.processEvents(QtCore.QEventLoop.AllEvents, 20)

if EV.exists():
    df = pd.read_csv(EV)
else:                                                # synthetic stand-in, same shape
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"dwell_s": 10 ** rng.normal(-2.2, 0.12, 900),
                       "depth_pA": np.abs(rng.normal(100, 20, 900)),
                       "frac_depth": np.abs(rng.normal(.25, .05, 900)),
                       "snr": np.abs(rng.normal(9, 2, 900)),
                       "accepted": np.ones(900, bool),
                       "filter_limited": np.zeros(900, bool)})

w = PlotsWindow(Session()); w.set_data({"-700 mV": df}); w.resize(1200, 800); w.show(); pump()
w._plot_type_combo.setCurrentText("Histogram")
w._hist_col.setCurrentText("dwell_s")
# NOT `_norm_combo.setCurrentText(...)`. On a non-editable combo that is a SILENT no-op
# for text the combo does not carry, so after the S6-03 rename the old spelling would
# leave this window on `count`: the script would histogram COUNTS, report area = 1 in
# every unit and EXIT 0 — a false green on the very defect it exists to detect (measured
# on a renamed build: the combo landed on 'count'). `set_normalization` raises instead,
# and resolves the bare "density" by unique prefix on a pre-rename build too, so this one
# line selects the right mode on BOTH builds and cannot silently miss on either.
w.set_normalization("density")
assert w.normalization().lower().startswith("density"), (
    f"the density mode was not selected: normalization() reads "
    f"{w.normalization()!r} — every number below would be about some other mode")
for b in w.findChildren(QtWidgets.QPushButton):
    if b.text().strip().lower().startswith("plot"): btn = b; break

# What Details must say, and where. The always-on line names the denominator; the factor
# sentence belongs only where the drawn area is not 1 on a linear axis; the log sentence
# replaces it (not augments it) on a log axis, because there the discrepancy is not a
# factor at all.
ALWAYS = "counts / (n × bin width)"
FACTOR = "Integrating the curve"
LOGX   = "AREA UNDER THE DRAWN CURVE"

bad, rows = False, []
for unit in ("s", "ms", "µs"):
    for log_x in (False, True):
        w._time_unit.setCurrentText(unit)
        w._hist_logx.setChecked(log_x)
        pump(5)
        btn.click(); pump(30)
        curve = [it for it in w._plot.items if isinstance(it, pg.PlotDataItem)][0]
        x, y = curve.getData()               # x AS DRAWN — log10(seconds) when log_x
        ax = w._plot.getAxis("bottom")
        area_drawn = float((y * np.diff(x)).sum()) * float(ax.scale)
        # `xData` stays the raw column, in seconds, in both x modes — this is the area
        # the CSV, the fit and the whole test suite are pinned to, and it must not move.
        area_stored = float((y * np.diff(curve.xData)).sum())
        ylab = w._plot.getAxis("left").labelText.strip()
        details = w.details_text()

        fail = []
        if ylab != "density (1/s)":
            fail.append(f"y label is {ylab!r}, not 'density (1/s)'")
        if "area = 1" in w.normalization():
            fail.append(f"the mode is still named {w.normalization()!r}")
        if ALWAYS not in details or "a density in 1/s" not in details:
            fail.append("Details does not state the denominator")
        if log_x:
            if LOGX not in details:
                fail.append("Details does not disclose the log-x caveat")
            if FACTOR in details:
                fail.append("Details quotes a ×factor on a LOG axis, where the drawn "
                            "area is not the density's integral by any factor")
        elif abs(area_drawn - 1.0) > 1e-6:
            if f"{ax.scale:g}, not 1" not in details:
                fail.append(f"Details does not name the drawn area ({ax.scale:g})")
        elif FACTOR in details:
            fail.append("Details cries wolf: the drawn area IS 1 here")
        if abs(area_stored - 1.0) > 1e-9:
            fail.append(f"the STORED-unit area moved to {area_stored!r} — a drawn "
                        f"number was rescaled, which this design forbids")

        bad = bad or bool(fail)
        rows.append((unit, log_x, ax.labelText, float(ax.scale), ylab,
                     area_drawn, area_stored, fail))

for unit, log_x, xlab, scale, ylab, drawn, stored, fail in rows:
    print(f"time unit = {unit:3s} log x = {int(log_x)}  x label = {xlab!r:22s} "
          f"tick scale = {scale:>9g}  y label = {ylab!r:16s} "
          f"area as drawn = {drawn:<12g} area in stored units = {stored:.12g}")
    for f in fail:
        print(f"      ✗ {f}")

print("REPRODUCED — the figure does not say what its numbers are in" if bad
      else "consistent: y reads 'density (1/s)' in every unit and on both x modes, "
           "Details names the discrepancy where there is one, and the stored-unit "
           "area is still 1")
sys.exit(7 if bad else 0)
