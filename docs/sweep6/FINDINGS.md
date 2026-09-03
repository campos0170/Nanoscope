# Sweep 6 — beta test of the round-4/5 additions (2026-08-31)

**Method.** Every item below was found by DRIVING the shipped widgets offscreen — real mouse
press/release through Qt's delegates, real typing through the item editor, the cyclic GC on
where lifetime was the question — on the owner's own data: the `-700 mV` apo ferritin ABF
(982 detected events), plus the `-600`/`-500` event tables. Nothing here was concluded from
reading `connect()` lines (the round-3 rule). The application object is the shipped one
(`create_app`), so the app-level wheel guard and the real settings behaviour are in force;
preferences are redirected to a temp .ini so the owner's store is never written.

Repro scripts sit beside this file. Each exits **7** when it reproduces and **0** when fixed:

```
QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 docs/sweep6/repro_S6-01_font_combo.py
QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 docs/sweep6/repro_S6-02_undo_rename.py
QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 docs/sweep6/repro_S6-03_density_unit.py
```

---

## Confirmed defects

### S6-03 — "density (area = 1)" has area 1000 on a ms axis, and the figure never says so
**Where** Plots ▸ Histogram ▸ Normalise = *density (area = 1)*, Time unit = ms or µs.
**Measured** The bar heights are a density **per second**, always. `_axis_label_for` rescales
the x ticks for display (`AxisItem.setScale(1000)`), so the drawn curve integrates to
**1000** (µs: 10⁶) over the axis as ticked, under a y axis labelled only `density`.

| Time unit | x label | tick scale | area as drawn |
|---|---|---|---|
| s | `dwell (s)` | 1 | 1 |
| ms | `dwell (ms)` | 1000 | **1000** |
| µs | `dwell (µs)` | 10⁶ | **10⁶** |

**Why it matters** This is the figure that leaves the app: the exported SVG carries
`dwell (ms)` on x, `density` on y, and a peak height of ~160 — a reader integrating it gets
1000. The x-axis machinery exists precisely to stop "a silent factor-of-1000 lie" (its own
words at `plots_panel.py:3381`); the same lie survives on y. The exported CSV is honest — it
is in stored units and its column is named `dwell_s` — so only the PICTURE misleads.
**Fix shape** Either scale the density by the display factor, or label the axis with the unit
it is actually in (`density (1/ms)`), which also makes the mode's name true again.

### S6-02 — Undo of a source rename reverts the plot but not the Sources list
**Where** Plots ▸ double-click a row in Sources ▸ type a name ▸ Enter ▸ Undo.
**Observed** `label_overrides`, the legend list and the canvas legend all revert; the list
row keeps the typed text. The window then shows two names for one source. It survives a
re-Plot; only a Refresh (`set_data`) puts the row back.
**Why it matters** The list row is where the user believes the source's name lives — the
same place they just edited. Undo appearing not to have worked (or to have half-worked) is
the shape of report that comes back as "the rename is broken".
**Note** A systematic pass — change a control, Undo, compare every widget against the config
— found this to be the ONLY stale widget: all 22 other reachable controls (combos, spin
boxes, checkboxes, the title edit) re-sync correctly.

### S6-01 — the Style panel names a font the plot is not drawn in
**Where** Plots ▸ Style ▸ Font, on a fresh window.
**Observed** `StyleSettings.font_family` defaults to `""` = "the application default family",
and the sync shows that as `setCurrentFont(QFont())`. On macOS the application default is
`.AppleSystemUIFont`, a private family `QFontComboBox` does not list, so the combo falls back
to its first entry — **"Academy Engraved LET"** on this machine. Confirmed under the real
cocoa platform, not just offscreen.
**Why it matters** The plot really is drawn in the system UI font (the exported SVG's
`font-family` proves it), so the panel is simply telling the user something false about a
figure they are preparing for publication — and the first alphabetical macOS font is a
decorative engraved face.
**Fix shape** Neither of the two obvious answers works, both measured. Showing the resolved
family is a no-op: `QFontInfo(QFont()).family()` is `.AppleSystemUIFont` too, under cocoa
AND offscreen, so the combo falls back to the same wrong entry. Seeding a real family into
the default changes how every existing figure is drawn and leaves every config already on
disk (which carries `""`) still lying. A sentinel row inside the `QFontComboBox` is worse
than useless — `setCurrentFont()` calls Qt's private `_q_updateModel()`, which rebuilds the
list and destroys the row, after which the next `setCurrentIndex(0)` **emits**
`currentFontChanged("Academy Engraved LET")`, committing the lie. The fix is to replace the
widget with a plain `QComboBox` over the same 181 families plus an **"Application default"**
row, every row carrying its `StyleSettings.font_family` value in `Qt.UserRole` so the
handler reads `itemData` and never `currentText()`.

### S6-04 (minor) — the undo ceiling throws steps away silently
**Where** Plots, after 200+ configuration changes.
**Observed** `_ConfigHistory.dropped` counts discarded steps (10 after 210 changes) and its
own comment says it exists "because a silent bound is indistinguishable from a broken Undo
button" — but **nothing reads it**: the Undo tooltip says "(200 to step back)", the details
pane says nothing, and no label mentions the discard. The counter is asserted only in
`tests/gui/test_plots_undo_templates_json.py:212`.

---

## Checked and CLEAN (driven, not read)

* **Yesterday's SIGSEGV fix holds.** 200 real legend-entry ticks with the GC forced to
  `(5,1,1)`, plus real ticks on the Sources and Features lists: no crash, and
  `_retired_rows` never grew past 0 — the rows are released as designed.
* **Undo/redo** — source ticks, legend ticks, fonts, sizes, colours, title text: all revert
  and re-apply; Ctrl+Z, Ctrl+Y and Ctrl+Shift+Z all reach it; the 200-deep bound holds.
* **KDE (density)** — contours land on the true 2-D mode (x mode 10⁻³, y mode 0.25 recovered
  to 3 decimals, so the grid is not transposed); <5 usable points refused by name;
  zero-variance data refused as degenerate rather than drawn; non-positive values on a log
  axis dropped WITH a count; two sources each contoured against their own peak.
* **Histogram normalisation** — `density` integrates to exactly 1 and `probability per bin`
  sums to exactly 1, in stored units, over dwell/depth/frac_depth.
* **Fitted-values box (#14)** — draws, and its numbers match the Details pane exactly
  (μ = 19.13 ms, λ = 704.6 ms, BIC −8563.43, KS 0.048 on the real −700 mV set).
* **Legend include (Q2)** — unticking the fit row drops it from the canvas and it stays
  dropped across a re-Plot.
* **Source rename → legend** — the real double-click-and-type gesture writes
  `label_overrides`, survives a Refresh, and the `(n = …)` suffix is defeatable by its own
  checkbox.
* **Export truth** — `Export plot data…` writes bin edges + heights + the fitted curve in
  stored units (area = 1 in seconds); `Export JSON…` → `Import JSON…` into a fresh window
  round-trips to an EQUAL `PlotConfig`; `Save image…` writes an SVG whose text matches
  the on-screen labels.
* **Event Inspector on 982 real events** — Accept/Reject flip the shared frame; Accept
  all/Reject all move all 982; ⟵/⟶ pan and return exactly; 120 consecutive real Next
  clicks at 16 ms each with the waveform keeping up; no exceptions.
* **Toolbar widths (the round-3 lesson)** — inspector rows now 1000 px and 673 px, window
  minimum 1016 px; gallery 711 px. All under the 1512 px laptop, all under the 1100 px rule.
* **Event Gallery** — all four sort modes verified NUMERICALLY monotone in their own column
  (not lexicographic); pagination is 40/page over 982; "Reject selected" on a 5-tile
  selection moved exactly 5 rows.
* **Wheel guard (#2b)** — retested on the SHIPPED application object: no visible spin box or
  combo changes under the wheel, focused or not. (An earlier probe that appeared to break it
  was building a bare `QApplication`, which is not the program the owner runs.)
* **Real recording end-to-end** — the −700 mV ABF opens in 0.7 s; channel kinds resolve
  correctly (IN13 dead/raw, IN14 voltage, IN15 current); project save + reopen restores the
  recording and the window title in 1.0 s.

## NOT covered by this pass
Detection/ML pipeline, baseline panel, PSD and I-V panels, measure/calibration, subpopulations,
the packaged `.app`/dmg (which still predates the SIGSEGV fix), endurance beyond a few minutes,
concurrency between instances, and the docs.
