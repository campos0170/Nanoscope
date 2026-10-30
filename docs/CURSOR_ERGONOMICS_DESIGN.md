# Cursor ergonomics for manual annotation — design (draft 3)

Status: implemented on 2026-10-29 (phase 1 and the edge fit; section 14 lists the rest). Written 2026-10-28 against `f9488d8`
(pyqtgraph 0.14.0, Qt 6.11.1). Code references are `file:line` at that commit. Companion
to `docs/MANUAL_EVENT_TABLE_DESIGN.md` (draft 3): every gesture here ends in its one add
door, `AnalysisPanel.add_manual_event(doc_id, channel, start, end)`, and this work starts
after that design's phase 1, which provides the door, the chosen target table and the
shared detection-signal cache. Drafts 1 and 2 were each reviewed against the code with
offscreen probes, and draft 2's edge fit and `C` were implemented and measured on synthetic
traces; section 13 lists what the reviews changed. Nothing was tried on a real Mac or on
real recordings.

## 1. What is tiring today

A manual pass over one recording:

1. Zoom to event scale: a box-zoom drag, the Navigator, or Shift+Up repeatedly.
2. Turn Measure on: Analysis menu (no shortcut), a toggle on a tab inside the Inspector
   dock, or a double-click on the trace.
3. Click the trace so its keys work.
4. Per event: double-click on it (a mask a fifth of the view wide appears), drag the left
   16 px handle onto the step, drag the right handle onto the step, press Ctrl+Shift+M to
   add it to the event table. (Ctrl+M appends to the measurement list and does not check
   for duplicates, `measure_panel.py:2369-2405`.)
5. Per window: Shift+Right, which carries the mask along onto unrelated trace.

That is four gestures per event, two of them pixel-precision drags. Over hours:

- A drag that misses the mask or a handle becomes a **box zoom** (left-drag is box zoom by
  default, `trace_view.py:583`, `main_window.py:1402-1408`), which needs Ctrl+Left to undo.
- After an add the mask stays on the event just added, so every event needs the mask moved
  by hand.
- Any click in a dock or table takes keyboard focus from the trace, and the trace keys
  (Shift+arrows, Home) **stop working silently** (`main_window.py:1370-1389`).
- Jumping to an event from the gallery, inspector or table **re-zooms the trace**
  (`analysis_panel.py:5772-5775`), losing the scan zoom.
- A horizontal two-finger swipe pans the **amplitude**, not time (`trace_view.py:83-105`).
- Zoom keys zoom about the view centre, not the pointer (`pan_scrollbar.py:395-397`).
- Nothing fits an edge to the step, anywhere in the app.

## 2. Summary

1. **Keys under the left hand, live whenever the trace has focus**, with the right hand on
   the trackpad or mouse:

   | Key | Does |
   |---|---|
   | `C` | catch the event under the pointer: find it and fit both edges |
   | `S` / `F` | put the start / finish edge at the pointer, fitted |
   | Shift+`S` / Shift+`F` | put that edge exactly at the pointer |
   | `D` | add the mask (done) |
   | Shift+`D` | add, then go to the next window |
   | `G` / Shift+`G` | next / previous window |
   | `X` | discard the mask |
   | `E` | draw mode on or off |

   They sit on the home row and keep their places on US, Spanish, German and French Mac
   layouts. For a clean event the whole gesture is: point, `C`, `D`.
2. **Draw mode** (`E`) changes one thing: plain left-drag draws a span instead of
   box-zooming. Releasing fits the edges.
3. **Fitting edges** moves each edge to the step in the current near where it was placed,
   through one pure function with gates that were checked against the correlated, 1/f-type
   noise these recordings have, and the row records how its bounds were made.
4. **Focus follows the pointer** onto the trace, and the cursor shape shows whether the keys
   are live, so keys stop dying silently.
5. **Moving**: `G` steps windows, a horizontal swipe pans time, and Shift+Up/Down zoom about
   the pointer.

## 3. The trace keys

### 3.1 Scope

Every key is a `QShortcut` parented to the trace with `WidgetWithChildrenShortcut`, the
pattern of Home and Shift+arrows (`main_window.py:1370-1389`). Probes confirmed that such
shortcuts fire before the focused pyqtgraph ViewBox's own key handler, and do not fire with
a table, spin box or line edit focused, so they cannot steal keys the way a window-wide bare
key would (the S8-KEYS1/KEYS2 family).

The keys are live whenever the trace has focus, in draw mode or not. A key that needs a
mask and finds none says so in the status line ("No mask: press C, or drag in draw mode").
`C`, `S` and `F` turn Measure on if it is off, through `MeasurePanel.set_active`, not the
menu action, which would raise the Inspector's Measure tab (`main_window.py:3584-3595`).

Aliases: Return and Enter (keypad Enter needs its own sequence) add like `D`; Shift+Return
and Shift+Enter like Shift+`D`; Esc cancels a draw in progress, else discards the mask. Esc
never leaves draw mode, since a habitual second Esc must not change what the next drag does.
Shift+Left/Right keep stepping windows as today.

The aliases act only when the trace got focus from a click or a pointer entry
(`Qt.MouseFocusReason`). MainWindow parks focus on the trace when a dock hides or the
Results dock closes (`main_window.py:6659-6664, 6731-6741, 12997-13009`), on the premise,
stated in those docstrings, that the trace takes no keys; a habitual Return or Esc after
closing a dock must not add or discard. `D` and `X` stay live either way, and the
docstrings are updated.

**Cmd slips.** Qt's Ctrl is the Mac's Cmd. A slip with Cmd held turns `D` into Run
detection (`main_window.py:1435-1440`), `G` into the gallery, `E` into Centre selection,
`S` into Save and Shift+`S` into Save As. Only Run detection does harm, and it is one undo
step that keeps hand-made rows (companion design 6). Run detection from the keyboard also
asks first when the target table has curation made in this session. `Q` and `W` are
avoided because Cmd+Q quits and Cmd+W closes the document.

Repeat: `G` and Shift+`G` repeat; every other key is `setAutoRepeat(False)`, so a held `D`
cannot add a run of events.

Why these letters: on the trace every letter is free except `-`, `+`, `=` and Backspace,
which pyqtgraph's focused ViewBox uses for its own zoom history (`ViewBox.py:1397-1413`).
Avoided: `Q` and `W`, where a slip with Cmd held quits or closes the document; `Z` and `Y`,
which swap on German layouts; `[ ]`, which need Option on German and Spanish Macs; bare
Left/Right, which mean previous and next event in the inspector. Trace-only keys need no
entry in `RESERVED_LABEL_HOTKEYS`, because label hotkeys live in the gallery and inspector,
which have their own focus.

### 3.2 Catch the event under the pointer: `C`

`C` finds the event, rather than guessing its width:

1. The pointer must be over the target pane; otherwise "Point at the *recording* trace".
2. Work in the target's signal (the array the add door measures on, `detection_signal`, raw
   for the Manual table), as `dev = polarity × (x − b)` with `b` the contract's
   whole-recording baseline slice, smoothed over three filter rise times. Smoothing runs
   outward from the pointer in blocks, not over the whole view.
3. **Trigger** on the largest smoothed `dev` over the pointer's pixel ±1 px, in samples. At
   scan zoom a pixel covers tens to hundreds of samples, and the single sample under the
   pointer usually misses a short event. It must reach 4 × `noise_rms_pA`; otherwise "No
   event under the pointer". 3σ gave 1.2% false triggers on low-passed noise; 4σ gave none.
4. **Grow** outward from that sample twice, bounded by the target table's neighbouring
   events: to where `dev` falls below half the local peak (the largest smoothed `dev`
   within 3 px of the pointer), which gives the placed edges, and to where it falls below
   2σ, the foot. Each edge's outer fit window is widened to reach the foot plus three rise
   times, because a sub-state shallower than half the peak lies outside the half-peak edge.
5. If growth reaches the edge of the view, refuse: "The event runs past the view: zoom out".
6. Fit both edges (section 5).

If one edge fails its fit, the mask shows that edge marked "not fitted" and keeps the good
edge as a preview. A single `S` or `F` on the failed edge completes it; `D` on an unfitted
edge asks for a confirming second press. Unfitted edges are recorded as `placed`, never
`drawn`.

Measured by the second review on synthetic traces (`truth.synth_trace` default noise,
white plus 30% 1/f, erf ramps), pointer at a random interior point, at 50 and 250 kHz:
single events of 2 to 200 ms at 10 to 20σ caught in 97 to 100% of trials; 0.2 ms events in
82 to 100% at 50 kHz; sub-state events in 88 to 94% (sharp edges) and 97 to 99% (1 ms
edges); two sharp-edged events 0.3 ms apart kept separate in 93 to 100%; no false catch in
400 trials with the pointer on baseline, and at most 0.7% under strong drift, none fitted.

Known limits: an event longer than about the baseline's rough window (0.5 s,
`baseline.py:1223-1224`) is partly absorbed into the baseline, and a weak event during fast
drift may not trigger (200 ms 5σ events went untriggered in 28 to 63%); two events with
soft edges closer than their ramps merge, because the trace never returns to baseline
between them. Cost: 0.02 to 1.3 ms at up to 100 samples per pixel.

In draw mode a double-click does what `C` does at that x, so the mouse alone can catch an
event. Outside draw mode a double-click keeps its current meaning (`trace_view.py:107-117`).

### 3.3 Put an edge at the pointer: `S`, `F`

`S` moves the start edge to the pointer's x and fits it there; `F` the finish edge. With
Shift they place the edge exactly at the pointer and record it as `drawn`. With no mask yet,
`S` then `F` builds one. This replaces nudging for the common case: an edge that is wrong is
fixed by pointing at where it belongs, at any zoom, instead of tapping a key per sample. Edge
identity comes from `getRegion()` order, not from the line objects, which swap
(`LinearRegionItem` `swapMode='sort'`), and an edge cannot be moved past the other.

### 3.4 After an add

- The mask is **discarded**, so the next event starts clean and `G` does not carry a mask
  over unrelated trace. This needs a new state, Measure on with no mask (section 6.3).
- The view does not move (the XI-1 rule, `test_ui_sweep_fixes.py:135-152`); Shift+`D` is the
  explicit "add and next".
- The new event flashes briefly, the gallery and inspector land on it quietly (companion
  design 4.3), and the status line says "Added event 318 to Schmitt. ⌘Z undoes".
- A refused add keeps the mask and gives the sentence.

## 4. Draw mode

### 4.1 Turning it on and off

- `E` on the trace, or Analysis, Draw mode (checkable, Ctrl+Shift+E; free, since
  `main_window.py` binds Ctrl+Shift only with S, Z, R, M and F). The menu item is enabled on
  the same broad gate as Ctrl+M (`main_window.py:3441-3453`) and names the first blocker.
- Turning it on does not place a mask.
- It stays on across rebuilds, view-mode changes and table switches while the target pane
  exists, since the Measure region already reattaches in those cases
  (`trace_view.py:670-693, 1823-1836`). It drops only when the target pane is gone, and the
  status line says why.

### 4.2 What it changes

| Gesture | Draw mode off | Draw mode on |
|---|---|---|
| left-drag on empty trace | box zoom, as today | draws a span on the target pane |
| left-drag on the mask body | moves the mask | draws a new span |
| left-drag on a mask edge | resizes it | resizes it |
| Option-drag | draws a span | box zoom |
| double-click | places a 20% mask, as today | catches the event (as `C`) |
| left-drag on another pane | as today | nothing; "Draw on the *recording* pane" |

Option inverts the drag in both states, so a one-off span needs no mode and a quick dive
into an event needs no mode change. Cmd (Qt's Ctrl) and physical Control are avoided: on a
Mac, Control+click becomes a right click.

### 4.3 How drawing works

- **Hook**: a `TraceViewBox.mouseDragEvent` override takes a left drag with `axis is None`
  when draw mode is on (or Option is held outside it) and calls `super()` otherwise. It
  passes the phase and data-x to a callback injected like `on_double_click`
  (`trace_view.py:79-81, 737`). Probes: the scene delivers start, moves and finish to such an
  override, and accepting without `super()` stops the box zoom.
- **The mask forwards body drags.** pyqtgraph gives a drag to the item that claimed it at the
  last hover, and if none did, to the first item near the press that accepts it, in z order
  (`GraphicsScene.py:303-340`). The movable mask (z 8) accepts any left drag
  (`LinearRegionItem.py:308-311`), so skipping its hover claim is not enough: probes showed
  the mask still moving, and a drag after pressing `E` with the pointer resting on the mask
  going nowhere. Instead a `LinearRegionItem` subclass overrides `mouseDragEvent`: in draw
  mode, or with Option held at the start, it accepts the drag and forwards its phase and x to
  the same callback. Its two edge lines are separate items and keep resizing as today
  (probed).
- **Preview without recomputing every frame.** `set_measure_region` emits
  `sigMeasureFinished`, and `MeasurePanel` recomputes at once on it
  (`trace_view.py:1636-1647`, `measure_panel.py:2029-2036`), about 8 ms at 200,000 samples,
  per frame. New `TraceView.preview_measure_region(lo, hi)` moves the two lines with
  `blockLineSignal`, as `LinearRegionItem.mouseDragEvent` itself does
  (`LinearRegionItem.py:318-331`), and emits only `sigMeasureChanged`, so the existing 60 ms
  debounce applies. `settle_measure_region()` emits `sigMeasureFinished` once, on release and
  after a fit.
- **Mapping**: x is clamped to `measure_cursor_extent(target)` and converted only through
  the target's axis, the rule `measure_bounds` uses (`trace_view.py:1608, 1838-1867`), never
  through the pane's first layer (`trace_view.py:1930-1944`).
- **Too narrow**: a release under three pixels is a click. A span shorter than
  2 × `min_len` + 2 samples cannot be fitted and is shown as drawn.
- **Cancel**: right-click during the draw, Esc, or a rebuild. A right-click reaches
  `TraceViewBox.mouseClickEvent` while the ViewBox owns the drag; for a draw forwarded from
  the mask body the mask owns it, and its base `mouseClickEvent` acts only on a move it
  started (`LinearRegionItem.py:333-340`), so the subclass also overrides
  `mouseClickEvent` to cancel a forwarded draw. The start x is cached at the start, because
  after a right-click the drag's own start position is reset; later frames are ignored until
  release; and each gesture carries a token that a rebuild invalidates, since the scene keeps
  sending to the removed ViewBox until release (`GraphicsScene.py:343-355`).
- **On release**: fit both edges (section 5) and settle.

## 5. Fitting edges

Called "fit", since the Measure tab already has "Snap to view" and "Snap to selected event"
buttons that place the cursors (`measure_panel.py:1121-1128`).

### 5.1 The function

A new pure module `nanoscope/analysis/edge_fit.py`, with no GUI:

```python
fit_edges(signal, fs, start, end, *, polarity, noise_rms, tau, lo, hi,
          edges=("start", "end"), method="step", rule=None, px_samples=None) -> EdgeFit
# EdgeFit: start, end; per edge: fitted (bool), z, step, reason
```

- **Signal**: the array the add door measures on (`detection_signal`, raw for Manual), from
  the shared cache (`analysis_panel.py:4934`). No silent raw fallback.
- **Polarity** from the target's provenance (`measured_polarity` for Manual,
  `baseline.py:4903`), never from the window, which flips on drift.
- **Noise and correlation**: `noise_rms` is the target's `noise_rms_pA`. `tau` is the
  open-pore autocorrelation time from
  `cusum.open_pore_autocorr_time(residual, starts, ends, n_scale=L, pad=L)`
  (`cusum.py:360-413`), floored at `filter_autocorr_time` when the provenance has a low-pass,
  as CUSUM's own caller does (`cusum.py:1069-1091`). The residual is the signal minus the
  contract's baseline. With no rows yet, the function falls back to the whole residual,
  events included, which inflates tau, so an empty Manual table passes the rows of the
  recording's selected detector table, or pseudo-events where |dev| exceeds 4σ. It is
  computed in the contract worker (companion design 5.4, which gains it; 42 ms at 750,000
  samples, about 1 s at 15 M), never on a key press, and written to `fit_json`. For scale,
  the code records the owner's high-passed open-pore tau at 11.3 and 4.8 on two files, and
  90.5 and 57.1 for the whole residual (`cusum.py:386-391`).
- **Edge rise**: the target table's median `rise_s` when it has rows, otherwise a first-pass
  10–90% estimate around a first split, floored at the filter rise time, and written to
  `fit_json`. With only the filter rise (about 34 µs at 10 kHz), soft edges and sub-states
  fail. The two-pass estimate works at 50 kHz but not at 250 kHz on multi-step edges.
- **Window** per edge: W = max(about 12 px of the current zoom in samples, 3 × edge rise),
  capped at 0.45 × the span, and the cap wins. It is clipped to `lo` and `hi`, the
  neighbouring events of the target table, and to the other edge.

### 5.2 The step fit (default)

The least-squares split of `cusum._best_split` (`cusum.py:653`), made **directed and
outermost**. Each rule below answers a measured failure.

1. **Directed**: candidates are only splits whose step has the right sign in `dev`, up for a
   start and down for an end. Undirected, the split takes the largest step of either sign, and
   a sub-state step runs the same way as the real edge: the end landed on the inner sub-state
   step in 185 to 200 of 200 trials (a 20 ms event with a 2 ms tail at 50 kHz).
2. **Cut at an opposite step**: before testing the outer side, look beyond the split for the
   strongest opposite-sign step. If it passes the gates of 4 and 5, judged only on the samples
   between, cut the window there and split again, up to three times, and clip every later
   statistic at the cut. Without this, a deeper neighbour the table does not hold, filling the
   outer half of the end window, leaves no down-step candidate at all: the end was fitted in
   only 6 to 51% of trials.
3. **Outer side at baseline**: skip a guard of one rise time past the split, then average
   max(32 samples, one rise time), stopping at any cut. The mean must be within
   max(3σ × sqrt(tau / n), 1.5σ) of `b_out`, the median of max(64, W) samples beyond the search
   window, clipped to neighbouring events; the contract's baseline track is used only when the
   window was cut at an opposite step. If the test fails, the outer segment is split again; if
   it still fails, the window is grown outward once. Testing right next to the split, or
   against the whole-recording track, failed on soft edges (the ramp tail), on 1/f drift, and
   on close neighbours; a window of 4 × rise averaged a short sub-state tail with the baseline
   behind it and cut the tail off.
4. **Significance judged locally**: z over at most L = max(64, 4 × edge rise) samples on each
   side of the split, clipped at a cut and at the other edge, divided by `sqrt(tau)`, at least
   6. The z over the whole window grows with the window on correlated noise: on the default
   pink-noise synthetic trace at 50 kHz it passed on pure noise in 81% of 2,500-sample windows
   and 100% of 25,000-sample windows. The local z's false rate was 0.5 to 1.3%.
5. **Size**: the local step must be at least max(3σ, 0.15 × the interior median `dev`). A 0.3
   factor rejected the outer step of every sub-state shallower than 30% of the depth. Steps
   under 3σ are never fitted, and such rows stay `placed`.
6. **Border**: a split within max(`min_len` + 1, one rise time) of the **outer** window edge is
   retried once with the window grown outward; the test is skipped when the window already
   reaches a neighbour. Testing both window edges with a tolerance of 3 rise times rejected every
   split whenever W came from the rise term, and every short event.

Measured by the second review on `truth.synth_trace` default noise,
100 events per case, edges placed within ±3 px, at 1, 10 and 100 samples per pixel, at 50 and
250 kHz: plain 1 ms edges 99 to 100%, sharp edges 100%, next to a deeper neighbour 1.5 ms away
100%, sub-state ends 95 to 99% and starts 95 to 98%. No false fit in 1,800 spans drawn on pure
noise (2 to 300 ms, every zoom, both rates). Edges placed further out than W are left unfitted,
almost never fitted wrongly (0 to 1%). On a soft edge the fit lands mid-ramp, within half a rise
time. Cost for both edges: 0.04 to 0.15 ms at up to 100 samples per pixel, up to about 1.3 ms
on a 200 ms event at a full 750,000-sample view.

**One rule per row.** A fit on both edges (on release, `C`) is all or nothing, so one dwell
never silently mixes two edge definitions: if one edge fails, it is shown as "not fitted" next
to the good edge, and the row is not added until that edge is fitted with `S` or `F`, or set
exactly with Shift. Those keys record each edge's source separately. With the rules above, a
good edge was held back by a failed one in at most 2% of rows (6% next to a deeper neighbour
at 100 samples per pixel with the rise unknown).

### 5.3 The table rule (phase 2)

For a hand-made row in a detector table, the edges can instead follow that table's own rule,
so the row has the same dwell definition as its neighbours. Done naively, this does not
reproduce the detector: in probes, measuring against flank medians gave both edges exact on
only 28% and 40% of 15-sigma events. So the rule runs **the detector's pipeline on a window**:

- the sliced whole-recording baseline memo the door already warms, never flank medians (with
  it, 99 of 99 15-sigma events reproduced exactly);
- the stored `merge_gap_s` (`detect.py:993-1003, 1072`), `group_starts_by_gap`,
  `split_low_occupancy_groups` and `min_dwell_s`;
- on a high-pass-localised table, the candidate-domain signal, noise and baseline
  (`detection_noise_rms_pA`, `detection_baseline`, `modules.py:1559-1672`), not the
  characterisation ones;
- CUSUM ends converted from inclusive to half-open (`detect_cusum.py:714`);
- entry and exit both at `k_out × noise`, so an event below `k_in`, which is the kind most
  often added by hand, still gets bounds; `k_in` rejects 98.5% of candidates
  (`detect.py:962-968`).

### 5.4 Why the choice of rule matters

On a filtered trace the Schmitt exit level sits before mid-ramp by about erfinv(1 − 4/SNR)
standard deviations of the edge: about 0.43 × the 10–90% rise at SNR 15 and 0.10 × at SNR 5.
Schmitt rows are therefore longer than step-fitted rows of the same event, by an amount that
depends on depth. A table mixing the two rules has a dwell distribution mixing two
definitions. Decision 1.

### 5.5 What the row records

New additive extras, written by the door after `recharacterize_span`, never in `SCHEMA` and
never in `orig_start_idx`/`orig_end_idx`, which mean detector bounds only
(`events.py:694-700`):

| Column | Value |
|---|---|
| `bounds_source` | per edge, for example `start=fit:step;end=placed`. Values: `drawn`, `placed`, `fit:step`, `fit:schmitt`, `fit:cusum`, `edited`, `stale` |
| `drawn_start_idx`, `drawn_end_idx` | `Int64`, the bounds before fitting, so a fit can be redone under another rule |
| `fit_json` | method and version, W, gate thresholds, `tau`, per-edge z and step, the signal key (`detection_signal_key`), baseline source, noise and fs, like `amplitude_provenance_json` (`event_amplitude.py:27-37`) |

Keeping them true:

- Merge takes the start side from the first fragment and the end side from the last, the one
  that ends last (`manual_events.merged_edge_cells`); a side whose fragment recorded nothing, a
  detected row, records nothing.
- An inspector mask drag or Revert sets that edge to `edited`; `_write_measured_row` writes
  only measured columns today (`event_inspector.py:3588-3603`).
- On the companion design's re-run carry-over (6) and Copy to (7.3), a row whose edge source is
  a table rule is re-fitted from `drawn_*` under the target's rule, or marked `stale`.
- A `bounds_source` selector joins the include switch in Distributions, fits and export
  curation keys (companion design 7.2), and the Measure tab shows the fit success rate for the
  current table. Filtering on `bounds_source` is not neutral: fits fail more often on long,
  multi-level and drifting events.

### 5.6 Rise and fall times

`rise_s` and `fall_s` look for the 10% and 90% points within a fixed 64-sample pad
(`characterize.py:74, 1243-1246`). A fitted start sits mid-ramp, so slow edges lose `rise_s`.
The owner records at 50 kHz (749,651 samples over 15 s, `detect.py:1836-1838`), where 64
samples are 1.28 ms and the median 1.01 ms rise stays finite; NaN appears only around a 2.6 ms
rise. At 250 kHz it would be common. Proposed: pad = max(64 samples, 1.28 ms), clipped to
neighbours as today, which is bit-identical at 50 kHz and only changes rows that are NaN today
or recorded faster.

## 6. Focus, feedback and state

### 6.1 Focus follows the pointer

When the pointer enters the target pane and the main window is active, the trace takes focus,
but only from a holder that has no keys of its own: nothing focused, the trace itself, a
button, a scroll bar, a closed combo box or a dock title bar. It never takes focus from the
docked Event inspector (`insp.isAncestorOf(focus)`), any table, tree or list
(`QAbstractItemView`), or a text or spin field. Those own bare keys: the inspector's Left,
Right, A, R, Space, T, digits and label hotkeys, the Results table's j, k, a, r and f, the Files
tree's arrows. A probe showed a pointer crossing silently killing them, and worse, a label
hotkey on `d` would become an add after the crossing. Over those holders the arrow cursor says
that a click arms the trace keys. An open popup is safe: pointer moves do not reach the scene
while it is open.

Entry is detected by a `TraceViewBox.hoverEvent` override (`isEnter` on the target ViewBox,
re-armed when a rebuild creates new ViewBoxes); `pg.ViewBox` has no `hoverEvent` of its own,
and pointer moves alone miss leaving the widget. The `setFocus` uses `Qt.MouseFocusReason`, which
section 3.1's alias rule reads.

### 6.2 What the user sees

- **Cursor**: a cross on the target pane's ViewBox when the keys are live, a plain arrow when
  focus is elsewhere. It is set on the target ViewBox, not the viewport, so other panes and the
  axes keep their own cursor (probed), and it is re-applied when a rebuild creates new
  ViewBoxes.
- **Label**: a small scene item in the bottom-left of the target pane, transparent to the
  mouse, like a legend: "→ Manual · 37 hand-made · 2 in view · draw". The crosshair readout
  owns the top edge (`trace_view.py:1925-1928`). The target comes from the companion design's
  chosen table and updates live.

### 6.3 Measure on with no mask

Today there is no such state: clearing the mask turns Measure off
(`measure_panel.py:1791-1806`), and turning it on seeds a mask over the middle fifth of the
view (`trace_view.py:1583-1606, 1716`). A missing stored range already means "seed a mask", so
an empty state cannot be "no range". A probe of that reading showed the first rebuild bringing a
seeded mask back, `set_measure_region` doing nothing with no item (so Snap to view, Snap to
selected event, Go to measurement and the plain double-click place nothing), the Navigator band
staying drawn, and Ctrl+M saving the event just added a second time from the stale readout.

So the state is an explicit flag, `_measure_empty`:

- `sigMeasureActiveChanged` is computed from `_measure_active` and whether the target pane
  exists, not from whether an item exists, so an empty mask does not uncheck Measure
  (`trace_view.py:1606`, `measure_panel.py:1873-1877`).
- `_create_measure_region` and `_reattach_measure` create nothing while empty, and the
  time-base path keeps the flag (`trace_view.py:640-656, 1787, 1823-1836`).
- `set_measure_region`, a draw start, `C`, `S` and `F` recreate the item at the range they are
  given, with no seed and no seeded `sigMeasureFinished` (`_create_measure_region` gains an
  initial range).
- A new `sigMeasureCleared` tells the Navigator to drop its band and MeasurePanel to drop
  `_last_stats` and its readout. `add_current` reads `measure_bounds()` afresh instead of
  trusting `_last_stats` (`measure_panel.py:2383-2392, 2138-2155`).
- Turning Measure on from `C`, `S` or `F` goes through `MeasurePanel.set_active`, whose toggle
  path (`measure_panel.py:1853-1862`) passes `seed=False`.
- Every reader of `measure_bounds`, `measure_x_range`, `measure_target` and
  `sigMeasureActiveChanged` is audited in phase 1: `main_window.py:2344, 3556, 3629, 3632-3653,
  3709, 6297`; `analysis_panel.py:1987, 4793, 5161`; `measure_panel.py:2073, 2420, 3108`;
  `navigator.py:117-122, 413-421`; `pan_scrollbar.py:178-182, 361-363`.

## 7. Moving along the trace

- **`G`, Shift+`G`**: next and previous window, the existing 90% step with 10% overlap at the
  same zoom, clamped to the target (`pan_scrollbar.py:206-220, 326-368`). With no mask, nothing
  is carried.
- **Zoom about the pointer**: Shift+Up/Down are wired straight to `PanScrollBar.zoom_in` and
  `zoom_out`, whose `_zoom` always uses the centre (`main_window.py:1381-1387`,
  `pan_scrollbar.py:395-397`). `_zoom` gains an anchor, `x0 = px − (px − lo) × m`, then the
  usual clamp. The shortcut passes the pointer's x from `QCursor.pos()` only when the pointer is
  inside a pane's ViewBox and `QApplication.widgetAt` is the trace viewport (over the left axis
  the mapping still returns an x outside the view, and a covering window still maps into the
  trace); otherwise the centre. The Scan strip buttons keep the centre. Near the ends of the
  recording the clamp moves the window, so the pointer is not held in place there. Option-drag in
  draw mode is a box zoom (4.2). Zoom Back already restores x and the y autorange
  (`trace_view.py:1110-1118`), so "dive in, add, Ctrl+Left" works.
- **Horizontal swipe pans time**, with the axis locked per gesture. Qt reports per event
  whichever component dominates, so a diagonal swipe alternates between time and amplitude. The
  first event of a trackpad gesture (`ScrollBegin`) carries no movement, and today's handler
  returns early on it (`trace_view.py:83-92`), so the lock is reset there, before that return,
  and the axis is chosen once either accumulated `pixelDelta` component reaches 6 px. The pan then
  uses that component of `pixelDelta`, not `delta()`, which reports whichever component dominates
  each event. A mouse wheel (no scroll phase, zero `pixelDelta`) decides per event from
  `orientation()`. Momentum continues on the locked axis; on macOS the first momentum event may
  arrive as an update, which offscreen probes cannot check. Shift plus a mouse wheel arrives as
  horizontal on a Mac, so mouse users get a time pan too. A vertical wheel still pans amplitude.
  The fake wheel event in `tests/test_zoom_navigation.py:129-135` has no `orientation()` and needs
  one; a settled swipe adds a Zoom Back entry, which falls under decision 5.
- **Zoom history**: every scan step is an entry today (`trace_view.py:1095-1108`), so after a
  pass Zoom Back walks back window by window. A scan step could instead replace the previous
  entry when that entry was also a scan step, tagged at its source in `PanScrollBar._scan`.
  The tooltip promises "reject an event and keep scanning" (`main_window.py:1340`), which
  per-step history already does, so this is decision 5, not a fix.

## 8. Editing an existing event's edges (phase 2)

The inspector already has one bounds writer, `_apply_edited_bounds`
(`event_inspector.py:3375-3474`), with neighbour clamping, the stale-rate refusal, the
re-measure, amplitude invalidation and one "Resize event mask" undo step, and no keyboard path.

- **Keys**: `S` and `F` set the current event's start or finish edge at the pointer, fitted;
  Shift for exact; `,` and `.` nudge the edge nearer the pointer by one pixel, accelerating while
  held. They go through the inspector's registry (`event_inspector.py:2186-2218`) and into
  `RESERVED_LABEL_HOTKEYS` (`labels.py:79`) with `<` and `>`; a legacy label on one of them loads
  unarmed.
- **One write per burst**: a nudge previews inside the `_syncing` fence, in sample indices from
  the frame (`event_inspector.py:3370-3372`). The pending burst commits through
  `_apply_edited_bounds` after 600 ms idle, and synchronously at the start of every other
  inspector handler (previous and next, decisions, labels, undo, redo, Revert, a mouse drag) and
  on hide or close. It is discarded if the frame or the row's `event_id` changed since the burst
  began, so an undo in the middle cannot be overwritten. It keeps the existing "Resize event
  mask" label, which tests pin (`test_undo_events.py:304, 516`). Without one write per burst,
  thirty taps would evict the table's whole undo history (`history.py:135`).
- **Nothing moves the view**: `preserve_view` in the inspector and no re-zoom of the main trace
  (`event_inspector.py:3470`, `analysis_panel.py:5606-5607`).
- **Refusals speak**: on Reconciled the keys say `_RECONCILED_EDIT_TIP`, where today the lock is
  a silent return (`event_inspector.py:3356, 3394-3396`); stale-rate and conditioning checks run
  once per burst, since a conditioning failure opens a modal (`event_inspector.py:3437-3462`).
- **A clamp is visible**: fitting searches inside `_neighbour_bounds`, and an edge stopped by a
  neighbour says so, naming a rejected neighbour when that is the cause.
- **`bounds_source`** is updated per edge (5.5).

## 9. Session aids

- The label shows the hand-made count for the target and in the view.
- "⌘Z undoes" after each add, and the new event flashes.
- **Resume a pass** (phase 2): the last window per (recording, channel) is kept in
  `workspace.extra`, and Analysis, Resume pass returns to it. Nothing saves the view today.
- Crash-recovery autosave already runs (`main_window.py:10721`); the label can say "saved" when
  a snapshot lands.
- Later: a distinct outline for hand-made rows on the trace, a Navigator band of windows already
  visited, and "not an event" regions (`detect_events` already takes `exclude_regions`).

## 10. Tests

- Trace keys fire with the trace focused and not with the Results table, Files tree, a spin box
  or a line edit focused; copy `test_manual_scan_navigation.py:665-701`. Keypad Enter adds.
- Focus follows the pointer only from holders without keys: after the pointer crosses the trace,
  A in the docked inspector still accepts, a label hotkey `d` labels the event and adds nothing,
  and `f` in the Results table still flags the row. Return and Esc do nothing after focus parking.
- The empty-mask state: after `D`, Measure stays on with no mask through a rebuild, a view-mode
  change and a rate declaration; Snap to view and Go to measurement place a mask; the Navigator
  band clears; Ctrl+M after `D` saves nothing twice.
- Draw mode: a drag on empty trace, and on the mask body, draws; edge handles resize; Option
  inverts; another pane does nothing; right-click and Esc cancel; a rebuild mid-draw cancels;
  a 20-frame draw causes at most two Measure recomputes; the mode survives a view-mode change.
- `C`: catches an event at the pointer, refuses on baseline and off the target pane, respects
  neighbours, and two close events stay two; `S`/`F` with and without Shift; `D` adds without
  moving the view and leaves Measure on with no mask; Shift+`D` adds and steps.
- `fit_edges`, on `truth.synth_trace` with its default pink noise at 50 and 250 kHz, both
  polarities, at 1, 10 and 100 samples per pixel: the measured rates of section 5.2 as floors;
  sub-state onsets and ends found at the outer step; a deeper neighbour not taken; a 250 kHz
  sub-state case with the rise unknown; no fit on pure pink noise and on hardware-like filtered
  noise; spans drawn 30% inside and outside; 300 to 500 ms events; a drift case; 2-to-10-sample
  events left as placed; NaN in the window; determinism; all-or-nothing per row. The second
  review's probe scripts (`edgefit.py`, `probe_fit.py`, `probe_catch*.py`) lived in a session
  scratch folder and are not in the repo; the tests rebuild them.
- Right-click cancels a draw started on empty trace and one started on the mask body.
- Table rule (phase 2): reproduces the detector's bounds using its own baseline, with
  `merge_gap_s` > 0, on a high-pass-localised table, and on a CUSUM table.
- `bounds_source` through merge, inspector drag, Revert, re-run carry and Copy to.
- `rise_s` finite on a fitted 0.6 ms edge at 250 kHz with the new pad; unchanged values at 50 kHz.
- Horizontal wheel pans time with the axis locked per gesture; vertical still pans amplitude.
- Inspector bursts (phase 2): one undo step, flushed by every other handler, discarded after an
  undo, no view change.

## 11. Phases

1. **The trace loop**, after the companion design's phase 1: the trace keys and their aliases,
   draw mode with Option inversion, the drag forwarding and preview/settle path, the empty-mask
   state and its consumer audit, `C`/`S`/`F`, the step fit with `bounds_source`, `drawn_*` and
   `fit_json`, focus-follows-pointer, the cursor and label, zoom about the pointer, the
   horizontal swipe, the rise-pad change.
2. **Edits and comfort**: the table rule, inspector `S`/`F` and pixel nudges with one write per
   burst, the `bounds_source` selector in consumers, Resume pass, scan-history coalescing if
   chosen, pinch zoom about the pointer on macOS (`QNativeGestureEvent`).
3. **Later, if wanted**: a candidate suggester marking threshold crossings no table holds
   (`detect.detect_fixed_threshold`, `detect.py:1525`, filtered with `_events_overlapping`), edge
   editing of a selected event on the main trace through the inspector's writer, drawing inside
   the inspector, a coverage band, "not an event" regions.

## 12. Decisions for the owner

1. **Fit rule on a detector table**: the table's own rule, so hand-made rows match their
   neighbours (proposed, phase 2; step fit until then, recorded as such), or the step fit
   everywhere, which is closer to the true edge and the same across tables?
2. **The key cluster** S D F G C X E: try it on your Mac, with your layout, before it is
   final?
3. **Trace keys live outside draw mode** (proposed): `E` then only decides what a plain drag
   does.
4. **Horizontal swipe pans time** (proposed)?
5. **Zoom history**: one entry per scan step, as today, or one per run of steps?
6. **Resume pass** (proposed)?
7. **Candidate suggester**: wanted at all?

Settled by the review: the add stays an explicit key press (`D`), not a mouse release, because
releasing would turn every bad fit into an undo step and a forgotten draw mode into stored events.

## 13. What the review changed

| Draft 1 | Draft 2 | Why |
|---|---|---|
| mask cleared after an add; mode drops when Measure goes off | a Measure-on, no-mask state; the mode drops only when its pane is gone | clearing the mask turns Measure off (`measure_panel.py:1791-1806`), so the mode would have ended after every add |
| mask skips its hover claim | mask forwards body drags | the scene falls back to the first item that accepts, which is the mask (`GraphicsScene.py:324-340`); probed |
| `set_measure_region` per frame, 150 ms nudge coalescer | preview and settle methods; the existing 60 ms debounce | every `set_measure_region` emits Finished, which recomputes at once (`measure_panel.py:2029-2036`) |
| `C` = a mask 5% of the view | `C` finds the event at the pointer | a fixed width only worked for events 2.5% to 7.5% of the view wide |
| `[ ] , .` nudges, `S`, Enter, mode-only keys | the S D F G C X E cluster, live whenever the trace has focus; nudges moved to the inspector in phase 2 | the per-event and per-window keys sat under the mouse hand, `[ ]` need Option on German and Spanish Macs, a one-sample nudge is 0.1 px at scan zoom, and mode-only keys were silently dead |
| z over the whole window, tau 1 on raw | a local z with the open-pore tau, a size gate, directed and outermost splits | on pink noise the old gate passed on noise in 84% to 100% of long windows; the step fit cut off sub-states |
| table rule on flank medians | the detector's pipeline on a window | flank medians reproduced only 28% to 40% of Schmitt bounds |
| `rise_s` NaN as a decision | a pad of max(64 samples, 1.28 ms) | the owner records at 50 kHz, where the old claim does not apply, and the new pad is identical there |
| Esc chain leaves the mode | Esc never leaves the mode | a habitual second Esc changed what the next drag did |
| banner as a QWidget with toggles, cursor on the viewport | a mouse-transparent label in the pane, cursor on the target ViewBox | the widget blocked drags beneath it, and the viewport cursor covered every pane |

A second review checked draft 2's mechanisms with probes and implemented its fit and `C`
literally on synthetic traces. What it changed:

| Draft 2 as first written | Now | Why |
|---|---|---|
| fit numbers 99%, 100%, 86 to 87% | numbers measured on the algorithm specified | the quoted numbers came from a different statistic; the specified algorithm, built literally, scored 6 to 89% |
| border test on both window edges, tolerance 3 rise times | outer edge only, tolerance one rise time | every split counted as near the border when W came from the rise term |
| outer side "within a local z bound" of a flank baseline | guard, short window, a floor of 1.5σ, a baseline from beyond the window | every reading of the old rule failed on soft edges, 1/f drift or close neighbours |
| no handling of a deeper outer neighbour | cut at an opposite step, up to three times | the end was fitted in 6 to 51% of trials |
| size gate 0.3 × median | 0.15 × median | 0.3 rejected sub-states shallower than 30% of the depth |
| "measured edge rise", tau "measured once" | named sources for both, stamped in `fit_json` | neither had a source; tau was misquoted and depends on scale and on the rows given |
| `C` reads the sample under the pointer, grows to half the peak | the pixel maximum; half-peak edges plus a 2σ foot for the fit window; refuse at the view edge | 0.2 ms events were caught in 0% of trials; sub-states were out of reach |
| focus follows the pointer except from text fields | only from holders without keys | it took the docked inspector's and tables' keys, and turned a label hotkey `d` into an add |
| empty mask as a missing range | an explicit flag, a clear signal, a fresh read in `add_current` | rebuilds re-seeded a mask, placement did nothing, Ctrl+M saved a duplicate |
| swipe axis chosen at `ScrollBegin` | chosen from accumulated `pixelDelta` | `ScrollBegin` carries no movement |
| Cmd slips: Q and W only | the full slip map; Run detection from the keyboard asks when curation is at stake | Cmd+D runs detection |
| Return and Esc live whenever the trace has focus | only after a click or pointer entry | focus parking would arm them for a user who never pointed at the trace |

## 14. Implementation status (2026-10-29)

Built, with tests (`tests/analysis/test_edge_fit.py`, `tests/analysis/test_characterize_rise_pad.py`,
`tests/gui/test_trace_keys.py`, `tests/gui/test_trace_draw_mode.py`, `test_trace_empty_mask.py`,
`test_trace_focus_cursor_label.py`, `test_trace_moving.py`, `test_trace_measure_panel_hooks.py`,
`test_trace_rig.py`): the keys C, S, F, Shift+S, Shift+F, D, Shift+D, G, Shift+G, X, E and the
Return / Enter / Esc aliases gated on a mouse-given focus (`gui/trace_keys.py`); draw mode with
Option inversion and the fit on release; the empty-mask state; the directed step fit and `C`
(`analysis/edge_fit.py`), with `bounds_source` per edge on the row; focus that follows the pointer
from keyless holders; the label on the trace and the caption beside the Add button; time-axis
swipes and wheel notches; zoom about the pointer; and the rise / fall pad of
`max(64 samples, 1.28 ms)`.

Different from this draft:

- **The open-pore tau is its own estimator** (`edge_fit.estimate_open_pore_tau`): the existing
  `cusum.open_pore_autocorr_time` pairs its boundaries so that it measures the stretches that contain
  events, not the gaps between them (white noise with 90 events reads 4.5 where the gaps give 1.0).
  That function was left as it is, because it feeds CUSUM's tau; the finding is for a separate change.
  The fit floors its tau at the detector's low-pass autocorrelation time when the table has one.
- **Rule 4's statistics are not clipped to the fit window.** Clipping them takes the wrong-split
  rate at 30 % window error from 4.6 % to 0 and leaves every 3 px result identical, but costs 6 to
  11 points for soft sub-states of unknown rise at one sample per pixel. Not applied.
- **Rates measured on synthetic traces** (100 events per case): a plain 1 ms event 99 to 100 %
  within 3 px at 50 and 250 kHz and 1 to 100 samples per pixel; sub-state ends 95 to 100 %; `C` on
  2 to 200 ms events at 10 to 20 sigma 96 to 100 % of the unrefused; pure noise never made a fitted
  row. At 250 kHz soft sub-states of unknown rise fail (0 to 7 %).
- **Fixed after the 2026-10-29 bug sweep** (`tests/gui/test_s9_*.py`): Ctrl+M and the Add button add
  through `TraceKeyController.add`, so every add carries the edge record, asks twice for an unfitted
  edge and clears the mask; a settled mask the controller did not place is recorded as drawn, so S / F
  move one edge of it, Measure's own seed excepted until it is moved; an add with one edge unset is
  refused and the preview says so first (`MeasurePanel.set_add_hold`); an edge at the recording's end
  stays at `n`; `fit_json` keeps each edge's own fit; the Event inspector's dock, not the Inspector
  panel, is the trace's key owner; and the trace has a key target with Measure off. The second press
  confirms only the mask it was asked about, and a table that takes no hand-made row is refused
  (`AnalysisPanel.refuse_add_for_the_table`) before any question about the mask.
- **Not built:** the Inspector's S / F and nudge keys with one write per burst, the table rule, the
  Resume pass, scan-history coalescing, the flash on add, and the focus-parking docstring updates.
- **Not tried:** a real Mac, a trackpad (the swipe direction is the one sign most likely to need
  reversing), Option delivered from a Mac keyboard, the on-screen cursor shapes, or any real recording.
