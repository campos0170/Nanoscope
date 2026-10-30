# Changelog

All notable NanoScope changes are recorded here. The project is currently in private beta; beta
tags identify tested development baselines and do not promise a stable file or API surface.

## Unreleased

### Changed

- **Ctrl+M files an event, and Ctrl+Shift+M and *Mark selection as missed event* are gone.** The Measure
  tab's *Add to list* became **Add event** and the old measurement list is a group, **Old
  measurements (N)**, shown only while it holds rows (Go to, Remove, Export CSV; no longer added to
  or edited). The *measure on* Raw/Processed choice is hidden whenever the Analysis panel supplies the
  table's own signal, which is always in the app.
- The rise / fall search window of an event pads its edges by max(64 samples, 1.28 ms): bit-identical
  at 50 kHz and below, wider above it, so an edge fitted mid-ramp at 250 kHz no longer loses its 10 %
  point to a window that was a fixed 64 samples (0.26 ms there).
- The event table, its Summary / Fits / Quality tabs and the Merge sub-events strip moved out of
  the Inspector into a new **Results** dock under the trace (View ▸ Panels ▸ Results, ⇧⌘R; close
  it to give the trace its full height — nothing is lost). The Inspector keeps only the parameter
  forms, so they get its full height back (Detection page 256 → 592 px at 1512x900). The Results
  dock opens tall enough for six table rows, capped at a third of the window (at least four whole
  rows at 1512x900), and its two header rows have a fixed height: the toolbar (Table selector,
  Columns…, Merge…, the table's constant columns, advisories) and one status line — the event
  count, or the stale notice, which is now one amber line (verdict, action, Details for the full
  sentence) instead of a wrapped banner up to 254 px tall. When the action is unavailable the
  line starts its reason, giving up the verdict first on a narrow panel. Table rows are 24 px on
  a Mac (one text line + 8 px). A short panel (or Merge… open) scrolls the Events page instead of
  squeezing the table. ⇧⌘R hides the panel with the focus parked on the trace and, shown again,
  hands it back to the table, so a/r/f keep working.
  Saved layouts keep their dock widths and gain the Results dock; the layout is now saved as
  version 1, so a downgraded build resets the layout once and names `windowState` on its info bar.
- Keep scientific warnings and fit verdicts in Details/readouts and saved figure
  provenance, not on plot titles, fit legends or Event Gallery waveform captions.
  Optional fit statistics, units, curves and numerical exports are unchanged.
- Standard report PDF/PNG files retain diagnostic context in document/image
  metadata. New Figure Library captures include full Details text; existing frozen
  figures and previously generated guides are not rewritten.

### Added

- **Hand-made events.** **Analysis ▸ Add event from cursors** (Ctrl+M) files the region selected on
  the trace as an event row in the table on screen (Schmitt, CUSUM, a study table), or
  in a new per-channel **Manual** table that the Table selector always offers ("Manual — no events
  yet") and the first add creates. One door, `AnalysisPanel.add_manual_event`, is behind Ctrl+M, the
  Measure tab's **Add event** button and the trace keys; the row is measured under the table's own
  provenance by `recharacterize_span`, is accepted, `added` and `reviewed`, joins the table in time
  order, and is one undo step. Every refusal changes nothing and names the way forward: no sample
  rate, a non-current channel, a stale table, a span outside what the detector searched, a span that
  overlaps an event, a Reconciled table. The table on screen is remembered per channel, however it got
  there, and saved with the project (workspace key `event_table_choices_v1`; no format bump). Rows carry `note`, `origin`,
  `bounds_source` (`drawn`, `placed`, `fit:step`) and the span as drawn.
- **Trace keys, draw mode and fitted edges.** With the trace focused: **C** catches the event under the
  pointer and fits both edges, **S** / **F** place an edge at the pointer fitted to its step
  (**Shift+S** / **Shift+F** exactly), **D** adds, **Shift+D** adds and goes to the next window, **G** /
  **Shift+G** step windows, **X** discards the mask, **E** toggles draw mode (**View ▸ Drag draws an
  event span**, Ctrl+Shift+E), where a left-drag draws a span instead of box-zooming and releasing fits
  its edges; **Option** inverts the drag for one gesture. Return / Enter / Esc are aliases that act
  only after a click or the pointer arriving, never after a dock parked focus on the trace. New
  `analysis/edge_fit.py` fits the step with gates checked against correlated noise. Keyboard focus
  follows the pointer onto the trace, with Measure on or off, unless a table, text field, menu or the
  docked Event inspector holds it, a horizontal swipe
  or notch pans **time**, and **Shift+↑ / Shift+↓** zoom about the pointer.
- **Measure tab: preview and unstored statistics.** A preview line shows the row the add would store
  (measured like the chosen table's own events, on that table's signal), refusals before the click,
  and the channel's hand-made counts. The statistics beside it are measured on the same signal against
  the table's baseline and noise, so the baseline shown is the baseline stored.
- **Hand-made events across the app.** The gallery marks them **M**, the inspector says so and offers
  **Delete**, the table and gallery menus delete them, and the source filters gain *Hand-made only*,
  *Detected only* and *Merged (hand + detected)*. **Re-running detection keeps them**: each is
  re-measured under the new run, labels move onto a detection of the same event, and the status line
  says what was kept and what was left out. **Analysis ▸ Copy hand-made events to…**,
  **Include hand-made events in exports** (events and training exports) and **Export event
  statistics…** (mean, median, std, variance, min, max, peak-to-peak, RMS, skewness, kurtosis, area,
  amplitude and SNR for one table, every table on the channel, or every Manual table in the project,
  with a `signal_basis` column).
- **Analysis ▸ Convert old measurements to events…** turns the saved manual measurements you tick into
  events in the channel's Manual table (re-measured as events; the dialog says which numbers change),
  hides the converted originals from the Distributions plot source, and is one undo step.
- **Inspector ▸ File Info ▸ Sample rate / time unit…** declares a recording's time base when the
  file does not say, or says it wrongly: the unit its time column is written in (the rate and t₀
  move together) or the sample rate it was acquired at (the only option for a legacy TDMS, which
  declares none). A column the 1 Hz floor refuses keeps the interval it measured, so its unit
  can be declared the same way. The declaration is saved with the project under a new
  workspace key (older v11 builds preserve it; no format bump), never written beside the raw
  file, shown in File Info as `100,000 Hz · user (file: 100 Hz · measured)` (the status bar
  prints the declared rate and its provenance), and followed by the recording's segments.
  Detections made at the old rate are marked stale (*Times stale — sample rate changed*) and
  offered a re-run — refused before the click when a stored cutoff is above the new Nyquist —
  and a saved manual measurement gets the same mark on its row; nothing is re-scaled in place.
- A CSV time column whose **header names its unit** — `time_ms`, `t_us`, `Time (ms)`, `time_s` —
  is read in that unit, with a Note in File Info saying so. A bare `time` is still read as
  seconds.
- The **Event inspector can dock beside the trace** (opt-in; it still opens as its own window by
  default): **Dock beside trace** / **Own window** on its toolbar, or **Window ▸ Dock Event
  inspector beside the trace**, moves the same window — event, filter and context kept — into a tab
  beside the Inspector panel (left or right column only), and the choice is remembered. Its
  minimum fell from 1107×827 to about 452×550 px (452×440 docked): the toolbar is two rows in a
  wide window and five short ones in a column, the amplitude row is two, the waveform's floor is
  180 px, and the readout scrolls and collapses under **Details ▾ / ▸** (hidden when docked, shown
  in a window, each remembered), with the Label row in the fixed bar under the waveform. The
  staleness sentence is a one-line chip (all of it in its tooltip, ⚠ on the Details button).
  Docked, ⌘S / ⇧⌘S / ⌘Z / ⇧⌘Z
  are the main window's own items (so neither copy goes dead as an ambiguous pair), a click anywhere
  in it or on its tab takes the keyboard, a tab sent behind hands the keyboard to the trace, and Tab
  never lands in the panel behind. In a short column the part under its toolbar scrolls, so no
  amplitude method, note or Details state pushes the main window past a 1280×800 screen. The
  waveform's title no longer widens the plot past its widget: a narrow plot draws a shorter form
  that keeps depth and dI/I0 whole (all of it in the tooltip), and a short axis reads **I (pA)**.
  The amplitude note reserves its height, so a staleness flip no longer moves the waveform.
- A **Window** menu: Bring All to Front; the main window, Event inspector, Event gallery,
  Analysis Explorer (and its open pages), Figure Library and the other windows NanoScope has
  opened, checked while on screen; Dock Event inspector beside the trace; and **Reset Window
  Layout…** (asks first). Child windows now reopen at their remembered size and place, validated
  like the main window's and brought back onto the current screen.
- Curation throughput in the Event inspector and the Event gallery: **Advance after decision**
  (off by default, one remembered choice for both windows) makes the A / R / Space / label keys
  move on to the next event of the filter — in the gallery only for a one-tile selection, turning
  pages; buttons never move on and decision keys no longer auto-repeat. New **Unreviewed**,
  **Unlabelled** and **Labelled: key=value** filters; **First unreviewed** (Ctrl/⌘+U); a
  fixed-height progress line (*Reviewed 312 / 870 · accepted 801 · labels: …*); a Label row
  (Apply / Clear) for the inspector's current event; ⌘Z / ⇧⌘Z in the inspector on the main
  undo history. Edit ▸ Undo / Redo no longer reset the inspector to All / the first event or move
  the gallery to tile 0 when the restored table holds the same events. An inspector readout line
  too long for the window now ends in "…" instead of widening the window, and in the gallery a
  decision that takes the focused tile out of the Show subset leaves the cursor beside it (the
  next ← / → selects its neighbour, not the page's first tile).
- Signal Conditioning window (Analysis ▸ Signal conditioning…, Ctrl+Shift+F; also the
  Preprocess card's Open button, the Filter step and the baseline workspace's Advanced
  signal-chain settings): the classical
  detectors' stage editor, moved out of the Inspector into one built-once window with an
  execution-order stage list, notices below the form, the full-height response view, protocol
  Import/Export (versioned JSON, never overwriting) and Copy as Python. The Preprocess tab keeps
  a fixed-height card. When the form's chain differs from the Schmitt/CUSUM detection on screen,
  the event table shows "⚠ Chain changed" with Revert / Run / Open, and the card and window name
  the differences. Ctrl+S / Ctrl+Shift+S save the project from the window, and its Run
  detection… follows the Detect button's gate. Recalling a protocol now sets every stage
  (stages it does not name, and the
  low-pass, are turned off), and its name stays on the combo only while the form equals it —
  a value the recording cannot hold (a cutoff above its Nyquist limit) drops it to "— manual —"
  and the status bar names what changed.
- Signal Conditioning previews of the main trace's visible window, computed on one background
  thread 250 ms after the last edit (a newer edit cancels the older preview): Time (raw against
  processed — Overlay, Split, Raw − processed — split onto the processed trace's real unit when
  Normalize or a high-pass changes it), Spectrum (Welch before/after with a deterministic line
  finder and Send to notch through the form's own checks) and Response (Magnitude plus Phase,
  Group delay and Step, applied zero-phase against the single-pass design). The status line says
  "Window estimate" while baseline drift or Normalize (or a stage settling longer than the 40 s
  padding, which it names) make a window inexact; Compute exact runs the whole recording on
  demand only, and the status line leads with what is happening (Computing exact…, Exact pass
  failed…, Updating…) before the current chain's verdict. A mains line is snapped onto its
  harmonic only when Δf ≤ 1 Hz. Send to notch on an off notch that keeps a list adds to that list
  and leaves the notch off. The window keeps its one preview thread from first open to quit.
- Opt-in Comparison figures page: shared-axis per-recording facets, feature-pair matrices with
  common-bin count histograms, and strip/swarm individual-event plots with optional descriptive
  boxes. Seaborn renders prepared observations; generation is explicit, source identities and
  exclusions are retained, and large/crowded inputs are refused instead of silently sampled.
  Appearance and workspace restore use frozen geometry. Native vector export, publication preview,
  Figure Library capture and bundles preserve exact tables and provenance; warnings stay in Details.
- Publication-workflow hardening: revise saved compositions from their embedded original panels,
  restore layout controls, preview and save as a new figure with portable revision lineage. Nested
  panels and absent source entries are supported; project-generation guards prevent modal saves
  into another project, including an identical reopened project.
- Extended detached shared ranges to exact built-in readable-axis formatters, including Population
  millisecond/log axes and PSD notation, while preserving uniform point styling. Versioned models
  prevent older readers from silently dropping unit-aware formatting. Unknown overrides and varying
  point styles remain unsupported; actual PNG dimensions are bounded before decoding.
- Next-phase plotting: frozen multi-panel composition with ordering, panel labels, grid layouts,
  preview-before-save and compatible shared ranges on detached Cartesian graphics. Each original
  panel's exact tables and resources remain in composition bundles; unsupported plots retain
  independent axes instead of silently losing annotations or image content.
- Selection-only lasso for Population Scatter/PCA with exact event IDs, modifier toggles,
  explicit multi-source choice, gesture cancellation and unchanged curation/analysis results.
- Opt-in Distribution summaries page with validated ECDF, uncensored empirical survival,
  linear-quartile box plots and Scott-KDE violins. Explicit generation, exclusion/comparability
  disclosures, presentation-only styling, frozen workspace restore and Figure Library capture
  are separate from existing detection, baseline, calibration and fitting behavior.
- Completed the core plotting-workflow additions: shared Data / Analysis / Appearance / Export
  sections, reusable appearance-only presets, persistent warning summaries with full details,
  and explicit event-count heatmap versus KDE terminology.
- Project-owned Figure Library, separate from Event Gallery, with named thumbnails, exact frozen
  data/configuration/provenance, source status, duplication, renaming, export and reveal-file actions.
- Graphical publication-export previews, explicit residual-panel inclusion and self-contained,
  checksummed figure ZIP bundles containing PNG/SVG, CSV/Parquet and analytical resources.
- Extended in-place Population styling to colours, markers, lines, legends and fit decorations,
  including appearance-only undo/redo and preservation of pending analytical settings.
- Opt-in event amplitudes: whole-event median, mean, peak and histogram mode; explicit
  selected-interval/state median or mean; and an editable equivalent level in Curate.
  Original median `depth_pA`, fractional depth, SNR, detection masks and state fits are unchanged.
- Measurement-domain Inspector traces, explicit units/reference labels, amplitude-only undo,
  project/CSV audit fields, guarded batch application and stale-choice invalidation after
  mask, merge or baseline changes. Plot pickers distinguish selected amplitudes from median
  depth and disclose missing values and non-comparable measurement choices.
- Shared opt-in plot typography, I-V/voltage-series appearance presets, and project-saved
  presentation overrides. Title/label/color-map edits in event/noise plots and text/canvas
  edits in Distributions update existing items without a new analysis or zoom reset.
- A shared **Export current figure** dialog with PNG/PDF/SVG, native or mm/in sizing,
  aspect lock, 85/180 mm width presets, and PNG DPI metadata. Standard report generation
  is labelled separately. Existing explicit-path exports and numerical CSVs are unchanged.
- Implemented project-v11 complete analytical projects: resumable plot and source state; frozen
  PCA, clustering and subpopulation results with preprocessing and seeds; PSD, density, I-V and
  voltage-dependence snapshots; fingerprinted ML-model references; and live analysis-panel state.
- Restored analytical snapshots lazily without reopening windows or silently rerunning detectors,
  solvers, fits or spectrum kernels.
- Added stable recording/channel/result lineage, missing-source and missing-model carry-through,
  and opaque future/plugin artifact preservation.
- Added explicit backward compatibility for v1-v10 projects and refusal by older readers rather
  than allowing them to strip v11 analytical data.

### Fixed

- **Every add files the same row.** Ctrl+M and the Measure tab's **Add event** button now add through
  the same path as **D**: the same record of how each edge was made (`bounds_source`, the span drawn,
  `fit_json`), the same second press for an edge that is not fitted ("Press again to add it as
  placed"), and the mask cleared afterwards with Measure left on. They had stored a caught event as
  drawn, without its fit, and kept the mask, so a second Ctrl+M was refused as an overlap. The
  second press now confirms only the mask it was asked about: a new mask made within the ten
  seconds was filed by its first press. A table that takes no hand-made row (Reconciled, a stale
  table) is refused on the first press, before any question about the mask.
- **S and F move one edge of any mask.** On a mask placed by a double-click, Snap to view, Snap to
  selected event, Go to measurement or a drag, S or F threw the other edge away and collapsed the
  mask to one sample. Now the other edge stays; only the mask Measure lays down when it is turned on
  is no instruction until it is moved. With one edge still unset (S or F alone) an add is refused and
  names the key that sets it, and the Measure tab's preview says so first; **D** had filed the
  one-sample stand-in as an event. `fit_json` keeps each edge's own fit, where it had described only
  the latest fit and could call an edge stored as `fit:step` unfitted.
- An event cut off by the recording's end ends at the last sample and is rejected automatically as
  `clipped_by_boundary`; it was filed one sample short and accepted. **Shift+D** at the last window
  keeps the add's confirmation and adds that there is no next window, instead of replacing it.
- **An add goes to the table on screen.** With Reconciled on screen, Ctrl+M filed into the source
  table remembered from before, and **Edit ▸ Undo** then reverted the Reconciled table instead. The
  remembered table now follows the screen (the selector, a new run, Reconcile), a saved project
  reopens on it, and Reconciled on screen refuses an add with the way forward. A **Batch detect**,
  or a Run that finished after you moved to another channel of the same recording, no longer puts
  that channel's own older table of the same detector on screen in place of the one you were
  working in.
- **Export event statistics** no longer mixes gains: a table measured before a calibration change is
  refused before the save dialog when it is the table on screen, and left out and named when the
  scope is every table on the channel or every Manual table, as a table on another sample rate is.
- The docked Event inspector lost A, R, Space and the label keys whenever the pointer crossed the
  trace, because the Inspector panel beside it had been registered as the trace's key owner. With
  Measure off, how a recording opens, focus did not follow the pointer onto the trace, **C** did
  nothing and **Shift+↑ / Shift+↓** ignored the pointer; the trace keys now act on the selected
  recording's current channel with Measure on or off.
- **Delete** of a hand-made event from the inspector dropped its *Hand-made only* filter and put both
  curation windows on a detected event. Each window now keeps its filter and lands, quietly, on the
  next event of its own list. **Edit ▸ Undo** of an add the windows were showing does the same.
- The Measure tab's preview went blank and **Add event** greyed out on a span holding a NaN or Inf
  sample, while Ctrl+M stored the row; the preview now shows that row, marked *would be rejected:
  non_finite*, and **Add event** is available whenever there is a mask.
- Re-running detection kept only one note when several hand-made events landed on one detection; every
  note is kept now, joined in time order with `; `. With **Include hand-made events in exports** off,
  the training export's sidecar counted the switched-off rows as withheld. After a re-run with another
  low-pass, the edge fit used the old run's open-pore tau.
- Event inspector and Event gallery: A, R and Space no longer take the key from the control that
  has the keyboard. Space on a Tab-focused button (Show in trace, Gallery…, Accept all / Reject
  all, Accept selected / Reject selected, Apply…) or check box presses it instead of toggling
  the event on screen or flipping every selected tile. The letters, Space and page keys typed
  into an open Filter, Show or other dropdown, or the gallery's Select menu, go to the list
  instead of accepting, rejecting or paging (typing "r" in the open Show list rejected the
  selected tiles, and PgDn turned the page and dropped the selection). Advance after decision
  (both windows), Dock beside trace, the Label combo with Apply and Clear, and Details had been
  taken out of the Tab chain so Space could not hit them; Tab reaches them again and Space
  presses them, while a mouse click on them still leaves the keyboard on the waveform or tiles.
  A, R and Space still curate from the waveform, the tile view and the window background.
- Pressing **Detect** (or any other analysis) while a **Merge all**, **Re-measure** or **Compare
  baseline estimators** was running replaced that job, and from then on the Manual study's and
  the Acid study's "prepare review" actions and the study import refused with "Wait for
  main-workspace analysis and editing operations to finish first." although nothing was running;
  selecting another recording, a second Detect or closing every recording did not clear it. The
  job it had replaced left its bookkeeping behind, and the same left "Measure this recording"
  reading "Measuring…" (and refusing a second click) and the Baseline-correction workspace's
  Refresh reading "Refreshing comparison…" until the recording changed. The analysis controller
  now announces a replaced job and the panel forgets everything it recorded about it, releasing
  the busy button that job owned (a job that needs the same button, such as an export replacing
  a comparison, keeps it); a comparison that was replaced says so in its status line instead of
  "Calculating…". The study guard still refuses while a job really runs. Also fixed: the answer
  or failure of a job that ended just as another replaced it could be taken for the NEW job's,
  ending it and losing its result (about a third of replacements in a tight loop); a replaced
  job's worker is now kept until its queued signals have been delivered and dropped.
- After **Inspector ▸ File Info ▸ Sample rate / time unit…**, a hand-added event (Ctrl+Shift+M
  or the Measure tab's button), a span edge dragged or Reverted in the Event inspector, and the
  three merges (Merge all, Merge events in selection, the inspector's Merge with next) wrote
  rows on the NEW clock into the detection table the amber line had just called entirely the old
  one: on a 2 s, 50 kHz slice declared to be 40 kHz a 300-sample event got a dwell of 7.5 ms
  beside nine neighbours measured at 50 kHz, and Merge all left four old-clock and four
  new-clock rows under a line that still said every time in the table was computed at the old
  rate. Stored detections keep the rate they were measured at (never silently re-scaled), so
  these doors now refuse such a table before reading or writing anything, so no row, no stamp
  and no Undo step moves. The status bar says what did not happen (*Nothing was added.* /
  *Nothing was merged.* / *The event's bounds were not changed.*) followed by the amber line's
  own sentence, whose Re-run is the way forward; Analysis ▸ Compare baseline estimators and
  Re-measure depths say it the same way (*Nothing was re-measured.*) instead of opening a
  `ValueError` box. A table that is not stale (no declaration, the declaration taken back, a
  fresh run under the declared rate, or a table with no rate stamp) is edited exactly as before.
  One rule, `analysis.events.stale_rate_hz`, now decides this for the amber line, the editing
  doors and the re-measure kernel's own guard.
- **Analysis ▸ Export events…**, **Export events for training…** and **Export sub-state
  levels…** no longer write a table the amber line has marked stale. After a sample rate or time
  unit was declared, the table kept the rate it was detected at, yet the export went ahead: the
  CSV's times stayed on the old clock while the `.meta.json` took `fs` from the table (the old
  rate) and `t0_s` with its "add t₀ to find it on the trace" sentence from the new one. For 40
  kHz declared over a 50 kHz detection that sentence put the first event 2599 samples off, and a
  time-unit declaration moved `t0_s` (10000.0 → 10.0) under an unchanged CSV, with nothing in
  either file saying a declaration existed. The three exports now refuse such a table before
  asking for a file, write nothing, and say on the status bar to re-run detection under the
  declared rate, naming the rate the table was detected at and the one the recording reads now;
  after the re-run the export goes through and its `t0_s + t_start_s` lands on the trace sample.
  A table that is current, one with no rate stamp, and one whose recording has no rate any more
  export exactly as before.
- A pass made by Signal Conditioning ▸ **Compute exact** at one sample rate is no longer shown
  as the exact trace of another. After a rate was declared (Inspector ▸ File Info ▸ Sample rate
  / time unit…) and Signal Conditioning opened again, the cache of whole-recording passes found
  the pass made at the file's 50,000 Hz under the key of the chain at the declared 25,000 Hz and
  drew it under *Whole recording (exact)* with the button reading *Done: the preview shows this
  chain's whole-recording (exact) samples* — 26 pA from what the chain makes of the same samples
  at the declared rate, on a recording whose noise is 4.5 pA. The cache key now names the rate
  the pass was run at (the chain's signature never did: it uses the rate only to drop notch
  lines above Nyquist), so a pass belongs to one rate: after a declaration, or after taking it
  away, the window says *Window estimate … · press Compute exact for the true trace* with the
  button enabled until the pass has been made at the rate in force, and a pass made earlier at
  the other rate is found again when the declaration is taken away. Calibration edits still drop
  the pass, and the cache still holds the two most recent.
- In the Results table, ⌘A, ⌘R, ⌘F and every Ctrl / Option / Shift chord on a, r, f, j or k were
  read as the bare curation key: ⌘A accepted the selected row instead of selecting all of them,
  ⌘R rejected it and ⌘F flagged it (Edit ▸ Undo then read "Undo Curate event", the status bar
  said nothing), and Select All never selected anything. The same happened to a chord pressed on
  a control inside the table's panel, such as ⌘R on the focused Columns… button. Only an
  unmodified a / r / f / j / k curates now: ⌘A selects every row (the caption reads "N of N rows
  selected · A accepts / R rejects all N"), the other chords do nothing in the table, and ⇧⌘F
  and ⇧⌘R still reach their menu items.
- A sample rate or time unit declared under **Inspector ▸ File Info ▸ Sample rate / time
  unit…**, or taken away again with *Use the file's own time base*, now reaches an open **Signal
  Conditioning** window. It used to go on previewing at the old rate: after a CSV whose time
  column is in milliseconds was declared (50 Hz → 50 kHz) its Time tab spanned 0–2000 s against
  the trace's 0–2 s under "Exact for these stages · window 2000.00 s", its Spectrum tab stopped
  at the old Nyquist, a 10 kHz low-pass the form had accepted was refused as "at or above
  Nyquist (25 Hz) for fs=50 Hz", and a detection run afterwards left the window holding two
  rates at once (source 50,000 Hz, chain 25,000 Hz on a recording declared at 25 kHz). Only
  closing and reopening the window or re-selecting the channel healed it, and nothing said it
  was stale. The window now re-reads its source and its chain together whenever the recording it
  shows is re-declared (a segment follows its parent's declaration), drops a preview that was
  still coming back at the old rate, cancels a running *Compute exact*, and re-frames a Time
  plot you had panned; a stage that no longer fits (a notch list above the new Nyquist, a
  baseline-drift window longer than the declared recording) is refused in the new rate's words.
  The source's identity now includes its sample rate, so a *Compute exact* pass computed at the
  old rate is no longer served as the exact trace of the new one when the window is reopened (26
  pA off a 4.5 pA noise sigma, labelled "Whole recording (exact)").
- The **Navigator strip** stayed on the old clock after **Inspector ▸ File Info ▸ Sample rate /
  time unit…** declared a recording's time base (or *Use the file's own time base* took the
  declaration away): the trace re-framed, the strip kept its curve, extent and shaded-window
  bounds. Declaring a CSV's millisecond column left it reading **Local context · 9× view** over
  a 2 s recording, drawing none of its 100,000 samples, and letting a drag of the shaded window
  to 13–15 s take the trace there too, past the last sample, so the pane went blank; a
  sample-rate-only declaration left the curve filling half the strip, or the window unable to
  reach the end of a recording it had made longer; a refused column declared in microseconds
  kept the strip in samples; Zoom-to-full repaired none of it, only re-selecting the recording.
  The strip now re-reads the recording whenever a drawn recording's time base is declared or
  removed — curve, extent, window bounds, the nine-view lens, its label and the Measure band —
  whichever of it and the trace hears the change first; a declaration on a recording that is not
  drawn leaves it alone.
- The Results dock's **Fits** tab now shows a fit only beside the table it was made on. The
  result store keeps one dwell fit and one depth fit per recording and channel, so a recording's
  Schmitt, CUSUM and Reconciled tables shared one slot and the tab printed whatever it held:
  Schmitt's `n = 316` under CUSUM's 370 candidates, the same fit beside a merged table (246
  accepted), a re-run (44 events), a table with one event rejected and the table an Undo
  restored — while Export figure refused that very fit as "saved fit is stale" — and a Fit whose
  dwell half failed left one table's dwell fit beside another's depth fit. Each fit now records
  the table it was made on (`detection_module_id`, a new key in the fit's saved metadata; no
  project format change, and older builds read the project as before). The tab prints a fit only
  when that table is the one on screen, with the calibration and accepted events it was fitted
  to (the figure export's two checks plus the table). Otherwise it says which table the fit
  belongs to, or that the table has changed since (curated, merged, re-run or undone), or that
  the calibration differs, and to press **Fit**. A Fit press replaces the pair: a half that
  fails drops its predecessor. A fit that records no table — saved by an earlier build, or filed
  by the Plots window — stays in the project and is still used by Export figure exactly as
  before, but is not printed beside any table until you press **Fit**. The accepted count still
  stands in for "the same events" (as it does for the export), so swapping one accepted event
  for another leaves a fit standing.
- Switching the Results dock's **Table** selector (or the Detection tab's detector) now takes
  the open **Event inspector**, docked or in its own window, and the **Event gallery** with it.
  They used to keep the table the dock had just stopped showing, so one R rejected the event in
  that table **and**, by event number, a different event of the table on screen: two tables
  written, one Edit ▸ Undo step that took back only the second (the first kept a "reviewed"
  rejection that travelled into the saved project and every export, and the menu read "Undo —
  nothing on this table; the threshold detector (1 step) on this channel"), and a click on a
  Results row showed the other table's event under the same number. Each surface now shows the
  table on screen from its first event (the inspector back to All, the gallery on its first
  tile, the trace and the Results selection left where they were), one keypress writes one table
  as one undo step, and a surface that is ever out of step with the dock keeps its decision on
  its own table, says so in the status bar and re-syncs instead of mirroring it onto the dock's
  table by event number. A CUSUM sub-state fit stays with the table it was measured on rather
  than being drawn under another table's events.
- A CSV time column in **milliseconds** loaded silently at a rate 1,000× too low: 10 µs steps
  written as `0.01` per row read as 100 Hz, `Provenance.MEASURED`, duration 552 s for 0.5 s of
  samples, with an empty Issues box — the only guard was the 1 Hz floor that catches microsecond
  columns, and no sample-rate override existed anywhere to correct it. An undeclared time column
  (a CSV's, or an HDF5 `/time`) whose measured rate is below 1 kHz now raises an Issue naming the
  step, the rate as read, what the rate is if the time base is in milliseconds or microseconds,
  and how to settle it (for a CSV, a unit in the column header; for either, the File Info
  declaration); the measured
  rate is kept, nothing is converted by a guess, and a header-declared unit never raises it. The
  user guide's Issues row claimed every non-seconds column was refused; it now says what is
  caught, what is asked, and what is read as declared.
- A saved window size and place from a differently sized display (a laptop and an external
  monitor more than about a fifth apart in width) is no longer reported as "a value this app did
  not write": the main window, the child windows and Signal Conditioning take their saved size
  and place, brought onto the current screen (a maximized main window stays maximized), and a
  window's remembered place is no longer deleted. Only bytes that really are damaged are named.
  Signal Conditioning is now listed in the **Window** menu. **Window ▸ Reset Window Layout…**
  also puts a window that is hidden back at its default size and place.
- A dock that is floating when you quit (Results, Inspector or File manager) now comes back
  floating and visible; quitting used to close it before the layout was saved. **Window ▸ Reset
  Window Layout…** now also forgets the Signal Conditioning window's size, place and splitter,
  and no longer lets a window you opened and hid earlier write its old place back at quit.
  A hand-added event (Ctrl+Shift+M) over an event the detection already has is refused with a
  status-bar line instead of counting the event twice, and a hand-add (or Merge all, or their
  undo) with the Event gallery open no longer jumps the main trace to the gallery's first tile.
- Reopening a project with ML detections no longer shows a false "open-pore estimator
  changed" banner when the stored windows differ from the defaults. The ML branch of the
  stored-detection seeding restored only the ML tab and left the baseline form on the app
  default, so the banner appeared and its "Re-measure depths" button would have re-measured with
  the defaults instead of the windows the table was measured with. ML results saved without
  stored parameters (pre-v3 projects) are unchanged. The baseline form is shared by every
  detector on a channel, so a table of another detector measured with different windows still
  reads as stale until it is selected.
- The ML preprocessing median (`_decimated_median`) is a rolling order statistic instead of
  ndimage's O(n*w) rank filter: about 40-50x faster at the 2501-tap Host-guest / Zn windows
  (6.2 s to 0.16 s at 3M samples; 31 s to 0.6 s at 1.5M samples after decimation). Values are
  identical on 240 synthetic cases and the 35 real-recording corpus tests (the sign of a zero
  can differ when -0.0 and 0.0 are mixed). Non-finite input and windows longer than the signal
  keep the previous path. The pinned md5 of `ml_preprocess.py` was re-pinned with this
  verification; the median's arithmetic now also depends on
  `acid_incubation_numerics.median_filter_1d`, which has its own tests.
- Voltage dependence and its menu gate read tables from any primary detector (Schmitt, CUSUM,
  ML): the one stored on the most recordings that can become a point (Schmitt on a tie) is
  pooled for the whole series, recorded in saved workspaces, and named in the Pooled provenance
  block when it is not Schmitt. An ML-only project previously had the action greyed out.
- Event inspector: opens on the event selected in the table (and, re-opened, where you were)
  instead of event 0; keyboard focus starts on the waveform instead of the "Event N" spin box,
  so A/R/T and label digits reach their shortcuts; the first event no longer draws against a
  0..1 pA placeholder range when the stale-measurement note arrives after the first draw.
- Analysis > Event labels... fits a laptop screen: its explanation line wraps instead of
  forcing a minimum width wider than the screen (1777 px at the app's 13 pt font).
- The trace view gives the current channel more height than the command-voltage pane (5:2
  instead of equal rows).
- "Known open-pore current" accepted 0 pA (and anything that underflows to 0 in the
  float32 pipeline that reads it, e.g. 1e-300) with no error, then silently produced a
  NaN fractional depth on every event and a NaN Summary median with no explanation.
  Refused at the editor, at the precision the analysis pipeline actually reads it under.
- Three numeric boxes (open-pore current, notch lines, acquisition bandwidth cutoff)
  each guarded only against PEP 515 underscore grouping (`1_20` -> 120.0); none guarded
  against the non-ASCII digit scripts bare `float()` also accepts (`١٢٠` -> 120.0). A
  shared strict-ASCII parser now backs all three.
- The per-event local baseline-intrusion advisory could report "assessed" with a
  near-zero fraction on exactly the two canonical Automatic-mode baseline failures
  (isolated long events; high-duty recordings), because it compared a fragment's two
  flanks to each other rather than to a wider, already-computed reference. Gated on the
  recording-level baseline-health verdict baseline.py already produces; advisory only,
  no detector output changes.
- The HDF5 reader read t0 from a `/time` column it had already declared unusable for a
  sample rate (all-NaN or reversed), producing a NaN or backwards origin with no issue
  naming the problem. Returns the documented "unavailable" sentinel with an explicit
  issue instead.
- `recharacterize_span` accepted a negative sample rate (a wrong-signed row, no error)
  and raised the wrong exception type for zero, infinite, or `None`; now raises
  `ValueError` naming `fs`, matching `detect_events`'s existing convention.
- Opening Analysis > I-V on a large, never-before-medianed recording blocked the GUI
  thread for the whole whole-channel median (~300 ms observed) with no busy state. The
  first gather now runs on a background thread once a channel exceeds ~500,000 samples;
  the raw arrays it needs are resolved on the GUI thread first and handed to the worker
  as plain data, never a live `Session` reference, after an earlier version of this fix
  was found and rejected in review for exactly that hazard.
- A project reset/switch deleted the I-V window directly, bypassing the retirement step
  every other background-worker-holding window goes through at teardown; a gather still
  in flight was left running with nothing left to deliver its result to.
- Practice-recording duplicate detection compared paths with `==` instead of the one
  function that answers "is this file already open" -- the literal pattern an
  architectural guard exists to forbid -- and carried a latent case-sensitivity defect
  on a case-insensitive volume as a result.
- Two dialogs (Help > About; the open-pore reference preview) set `WA_DeleteOnClose`,
  the one lifetime pattern `nanoscope/gui` otherwise avoids everywhere: the app's
  teardown sweep calls `close()` on every owned window as a stop mechanism, which is a
  hide for every other window and would have been a delete under a live Python wrapper
  for these two.
- Several architectural AST guards had gone stale as legitimate shared infrastructure
  (plot-appearance/workflow-bar/label modules, a lasso-selection primitive, a
  publication-export dialog) was adopted by multiple analysis windows without the
  guards' exact-import-set assertions being updated to match. Each newly-allowed module
  was individually verified -- no widget of its own, no further reach beyond the same
  shared, non-panel set -- before being added, rather than the assertions being loosened
  wholesale.

### Validation

- 2026-09-13: all six items confirmed still open from the 2026-09-12 beta sweep (a
  zero open-pore current; non-ASCII digit acceptance in three boxes; the baseline-
  intrusion advisory's false negative; the HDF5 t0 defect; `recharacterize_span`'s
  fs validation; the I-V first-open stall) are fixed and independently reviewed;
  root-cause fixes throughout, not input-specific patches. Nine other items from that
  sweep, including the acquisition-metadata save failure, were found already resolved
  by intervening work and are not claimed here. 358+ targeted tests re-verified across
  measurement quality, the reader contract, characterize/span/flag-invariant guards
  and I-V; 455 of 457 tests across the sibling-import guard files (the two remaining
  failures are a pre-existing, unrelated geometry pin).
- 2026-09-13: root cause found for the intermittent PySide6 signal-manager crash noted
  below -- once a Qt wrapper's C++ side is destroyed, its Python object becomes
  permanently invisible to Python's own cyclic garbage collector. Confirmed on PySide6
  6.11.1 and 6.11.2 both, and confirmed independent of this suite's `gc.disable()`
  policy (a separate, unrelated deadlock avoidance; re-enabling it does not free these).
  Measured: ~734 leaked `QObject`s and ~3.6 MB RSS per `MainWindow` construct/destroy
  cycle, linear with no plateau across 40 cycles. This is a PySide6 characteristic, not
  fixed here; the manual full-regression CI job now runs the `gui` shard as 16 smaller
  processes instead of one, bounding rather than eliminating the accumulation a single
  long process would otherwise reach.
- Comparison-figure source pass: 571 focused tests passed across rendering, page interaction,
  workspace persistence, Library capture, native vector/export composition and scientific
  regressions. All 37 recorded analysis-module hashes are unchanged. Read-only apo-ferritin
  QA used all 2,104 saved accepted events for the three new views, checked PNG/PDF/SVG exports,
  and left original event tables and project bytes unchanged. This source pass preceded the
  packaged comparison release below.
- 2026-09-06 comparison packaging: rebuilt the Apple Silicon app and replaced both the local
  and SanDisk DMGs. All 16 applicable in-bundle checks passed, including Seaborn facets, pair
  matrices, strip/swarm views, source-free restore and twelve PNG/PDF/SVG exports; three optional
  real-recording/model checks were skipped because no inputs were supplied. All 27 packaging
  regressions passed. Strict ad-hoc signature verification and both DMG integrity checks passed;
  the images have matching SHA-256 `3ffe9e901c7032169496b3303b4dc27ebec56e3eb8c0c9c70fad52b79c9a48d6`.
  All 125 embedded NanoScope modules plus the packaged self-test match current source bytecode,
  apart from normalized filenames. The installed `/Applications` copy was not replaced in this pass.
- 2026-09-06 packaging: rebuilt the macOS application and replaced the local and SanDisk
  DMGs. All 15 applicable packaged smoke checks passed; three optional real-recording/model
  checks were skipped because no inputs were supplied. Strict ad-hoc signature verification
  and both DMG integrity checks passed, and the DMGs have matching SHA-256 hashes. All 122
  embedded NanoScope modules match the current source code apart from rewritten filenames.
  The installed `/Applications` copy was left unchanged.
- Publication-hardening pass: 88 composition/revision checks passed, including real Population
  linear/log millisecond axes, real PSD replay, original snapshot/resource preservation, model
  version guards and project-switch cancellation. Additional targeted runs passed 108 export/
  capture checks, 74 existing axis/selection checks and 79 Library/Explorer/workspace checks
  (these groups overlap). Native and physical-size shared-panel exports and revision layout
  were visually verified. No scientific kernels or analysis defaults changed in this pass.
- Next-phase verification covered numerical references and unit-change invariants for all four
  new summaries, exact frozen restore/export, invalid-input refusal, composition resources and
  shared-axis safety, real lasso gestures and event-table routing, and unchanged automatic zoom.
  The existing baseline, detection, characterization, amplitude and workspace regression group
  passed all 394 checks; Explorer/workspace integration passed 101. Offscreen visual QA covered
  all four summary types and the multi-panel composer. These are targeted runs, not a full suite.
- Core plotting verification: 182 targeted checks passed across frozen previews/exports,
  appearance invariance, Figure Library persistence/provenance, all-page capture and staged-write
  safety. Broader plotting, numerical, workspace and responsive-layout suites also passed in
  targeted runs. Visual QA covered publication previews and smaller Explorer windows.
- Long combined GUI runs can intermittently hit the existing PySide6 6.11.1 signal-manager
  crash seen before this work; this is not claimed as a clean full-suite run.
- 870 distinct focused amplitude, detector, characterization, persistence, curation and
  plotting regression checks passed; offscreen visual checks verified editable-level and
  selected-interval layouts. These source checks preceded the packaged build described above.
- 8,114 tests passed and 8 corpus/platform tests skipped in the complete Stage 3 regression.

## 0.1.0 Beta 1 — 2026-09-03

### Added

- Interactive ABF, HDF5, TDMS, CSV, and text-trace viewing and calibration.
- Schmitt, CUSUM, ML, and manual event detection with provenance-preserving reconciliation.
- Event inspection, batch curation, structured labels, and supervised-learning exports.
- Histograms, density and scatter plots, fitting, PCA, clustering, PSD, I–V, and voltage-dependence
  analysis.
- Project persistence with guarded atomic saves and external-change conflict detection.
- Crash-recovery autosaves after 15 seconds idle, bounded by a 60-second maximum, with two atomic
  generations and Recover, Discard, and Show in Finder startup actions.
- Adaptive local-context trace navigation for efficient manual annotation.

### Validation

- 8,060 tests passed and 6 corpus/platform tests skipped in the complete Stage 2 regression.
- The packaged macOS application passed all 18 bundle self-tests, strict ad-hoc signature
  verification, and DMG checksum verification.

### Distribution limits

- The beta build targets Apple Silicon and macOS 15 or later.
- The application is ad-hoc signed but not Apple-notarized; Gatekeeper may require the documented
  first-launch workaround.
- Raw recordings are referenced by path and are not embedded in projects or recovery snapshots.
