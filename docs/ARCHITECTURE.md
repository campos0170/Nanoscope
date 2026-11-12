# NanoScope — Architecture

Design notes for developers and maintainers.

## Layering

The package is layered so the analysis core never depends on Qt, and the GUI never re-implements
analysis:

```
nanoscope/
  io/        format readers (ABF/HDF5/TDMS/text) behind one dispatch registry, over one shared
             answer-contract (io/contract.py) — no GUI, no analysis
  core/      data model + state hub (Session, Document, calibration, identity, types), the analysis
             *seam*, the undo door (history), the project (de)serializer, the write seam
  analysis/  pure DSP/ML: filters, baseline (+ baseline_probe), detect, characterize, events, stats,
             populations, measure, remeasure, psd, iv, merge, scoring, truth, and the ML pipeline
  plot/      matplotlib publication figures + style (static export path, separate from the live viewer)
  gui/       PySide6 + pyqtgraph widgets; the only layer that imports Qt
```

Rule of thumb: `analysis/` and `io/` are importable and testable with no display; `gui/` orchestrates
them. Everything in `analysis/`/`core/` runs headless in CI.

## State model

`core/session.Session` is the **single mutable state hub**. Each GUI panel holds exactly one
reference — to `Session` — and communicates through its mutators and Qt signals. **No panel imports
or references another panel**; `main_window.py` and the `main_window_<part>.py` mixins it is made of
(see *The main window and the Plots window: one class in several files*) are the *only* GUI modules
allowed to import other panels, and they wire them together. This keeps the panel graph a star, not
a web, and is enforced by a test that walks the `gui/` ASTs.

- **Documents.** `core/document.TraceDocument` is an immutable value: metadata + per-channel
  contiguous **float32** signal arrays. The time axis is **float64** (a float32 time axis is
  non-monotonic at 15 M samples and breaks rendering).
- **Calibration** is separate from the document (`core/calibration`): the same raw data can be viewed
  under different gains/offsets. Calibrated arrays are produced on demand and cached; identity
  calibration returns the raw array by reference (no copy) — read-only, like every cached
  calibrated array (see *Load-bearing invariants*).
  **"Back at its default" means the default the channel OPENED with.** A voltage channel's default
  is detected from the channel's own median: past `_MAX_PLAUSIBLE_BIAS_MV` (1200 mV) it is the
  10×-inflated encoding and opens at gain 0.1, below it the number IS the bias and it opens at 1.0.
  `default_for(info)` asked again without the median answers 0.1 for every voltage-mV channel, so
  Reset to default installed that on a host-guest channel that opened at 1.0 — 10× on every
  voltage, conductance and V-dependence figure — and the sidecar, comparing with the same answer,
  stored the untouched voltage channel on every write. `Session.add_document` therefore records
  `default_for(info, median)` per channel before any sidecar is applied
  (`CalibrationSet.opened_defaults`, read through `Session.opened_default`), and Reset, the sidecar
  payload (`_sidecar_payload`, so `save_sidecar`, `sidecar_effect` and `notes_lost_by`) and the
  workflow chip all compare with it. A segment takes its PARENT's opened default: the encoding
  belongs to the file, and a window cut where the bias was low would read as a plausible bias.
  Reset resets gain, offset and units only; the note is the user's record and the kind override
  has its own reset.
- **Signal loading** is eager (whole channel into RAM) and happens **off the Qt main thread** via a
  worker; ~15 M samples/channel is the design ceiling, above which the loader offers a sample range.
- **Identity** is three questions, not one, and `core/identity.py` is where they are kept apart.
  Conflating them into "the resolved path" caused three separate data-safety defects.
  - *Which load?* — `doc_id`. Opaque, unique to **this load in this session**; only `core/identity`
    may build it (`make_doc_id`) or take it apart (`source_of`), and an AST guard fails the build if
    anything else derives an identity from `meta.path.resolve()`. It is stamped once in
    `TraceDocument.__post_init__` — the one funnel every recording passes through — and preserved
    across `dataclasses.replace`, so it is stable for the life of the load. This is what makes the
    ~50 `session.has_document(res.doc_id)` guards mean what they always assumed: a detection that
    lands after its recording was closed **and reopened** no longer installs itself onto the new
    document.
  - *Which recording, across sessions?* — `core/segment.persistent_key(doc)`: the resolved source
    path for a file (`Document.source_key`), the segment uid for a derived segment. Never the
    display name (a segment may be named after its parent) and never `doc_id`. Results,
    calibrations, layers, manual measurements **and the analysis windows' overlay sources**
    (`core/sources.py`) are keyed by it. "Is this *file* already open?" is therefore
    `Session.doc_id_for_source`, not `has_document`.
  - *Which bytes?* — `identity.SourceFingerprint`, stamped into each project's `DocumentRef` (v4).
    A path is a name, not a recording: re-exporting over the same filename used to reattach saved
    events to completely different data in silence. On mismatch the restore **warns and explains**
    (naming what differs and how many restored events now lie past the end of the file); it never
    refuses to open, and an absent or unreadable stamp reads as *unknown*, never as *verified*.
- **Undo** is `core/history.EventHistory`, one snapshot stack over every `EventSet` mutation in the
  session, keyed `(doc_id, channel, module_id)`. It is deliberately **not** a `QUndoStack`, and this
  is the load-bearing decision in the file: `QUndoStack.push()` *performs* the command, so every
  mutation site would have to build a `QUndoCommand` that knows how to do and un-do itself **before**
  it mutates. This app mutates an `EventSet` from **eleven** call sites — six install a new result
  (a finished detection, a batch detection, a merge from any of its three entry points, a hand-added
  missed event, a re-measure, a project restore) and the other **five write the shared `DataFrame`
  in place** from inside a Qt slot on a widget (the table's `accepted` checkbox, the Inspector's
  single accept/reject, Accept-all/Reject-all, a mask drag, the table's flag cell). Retro-fitting a
  command object onto each is exactly the "fix applied at N call sites" shape this project has
  watched come back three times: the twelfth mutation, written next month, forgets to push and is
  silently un-undoable. Instead the history keeps a pristine deep copy of every live result and
  **compares**.
  A mutation does not have to announce itself to be captured; announcing it only sharpens the menu
  label, and even the label is derived from the diff (`describe_change`). A forgotten announcement
  costs a word, never a row. `main_window.py` is the only module that makes checkpoints (AST-guarded: the
  guard reads every other `gui/*.py`, the `main_window_*.py` parts included; the parts only clear the
  recorders and suspend them while a project restores, in `main_window_project.py`, and ask
  `is_applying()` in `main_window_layout.py`) — it
  is the one object that already receives every announcement of an `EventSet` mutation
  (`resultsChanged`, `eventAcceptanceChanged`, `eventFlagChanged`, the Inspector's `eventEdited` and
  `curationBatchChanged`), all routed to the one recorder. The redundancy there is deliberate and
  free: a checkpoint that finds nothing changed pushes nothing, so wiring a signal twice cannot
  invent an undo step. Undo re-installs through
  `Session.add_result`, so the existing `resultsChanged` refresh is the only repaint path.
  `checkpoint_all()` is reached from four places, and the last two are what make the promise above
  true rather than aspirational: an undo/redo, the Edit menu opening, **a project save and a quit**.
  Two things the diff alone cannot see are read off the RESULT instead: whether the `EventSet`
  OBJECT was replaced (an install) or edited in place (a curation click), and whether the new table
  carries any of the hand curation the old one held. Pressing Detect twice with the same settings
  produces an identical frame with the accept/reject column reset — indistinguishable from curation
  in the frame, unmistakable in the result — so it is named "Re-run detection" and announced.
  Optional non-schema columns (`added`) are reconciled inside the comparison: it is materialised
  only when a detection is DISPLAYED, so comparing column lists made every batch-detected table
  that was never on screen report a mutation on every checkpoint.
- **The hand-measurement table gets the same rule from a sibling class**,
  `core/history.MeasurementHistory`, not from a sixth key on `EventHistory`. Everything that made
  the event stack right carries over — a pristine deep copy that is compared, a derived label, a
  bounded stack — but the PAYLOAD does not: every step of `EventHistory` goes through
  `AnalysisResult`/`EventSet` (the frozen 20-column schema, `ensure_added_column`,
  `describe_change`'s column groups, `Session.add_result` as the install door), and a measurement row
  shares none of it. A reserved key would have meant branching on that key in five places inside the
  newest and most-repaired module in the app. The frame is read and written through two callables
  the wirer supplies, so the class stays free of Qt widgets and of the measurement schema. There is
  still exactly ONE Edit ▸ Undo: `main_window` keeps a small journal of which stack each step landed
  on and a press takes back the most recent step that can actually be applied.
- **The analysis settings get the rule a third time**, from `core/history.SettingsHistory`, for
  `MeasurementHistory`'s reason: the rule is reused and the payload is not — here it is not even
  one shape. A scope is `("calibration", doc_id, channel)`, `("time_base", doc_id)`,
  `("acquisition", doc_id, channel)` or `("form", doc_id, channel, module_id)`: the Detection, ML
  and Baseline forms of one event table, `module_id` `None` until the recording and channel have
  a table. Each scope has its own stack, `MAX_DEPTH` deep, read, written and named
  through three callables `main_window` supplies, dispatched on `scope[0]` through one table
  (`_settings_kinds`), so the class holds no widget and no schema. `None` from a reader means
  "cannot be read now", never a value: a setting whose own value can be `None` — the file's own
  time base, no bandwidth override — is read wrapped (`{"declaration": None}`) or it could never be
  undone back to. A scope is adopted silently when it appears (`rebase` on `documentOpened`, after a
  segment cut, when a restore settles), or the first Apply on a fresh recording would be its
  baseline and could never be undone; and a form is rebased whenever it arrives through a switch, a
  seed or a restore, an event undo that re-installs a run included, so a re-seed is never recorded
  as the user's edit. The form's steps made before a table existed move to the first table its
  recording and channel get (`move_scope`), but only to a table BORN there, one that lands while the
  session is announcing it (`Session.announcing`): a switch to a table that already existed would
  put its steps in front of that table's own. A write-back runs
  `checkpoint_all()` while the history is still applying, so what it moved in ANOTHER scope — a
  time base taken back re-clamps the form's frequency boxes — is adopted there rather than recorded
  later as the user's edit. The stack's `changed` drives the menu text but NOT the
  dirty flag — every setting it holds already dirties the project through the session's own
  signal (a form's through `formEdited`), and `changed` also fires when a closing recording takes
  its scopes with it.

  **A calibration step carries its sidecar.** What a calibration scope holds is the calibration and
  the bytes of `<datafile>.npcal.json` on each side of the commit that made it (`_CalibrationState`;
  the file is not compared, or an Apply that moved no number would be a step that takes nothing
  back). The Calibration panel announces each commit (`commitStarting`), so the step keeps the file
  as the commit found it. An undo or redo installs the calibration alone
  (`set_calibration(..., write_sidecar=False)`) and `_put_back_sidecar` then puts the file back
  through `core.calibration.restore_sidecar`, byte for byte and without asking (the owner's
  decision 6): a rewrite, a deletion, or a deleted file written back with its note. It writes only
  while the file still holds what the step left; otherwise it returns `SIDECAR_KEPT` and leaves the
  file as it is, which is what another channel's step still in effect, or an edit made outside the
  app, looks like. Written back through `save_sidecar` as Apply is, the file was re-derived by the
  payload rule, which took a note off a channel at its default numbers, deleted a file whose only
  entry that was, and after a project restore wrote a gain the file had never held. The status
  line, and the Calibration panel's own line (`show_status`), say what happened to the file in the
  panel's words (`calibration_panel.sidecar_outcome_text`).

  The journal's entries are `(_UNDO_EVENTS, key)`, `_UNDO_MEASUREMENTS` and
  `(_UNDO_SETTINGS, scope)`. An event entry names its table: a bare entry answered for whichever
  table was on screen and put a newer step on ANOTHER recording in front of the newest step on this
  one. A press acts on the newest entry that can act ON SCREEN — **Ctrl+Z never silently changes a
  recording that is not on screen** (decision 1). An event table is on screen while it is the one
  `_undo_key()` names, a form scope while `scope[1:] == AnalysisPanel.form_key()` (the same key: a
  form is filed under the table shown), and a calibration, time base or acquisition scope while its
  recording is the Analysis panel's target. The journal is bounded to `2 * MAX_DEPTH` entries
  through one door, `_journal_undo`; past it the stacks are asked directly: the table on screen,
  then the measurements, then the settings on screen. When nothing can be taken, the menu says where
  the history is instead of going quiet, in the order of `_sync_undo_actions`: another detector of
  this channel holds steps; a ceiling dropped this table's or this recording's settings history;
  "Undo — nothing to undo on this event table" when any event table of the session holds history
  or the recording on screen holds settings that wait for another table or channel; and "Undo —
  nothing on this recording" only when no event table holds any history and every step left is a
  setting of a recording that is not on screen. Redo is per scope (decision 2): a settings step
  ends only its own scope's redo entries and never calls the tables' `clear_redo`, and a table step
  leaves every settings entry alone; the cross-stack rule between the two tables is unchanged.

  One entry can be COMPOUND (decision 3): `(_UNDO_RUN, ("form",) + K, n)`. When "Re-run detection"
  or "Re-measure depths" lands on table K, the `n` contiguous settings entries of `("form",) + K`
  immediately before it fold into it (`_note_event_step`), so one press undoes the run and the
  setting changes that led to it ("Undo Re-run detection and 2 setting changes"). Anything else
  between them — a curation, a calibration, another table's form — stops the fold, and so does the
  `(_UNDO_BREAK, K)` that `_on_form_arrived` leaves behind the form steps of a table BORN on screen:
  a first detection is no undo step, so without it the settings that produced a table's first run
  would join its second run's press. A break lives in the undo journal only and nothing undoes it.
  The press undoes the event step first, rebases the form the restore re-seeded, then the settings
  steps newest first; its redo replays them oldest first, then the event step (`_take_run_step`),
  all or nothing. A pending press keeps exactly the halves its stacks still hold
  (`_split_redo_runs` asks them): a step on the table ends the table's half, a step of the form
  ends the form's. No stack is saved with a project.

  Every Undo and Redo key list is built by `gui/key_sequences.py`, which registers each RESOLVED key
  once. On a Mac the standard Redo resolves to ⇧⌘Z, so the standard key beside an explicit
  "Ctrl+Shift+Z" was one key twice, which Qt answers with "Ambiguous shortcut overload" and nothing;
  offscreen, where Redo is Ctrl+Y, no test saw it.
- **One set of form widgets, one form per event table.** The Detection, ML and Baseline forms show
  whichever table is selected, so what they hold is filed under `AnalysisPanel.form_key()`:
  `(doc_id, channel, module_id)`, and `(doc_id, channel, None)` while the recording and channel have
  no table. `form_state()` is the snapshot — `detection`, `intents`, `ml`, `baseline`, JSON-safe,
  with neither the detector (the table on screen decides it) nor the model path (a model file can
  be gone) — and `restore_form_state()` is a SET and not a seed, because the Baseline form's
  `set_params` is a seed that refuses a confirmed choice. `intents` is what each frequency box was
  last SET to: `params()` holds what a box reads, which on a slower recording is the Nyquist clamp,
  so a snapshot of `params()` alone put the clamp back as the user's value. Every write the app
  makes itself runs under the writing guard (`_writing_form()`) and is no edit. The panel knows
  nothing of undo; it gives five signals: `formEdited` (the user edited), `formWritten` (the app
  wrote the form), `formLeaving` (before the form changes, while `form_key()` still names the old
  key, so the outgoing form is read before a Nyquist clamp touches it), `formArrived(old, new)`
  and `formStepNamed(label)` (a recall, a preset load, a Revert or a Reset names its own step).
  `MainWindow` decides what each means for the history: an edit starts a 700 ms pause, and a run,
  a switch, Ctrl+Z, the Edit menu, a save, a quit or any other step recorded inside it ends it
  (`_note_step` commits a waiting edit first, so the journal keeps the order things were done in),
  and a form with an edit waiting offers no Redo, since committing the edit ends that branch.
  Leaving files an edited form as its table's draft and drops an unedited one; arriving takes the
  draft back instead of seeding from the stored run, so a switch never throws away what was typed.
  A run the panel started from the form settles its draft, a closed recording drops its drafts,
  and the rest are saved with the project (`analysis_form_drafts_v1`, under *The project file*).
  A recording and channel's first table consumes the draft and the steps of the form that stood
  for it, `(doc, ch, None)`, wherever it is born: the panel's arrival does it for a table on
  screen, and the panel (`_table_landed_elsewhere`) and `MainWindow` (`_on_table_landed`) for one
  that lands while another recording is shown. A form the panel keeps over a new table clears its
  `_form_seed`, which belongs to an older table than the one installed under it, and a project
  restore holds the drafts across the recipe it replays, which can leave the detector combo on a
  detector that has no table.
- **A channel the user unticked is `Layer.excluded`, not a deleted layer.** `visible` answers "is
  this the recording on screen"; `excluded` answers "did the user say no". Conflating them forced the
  untick to be destructive — `show_only` re-shows every layer of the recording it switches to, so a
  merely-hidden channel came back — and that took the layer's colour and its `time_offset_s`
  alignment with it, both of which are persisted. `excluded` is an additive, optional manifest key:
  absent means False, which is exactly what an older project meant.
- **A stored fit names the table it was made on.** The `ResultStore` keeps ONE `fit.dwell` and ONE
  `fit.depth` per `(doc_id, channel)`, so every table of a recording (Schmitt, CUSUM, Keras,
  Reconciled) shares one fit slot, and a fit used to carry nothing that said which of them it
  described: the Results dock's Fits tab printed Schmitt's n = 316 under CUSUM's table, beside a
  merged table and under a re-run, while Export figure refused the same object as stale.
  `fit_dwell` / `fit_depth` now record the `module_id` of the table they were handed in
  `FitResult.meta['detection_module_id']` — an added key in a dict the project already writes
  wholesale, so no version bump and an older build carries it through untouched — and
  `modules.judge_stored_fit` is the ONE rule for "does this fit describe the table on screen": the
  export's two guards (`stored_fit_problem`: calibration stamp, accepted count) plus that lineage.
  A Fit press replaces the PAIR (a half that fails retires its predecessor from the store), and the
  Fits tab is redrawn by one helper from every route that changes what it sits beside. **The
  conservative rule for a fit that names no table** — anything a project saved before the stamp,
  anything the Plots window files — is that it is *kept* (store, project, and the figure export,
  exactly as before) but never *assumed* to match: the tab shows a sentence in place of its numbers
  until Fit is pressed. The accepted COUNT was the export's only proxy for "the same events", and
  it cannot see a curation that swaps one accepted event for another, or an event resized or
  re-measured in place. So `fit_dwell` / `fit_depth` also record a digest of the values they were
  handed (the `FitResult.meta` keys under *The project file*), and `judge_stored_fit` and
  `stored_fit_problem` compare it when a fit carries one (the owner's decision 10); a fit without
  one keeps the count rule.
- **What the analysis windows may plot** is `core/sources.discover_plot_sources` — a PURE (Qt-free,
  Session-free) rule that turns the open documents plus a result lookup into a list of `PlotSource`
  value objects: one per `(document, channel)` with a stored detection, segments included, plus the
  manual-measurement table. Each carries its OWN detection provenance (so a per-source fit resolves
  its own `t_min`), its own Okabe-Ito colour and its own calibration fingerprint. `MainWindow` calls
  it; the windows never walk the Session themselves.

## The main window and the Plots window: one class in several files

`gui/main_window.py` (`MainWindow`) and `gui/plots_panel.py` (`PlotsWindow`) were each one class in
one file, of 17,874 and 16,095 lines. Each is now one class in several files, and the original
module is 10,015 and 9,218 lines. Methods were cut out of the class and pasted, unchanged, into
plain-Python mixin classes in sibling modules, which the class lists before `QtWidgets.QMainWindow`
in its bases; the module-level names the parts share went to a leaf module that the original module
imports back name by name, so `nanoscope.gui.main_window.X` and `nanoscope.gui.plots_panel.X` still
resolve for every name that moved. Compared as syntax trees with the two files as they were
(imports, docstrings and the class header aside), every class-body and module-level statement is
present, unchanged, exactly once across the new files: nothing was lost and no method changed. What
changed is where a method is defined: its `__module__` and `__qualname__` name the mixin
(`_ProjectMixin._open_project`), `inspect.getsource` finds it through the MRO, and the MRO lists the
mixins.

`MainWindow` has 579 callables (methods, static and class methods, properties): 316 stay in the
class, 263 are in the seven parts.

- `main_window.py` — `__init__`, the menus and `_wire`; loading, detection and batch detection;
  selection, hand-made events and segments; the curation hosts and Signal Conditioning; the refusal
  ledger, merge and level fit; the subpopulation run; the undo journal and the dirty state;
  `_finalize_restore`; the close guard and teardown; every Qt override; every `@Slot`.
- `main_window_figures.py` (1,380 lines) — the Analysis Explorer, the figure surfaces and their
  sources, recipes and saved figures, the Figure Library, figure capture and export.
- `main_window_exports.py` (1,577) — the level-fit reconciliation; the exports of events,
  provenance, levels, density, training data and report figures.
- `main_window_surfaces.py` (1,297) — the surface gates, the analytical workspace, Analysis home
  and the guided first analysis, the openers of the Plots, Event plots, study, I-V,
  voltage-dependence and PSD windows.
- `main_window_subpop.py` (681) — the subpopulation report window, its assignment on the trace and
  the gallery, the report export.
- `main_window_project.py` (1,604) — gathering a project, Save, Save As, the write, Open and the
  restore of each document, the recent-projects menu.
- `main_window_layout.py` (861) — child windows with remembered geometry, the Window menu, Reset
  window layout, saved settings, dock widths and corners.
- `main_window_recovery.py` (475) — crash-recovery autosave: store and scheduler, snapshot
  hand-off, health line, the offer at start-up, recover and discard.
- `main_window_shared.py` (387) — the leaf: 51 names (constants, helpers and two classes, `_InfoBar`
  and `_RecoverySnapshot`). Three of the seven parts import from it (layout, project, recovery).
- `main_window_seams.py` (32) — call-time forwarders for `ask_export_path`, `locked_project_read`
  and `_on_screen`.

`PlotsWindow` has 428 callables: 186 stay in the class, 242 are in the six parts.

- `plots_panel.py` — `__init__` and the construction of the widgets, the seven signals and the one
  `@Slot` (`_on_notes_ready`), `closeEvent` and `style`, the public setters and getters, the five
  draw methods (`_redraw_current`, `_draw_histogram`, `_draw_scatter`, `_draw_kde`, `_draw_pca`),
  `save_image` and `export_data`, the Configuration history and the templates; at module level the
  eleven dataclasses it defines (`PlotConfig` and its sections, `StyleSettings`,
  `AnalysisIdentity`), `_ConfigHistory` and the JSON codec. A saved analysis workspace records a
  dataclass as `module:qualname` (`core/analysis_workspace._type_name`), so these stay here and what
  is written stays `nanoscope.gui.plots_panel:<Name>`.
- `plots_constants.py` (892 lines) — the leaf: 103 names (column, plot-type, fit, normalisation,
  colour and unit constants, `FrozenMap`, the text and legend key helpers, the quantity helpers and
  the small value and pyqtgraph classes the window draws with). All six parts import from it.
- `plots_fit_mixin.py` (1,942) — the fit overlays, summaries and the jobs behind them, the
  goodness and rival-family notes, the fit range, the component-count ladder, the Nonlinear fit hook.
- `plots_annotation_mixin.py` (1,006) — the fit label and the legend, the provenance caption, the
  confidence band and the residual strip, the detection-floor shading, the message and Details
  writers, click and lasso selection.
- `plots_sources_mixin.py` (1,151) — data in (`set_data`), the source catalog, the column pickers,
  the control-change handlers, the RUN gate and the fit-apply gate, current-frame access.
- `plots_style_mixin.py` (915) — the Style group's handlers and helpers, axis labels and title, the
  axis-range boxes.
- `plots_notes_mixin.py` (805) — the histogram's bin edges and normalisation, the KDE overlay, the
  notes about the columns and sources drawn.
- `plots_views_mixin.py` (651) — the heatmap, the clustering and PCA runs, and the numbers behind
  the picture (series labels, the plot data table, PCA loadings).

The rules hold for both windows, and `tests/gui/test_main_window_parts.py` and
`tests/gui/test_plots_window_parts.py` guard them.

- **No `@Slot` leaves the class.** A `@Slot` in a mixin registers on the window's meta-object, but
  `signal.connect(self.<slot>)` then goes through PySide's global receiver, which Qt does not remove
  when the window's C++ object is deleted: with `_on_application_state_changed` in a mixin, a
  deleted `MainWindow` was still a receiver of `QGuiApplication.applicationStateChanged`. Two rules
  hold it: no part has a decorated slot, and every slot on the window's meta-object has its function
  in the class itself (that one does not depend on how `Slot` is spelled).
- **A part is a plain class**: one `*Mixin`, no base, no `__init__`, no `super()`, no `Signal`, no
  member named like a `QMainWindow` member (a Qt override a mixin cannot supply, or a Qt method it
  would shadow), and no name that another of the window's classes also defines. It is not
  stateless: three of the main mixins hold ten class-level constants (`_ExportsMixin` 3,
  `_SubpopMixin` 6, `_ProjectMixin` 1) and two of the plots mixins hold ten (`_PlotsFitMixin` 8,
  `_PlotsAnnotationMixin` 2); they moved with the methods that read them.
- **Imports.** A main part never imports `main_window` or another part at module scope, and holds
  neither them nor `MainWindow` nor another part's mixin as a module-level name; a plots part never
  imports `plots_panel` or another part, in any spelling or at any depth. A main part may import
  `main_window_shared` and `main_window_seams`, a plots part `plots_constants`; the leaves import
  none of the window's modules.
- **The names tests patch.** Tests patch `ask_export_path`, `locked_project_read` and `_on_screen`
  on `nanoscope.gui.main_window`: a main part that calls one by its bare name imports the forwarder
  of that name from `main_window_seams`, which looks it up on `main_window` when it is called, and
  `main_window.py` keeps all three. The names tests patch on `plots_panel` (`confirm_large_svg`,
  `ask_export_path`, `choose_live_plot_export`, `ImageExporter`, `kde_curve`, `PLOT_CONFIG_VERSION`,
  `_LEGACY_NORMS`, `write_live_plot_image`) are read only in `plots_panel.py`. A part that read its
  own copy would silently stop being reached by the patch.
- **One object, not two.** Every name a leaf defines is the same object on the original module (51
  on `main_window`, 103 on `plots_panel`); nothing rebinds one afterwards.
- **Order.** Every part file is a base of its window, ahead of `QMainWindow`.
- **History.** Checkpoints are made only in `main_window.py` (see *State model*, Undo).
  `_finalize_restore` makes one, so it stays there; `main_window_project.py` clears the recorders
  and suspends them while a project restores.
- **The shell guards.** The shell modules, the Explorer, Analysis home and the analysis catalog may
  import no page module. Every `main_window_*` file, `plots_constants` and every `plots_*_mixin`
  file is a page module for them (`plots_series`, a shared helper that predates the split, is not):
  `test_p1_guards.py`, `test_p1_shell.py`, `test_analysis_home.py`. `test_measure_navigation.py`
  pins what each plots part may reach: the leaf and widgetless helpers, no panel.
- **Source-reading guards read the parts as well as the original module.** For `MainWindow`:
  `test_fup_wiring_and_suite_green.py`, `test_clock_invariant.py` and, for the history,
  `test_undo_events.py` (every `gui/*.py` but `main_window.py`). For `PlotsWindow`:
  `test_plots_undo_templates_json.py`, `test_s7_fixup_plots_panel.py`, `test_subpop_views.py`,
  `test_p1_guards.py`, `test_p0_final_seams.py`, `test_shared_invariants.py` and
  `test_s6_plots_undo_depth_disclosure.py`. A few tests read text that must stay in
  `main_window.py`: the connections of four `EventSet` mutation signals to a checkpoint
  (`test_undo_events.py`), `_run_subpopulations` through `_on_subpop_finished`
  (`test_subpop_gui.py`), and `_on_analysis_failed`, whose modal the guard records by file name
  (`test_modal_guard.py`).

## The reader door — `io/contract.py`

Four readers decode four formats (`abf`, `hdf5`, `tdms`, `text`), and the recurring defect was never
in the decoding: it was **four different answers to one question**. `io/contract.py` is where the
questions and their single answers live. Nothing in it decodes anything — it is deliberately the
boring half of `io/`: the sentences, the clamp, and one logging bridge. All four readers import it.

The four questions, each with the defect that put it there:

1. **How many samples are really here?** A truncated TDMS reported whatever nptdms recovered with
   `issues=()`; ABF correctly refused. That number is the denominator of every event rate, capture
   rate and duration in the app, so a short read silently re-attributes the whole result to a
   recording the user does not have. `short_read_issue` is the one sentence for it.
2. **What is the sample rate, and how sure are we?** The text and HDF5 readers measure `dt` from the
   first ~1000 rows and then describe the whole file with it: on a CSV whose interval changes at row
   1500, 0.0600 s was reported for a 0.3300 s recording — 5.5× wrong, `issues=()`. Changing how the
   rate is derived would move every dwell time on every well-formed file, so the fix is not the rate:
   it is *saying that the rest of the file disagrees* (`rate_disagrees_issue`). The question has a
   second half — *in what unit?* — answered in three tiers (the header rule in `io/text.py`, the
   two rate bounds in `io/base.py`): a CSV header that names the unit (`time_ms`) is a declaration
   and is read as such; an undeclared rate below `MIN_PLAUSIBLE_RATE_HZ` (1 Hz) is refused, not
   guessed, but the interval is kept (`RecordingMeta.refused_rate_hz`) so the unit can be declared;
   and an undeclared rate below `SUSPECT_UNDECLARED_RATE_HZ` (1 kHz) — the millisecond column,
   which looks cleanest of all — is kept and put to the user as an Issue. The user settles it with
   a `core/sample_rate.py` declaration (unit or rate), which the Session applies by rebuilding the
   document's frozen meta so every consumer sees it (segments follow their parent), and
   `core/sample_rate_state.py` saves in the project.
3. **Which channels exist, and why was one rejected?** HDF5 silently dropped a `/current` stored as
   int32 raw ADC counts, or shaped `(1, N)` — absent from the Files list with nothing anywhere saying
   why. A refusal the user cannot see is indistinguishable from a file that never had the channel
   (`rejected_channel_issue`).
4. **What encoding is this text in?** The probe opened with `errors="replace"` and `pd.read_csv` did
   not, so a Latin-1 CSV off Windows instrument software probed clean with a channel called `curr?nt`
   and then failed at load (`fallback_encoding_note`).

Plus the structural rule all four re-implemented, three of them differently — **the load window has
two halves**:

- `meta.loaded_range` is derived from what was **read**, never from what was **asked**. The text
  reader assigned the request verbatim, so `(500, 8000)` on a 2000-row CSV produced `n_samples=1500`
  beside `loaded_range=(500, 8000)` — a value that then went into the project manifest and was
  replayed on restore. **`window_loaded` is the only writer of that field**, and one level below the
  readers `core.types.RecordingMeta.__post_init__` refuses any meta whose window and sample count
  disagree, so a fifth reader cannot get it wrong quietly. `short_window_issue` is the separate
  sentence for "the request outran the file" — the file is not damaged, and saying so matters on a
  project restore whose recording has been re-exported shorter.
- `meta.t0_s` is the half nobody unified: it is the experimental time of the **first loaded sample**,
  because that is what `Session.x_axis` builds (`arange(n)/fs + meta.t0_s`), what the Event Inspector
  inverts, and what `core.segment` rebases against. HDF5 read it at the window's own index and the
  other three left it at the file's origin, so the same window of the same experiment carried two
  different clocks depending on the format (measured on the owner's ABF: a window starting 20 s in
  reported the file's own 61.303 s). `window_t0` is the one answer, and `start == 0` returns the
  file's `t0` unchanged so a whole-file load is bit-identical to what it always was.

`ReaderReport` is the accumulator both `probe()` and `load()` write into. It keeps **issues**
(non-fatal *problems* — the recording is not what it looks like) apart from **notes** (honest
*advisories*), deduplicates (probe and load share one scan, and a rebuilt `RecordingMeta` must not
say the same sentence twice), and both halves are rendered verbatim by `gui/file_info_panel.py` —
the **Inspector ▸ File Info** tab, in its own **Notes** and **Issues** boxes, NOT the Files dock,
which lists recordings and channels only. That rendering is the entire point, since a problem that
reaches only Python logging has not been reported to anybody.
That is also what `decoder_messages` exists for: nptdms recovers what it can from a truncated file
and says so *to the `nptdms` logger*, i.e. to a terminal nobody has open. It attaches a
thread-scoped handler rather than reconfiguring logging, lowers a logger's level only when the
effective level would have swallowed the warning outright and only downward, and puts it back.

## The analysis seam

`core/analysis.py` declares the extension points that let analysis modules plug into the GUI without
either side importing the other:

- `AnalysisModule` / `AnalysisContext` — a channel-in, result-out protocol. One `register_all()` at
  app start wires up the classical `analysis.modules.DetectSchmitt` and `DetectCusum` modules plus
  the runtime-loaded Keras `analysis.ml_modules.MLDetect`. Registration is idempotent (a second test
  session shares the process-global registry), and importing `ml_modules` only *probes* for
  TensorFlow rather than importing it, so start-up stays cheap.
- `AnalysisResult` carries a payload (e.g. an `EventSet`) plus a `calibration_fingerprint` for
  staleness detection.
- `OverlayHint` (int64 sample-index arrays) + an overlay-renderer registry (`gui/overlays.py`) turn
  analysis output into trace graphics. Detected events render as **one batched span item**, not N
  region items.
- `ResultStore` on the `Session` holds results; `resultsChanged` drives the UI.

Detection runs off-thread through `gui/analysis_worker.AnalysisController`; results are installed on
the main thread.

### Background work: one seam, four rules

Every background job in the app — a load, a probe, a detection, an ML run, a surrogate study, an
open-pore probe, a re-measure, a batch pass — is started and retired through
`gui/worker_thread.py`. That module is where the thread-lifetime rules live, once, instead of three
times over in the three controllers:

1. **A worker thread is never terminated**, and no wait is ever unbounded. A thread that will not
   stop is *detached*: severed from its controller, parked, and left to wind down with nothing
   listening. `retire_worker_thread` returns whether it actually stopped, and the caller must not
   claim the work was cancelled when it was not.
2. **Cancellation is an ambient token** (`analysis/cancel.py`), installed by the worker and polled
   by every kernel underneath — no call site forwards it, so no call site can forget it. Anything
   long-running added later must poll.
3. **One analysis at a time, app-wide** — THE RUN GATE. `AnalysisController` and
   `BatchDetectController` write into the same `ResultStore`, so two of them in flight is two
   answers for one key with the last writer winning. Every job registers a `domain` and a `label`
   as a side effect of `start_worker_thread` (both are *required* keyword arguments, so a
   controller written later cannot fall outside the gate by omission); `blocking_job` decides, and
   `refusal_text` writes the sentence. A refused start emits `refused(str)`, returns `False`, and
   **stops nothing**. Two jobs of the same domain with the same owner are allowed — a controller's
   own `start()` aborts its predecessor first, which drops its answer. `abort()` emits no terminal
   signal, so the controller says so itself: `AnalysisController.replaced(label)` goes out from the
   one place a job is replaced, after the old job is gone and before the caller has recorded the new
   one, and `AnalysisPanel` forgets what it recorded about the dropped job there. Left behind, a
   Merge-all check-list or a re-measure target reads as "an operation is running" to
   `study_event_bridge._assert_idle` for the rest of the session, and a probe button keeps
   saying "Measuring…". The dropped worker is kept alive until the signals it had already queued
   have been delivered: the sender guard compares objects, and a recycled address would let a
   stale answer end the job that replaced it.
   Loads are a separate domain and are deliberately not exclusive with analyses.
4. **Quitting never aborts the process.** `MainWindow._teardown` aborts every controller, syncs
   `QSettings`, and *uses* `drain_detached`'s return value; `gui/app.main` gives anything still
   running one more bounded chance and then leaves via
   `worker_thread.exit_without_destroying_live_threads`, because a C++ `QThread` destroyed while
   its OS thread runs calls `qFatal` (measured: SIGABRT, 4/4 runs, plus a macOS crash report).

The gate is deliberately NOT expressed as a disabled Detect button: Qt never delivers a click to a
disabled widget, so a control greyed out by a batch could not tell the user it was the batch. The
click lands and the refusal names the run that is in the way.

## Rendering

- pyqtgraph with peak-preserving **min/max decimation** (`setDownsampling(method="peak")`) so spikes
  survive zoom-out. There is no custom decimation pyramid — measured unnecessary at the sample
  ceiling.
- Annotation items added to a pane (event spans, the measure region, markers) use
  `ignoreBounds=True` so they never drive the y-autorange (a real bug fixed once: view-tracking items
  otherwise ratchet the axis wide).
- The trace's x-range is owned explicitly by `TraceView` (load/reset set the full data extent);
  autoRange is not relied on because it interacts badly with `clipToView`.

## Pipeline

```
open_document (io) ──▶ Session.add_document
                          │
   Calibration ◀── calibrated(doc, ch)  (raw × gain + offset, cached, float32)
                          │
   Shared recipe: notch → median → [high-pass] → baseline-detrend → running-mean → SavGol → Normalize → low-pass
                          │
                          ├─ Candidate-localization branch: includes enabled high-pass
                          │    └─ Schmitt/CUSUM → candidate start/end spans
                          │
                          └─ Quantitative branch: same recipe with high-pass forced OFF
                               └─ characterization baseline/noise + characterize(candidate spans)
                                  → 20-column EventSet (depth, fraction, SNR, ECD, rise/fall, n_levels)
                          │
   ├─ Event table + span overlays + summary (gui.analysis_panel)
   ├─ Event inspector (gui.event_inspector — per-event waveform + curation)
   ├─ Manual measurement (analysis.measure — cursor-selected stats, own schema)
   └─ Analysis Explorer (gui.analysis_explorer — lazy tab shell, shared context/freshness/refresh)
      └─ Plots + ML (analysis.populations + analysis.stats; gui.plots_panel — hist/scatter/PCA/clustering/fits,
        MULTI-SOURCE: core.sources.discover_plot_sources offers every open recording/segment with a
        detection, keyed by segment.persistent_key, so two pores overlay with shared bins, a legend
        and ONE FIT PER SOURCE resolved from that source's own provenance; Scatter/PCA points retain
        source-row identity for MainWindow's event-table + trace navigation)
```

## Analysis core modules

| Module | Responsibility |
|---|---|
| `analysis/filters.py` | Zero-phase low-pass (Bessel), high-pass, notch, Savitzky-Golay, running-mean, median; MAD noise; rise time; PSD. |
| `analysis/baseline.py` | `BaselineConfig`; the two-pass masked-median open-pore estimator (kernel imported from `ml_preprocess`), the block-median/percentile/ASLS + PCHIP detrenders, `measured_polarity`, `baseline_health`, `remove_baseline_wander`, and `characterisation_baseline` — THE door. |
| `analysis/baseline_probe.py` | Measures a recording's own timescales on the no-high-pass characterization trace and **proposes** `rough_s`/`base_s` from them. Reports; never applies. |
| `analysis/detect.py` | Schmitt two-threshold detector; measured polarity; merged-exclusion analyzed-duration. Candidate-localization kernel; the module adapter owns the high-pass split. |
| `analysis/detect_cusum.py` | One-sided sequential candidate detector with changepoint onset/exit estimates; the module adapter gives it the same high-pass split and event schema as Schmitt. Each run also counts what the same finder and gates report on the negated deviation and stamps `n_opposite_events`, `n_opposite_isolated`, `n_opposite_reference` and `opposite_guard_s`; `modules._build_warnings` raises the advisory from these keys alone: `n_opposite_isolated` at least 5 and at least 10% of `n_opposite_reference`, or, failing that, `n_opposite_events` at least 5 and at least half of it (dense firings at a very low `h`, where the isolated count collapses; that count is then the one shown). All are frozen at detection, so a merge, a hand edit or a re-measure cannot change the sentence. With `hp_enabled` in the provenance it never raises the noise sentence: on the high-passed candidate trace the far edge of each event also fires in the opposite direction, so the count is not a noise estimate there and the sentence says that instead. A count, never a rate, and it gates nothing. The opposite pass costs little at default settings; at a very low `h` a run can take up to about twice as long. |
| `analysis/merge.py` | Post-detection sub-event merge (window-based and selection-based), re-characterizing each merged span through `characterize`; the occupancy floor that refuses a merge whose span would be mostly open pore. Reuses `detect.group_starts_by_gap`, so the gap rule is not re-derived. |
| `analysis/characterize.py` | Per-event dwell, **median** depth, ECD, SNR, rise/fall, sub-state count → the events DataFrame. |
| `analysis/events.py` | The frozen 20-column event schema + `EventSet` (accept filtering, pooling that refuses cross-pore). |
| `analysis/lineage.py` | What a stored result was made with — `made_with` reads the rate, calibration and estimator stamps, the chain and the thresholds it already records, and adds none — and one judge of each against now: `judge_inputs` gives every input its own verdict (same, changed, unknown) with the action that repairs it. It reuses the amber line's rate rule and the chain notice's chain and threshold rules rather than restating them, so the Details list and the notice cannot disagree. Pure, Qt-free. |
| `analysis/stats.py` | Censored-MLE exponential dwell fit; hand-rolled Gaussian-mixture amplitude fit; log-bin histograms with the Jacobian. |
| `analysis/populations.py` | GMM/k-means clustering (BIC k-selection), PCA, and linear/power-law fits — all scikit-learn/NumPy. |
| `analysis/measure.py` | Manual-selection statistics (dwell, amplitude by method, skewness, kurtosis, …) + a 28-column measurement schema (`MEASUREMENT_SCHEMA`), separate from the frozen 20-column event schema. |
| `analysis/truth.py` | Synthetic trace + AAFT surrogate generators used to validate detection (recall/precision) and for null controls. `null_control` takes the detector to run (`detector=`, Schmitt when omitted); `sweep_threshold` and `recovery_map` are Schmitt only. |

### Classical high-pass is a dual-signal adapter contract

`preprocess_for_detection` still builds the candidate signal and the low-level Schmitt/CUSUM
kernels remain single-signal detectors. The split belongs to `analysis.modules`, at their shared
adapter boundary:

1. Build the user's complete localization recipe, including high-pass, and let the selected kernel
   return candidate spans plus its threshold provenance.
2. Release that recording-sized temporary, reconstruct the recipe through `detection_signal` with
   only `hp_enabled=False`, and estimate the characterization baseline and MAD noise there.
3. Call `characterize` on the original spans. `baseline_pA`, `depth_pA`, `frac_depth`,
   `mean_dev_pA`, `max_dev_pA` and SNR therefore describe the no-high-pass branch. High-pass itself
   no longer erases the DC reference, although every other configured stage still applies (notably
   the dimensionless Normalize stage). ECD also uses its
   samples, but its integration limits — like start/end, dwell and rise/fall — still come from the
   high-pass-derived candidate span.

Auto polarity is anchored in the characterization domain. The first candidate pass may vote on the
high-pass trace as an optimization; if that vote disagrees with the no-high-pass physical vote, the
adapter drops the candidates and repeats localization once with the physical sign. Both votes and
the retry count remain in provenance. A user override is used on both paths.

New runs stamp `highpass_localization_only` / `characterization_hp_omitted`, preserve candidate
baseline/noise under `detection_*`, and reserve top-level baseline/noise/polarity for the values in
the event rows. `detection_signal` treats an old high-pass result without that marker as legacy and
reconstructs its original high-pass measurement signal. This preserves reproducibility instead of
silently upgrading old candidate-only amplitudes; the UI warns that those rows must be re-run.

`analysis.conditioning.conditioned_trace` is the localization/orange-preview builder.
`characterization_trace` delegates to it after disabling only high-pass; the baseline probe uses
this second door. The selected-stage and combined-response plots describe the localization branch
when high-pass is enabled. Normalize is intentionally **not** omitted on the quantitative branch:
its documented dimensionless-output caveat remains in force.

## The baseline seam — one config, one door, one fingerprint

Every blockade depth in this app is a distance from a baseline, so "which baseline?" is not a detail.
The app used to answer it four times: a 10 ms `block_median` inside `detect.py`, a 200 ms window in
`ml_modules.py` (widened to fix a real zoom-dependence defect, but only there), a ±3 ms flank median
in `characterize.recharacterize_span`, and the 10 ms default again in `truth.py`. One event on the
owner's −600 mV apo recording therefore read **32.04 pA** (Schmitt), **37.54 pA** (rebuild) and
**67.69 pA** (ML), and the population median depth was 23.23 pA *or* 111.31 pA depending on which
detector found it.

- **`BaselineConfig` is the single owner of the estimator's shape** — method, `rough_s`, `base_s`,
  `decim`, `mask_sigma`, `dilate_ms`, `window_s`, `despike_size`. One object flows from the GUI
  through detection, characterisation, ML characterisation and re-characterisation. Deliberately NOT
  on it: `percentile` and the four `asls_*` knobs, which tune a *detrender*, not the I0-tracking
  shape that has to be identical across the four call sites.
- **`baseline.characterisation_baseline` is THE door**, and `baseline.py` names every
  baseline-parameter literal in the package. Parameters alone would have fixed four sites, not the
  *class*: `estimate_baseline` is public and takes `window_s=` with a silent default, so the fifth
  site to hit a baseline problem would fix it locally exactly as `ml_modules` did — for a good
  reason, with a measurement behind it — and the app would report two depths again. The AST guards in
  `tests/analysis/test_baseline_seam.py` fail the build if another module calls `estimate_baseline`,
  defines a baseline-window constant, or hands the door a literal window.
- **The kernel is imported, never re-implemented.** `baseline.py` pulls `baseline_masked_median`,
  `baseline_robust_median`, `_decimated_median` and `despike_median` out of the frozen
  `analysis/ml_preprocess.py`, which is bit-exact against the owner's persisted training corpus and
  already hardened against the non-deterministic 1-D `scipy.ndimage.median_filter`. A second copy of
  a twenty-line function with five silent bit-exactness traps was judged the single largest risk in
  that change; `tests/analysis/test_masked_baseline.py` asserts there is exactly one definition of
  `_decimated_median` in the tree.
- **`BaselineConfig.fingerprint()` follows the `calibration_fingerprint` precedent exactly.** Rows
  carry the stamp (in `provenance`, via `to_provenance` — never a new column, the 20-column schema is
  frozen); a stored result whose stamp is absent or differs from the live config is marked STALE and
  the user is offered a re-run. Nothing is ever silently re-measured, because `depth_pA` moving 15×
  between two openings of the same project is indistinguishable from data corruption. `fs` is folded
  in by resolving `decim` first, so two recordings at different rates cannot share a fingerprint
  while running different strides. `from_provenance` always answers (callers need a config);
  `reproducible_from` **refuses** — returning `None` for a pre-change row or one whose detector was
  handed a baseline array — because a rebuild that guesses is worse than one that falls back and says
  so.

- **A recording too short for the configured window is FITTED, never substituted.**
  `adapt_to_length(config, n, fs)` is a three-rung ladder, and each rung is named in `provenance` as
  `baseline_fallback`: (1) the configured estimator fits and nothing is touched — **the only rung a
  normal-length recording can reach, which is why no shipped number moved**; (2) `FALLBACK_CLAMPED`,
  the *same* estimator with `rough_s` clamped to the whole recording and `base_s` scaled to preserve
  the user's own `base_s/rough_s` ratio, `decim` re-derived; (3) `FALLBACK_TOO_SHORT`, a global
  median under its true name, `method='constant'`. What this replaced was a substitution: the old
  `_length_fallback` stood the pre-change 10 ms block median up in place of the configured
  estimator, i.e. the fallback that existed to avoid a degenerate answer *was* the degenerate
  answer. Measured on a synthetic trace with known truth (baseline −35 pA, σ 4.5 pA, one 20 ms
  −88 pA blockade), depth reported as a fraction of the true 88 pA at **n = 3072** — the owner's own
  ML window: **0.0094 before, 1.02 after**; at n ≥ 5000 both agree to four decimals. Below ~2000
  samples the clamp collapses too, for the documented reason (`rough_s` must be ≥ ~3–4× the longest
  state it is meant to expose, and a 20 ms event fills a 30 ms window) — that regime is **not
  claimed to be sound**, it is warned about, and it is not retuned, because tuning
  `MASK_FRACTION_CEILING` against one recording is the auto-tune this module refuses.
  **Clamping is not the auto-tune R5 forbids:** there is no candidate set, no metric is consulted,
  and the ratio that decides the two-pass structure is the user's own. `BaselineResult.fallback_token`
  carries which rung was taken; `provenance()` used to hardcode `FALLBACK_TOO_SHORT` whatever
  happened, so no consumer could tell a clamped measurement from an untracked one.
  `estimate_baseline_result` still **refuses** a too-short window — it is the estimator's strict
  contract and the corpus runs through it — but names the config that would work and points at the
  door. `'constant'` is exempt from the length check in both: a global median reads no window, and
  `estimate_baseline` answered a 200-sample `'constant'` call while `estimate_baseline_result`
  refused it.

### `analysis/baseline_probe.py` — propose, never decide

There is no defensible global default. The apo `_0006_` file has a **521 ms partial-clog state at
t = 37.13 s**, and a 500 ms rough window sitting inside a 521 ms state *is* that state: nothing gets
masked and the baseline lands 22 pA off (−56.6 against a true −34.5). It needs `rough_s = 2.0`.
`ferritin_data3` is a gating pore stepping 296 pA with a ~35 ms level dwell, where a 2 s window
cannot follow the steps at all and the reference notebook uses `rough_s = 0.03` — 67× apart across
four real recordings. So the app **measures each recording and proposes**: `probe_timescales` returns
a `BaselineProbe` (a robust one-sided sigma, a dwell distribution, a gating verdict), and
`propose_baseline` turns it into a `BaselineProposal` — a config, a `param_source` of `'proposed'` or
`'default'`, and a `reason` written to be shown verbatim beside the numbers. It reproduces the known
good parameters on three of the four recordings from the data alone.

Nothing here is ever auto-applied, for two measured reasons rather than stylistic ones. First, the
substitute sigma estimator is defensible but over-reads 3–4× on a gating pore (the obvious
delta-based sigma reads 7× *low* on these already-low-passed exports, which is why it is not used).
Second, **the reference's own QC triple is a degenerate objective**: `qc_open_pore_level` / `drift` /
`mad_noise` are monotone in window length, so optimising them selects the shortest possible
window — a degenerate `0.01/0.002/1` set beats the shipped parameters on all three for every corpus
file. Auto-tuning against them would reproduce the exact failure the seam removes. The defence is
therefore architectural: a human confirms.

The GUI feeds the probe the whole **characterization trace**: the live conditioning recipe and
low-pass with high-pass forcibly omitted. This keeps the proposed windows in the same domain used
for quantitative baseline and amplitude measurement even while the orange localization trace and
combined response include high-pass.

The probe also may not construct a `BaselineConfig`. Every config it returns is
`DEFAULT_BASELINE.replace(...)` touching only `rough_s` / `base_s` / `decim`, which makes it
structurally impossible for a probe to invent a `mask_sigma`, a `dilate_ms` or a `despike_size`, and
keeps `DEFAULT_BASELINE` the one literal the seam guard protects. An undecidable probe returns
`DEFAULT_BASELINE` *by identity*. `decim` is always **derived**, never proposed. The probe runs off
the GUI thread through the ordinary run gate (`label="baseline_probe"`) — measured 106 ms at 1.2 M
samples and 579 ms at 8.6 M — and `proposal_summary` builds the dialog's rows, so the sentence the
user reads and the numbers the estimator runs come from one function over one object.

## Load-bearing invariants

Preserved deliberately; changing them silently corrupts the science:

- **Detected (not assumed) polarity**; work in rectified space, never rectify stored data. For a
  high-pass classical run, Auto polarity is owned by the no-high-pass characterization branch; an
  opposite vote from high-pass ringing triggers one forced-sign localization retry.
- **The mask direction is a property of the recording, not of the parameters.**
  `baseline.measured_polarity` votes across a fixed ladder of detrend scales, drops any scale that
  measured nothing (`derive_polarity` returns `+1` when it has no evidence, and that default was
  being reported as a measurement), and ignores the `BaselineConfig` it is handed. It used to
  detrend at the configured `rough_s` with a stride derived from `base_s`: on `ferritin_data3`,
  stride 264 answered −1 and stride 333 answered +1, and the one-sided mask pointed backwards moves
  the baseline by 312 pA.
- **No schema column ever holds ±Inf.** `EventSet` demotes an overflowed magnitude to `NaN`, this
  app's "not measured" sentinel; the row keeps its `non_finite` flag and its rejection. `±Inf` is
  the only non-value that survives `dropna()`.
- **A control checks its own premise.** `truth.null_control` verifies that its AAFT surrogate is a
  null of the recording it came from, and reports no rate rather than a wrong one. Phase
  randomisation is a null for a stationary spectrum plus Gaussian noise; a drifting real pore is
  not, and the control was returning `estimated_fdr = 1.00` on every real recording measured.
  The control runs the detector it is given, so the GUI's Null control follows the selected
  detector. Under CUSUM the result also carries the real run's opposite-direction count
  (`n_opposite`) and resolved `polarity`, because the event-free count is a weak lower bound for a
  detector whose excised detections include its own noise firings; the Statistics readout leads
  with that count. The GUI stamps each stored result with its `detector` and `thresholds` (for
  CUSUM `cusum_k`, `cusum_h` and `filter_cutoff_hz`) and shows it only with a table of that
  detector; under another detector's table the readout names the stored result and the table it
  shows with, and says "not measured" only when nothing is stored. The GUI refuses, through
  `runRefused` and before anything starts, a CUSUM control while High-pass is enabled (the
  opposite-direction count is not a noise estimate there, and an Auto vote on the high-passed
  trace can pick the wrong sign) and a detector id that is neither `detect.schmitt` nor
  `detect.cusum`. A recovery map is Schmitt's, so it weights the dwell fit of a Schmitt table
  only; one that lands under another table is kept and a status line says so.
- **MAD** noise, never `std` (std absorbs the events being detected).
- **Median** blockade depth, not max (max is biased by dwell → spurious dwell–depth correlation).
- **Two-threshold** Schmitt with `k_out < k_in` (single threshold shatters events).
- **PCHIP, never a cubic spline**, whenever block knots are expanded back to full rate
  (`baseline._expand`) — Pchip is shape-preserving and cannot overshoot; a natural cubic spline
  rings at a drift step and the overshoot lobe is a phantom deviation detection reports as an event.
  This is the *detrending* family (`block_median` / `percentile` / `asls`) and the stored-result
  compatibility path. The estimator every depth is now measured against is the two-pass masked
  median — see *The baseline seam* — which interpolates nothing.
- **Censored-MLE** dwell fit (a naive `expon.fit` is biased long by `t_min`).
- **Zero-phase** filtering everywhere (no phase delay). This does not promise identical threshold
  crossings: a filter can reshape an edge, which is why high-pass-derived boundaries, dwell,
  rise/fall and ECD limits remain explicitly disclosed.
- Signals stored **float32 contiguous**; time axis **float64**.
- **A recording's samples are read-only in memory.** `TraceDocument.__post_init__` sets the
  read-only flag on every channel array (`_freeze_samples` — the flag, never a copy), and
  `Session.calibrated_array` freezes each array it caches. `frozen=True` froze the attributes, never
  the samples, and at an identity calibration the session hands out the recording's OWN array, so
  one in-place `arr -= baseline` anywhere changed the recording for every curve, panel and worker,
  silently, for the rest of the session. Now that write raises `ValueError: assignment destination
  is read-only` at the line that tried it — the contract the worker snapshots already kept
  (`FrozenChannelSource.capture`). Segments and `dataclasses.replace` pass through the same
  `__post_init__`. A caller that needs to write takes a copy first.
- **A recording has at least one sample.** `n_samples == 0` is refused at `TraceDocument`
  construction — the one door the four readers and every segment pass through — and again in
  `io.base.probe_path`, which builds no document. HDF5 and TDMS both used to accept one with
  `issues=()`, and selecting it raised out of the calibration panel halfway through
  `_on_selection`, leaving the analysis target on the previous recording while the title bar named
  the empty one.
- **A selection retargets the whole window, or the user is told which panel did not follow.**
  `MainWindow._on_selection` runs its seventeen consumers through one runner; an exception in any of
  them is recorded, printed, and does not abandon the rest. PySide6 already swallows an exception
  that escapes a slot, so the real choice was never "raise or swallow" — it was "half a retarget
  and silence" or "a whole retarget and a sentence".
- **A calibration gain is never zero.** Refused at `ChannelCalibration.__post_init__`, the one
  door a sidecar, a project, the Gain box and any future loader must all pass through — after the
  float32 cast `apply()` will make, so a gain that underflows to zero is caught too. It is a
  different rule from the non-finite one: `gain=0` makes every sample equal to the offset, a
  perfectly *finite* constant that no read-out downstream can tell from a recording (it printed
  "Total RMS noise: 0 pA" with `meta.issues = ()` on the owner's real ABF). A NEGATIVE gain stays
  legal — headstages invert, and a sign flip loses nothing.
- **What the calibration did to a channel travels with the channel.** `meta.issues` describes the
  recording *as loaded*; every analysis reads the *calibrated* array, and nothing re-checked it, so
  a gain that overflowed float32 left 2,420 non-finite samples with `meta.issues = ()` while the
  PSD panel told the user to go read the Issues box. `Session.calibrated` now fills
  `CalibratedChannel.issues` (memoised per `(doc, ch, gain, offset)`, the same key as the cached
  array), and the File Info Issues box shows both halves. Empty means "the calibration added
  nothing", never "not checked".

## The project file — `core/project.py`

`core/project.py` is the **pure (de)serializer** for a `.nanoscope` file: no Qt, no `Session`. It
moves a plain `ProjectState` — documents-as-references, segments, calibrations, layers, view mode,
per-result event tables + provenance + params, manual measurements — to and from one self-contained
file, so the GUI restore code in `main_window_project.py` (finished by `MainWindow._finalize_restore`)
stays a thin translator between `Session` and a dataclass. That split is what makes the format testable headless, and `tests/core/test_project.py`,
`test_project_segments.py` and `tests/gui/test_adversarial_project_io.py` all lean on it.

**The file is a ZIP** holding `manifest.json` (everything except the bulk tables),
`results/<i>.parquet` (one per `AnalysisResult`, the FULL curated frame — `accepted`, edited
`start_idx`/`end_idx`, recomputed columns, non-schema columns like `added` — stored losslessly with
pandas dtypes via pyarrow) and `measurements.parquet` when the hand-measurement table is non-empty.

**The 180 MB of raw signal is never stored.** Documents are persisted as REFERENCES — absolute path,
the loaded `SampleRange`, and the `identity.SourceFingerprint` — and reopened from disk on load. What
a user cannot cheaply recreate (calibration, layer view state, curated detections, manual
measurements) is what we keep. A **segment** is likewise a derived recording, stored as
`(segment_uid, parent_key, i0, i1, name, t0_abs_s)` and re-derived from its parent on load rather
than saved as signal.

Three decisions in it are load-bearing:

- **Records are keyed by the PERSISTENT key, never by `doc_id`.** The `doc_id` field inside
  `calibrations` / `layers` / `results` holds `core.segment.persistent_key(doc)` — a file's resolved
  source path, or a segment's `segment_uid`. A live `doc_id` now carries a per-load token (see
  *Identity*, above), so persisting it would make **every** project round-trip fail to re-attach: the
  reopened load's token is by construction a different one. The values `persistent_key` returns are
  exactly what they were before that token existed, so projects written by either build read
  identically.
- **The version is a RANGE, and every extension is additive.** This build writes `VERSION = 13` and
  reads back to `MIN_READ_VERSION = 1`; older formats are strict subsets. v2 added `segments`, v3
  each result's `params` and `calibration_fingerprint`, v4 each document's `source` fingerprint, v5
  each calibration's `note` — the one field a project could not carry, which is why committing a
  calibration after a restore used to blank the user's sidecar. v6 added the `fits` list: the
  DERIVED results (`fit.dwell` / `fit.depth` / `fit.levels`), which until then were dropped on save
  without a word. v7 added `channel_name` / `channel_kind` to all four per-channel records
  (`CalibrationRec`, `LayerRec`, `ResultRec`, `FitRec`), because an index is a *position* and not an
  *identity*: re-export a recording with its signal columns in the other order — same sample count,
  same channel count, byte-identical size — and a `pA` calibration lands on the VOLTAGE channel,
  a detection is reattached to the voltage trace, and the result's stored fingerprint MATCHES the
  misplaced calibration, so the staleness system positively certifies those events as current-valid.
  v8 added `CalibrationRec.kind`, the user's *"treat this channel as"* override — NOT `channel_kind`,
  which records what the file header declared so a channel can be found again; this one is the
  correction made when the header is wrong or silent, and it drives both the pane order and the
  overlay grouping. v9 added `colours`, the project's identity-keyed colour map
  (`core.layer.ColourRegistry`): which slot each recording+channel holds in the categorical cycle,
  and the ink the user assigned it by hand. Before it, a recording's colour was persisted for
  exactly ONE of the four surfaces that draw it — `layers[].color` carried the trace view's ink,
  while the Plots panel recoloured from *position* on every refresh — so a hand-picked colour did
  not survive a reopen and "the Zn run is the green one" was true in one window and false in the
  next. Keyed by the persistent key like `calibrations` / `layers` / `results`, so it re-attaches to
  a segment re-derived on load. v10 added each result's `module_version` and top-level
  `baseline_fingerprint`. This preserves the exact producer version for derived/plugin results and
  the baseline lineage needed by Reconciled snapshots instead of silently substituting the current
  registry version or dropping provenance on round-trip. v11 added the versioned `workspace`
  envelope: ordered, stable-id analytical artifacts with small JSON payloads, Parquet-backed tables,
  and opaque byte members. It carries population-plot recipes and computed PCA/clustering/report/QC
  snapshots without serializing Qt or executable Python objects; unknown producer kinds and their
  extension fields/resources are deliberately retained for a later build or plugin. v12 added state
  INSIDE three artifacts the v11 build already owns: the Noise spectrum's spectrogram colour limits and
  log-frequency axis (`builtin.psd`), the Event waveforms ensemble's subset, band and frozen
  band/alignment/counts (`builtin.event-plots`), and the PlotConfig version 3 sections, KDE overlay
  and Event-count heatmap (`builtin.plots`). v11's opaque carry-through keeps unknown artifacts and
  fields, but a v11 build restores its OWN artifacts from the keys it knows and re-captures them from
  its widgets on save, so it would show a heatmap as a histogram and strip the rest. v13 replaced the
  eight singleton analytical pages with independent, named live `FigureSpec` artifacts. An older build
  cannot interpret their ownership, pending/applied recipes or frozen resources, so every new save is
  v13 even with one figure; v1–v12 files still open here, and converting one in memory does not
  rewrite the source file. Each missing key
  has a defined reading:
  absent `params` is `{}`, an absent
  calibration fingerprint falls back to the restored calibration, an absent `source` reads as
  *unknown* (which raises no warning rather than a false one), an absent `note` means "this file has
  no opinion", leaving the sidecar's note alone, an absent `fits` is `[]`, an absent
  `channel_name` / `channel_kind` is `None`, which `calibration.resolve_channel` treats as "resolve
  by index, exactly as before, silently", and an absent `kind` means "no override, the header
  stands", and an absent `colours` is an EMPTY registry, meaning "nobody has assigned anything
  yet" — the first discovery then hands out slots in the order it always did, which is visually what
  that project last showed, because a project without a registry was drawing positional colours
  anyway. An absent `module_version` falls back to the currently registered module version for
  legacy projects, and an absent top-level baseline fingerprint falls back to provenance when
  available. An absent `workspace` in a v1–v10 file is an empty analytical workspace; a v11 or
  later file must carry a valid workspace envelope. The v12 keys absent from a v1–v11 artifact read
  as Auto colour limits, a linear frequency axis, every accepted event with no band, and a
  PlotConfig without KDE or heatmap sections. A version above
  `VERSION` is refused with *"needs a newer NanoScope"* rather than half-read.
- **Additive does not mean an older build may write the file.** Additive-ness governs whether THIS
  build can read OLDER projects; it says nothing about whether older builds can safely read this one,
  and they cannot: a build that predates `fits` accepts the file, ignores the key and the
  `fits/<i>.parquet` members, and DROPS THEM ON ITS NEXT SAVE — silently, because it never knew they
  were there. That is why every one of v6 / v7 / v8 / v9 / v10 / v11 / v12 bumps `VERSION` even though each is
  purely additive: the bump converts a silent strip into `ProjectVersionError`'s clean refusal, and costs
  nothing in the other direction, since `MIN_READ_VERSION` is still 1. v13 bumps it for the same
  protective reason, though it replaces the singleton pages with named figures rather than adding a key.
- **`load_project` promises exactly ONE exception type.** It kept that promise for the failures
  somebody had thought of — missing manifest, bad JSON, absent parquet member — and broke it for
  every other malformation of the same file: a `layers` entry without `color` raised a bare
  `KeyError('color')`, a `loaded_range` written as a string raised `TypeError`, a non-numeric
  `channel` raised `ValueError`. Six shapes, each escaping as whatever the decoding expression
  happened to throw. `_manifest_section` wraps each section — not each field, because the failure is
  not a field: it is *"this manifest is not shaped like a manifest"*, and the useful thing to say is
  which part of it and what Python found there. The GUI's blanket `except Exception` is precisely why
  it stayed wrong for so long; the contract is for the API's other callers and for the next reader.

Two smaller rules keep the round-trip honest. Provenance dicts are coerced JSON-safe (`_json_safe`:
numpy scalars and arrays down to builtins, an unrecognised leaf to `str`, so a stray object can never
abort a save). And the per-result **overlay hints are not stored** — their sample-index arrays are
rebuilt from the event frame on restore, so a numpy array never reaches the JSON in the first place.

**Some additions need no version bump**, because they ride in a container the format already writes
whole and an older v11 build carries through untouched:

- **`FitResult.meta`**, which `stats.fit_to_json` writes wholesale. `fit_dwell` / `fit_depth` record
  there the table they were made on (`detection_module_id`), the column they fitted
  (`input_column`) and `input_digest` — `sha256:` over the `start_idx` and the fitted column of the
  rows the fitter was handed, in stable start order and not by `event_id`, because a merge
  renumbers the ids and re-sorts the rows — plus the table's own rate and estimator stamps, spelled
  as the table spells them (`fs`, `baseline_fingerprint`). `modules.judge_stored_fit` and
  `stored_fit_problem` compare the digest when a fit has one, and a fit without it keeps the count
  rule. The Plots window's fits record no digest: it may fit a narrowed range or another column of
  the same events, which a digest of the table's rows would not describe.
- **The v11 `workspace.extra` map**, whose unknown keys a v11 build preserves byte for byte. Each
  addition there is one versioned key with one contract — records keyed by the PERSISTENT key and
  matched with `identity.same_source`, unknown or malformed records kept verbatim and never
  applied, an unreadable section quarantined rather than overwritten: `sample_rate_declarations_v1`,
  `acquisition_bandwidth_overrides_v1`, `event_table_choices_v1`, and `analysis_form_drafts_v1`,
  the un-run Detection, ML and Baseline form each event table was left with — one record per
  `(doc_id, channel, module_id or None)`, written by `core/form_draft_state.py` and restored before
  the form on screen arrives. A draft differs from a table choice in one rule: a choice outlives the
  session's interest in it, so only another live choice replaces a saved one, while a draft is
  dropped when its own run settles it, and that absence is information. The live session therefore
  replaces every saved record about a channel it holds, present or not, and keeps the records of a
  recording that is not open.

**The save is atomic**, via `atomic_write.staged_write` (see the next section; `_staged_write` here is
a one-line pass-through kept for this module's callers). This is not defensive decoration:
`zipfile.ZipFile(path, "w")` truncates the target the instant it opens, so the pre-atomic version
destroyed a good project *before* it knew whether it could write a new one — an injected ENOSPC on
the second member turned a 6949-byte project into a 3251-byte corpse that then failed to load with
"missing its manifest.json". A save that fails must cost the user nothing. `save_project` also takes
an optional `notes` list, which collects sentences about what the write did to the **file** — that a
symlink was followed to its destination, or that other hard-linked names now hold the previous
contents.

## The write seam — `core/atomic_write`

PORT_PLAN rule 21: **no write may destroy what it is replacing until the replacement exists.** The
rule was stated for "every file the app writes over user data" and implemented in
`core/project._staged_write` alone; `calibration.save_sidecar`, `scoring`'s two exports,
`measure.write_measurements_csv` and `plot.figures.save_figure` all wrote in place. It now lives in
one module and an AST guard (`tests/core/test_persist_invariant.py`) fails the build if any module
under `nanoscope/` writes a path it did not stage.

Three guarantees, three different failures:

1. **A write never half-lands.** Staged in a temporary sibling, moved with one `os.replace`.
2. **A write never changes WHICH FILE it wrote.** `os.replace` installs a new inode, so a naive
   stage-and-replace turned a **symlink** into a regular file (the canonical project silently
   keeping the previous save), destroyed **extended attributes** on every save, could not stage a
   name of 239 characters at all, and overwrote a `chmod 400` project without a word.
   `resolve_write_target`, `dress_like`, `temp_sibling` and `refuse_if_write_protected` close
   those. A **hard link** is the one thing an atomic replace cannot preserve — writing in place is
   the only alternative and it is the destructive path guarantee 1 forbids — so the trade stands
   and is *reported* through `notes`.
3. **A SET of files never half-lands.** One figure export writes twelve files under one `figures_*`
   stem; nine failing left three from today beside nine survivors of a previous export with no
   marker. `StagedSet` stages every member first, commits together, and restores what it replaced
   if a commit fails part way.

The same rule at the value level is why `CalibrationRec` carries every field
`ChannelCalibration` has (checked by a test): a field the record lacks is not "not persisted", it
is *blanked* by the next ordinary Apply. `note` is three-state — a string, `""`, or absent, where
absent means "this project file has no opinion" and leaves the sidecar's note alone.

And the same rule about *what to ask before*: `sidecar_effect` answers a question about the FILE,
so the confirmation built on it only fires when the whole file goes. Reset one channel of a
recording whose other channel is still calibrated and the file survives (`SIDECAR_WRITTEN`) while
the note goes out of it — measured: no prompt, status line *"Applied · saved to …"*.
`calibration.notes_lost_by` (→ `Session.sidecar_notes_lost_by`) is the companion question — *what
does this write destroy* — derived from the same `_sidecar_payload` the write is derived from, so
the prediction cannot drift from the act. The panel asks one of the two, never both.

## The save-dialog seam — `gui/export_paths`

`core/atomic_write` guarantees a write does not half-land. This module answers the question that
comes *before* it: **which files is the user actually about to lose?**

`QFileDialog.getSaveFileName` confirms an overwrite of **the one name the user typed**, and that
confirmation is only truthful for an export that writes exactly that file. Several of ours do not.
*Analysis ▸ Export figure…* proposed `figures.pdf` — a file it **never writes** — and then wrote
`{stem}_{dwell,depth,scatter,trace}.{pdf,png,csv}`: up to **twelve** files, none of them mentioned,
silently overwriting whatever was there, with the `.csv` source-data files the ones a user is most
likely to have hand-edited. *Export events for training…* proposed `events_training.csv` and also
wrote `events_training.meta.json` beside it, unmentioned.

Patching those two commands would have left the next export free to reintroduce the defect, so the
guarantee moved to where every export must pass: **the path a GUI export writes to is obtained from
`ask_export_path`**, which is handed an `outputs` callable that enumerates the REAL output set
(`figures.figure_output_paths`, `scoring.training_export_paths`, …) and confirms whatever part of
that set already exists. `outputs=None` is the honest declaration "this export writes exactly the
chosen path and nothing else". Every GUI export that asks for a file name gets it from
`ask_export_path`; an AST guard in `tests/gui/test_export_overwrite.py` fails the build if any
module in `nanoscope/gui` calls `getSaveFileName` outside this file — that guard, not the
docstring, is what keeps the seam. An export into a chosen folder has no file name to ask for: the
batch figure export picks its folder with this module's `ask_export_directory` and confirms its own
file list with `existing_of` and `confirm_overwrite`; Manual Study's prepared-recording and
event-table exports pick theirs with a plain folder dialog and refuse to replace a file that is
already there.

Three details are deliberate:

- **The native overwrite prompt is suppressed** (`DontConfirmOverwrite`). Leaving it on would ask
  about one file and then this module would ask about the real set: two prompts, the first of them
  misleading. One prompt, naming what is actually at stake.
- **`confirm_overwrite` names every file**, because "3 files will be replaced" is not something a
  user can check. It is a module-level function rather than a method so a test can substitute it
  without a live modal.
- **Buttons are named, never indexed.** `QMessageBox.buttons()` returns *layout* order, not insertion
  order, and on this platform layout order put the destructive button last — so `buttons()[-1]` made
  **Replace** the Enter-key default on a prompt whose entire job is to stop an accidental overwrite,
  and left no escape button at all. Return and Esc must both keep the user's files.

`ask_export_path` returns the chosen path **verbatim**; normalising it is the caller's job and must
match what its own `outputs` enumerator assumed, or the set that was confirmed is not the set that
gets written.

## Environment

The Anaconda distribution ships a `qt.conf` that points Qt6 at conda's Qt5 plugins, breaking Qt
plugin discovery. `nanoscope/bootstrap.py` sets the plugin path to PySide6's own plugins before any
Qt import, so the app launches without manual environment variables. It is imported first by
`nanoscope/__init__.py` and by the test bootstrap.

## Testing

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Runs headless. GUI tests use `pytest-qt`. A `@pytest.mark.corpus` subset drives real recordings and
skips when they are absent. Detection is gated in tests to recover synthetic ground truth at
recall/precision > 0.95.

Two hazards are the environment's, not the code's, and both are handled once in `tests/conftest.py`
rather than per test. Both were live defects in this suite.

**1. A modal reached under `QT_QPA_PLATFORM=offscreen` hangs the whole run, silently.** There is no
event source that can dismiss a `QDialog.exec()`, so the process sits in `poll()` at 0 % CPU with no
output — indistinguishable, from the outside, from a slow test. Five modals are defaulted for every
test by the autouse `_no_close_guard_modal` fixture. Three are **confirmations the user's own click
asks for**: `MainWindow._confirm_close_choice` → *Discard* (pytest-qt closes every registered widget
at teardown, which fires `closeEvent`), `_confirm_close_discarding_events` → True, and
`MeasurePanel._confirm_remove` → True. Four are those confirmations plus one of a different and more
dangerous shape — a **modal a RESULT opens**: `AnalysisPanel._show_sweep`, where a finished
`sweep_threshold` raises `SweepDialog` on a queued signal. A test never has to click anything to meet
that one; it only has to let the job finish, so whether it hangs is a race against machine speed —
won on a loaded machine, lost on an idle one.
`test_run_gate.py::test_the_surrogate_chain_runs_off_the_gui_thread` starts a real sweep to prove
conditioning runs off the GUI thread, waits only until conditioning *begins*, then pumps 20 more
times before cancelling; it hung deterministically in isolation until this fixture covered it.
`_show_comparison` (`EstimatorCompareDialog`) is deliberately **not** in the fixture: it is reached
only after a test has itself called `compare_baseline_estimators()`, and the migration test patches
`EstimatorCompareDialog.exec` to return *Accepted* so it can assert the re-measure that branch
launches — neutralising the method would delete that branch and the test with it. A per-test patch of
the *dialog* is right when the dialog's answer is what the test is about; the fixture is for modals
nobody meant to open. **Any new result-opened modal must get the fixture's treatment**, and note its
one limitation: it patches only what is already in `sys.modules` when it runs, so a test that is the
first to import `nanoscope.gui.main_window` is not covered.

**2. The suite must not read the developer's own saved window layout.** `MainWindow._save_settings`
writes `geometry` and `windowState` into `QSettings` on every close, so any real launch of the app, a
screenshot harness, or a `--self-test` run leaves a layout behind that the next `MainWindow()` in a
test restores instead of its defaults. The session `qapp` fixture therefore points the whole run at a
private namespace (`NanoScopeTest` / `NanoScopeSuite`) and clears it. Measured: three geometry- and
focus-sensitive tests failed on this machine and passed the instant `geometry` and `windowState` were
deleted from the real store — `test_analysis_launcher.py::test_a_real_window_drives_the_ladder_and_never_clips_below_the_minimum`,
`test_measure_tab.py::test_dragging_the_inspector_narrow_still_sticks_with_the_measure_tab_raised`
and `test_p3_regressions.py::test_the_documented_curation_keys_are_reachable_on_the_table`.
`test_window_layout.py` and `test_shown_recording.py` already carried their own namespaces for this
reason; the rule is now the suite's, not two modules'.

To diagnose a hang: `sample <pid>` and look for `Sbk_QDialogFunc_exec` / `QDialog::exec()` in the
main-thread call graph. Re-running with `-v` and reading the last line of the log names the test.
