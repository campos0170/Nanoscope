# Editing foundation — design (draft 1)

Status: approved by the owner on 2026-10-30, with every decision in section 12 as proposed.
Built in the working tree (uncommitted): section 13 lists what was built, where it differs from
this design, what was not built and what is the owner's to decide.
Written against `27ec0eb`. Code references are `file:line` at that commit, under `nanoscope/`
unless they start with `tests/` or `docs/`.

## 1. Goal and the problem today

Every edit in NanoScope should be one press away from being taken back. A switch, a reset or
a re-seed should never throw away settings the user typed, and a reset should reset one
thing. Every result should say when something it was made from has changed.

On 2026-10-30 the owner was asked which incident prompted this and named none. So this
design starts from what the code does today and gives the five pieces equal weight. Mapping
the code found three kinds of trouble:

- defects in the undo that already exists;
- edits that no undo reaches;
- dependencies that nothing checks.

### 1.1 What exists

| Piece | Today |
|---|---|
| Global undo/redo | **One Edit ▸ Undo over two snapshot stacks.** `EventHistory` holds the event tables: 30 steps per table and 128 MiB for the whole session (`core/history.py:135, 164`). `MeasurementHistory` holds the old measurement list (`core/history.py:1279`). A journal records which stack moved last, and that picks the stack a press acts on (`gui/main_window.py:775-776, 13535-13679`). A step on a table that is not on screen is skipped. History is never saved, Open Project clears it, and closing a recording takes that recording's history with it. |
| Plot history | **The Plots window has its own stack.** It holds 200 whole `PlotConfig` values, with its own buttons and keys, and is kept apart from Edit ▸ Undo on purpose (`gui/plots_panel.py:2211, 2237-2300, 5922-5944`). Closing the window blanks it (owner note #16, `plots_panel.py:11900-11930, 12021`). The other eight plot windows have no undo at all. |
| Non-destructive processing | **Samples are never rewritten.** `TraceDocument` is frozen, and processed traces are derived and cached by content. **Settings are another matter:** no undo covers any processing setting, and several actions overwrite them (1.2). |
| Resets | **Every reset is coarse.** Revert to detection's chain rewrites all 37 chain keys. Calibration ▸ Reset to default resets gain, offset, units, note and kind together. Baseline ▸ Back to default resets only the estimator windows, and is hidden outside Automatic mode. A chain stage, the ML form and the open-pore reference have no reset at all. |
| Dependency tracking | **Stamps are flagged, never acted on.** Detections carry calibration, baseline and sample-rate stamps, which the amber line judges (`gui/analysis_panel.py:7353-7472`). The chain judge compares the form with the run on chain keys only (`analysis/conditioning_recipes.py:567`). `modules.judge_stored_fit` checks a fit's table name, calibration and accepted count (`analysis/modules.py:2916`). Reconciled tables carry digests of their sources. The Plots window checks calibration only. Nothing recomputes a stored result: everything is flagged, and the user repairs it. |

### 1.2 Defects found while mapping

Phase 1 fixes each one, with a regression test that fails on today's code.

1. **⇧⌘Z does nothing in the main window on a Mac.**
   - Edit ▸ Redo registers both `QKeySequence.Redo` and `"Ctrl+Shift+Z"` (`gui/main_window.py:1325-1326`).
   - On the cocoa platform, Redo resolves to Ctrl+Shift+Z (measured), so the same key is registered twice. Qt reports "Ambiguous shortcut overload" and triggers nothing. A probe registered a key twice: 0 of 1 presses fired. Registered once, 1 of 1 fired.
   - The Plots window dedupes its keys for exactly this reason (`gui/plots_panel.py:5925-5944`).
   - Tests run offscreen, where Redo is Ctrl+Y, so none of them sees the problem. Only the menu item works.
2. **After a Table-selector or detector switch, Undo still shows the old table's state.** `_on_result_switched` (`gui/main_window.py:8220-8260`) never calls `_sync_undo_actions`. If the newly shown table has history, Undo stays disabled until the Edit menu is opened, and ⌘Z on a disabled action does nothing.
3. **⌘Z does nothing in the Signal Conditioning window.** The window forwards ⌘S and ⇧⌘S to the main window but not Undo or Redo (`gui/signal_conditioning.py:638-650`). The chain is edited in that window.
4. **Switching recordings throws away edits that have not been run.**
   - Leaving a target clamps the frequency boxes to the new recording's Nyquist (`gui/panels.py:3921-3952`).
   - It then re-seeds the whole form from the new target's stored run (`gui/analysis_panel.py:1744, 1783-1800, 2001-2052`).
   - So an edit to the low-pass on A is gone after a look at B and a return to A. If A has no stored run, the form keeps B's values.
   - A Table-selector switch runs the same re-seed.
5. **Reset to default puts the wrong gain on a voltage channel.**
   - Opening a recording seeds each channel with `default_for(info, median)` (`core/session.py:159-174`). A host-guest voltage channel that reads a plausible bias therefore opens at gain 1.0.
   - Reset installs `default_for(info)` without the median (`gui/calibration_panel.py:969-974`), which gives gain 0.1. That is a 10x error in every voltage, conductance and V-dependence figure.
   - The sidecar payload compares against the same median-less default (`core/calibration.py:355-366`). So any sidecar write for such a recording also records the voltage channel, which the user never touched.
6. **Reset to default also clears the channel's note and its "Treat this channel as" override** (`gui/calibration_panel.py:969-974`).
7. **Revert to detection's chain is all-or-nothing and cannot be taken back.**
   - It rewrites all 37 chain keys and the polarity (`analysis/conditioning_recipes.py:567-601`, `gui/analysis_panel.py:8124-8134`).
   - Any key the run never recorded is filled with its default.
   - A protocol recall behaves the same way (`gui/panels.py:5203-5242`).
8. **Re-measure ignores a calibration change.**
   - It refuses only a stale sample rate (`gui/analysis_panel.py:6570-6617`).
   - It measures with the live calibration but keeps the table's old calibration stamp (`6848-6857`).
   - The table ends up holding depths measured under the new calibration with the old calibration's stamp. The amber line keeps saying "Depths stale — calibration changed".
9. **The guide quotes ceiling wording the app does not produce.** Compare `docs/USER_GUIDE.md:1678-1681` with `gui/main_window.py:13925-13954, 14008-14027`.

### 1.3 Edits nothing can take back

- **Detection form:**
  - every chain and threshold edit;
  - stage toggles in Signal Conditioning, and notch lines sent from the spectrum;
  - protocol recall, and Revert to detection's chain.
- **ML form and presets.**
- **Baseline:** Apply, Back to default, window edits, and open-pore reference rows.
- **Calibration:** Apply and Reset, including the sidecar deletion (`gui/calibration_panel.py:802-845`).
- **Time base:** the sample-rate declaration and the acquisition-bandwidth override.
- **Plot appearance** in Event plots, Noise spectrum, I-V, V-dependence, Distributions, Comparison figures, Subpopulations and the study figures. All of them use the shared `gui/plot_appearance.py:70` widget.

Some edits are deliberately left out of this design (3.8): fits; layer colour, offset and visibility; view mode; the saved protocol and preset libraries; and the study windows.

### 1.4 Dependencies nothing checks

| Change | Affects | Today |
|---|---|---|
| Detection thresholds (k_in, k_out, CUSUM k/h, min/max dwell, merge gap) | the table on screen | Not checked. The chain judge compares chain keys only, and `_on_params_changed` re-judges nothing (`gui/analysis_panel.py:7135-7165`). |
| An accepted event swapped, resized or re-measured, with the accepted count unchanged | dwell and depth fits | The fits still read "current", because the judge compares only the accepted count (`analysis/modules.py:2884-2899`). |
| Baseline estimator or sample rate | dwell and depth fits | Not checked. Fits carry no baseline or rate stamp (`analysis/modules.py:2958-2979`). |
| Calibration, when the fit and the table both predate the change | fits | The fit shows as current. Only the table's amber line flags the change. |
| Baseline estimator or sample rate | Plots window | Not checked; it looks at calibration only. Its own docstring names the gap (`gui/plots_panel.py:10398-10441`). |
| The table being stale in any way | Export figure | Not checked. Only the fit guard runs (`gui/main_window.py:10685-10713`). |
| Calibration | Re-measure | Not checked (defect 8). |

## 2. Summary

1. **Global undo.** Edit ▸ Undo also takes back the analysis settings: the Detection, ML and Baseline forms, calibration, the sample-rate declaration and the acquisition bandwidth. The same press works from the Signal Conditioning window, the Inspector, the gallery and the inspector window, and ⇧⌘Z works on a Mac.
2. **Plot history.** Every plot window gets its own Undo and Redo for its appearance, kept apart from Edit ▸ Undo. The Plots window keeps the whole-recipe stack it already has.
3. **Non-destructive processing.** A recording's samples become read-only in memory. The forms keep un-run edits for each event table, so a switch of recording or table never discards them. When a value is clamped to a slower recording's Nyquist, the app says so and puts the value back on the way out.
4. **Individual resets.** Each chain stage can be reverted to the run or reset to its defaults. Calibration resets to the default the channel opened with and keeps its note and kind. The baseline estimator and the reference reset separately. Every reset is one named undo step.
5. **Dependency tracking.** Each table has one record of what it was made from. Every input in it (rate, calibration, baseline, chain, thresholds) is judged on its own and listed with its own action. Fits record which events they fitted. The Plots window, Export figure and Re-measure all ask the same judge.

## 3. Global undo and redo

### 3.1 What "global" means here

There is one Undo, reachable from every window. It covers every edit that changes what the next number will be.

It is not one flat list across recordings. Today's rule stays: **Ctrl+Z never silently changes a recording that is not on screen** (`docs/USER_GUIDE.md:1654-1656`). A press takes back the most recent step that belongs to the recording on screen, and the menu names that step. Decision 1.

### 3.2 What it covers

**Form** (one scope per event table; see 3.4)
- **Snapshot:** `DetectionPanel.params()`, `ml_params()`, and the Baseline form's full state (a new `form_state()`; see 3.9).
- **Written back through:** a new `AnalysisPanel.restore_form_state`.
- **Recorded:**
  - 700 ms after the last edit, the same pause the Plots window uses (`_HISTORY_COALESCE_MS`, `plots_panel.py:2234`);
  - at once before a run and before a switch;
  - at the four moments that already checkpoint: Ctrl+Z, opening the Edit menu, save and quit.
- **Labels:**
  - Built from the diff (`chain_differences` and `describe_differences`, `conditioning_recipes.py:502-530`). Examples: "Change low-pass 10000 → 5000 Hz", "Turn on notch", "Change k_in 5 → 6", "Change 4 settings".
  - The action that knows its own name labels its step: "Recall protocol ‘HG student’", "Revert chain to Schmitt's run", "Reset Notch stage".

**Calibration** (one scope per recording and channel)
- **Snapshot:** `Session.calibration(doc, ch)`, meaning gain, offset, units, note and kind.
- **Written back through:** `Session.set_calibration(..., write_sidecar=True)`.
- **Recorded:** on `calibrationChanged`.
- **Labels:** "Apply calibration", "Reset calibration", "Treat channel as voltage".

**Time base** (one scope per recording)
- **Snapshot:** `Session.sample_rate_declaration(doc)`.
- **Written back through:** `Session.set_sample_rate_declaration`, where None means the file's own time base.
- **Recorded:** on `sampleRateChanged`.
- **Labels:** "Declare 50 kHz", "Use the file's own time base".

**Acquisition bandwidth** (one scope per recording and channel)
- **Snapshot:** `Session.acquisition_bandwidth_override`.
- **Written back through:** `Session.set_acquisition_bandwidth`.
- **Recorded:** on `acquisitionBandwidthChanged`.
- **Label:** "Change acquisition bandwidth".

Undoing a setting never re-measures anything; the stamps take care of the rest. For example, undoing a calibration change makes the amber line it raised go away, because the table's stamp matches again.

### 3.3 `SettingsHistory`

`SettingsHistory` is a third sibling class in `core/history.py`. It is a sibling for the same reasons `MeasurementHistory` gives (`core/history.py:1279-1307`): the rule is reused, the payload is not. Each scope registers a `read()` and a `write(state)` callable, both supplied by MainWindow, so the class holds no widget and no schema.

- **Recording.** It keeps a pristine copy per scope and compares against it on `checkpoint(scope)`.
  - A change pushes `(before, after, label)`.
  - No change pushes nothing, so wiring a signal twice cannot invent a step. This is the `EventHistory` rule.
- **Undo and redo.** `undo(scope)` writes `before` with recording muted, then makes it the pristine copy. `redo` does the same with `after`.
- **Rebase.** `rebase(scope)` adopts what is on screen without recording a step. It is used when a form arrives through a switch, a project restore or a seed.
- **Depth.** 30 steps per scope (`MAX_DEPTH`). Hitting the ceiling is counted and reported, as the event stack does.
  - A snapshot is a dictionary of about 80 scalars, so no byte budget is needed (compare `plots_panel.py:2194-2211`).
- **Signals and lifetime.**
  - It emits `stepRecorded` and `changed`, as its siblings do.
  - Closing a recording calls `drop_doc`, and Open Project calls `clear`.
  - Recording is suspended during a project restore.

### 3.4 Scopes, and one form per event table

The forms are one set of widgets that show whichever target is selected (`gui/panels.py:4848, 4974`). Their history therefore needs a key.

- **Form key.** The key is the **event table on screen**: the same `(doc_id, channel, module_id)` that Edit ▸ Undo already walks. Before a recording and channel has any table, the key is `(doc_id, channel, None)`.
  - Each key keeps its own form state (5.2). Undo therefore walks back the table on screen and the settings for that table.
  - Switching tables and switching recordings both move to another key.
  - The steps made before any table existed move to the first table that recording and channel gets, since they produced it.
- **Other keys.** Calibration and the acquisition override are keyed `(doc_id, channel)`, and the time base `(doc_id,)`. For these, "on screen" means the recording on screen.

### 3.5 One press, one journal

The journal in MainWindow (`gui/main_window.py:775-776`) already chooses between two stacks. It gains settings entries, each naming its scope.

- `_undo_target` and `_redo_target` (`13647-13679`) scan newest first for an entry that can act on screen. Their fallback also asks the settings scopes that are on screen.
- `_sync_undo_actions` (`13963-14031`) names settings steps like any other, as in "Undo Change low-pass 10000 → 5000 Hz". When only other recordings have steps, the menu says "Undo — nothing on this recording".
- Redo stays separate for each scope (decision 2):
  - a new settings step ends only its own scope's redo;
  - a settings step does not end an event table's redo;
  - an event step does not end a settings redo.

  The cross-stack rule between events and measurements (`13535-13575`) is unchanged.

### 3.6 Undoing a run also undoes the settings that led to it

**Today**, one Ctrl+Z after "change settings, run again" restores both the table and the form. Restoring the old table re-seeds the form, and `tests/gui/test_beta_calib_filters_preview_undo.py:206-280` pins this.

**The problem** once settings are steps: the re-seed alone would leave the settings steps behind it pointing at a form that has already changed.

**The rule:** undoing a step that installed a new run on a table, such as "Re-run detection" or "Re-measure depths", does more than restore the table.

- In the same press it also undoes the settings steps of that table that sit right before it in the journal. Redo replays them in order.
- The menu says so: "Undo Re-run detection and 2 setting changes".
- A settings change separated from the run by other work, such as a curation step, stays a step of its own.

Decision 3.

### 3.7 Reaching it from every window

- The Signal Conditioning window forwards ⌘Z and ⇧⌘Z to the main window, as it already does for ⌘S (defect 3).
- The Event gallery gets a Redo button beside its Undo.
- The Inspector, the docked inspector and the gallery already send their key presses to Edit ▸ Undo (`gui/main_window.py:6767-6768, 7048-7058, 7716`).
- A focused text field keeps ⌘Z for its own typing, as today (`gui/main_window.py:1316-1320`), and
  only while it has typing to undo: a spin box stepped with its arrows or an editable combo picked
  from its list lets the key through (`gui/app.py` `_UndoKeyGuard`, sweep 10 S10-PLOT1). The edit becomes an undo step after the 700 ms pause, or when focus leaves the field.
- Redo's key list is deduped on the resolved key sequence, through one helper shared with the Plots window (defect 1).
- `_on_result_switched` refreshes Undo and Redo (defect 2).

### 3.8 What stays out

- **View state:** zoom (which has its own Back/Forward), layer visibility, colour and time offset, view mode, window layout and column order.
- **Library edits stored in QSettings:** saving, renaming and deleting protocols and presets. These are preferences, not the project. Instead of undo, deleting a protocol or a Plots template will ask first; today neither asks (`gui/panels.py:5346-5353`, `gui/plots_panel.py:6742-6752`).
- **Fits.** A Fit press replaces the pair, and a figure refits when it needs to (7.2). Undo of a fit is not proposed.
- **One-way actions that already ask first:** closing a recording, Open Project, and creating a segment.
- **Study and incubation windows,** which keep their own audit records.
- **Saving undo history with the project.** History stays session-only, as today.

### 3.9 Cost and risk

- **Code size:**
  - `SettingsHistory`: about 250 lines.
  - Wiring in MainWindow: about 300 lines.
  - `restore_form_state` plus the Baseline form's `form_state` and `restore_form_state`: about 150 lines.
- **The Baseline form needs a real "set".** Its `set_params` is a seed that deliberately refuses to overwrite a confirmed choice (`gui/baseline_panel.py:1009-1050`), so undo cannot go through it.
- **Risk: a seed could race a pending checkpoint and be recorded as a step.**
  - Guarded by order: checkpoint the outgoing key, switch, then rebase the incoming key.
  - Guarded by muting recording during every write the app makes itself.
  - A test drives each seed path.
- **Memory:** negligible.

## 4. Plot-specific history

### 4.1 Today

The Plots window's `_ConfigHistory` (`gui/plots_panel.py:2237`) is right, and it stays as it is:

- a stack of frozen values, 200 deep;
- typing merged into one step after a 700 ms pause;
- its own buttons and window keys;
- kept apart from Edit ▸ Undo, because "Ctrl+Z in the Plots window has to undo the colour the user just picked" (`2260-2265`).

The other eight plot windows all use one appearance widget, `PlotAppearanceWidget` (`gui/plot_appearance.py:70`), and have no undo.

### 4.2 Each plot window keeps its own

- **`PlotAppearanceWidget` gains an opt-in history** of its `PlotAppearance` values:
  - Undo and Redo buttons beside Reset appearance;
  - ⌘Z and ⇧⌘Z, scoped to its own window and deduped;
  - 200 steps, with the same 700 ms merge for spin-box typing.
- **Every change is a step:** an edit, a preset, a saved appearance and Reset (`_edit`, `_apply_preset`, `apply_user_preset` and `_reset`, `plot_appearance.py:290-366`). `set_state`, used for a restore or a recipe load, adopts the new value without a step.
- **One history per plot.** Each window has its own instance. Edit ▸ Undo never reaches a plot, and a plot's ⌘Z never reaches a table.
- **The Plots window turns it off.** Its `PlotConfig` stack already includes appearance, and one window must not have two stacks.
- **A window's history ends with its data,** as the Plots window's does: when the window is reset for new data, or a project opens.

### 4.3 What stays out

- **The analysis recipes of the other windows** (Noise spectrum settings, Distributions, Comparison figures). Each has a Generate or Apply button and its own "unapplied changes" line. Giving them a recipe history is decision 9.
- **The Plots window's history across a close.** The window stays blank on reopen (owner note #16), so its history still ends when it closes.

## 5. Non-destructive processing

### 5.1 The samples cannot be written

- **Today the arrays are writeable.** With an identity calibration, `Session.calibrated_array` hands out the recording's own array (`core/session.py:740-753`). One in-place `arr -= x` anywhere would change the recording for every surface, silently, for the rest of the session.
- **`TraceDocument.__post_init__` marks each channel array read-only,** without copying it.
- **Session's cached calibrated arrays get the same flag.**
- **An in-place write then raises** at the line that tried it. Worker snapshots already work this way (`gui/analysis_worker.py:483-485`).

Decision 7.

### 5.2 Your edits stay with their table

- **Every form key keeps its last state.** Each key (3.4) remembers the form as it was when the user left it, and restores it on return.
- **A key never visited** is seeded from its stored run, as today.
- **A key the user edited keeps the edit,** and the chain notice says how it differs from the table: "⚠ Settings changed: low-pass 10000 → 5000 Hz. Revert, or run detection".
- **The rule changes.** Today the rule is "the settings on the tab are the settings that produced the table on screen" (`gui/analysis_panel.py:2001-2030`). It becomes: **they are, unless you changed them, and then both the tab and the table say so.**
- **Run still asks** before replacing hand curation (`tests/gui/test_wa_undo_rerun_prompt.py`).

Decision 4 asks whether these drafts are saved with the project.

### 5.3 A clamp is announced and undone

The outgoing form state is captured before `_apply_nyquist_ranges` clamps the frequency boxes. Returning to the faster recording therefore restores the value that was typed. A clamp on arrival is announced on the status line: "Low-pass 20 kHz is above this recording's Nyquist; the form now reads 12.5 kHz."

### 5.4 Every processing change is a step

- **Each of these becomes one named step (3.2):**
  - protocol recall, Revert, a stage toggle;
  - notch lines sent from the spectrum, an import into the form;
  - a calibration Apply or Reset, a time-base declaration.
- **Undo restores the calibration sidecar exactly** as it was before the step, including a deleted file and its note. The code today notes that a Reset that deletes the sidecar "is not undoable" (`gui/calibration_panel.py:808-811`); within a session, that stops being true.

Decision 6.

## 6. Individual resets

| Where | Today | Proposed |
|---|---|---|
| A Signal Conditioning stage | No reset. Unticking a stage keeps its values. | **Revert stage** puts back the run's settings for that stage; it is enabled only when that stage differs. **Reset stage** restores the stage's default settings (`CHAIN_DEFAULTS`) and leaves its on/off as it is. The stage list marks any stage that differs from the run. |
| Chain notice ▸ Revert | Rewrites all 37 keys and the polarity, and uses defaults where the run recorded nothing. | Reverts only the keys that differ, chain and thresholds (7.2), as one step. |
| Calibration ▸ Reset to default | Uses the median-less default and clears the note and kind. | Restores the default the channel opened with; Session keeps it per channel from the moment the recording opens. Only gain, offset and units are reset. "Treat this channel as" keeps its own reset (the combo's "as the file declares"), and the note stays. Decision 5. |
| The sidecar payload | Compares against the median-less default. | Compares against the default the channel opened with, so an untouched channel is never written. |
| Baseline ▸ Back to default | Resets the estimator windows; hidden outside Automatic mode. | Resets the estimator windows in every mode. A separate **Clear reference** button sets the mode back to Automatic. |
| ML form | Reload a preset. | Unchanged; out of scope. |

Every reset is one step, named "Reset …" or "Revert …".

## 7. Dependency tracking

### 7.1 One record, one judge

A new module, `analysis/lineage.py`, is pure and Qt-free.

- **`made_with(result)`** reads the inputs a table already records:
  - the sample rate (`provenance['fs']`);
  - the calibration and baseline fingerprints;
  - the chain (`complete_recipe` of its params) and the thresholds.

  It adds no new stamps to tables.
- **`judge_inputs(made, now) -> tuple[InputVerdict, ...]`** gives each input a verdict: "same", "changed (old → new)", or "unknown" for a legacy table that did not record it.
- **The amber line keeps its headline** and its order of precedence (`gui/analysis_panel.py:7361-7472`).
- **Its Details panel lists every input** with its verdict and its own action:
  - **Re-run** for rate, calibration, chain and thresholds;
  - **Re-measure** for baseline;
  - **Revert** for form inputs (6).

  Today a table that is stale on two counts names only the first.

### 7.2 Gaps this closes

| Gap | Proposed |
|---|---|
| Thresholds → table | Thresholds are judged with the chain: "⚠ Settings changed: k_in 5 → 6". Decision 8. |
| Events → fits | `fit.dwell` and `fit.depth` record a digest of the values they fitted (the fitted column of the accepted rows, in event order), plus the table's baseline stamp and rate. `judge_stored_fit` compares the digest when both sides have one. A swap, a resize or a re-measure that keeps the count then reads "the events changed since the fit". A fit without a digest keeps today's count rule. Decision 10. |
| Baseline and rate → Plots window | The Plots window uses the per-table verdict for every source it draws, not calibration alone. Its own docstring asks for exactly this (`plots_panel.py:10418-10422`). |
| Table staleness → Export figure | Export figure refuses a table whose sample rate is stale, as the event exports already do (`gui/main_window.py:8643`). A table that is stale on calibration or baseline is named in the status line and in the figure's provenance. |
| Calibration → Re-measure | Re-measure refuses with: "Re-run first — the calibration changed since this table was detected" (defect 8). |

### 7.3 What stays untracked

- **A recording file changed on disk while open.** This is checked at save and open (`gui/main_window.py:11373-11441`), but not watched in between.
- **Acquisition bandwidth.** It is metadata by design (`gui/main_window.py:340-343`).
- **Automatic recomputation.** None is added; the rule stays "never silently re-measure".

## 8. Persistence

- **Undo history:** not saved, as today.
- **Drafts:** only if decision 4 is yes.
  - They go in a `workspace.extra` key, `analysis_form_drafts_v1`, keyed by persistent key, channel and module id.
  - The key is additive, and needs no version bump. A v11 build that does not know it preserves it byte for byte (`workspace.extra`'s contract; see `core/table_choice_state.py`). A build older than v11 refuses a v11 file ("needs a newer NanoScope", the v6 note in `core/project.py`), so it never strips the key either.
- **Fit digests:** new keys in `FitResult.meta`, which the project writes whole (`docs/ARCHITECTURE.md:111-128`). No version bump.
- **Sidecars:** the same format. Fewer untouched channels are written (6).

## 9. Tests

All tests use synthetic `TraceDocument`s, with one offscreen pytest process per file. No test names a real recording.

### 9.1 New

- **`tests/core/test_settings_history.py`:**
  - comparison, and no phantom steps;
  - rebase, and muting during writes;
  - the depth ceiling;
  - redo per scope;
  - labels.
- **`tests/gui/test_editing_undo_reach.py`:**
  - Redo's key list holds no duplicate under the Mac resolution (the shared helper, fed the cocoa sequence);
  - Undo refreshes after a Table-selector switch;
  - ⌘Z and ⇧⌘Z work from the Signal Conditioning window;
  - the gallery has a Redo.
- **`tests/gui/test_editing_settings_undo.py`:**
  - a chain edit, a protocol recall, Revert and a stage toggle each undo and redo;
  - one press undoes "change, run", with the CF-4 tests unchanged;
  - the rule that a recording not on screen is never changed;
  - undoing a calibration Apply or Reset restores the sidecar byte for byte, including a deleted one;
  - undo of the time base and of the acquisition bandwidth.
- **`tests/gui/test_editing_drafts.py`:**
  - edits survive a recording switch and a table switch;
  - a clamp is announced on arrival and the typed value restored on return;
  - a draft round-trips through a project, if decision 4 is yes.
- **`tests/gui/test_editing_resets.py`:**
  - stage Revert and stage Reset;
  - Reset to default on a host-guest voltage channel stays at gain 1.0 and keeps the note and kind;
  - the sidecar leaves an untouched voltage channel out;
  - Baseline Back to default in reference mode, and Clear reference.
- **`tests/core/test_read_only_samples.py`:**
  - an in-place write to a channel, or to a cached calibrated array, raises;
  - a calibration change still works.
- **`tests/gui/test_plot_appearance_history.py`:**
  - each of the eight windows undoes and redoes an edit, a preset and a reset;
  - the Plots window has only one stack.
- **`tests/analysis/test_lineage.py` and `tests/gui/test_editing_dependencies.py`:**
  - thresholds are judged;
  - the fit digest catches a swap, a resize and a re-measure that keep the count;
  - the Plots window flags a source whose rate or baseline is stale;
  - Export figure refuses a rate-stale table;
  - Re-measure refuses after a calibration change.

### 9.2 Tests that move

These are expected to change. Each will be checked by failing test id against `27ec0eb`.

- **Reset's default and what it keeps:** `test_sidecar_safety.py`, `test_persist_gui.py`, and `core/test_persist_invariant.py:590-640`.
- **The fit digest:** `test_s8_rd1_stored_fit_belongs_to_a_table.py` and `test_s8_rd1_fits_tab_follows_the_shown_table.py`.
- **Revert's keys and the stage buttons:** `test_signal_conditioning_window.py`.
- **Every new signal classified:** `test_surface_staleness.py`.
- **Menu wording:** `test_undo_events.py` and `test_s5_persistence_undo_depth_disclosure.py`.
- **Writes into a document's array after building it (5.1):** any test that does this. The whole-suite run will find them.

## 10. Phases

1. **Undo that reaches everywhere.** Small, and it fixes what is broken first.
   - Defects 1, 2, 3 and 9.
   - `SettingsHistory` with the calibration, time-base and acquisition scopes.
   - The calibration resets and the opened-with default (defects 5 and 6).
   - Re-measure's calibration guard (defect 8).
   - Read-only samples.
2. **Settings you cannot lose.** The largest phase. It changes how the form follows the table (decisions 2 to 4).
   - The form scopes per table, with drafts and the clamp.
   - One press undoing a run together with its settings.
   - Revert, protocol recall and stage toggles as steps.
   - Stage Revert and stage Reset.
   - The Baseline resets.
   - Thresholds in the chain notice.
3. **Plots and dependencies.**
   - Appearance history in the eight windows.
   - `lineage.py` and the list in Details.
   - Fit digests.
   - The Plots window and Export figure.

Each phase ends the same way:

- the whole suite is compared by failing test id against `27ec0eb`;
- the guide and the CHANGELOG are updated;
- nothing is committed unless the owner asks.

## 11. Documentation

- **`docs/USER_GUIDE.md`:**
  - §9b becomes "Undoing a change": what is covered, the rule about the recording on screen, the one press that undoes a run with its settings, and the ceiling wording the app actually shows.
  - §5: stage Revert and Reset, edits that stay with their table, and the clamp line.
  - §3: Reset to default.
  - §5a: Back to default and Clear reference.
  - §9a: the list of inputs.
  - §10: appearance Undo in every plot window.
  - §13: ⌘Z in the Signal Conditioning window, and the gallery's Redo.
- **`CHANGELOG.md`:** Unreleased gets Added, Changed and Fixed bullets for each phase.
- **`docs/ARCHITECTURE.md`:** the undo paragraph gains the third sibling class and the journal's settings entries.

## 12. Decisions for the owner

1. **Reach.** Should Undo take back the newest step on the recording on screen (proposed), or the newest step anywhere, even on a recording that is not shown?
2. **Separate redo.** Should a settings change end only that table's settings redo, and leave its event-table redo alone (proposed)?
3. **One press for a run.** Should undoing a re-run also undo the setting changes made just before it (proposed; this keeps today's behaviour)? Or should each step take its own press?
4. **Edits stay with their table.**
   - When you switch, should the form keep your un-run edits for each table (proposed), instead of snapping back to the run's settings?
   - Should those edits be saved with the project (proposed), or kept only for the session?
5. **Calibration reset.** Should Reset to default return gain, offset and units to the values the channel opened with, and keep the note and "Treat as" (proposed)?
6. **Undo and the sidecar.** Should undoing a calibration change rewrite or remove the sidecar without asking, with the status line naming the file (proposed)? Or should it ask, as Apply does?
7. **Read-only samples.** Is it acceptable that an in-place write becomes an error in that action, instead of silent damage (proposed)?
8. **Thresholds.** Should thresholds be judged with the chain, and should Revert put them back (proposed)?
9. **Plot recipes.** Appearance history goes into every plot window now (proposed). Should recipe history for the Noise spectrum, Distributions and Comparison windows come later, or now?
10. **Fit digest.** Should fits be judged by the values they fitted (proposed)? Existing fits would then read "events changed" after any re-measure or resize.
11. **Which phases now.** All three (proposed), or phase 1 first and the rest after you have looked at it?

## 13. Implementation status (2026-10-31)

All five pieces are built in the working tree on `codex/beta-sweep-fixes`, on top of `27ec0eb`. Nothing
is committed. Every decision in section 12 was built as proposed, and none had to be reopened with the
owner. The work ran in five stages:

- **Wave A:** nine parts, each built by one agent, checked by an independent reviewer and fixed where
  the review found something.
- **B1:** the main window's undo reach and the three settings scopes, with two reviews and a fix round.
- **Wave C:** the form history, the stage controls, the dependency wiring and the docs, built by five
  agents at once in separate copies and merged by script.
- **C2:** the journal reconciled (B1's fix round and wave C wrote the same journal in parallel), then
  the merged tree wired together and run.
- **C2 review and fix:** independent reviewers went through the merged tree, a skeptic re-ran each
  finding and classed it against `27ec0eb`, and a fix stage mended the five that wave C caused (13.2,
  each with a test that fails without its fix). The five that `27ec0eb` does the same are left as
  known edges (13.3) and one owner question (13.4, question 6).

### 13.1 Built

| Part | Files | New tests |
|---|---|---|
| `SettingsHistory`, the third undo stack (3.3) | `core/history.py` | `tests/core/test_settings_history.py` |
| Reset to default restores the opened-with default and keeps note and kind; a sidecar no longer records untouched channels; the workflow chip uses the same default (defects 5 and 6, section 6) | `core/calibration.py`, `core/session.py`, `gui/calibration_panel.py`, `gui/workflow_status.py` | `tests/gui/test_editing_calibration_reset.py` |
| Read-only samples (5.1) | `core/document.py`, `core/segment.py`, `core/session.py` | `tests/core/test_read_only_samples.py` |
| Appearance Undo/Redo in every plot window; deleting a Plots template asks first (4, 3.8) | `gui/plot_appearance.py`, `gui/plots_panel.py` and the eight plot windows | `tests/gui/test_plot_appearance_history.py` |
| One record and one judge of a table's inputs; fits record a digest of what they fitted (7.1, 7.2) | `analysis/lineage.py` (new), `analysis/modules.py` | `tests/analysis/test_lineage.py`, `tests/analysis/test_fit_input_digest.py` |
| Per-stage Revert and Reset; Revert only what differs; thresholds in the chain judge (defect 7, sections 6 and 7.2) | `analysis/conditioning_recipes.py` | `tests/analysis/test_conditioning_recipes_stages.py` |
| Baseline form state for undo; Back to default in every mode; Clear reference (3.9, section 6) | `gui/baseline_panel.py` | `tests/gui/test_editing_baseline_form.py` |
| Typed frequencies survive a Nyquist clamp and the clamp is announced; deleting a protocol or an ML preset asks first; the form's own signals (5.3, 3.8) | `gui/panels.py` | `tests/gui/test_editing_detection_panel.py` |
| One helper for undo and redo keys, each resolved key registered once (defect 1) | `gui/key_sequences.py` (new) | `tests/gui/test_editing_undo_reach.py` |
| Undo that reaches every window: the Redo key and the refresh after a switch (defects 1 and 2), ⌘Z and ⇧⌘Z in the Signal Conditioning window (defect 3), a Redo in the gallery | `gui/main_window.py`, `gui/signal_conditioning.py`, `gui/event_gallery.py` | `tests/gui/test_editing_undo_reach.py` |
| Calibration, time base and acquisition bandwidth in the one journal; undo puts the sidecar back; the Re-measure guard (defect 8); the judge's rule for the estimator's dormant windows; the clamp status line | `gui/main_window.py`, `gui/calibration_panel.py`, `gui/analysis_panel.py`, `core/calibration.py` | `tests/gui/test_editing_settings_undo.py`, `tests/gui/test_editing_baseline_dormant_windows.py` |
| The form history in the Analysis panel: `form_key`, `form_state`, `restore_form_state`, five signals, the writing guard, per-table drafts, the clamp put back (3.4, 5.2, 5.3) | `gui/analysis_panel.py`, `gui/panels.py` | `tests/gui/test_editing_drafts.py` |
| Form scopes in the main window: the 700 ms pause, the five signals, labels, one press for a run (3.6), drafts saved with the project (decision 4) | `gui/main_window.py`, `core/form_draft_state.py` (new) | `tests/gui/test_editing_form_undo.py`, `tests/gui/test_editing_drafts_persistence.py`, `tests/core/test_form_draft_state.py` |
| Stage Revert and Reset in the Signal Conditioning window, the stage marks, the "Settings changed" notice, a legacy CUSUM table read as its own | `gui/signal_conditioning.py`, `gui/analysis_panel.py`, `gui/panels.py`, `gui/main_window.py` | `tests/gui/test_editing_stage_resets.py` |
| Dependency wiring: the Details list of inputs, the Fits-tab wording for the digest, the Plots window's rate and estimator marks, Export figure's staleness check | `gui/analysis_panel.py`, `gui/plots_panel.py`, `gui/main_window.py` | `tests/gui/test_editing_dependencies.py` |
| The journal reconciled, and the seams between the parts (13.2) | `gui/main_window.py`, `gui/analysis_panel.py` | `tests/gui/test_editing_journal.py`, `tests/gui/test_editing_integration.py` |
| Documentation (section 11) | `docs/USER_GUIDE.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md` | the docs guard tests |

### 13.2 Built differently

Each item says what the build does where this design says something else, or nothing.

**The journal and the menu (3.1 to 3.7)**

- **An event step names its table.** The journal entry is `(_UNDO_EVENTS, key)`, not a bare "events". A
  bare entry answered for whichever table was on screen, so a newer curation on another recording
  stood in front of the newest step on this one (B1's review). Closing a recording takes its entries
  out of both journals.
- **Five kinds of entry, not three.** A press for a run is `(_UNDO_RUN, form scope, n)`, and
  `(_UNDO_BREAK, key)` only separates: a table's first run is no undo step, so without a break the
  settings that produced it joined the second run's press. The break lives in the undo journal only
  and nothing undoes it. The journal's bound has one door, `_journal_undo`.
- **One press for a run (3.6) rebases the form between its halves.** Undo takes the table back,
  rebases the form (its re-seed must not become a step), then undoes the settings steps newest
  first; redo replays them oldest first, then the table. A write that raises puts back what the press
  had moved, and nothing is undone. A settings step separated from the run by other work stays a step
  of its own, and its first press can move nothing visible, because the re-seed has already put its
  "before" on the form.
- **A pending press keeps the halves its stacks hold.** A step on the table ends the table's redo
  half, a step of the form ends the form's, and a step anywhere else leaves it whole
  (`_split_redo_runs` asks the stacks).
- **`move_scope` only for a born table (3.4).** The steps made before a table existed move onto its
  first table only when that table is the one the session is announcing (`Session.announcing`). A
  switch to a table that already existed, the detector combo back to a table with a run, moves
  nothing: the moved steps would stand in front of that table's own. `move_scope` itself never
  splices two redo branches, trims the merged scope to the depth ceiling and counts what it drops.
  A table that lands while the panel shows another recording (a batch run, or a Detect that lands
  after a look away) never reaches the panel's arrival, so `MainWindow._on_table_landed` hands
  the steps over on `resultsChanged`, through the same door (`_hand_over_form_steps`), whenever
  that recording and channel's table-less scope holds any.
- **The menu has more wordings than 3.5 names.** "Undo — nothing on this recording" is said only
  when no event table holds any history and every step left is a setting of a recording that is not
  on screen. "Undo — nothing to undo on this event table" also covers a recording whose only steps
  are another table's or channel's form, which wait for their table. A settings ceiling has its own
  wording, "Undo — this recording's oldest settings history was dropped at the 30-step ceiling (N
  steps lost)". The detector-switch wording counts other tables' form steps, and a press for a run
  names itself ("Undo Re-run detection and 2 setting changes").
- **An edit inside its pause can be taken back.** Undo is enabled only while a step exists, an edit
  waits 700 ms before it is one, and a disabled action never receives its key, so ⌘Z was dead for the
  first 700 ms of a form's first edit. The menu now offers a bare "Undo" while an edit waits and
  nothing else can be taken back, and `_end_form_pause`, the one door every ending of the pause goes
  through, brings it up to date.
- **An edit inside its pause is the older step.** Every other step is journalled when it is pushed,
  an edit when it is committed, so a curation or a calibration made inside the pause was journalled
  before the edit it found waiting: Ctrl+Z took the edit while the menu named the curation, and an
  edit that a curation had separated from a run joined the run's press (3.6's last bullet).
  `_note_step` commits a waiting edit before it journals anything, which makes "before any other
  step is recorded" a moment that commits, beside the ones 3.2 lists. A form with an edit waiting
  also offers no Redo (`_form_edit_waits`), as the Plots window and the appearance widget do: the
  commit ends that branch, and a press would have been enabled, named a step and taken nothing. A
  pending press is named without its form half meanwhile.
- **`SettingsHistory.changed` drives the menu only.** 3.3 says it emits `stepRecorded` and `changed`
  as its siblings do, and theirs dirty the project. Here `changed` is not wired to the dirty flag:
  closing a recording drops its scopes and emits it, and `test_close_guard` pins that closing a
  recording with no project open does not dirty the session. Each setting already dirties the project
  through the session's own signal, and a form through `formEdited`.
- **`SettingsHistory` guards against its callers.** A reader that raises counts as `None`, a describer
  that raises or returns nothing falls back to "Change settings" (a bad label never loses a step), and
  a state that cannot be deep-copied is kept by reference. The comparison is exact, bar `NaN == NaN`,
  a list against a tuple, numpy values and dataclasses field by field, since reading an edit as
  unchanged would lose it. `clear()` emits `changed` only when it held something, as
  `EventHistory.clear` does.
- **Any form edit dirties the project,** the ML form's included. The ML widgets were wired to nothing
  that reached the dirty flag at `27ec0eb`; a form's un-run edits are project state now (decision 4).
- **A named step is committed at once** (`formStepNamed`), under its label. B1's pending-label table
  serves the calibration scopes only. "At once before a run" also covers a re-measure and a batch
  start, which emit no `willReplaceEvents`.
- **Labels go past 3.2's examples.** "Change N settings" counts settings, not keys: the ML form's
  derived keys, the frequency intents and the baseline's source and accepted-over mark do not count
  on their own. A setting of a stage that is off is worded as if its stage were on ("Change notch Q
  30 → 35"), and the fallbacks are "Change <key words>" and "Change settings". A calibration change
  nobody named reads "Change calibration (gain 1 → 2)", "Treat channel as current" or "Edit
  calibration note"; the time base reads "Declare 25,000 Hz", "Declare the time column in
  milliseconds" or "Use the file's own time base"; the acquisition bandwidth reads "Change
  acquisition bandwidth" or "Use the file's acquisition bandwidth".
- **The time base and the override are read wrapped.** `None` means "cannot be read now" in
  `SettingsHistory` and "nothing declared" in the session, so a state is `{"declaration": …}` or
  `{"override": …}`. A segment's time base is not tracked, since it follows its recording's
  declaration and one declaration is one step. A segment cut rebases the calibration and override it
  inherits, or the next checkpoint would record the inheritance as a step.
- **Undo puts the sidecar back, byte for byte (3.2, 5.4).** 3.2 wrote the calibration back through
  `Session.set_calibration(..., write_sidecar=True)`. That re-derives the file by the payload rule,
  which took a note off a channel at its default numbers, deleted a file whose only entry that was
  and, after a project restore, wrote a gain the file had never held, none of it asked (B1's review).
  A calibration step now carries the file's bytes on each side of its commit, the Calibration panel
  announces each commit (`commitStarting`), and `core.calibration.restore_sidecar` puts the bytes
  back. It writes only while the file still holds what the step left, and otherwise leaves it
  (`SIDECAR_KEPT`, in its own words): another channel's step still in effect, or an edit made outside
  the app. 5.4's "exactly as it was" therefore holds, a hand-written file included, and decision 6
  needed nothing more. The Calibration panel's own line repeats what the undo did to the file, a
  failed write reads "Undone: …, in this session only", and an Apply of values already there is no
  step.
- **⌘Z in the Signal Conditioning window** is forwarded to Edit ▸ Undo as 3.7 says, but until chain
  edits were steps it took back whatever came last, curation and calibration included. With the form
  scopes it takes back the chain edit made there. The "Undone: …" line is on the main window's status
  bar, behind it (13.3).
- **The gallery's key fallback matches every key an action holds** (`_handle_own_key`), not only the
  first. Offscreen, Redo holds Ctrl+Y and Ctrl+Shift+Z, and the fallback ignored ⇧⌘Z where the shortcut
  map did not reach the gallery.

**Reset to default and read-only samples (defects 5 and 6, 5.1)**

- **A segment takes its parent's opened default.** The voltage encoding belongs to the file. A window
  cut where the bias was low would read as a plausible bias and reset to 1.0 where its parent resets
  to 0.1.
- **A note the next open writes back by itself is not "lost".** An older build stored the untouched
  host-guest voltage channel with its auto-generated note, so the first Apply after this change
  would have asked "Apply will erase 1 calibration note". A typed 0.1 on that channel is now
  stored: the median-less default used to drop it, because 0.1 equalled it.
- **The workflow chip compares with the opened default** and, as before, ignores the "Treat this
  channel as" kind, which the sidecar does count.
- **The Reset button's tooltip** says what it keeps (note and kind) and how to clear the kind: set the
  combo back to "as the file declares" and press Apply.
- **No in-place write existed.** An AST scan and a dynamic trace found none in `nanoscope/`; only
  tests wrote into a document after building it. `Session.x_axis`'s cached time arrays are shared by
  reference and are not flagged read-only.

**Plot history (4)**

- **⌘Z and ⇧⌘Z are live while the appearance widget is on screen**, which in the Analysis Explorer is
  while the page's Section shows Appearance. Parenting the shortcuts higher brought back ambiguity
  between pages. The Plots window turns the widget's history off and keeps its own stack.
- **A refused undo moves nothing.** A host that puts back the state the undo was leaving, as Comparison
  figures does when the figure will not render, leaves the stack where it was, and `undo()` returns
  False.
- **A step is named by its field** when one field moved ("title size 12 → 14 pt"), by the preset when
  several land on one, and "reset appearance" or a count otherwise.
- **The history ends** in every host's restore door, and also in `StudyFigureWindow.reset_style` and
  `SubpopReportWindow.set_report(None)`.
- **The widget is 13 px wider** in its non-compact layout, for the Undo and Redo buttons beside Reset
  appearance; the compact layout keeps its width.
- **Deleting asks** for protocols, saved ML presets and Plots templates; 3.8 named two of them. The ML
  prompt also says which default preset the tab falls back to. Deleting a name that is not in the
  user's own store changes nothing, where it used to drop the combo to "— manual —".

**Forms and drafts (3.2, 3.4, 5.2, 5.3)**

- **The form's snapshot is more than `params()` (3.2).** `form_state()` is `detection`, `intents`,
  `ml` and `baseline`, JSON-safe, without the detector (the table on screen decides it) and the model
  path. `intents` is what each frequency box was last set to, because `params()` holds what a box
  reads, which on a slower recording is the clamp. With it, an undo of an edit made to a clamped box
  itself puts the typed value back. The Baseline form's snapshot holds both windows, their source, the
  accepted-over mark and the reference editor whole (typed text, confirmed flag, restore error).
- **A same-table result does not re-seed a form holding edits nobody ran** (5.2 only says a key the
  user edited keeps the edit). An event undo or redo, a batch run, a merge or the banner's Re-run
  lands on the table on screen and leaves the form where it is, still marked edited, and the chain
  notice says how it differs. A form an undo has walked back to the seed re-seeds as before. A form
  the panel keeps over a new table clears its `_form_seed`: the seed belonged to an older table than
  the one just installed under it (after the redo of a press it still held the first run's form), so
  a value typed back to it read as a form an undo had walked back, and the next event undo or redo
  re-seeded the edit away.
- **Which runs settle a draft.** Detect, Run ML and Re-measure, started from the form. The banner's
  Re-run replays the stored run, not the form, and a batch run is not from the form either.
- **The detector combo.** It re-seeds min dwell and merge gap as it moves, before any slot of the panel
  runs. The table switch is deferred to a slot connected after it, and the form is filed as it stood
  before the combo moved. The knock-on is neither an edit nor dirty, unless no other table comes on
  screen (13.4, question 3).
- **`form_key()` still names the old key while `formLeaving` is announced,** so the listener can read
  the form it is about to lose. On an arrival the order is `formLeaving(old)`, then `formWritten`
  (the guard closes), then `formArrived(old, new)`; both of the last two rebase, and `move_scope` keeps
  the newer pristine copy, so the order is harmless.
- **More writes run under the guard than the contract named:** the detector combo's move by the app
  (`_set_active_detection`), the baseline-mark replay and the resets in `_show_empty`. When the
  outermost guard closes it also files the form under the table on screen, which covers a project
  restore that picks a table directly. A write-back for a table that is not on screen raises, which
  keeps its step, where writing nothing would have spent it.
- **`form_drafts()` includes the form on screen** when it holds edits, and `set_form_drafts()` puts the
  on-screen key's draft back at once, since for the same target no arrival ever comes.
- **A recording and channel's first table consumes the form that stood for it.** Born on screen by
  anything but a run of the form, it left behind the draft that `_leave_form` had filed under
  `(doc, ch, None)`; born while another recording is shown, it never reached the panel at all.
  Either way the draft was a draft of no table: nothing took it out, the next project save wrote it
  as a record with no module, and it came back on the form of the first detector that had no table.
  It is dropped (`_on_results_changed`, `_table_landed_elsewhere`). The table seeds the form, as a
  first table always has, and the edit is reachable as the undo step that moved onto the table.
  Whether it should come back on the new table instead, as it does on one that already exists, is
  the owner's (13.4, question 6).
- **The ML notch box drops an infinite entry**, as it does any other entry that is not a frequency
  (`DetectionPanel._parse_notch_freqs`). "inf" and "1e999" parsed, were positive and were then refused
  by `MLPreprocessConfig`, so `ml_params()` raised from the one box meant to be lenient. The form's
  snapshot reads it at every seed, leave and save, and a switch of recording stopped half way.
- **A project restore runs under the writing guard.** Both the Analysis-panel recipe and the reset's
  default replay do. Without it the restore's own writes read as the user's edits, and the next save
  wrote a draft for a table nobody had touched. A real save and reopen proves it: the reopened form is
  the saved one, counts as edited, makes no step and leaves the project clean. The drafts are held
  across the recipe: it puts the detector combo where it was saved, which can be a detector the
  recording has no table for, and that moves the table under the form with no switch. Nothing filed
  the form the arrival had put on screen, the guard re-filed it under the key it had landed on, and
  its table's draft was nowhere after the reopen (the next save dropped it); an edit of the no-table
  form counted as no edit, and the first look away dropped it. `_finalize_restore` takes the drafts
  before the recipe and puts them back after it.
- **Draft persistence differs from the table choices, on purpose.** The live session replaces every
  readable record about a channel it holds, present or not, so a draft its own run settled is not
  resurrected; a record about a recording that is not open is kept. The first of two records for one
  table wins, and a record applies only to a channel under the name it was made for. An unreadable
  section is kept beside a fresh one as `analysis_form_drafts_v1_unreadable` and named in a banner.
- **The clamp sentence (5.3)** is "Low-pass 20 kHz is above this recording's Nyquist (12.5 kHz); the form
  now reads 12.4999 kHz.": the box's own digits, "at or above" for a value exactly on Nyquist, and
  "above what the form accepts while the sample rate is unknown" with no rate. It names only a stage
  a run reads (low-pass on, high-pass on, the notch on with its series spelling) holding a value
  someone set, never a box's built-in default. It is read after the seed or the draft has landed, at
  the end of a target change and after a rate declaration, and not forwarded as a signal, because
  `clampNotice` fires before the seed.
- **A value equal to the ceiling a clamped box already reads is not taken as set.** Writing a
  snapshot taken after a clamp back on a slower recording, an undo of any other step, therefore keeps
  what was set. A value above the ceiling is remembered as asked, one a protocol recalled on a slower
  recording included, so a faster recording gets it back.
- **The put-back no longer depends on the way there.** At `27ec0eb` a 30 kHz value read 12,499.9 Hz on
  a 50 kHz recording reached through 25 kHz and 24,999.9 Hz reached directly. Every box is now given
  its remembered value and the range clamps it.
- **`mlParamsChanged` fires once** per hand edit, per preset load and per `set_ml_params`, the analysed
  range included, and `mlPresetLoaded` carries the preset's key.

**Resets and the notice (6, 7.2)**

- **Revert can be withheld.** `revert_params` is None when a stage it would revert would still differ
  afterwards, which is a difference in a setting the run never recorded. The window's Revert and the
  chip menu's entry are then disabled, and the judge's sentence says why: "Run detection to apply it —
  Revert cannot restore a setting the run did not record." A stage's on/off the run never recorded
  comes back as off, a recorded `detrend_decim` of None (the automatic stride) comes back as 0, and
  polarity comes back with the baseline-drift stage when it ran. It is withheld too when a frequency
  the run recorded is at or above the recording's Nyquist now (a slower rate declared since), since the
  form's box cannot hold it: "Revert cannot restore the run's low-pass 10 kHz, which is above this
  recording's Nyquist (8 kHz)" (`revert_withheld`). The Details list's chain and threshold rows follow
  the same verdict ("Re-run to apply"). Sweep 10 S10-STAGE1/2.
- **The notice's words.** The headline is "⚠ Settings changed: k_in 5 → 6", with no detector label, and
  the sentence is "⚠ Settings changed since the Schmitt detection on screen ran: k_in 5 → 6. Revert,
  or run detection to apply them." The chip reads "⚠ Settings changed", the window's button "Revert
  to detection's settings" and the chip menu's entry "Revert form to this detection's settings". The
  step is named "Revert chain to Schmitt's run" or "Revert settings to CUSUM's run".
- **The judge reads a max dwell or merge gap of 0 and of None as the same,** off, because the form
  cannot tell them apart and a difference no Revert could clear would be a difference it should not
  name. Any other threshold that is None, not a number or not finite counts as not recorded.
- **Reset stage** is enabled whenever a stage is shown. When the stage already holds its defaults it
  names no step and says so.
- **The stage marks change font and ink only** (bold, the notice's amber), so a row keeps its height
  and its text. `MainWindow` pushes the window its stage states with every verdict, and again when it
  opens. The verdict is announced again when any field of it changes, not only its sentence: the
  sentence names three differences, and a fourth could move under it.
- **Clear reference is never disabled.** The module's rule is that a locked button is dimmed and
  explains itself. A click with nothing to clear says "There is no open-pore reference to clear: this
  recording already uses the Automatic estimator." and emits nothing. "Something to clear" means the
  reference form feeds the run, which includes a stored reference that could not be restored.
- **Back to default under a reference** keeps only itself from the estimator's row, beside the caption
  "Automatic estimator (not in use): 0.5 s / 0.1 s", in Apply's cell. Measure this recording and Use
  these values hide with the windows. Its refresh used to re-raise the "actually ran at … not the X
  above" line over hidden spins, and that is fixed in the line's one writer.
- **A press of it no longer stales the table** (the earlier "must land with B1" item, whose strict
  xfail is gone). `AnalysisPanel._live_baseline_stamp` is the one place the live estimator stamp is
  computed, and under the same open-pore reference it takes the stored dormant estimator fields. The
  Details list's live baseline comes from the same place.
- **A legacy CUSUM table is judged, seeded and re-run on its own k and h.** The panel's provenance
  fallback dropped `cusum_k` and `cusum_h`, so the notice never compared them; the notice, the form and
  the Details list now read one record, the detector's stamp.

**Dependencies (7)**

- **The fit digest names a row by `start_idx`, not `event_id`,** and hashes the rows in start order,
  which makes 7.2's "in event order" true. A merge renumbers `event_id` and re-sorts the rows, so by id
  two rejected fragments merged into one rejected row moved both fits' digests, and Export figure
  refused them. The digest is `sha256:<hex>`; a table without integer starts gets none and keeps the
  count rule, which still applies to a fit that has a digest (the digest only adds a reason,
  `FitVerdict.digest_moved`). A table with string or NaN event ids but integer starts, a manual
  catalog, gets one too. An edge drag re-measures that event's depth as well (`_write_measured_row`),
  so a resize moves both fits' digests. `fit_dwell` and `fit_depth` stamp it, and the Plots window's
  fits, which may fit a narrowed range or another column, do not. `fit_depth`'s `input_column` is always
  `frac_depth`. The fit's meta also records the table's `fs` and, now that the panel hands the stored
  result to the fitters, its `baseline_fingerprint`.
- **Lineage reuses the notice's rule for thresholds** (`threshold_differences`) and has no comparison
  of its own. A table with no estimator stamp is "unknown" with the action Re-measure, as the amber
  line has it for a table that predates the stamp.
- **One judge.** `AnalysisPanel._staleness(res, *, on_screen)` is the verdict. The amber line shows it,
  and the Plots window asks it of every table it draws (`table_staleness`). The rate and the
  calibration are judged for every table; the estimator and the reference only for the table on
  screen, because the Baseline form belongs to it.
- **The Details list** reads "Inputs, as made → now:" with a row "was → now — action" per input, the
  estimator in words. For a rate the recording no longer knows, the row's action is "Declare the rate
  (Inspector ▸ File Info ▸ Sample rate / time unit…), then Re-run"; the amber line's own button is
  still a gated Re-run.
- **The Plots window's wording** is a "STALE: <columns> on this plot …" note and a "⚠ STALE: <columns>"
  headline in Details, beside the unchanged SUPERSEDED CALIBRATION note.
- **The Fits tab** says "the fitted events changed since the fit (swapped, resized or re-measured)"
  only where the counts cannot tell; when the count moved as well, its count sentence stays.
- **Re-measure's refusal** reads "Nothing was re-measured. Re-run first — the calibration changed since
  this table was detected."

**Tests that moved (9.2)**

- **Reversed by decision 4:** `test_p3_regressions::test_c24_selecting_another_recording_still_seeds_from_that_one`
  is two tests now (an unedited A still seeds from its run; an edited A's draft comes back, B's does
  not follow), and `test_baseline_migration`'s confirmed-configuration test reads source "user" on the
  return to A.
- **Reversed by decision 5:** `test_persist_gui` (Reset keeps the note) and
  `test_trace_view_kind_and_framing` (Reset keeps the override, whose own control clears it).
- **Writes after building (5.1):** `test_segment` and `test_analysis_launcher`; and
  `test_manual_event_door`'s seed helper, whose seed was an unannounced pending change that Ctrl+Z's
  commit would have taken for a step.
- **Import guards** that pin a module's gui imports exactly gained `key_sequences`:
  `test_measure_navigation`, `test_signal_conditioning_window`, `test_signal_conditioning_previews`,
  `test_subpop_galleries`.
- **The new prompts:** `test_filters_dropdown`, `test_detection_protocols_and_normalize` and
  `test_plots_undo_templates_json` answer the delete question. `test_conditioning_recipes` pins
  Revert's new rule. `test_surface_staleness` classifies the form's five signals, the gallery's Redo
  and the three `SettingsHistory` signals.
- **9.1's `tests/gui/test_editing_resets.py` was not written.** Its cases live in the baseline,
  calibration and stage-reset files. The files 9.2 expected to move and did not: `test_sidecar_safety`,
  `core/test_persist_invariant`, the two s8_rd1 files and `test_undo_events`.

**Documentation (11)**

- **§9b's old anchor still resolves.** `docs/MANUAL_ZN_WORKFLOW.md` links
  `USER_GUIDE.md#9b-undoing-a-change-to-the-event-table`, so the renamed section keeps it as an explicit
  anchor beside the new one. Export figure's refusal is written in §12, where the command is
  documented, and §10 points to it. The guide's edition line is the owner's to move.
- **The amber line's Details popup is measured again each time it opens.** The review of the merged
  tree found that a `QMenu` keeps the height of its first open and does not re-measure a widget action
  that grew, so the Inputs list was cut off at the bottom by up to 52 px when an input had changed
  between two opens. The sentence label had clipped the same way on `27ec0eb`, but it was the new
  list, which changes with every open, that made it show. `_StaleBanner._open_details` now removes
  and re-adds the action before the popup opens. The test
  (`test_details_is_measured_again_each_time_it_opens`) opens Details four times with 2, 9, 4 and 12
  rows and fails with those two lines taken out.

### 13.3 Not built

- **The focus-leave commit (3.7).** A text edit becomes a step after 700 ms, at a switch, before a run,
  and at Ctrl+Z, the Edit menu, a save and a quit. "When focus leaves the field" is not a moment of its
  own: a global `QApplication.focusChanged` hook was judged risky, given the suite's PySide teardown
  crashes.
- **Naming the stale table in the exported figure's provenance (7.2).** Export figure names it on the
  status line only; `plot/figures.py` was no builder's file.
- **The estimator verdict for an off-screen table that holds a draft.** The Baseline form's draft is
  typed text, and the only parser from it to a `BaselineConfig` is the live panel and its reference
  editor. A second copy of that parser, or a scratch widget, was the price. Such a table is judged on
  rate and calibration, as any other table off screen is.
- **A sign of an Undo pressed in the Signal Conditioning window.** "Undone: …" is on the main window's
  status bar, behind it (B1's finding R1-6, second half).

Edges that are known and left as they are:

- **An edit's "edited" mark outlives its undo.** After the undo of a run's press, or a typed edit
  undone, the form equals the table's run but is still marked edited, so the next save carries a draft
  equal to the run. Harmless to every number. A fix would compare the restored form with the seed,
  which a form kept over a new table no longer has (13.2).
- **A named action inside the pause of an earlier edit** (a Revert, a Reset stage, a recall pressed
  within 700 ms of typing) commits that edit as a step of its own, labelled by the action. A rate
  declaration's clamp adopts a pending edit without a step. Both are rare.
- **A form step separated from a run by other work** moves nothing visible on its first press (3.6's
  accepted trade). `form_state()["intents"]["chosen"]` is hidden state a landing run moves and an
  older step restores; it only matters for how a later clamp is put back.
- **"Redo — nothing on this table; the CUSUM detector (1 step) on this channel"** can name undo steps,
  as `siblings_with_history` always did for events.
- **An undo that deletes the sidecar says "(all channels are back at default)"** whenever the file
  did not exist before the step, which describes the disk and not the session. Another channel can
  still be calibrated in this session only: after a project restore whose gain differs from the
  file's, after an Apply whose write was refused, or when the sidecar was deleted outside the app
  before the step (sweep 10 S10-CAL2, judged not a defect).
- **The Signal Conditioning window's own Undo and Redo are never disabled;** a press with nothing to
  take is a no-op.
- **`Session.x_axis`'s cached time arrays** are shared by reference and still writeable (13.2).
- **A setting typed while a run is in flight is overwritten when the run lands.** A run from the form
  settles the form it was launched from, and the landing re-seeds it from the table whatever was
  typed since. Identical on `27ec0eb`. A fix would keep the form's state at launch and settle only
  a form that still equals it.
- **A project restore whose recording is missing applies the saved recipe to whichever recording
  is on screen.** The recipe's baseline windows and detection settings then show on that recording
  as if they were its own, and the next save writes them as its recipe. Identical on `27ec0eb`;
  that recording's own draft is kept for a session that can apply it.
- **Undo then Redo of a form step can move more than the step changed.** A form step is the whole form
  as it stood before and after it, so a press writes all of it back. When something else wrote the form
  in between (a table's seed, a kept edit, the other detector's defaults for min dwell and merge gap),
  the press restores those fields too, and Redo does not return to where Undo started. Found by running
  the reviewers' random walks again on new seeds after the fix stage: 4 of 202 sequences fail the
  walk's "undo then redo changes nothing" check (the form walk's seeds 42, 65 and 67, the undo walk's
  seed 1020). Seeds 65, 67 and 1020 fail the same way before the fix stage and are the accepted trade
  of 3.6 in its wider form. Seed 42 is new with the hand-over of a first table's steps (13.2): a CUSUM
  table that a batch run puts on a recording takes steps made in Schmitt mode on that recording's
  table-less form, and undoing one sets min dwell and merge gap back to Schmitt's. A fix would write
  only the keys a step changed (the before and after difference) instead of the whole snapshot. It
  reaches every `SettingsHistory` scope, so it is left for the owner's decision.

### 13.4 Questions for the owner

1. **An emptied Title, X or Y label box.** It looks the same as one never touched (both show the
   automatic placeholder). Comparison and Study figures then draw no label, while the other plot
   windows draw the automatic one. This is unchanged since `27ec0eb`. One rule for all of them?
2. **Polarity is judged only while the baseline-drift stage runs.** `detect_events` also reads it, so a
   polarity edit with that stage off leaves the table looking current. Measured on the merged build:
   the notice still reads "match". Should polarity be judged on its own?
3. **A detector switch re-seeds min dwell and merge gap.** The combo has always moved a knob that still
   holds the other detector's default to this one's (Schmitt 20 µs and off, CUSUM 1 ms and 0.5 ms).
   An earlier version of this question said the table still on screen would then read "Settings
   changed", and asked whether Revert should restore the detector. It does not read that, because the
   combo shows the other detector's table (or "no stored result") along with it. What is left: when no
   other table comes on screen (a fresh recording, say), the re-seed counts as an edit ("Undo Change 2
   settings"), and Undo puts the two settings back but not the detector choice, since the detector is
   never part of a form's state. Should the detector go back with the step, or should that re-seed be
   no step?
4. **A baseline-form edit does not refresh the Plots window live.** It raises `stalenessChanged`, which
   never reached a Plots redraw. The "STALE" mark appears at the window's next draw: Refresh, any view
   change, or a calibration or rate change, which already redraw it. Wiring the signal to a redraw
   would make it live, at the cost of a redraw per knob step. Live, or at the next draw?
5. **Not verified on a real Mac.** In the Analysis Explorer and the study figure windows, which have no
   menu bar of their own, ⌘Z should reach the plot's appearance history and not Edit ▸ Undo through the
   main window's native menu. Only the offscreen routing is tested.
6. **A first table the form did not make.** A batch run, or a hand-made event, that gives a recording its
   first table seeds the form from that table, so an un-run edit typed before it leaves the form; it is
   the table's first undo step (Undo, then Redo, brings it back) and no draft is kept for it. This is
   what the app did on `27ec0eb`, and `test_c24_a_genuinely_new_table_on_the_same_target_still_seeds_the_form`
   pins it for a project restore. On a table that already exists the same landing keeps the edit and
   the chain notice names it. Should a first table keep it too, by moving the draft to the new
   table's key?

### 13.5 Checks, and what is left

- **The merged tree, 203 test files** (every file that is new, differs from `27ec0eb` or names a changed
  module): 186 green, 8 skipped by the rule against files that name the owner's path, 3 red with the
  same ids on `27ec0eb` (`test_lay2_detection_actions_above_the_fold`, `test_lay_detection_panel_reachable`,
  `test_wire_fold2r_stage_selector`), and 6 that crashed under three-job load in PySide's object
  construction, a known flake, each green when run alone. 3,184 tests passed in the files that ran.
- **The 20 new editing files, one at a time:** 375 tests, all green.
- **After the review fixes, the same 203 files:** 190 green, 8 skipped by the owner-path rule, 3 red
  with the same ids on `27ec0eb`, and 2 that crashed under three-job load in PySide's object
  construction, each green when run alone. 3,316 tests passed in the files that ran. Of the 24 tests
  the fixes added (five test files, `test_editing_journal` through `test_editing_detection_panel`),
  23 fail with their own fix taken out and all pass with it in (the 24th, one case of the notch box
  that already held, is the control). The 103 files that touch the undo machinery, a restore or the
  forms were run again after the last source change: 98 green, 4 skipped, 1 load crash, green alone.
- **Not run:** the test files that name the owner's path, which the runner skips (64 in the tree).
  The final list reached eight of them, `test_adversarial_project_io`, `test_analysis_integration`,
  `test_fix_fit_bootstrap_panel`, `test_plots_density_units`, `test_plots_fit_honesty`,
  `test_restore_carry_and_identity`, `test_s6_plots_undo_depth_disclosure` and
  `test_wire_measure_basis_and_export`, and `tests/test_s7_fixup_manual_agrees_with_app.py` is a ninth
  outside it. `test_s6_plots_undo_depth_disclosure` and `test_shared_invariants` exercise the Plots
  window's keys and history, so they are the ones to run by hand.
- **The whole suite on the finished tree (2026-10-31),** compared by failing test id with `27ec0eb`
  as section 10 asks. A fresh export of the working tree, one process per file, four jobs: 701 files,
  64 skipped by the owner-path rule, 622 green, 6 red and 9 crashed in that run. The 9 crashes were
  PySide object-construction segfaults under load and each passed when run again alone; the one error
  among the 6 (`test_iv`, 13 passed and 1 error) passed 14 of 14 alone. The 5 files that stay red fail
  with the same 60 ids on `27ec0eb` (`core/test_persist_invariant` 1, `test_lay2_detection_actions_above_the_fold`
  28, `test_lay_detection_panel_reachable` 20, `test_modal_guard` 1, `test_wire_fold2r_stage_selector` 10):
  **zero new failing ids.** A run of the tree before the fix stage had the same result.
- **Two flakes, measured on untouched `27ec0eb`.** `analysis/test_combined_filter_response` fails one of
  three tests at random (9, 4 and 12 of 100 runs there, 17, 1 and 8 on this tree; the module and the
  test file are byte-identical in both). `gui/test_signal_conditioning_window` segfaults while a main
  window is built in 2 of 8 solo runs on both trees.
- **The reviewers' random walks, run again on new seeds after the fix stage:** 202 sequences, 198 clean
  (the undo lens's table, table-less and long walks and its two-recording walk, and the form lens's
  three). The 4 that fail are the round-trip edge in 13.3.
- **Left:** the manual is the owner's to regenerate:
  `docs/inventory.json` still describes the old Edit ▸ Undo and Redo, Reset to default, Revert to
  detection's chain and Back to default, and has no entry for Clear reference, Revert stage, Reset
  stage, the gallery's Redo, appearance Undo and Redo or the delete prompt. Nothing is committed.

### 13.6 Sweep 10 (2026-10-31)

A fast bug sweep of `7e4e0f6` found 13 defects: 3 major, 10 minor. The findings, repros and
skeptics' verdicts are in the session notes (`fast-sweep/FINDINGS.md`). Five builders fixed them in
parallel, and an integrated-tree review followed. What changed in the design's terms:

- **Calibration.** `save_sidecar` writes nothing when the file already opens as the write would
  leave it, so a v1 file is not re-encoded. A commit that moves no number but does change the file
  is a step all the same (`SettingsHistory.checkpoint(force=True)` from `_checkpoint_calibration`):
  a project's gain written over the file's own, or the Apply the `SIDECAR_KEPT` line advises. The
  journal's expected bytes are therefore always the bytes the app last wrote (S10-DATA1).
- **Journal.**
  - The "born table" test is `Session.announcing_new`: the key held no table before the add (S10-FORM2).
  - 3.6's fold keys on `EventHistory.last_step_run`, not on the step's label. A run's step is named
    as a run (S10-UNDO2).
  - A landing drops the redo steps it has already reached (`SettingsHistory.drop_redo_reached`, S10-LIFE1).
  - After a table step, the redo journal mirrors what the stacks still hold (S10-UNDO1).
  - Draft records are type-checked (`_VALUE_KINDS`), and a draft the form cannot take is dropped
    whole (S10-FORM1).
- **Export.** Generate Report figures asks `report_fit_problem`. The fit's column must be the
  figure's (`dwell_s` / `frac_depth` or an alias of the same quantity), and a fit that records
  another table is refused. A fit that records none keeps the count rule (S8-RD1) (S10-LIN3).
- **Details.**
  - Once the rate moved, the estimator is judged at the table's own rate, and its action is Re-run (S10-LIN1).
  - The Fits box is redrawn on a calibration change (S10-LIN2).
  - Revert is withheld above Nyquist (7 above, S10-STAGE1/2).
- **Keys.** `_UndoKeyGuard` (13.1, 3.7), and appearance sizes step from Default to the figure's
  floor (S10-PLOT1/2).

Left open:

- **A Plots fit over a typed Fit range** records the full accepted count and names no table, so it
  passes the export's count rule and is drawn on the full-range histogram. A fix needs an owner
  call; the options are in the notes (`fast-sweep/fixes/fitguard.md`).
- **An edit typed into a CUSUM no-table form while a batch runs** can still be overwritten by the
  run's values when that run seeds the form. This is 13.3's typed-during-run edge, reached by the
  batch.
- **Batch detect stamps the combo's detector into a Schmitt run's params** (`detector:
  detect.cusum` when the combo shows CUSUM). `_fill_form` copes with it; the batch itself is
  unchanged.
- **A focused plain `QLineEdit` keeps ⌘Z with nothing typed in it.** The guard covers spin boxes
  and editable combos only.
