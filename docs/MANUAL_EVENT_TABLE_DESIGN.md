# Manual events — design (draft 3)

Status: implemented on 2026-10-29 (phase 1 and most of phases 2 and 3; section 16 lists what
was built differently or not at all). Written 2026-10-28 against `f9488d8`.
Code references are `file:line` at that commit. Draft 1 and draft 2 were each reviewed
against the code; section 15 lists what those reviews changed.

## 1. Goal and the problem today

Manual selections should be detections in their own right. They should show in the
Results dock, the Event Gallery, the Event Inspector and every export. When the user
chooses, they should be counted together with the automatic detections, and moving
between manual and automatic work should be seamless.

Today there are two manual paths and neither does all of that.

| | Add measurement to list (Ctrl+M) | Mark selection as missed event (Ctrl+Shift+M) |
|---|---|---|
| Writes to | one session-wide measurement table, `MEASUREMENT_SCHEMA` (`analysis/measure.py:233`), 32 columns | the shown detector table, as a hand-added event row (`gui/analysis_panel.py:4754`) |
| Is an event table | no | yes |
| Gallery, inspector, Reconcile, event exports | never | yes, at once |
| Needs a detector run first | no | yes, and refuses Reconciled |
| Survives a re-run of that detector | yes | no: the re-run replaces the table |
| Survives closing the recording | yes (`measure_panel.py:340-372`) | no |
| Undo | separate `MeasurementHistory` stack | the table's own event history |

Ctrl+Shift+M already gives the seamless part: the row lands in the table being curated
and shows at once everywhere. Its defects are that it needs a detector run, a re-run
throws the rows away, and nothing lets the user leave hand-made rows out. The measurement
list has none of those defects but is not an event table.

## 2. Summary

1. **One kind of manual event**: an event row made by hand, `added=True`,
   `source="manual"`, always in an event table.
2. **One add door**, `add_manual_event(doc_id, channel, start, end)`, called by Ctrl+M now
   and by the cursor gestures of the separate cursor design later. It adds to the
   channel's **target table**, chosen by the user in the Table selector and remembered per
   channel. The target is named on the trace next to the cursors.
3. **The Manual table** (`detect.manual`): one per channel, always offered in the
   selector, for manual work with no detector or as a separate manual record. It never
   re-runs and is never replaced.
4. **Hand-made rows survive a re-run**, re-measured under the new run, with their labels,
   notes and rulings kept even when the new run detects the same event.
5. **The user chooses whether hand-made rows count**: a source filter in the gallery and
   inspector, an include switch on exports, figures, Distributions and fits, and **Copy
   hand-made events to…** between any two tables.
6. Old measurement lists **convert** into Manual tables; the user ticks the rows, and the
   originals stay.

## 3. The manual row

Every manual row carries the frozen 20 `SCHEMA` columns (`events.py:30`) plus extras:

| Column | Value |
|---|---|
| `event_id` | `max + 1` within the table |
| `recording`, `pore_id` | copied from the target table's existing rows; for an empty table `meta.path.name` and `meta.stem_key` (`analysis_panel.py:3123`). Not the in-memory `doc_id` today's empty-table fallback writes (`analysis_panel.py:4888`); a study table uses its record tag (`study_event_bridge.py:116`), and reconcile refuses mixed identity (`reconcile.py:543-551`) |
| `start_idx`, `end_idx` | the span, half-open, clipped to the loaded data |
| measured columns | `recharacterize_span(..., provenance=<target provenance>, prior=None)` |
| `accepted`, `flag` | from the rebuild: an automatic rule can still reject, for example a span clipped at the recording edge |
| `added` | `True` |
| `source` | `"manual"` (`SOURCE_MANUAL`) |
| `review` | `"reviewed"`: drawing it is a ruling |
| `orig_start_idx`, `orig_end_idx` | NA, so Revert stays off |
| `note` (new, string) | free text, user-editable; a converted measurement's `label` goes here |
| `origin` (new, string) | set on copies and carried rows: `"detect.manual:41"`, `"detect.schmitt:318"` |
| `legacy_measurement_ids` (new, string) | set on converted rows, for example `"12"`, or `"12,15"` after a merge |

This is the pair every hand-made row already has, so `source_disagrees_with_added`
(`events.py:1710`), the score's miss count (`scoring.py:443`), reconcile's "curator
evidence, not a vote" rule (`reconcile.py:343-347`) and the training export's `added`
column, "the detector missed it" (`scoring.py:545`), keep their meaning in an automatic
table.

"Hand-made" in this design means `source_kind == "manual"`, falling back to `added` only
when `source` is NA. A merge of a detected and a hand row is `added=True` but
`source="mixed"` (`merge.py:533`, `events.py:1612-1642`), and it counts as mixed, not
hand-made.

`SCHEMA` and `TRAINING_COLUMNS` do not grow (`test_events.py:84`,
`test_s3_clock_export_sidecar.py:187`). The new columns are additive extras: whole frames
go to parquet (`project.py:966-1015`) and re-measure carries extras
(`remeasure.py:932-947`). Merge keeps only the first fragment's value of a non-boolean
extra (`merge.py:532-540`), so it must join `note` values and concatenate `origin` and
`legacy_measurement_ids`.

Before restoring dtypes on a grown frame, the add casts non-`SCHEMA` extras to nullable
dtypes (`Int64`, `boolean`, `string`). Today the hand-add casts every column back to its
old dtype (`analysis_panel.py:4842-4843`), which raises on a native study table's int64
`study_*` columns and turns NaN into `True` in its bool ones (`study_event_bridge.py:150-157`).
The acid bridge works around this (`acid_event_bridge.py:144-156`); the native study bridge
does not, and no test covers it. A row added to a study table is also stamped with the
study review policy (`events.py:1386, 1413-1414`).

Rows are inserted in time order. Nothing depends on append order (`merge.py:379`,
`history.py:318`, `main_window.py:7984-8026`).

**Overlap.** A span overlapping an event of the same table is refused, as today
(`analysis_panel.py:4808-4820`). A wrong detected boundary is fixed by dragging that
event's mask in the inspector.

**Delete.** Event tables have no delete action anywhere today, and the overlap rule counts
rejected rows, so a wrongly drawn manual event that was rejected would block its span for
good. New: **Delete hand-made event**, for hand-made rows only, in the gallery, the
inspector and the Results table, one labelled undo step.

## 4. Adding

### 4.1 The door

```python
AnalysisPanel.add_manual_event(doc_id, channel, start, end) -> AddOutcome
```

Every gesture calls it: Ctrl+M, the Measure tab's Add button, and later the cursor
design's span drawing, edge snapping and nudges. It takes sample indices, not the Measure
region, which today's add reads internally (`analysis_panel.py:4782`).

- The helpers it needs, `_stored_detection`, `_active_signal` and `_sample_rate_staleness`,
  read `self._sel` (`analysis_panel.py:1862-1872, 4945-4955, 6174-6177`). They take the key
  explicitly, or the door refuses unless `(doc_id, channel)` is the current target.
- Every refusal carries a sentence. Today several return a bare `None` and the window
  shows a generic fallback (`analysis_panel.py:4783-4800, 4917`; `main_window.py:3574-3579`).
- New refusals: a recording with no sample rate ("the readout works, but an event needs a
  rate: declare one in File Info"); a channel that is not a current; a span outside the
  target table's searched span, for example an ML table run over the visible range
  (`analysis_panel.py:2765`, `ml_modules.py:648-651`), which would otherwise inflate its rate:
  "Outside the span Schmitt searched. Add it to Manual, or re-run over the whole recording";
  and no silent fallback, so a `detection_signal` error or a baseline that falls back to the
  flank median refuses with the reason (today it measures on raw,
  `analysis_panel.py:4974-4977`, `characterize.py:968-978`).
- The Ctrl+M handler keeps today's four gesture gates and their sentences: an open target,
  the channel shown, Measure on with a region, the region on this recording
  (`main_window.py:3534-3560`, pinned by `test_merge_gui.py:349-360`).
- The door owns the wait cursor, since `recharacterize_span` is unbounded.
- The Measure tab's Mark-missed button is routed through the same handler, and
  `markMissedRequested` with its ledger entry (`main_window.py:230`) retires.

### 4.2 The target table

The target is the channel's **chosen table**:

- The Table selector always offers **Manual** on a current channel, even before one
  exists. Choosing it shows the Manual table, creating it empty (5.5).
- The choice is remembered per (recording, channel) in `workspace.extra`. Today the shown
  table is one panel-wide id that defaults to Schmitt and is not saved
  (`analysis_panel.py:1135, 1795-1804`), so where Ctrl+M files would depend on side effects:
  a detector run, a switch of recording, a reopen.
- A detector run on a channel whose chosen table is Manual does not move the user. The
  status line says "Schmitt finished: 317 events. Show it". Today any detection write makes
  its table the shown one (`analysis_panel.py:4196-4216`).
- The target is named next to the Measure cursors on the trace ("→ Manual", "→ Schmitt"),
  on the Measure tab and in the status line after each add ("Added event 318 to Schmitt").
  Ctrl+M works with the tab hidden, so the tab alone is not enough.

| Chosen table | Ctrl+M |
|---|---|
| Schmitt, CUSUM or ML | adds a hand-made row to it |
| a study table | adds a hand-made row to it; the study round trip carries these (`study_event_bridge.py:136-138`) |
| Manual | adds to the Manual table |
| none on this channel | adds to the Manual table, creating it |
| Reconciled | refused: "Reconciled is derived. Show Schmitt or CUSUM and add it there". A source change makes Reconciled stale, and rebuilding it replaces its curation (`analysis_panel.py:2508-2513`) |

Ctrl+Shift+M retires, because two adjacent keys with different targets invite misfiled
events late in a long session. Decision 3 asks whether a key that always adds to Manual is
wanted.

### 4.3 Landing

After an add, the gallery and inspector, if already open, move to the new event without
moving the trace, each only when its own list holds it (`curation_view.landing_after_add`):
the inspector's filter, the gallery's Show filter with its class or representative sample.
A window whose list hides the new event (Detected only, Rejected only, Unreviewed, a label
filter, or Accepted only for an add stored rejected as `clipped_by_boundary`) keeps its
filter, its event and its selection, so the next A or R rules on the event the curator was
on. Each window decides in its own `land_on_added_event`, which emits nothing. Never
`select_event` or the `_open_*_at_event` helpers (`main_window.py:7727-7744`), which zoom
the trace and break the pinned rule that a hand-add leaves the view alone
(`test_ui_sweep_fixes.py:135-152`).

A window whose filter held no event (a Delete emptied it) keeps that filter through every
re-feed that keeps its place, and when the frame fills it again (the undo of that Delete,
the next add, the undo of a curation step) it shows the filter's first event, in its own
order (`curation_view.landing_after_refill`). While the filter is still empty, the window
says so. Owner's rule, 2026-10-29
(`tests/gui/test_curation_windows_keep_their_filter_through_adds_and_refills.py`).

Today any change in row count resets both windows to the first event with the filter on
All, because `same_events` is positional and exact-length (`curation_view.py:198-209`,
`event_inspector.py:2381-2392`, `event_gallery.py:2225-2292`). New
`curation_view.superset_by_id(old, new)`: old ids are a subset of new ids and each keeps
its `start_idx`. Both windows' keep path uses it, so filter, context, current event and
gallery selection survive an add.

### 4.4 Undo

The add writes the chosen table, which is the table on screen, so Ctrl+Z undoes it with no
cross-table rule (`main_window.py:13336-13344`). Redo stays per table: a Manual add leaves
Schmitt's redo alone, and an add to Schmitt ends Schmitt's own redo (`history.py:907`), the
same separation ML and Schmitt have today (`history.py:960-981`).

Tests: `test_redo_branch.py:120-153` (A2) keeps its cross-stack check by driving a legacy
row Remove, which is still a measurement step. `test_s3_dl_history_scope.py:191-241` and
the other history tests that seed rows through `add_current` re-seed with
`set_measurements_df`. New: an add to Schmitt is undone by Ctrl+Z; a Manual add spares
Schmitt's redo.

Labels: an empty table growing reads as a detector run in `describe_change`
(`history.py:385-388`). New rule: an empty table growing only by hand-made rows is "Add
event" or "Add *N* events". A `note` edit is "Edit note", since `_CURATION` is `accepted`,
`flag`, `added` (`history.py:177`).

## 5. The Manual table

### 5.1 Identity and registration

- Result key `(doc_id, channel, "detect.manual")`, display name "Manual".
- Not runnable: not in `ANALYSIS_REGISTRY` (`core/analysis.py:201`); storage, restore and
  selection do not need it (`session.py:984`, `main_window.py:12127`).
- Registered in `core/sources.py:160` `DETECTION_MODULE_IDS` (last), `core/sources.py:174`
  `DETECTOR_NAMES`, `events.py:904` `SOURCE_KIND_BY_MODULE` as `KIND_MANUAL`
  (`test_src_events_contract.py:195`) and `event_inspector.py:225`. `main_window.py:149` is
  a phrase map, so it gets "your manual events".
- Not a Reconcile input: `_reconcile_sources` (`main_window.py:3837`) and
  `can_reconcile_sources` (`reconcile_dialog.py:32`) skip it.
- `voltage_dep.py:156` is an allowlist, so Manual is excluded: it has no searched span,
  so no capture rate.

### 5.2 Predicates instead of literal checks

`CONSENSUS_MODULE_ID` is compared literally at 41 behavioural sites with five meanings.
They become predicates by meaning, each site mapped explicitly:

| Predicate | True for | Used for |
|---|---|---|
| `is_rerunnable` | schmitt, cusum, keras | Re-run, stale actions, form seeding |
| `geometry_locked` | reconciled | mask drag, merge, revert, amplitude edits |
| `mixed_chain` | reconciled | trace reconstruction and re-measure refusals |
| `reconcile_input_allowed` | schmitt, cusum, keras | Reconcile sources |

Two matter at once. `_on_rerun` (`analysis_panel.py:2947-3006`) sends any id it does not
know to `_run_detection`, which runs Schmitt and replaces the Schmitt table.
`_on_stale_action` (`2887-2945`) has a `_reference_redetection_needed` branch that would
launch a Schmitt detection from a Manual banner. Both must refuse Manual.

### 5.3 The measurement contract

`recharacterize_span` takes its basis from the table's provenance and falls back quietly
when a key is missing (`characterize.py:1005-1020, 152-159, 897-983`). So the Manual table's
provenance is fixed at creation and reused by every add, mask drag
(`event_inspector.py:3579`), merge and Rebuild:

| Key | Value |
|---|---|
| signal | the raw calibrated channel: no chain keys, so `detection_signal` returns raw (`modules.py:845-878`), which is what the gallery draws |
| `fs`, `units` | the live rate and channel units |
| `polarity`, `polarity_source` | `measured_polarity` over the whole channel (`baseline.py:4903`) |
| baseline block | the confirmed Detection-form `BaselineConfig.to_provenance(fs)` and its `baseline_fingerprint` (`baseline.py:1348-1371`), also on the `AnalysisResult` with the live `calibration_fingerprint` |
| `noise_rms_pA` | `noise_rms(trace − characterisation_baseline)` over the whole recording |
| `detected_at` | creation timestamp, the lineage token the refusal ledger reads (`main_window.py:7986`) |
| `open_pore_tau` | the open-pore autocorrelation time the edge fit's significance test needs (`docs/CURSOR_ERGONOMICS_DESIGN.md` 5.1), computed in the same worker |

The noise is computed inline in three places today (`detect.py:854-871`,
`modules.py:1575-1585`, `ml_modules.py:483-492`). One helper,
`measurement_noise_rms(signal, fs, cfg, polarity)`, replaces them, with
`on_nonfinite="interpolate"`, since `characterisation_baseline` refuses non-finite samples
by default (`baseline.py:5019`).

Deliberately absent: the analysed span, `analyzed_duration_s`, `min_dwell_s`,
`max_dwell_s` and every threshold. Nobody knows how much was searched by hand, so the table
has no event rate, and the Summary says so instead of calling `rate_hz`, which raises
(`events.py:1815`).

Manual numbers are measured on the raw trace, where a detector table is measured on its
low-passed one, about 2.6% in median `frac_depth` at the default filter
(`modules.py:813-815`). Copying into an automatic table re-measures under that table's
provenance (7.3), so a copy is comparable. Decision 6 asks whether the Manual table should
be able to use the Preprocess chain instead of raw.

### 5.4 Building the contract off the GUI thread

The contract needs a whole-recording baseline and noise, about 2 s on a 15 M-sample
recording (`baseline.py:4930`, about 0.3 s per 3 M samples), which a key press must not
pay. As soon as Manual becomes the target (chosen in the selector, or Measure turned on
over a channel with no table), a worker computes the pending contract and caches it under
(recording, channel, rate, calibration fingerprint, baseline fingerprint). Until it is
ready the Measure preview reads "Preparing the Manual table…". The table is created from
the same cached contract, so the preview and the stored row agree.

### 5.5 Creation and the empty table

- Measure the row against the contract first. Only if the row is good, install the empty
  table, then the row. The first checkpoint of a key is a silent baseline
  (`history.py:861-866`), so this makes the first add undoable, and a refused add leaves no
  table behind.
- Choosing Manual in the selector shows it at once and installs the empty table as soon as
  its contract is ready (5.4); that is the user's choice.
- Undoing the first add leaves an empty Manual table, because history has no "absent"
  state (`history.py:1000-1034`). An empty Manual table is left out of the workflow rail
  (`workflow_status.py:297-310`) and the close stake (`history.py:1192-1221`).

### 5.6 Rebuild Manual table

Re-measure cannot repair a Manual table. It refuses a table on another clock
(`analysis_panel.py:5295-5297`, `remeasure.py:1050-1075`), holds `t_start_s` and `dwell_s`
fixed (`remeasure.py:1130-1148`), replaces only the baseline block from the live form
(`analysis_panel.py:5303`) and never restamps the calibration. Every edit door refuses a
rate-stale table (`analysis_panel.py:4776-4781`), so without a repair a rate declaration
would lock the Manual table for good.

**Rebuild Manual table** runs off the GUI thread, recomputes the contract from live data,
and re-measures every row from its stored indices with `recharacterize_span(prior=row)`,
re-applying the extras. Indices and `event_id` are held; times move when the rate moved,
which is right for spans bracketed in samples. It is one undo step with its own label and
`_JOB_PROSE` entry (`worker_thread.py:290`), and the banner action for Manual on all three
staleness kinds: rate, calibration, baseline. The banner says "Rebuild", never "Re-run".

### 5.7 What else knows it is not a detector

- Selector: "Manual — 13 events", not "candidates" (`analysis_panel.py:1714`).
- Score panel blank: every row is `added`, so `score_detection` would show recall 0.
- Workflow rail: a channel with only a Manual table reads "Manual events".
- Wording: Revert reads "Drawn by hand: there are no detector bounds to return to"
  (`event_inspector.py:234`). The empty Manual table reads "No manual events yet. Bracket
  an event on the trace and press Ctrl+M", at `event_inspector.py:2387` and the gallery's
  `_ran_empty` branch (`event_gallery.py:4427-4436`). The stale sentence says "Rebuild".
  `_hint_hidden_results` (`main_window.py:12965`) and `judge_chain`
  (`conditioning_recipes.py:583`) get Manual wording.
- Study snapshot refuses a Manual table by module id: "Manual events are not a study
  detection; add them to the study table" (`study_event_bridge.py:250`).
- Export events sidecar: `detector: "detect.manual"`, the contract's keys, curation
  `by_kind: {"manual": n}`; missing keys are null already (`scoring.py:1146-1150`).
- Training export from Manual alone: allowed, with the dialog and sidecar saying no
  detector ran, so `added` means "drawn by hand", not "missed".
- Report figures: the dwell floor falls back to one sample period
  (`modules.py:2271-2281`); captioned "hand-selected events, no detection floor", with a
  warning before a censored dwell fit.

### 5.8 Closing a recording

Every result is dropped when its recording closes (`session.py:223`). Measurement rows are
not, so a user who works through recordings one at a time and saves keeps all of them
today. To keep that, a non-empty Manual table of a closed recording is kept in the project
the way records restore cannot install are kept (`_capture_unrestorable`,
`main_window.py:12315-12333`), written on save, and re-installed when the recording is
opened again. The close prompt says "Your 40 manual events are kept". Decision 5.

## 6. Hand-made rows survive a re-run

Today a re-run replaces the table, and the prompt warns that hand-added events are lost
(`analysis_panel.py:2427-2513`).

**Where.** One pure function in `analysis`, `carry_hand_rows(new_result, old_df, signal)
-> (EventSet, report)`, called at the two install sites before their single `add_result`:
`AnalysisPanel._on_controller_finished` (`analysis_panel.py:3966-3985`: Schmitt, CUSUM,
ML and the stale Re-run) and `MainWindow._on_batch_finished` (`main_window.py:3031-3051`).
The old table is still in the store at both. `Session.add_result` cannot host it, because
undo and restore install through it. The signal comes from an accessor addressed by
(recording, channel, provenance), not `_active_signal`, which reads the current selection,
which the user may have left during the run. The re-measure is a whole-recording pass per
table, so it runs in a follow-on worker job before install, with an install-time check that
the stored table is still the one it read, so a row hand-added during the run is not lost.

**What is carried.** Hand-made rows (`source_kind == "manual"`), re-measured with
`recharacterize_span(prior=row)` under the new provenance. That keeps a hand rejection and
non-derived flag tokens, labels included, and re-derives automatic tokens
(`characterize.py:851-875, 986-1002`). Amplitude choices are invalidated, as re-measure and
merge do (`remeasure.py:963-976`). Recording and pore come from the new table. Each carried
row gets `origin` set and a new `event_id`; the report lists old and new ids.

**Overlap with a new detection.** Matching uses reconcile's overlap coefficient of at
least 0.5 (`reconcile.py:182-192`), not any shared sample.

- A matched hand row and a viable detected row: the detected row stays (decision 1), and
  the hand row's note, labels, review and any hand rejection move onto it, the way
  reconcile inherits curation (`reconcile.py:869-881`).
- A matched hand row and an automatically rejected detected row: the hand row stays and
  the rejected candidate goes, since an auto-rejected row does not outvote a ruling
  (`reconcile.py:852-856`).
- Below 0.5: both stay, listed for the user to resolve.
- A hand row outside the new run's analysed span or inside its excluded regions
  (`detect.py:945-949`) is left out and listed, since it would count in a rate and as a miss
  for samples the detector never searched (`characterize.py:502-520`, `events.py:1815`).

**Undo and wording.** The carry-over goes in the same `add_result` as the new run, so a
re-run stays one undo step. A same-settings re-run is recognised as "Re-run detection"
only while the new table holds no hand curation (`history.py:560-575, 877-880`), so carried
rows would rename it "Curate 2 events" and silence its warning (`main_window.py:13510-13515`).
The new result is stamped with `carried_ids` and `dropped_ids`; the recorder treats a
replaced table carrying that stamp as a re-run, and `_curation_was_reset`, `_summarise`,
`hand_work_at_stake` and `_hand_curation` count only rows not carried. One status line says
what was lost and what was kept: "Re-ran Schmitt: 317 events. Kept 28 hand-made events; 3
are now detected (41, 77, 203), their labels moved onto them". Tests:
`test_undo_events.py:153-222`, `test_wa_undo_rerun_prompt.py:271-272`.

## 7. Choosing whether hand-made rows count

### 7.1 Source filter and mark

- One filter kind in `analysis/curation_view.py:158`, shared by both windows: Source
  manual, automatic, mixed. New combos are hooked by `CurationKeyGuard`
  (`curation_keys.py:66-73`).
- An "M" glyph in a bottom corner of a gallery tile (the top corners, border and caption are
  taken, `event_gallery.py:1086-1130`); a `source:` line and the `origin` in the inspector.

### 7.2 Include switch

"Include hand-made events", default on, which is today's behaviour, on:

- Export events and the training export, with new curation keys `include_added` and
  `n_added_excluded`. `n_withheld` keeps meaning rejected rows, which its note says
  (`scoring.py:880-885, 1015-1053`), and the training filter goes inside
  `_training_row_mask`, the one definition the writer and sidecar share
  (`scoring.py:585-631`);
- report figures: with the switch off the stored fit no longer matches, which today falls
  back to a fresh one-component fit (`main_window.py:10092-10104`); refit with the stored
  fit's own model order and say so in the caption;
- Distributions and fits: `include_added` on `dwell_fit_inputs`, `depth_fit_inputs`,
  `SelectionConfig` and the summary config and its validator, defaulting on for old recipes
  (`modules.py:2488-2596`, `plots_panel.py:1329-1333, 8385-8475`,
  `distribution_summary_panel.py:310, 625`).

The Summary counts, the rate, the score and the density figure always include them; the
switch is about what leaves the app. It is disabled on the Manual table, where every row is
hand-made.

### 7.3 Copy hand-made events to…

**Copy hand-made events to *table*…** copies hand-made rows from any table to any other
table of the same channel, including from an automatic table to Manual:

- re-measured under the target's provenance, so a copy is comparable with the target's
  detected rows;
- `origin` set on each copy; a row whose origin is already in the target is reported
  "already copied" and skipped, so running it twice copies nothing twice;
- only accepted rows by default; only rows inside the target's analysed span and outside
  its excluded regions;
- a row overlapping a target event is not copied; it is listed with a link, and a span
  overlapping a rejected candidate needs that candidate's mask dragged first;
- one undo step on the target (`history.py:381-386, 877-880`); the source is unchanged.

Copies are independent: a later edit on one side does not reach the other, and the dialog
says so. The Measure tab shows the channel's hand-made counts per table ("Hand-made here:
Schmitt 12, CUSUM 3, Manual 40"), so split work is visible.

### 7.4 Pooling consumers

- **Density figure**: one group per (recording, current channel), the table chosen through
  a public `AnalysisPanel.stored_to_show(doc_id, channel)`; groups keyed by persistent key,
  colliding labels disambiguated, and the detector named in the legend when groups use
  different ones. Today it takes every table and same-name groups collapse in a dict
  (`main_window.py:9682-9723`).
- **Distributions**: the Manual table is an ordinary source (`<key>@detect.manual`).
  Ticking it with a table that holds copies plots them twice; the source row counts copies
  exactly by `origin`.
- **Fits**: one slot per channel, unchanged; the lineage guard already refuses a fit under
  the wrong table (`modules.py:2801-2807, 2915-2925`).

### 7.5 Reconcile, later

Reconcile could take Manual as an input: it deduplicates by overlap, lets a manual boundary
win and keeps manual-only events (`reconcile.py:343-347, 857-863`). It stays out for now
because the Reconciled table is a frozen snapshot that any source change makes stale
(`analysis_panel.py:6211-6245`), coverage and rate drop when an input has no span
(`reconcile.py:1072-1077`), and a manual row over a hand-rejected automatic row is a
curation conflict on purpose (`reconcile.py:881-895`). The cost: a missed event found while
curating Reconciled is added to a source table, and rebuilding Reconciled then replaces its
curation. Decision 4.

## 8. The Measure tab

The tab becomes the cursor tool:

- the Measure toggle, the target, and an Add button;
- **a live preview of the row Ctrl+M will store**, measured under the target's provenance.
  It equals the stored row below `_LIVE_MAX_SAMPLES` (200,000 samples); above it the
  preview is strided and labelled approximate, and the stored row is exact, which is
  today's rule (`measure_panel.py:25-52`, pinned by `test_measure_latency.py:241-310`). The
  preview is fed the same conditioned array the add uses, pushed through the existing
  `set_detection_present` seam (`analysis_panel.py:4338`, `measure_panel.py:1451`), so the
  whole-recording baseline memo is shared rather than built twice;
- **an unstored statistics readout** under the preview, with the amplitude-method combo:
  mean, median, std, variance, min, max, ptp, rms, skewness, kurtosis, area. It is computed
  with `measure_selection` on the target's signal, with `baseline` set to the preview row's
  `baseline_pA` and `noise_rms` to the table's `noise_rms_pA`, so it is not a second set of
  amplitudes on a local flank baseline (`measure.py:676-682`). This keeps measuring without
  storing, for baseline and noise checks;
- no Raw or Processed combo: the basis is the target's, and the preview names it
  ("Measured on Schmitt's signal: 5 kHz Bessel");
- the purple trace overlay (`measure_panel.py:2864-2905`) shows the Manual table's rows when
  Manual is not the shown table. AnalysisPanel pushes the shown module id and the Manual
  frame from `_show_result`, `_clear_results`, `resultSwitched` and `resultsChanged` for
  `detect.manual`, which also covers undo and redo. Its toggle moves next to the preview;
- the old measurement list (section 10), while unconverted rows remain.

The Analysis menu tooltip for Measure event (`main_window.py:1489-1492`) changes to match.

## 9. Exports

- Events, training, levels and figures: today's doors on the shown table, plus the include
  switch.
- **Export event statistics…** runs the statistics of section 8 over the rows of one table,
  several tables or every Manual table in the project, and writes the 33-column measurement
  CSV (`MEASUREMENT_SCHEMA` plus `signal_basis`), with `signal_basis` naming the table's
  chain. It replaces the measurement list's Export CSV, works for automatic tables too, and
  gives back one pooled file across recordings.

## 10. Converting old measurement lists

1. **Offer.** A restore note in the info bar, the app's way of reporting restore outcomes
   (`main_window.py:12420-12493`): "This project has 42 manual measurements. Analysis,
   Convert measurements to Manual events…". A decline is kept in `workspace.extra`, as the
   sample-rate declaration is (`sample_rate_state.py:4`). Conversion runs from a queued slot
   after `projectRestored`, never inside `_finalize_restore`, where history is suspended
   and dirty marking is off (`main_window.py:12387-12396, 13718, 12526-12533`).
2. **The user ticks the rows.** The dialog lists each row with recording, channel, span
   and label, placeable rows ticked. A ticked row becomes an accepted, reviewed Manual
   event; ticking it is the ruling. The event model has no neutral state:
   `accepted=False` without an automatic flag reads as a hand rejection
   (`scoring.py:438-442`), so measurements that were baseline or noise checks must be
   unticked rather than converted.
3. **Placement**: grouped by `(doc_key, channel_index)`, placed only on an exact `doc_key`
   whose channel name at that index matches. The navigation fallbacks that guess
   (`measure_panel.py:2693-2730`) are not used.
4. **Measurement**: each converted row is measured as an event under the Manual contract,
   so its event numbers differ from the stored measurement numbers; the dialog says so and
   lists "numbers will change" for rows with a different calibration stamp or basis.
   Rate-stale rows convert too, re-timed from their indices (`measure.py:382-404`).
5. **Not placeable**, each with its reason: blank `doc_key`, `channel_index` −1, recording
   not open, channel-name mismatch, no rate on the live recording, indices outside the
   loaded data, a row overlapping another ticked row of the same channel.
6. **Record.** Which measurement became which Manual event is kept in a conversion record
   in `workspace.extra`, which a v11 build preserves (`project.py:212-216`). Not in a column
   of the measurement table: `coerce_measurements` drops extra columns
   (`measure.py:902-922`), so a column would not survive even this build's restore.
   Converted originals stay in the measurement table and are hidden from the plot source
   (`core/sources.py:1605-1615`), the restore note and the remaining count. Whether a row is
   converted is decided from the `legacy_measurement_ids` present in live Manual tables,
   so editing or deleting a converted event cannot hide its original.
7. **Undo.** The installs run with event and measurement history muted for the keys they
   touch and are recorded as one labelled baseline, so Ctrl+Z does not take a conversion
   apart. **Revert conversion** removes the converted Manual rows still present and
   updates the record. Copies made from converted rows stay, being independent.
8. **The old list until then**: shown collapsed as "Old measurements (12)", read-only except
   Remove, with its Export CSV. Saved Distributions recipes naming `manual-measurements`
   are remapped to the Manual sources of the converted rows, with a note when one maps to
   several.

## 11. Persistence

- No new manifest key; a Manual table is an ordinary `results[]` entry
  (`main_window.py:11152`, `project.py:1192`). The conversion record and the chosen table
  per channel live in `workspace.extra`.
- A v11 build installs an unknown module id but never lists it (`analysis_panel.py:1687-1697`),
  so it would hide the manual work while keeping it, and still count it on the rail and in
  the density figure. It strips nothing: the conversion record is in `workspace.extra`, so
  the project rule of bumping on a silent strip (`project.py:148-158`) does not force a
  bump. Decision 2 asks whether to bump anyway. If yes, it ships with the first build that
  writes Manual tables, and the version-pinning tests and the architecture doc move with it
  (`test_persist_invariant.py:529`, `test_recovery.py:113-118`,
  `test_project_workspace.py:198`, `test_docs_merge_and_format_agree_with_app.py:181-186`).

## 12. Tests that move

- Registration: `test_src_events_contract.py:190-209`, `test_wa_sources_per_detector.py`,
  `test_sources.py`, `test_workflow_status.py`, `test_reconcile.py:47, 550`.
- Ctrl+M, Ctrl+Shift+M and the add door: `test_analysis_launcher.py:837-840, 895-901`,
  `test_manual_scan_navigation.py:543-572`, `test_ui_sweep_fixes.py:118-175`,
  `test_p3_regressions.py:1010-1045`, `test_reconcile_workflow.py:286-340`,
  `test_merge_gui.py:349-360`, `test_acid_event_bridge.py:171-190`,
  `test_flag_invariant_gui.py:431-457`, `test_scoring.py:150-152, 255-265`,
  `test_wa_undo_row_count_edits.py:113-124`, `test_wire_analysis_panel_seams.py:198`,
  `tests/conftest.py:940-965`.
- Measure tab and preview: `test_measure_tab.py`, `test_measure_navigation.py`,
  `test_measure_latency.py:241-510`, `test_manual_measurement_edits.py:64`,
  `test_pc_measure_processed.py`, `test_wire_measure_basis_and_export.py`; about 15 more
  files seed rows through `add_current` (for example `test_s2_measure_cal_stamp.py`,
  `test_s2_measure_clock_column.py`, `test_s3_segment_clock_surfaces.py`,
  `test_close_guard.py`, `test_document_identity.py`) and move to a `set_measurements_df`
  fixture.
- Undo and re-run: `test_redo_branch.py:120-153`, `test_s3_dl_history_scope.py:191-266`,
  `test_undo_events.py:153-222`, `test_wa_undo_rerun_prompt.py:271-272`,
  `test_s8_data1_stale_table_refuses_in_place_edits.py`.
- Measurement contract: `test_pc2_measure_export_contract.py:269-327` and
  `test_s2_measure_stamp_schema.py` move to the statistics export.
- Guards: no direct `characterize` (`test_flag_invariant.py:401-424`), `provenance=` passed
  (`test_span_invariant.py:525`), no hand-built baseline windows
  (`test_baseline_seam.py:1023-1090`), every new signal classified
  (`test_surface_staleness.py:320-339`).
- New: Ctrl+M with no table creates Manual and is undoable; a refused first add leaves no
  table; Ctrl+M on Schmitt lands on the new event without moving the trace; the chosen
  table survives a detector run and a reopen; a native study hand-add; Re-run and the stale
  action refuse Manual; Rebuild re-times after a rate declaration; a re-run keeps hand rows,
  moves labels onto matched detections, stays one "Re-run detection" step; copy is
  idempotent by origin; delete of a hand-made event; include switch counts; density groups;
  conversion places, skips, records, survives a v11 round trip, and Ctrl+Z after it and
  Revert behave as stated.

## 13. Phases

1. **Manual work without regressions.** The door and its refusals; the chosen table, the
   selector's Manual entry and the trace label; landing; delete; the Manual table with its
   registration, predicates, off-thread contract and Rebuild; the empty-table rules; the
   re-run carry-over; the Measure tab with preview and statistics readout; Export event
   statistics; the old list read-only with Remove and Export CSV; the study fixes; the
   redo tests. Decision 2's bump and decision 5's carry on close, if chosen, ship here.
   The carry-over cannot wait: without it, Ctrl+M rows written into Schmitt in this phase
   would be lost at the next re-run, where today's measurement rows are not.
2. **Choosing whether hand-made rows count.** Copy between tables, the source filter and
   mark, the include switch, density grouping, place per table: remember filter, current
   event, gallery selection, sort and page per table and restore them on switching back
   (precedent `_remember_gallery_subpop_view`, `main_window.py:9169-9194`).
3. **Conversion** of old lists, and the inspector Statistics section, which shows the
   section 8 statistics for any browsed event.

The cursor ergonomics, the owner's tiring part, are a separate design that calls
`add_manual_event`. Phase 1 keeps today's preconditions: Measure on and a dragged region.
Decision 7.

## 14. Decisions for the owner

1. **Re-run overlap.** When a new detection matches a hand-made event, keep the detected
   bounds and move the hand row's labels, note and ruling onto it (proposed), or keep the
   hand-drawn bounds?
2. **Version 12.** Make older builds refuse projects with Manual tables, instead of opening
   them with the manual work hidden?
3. **A Manual key.** Is a second key that always adds to Manual, whatever is shown, wanted?
   Without it, Manual is reached through the selector.
4. **Reconcile.** Leave Manual out of Reconcile for now (proposed), accepting that adding to
   a source table costs the Reconciled table's curation on rebuild?
5. **Closing a recording.** Keep its Manual table in the project (proposed), as measurement
   rows are kept today?
6. **Basis.** Measure the Manual table on the raw trace only (proposed), or allow the
   Preprocess chain?
7. **Cursor work.** Ship the cursor ergonomics with phase 1, or after it?
8. **Stored non-events.** With the list gone, a span can be measured without storing
   (the readout) or stored as an event. Is a stored "measurement that is not an event"
   still needed?

## 15. What the reviews changed

| Draft | Change | Why |
|---|---|---|
| 1 | Ctrl+M always to Manual became Ctrl+M to the chosen table | a Manual add was invisible in the gallery and inspector the user was working in, and needed cross-table undo, which reopened the H-3 redo bug (`history.py:907, 960-981`) |
| 1 | live Combined table dropped | it would push undo steps per recompute, renumber events (`reconcile.py:964`), need read-only gates nobody has, and lose coverage (`reconcile.py:1072-1077`) |
| 1 | "Re-measure repairs" became Rebuild | Re-measure refuses rate changes and holds times (`remeasure.py:1050-1075, 1130-1148`) |
| 1 | Manual follows a reference became stand-alone | the copied key list was incomplete (`modules.py:674-700`) and went stale on re-run |
| 2 | "no table shown" target became a chosen table per channel | the shown table is one panel-wide unsaved id, so Manual was unreachable once a detector ran (`analysis_panel.py:1135, 1795-1804`) |
| 2 | drop-on-overlap became move-labels-on-match | dropping lost labels, notes and rulings that reconcile keeps (`reconcile.py:869-881`) |
| 2 | carry-over got an install site, a thread and history rules | it broke the "Re-run detection" label and warning (`history.py:560-575, 877-880`) |
| 2 | carry-over moved into phase 1 | Ctrl+M rows would otherwise be lost at the next re-run |
| 2 | landing uses quiet calls and a superset rule | `show_event` and `select_event` move the trace (`test_ui_sweep_fixes.py:135-152`) |
| 2 | `converted_to` column became a record in `workspace.extra` | `coerce_measurements` drops extra columns (`measure.py:902-922`) |
| 2 | conversion recorded with history muted | its installs landed on Ctrl+Z (`main_window.py:2309, 13474-13485`) |
| 2 | statistics readout and export moved into phase 1; Delete added; refusals for no rate, non-current channels, out-of-span | regressions against today's measurement list and Ctrl+Shift+M |

## 16. Implementation status (2026-10-29)

Built, with tests (`tests/analysis/test_manual_events.py`, `tests/gui/test_manual_event_door.py`,
`tests/gui/test_measure_tab_preview.py`): the `detect.manual` table and its contract
(`analysis/manual_events.py`), the one add door `AnalysisPanel.add_manual_event` and its refusals,
the chosen table per channel (saved under `event_table_choices_v1`), Ctrl+M and the Add button on
that door, Delete, the carry-over of hand-made rows through a re-run (single run and batch), Rebuild
for a stale Manual table, Copy hand-made events to…, the include switch, Export event statistics…,
the source filters, the gallery mark and the inspector line, the Measure tab preview and the
unstored statistics against the table's baseline and noise, and the conversion of old measurements.

Different from this draft:

- **The contract and the carry-over run on the GUI thread.** No prewarm job and no "Preparing the
  Manual table…" state. The contract (polarity vote, whole-recording baseline, noise, open-pore
  tau) is built once per recording, channel, rate, calibration and estimator, under a wait cursor,
  the first time Measure is turned on or an event is added: 0.4 s at 750 k samples and about 1 s at
  3 M (2.8 s at 6 M of pure noise, where the polarity vote dominates). The carry-over measures each
  hand-made row once and costs 7 ms per row.
- **Above 200,000 samples the preview names the span too long and shows no row**, rather than a
  strided, approximate row. The statistics are still shown (strided and labelled, as today) and the
  stored row is always exact.
- **The Raw / Processed choice is hidden, not deleted.** `MeasurePanel.set_basis_provider(None)`
  brings it back for a panel with no owner, which is how its own tests reach it.
- **The old list is not collapsed.** It is a group, "Old measurements (N)", hidden while empty.
- **Fixed after the 2026-10-29 bug sweep** (`tests/gui/test_s9_*.py`): the chosen table follows the
  table on screen (every install writes it, for a table the channel holds), so the add always writes
  the table on screen (4.4) and a project reopens on it; a write to another channel of the recording
  on screen leaves the channel shown alone (`Session.announcing`); Export event statistics refuses or leaves out a table on an old calibration;
  a re-run joins the notes of several hand-made rows that land on one detection
  (`manual_events.join_notes`); the training sidecar counts only the rows the include switch kept;
  the edge fit's tau cache is keyed on the run; Delete keeps the curation windows' filter; and the
  preview stands on a span the unstored statistics refuse.
- **Not built:** the purple overlay of the Manual rows when another table is shown; decision 5
  (the Manual table is discarded with its recording through the existing prompt); the restore-note
  offer, the "Revert conversion" action and the remapping of saved Distributions recipes (conversion
  works on the selected recording and channel, is one undo step, and a converted original is told
  from an unconverted one by the `legacy_measurement_ids` of the live Manual rows); the "hand-selected
  events, no detection floor" caption on the dwell histogram; the include switch governs the events
  and training exports only, not fits, Distributions or report figures.
- **Not tried:** a real Mac, a trackpad, or any real recording.
