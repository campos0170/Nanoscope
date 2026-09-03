# Stage 1 persistence contract

Stage 1 protects the state NanoScope can already restore. `MainWindow` is the only project
writer; the Gallery and Event Inspector expose save controls and emit save intent back to it.
The table below is the contract used by the Stage 1 regression suite.

| Mutable state | Dirty-state route | `ProjectState` field | Restore route | Regression coverage |
| --- | --- | --- | --- | --- |
| Recording membership in an existing project | `Session.documentOpened` / `documentClosed` → `_mark_dirty_for_document_set` | `documents` | `_begin_restore` → normal open queue | `test_close_guard.py`, `test_project_io.py` |
| Derived recording segments | explicit `_mark_dirty` after creation; document-close safeguards | `segments` | `_rederive_segments` then `_apply_restore_for_doc` | `test_segmentation_gui.py`, `test_dataloss_regressions.py` |
| Channel calibration and source identity | `Session.calibrationChanged` → `_mark_dirty` | `calibrations` | `_apply_restore_for_doc` with channel/source identity checks | `test_project_io.py`, `test_adversarial_project_io.py`, `test_restore_carry_and_identity.py` |
| Layer inclusion, visibility, time alignment, and chosen ink | user handlers emit `FileDock.viewStateEdited` → `_mark_dirty`; opening/navigation remains neutral | `layers` plus `colours` | colour registry is installed before layers; `_apply_restore_for_doc` restores the trace ink without rewriting the independent plot-colour registry | `test_view_state_dirty.py`, `test_project_io.py`, `test_layer_ink.py`, `test_colour_identity.py` |
| Overlay versus stacked layout | `Session.viewModeChanged` → `_mark_dirty`; restore is suppressed by `_restoring` | `view_mode` | `_finalize_restore` | `test_view_mode_roundtrip.py`, `test_project_save_surfaces.py` |
| Detection tables, detector provenance, source/status, reconciled rows, manual additions, curation, flags, labels, and edited bounds/measurements | `Session.resultsChanged`, table edit signals, Gallery/Inspector batch and label signals, and `EventHistory.changed` → `_mark_dirty` | `results` | `_apply_restore_for_doc` rebuilds each `EventSet` without rerunning detection | `test_project_io.py`, `test_undo_events.py`, `test_wa_table_batch_curation.py`, `test_project_save_surfaces.py` |
| Derived dwell/depth/level fits and fit provenance | `Session.resultsChanged` → `_mark_dirty` | `fits` | `_apply_restore_for_doc` rebuilds tables or stored fit curves | `test_project_io.py`, `test_level_fit_reconcile.py` |
| Manual measurement rows and annotations | `measurementsChanged` and `MeasurementHistory.changed` → `_mark_dirty` | `measurements` | `_finalize_restore` replaces the table, including with an empty table | `test_measure_navigation.py`, `test_measure_tab.py`, `test_project_io.py` |
| Project label vocabulary | `_edit_labels` compares old/new vocabulary then calls `_mark_dirty` | `label_vocabulary` | `_finalize_restore` then `_push_vocabulary` to both child windows | `test_analysis_launcher.py`, `test_project_io.py` |
| Records that could not be attached during project open | incomplete restore remains dirty; `_capture_unrestorable` retains records | merged back into their original list fields | subsequent restore retries the normal routes | `test_restore_carry_and_identity.py`, `test_dataloss_regressions.py` |

## Save-state authority

- `_visible_project_dirty()` is the one user-facing unsaved predicate.
- `_update_title()` publishes the current path and that predicate through
  `projectSaveStateChanged` to every open editing window.
- Gallery and Inspector never gather `ProjectState`, choose a final path, update recent files,
  manage the disk stamp, or call `core.project.save_project`.
- A project revision is identified by its bytes as well as filesystem metadata. The final
  compare and atomic replacement share an inter-process lock, and a conflict approval applies
  to one exact revision; first-save collisions and a third writer require a fresh confirmation.
- Project open reads and stamps one lock-protected revision, so a save landing between those two
  operations cannot be mistaken for the version in memory.
- A cancelled or failed save leaves the path and dirty state unchanged.
- A deliberately incomplete save (for example, an unrestorable segment whose parent is closed)
  may write a file but remains visibly unsaved.

## Explicitly outside Stage 1

Plot/PCA/GMM window recipes, crash-recovery snapshots, portable raw-data bundles, immutable event
UUIDs, reviewer audit history, and label-package import belong to later stages. Preprocessing
protocols, ML presets, and plot templates remain application preferences in `QSettings`, not
resumable project analysis.
