# Sweep-7 fix batch — carry-over list (noticed by the fixers, deliberately NOT fixed in this batch)

Collected from docs/sweep7/fixes/reports/*.md on 2026-09-01. Each is a candidate for the next sweep's known-open list.

## From A1 (Plots window)
- A template recall whose config equals the on-screen state opens a new origin but pushes no undo step (pre-existing), so Undo after it steps past the import rather than back onto it; pinned as such in the rewritten losses test.

## From A2 (exported figures)
- main_window's Export figure… loop: a figure builder that raises is dropped with only the status-line count changing ("Wrote 3 figure(s)") — no dialog names the missing figure or its error. (EXP3-04 covered the 4-of-4 case; the partial case is still silent.)

## From B (gallery / inspector)
- `EventTable.select_event` / `AnalysisPanel.select_event` emit nothing when the row is already current, so the gallery door is not re-fired (POP2 repro case 1 never reaches the gallery).
- Inbound `show_event` on an in-subset tile replaces a held multi-selection with that tile (pre-existing mirror semantics).
- "Reject selected" is still enabled with an empty selection (keyed to tile count) — harmless now that the selection survives.
- Pre-existing unused imports in event_gallery.py / event_inspector.py.

## From D (calibration / rail)
- CAL5: reopening a moved recording at its new path still says nothing (no sidecar exists there to speak from).
- PSD / I-V windows resolve channels by header kind (S7-CAL2 section A); `Session.effective_kind` / `resolve_current_channel` are the drop-in.

## From E (detection provenance)
- `provenance['merged']` is overwritten by each successive merge, so the report's "folded X into Y" describes the LAST merge only.
- `EstimatorComparison.warnings` (compare dialog) shows the baseline's own warnings only — by design.

## From F (notch / spectra / docs)
- `docs/inventory.json` has no rows for the Preprocess ▸ Notch "use an explicit line list instead" checkbox or its "lines" box.
- `tests/gui/test_baseline_panel.py:~511` docstring still quotes the stale 233 ms probe figure.

## Refuted / unadjudicated in sweep 7 (for the record)
- S7-VDEP1, S7-VDEP2, S7-JRN5, S7-JRN6: REFUTED by their skeptics (repro scripts remain under docs/sweep7/repros and still print REPRODUCED — they test claims the skeptics showed to be disclosed already).
- S7-FIX4: confirmed but rejected — the proposed remedy would have made the histogram page say something worse.
- S7-JRN7: filed, never adjudicated; docs-only, fixed by agent F in this batch.
