# Learnability upgrade validation — 2026-09-11

This report records pre-release source validation; no app build, installation, DMG replacement,
commit or push was performed during that implementation phase. The subsequent rebuild is
recorded in [Release verification](RELEASE_VERIFICATION_2026-09-11.md). Existing publication-export
work in the worktree was preserved.

## Changes

- Shared canonical names across the launcher, menu, Explorer and eight plot-window titles.
- Persistent, searchable Analysis home: all eight analytical pages plus the guide, gallery,
  report figures and Figure Library. Browsing and refreshing availability do not instantiate
  analytical pages or run their refreshers. Analyze now opens this home before detection.
- Seven-step modeless first-analysis guide under Help and Analysis. Reading, navigation and
  checkmarks do not change scientific settings. Opening a live analytical tool can compute
  that view through its existing workflow; the guide explicitly discloses this.
- Optional deterministic synthetic practice CSV, created only through an explicit action.
  It has a required SYNTHETIC- filename prefix, normalized-output overwrite confirmation and
  protection against replacing any currently open recording. CSV reading still requires
  users to declare/verify physical units; no calibration is applied automatically.
- Essential/Advanced Inspector visibility. Advanced remains the default. Active filters,
  critical warnings and retained settings stay disclosed. No numerical presets are applied.
- Status distinguishes acceptance from explicit review, flags unknown review ledgers and
  conflicts, and does not equate changed gain settings with physically verified calibration.

## Scientific boundary

No detector, baseline, amplitude, fitting or calibration algorithms were edited in this
learnability change. Existing scientific action routes and per-tool gates are retained.
Opening an empty Subpopulations view does not run clustering; generating/refreshing does.
Guide checkmarks do not enter the scientific review ledger. Pure control-level changes do
not alter scientific parameters, emit analysis signals or modify results/undo history.

The existing application saves Inspector tab selection as workspace view metadata. Explicit
guide actions that change tabs can therefore mark a project unsaved. A traced check verified
that these routes leave detection parameters and results unchanged. This behavior is not
suppressed or falsely described as an entirely immutable project.

## Verification

Tests used `/opt/anaconda3/bin/python`, `QT_QPA_PLATFORM=offscreen` and the repository's
preference/recovery-isolating pytest harness. Groups below overlap; do not sum their counts
as unique tests.

- 187 scientific regression tests passed: detector kernels, baseline, Savitzky–Golay baseline,
  event amplitude/lifecycle and core calibration.
- 79 plotting/export/curation regressions passed: shared plot workflow, overwrite protection,
  amplitude curation and amplitude columns.
- 19 guide/Essential-control tests passed: navigation isolation, gating, CSV roundtrip and
  known generator shape, safe file creation, existing-loader routing, project-save routing,
  no implicit mixture run, control visibility/preference and scientific-state invariance.
- 42 home/Explorer/workflow-rail integration checks passed.
- All 58 launcher checks passed across three disjoint fresh-process groups.
- 55 workflow status/rail and calibration/override checks passed.
- 83 canonical-window-title/Subpopulations-view checks passed.
- 45 reconciliation, Subpopulations recovery/disclosure and Inspector redesign checks passed
  after canonical-copy expectations and the legacy optional-column assertion were corrected.
- Legacy panel/tab tests: 103 passed. An additional panel/Inspector group initially passed
  124 with one outdated test assertion: it expected optional amplitude columns in a legacy
  detector frame that does not contain them. The assertion now checks scientific ordering
  of columns actually present, without adding data or changing the table implementation.

Visual checks inspected the integrated home, seven-step guide at normal and 600×440 minimum
sizes, and Essential Inspector. Independent panel checks covered 360/420-pixel widths. Long
guide content scrolls; primary navigation remains available. This is developer visual and
regression testing, not a formal usability study with newcomers.

## Environment caveats

One earlier combined launcher process crashed during Qt/RecoveryController initialization
before reporting test progress. Fresh disjoint launcher groups passed. Similar native Qt
test-lifetime instability was recorded in the prior publication upgrade; no root cause or
fix is claimed here. Non-fatal joblib core-count fallback, font-cache and intentionally
overflowing calibration-test warnings were observed. No real recording or installed app
was modified by validation.
