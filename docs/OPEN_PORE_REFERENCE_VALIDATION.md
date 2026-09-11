# User open-pore reference validation — 2026-09-11

## Scope

Opt-in per-recording/channel signed current or selected-interval reference for
Schmitt/CUSUM. No change to detector thresholds, detection kernels, global residual
noise estimation, calibration, filtering defaults or Automatic baseline algorithms.
New reference choices intentionally change analytical results when explicitly run.
This is not an adaptive-noise implementation or automatic open-pore identification.

## Automatic numerical regression

Replayed all 36 evaluation-a/evaluation-b recordings from the existing synthetic
difficult-recording benchmark with both default detector profiles (72 runs).
Compared against the previously generated native event CSVs using round-trip float
parsing: **12,173 rows and all 24 columns matched exactly**, including numerical
values, event boundaries, accepted states and flags. Baseline fingerprints, noise
estimates and polarity also matched. Equality treats corresponding NaNs as equal.

The independent runner/report are in the task workspace:
`work/open-pore-reference-validation/automatic-regression-final-report.json`.
The final repeat records source hashes before/after: all analysis source files
were unchanged throughout the check.

The core implementation was also compared directly against unmodified Git HEAD
`3775a559391aec20c63e9973a40ec3c8606807a5`: four legacy configurations produced
identical baseline arrays, complete provenance and warnings.

## Regression suites

- 180 existing GUI tests passed across baseline panel/migration/correction,
  staleness, polarity readout, normalization/protocols, ML baseline plumbing,
  batch detection, overlays and per-recording state.
- 277 existing analysis tests passed across Schmitt/CUSUM, characterization,
  dual-signal detection, remeasurement/curation, scoring/export disclosure,
  baseline occupancy/cache and nonfinite checks.
- 19 existing event-amplitude curation tests passed.
- The core agent's existing baseline seam/masked/probe run passed 174 tests.
  One existing test failed solely because its pinned MD5 for `ml_preprocess.py`
  was stale: expected `4eb0ef79…`, while both Git HEAD and the unchanged working
  file were `48e364c4264dcef2c25a2fadad9bc4a1`. The kernel and hash-pinning test
  were not edited to hide this failure.

All **193 new focused tests passed**: 57 reference-core, 54 production integration/export,
32 reference-panel, 22 end-to-end workflow, 10 amplitude-reference safety, and
18 preview/decimation tests. These cover reference construction/validation, signed inputs, selected
constant/linear references, production detector equivalence to the same explicit
baseline array, residual noise/polarity, project round-trip/replay, unsupported
domains and workflows, and fail-closed curation reconstruction. GUI checks cover
the recording-scoped editor, preview without result installation and fresh-detection
requirements. See the `test_open_pore_reference*` and `test_open_pore_preview.py`
files for executable cases. Visual QA inspected the reference editor at 400 px wide
and the modeless preview with its Details section both collapsed and expanded.

## Limits of the evidence

Exact Automatic compatibility is not proof that an arbitrary supplied reference is
correct or that difficult events become recoverable. User-provided reference evidence
can be wrong; fixed and interpolated references have explicit assumptions. Low SNR,
bandwidth limits, overlap and changing noise remain separate detection challenges.

Validation concerns the source implementation. No application bundle, installation,
DMG or GitHub publication is included in this change.
