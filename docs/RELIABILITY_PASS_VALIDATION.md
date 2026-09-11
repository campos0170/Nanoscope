# Reliability and responsiveness pass — 2026-09-11

## Scope

This pass implements safety, responsiveness, release identification and newcomer
workflow improvements on top of the opt-in recording-specific open-pore reference.
It does not introduce a detector, change valid scientific defaults, rebuild an app,
install a release, or push a commit. Figure Library expansion, adaptive-noise
detection and broader scientific validation remain separate work.

## Behavior changes

- Saved active-reference metadata without its recipe, contradictory recipes and
  malformed reference restores remain readable but block analysis until an explicit
  user decision. Ordinary saved controls are restored separately from an invalid
  reference draft. Existing event measurements and curation are preserved.
- A saved low-pass filter that cannot be replayed now raises instead of silently
  remeasuring unfiltered samples or substituting an arbitrary order. Valid omitted
  or null legacy defaults remain supported; Gaussian's unused order is unchanged.
  Failed Inspector boundary edits leave bounds, amplitudes and undo history intact.
- Initial sub-state fitting uses the existing worker/controller. It freezes inputs,
  supports cancellation and checks load, calibration, rate, source-result identity,
  event-table content, parameters and provenance before installing its outcome.
  The numerical fitting functions are unchanged.
- Batch jobs retain frozen raw/calibration descriptors and materialize one calibrated
  recording at a time. Samples are shared by the existing immutable-load contract;
  the batch does not copy every raw recording. Existing batch calibration-staleness
  policy is unchanged: a correctly stamped old-calibration result is marked stale.
- Reference preview uses the exact conditioned current and baseline produced by
  its detector run. Capture is context-local, output-only, and never persisted.
  The preview does not install its temporary event table.
- Rapid baseline display edits are coalesced for 200 ms and marked as updating.
  One display-conditioning cache is keyed by source, calibration and signal chain.
  Display residuals call the unchanged subtraction function in bounded chunks;
  a 64 MiB LRU reuses residuals across pane rebuilds and invalidates on explicit
  baseline replacement, calibration change or layer removal. This is a cache cap,
  not a cap on all visible plots or total process memory.
- Help → About identifies the running build and offers an explicit local-only
  diagnostics copy. Future builds embed time, source digest, Git state and an exact
  dependency inventory. See `packaging/RELEASE-METADATA.md` for format and limitations.
- The guide explains Automatic, known-current and selected-region references,
  preview/redetection, units and limitations. First-use Essential/Advanced is an
  appearance choice, not a change to analysis settings. Workflow stages are keyboard
  accessible. Existing user preferences are retained.

## Numerical and workflow evidence

The frozen difficult-recording benchmark was rerun after the analysis-source changes:
**72/72 runs passed exact comparison**, across 36 evaluation recordings and both
Schmitt and CUSUM default profiles. All 12,173 event rows and all 24 exported columns
matched; baseline fingerprints, noise and polarity matched too. The source did not
change during this run. Report in the working workspace:
`work/reliability-pass/automatic-regression.json`.

Focused tests cover capture/no-capture equality on both engines, positive and negative
current, filtered/unfiltered traces, automatic/supplied references, nested/thread
capture isolation, exact chunked subtraction, cache invalidation and bounded temporary
allocation. Background-fit tests check Qt responsiveness, exact old numerical results,
cancellation and eight stale-source scenarios. Batch tests verify lazy materialization,
frozen nested settings, damage disclosure and derived-array release.

The existing worker/batch/lifetime and level-fit/integration/export suites passed
**171 tests**, including the new performance tests. The baseline-view/reference-preview
integration group passed **50 tests**. These groups overlap other checks and must not
be summed into a unique test total.

A single local 3,000,000-sample float32 residual microbenchmark measured traced peak
allocation of **80.1 MiB** for the existing full-array call and **19.0 MiB** for bounded
drawing; a cache hit allocated less than 1 KiB. The output is 11.4 MiB in both cases
and exactly equal. This measures Python/NumPy traced allocation for this operation,
not application RSS or a universal speed guarantee.

Push/PR CI now includes curated scientific, reference safety and workflow tests.
The exact configured scientific CI selection passed locally: **685 tests**, with
two existing warnings. The focused release/onboarding group passed **65 tests**,
and the dedicated final restore/panel/migration group passed **115 tests**.
These selections overlap; the counts are reported separately.
Full regression shards remain manually available. Historical scientific hash pins
were not rewritten; this is not a claim that every historical test in every platform
passes. No bundled-app smoke test was run because no build was requested in this pass.

## Remaining release decision

The existing build environment contains TensorFlow macOS 2.15.0 alongside Keras
3.12.0, outside TensorFlow's declared Keras requirement. Its inventory records this
inconsistency; it is not advertised as a solver-valid dependency lock. No packages
were upgraded or downgraded. Resolve and validate supported ML models separately
before claiming a clean reproducible ML environment.
