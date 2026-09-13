# Release verification — 2026-09-13

Apple Silicon NanoScope 0.1.0 was rebuilt from `05ce8ed19e36` (HEAD of
`codex/beta-sweep-fixes`), which contains today's beta-sweep bug-fix batch (six
confirmed-open bugs fixed at root cause: zero open-pore current, stale baseline-health
gating, degenerate HDF5 time columns, missing sample-rate validation in
`recharacterize_span`, an I-V-window GUI-thread stall on cold large channels, and
non-ASCII/underscore numeric-entry parsing), ten stale architectural-guard-test fixes,
and a CI mitigation (GUI test suite split into 16 shards) for an intermittent PySide6
signal-manager crash whose root cause was isolated but is not fixable from NanoScope's
own code (see [Crash root-cause finding](../CHANGELOG.md) in the Unreleased/Validation
section). The display version remains 0.1.0; use the executable hash below to
distinguish this build from earlier 0.1.0 builds.

## Outputs

- Local app: `dist/NanoScope.app`
- Local DMG: `dist/NanoScope-0.1.0-arm64.dmg`
- DMG size: 366,539,496 bytes
- DMG SHA-256: `19192027a6d15be8deea3799fd2e55ebfdbab5ce04fefa91917cfd7e2c153fc8`
- App executable SHA-256: `c3d6c54da6d2057112ace98fabe2b213f28cb09b2204c9e474bf391204bfd57a`

Only the local DMG was requested for this release. Generated binaries, recordings and
models are not part of the source commit.

## Build and checks

- Clean `packaging/build_app.sh` build using the existing Python 3.11.7/arm64
  environment. No dependency upgrades or scientific algorithm changes were made
  during packaging.
- Bundle sanitization and strict, deep ad-hoc signature verification passed.
- The standard `hdiutil create` DMG path succeeded directly this time; the hybrid/
  no-mount fallback documented in the 2026-09-11 release (and in `build_app.sh`) was
  not needed. `hdiutil verify` reports a valid DMG checksum.
- 148 frozen modules matched current source bytecode exactly, normalizing only
  `co_filename`: all 147 `nanoscope`/`nanoscope_selftest` entries in the bundle's PYZ
  archive, plus the `nanoscope_app` entry-point script embedded directly in the
  executable. Zero mismatches, zero modules without a resolvable source file.
- 37 packaging tests passed (`tests/packaging`).
- The rebuilt app's self-test exited 0: **17 applicable checks passed, 3 skipped**
  (20/20 in the tool's own summary, which counts skips as passed). Passing runtime
  checks include all reader registrations, synthetic HDF5 and Parquet roundtrips,
  detection on a worker thread, CPU-only TensorFlow import, the google3 protobuf stub
  surviving freezing, full `MainWindow` construction/loading, distributions/
  event-density/PSD, CSV exports, and Seaborn comparison plots with frozen-snapshot
  restoration and twelve PNG/PDF/SVG exports.
- External-recording loading, Keras model prediction and ML-controller detection with
  a supplied model were skipped because no external data/model arguments were
  supplied.
- Embedded build metadata (`_build_info.json`) confirms `git_commit` matches this
  build's HEAD exactly. `git_dirty` reads `true`, but the only modified file at build
  time was `packaging/resources/NanoScope.icns`, regenerated with different byte
  content (203,662 → 258,752 bytes) as a normal side effect of the build script's own
  icon-generation step — not uncommitted application code.

Preferences, recovery files and synthetic test artifacts were isolated from user data.
Ad-hoc signing is not Apple Developer ID notarization; `spctl -a -vv` correctly
rejected the app for that reason. Build checks demonstrate packaging and execution,
not calibration validity or detector accuracy on difficult recordings.

Earlier source validation is recorded in
[Release verification — 2026-09-11](RELEASE_VERIFICATION_2026-09-11.md),
[Publication upgrade validation](PUBLICATION_UPGRADE_VALIDATION.md) and
[Learnability upgrade validation](LEARNABILITY_UPGRADE_VALIDATION.md). Installation on
an individual Mac is a separate operation; the presence of this report does not prove
that its `/Applications` copy has been replaced.
