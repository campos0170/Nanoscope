# Release verification — 2026-09-11

Apple Silicon NanoScope 0.1.0 was rebuilt from the publication-export and learnability
changes in this commit. The display version remains 0.1.0; use the executable hash below
to distinguish this build from earlier 0.1.0 builds.

## Outputs

- Local app: `dist/NanoScope.app`
- Local DMG: `dist/NanoScope-0.1.0-arm64.dmg`
- DMG size: 347,322,843 bytes
- DMG SHA-256: `9b15b75febbec39a6e3d90ce6714d8993419769de2646ce0dd57e2f0bb41a4bb`
- App executable SHA-256: `35e8439ff62a2d95bb706ee9a219921ca968a6aa2524b183afd5941aec7a7e63`

Only the local DMG was requested for this release. Generated binaries, recordings and
models are not part of the source commit.

## Build and checks

- Clean `packaging/build_app.sh` build using the existing Python 3.11.7/arm64 environment.
  No dependency upgrades or scientific algorithm changes were made during packaging.
- Bundle sanitization and strict, deep ad-hoc signature verification passed.
- macOS rejected the standard `hdiutil create` path; the build's existing no-mount hybrid
  fallback succeeded. `hdiutil verify` reports a valid DMG checksum.
- 132 frozen source modules matched current source bytecode, normalizing only `co_filename`.
  This includes 131 NanoScope modules and the packaged self-test, with explicit checks for
  the new guide/home/catalog, publication-preflight and source-style modules and the
  Seaborn/Agg/PDF/SVG dependencies.
- 27 packaging tests passed.
- The rebuilt app's self-test exited 0: **16 applicable checks passed, 3 skipped**. The
  built-in summary prints 19/19 because skipped checks count toward that summary; it does
  not mean an external recording or a trained model was tested.
- Passing runtime checks include all reader registrations, synthetic HDF5 and Parquet
  roundtrips, detection on a worker thread, CPU-only TensorFlow import, full MainWindow
  construction/loading, distributions/event-density/PSD, CSV exports and Seaborn comparison
  plots with frozen snapshot restoration and twelve PNG/PDF/SVG exports.
- External-recording loading, Keras model prediction and ML-controller detection with a
  supplied model were skipped because no external data/model arguments were supplied.

Preferences, recovery files and synthetic test artifacts were isolated from user data.
Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging
and execution, not calibration validity or detector accuracy on difficult recordings.

Earlier source validation is recorded in [Publication upgrade validation](PUBLICATION_UPGRADE_VALIDATION.md)
and [Learnability upgrade validation](LEARNABILITY_UPGRADE_VALIDATION.md). Installation on an
individual Mac is a separate operation; the presence of this report does not prove that its
`/Applications` copy has been replaced.
