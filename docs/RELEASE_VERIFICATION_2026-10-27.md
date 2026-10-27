# Release verification — 2026-10-27

Apple Silicon NanoScope 0.1.0 was rebuilt from `132b42bdb808` (HEAD of
`codex/beta-sweep-fixes`), which contains the CSV time-unit fix and the sample-rate /
time-unit declaration: a time column written in milliseconds loaded silently at a rate
1,000× too low (10 µs steps written as `0.01` per row read as 100 Hz, `Provenance.MEASURED`,
`issues=()`); the reader now reads a unit the column's header declares, asks about an
undeclared column that reads below 1 kHz, keeps the interval of a column the 1 Hz floor
refuses, and **Inspector ▸ File Info ▸ Sample rate / time unit…** lets the user declare the
time base, saved with the project and followed by segments (see the Unreleased section of
[CHANGELOG.md](../CHANGELOG.md)). The display version remains 0.1.0; use the executable hash
below to distinguish this build from earlier 0.1.0 builds.

This build also carries **one dependency change in the build environment**, made
deliberately and recorded here because nothing in the source tree records it:
`/opt/anaconda3` had SciPy 1.15.1 (installed 2025-02-09), which `pyproject.toml` excludes
(`scipy>=1.11,!=1.15.0,!=1.15.1`) for a known one-dimensional median-filter defect, and
which every earlier 0.1.0 DMG built from that interpreter therefore bundled. The bundle's
own self-test asserts the bundled SciPy is not one of those versions, and the first rebuild
of the day failed exactly that check (21/22). SciPy was moved to **1.15.3** — the same minor
series, so numpy 1.26.4, pandas 2.2.3 and scikit-learn 1.7.0 are untouched, and the series in
which the defect was fixed (1.15.2) — and the app was rebuilt. No scientific algorithm
changed during packaging.

## Outputs

- Local app: `dist/NanoScope.app` (1,235 MB)
- Local DMG: `dist/NanoScope-0.1.0-arm64.dmg`
- DMG size: 399,858,275 bytes
- DMG SHA-256: `3346220e7aad76e75b59938b424ccca547edb0636f072aac0afea7f4e26ebc0a`
- DMG CRC32 (`hdiutil verify`): `$7842AC6B`
- App executable SHA-256: `2f18ab2df404251b079e7e1f094ce2e7872890396144adab87120e80f6faeacc`
- Embedded build id: `20261027T134030249854Z-132b42bdb808-dirty-8b2a6451451a`

Only the local DMG was requested for this release. Generated binaries, recordings and
models are not part of the source commit.

## Build and checks

- Clean `packaging/build_app.sh` build (icon → PyInstaller 6.21.0 freeze → bundle
  sanitisation → ad-hoc signature → DMG) with Python 3.11.7/arm64 on macOS 26.3.1; bundled
  Qt 6.11.1 (PySide6 6.11.1), TensorFlow 2.15.0, numpy 1.26.4, SciPy 1.15.3.
- Before the rebuild, the editable `nanoscope` install was confirmed to import under SciPy
  1.15.3 and 1,131 tests passed across `tests/io`, `tests/core`,
  `tests/test_acid_incubation_numerics.py`, `tests/test_acid_scientific_invariants.py`,
  `tests/analysis/test_filters.py` and `tests/analysis/test_truth.py`; the one failure,
  `tests/core/test_persist_invariant.py::test_no_module_in_the_app_writes_a_path_it_did_not_stage`,
  is red at HEAD independently of this build (unstaged CSV writes in the acid-study module).
- Bundle sanitisation and strict, deep ad-hoc signature verification passed
  (`valid on disk`, `satisfies its Designated Requirement`, `Signature=adhoc`).
- The standard `hdiutil create` DMG path succeeded directly; `hdiutil verify` reports a
  valid checksum.
- 37 packaging tests passed (`tests/packaging`).
- The rebuilt app's self-test, run with an empty environment (`env -i`, no `PATH`, no
  conda, no `PYTHONPATH`), exited 0: **19 applicable checks passed, 3 skipped** (22/22 in the
  tool's own summary, which counts skips as passed). Passing runtime checks include all four
  reader registrations, the *acid CUSUM and compiled clustering runtime* check that the
  earlier SciPy failed (CUSUM core, HDBSCAN, dip test and UMAP), synthetic HDF5 and Parquet
  round-trips, detection on a worker thread, CPU-only TensorFlow import, the google3 protobuf
  stub surviving freezing, full `MainWindow` construction that opens a document and the
  unified incubation launcher, project save/close/restore with nested visible segments,
  Populations/density/PSD windows, Seaborn comparison figures with frozen-snapshot
  restoration and twelve PNG/PDF/SVG exports, and the figure + events CSV export paths.
- External-recording loading, Keras model prediction and ML-controller detection with a
  supplied model were skipped because no external data/model arguments were supplied.
- The frozen executable's PYZ archive contains the change's three new modules
  (`nanoscope.core.sample_rate`, `nanoscope.core.sample_rate_state`,
  `nanoscope.gui.sample_rate_dialog`). The module-by-module bytecode comparison recorded
  on 2026-09-13 was not repeated for this build.
- Embedded build metadata (`_build_info.json`) confirms `git_commit` matches this build's
  HEAD exactly. `git_dirty` reads `true`, for the same reason as on 2026-09-13: the only
  modified tracked file at build time was `packaging/resources/NanoScope.icns`, regenerated
  by the build script's own icon-generation step — not uncommitted application code. (An
  untracked `examples/` folder was also present; it is not part of the package.)

Preferences, recovery files and synthetic test artifacts were isolated from user data.
Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging
and execution, not calibration validity or detector accuracy on difficult recordings — and
not the correctness of the acid-study median filter beyond the version assertion above.

Earlier verification is recorded in
[Release verification — 2026-09-13](RELEASE_VERIFICATION_2026-09-13.md) and
[Release verification — 2026-09-11](RELEASE_VERIFICATION_2026-09-11.md). Installation on an
individual Mac is a separate operation; the presence of this report does not prove that its
`/Applications` copy has been replaced.
