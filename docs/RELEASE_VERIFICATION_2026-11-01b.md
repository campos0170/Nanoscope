# Release verification — 2026-11-01 (second build)

Apple Silicon NanoScope 0.1.0 was rebuilt from `54c14838834a` (HEAD of `codex/beta-sweep-fixes`, pushed
before the build). It is the second build of the day: the first, from `4395a71`, is recorded in
[Release verification — 2026-11-01](RELEASE_VERIFICATION_2026-11-01.md). Since then the branch gained one
code commit, plus that record (`5eae42a`).

**`54c1483`: steps 1–6 of the Plot Studio gap study** (57 files; see the Unreleased section of
[CHANGELOG.md](../CHANGELOG.md)).
- **Plot windows:** Reset view, Zoom back / Zoom forward, a pointer readout and the app's own right-click
  menu.
- **Main trace:**
  - Go to event / Go to time;
  - a sample index on the crosshair;
  - detection-threshold tracks, hand-moved edge marks and an ML confidence track.
- **Distributions:** a KDE overlay and an event-count heatmap.
- **Event waveforms:** alignment on the extremum, a variability band and the gallery's subsets.
- **Spectrograms:** colour limits and a labelled log-frequency view.
- **Formats:** plot templates in more windows. Projects are now version 12 and carry copies of the plot
  presets and templates; plot configuration files are version 3.
- **Fixes and smaller changes:**
  - the trace crosshair's readout is drawn (it was clipped above the pane);
  - a segment's crosshair time is right;
  - one cursor stays across the trace after a rebuild;
  - View ▸ Zoom Back / Forward are greyed out with nowhere to go;
  - the Event inspector has an ID… button.

The display version remains 0.1.0; use the executable hash below to distinguish this build from earlier
0.1.0 builds.

**Build environment: no change.** The embedded environment capture matches the `4395a71` build's in every
field but its capture time:
- Python 3.11.7, Darwin 25.3.0 and arm64;
- the same 85 recorded packages, including SciPy 1.15.3, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.0,
  tensorflow-macos 2.15.0 and PySide6 6.11.1;
- PyInstaller 6.21.0.

## Outputs

- **Local app:** `dist/NanoScope.app`, 1,236 MiB on disk (`du -sm`), 4,378 files.
- **Local DMG:** `dist/NanoScope-0.1.0-arm64.dmg`.
  - 402,132,266 bytes.
  - SHA-256 `2a81801eef9601531aee960ac33f6850d787d328e12fade77162c7f3980a0fbf`.
  - CRC32 (`hdiutil verify`): `$CDEC3B0C`.
- **App executable:** SHA-256 `2c6582f25230f3d1af936b7959e994febb7fac08cf02684849de7bc41746afa7`,
  64,876,720 bytes.
- **Embedded build id:** `20261101T144913585272Z-54c14838834a-dirty-478a9c1a0c9e`, built 2026-11-01 14:49 UTC.

The build script removes the previous app and DMG, so the earlier DMGs were kept, byte for byte:

| Build | Path | SHA-256 |
|---|---|---|
| `4395a71` | `dist/previous/` | `d18bd99d…5ae40` |
| 2026-10-31 | `dist/previous-20261031/` | `c23adbd7…51117` |
| 2026-10-28 | `dist/previous-20261028/` | `245cc142…934138f` |
| 2026-10-27 | `dist/previous-20261027/` | `3346220e…e26ebc0a` |

`dist/` is not tracked.

## Build and checks

The bundle and self-test checks were each made by one agent.
- **Re-measured by a second agent,** who agreed:
  - the DMG hash and checksum;
  - the git commit and source digest;
  - the signature;
  - a spot check of the frozen code, including every new module;
  - one self-test run;
  - the packaging-test count.
- **Single measurements:** the build-log checks, the environment comparison, the whole suite and the
  crash-rate comparison.

### The build

- **Build.** A clean `packaging/build_app.sh` build exited 0 in 4 min 11 s. The exit code was captured
  by the command that ran the script; the log itself ends at *Done*.
- **PyInstaller warnings.** Its 80 warning lines are identical, line for line, to the `4395a71` build's
  once PyInstaller's elapsed-time prefixes are removed.
  - None names a missing `nanoscope` module.
  - None of the 717 missing-module entries in `warn-NanoScope.txt` does either; they are the same set as
    before.
- **Bundle sanitisation.** Its output is byte-identical to the `4395a71` build's.
- **Signature.** Strict, deep ad-hoc signature verification passed in the build and after it: identifier
  `org.nanoscope.NanoScope`, thin arm64.
- **DMG.** `hdiutil verify` reports a valid checksum. A copy of the app taken out of the read-only mounted
  image has a valid signature and is identical to `dist/NanoScope.app` file for file (`diff -r`).

### Frozen code

All 206 frozen `nanoscope.*` modules, `nanoscope_selftest`, the `nanoscope_app` entry script and the runtime
hook have the same code as `54c1483`'s sources compiled with the build interpreter.
- **Method check.** The same comparison of the `4395a71` executable against its sources gave all 202
  entries identical.
- **Against the `4395a71` executable,** 24 modules changed and 7 were added. That is exactly
  `git diff --name-status 4395a71 54c1483 -- nanoscope/`.
- **New modules,** all frozen and identical to their source:
  - `analysis.boundary_marks`;
  - `analysis.detection_thresholds`;
  - `core.confidence`;
  - `core.plot_preset_state`;
  - `gui.plot_interaction`;
  - `gui.plot_preset_sync`;
  - `gui.plot_templates`.
- **Not frozen,** as before: `nanoscope.__main__` and `nanoscope.incubation_cli`.

### Build metadata

- `git_commit` in `_build_info.json` is `54c14838834aec453a9ebc052611604ad04ffbb5`.
- `source_sha256` (`478a9c1a…025c`) equals the digest recomputed from the tree.
- `git_dirty` reads `true`, for the same two causes as before: the icon file the build regenerates, and the
  untracked `examples/` folder.

### Tests

- **Whole suite.** It was run on an export of `54c1483`, one pytest process per file, two at a time.
  - 656 of the 719 test files outside `tests/packaging` ran. 647 were green, and 11,047 tests passed.
  - The other 63 name the owner's home path and were not run.
  - Five files are red, with the same 60 failing ids as on `27ec0eb`, compared id for id.
  - Four files crashed under load and passed when run alone:
    - `test_baseline_migration` (31);
    - `test_measure_tab` (43);
    - `test_plot_templates_and_project_presets` (37), a new file of this merge;
    - `test_warning_free_shared_plots` (21).
- **Crash rate.** Before committing, the session that wrote `54c1483` compared an existing intermittent
  segfault on the merged tree with `5eae42a`, using the same repeated runs on each tree, with the same
  crash signature.
  - First stage: 20 runs each, 6 crashes merged against 2 for the base. That looked unclear, so it ran
    30 more pairs: 4 against 5.
  - Pooled: 10 of 50 against 7 of 50.
  - The result is inconclusive (Fisher two-sided p ≈ 0.6). Neither no change nor an increase up to
    about two to three times the base rate is ruled out.
- **Packaging tests.** After the build, 53 passed. `test_s4_bundle_selftest_args` was not run, under the
  home-path rule.

### Self-test

The rebuilt app's self-test (`--self-test`) was run with `env -i` plus only a scratch HOME and a scratch
recovery root, under a sandbox that denied writes under the home folder except `~/Library/Caches/NanoScope`.
- **Results.** It exited 0 three times: twice from `dist/` and once from the copy taken out of the DMG.
  Each run gave **19 applicable checks passed, 3 skipped**: the recording and model checks, because none
  was passed.
- **From Applications.** After installation it passed again from `/Applications`, with `env -i`, the real
  home and no sandbox: 22/22 in the tool's summary.

The crash diagnostics are unchanged since `4395a71`, so the frozen crash-log check of the first 2026-11-01
record was not repeated.

## Installation

At the owner's request, `/Applications/NanoScope.app` now holds this build.
- It was copied out of the DMG with `ditto`.
- It is identical to `dist/NanoScope.app`, and its signature verifies.
- The `4395a71` build was moved to the Trash.

Projects this version saves are version 12; an older NanoScope refuses them as needing a newer version.

## Not verified for this build

- **The 64 test files that name the owner's home path:** 63 in the suite and
  `test_s4_bundle_selftest_args`.
- **The *clean exit* line in the frozen app,** as before.
- **⌘Z through the native menu on a real Mac** (design doc 13.4, question 5).
- **Windows.**
- **The new plotting features in the packaged app.** Their tests ran against source, which the comparison
  shows is what was frozen. Every GUI check is headless, and no interactive session was part of the
  verification.

Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging and execution, not
calibration validity or detector accuracy on difficult recordings.

Earlier verification is recorded in
[Release verification — 2026-11-01](RELEASE_VERIFICATION_2026-11-01.md) and the reports it lists.
