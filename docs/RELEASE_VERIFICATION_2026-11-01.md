# Release verification — 2026-11-01

Apple Silicon NanoScope 0.1.0 was rebuilt from `4395a71ce2d3` (HEAD of `codex/beta-sweep-fixes`, pushed
before the build). Since the 2026-10-31 build (`d09f1d2`), the branch gained one code commit, plus the
release record of that build (`c532501`, `650403e`).

**`4395a71`: crash recovery made hard to lose, and crashes made traceable** (see the Unreleased
section of [CHANGELOG.md](../CHANGELOG.md) and [user guide 11b](USER_GUIDE.md)).
- **Locks and timing.** Each window locks its own recovery folder. Checkpoints are taken sooner, and
  at once on an export, at the start of a run and when the app is left.
- **Prompts and discards.** Discard keeps copies for 7 days. The startup prompt names the work. No
  copy that holds work is pruned to make room (it is kept 90 days) or hidden silently.
- **Visibility.** A failed checkpoint stays visible until one succeeds.
- **Scripts.** A script's bare `MainWindow()` no longer writes recovery.
- **Crash log.** The frozen app writes the build id and pid in its start banner, a stack trace into
  its log on a crash, a *clean exit* line on a normal quit, and any unhandled exception.

The display version remains 0.1.0; use the executable hash below to distinguish this build from earlier
0.1.0 builds.

**Build environment: no change since 2026-10-31.** The embedded environment capture matches that
build's in every field but its capture time:
- Python 3.11.7, Darwin 25.3.0 and arm64;
- the same 85 recorded packages, including SciPy 1.15.3, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.0,
  tensorflow-macos 2.15.0 and PySide6 6.11.1;
- PyInstaller 6.21.0.

## Outputs

- **Local app:** `dist/NanoScope.app`, 1,236 MiB on disk (`du -sm`), 4,378 files.
- **Local DMG:** `dist/NanoScope-0.1.0-arm64.dmg`.
  - 402,722,448 bytes.
  - SHA-256 `d18bd99df825070dcb8286a90d3c1704fa6be1dee36bcf1348e2fc5c17d5ae40`.
  - CRC32 (`hdiutil verify`): `$AA3932C1`.
- **App executable SHA-256:** `a5dc48d105fd0ed9e10e9c7a61be0a58f78995a950ef9a3f921b804176c01082`.
- **Embedded build id:** `20261101T084058773815Z-4395a71ce2d3-dirty-865f0713772b`, built 2026-11-01 08:40 UTC.

The build script removes the previous app and DMG, so the earlier DMGs were kept, byte for byte:

| Build | Path | SHA-256 |
|---|---|---|
| 2026-10-31 | `dist/previous/` | `c23adbd7…51117` |
| 2026-10-28 | `dist/previous-20261028/` | `245cc142…934138f` |
| 2026-10-27 | `dist/previous-20261027/` | `3346220e…e26ebc0a` |

`dist/` is not tracked.

## Build and checks

The bundle, self-test and crash-log checks were each made by one agent.
- **Re-measured by a second agent,** who agreed:
  - the DMG hash and checksum;
  - the git commit and source digest;
  - the signature;
  - a spot check of the frozen code;
  - one self-test run;
  - the packaging-test count;
  - the SIGSEGV crash runs.
- **Single measurements:** the build-log checks, the whole suite, and the SIGTERM and SIGINT runs.

### The build

- **Build.** A clean `packaging/build_app.sh` build exited 0 in 4 min 21 s.
- **PyInstaller warnings.** The 80 warning lines fall into the same categories, with the same
  messages, as on 2026-10-31.
  - None names a missing `nanoscope` module.
  - None of the 717 missing-module entries in `warn-NanoScope.txt` does either.
  - `nanoscope.core.recovery` is a new importer of `msvcrt`, a Windows-only, conditional import.
- **Bundle sanitisation.** It passed with the same output as on 2026-10-31.
- **Signature.** Strict, deep ad-hoc signature verification passed in the build and after it:
  identifier `org.nanoscope.NanoScope`, thin arm64.
- **DMG.** `hdiutil verify` reports a valid checksum. A copy of the app taken out of the read-only
  mounted image has a valid signature and is identical to `dist/NanoScope.app` file for file
  (`diff -r`).

### Frozen code

All 199 frozen `nanoscope.*` modules, `nanoscope_selftest`, the `nanoscope_app` entry script and the
runtime hook have the same code as `4395a71`'s sources compiled with the build interpreter.
- **Method check.** The same comparison of the 2026-10-31 executable against `d09f1d2` gave all 201
  entries identical.
- **Against the 2026-10-31 executable, exactly these differ,** which is `git diff --name-status d09f1d2
  4395a71` for the package plus the entry script:
  - `core.recovery`;
  - `gui.app`;
  - `gui.main_window`;
  - `gui.recovery_controller`;
  - `gui.worker_thread`;
  - `nanoscope_app`;
  - the new `nanoscope.clean_exit`.

### Build metadata

- `git_commit` in `_build_info.json` is `4395a71ce2d31b6b479817890a96a7483ab60cfa`.
- `source_sha256` (`865f0713…bbb5`) equals the digest recomputed from the tree.
- `git_dirty` reads `true`, for the two causes recorded on 2026-10-31: the icon file the build
  regenerates, and the untracked `examples/` folder.

### Tests

- **Whole suite.** It was run on an export of `4395a71`, one pytest process per file, two at a time.
  - 644 of the 707 test files outside `tests/packaging` ran. 637 were green, and 10,827 tests passed.
  - The other 63 name the owner's home path and were not run.
  - Five files are red, with the same 60 failing ids as on `27ec0eb`, compared id for id:
    `core/test_persist_invariant` 1, `test_lay2_detection_actions_above_the_fold` 28,
    `test_lay_detection_panel_reachable` 20, `test_modal_guard` 1, `test_wire_fold2r_stage_selector` 10.
  - Two files crashed under load and passed when run alone: `test_s3b_exp_save_image_staged` (40) and
    `test_study_curation_identity_case` (6).
- **Packaging tests.** After the build, 53 passed: the 32 of 2026-10-31 and the 21 new
  `test_crash_diagnostics` tests. `test_s4_bundle_selftest_args` was not run, for the home-path rule.
  - The confirming agent's run had 52 passed and 1 failed: in `test_s4_bundle_ml_source_guard`, the source
    self-test subprocess that the test drives ended with SIGSEGV while the machine was loaded. That file
    passed 2/2 when run alone, twice.

### Self-test

The rebuilt app's self-test (`--self-test`) was run with `env -i` plus only a scratch HOME and a scratch
recovery root, under a sandbox that denied writes under the home folder except
`~/Library/Caches/NanoScope`. The scratch HOME is a change from 2026-10-31: the first run built the font
cache cold.
- **Results.** It exited 0 three times: twice from `dist/` and once from the copy taken out of the DMG.
  Each run gave **19 applicable checks passed, 3 skipped**: the recording and model checks, because
  none was passed.
- **From Applications.** After installation it passed again from `/Applications`: 22/22 in the tool's
  summary, the same 19 passed and 3 skipped. That run used `env -i` with the real home and no sandbox.

### The crash log in the frozen app

No test exercises this in the frozen app, so the built GUI app was launched seven times: five by the
checking agent and two more SIGSEGV runs by the confirmer. Each launch used a scratch HOME, a scratch
recovery root, offscreen Qt, and a sandbox that also blocked the preferences daemon.
- **A deliberate SIGSEGV,** sent both during startup and while the app sat idle in its event loop, left
  in the log:
  - the start banner with the right pid and build id;
  - *Fatal Python error: Segmentation fault* with the Python stack;
  - no *clean exit* line.
- **An unhandled exception** raised in the launcher's event filter was logged once, with its time and
  thread (one run).
  - PySide's own *Error calling Python override of QObject::eventFilter():* prefix ends without a
    newline, so the hook's `----- unhandled …` header can start mid-line. A search anchored at the start
    of a line misses it.
- **SIGTERM** (one run) leaves the banner and the startup lines that follow it, but no stack and no
  *clean exit* line. As designed, it reads as a force quit: the app installs no SIGTERM handler.
- **The *clean exit* line was not shown in the frozen app.** An offscreen process cannot be sent a
  normal quit. Tests show it from source.
- **The owner's files were unchanged** before and after: their log, recovery folder, caches and
  preferences.
- **The four deliberate crashes left macOS crash reports** in `~/Library/Logs/DiagnosticReports`:
  `NanoScope-2026-11-01-035117`, `-035149`, `-035916` and `-035937.ips`. macOS wrote them outside the
  sandbox.

## Installation

At the owner's request, `/Applications/NanoScope.app` now holds this build.
- It was copied out of the DMG with `ditto`.
- It is identical to `dist/NanoScope.app`, and its signature verifies.
- The 2026-10-31 build was moved to the Trash.

## Not verified for this build

- **63 test files that name the owner's home path,** and `test_s4_bundle_selftest_args`.
  `test_s6_plots_undo_depth_disclosure` was run, sandboxed, on `d09f1d2` (see the 2026-10-31 record),
  not on this build. No code it exercises changed since.
- **The *clean exit* line in the frozen app** (see above). The first normal quit of the installed app
  will show it at the end of `~/Library/Logs/NanoScope/nanoscope.log`.
- **⌘Z through the native menu on a real Mac** (design doc 13.4, question 5).
- **Windows and network home folders:** the recovery lock's Windows branch and Windows crash output.
- **The recovery changes in the packaged app.** Their regression tests ran against source, which the
  comparison shows is what was frozen. The crash diagnostics were exercised in the frozen app (above),
  except the *clean exit* line. Every GUI check is headless, and no interactive session was part of
  the verification.

Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging and
execution, not calibration validity or detector accuracy on difficult recordings.

Earlier verification is recorded in
[Release verification — 2026-10-31](RELEASE_VERIFICATION_2026-10-31.md) and the reports it lists.
