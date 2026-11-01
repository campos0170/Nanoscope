# Release verification — 2026-10-31

Apple Silicon NanoScope 0.1.0 was rebuilt from `d09f1d2e438b` (HEAD of
`codex/beta-sweep-fixes`, pushed before the build). Since the 2026-10-28 build (`f9488d8`) the
branch gained these commits (and `a6e3c36`, which records the 2026-10-28 verification):

- **From `f9488d8` to `27ec0eb`:**
  - the Manual event table, with one door for adding events and trace keys with fitted edges
    (`e2c92cb`);
  - the fixes for bug sweep 9 (`3693928`, `70fdaee`);
  - edges keeping their own records when events merge (`87d1a97`);
  - trace keys following a "Treat this channel as" override (`f70403b`);
  - the app defects behind the pre-existing red tests, fixed, and stale tests updated (`c6130d0`);
  - the ML gate's reserved row restored, and seven stale-test decisions settled (`27ec0eb`).
- **The editing foundation** (`7e4e0f6`): one Edit ▸ Undo over settings, calibration and the time
  base; per-table form drafts; resets; input lineage.
- **The fixes for the 13 defects of bug sweep 10** (`3938d32`).
- **The owner's answers to the editing foundation's open questions** (`d09f1d2`). See
  [the design doc's 13.4 and 13.7](EDITING_FOUNDATION_DESIGN.md) and the Unreleased section of
  [CHANGELOG.md](../CHANGELOG.md).

The display version remains 0.1.0; use the executable hash below to distinguish this build from
earlier 0.1.0 builds.

As on 2026-10-28, this build carries **no change to the build environment**. The embedded
environment capture matches that build's in every field but its capture time:

- Python 3.11.7, Darwin 25.3.0 and arm64.
- The same 85 recorded packages: SciPy 1.15.3, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.0,
  tensorflow-macos 2.15.0 and PySide6 6.11.1.
- PyInstaller 6.21.0.

Packaging changed no code: the frozen application modules are the committed source, as compared
below.

## Outputs

- **Local app:** `dist/NanoScope.app`, 1,235 MiB on disk (`du -sm`), 4,378 files.
- **Local DMG:** `dist/NanoScope-0.1.0-arm64.dmg`.
  - Size: 400,332,092 bytes.
  - SHA-256: `c23adbd7aa24273acf126dd96d57330ecba04e38367fb65ed6174587b6c51117`.
  - CRC32 (`hdiutil verify`): `$ECBD77B5`.
- **App executable SHA-256:** `ed0493ba243df006d60026022468e210c80d5903cb46869ee76c2df7b24b289c`.
- **Embedded build id:** `20261101T013909067040Z-d09f1d2e438b-dirty-dcb5156fddd6`. It was built at
  2026-11-01 01:39 UTC, which is 2026-10-31 local time.

The build script removes the previous app and DMG before it builds, so the earlier DMGs were kept
byte for byte:

- **2026-10-28 build:** `dist/previous/NanoScope-0.1.0-arm64.dmg`, SHA-256 `245cc142…934138f`.
- **2026-10-27 build:** `dist/previous-20261027/NanoScope-0.1.0-arm64.dmg`, SHA-256 `3346220e…e26ebc0a`.

`dist/` is not tracked. Generated binaries, recordings and models are not part of the source commit.

## Build and checks

### The build

- **Build.** A clean `packaging/build_app.sh` build ran icon, PyInstaller freeze, bundle
  sanitisation, ad-hoc signature and DMG. It exited 0 in 4 min 02 s.
- **PyInstaller warnings.** It printed 80 warning lines, the same breakdown as on 2026-10-28:
  - **61 import notices.**
    - 50 hidden `setuptools._distutils` imports.
    - 3 other hidden imports it did not find: `scipy.special._cdflib`,
      `importlib_resources.trees` and `PySide6.support.deprecated`.
    - 4 Windows and Linux ctypes libraries.
    - 4 submodule collections that failed on a missing optional package: OpenVINO twice,
      `jupyter_rfb`, and a vendored `typeguard`.
  - **19 lines of one notice:** 17 SciPy binaries carry no valid macOS SDK version. The strict
    signature verification below passed.
  - **No nanoscope module.** No warning names a missing `nanoscope` module, and none of the 717
    missing-module entries in `warn-NanoScope.txt` does either.
- **Bundle sanitisation passed.**
  - The Qt `cocoa` plugin resolves to the bundled QtGui 6.11.1.
  - The binaries need macOS 15.0, and the plist declares 15.0.
  - The step removed 16 unreferenced Qt symlinks, one absolute build-host `LC_RPATH` entry (in
    numpy's bundled OpenBLAS), and 7 `direct_url.json` files.
- **Signature.** Strict, deep ad-hoc signature verification passed in the build and again after it:
  - `valid on disk`;
  - `satisfies its Designated Requirement`;
  - `Signature=adhoc`;
  - identifier `org.nanoscope.NanoScope`;
  - thin arm64.

### The DMG

- **Creation.** The standard `hdiutil create` path succeeded directly, and `hdiutil verify`
  reports a valid checksum.
- **Contents.** `NanoScope.app`, `INSTALL.txt` (identical to the 2026-10-28 image's) and an
  `Applications` symlink to `/Applications`.
- **Copy check.** The image was mounted read-only and the app copied out with `ditto`. The copy's
  signature is valid, and it is identical to `dist/NanoScope.app` file for file (`diff -r`).

### Tests

- **Whole suite.** After the build, alongside the release checks, the whole suite was run on an
  export of `d09f1d2`, one pytest process per file, and compared by failing test id with `27ec0eb`, the commit before the editing foundation.
  - 642 of the 705 test files outside `tests/packaging` ran: 633 green, and 10,809 tests passed
    in all. The other 63 name the owner's home path and were not run, because they read
    recordings by absolute path.
  - Five files are red, with the same 60 failing test ids as on `27ec0eb`, compared id for id:
    - `core/test_persist_invariant`, 1;
    - `test_lay2_detection_actions_above_the_fold`, 28;
    - `test_lay_detection_panel_reachable`, 20;
    - `test_modal_guard`, 1;
    - `test_wire_fold2r_stage_selector`, 10.
  - Four files failed only under load: three PySide crashes and the known
    `test_combined_filter_response` flake. Each passed when run alone:
    - `test_comparison_figures_integration`, 8 passed;
    - `test_iv`, 14 passed;
    - `test_lay_inspector_opens_reachable`, 9 passed;
    - `test_combined_filter_response`, 24 passed.
  - **No failing test is new.** The commits' own regression and acceptance tests are recorded in
    their commit messages and in design doc 13.5 to 13.7.
- **Packaging tests.** After the build, 32 passed (`tests/packaging`).
  `test_s4_bundle_selftest_args` was not run, for the same home-path rule.

### The self-test

The rebuilt app's self-test (`NanoScope.app/Contents/MacOS/NanoScope --self-test`) was run with an
empty environment: `env -i`, no `PATH`, no conda, no `PYTHONPATH`.

- **Results.** It exited 0 twice: from `dist/` in 30.5 s, and from the copy taken out of the DMG in
  37.9 s. Both runs gave **19 applicable checks passed, 3 skipped** (22/22 in the tool's own
  summary).
- **Slower than 10-28.** The 2026-10-28 runs took 25 s and 23 s.
  - Other checks were running at the same time.
  - The first run rewrote the Numba cache for the `dist/` path.
  - A confirming run from `dist/` under similar load took 23.3 s.
- **Checks that passed** include:
  - all four reader registrations;
  - the CUSUM, HDBSCAN, dip test and UMAP runtime;
  - HDF5 and Parquet round-trips;
  - detection on a worker thread;
  - CPU-only TensorFlow;
  - the google3 stub;
  - full `MainWindow` construction with the unified launcher;
  - project save, close and restore;
  - the Populations, density and PSD windows;
  - comparison figures with 12 PNG/PDF/SVG exports;
  - the figure and events CSV export paths.
- **Skipped checks.** External-recording loading, Keras model prediction and ML-controller
  detection were skipped, because no recording or model was passed.

### The frozen code

The frozen executable's application code is the committed source.

- **This build.** All 198 frozen `nanoscope.*` modules, the `nanoscope_selftest` module and the
  `nanoscope_app` entry-point script (200 code objects), plus the runtime hook, have the same code
  as the HEAD sources compiled with the build interpreter. The comparison ignores the recorded file
  name and puts set constants in order.
- **Method check.** The comparison was first run on a known pair: the 2026-10-28 executable against
  the `f9488d8` sources gave all 189 code objects identical.
- **Against the 2026-10-28 executable.** 61 frozen modules differ: 50 changed and 11 new. That is
  exactly `git diff --name-status f9488d8 d09f1d2 -- nanoscope/`.
  - The new modules are `analysis.edge_fit`, `event_statistics`, `lineage`, `manual_events`,
    `measurement_conversion`, `core.form_draft_state`, `table_choice_state`,
    `gui.convert_measurements_dialog`, `hand_row_preview`, `key_sequences` and `trace_keys`.
  - `nanoscope.__main__` and `nanoscope.incubation_cli` are in neither bundle, as before.

### Build metadata

`_build_info.json` confirms that `git_commit` matches `d09f1d2e438b317d42954f6b32fdb97301aafa73`
exactly.

- `source_sha256` (`dcb5156fddd60f11e5dac5cd71d8c550e6db08b7c5cbe970da2df995d9f58173`) equals the
  digest recomputed from the tree after the build.
- `git_dirty` reads `true`, as on 2026-10-28. It has two causes:
  - The only modified tracked file at build time was `packaging/resources/NanoScope.icns`, which
    the build's own icon step regenerates.
  - The untracked `examples/` folder makes `git status` dirty on its own. It is not part of the
    package or the source digest.

### Isolation

Preferences, recovery files and synthetic test artifacts were kept away from user data.

- The self-test redirects its settings, recovery and study-library paths to a scratch folder: 172
  files, 23.0 MB, removed at the end of each run.
- Both runs were made with writes under the home folder denied, except
  `~/Library/Caches/NanoScope`. The installed app was denied too. The system log shows no denial
  for NanoScope.
- The only user-level files the checks wrote are compiled-function (Numba) cache entries under
  `~/Library/Caches/NanoScope/numba`. The entries made for the copy's scratch path were removed.
- **Lock files.** Each self-test run also leaves a 1-byte project-lock file under the temporary
  folder's `nanoscope-project-save-locks-<uid>/` (`/tmp` with an empty environment), which the
  app does not remove. That was already so on
  2026-10-28.
- The installed `/Applications/NanoScope.app` was not touched. Its folder date, executable hash and
  embedded build id are the same before and after this verification.

## Not verified for this build

- **The 64 test files that name the owner's home path** were not run: 63 in the suite and one in
  `tests/packaging`. They were not run on this build, nor on any of the ten code commits since
  `f9488d8`. Among them, `test_s6_plots_undo_depth_disclosure` exercises the Plots window's undo
  keys and history, which these commits changed.
- **⌘Z through the native menu on a real Mac** (design doc 13.4, question 5). Every key test is
  offscreen, and this release widens which focused controls pass ⌘Z to the window's Undo.
- **The fixes in the packaged app.** Their regression tests ran against source, which the
  comparison above shows is what was frozen. Every GUI check is headless, and no interactive session
  on this build was part of the verification.

Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging and
execution, not calibration validity or detector accuracy on difficult recordings.

Earlier verification is recorded in
[Release verification — 2026-10-28](RELEASE_VERIFICATION_2026-10-28.md) and the reports it lists.
Installation on an individual Mac is a separate operation; this report does not prove that its
`/Applications` copy has been replaced. When this report was written that copy was still the
2026-10-19 build from `61cdea32986d`.
