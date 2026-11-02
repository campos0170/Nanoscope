# Release verification — 2026-11-01 (third build)

Apple Silicon NanoScope 0.1.0 was rebuilt from `56bdf1e60a81` (HEAD of `codex/beta-sweep-fixes`, pushed
before the build). The last build, from `54c1483`, is recorded in
[Release verification — 2026-11-01 (second build)](RELEASE_VERIFICATION_2026-11-01b.md). Since then the
branch gained that record (`f20facc`) and one code commit.

**`56bdf1e`: plot text editable on the figure, and a redesign of manual annotation** (87 files; see the
Unreleased section of [CHANGELOG.md](../CHANGELOG.md)).
- **Plot windows:**
  - titles, axis labels, legend rows and the fit label are retyped on the figure itself, and styled from
    a right-click (Text style…); tick labels are styled the same way; Axis properties… and Add title… are
    there too; one Undo step each. Where views share one label (Noise, Event waveforms, Subpopulations),
    the label is renamed in Appearance and styled on the figure;
  - a fit's equation and numbers go in a fit label instead of the title;
  - presets keep style only; templates keep titles, axis labels and text styles, but not legend names;
  - a range pinned in Axis properties belongs to the view it was typed on in Noise, Event waveforms and
    Subpopulations, and to the pair of Types in Distribution summaries;
  - voltage axes, and the Voltage dependence fit numbers, read in mV.
- **Results ▸ Columns…** stays open for several picks.
- **Manual annotation:** a new Measure mask look, Annotate mode (E) where a drag adds the event at once,
  and a right-click menu on the trace.
- **Smaller changes:** Figure Library names, Compose, Comparison figures, the Distributions Features
  checklist, the Analysis Explorer refusal banner, greyed Analysis home rows and Ctrl+1–4 on plot pages.
- **Changes that affect existing habits or files:**
  - Ctrl+Shift+M no longer adds an event; it turns Analysis ▸ Measure event on and off.
  - A title saved with HTML tags such as `<sub>` now shows the tags literally; retype it with `_{…}`.
  - The data exports gain unit columns: `fit_params_units_json` (Voltage dependence) and
    `fit_slope_units`, `fit_intercept_units` (I-V).

The display version remains 0.1.0; use the executable hash below to distinguish this build from earlier
0.1.0 builds. The project format is unchanged (version 12), but projects this build saves carry new figure
settings (fit label, text styles, per-view ranges, caption position, hidden legend entries). The `54c1483`
build opens them and ignores those settings.

**Build environment: no change.** The embedded environment capture matches the `54c1483` build's in every
field but its capture time:
- Python 3.11.7, Darwin 25.3.0 and arm64;
- the same 85 recorded packages, including SciPy 1.15.3, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.0,
  tensorflow-macos 2.15.0 and PySide6 6.11.1;
- PyInstaller 6.21.0.

## Outputs

- **Local app:** `dist/NanoScope.app`, 1,236 MiB on disk (`du -sm`), 4,378 files.
- **Local DMG:** `dist/NanoScope-0.1.0-arm64.dmg`.
  - 401,158,622 bytes.
  - SHA-256 `671268a2c4a9b2634259e0ec528f70c0f65c811d212ac509f15a177d790d6d87`.
  - CRC32 (`hdiutil verify`): `$851A3022`.
- **App executable:** SHA-256 `dab93fa5b8ae1130f2dbb0d937ae2ccdb26d5a5cc961aa7697e46ad06dc6892c`,
  65,048,912 bytes.
- **Embedded build id:** `20261102T032916708182Z-56bdf1e60a81-dirty-4f878c1a7e32`, built 2026-11-02 03:29 UTC.

The build script removes the previous app and DMG, so the earlier DMGs were kept, byte for byte:

| Build | Path | SHA-256 |
|---|---|---|
| `54c1483` | `dist/previous/` | `2a81801e…0a0fbf` |
| `4395a71` | `dist/previous-20261101a/` | `d18bd99d…5ae40` |
| 2026-10-31 | `dist/previous-20261031/` | `c23adbd7…51117` |
| 2026-10-28 | `dist/previous-20261028/` | `245cc142…934138f` |
| 2026-10-27 | `dist/previous-20261027/` | `3346220e…e26ebc0a` |

`dist/` is not tracked.

## Build and checks

The bundle and self-test checks were each made by one agent.
- **Re-measured by a second agent,** who agreed:
  - the DMG hash and checksum, and the app inside the read-only mounted image;
  - the git commit and source digest;
  - the signature;
  - a spot check of the frozen code on changed, unchanged and every new module;
  - one self-test run;
  - the packaging-test count.
- **Single measurements:** the build-log checks, the environment comparison, the whole suite and the
  crash-rate comparisons.

### The build

- **Build.** A clean `packaging/build_app.sh` build exited 0 in 4 min 0 s, as captured by the command that
  ran the script.
- **PyInstaller warnings.** Its 80 warning lines are identical, line for line, to the `54c1483` build's
  once PyInstaller's elapsed-time prefixes are removed. None names a missing `nanoscope` module.
  - `warn-NanoScope.txt` has 717 missing-module entries, the count recorded for `54c1483`, and none names a
    missing `nanoscope` module. The `54c1483` file was overwritten by this build, so the two sets were not
    compared entry for entry.
- **Bundle sanitisation.** Its output is byte-identical to the `54c1483` build's.
- **Signature.** Strict, deep ad-hoc signature verification passed in the build and after it: identifier
  `org.nanoscope.NanoScope`, thin arm64.
- **DMG.** `hdiutil verify` reports a valid checksum. A copy of the app taken out of the read-only mounted
  image has a valid signature and is identical to `dist/NanoScope.app` file for file (`diff -r`).

### Frozen code

All 209 frozen `nanoscope.*` modules, `nanoscope_selftest`, the `nanoscope_app` entry script and the runtime
hook (212 entries) have the same code as `56bdf1e`'s sources compiled with the build interpreter.
- **Method check.** The same comparison of the `54c1483` executable against its sources gave all 209
  entries identical (its 206 `nanoscope.*` modules, `nanoscope_selftest`, the entry script and the runtime
  hook). The new executable compared with `54c1483`'s sources differs in exactly the changed and added
  modules: 182 identical, 27 different, 3 without source.
- **Against the `54c1483` executable,** 27 modules changed and 3 were added. That is exactly
  `git diff --name-status 54c1483 56bdf1e -- nanoscope/ packaging/nanoscope_app.py`.
- **New modules,** all frozen and identical to their source: `gui.column_checklist`, `gui.figure_text` and
  `gui.fit_label`.
- **Not frozen,** as before: `nanoscope.__main__` and `nanoscope.incubation_cli`.

### Build metadata

- `git_commit` in `_build_info.json` is `56bdf1e60a8122b242c5e83cf983f2fddc8c56d6`.
- `source_sha256` (`4f878c1a…3f35`) equals the digest recomputed from the tree.
- `git_dirty` reads `true`, for the same two causes as before: the icon file the build regenerates, and the
  untracked `examples/` folder.

### Tests

- **Whole suite.** It was run before the commit on an export of `f20facc` with the round's final patch
  applied; `git archive 56bdf1e` is identical to that export. One pytest process per file, two at a time.
  - 677 of the 740 test files outside `tests/packaging` ran. 671 were green (2 of them collect no tests,
    as on `54c1483`), and 11,432 tests passed.
  - The other 63 name the owner's home path and were not run.
  - Five files are red, with the same 60 failing ids as on `54c1483`, compared id for id.
  - One file crashed under load and passed when run alone, twice: `test_f2_zero_level_empty_state` (28).
- **Crash rate.** The round was built in three stages, each compared with the tree before it by repeated
  runs of GUI test files. A crash here is a segfault. Failures from the known PySide wrapper mix-up (a
  widget handed back as the wrong type) are counted separately as wrapper failures.
  - **First stage, against `f20facc`.** The 11 existing files nearest the changed code crashed in 8 of 264
    runs, against 5 of 264; these two sets ran one after the other, not interleaved. There was 1 wrapper
    failure on the new tree and none on `f20facc`.
    - The 17 new GUI test files crashed in 5 of 408 runs, all in `test_iv_vdep_fit_label` (5 of 24),
      which has no counterpart on `f20facc`.
    - A repeat on both trees at once gave 7 of 48 for `test_voltage_dep` and `test_iv_vdep_fit_label`,
      against 19 of 48 for `test_voltage_dep` and `test_iv` on `f20facc`.
    - A synthetic stand-in for the storm harness, with the Signal Conditioning window open, crashed in 4
      of 12 runs against 2 of 12.
  - **Second stage, against the first, interleaved:** 9 of 312 crashes against 10 of 312, plus 5 wrapper
    failures against none.
  - **Last stage, against the second, interleaved:** 0 of 48 against 1 of 48, plus 1 wrapper failure
    against none.
  - No crash comparison on its own is statistically clear. Wrapper failures, however, appeared only on the
    newer tree of each comparison: 8 in all, against none (Fisher two-sided p ≈ 0.008 if pooled, though
    the comparisons are chained, not independent). Both kinds come from the same PySide memory corruption.
    Counting crashes and wrapper failures together over the comparisons that ran the same files on both
    trees (the 11 existing files, the storm stand-in, the second and last stages, and the Signal
    Conditioning test below), the newer trees failed 32 of 652 runs and the older 22 of 652 (p ≈ 0.2).
    The I-V and Voltage dependence repeat, which ran different files on each tree, went the other way. This
    build may be somewhat more prone to that corruption; these samples cannot settle it.
  - The first stage found one crash site in new code (a `QTextDocument` built on every text refresh) and
    removed it. After that, every crash was at a construction or harness site, the kind that also crashes
    on the earlier trees.
  - The app's intermittent crash needs a Signal Conditioning window to exist (sweep 8). That window's test
    file crashed in 3 of 16 interleaved runs on the second stage's tree and 4 of 16 on `f20facc`, plus 1
    wrapper failure on the second stage's tree. The last stage was not part of this comparison.
- **Packaging tests.** After the build, 53 passed. `test_s4_bundle_selftest_args` was not run, under the
  home-path rule.

### Self-test

The rebuilt app's self-test (`--self-test`) was run with `env -i` plus only a scratch HOME and a scratch
recovery root, under a sandbox that denied writes under the home folder except `~/Library/Caches/NanoScope`.
- **Results.** It exited 0 four times: three times from `dist/` (twice by the checking agent, once by the
  confirming agent) and once from the copy taken out of the DMG. Each run gave **19 applicable checks
  passed, 3 skipped**: the recording and model checks, because none was passed.
- **From Applications.** After installation it passed again from `/Applications`, with `env -i`, the real
  home and no sandbox: 22/22 in the tool's summary.

The crash diagnostics are unchanged since `4395a71`, so the frozen crash-log check of the first 2026-11-01
record was not repeated.

## Installation

At the owner's request, `/Applications/NanoScope.app` now holds this build.
- It was copied out of the DMG with `ditto`.
- It is identical to `dist/NanoScope.app`, and its signature verifies.
- The `54c1483` build was moved to the Trash.

## Not verified for this build

- **The 64 test files that name the owner's home path:** 63 in the suite and
  `test_s4_bundle_selftest_args`.
- **The *clean exit* line in the frozen app,** as before.
- **⌘Z through the native menu on a real Mac** (design doc 13.4, question 5).
- **Windows.**
- **The new plotting and annotation features in the packaged app.** Their tests ran against source, which
  the comparison shows is what was frozen. Every GUI check is headless, and no interactive session was part
  of the verification. Hand checks on a real Mac are still owed:
  - the double-click timing on figure text;
  - the Results ▸ Columns… popup with a trackpad: focus back on the button, and a click on the button
    closing it;
  - Annotate drags and the trace right-click menu with a trackpad, including Control-click and two-finger
    click;
  - the mask on a Retina display, including its hover highlight, which was checked headless but has no
    pixel test.

Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging and execution, not
calibration validity or detector accuracy on difficult recordings.

Earlier verification is recorded in
[Release verification — 2026-11-01 (second build)](RELEASE_VERIFICATION_2026-11-01b.md) and the reports it
lists.
