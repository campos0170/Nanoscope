# Release verification — 2026-11-11

Build **46.31.52** of NanoScope 0.1.0 for Apple Silicon, built from commit `36aa140` on
`codex/beta-sweep-fixes` (pushed). It replaces build 45.55.44 (2026-11-06) in `/Applications`.

| Item | Value |
|---|---|
| Build ID | `20261111T163237642239Z-36aa140fa2b2-dirty-8c6f5e37d549` |
| Built | 2026-11-11 16:32:37 UTC (build took 3 min 32 s) |
| Source SHA-256 | `8c6f5e37d54985a5869391c104f783eb55b29b1ec3b29bf91492254c2894209c` |
| Executable SHA-256 | `85e81d18dc66b6caf128273931335b83f342f4da32a63526716aedbf2ecb38e1` |
| `dirty` | the application icon, which every build regenerates; restored afterwards, no source file differed from the commit |

## What is new since 45.55.44

- CUSUM counts its own opposite-direction detections and warns when noise is likely producing
  rows; Null control runs the selected detector; Sweep threshold and Recovery map are
  Schmitt-only and greyed out with CUSUM (`c0b5f2a`).
- Batch detect runs whichever of Schmitt or CUSUM is selected (`ec83247`).
- The Noise page can analyse the processed signal as well as the raw one (`8dc5c1f`).
- Detect, Apply and Run detection… show ● for un-run edits; Ctrl/⌘+Return presses the run
  button of the panel you are in (`36aa140`).
- The Windows-build and test-portability commits since 45.55.44 change no Mac behaviour.

## Outputs

| File | Size | SHA-256 |
|---|---|---|
| `dist/NanoScope-0.1.0-arm64.dmg` | 426,161,356 bytes | `e592288c6f7d18f8890da4457c89281203314339abc3c47965350551ff31cdc3` |
| `dist/NanoScope.app` | 1.4 GB | executable as above |

At the owner's request no earlier build was kept: the previous DMGs in `dist/` and a copy of
the 45.55.44 app were deleted, so there is no local rollback copy.

## Build and checks

- **Environment.** The Anaconda base environment fails the release check (PySide6 6.11.1,
  Keras for tensorflow-macos). The build used the private, no-download environment assembled
  for the 2026-11-06 build from the owner's installed distributions, with the PySide6 6.9.3
  overlay first on `PYTHONPATH`: `build_identity.py --check-environment` reported 76 packages
  and no compatibility issues (TensorFlow 2.19.0, Keras 3.12.0, PyInstaller 6.21.0).
- **Command.** `packaging/build_app.sh` (full: icon, freeze, verify, sign, DMG), exit 0.
- **Integrity.** `verify_bundle.py` passed inside the build; `codesign --verify --deep --strict`
  passes; signature is ad-hoc (not notarized); `hdiutil verify` reports the DMG checksum VALID.
- **Tests.** The source at `36aa140` passed the whole suite, one process per file, 870 files
  (63 files that name real recordings were not run), with no failing test id that the previous
  commit does not also fail; files that crashed under load passed when run alone.
- **Self-test.** `--self-test`: 22 of 22 checks passed from `dist/` (33 s) and 22 of 22 again
  from `/Applications` after installing from the DMG (42 s). The ML and real-recording checks
  are skipped by design when no model or file is passed.

## Installation

`/Applications/NanoScope.app` was replaced with the app copied out of the DMG (attached
read-only); its signature was verified before the old app was removed, and its
`_build_info.json` names the build ID above.

## Manual

Edition 2026-11-11 was rebuilt for this build (commit `6de6eff`): 235 pages, all generator
checks passed. No illustration was re-captured.

| File | SHA-256 |
|---|---|
| `docs/manual.html` (and its identical copy) | `07a039cf9e27d616f48b31c3539f2c5332235e123fdde70debfd46184c237b07` |
| `docs/nanoscope-manual.pdf` | `a07d2b81b4904d4e984bf6034ac1abed96cdc205c1821ddc50a6775aa2afe0dc` |
| `docs/nanoscope-manual.zip` | `0f630f650696d9a8868eb7b2c8b5887fa605b8884aff4f729e113d11be625a62` |

## Not verified or not updated for this build

- The SanDisk kits: the drive was not mounted.
- A GitHub Release: `gh` is not signed in on this machine.
- The Windows ZIP, which GitHub Actions builds: not checked from here.
- Notarization: not done (ad-hoc signature only; users click Open Anyway).
- Interactive use on the real display: the checks ran offscreen and through the self-test.
