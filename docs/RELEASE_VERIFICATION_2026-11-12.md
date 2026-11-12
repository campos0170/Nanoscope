# Release verification — 2026-11-12

Build **46.45.9** of NanoScope 0.1.0 for Apple Silicon, built from commit `46a6954` on
`codex/beta-sweep-fixes` (pushed). It replaces build 46.31.52 (2026-11-11) in `/Applications`.

| Item | Value |
|---|---|
| Build ID | `20261112T150902864336Z-46a69542574d-dirty-749c61f46545` |
| Built | 2026-11-12 15:09:02 UTC (build took 3 min 35 s) |
| Source SHA-256 | `749c61f465459b00aed9885a09d238c44fa2bf4d051dc9ae80fb7ff4db12ee61` |
| Executable SHA-256 | `8188babf201901d8279ba12159bb31bac81b0f780e9ddb96d149cb8be99151ed` |
| `dirty` | the application icon, which every build regenerates; restored afterwards, no source file differed from the commit |

## What is new since 46.31.52

- **Plotting fixes (`46a6954`).** Noise ▸ Export data writes the spectrum's real values with
  Log-log axes ticked, and Voltage dependence does the same for its fit rows with Log y ticked;
  closing the recording a Noise figure is pinned to no longer blocks Save Project; report
  figures with an infinite or NaN provenance value can be saved to the Figure Library and
  bundled; Event waveforms can be snapshotted and bundled again (the provenance is written once);
  the shipped *Publication figure (light)* template carries only a look; `_{…}` and `^{…}` draw
  as sub- and superscripts on Comparison, study and report figures; Reset figure keeps the
  figure's style; incubation time is shown and typed in hours.
- **Usability pass (`0b8eada`).** Analysis ▸ Export submenu, advisory headlines with Show full
  text, plainer Detection row labels, the Preprocess workspace name, and Help ▸ User guide… /
  Keyboard shortcuts… with the guide inside the app. A brand-new installation (no saved
  settings) starts in the Essential controls and opens the Guided first analysis once; an
  installation that has been used is unchanged.
- **Defects found while clearing the failing tests (`a05de0a`).** Manual Study workspaces with a
  recording open again; Update plot on a Distributions figure no longer redraws superseded
  tables; opening a project no longer shifts a typed X range on a Distributions histogram; Log x
  / Log y over a typed major tick spacing no longer stops with an internal error; a project
  figure with a legend can be composed with shared axes again; the Inspector can be dragged
  narrower than 227 px; Curve fit ▸ Export last completed fit… uses the export prompt every
  other export uses.
- `a32fb94` splits two large source files and changes no behaviour.

The full list is under *Unreleased* in `CHANGELOG.md`.

## Outputs

| File | Size | SHA-256 |
|---|---|---|
| `dist/NanoScope-0.1.0-arm64.dmg` | 432,361,102 bytes | `36a7e834cf490a8e581ec92901126ad34177aa3390cef518f6937327182b8be5` |
| `dist/NanoScope.app` | 1.4 GB | executable as above |

No earlier build was kept, as the owner asked on 2026-11-11: the build replaced the 46.31.52
app and DMG in `dist/`, and the installed app was replaced directly, so there is no local
rollback copy.

## Build and checks

- **Environment.** The same private, no-download environment as the 2026-11-11 build, with the
  PySide6 6.9.3 overlay first on `PYTHONPATH`: `build_identity.py --check-environment` reported
  76 packages and no compatibility issues (TensorFlow 2.19.0, Keras 3.12.0, PyInstaller 6.21.0,
  matplotlib 3.10.0, pyqtgraph 0.14.0).
- **Command.** `packaging/build_app.sh` (full: icon, freeze, verify, sign, DMG), exit 0.
- **Integrity.** `verify_bundle.py` passed inside the build; `codesign --verify --deep --strict`
  passes; signature is ad-hoc (not notarized); `hdiutil verify` reports the DMG checksum VALID.
- **Tests.** A tree byte-identical to `46a6954` in `nanoscope/` and `tests/` passed the whole
  suite, one process per file: 890 files, none failing (63 files that name real recordings were
  not run); seven files that crashed under load passed when run alone. The last correction, to
  changelog and guide text only, was followed by a re-run of the 49 files that read those
  documents, all passing.
- **Tests under the shipped Qt.** The suite runs with PySide6 6.11.1 and the app ships 6.9.3, so
  the 23 test files the plotting fix added or changed were run again in the build environment:
  864 tests in 23 files, all passing.
- **Self-test.** `--self-test`: 23 of 23 checks passed from `dist/` (31 s) and 23 of 23 again
  from `/Applications` after installing from the DMG (33 s). The new check is that the user
  guide is in the bundle. The ML and real-recording checks are skipped by design when no model
  or file is passed.

## Installation

`/Applications/NanoScope.app` was replaced with the app copied out of the DMG (attached
read-only); the copy's signature was verified before the old app was removed, and its
`_build_info.json` names the build ID above. The guide inside the installed app is identical
to `docs/USER_GUIDE.md` at the commit, and the DMG's `INSTALL.txt` to `packaging/INSTALL.txt`.

## Manual

Not rebuilt for this build, at the owner's word on 2026-11-12. The illustrated manual is still
edition 2026-11-11 and names build 46.31.52; its prose has not been brought up to date with the
plotting fixes. The written guide that Help ▸ User guide… opens is current.

## Not verified or not updated for this build

- The SanDisk kits: the drive was not mounted.
- A GitHub Release: a draft, `v46.45.9`, was uploaded on 2026-11-12 (DMG, `INSTALL.txt`,
  `SHA256SUMS.txt`; GitHub's SHA-256 of each equals the local file's) and was not published when
  this was written; v46.31.52 is the latest one published.
- The Windows ZIP, which GitHub Actions builds: not checked from here.
- Notarization: not done (ad-hoc signature only; users click Open Anyway).
- Interactive use on the real display: the checks ran offscreen and through the self-test.
