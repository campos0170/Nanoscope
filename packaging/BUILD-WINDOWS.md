# Building NanoScope for Windows

PyInstaller does **not** cross-compile. A Windows `.exe` has to be produced on Windows; the
macOS `.app` and this build are separate artifacts made on separate machines from the same
source tree.

## One-time setup on the Windows machine

Python **3.11** (the version the app is developed against; 3.12 works, 3.13 has no TensorFlow
wheels yet and would silently build without the ML detector).

```powershell
git clone <repo> nanoscope
cd nanoscope
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" pyinstaller pillow
```

Optional, for the ML detector — the app builds and runs without it and says so:

```powershell
pip install tensorflow keras
```

## Prove the source works there before packaging it

This is the step worth not skipping. The suite has only ever run on macOS, so it is the first
real evidence that the app works on Windows at all:

```powershell
pytest -q
```

Then run it from source once:

```powershell
nanoscope
```

## Build

```powershell
.\packaging\build_app.ps1
```

That cleans stale output, runs PyInstaller against `packaging/NanoScope-win.spec`, launches the
built `NanoScope.exe --self-test`, and zips the result. Options: `-NoZip` (fast iteration),
`-KeepCache` (reuse PyInstaller's analysis), `-SkipSelfTest`.

Output:

```
dist\NanoScope\NanoScope.exe          the app (onedir — ship the whole folder)
dist\NanoScope-0.1.0-win64.zip        distributable
```

## Built on GitHub instead

`.github/workflows/windows-build.yml` does all of the above on GitHub's Windows machines, so no
Windows PC is needed. It installs the macOS release's pinned versions
(`packaging/environments/macos-arm64-py311-validated.constraints.txt`, minus pytest 7.4 and
`tensorflow-io-gcs-filesystem`, which has no Windows wheel at that version), runs the core, I/O,
packaging and recovery tests, builds with `build_app.ps1`, runs `NanoScope.exe --self-test`, and
keeps the ZIP as a run artifact for 30 days.

- It runs on every push that changes `packaging/` or the workflow.
- Publishing a GitHub release builds that release's tag and attaches
  `NanoScope-<tag>-win64.zip` to it, adding the ZIP's line to `SHA256SUMS.txt`.
- For a release that already exists: Actions → Windows build → Run workflow, with `release_tag` set
  to its tag (for example `v45.55.44`). This builds the branch it is run from, and refuses unless
  that branch's app code (`nanoscope/` and `pyproject.toml`) is identical to the tag's, so the ZIP
  carries the same app as the DMG. Only packaging may differ, which is how a release made before
  a Windows packaging fix still gets a working Windows build.

Nothing is attached unless the tests and the self-test pass.

## What differs from the macOS build

| | macOS | Windows |
|---|---|---|
| spec | `NanoScope.spec` | `NanoScope-win.spec` |
| build | `build_app.sh` (bash) | `build_app.ps1` (PowerShell) |
| output | `NanoScope.app` + DMG | `dist\NanoScope\` + ZIP |
| icon | `NanoScope.icns` | `NanoScope.ico` (checked in; `make_icon.py --ico`) |
| signing | ad-hoc `codesign` | **none** — see below |
| logs | `~/Library/Logs/NanoScope` | `%LOCALAPPDATA%\NanoScope\Logs` |

Shared on purpose, because they are facts about *this app* rather than about an OS: the hidden
imports, the exclude list, and `rthooks/rthook_nanoscope_env.py`.

## The exe is unsigned

Windows SmartScreen will show "Windows protected your PC" on first run — **More info → Run
anyway**. Removing that warning needs an Authenticode code-signing certificate (an OV cert is
~£200/yr; an EV cert clears SmartScreen immediately, OV builds reputation over time). Nothing in
the build depends on it; add `signtool sign` after the PyInstaller step when a certificate exists.

## Two behavioural differences to know about

1. **`os.replace` fails when the destination is open in another process.** The atomic-save
   primitive (`nanoscope/core/atomic_write.py`) renames a temp file over the target. On Windows,
   exporting a CSV over one that is open in Excel raises instead of succeeding. No half-written
   file either way — but the failure is visible, and the message comes from the OS.
2. **Hard-link detection is inert.** `st_nlink` reports 1 on Windows, so the "this file has other
   names" advisory never fires. Nothing else depends on it.

## Known-untested

Nothing in `tests/` imports `packaging/`, so the suite does not cover the spec, the launcher or
the self-test on any platform — those are proved by `build_app.ps1` running the built exe's
`--self-test`, which is why that step is on by default.
