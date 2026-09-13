# Packaging NanoScope as a macOS app

This directory builds `NanoScope.app` — a self-contained, double-clickable macOS application
that needs no Python, no Anaconda and no `pip install` on the machine it runs on.

**Target:** Apple Silicon (arm64) only, macOS 15+ (the bundled PySide6 is built for 15.0). Not universal2, not Intel, not Windows/Linux.
**Signing:** ad-hoc only (no Apple Developer ID). See [Installing](#installing-for-a-labmate).
**Size:** 1125 MB installed, 346 MB as a compressed DMG. TensorFlow is 716 MB of that, by design —
the ML detector works out of the box.

---

## Rebuilding

One command, from anywhere:

```bash
/Users/abrahamcampos/Desktop/nanoscope/packaging/build_app.sh
```

It generates the icon, freezes with PyInstaller, checks the bundle, ad-hoc signs it, and writes
`dist/NanoScope-0.1.0-arm64.dmg`. Expect **5–8 minutes** cold.

| Flag | Effect |
|---|---|
| *(none)* | full build: icon → app → verify → sign → DMG |
| `--no-dmg` | stop after the signed `.app` (use this while iterating) |
| `--keep-cache` | reuse PyInstaller's analysis cache — roughly halves rebuild time |

Override the interpreter with `NANOSCOPE_PYTHON=/path/to/python ./packaging/build_app.sh`.
The default is `/opt/anaconda3/bin/python3`.

### Prerequisites

```bash
/opt/anaconda3/bin/python3 -m pip install -e /Users/abrahamcampos/Desktop/nanoscope
/opt/anaconda3/bin/python3 -m pip install 'pyinstaller>=6.11'
```

The `-e` install matters: the spec calls `copy_metadata("nanoscope")` so
`importlib.metadata.version("nanoscope")` works inside the bundle, which is what stamps the app
version onto every detection result's provenance. `build_app.sh` refuses to start without it.

### After changing the app

Editing anything under `nanoscope/` needs a rebuild — the Python is compiled into the bundle.
Adding a **new dependency**, or a new module that is imported dynamically, may also need an entry
in `hiddenimports` in `NanoScope.spec`. Run the self-test after every rebuild:

```bash
dist/NanoScope.app/Contents/MacOS/NanoScope --self-test \
    ~/some_recording.abf ~/some_model.keras
```

20 checks, approximately one minute, exit code = number of failures. Both file arguments are optional; the checks
that need them report `SKIPPED` instead of failing — but pass them when you have them: the ML
check runs the model over a slice of the REAL recording when a data file is supplied, and falls
back to synthetic noise (where a detector that finds nothing still looks fine) when it is not.

The checks are headless (offscreen Qt), and one of them auto-dismisses modal dialogs — the app
raises its unsaved-work prompt when the window closes with a document loaded, and a modal inside a
headless run is a permanent deadlock (`QDialog.exec()` waiting for a click that cannot come). The
check reports which dialog it answered rather than hanging on it.

The project-restore check saves a synthetic recording with nested segments, closes its window,
then restores the project in a new window and verifies segment sizes, visible-layer count,
and selected calibration, colour and time-offset values. Each check logs its start and result immediately, so a native crash preserves
the last check reached.

The comparison-figure check exercises the packaged Seaborn dependency with facets, a pair matrix,
strip and swarm observations, source-free snapshot restore and twelve PNG/PDF/SVG exports. It
also verifies that styling/restoration do not prepare the measurements again and that source
event tables remain exactly unchanged.

---

## Files

| File | Role |
|---|---|
| `build_app.sh` | The one command. Preflight → icon → freeze → verify → sign → DMG. |
| `NanoScope.spec` | PyInstaller spec: hidden imports, excludes, TOC pruning, `Info.plist`. |
| `nanoscope_app.py` | Frozen entry point. Redirects stdio to a log, shows a crash dialog, handles `--self-test`. |
| `nanoscope_selftest.py` | The 19 in-bundle checks. One per way this bundle can break. |
| `rthooks/rthook_nanoscope_env.py` | Runtime hook: `MPLBACKEND=Agg`, writable `MPLCONFIGDIR`, TF CPU env. |
| `qt_prefix_fix.py` | Build-time fix that stops PyInstaller shipping Anaconda's **Qt5** plugins. |
| `verify_bundle.py` | Post-build integrity checks; run automatically by `build_app.sh`. |
| `run_pyinstaller.py` | Wrapper that masks the obsolete `pathlib` backport in this conda env. |
| `make_icon.py` | Generates `resources/NanoScope.icns` (nanopore ring + blockade trace). |
| `INSTALL.txt` | Shipped inside the DMG. Written for the labmate, not for you. |

---

## Why PyInstaller

It is the only one of PyInstaller / py2app / Nuitka with maintained first-party hooks for every
hard part of *this* dependency set at once: PySide6's split framework layout, TensorFlow's lazy
`_api` module tree, scipy/sklearn's compiled extensions, h5py's plugins. py2app has no TensorFlow
story; Nuitka would compile for 40+ minutes per iteration and still need identical hook knowledge.

Layout is **onedir inside a .app**, never onefile — a onefile build of a 1.1 GB payload would
unpack the whole thing to a temp directory on every launch.

---

## The four landmines that actually bit, and how the build handles them

### 1. Anaconda's `qt.conf` — hit at *build* time, not run time

`nanoscope/bootstrap.py` exists because `/opt/anaconda3/bin/qt.conf` declares
`Prefix = /opt/anaconda3`, and Qt reads `qt.conf` from the directory of the running executable.
PyInstaller determines where Qt lives by importing PySide6 **in a subprocess whose executable is
`/opt/anaconda3/bin/python3`** and asking `QLibraryInfo`. It is therefore told:

```
version       = 6.11.1            <- PySide6's Qt6, correct
PluginsPath   = /opt/anaconda3/plugins   <- conda's Qt 5.15 plugins, WRONG
LibrariesPath = /opt/anaconda3/lib
```

The first build shipped Qt5's `libqcocoa.dylib` inside a Qt6 app and died on launch with
`Plugin uses incompatible Qt library (5.15.0)` → `Could not find the Qt platform plugin "cocoa"`.
Setting `QT_PLUGIN_PATH` does not help; `QLibraryInfo` ignores it.

`qt_prefix_fix.py` re-roots the whole location table onto `<site-packages>/PySide6/Qt` before
`Analysis()` runs, and `verify_bundle.py` fails the build if the bundled cocoa plugin is ever
anything but Qt6 again.

`bootstrap.py` itself was changed to stand down when frozen (`sys.frozen` / `sys._MEIPASS`) —
inside the bundle there is no conda `qt.conf` to defend against, and recomputing the path from
`PySide6.__file__` would aim Qt at `Contents/Resources` while PyInstaller aimed it at
`Contents/Frameworks`.

### 2. `fsspec.gui` → 325 MB of VTK

`pyarrow.fs` imports `fsspec`, whose `gui` submodule imports `panel` at module scope — and
`panel` reaches HoloViews, hvplot, bokeh, intake, altair and **VTK**. Nothing in the app imports
`fsspec.gui`; PyInstaller's module graph simply walks into it. Separately,
`neo.rawio.spikeglxrawio` dragged in `probeinterface` → `zarr` → `numcodecs`. Both chains are cut
in `_LEAF_EXCLUDES`. That one exclusion list is worth 325 MB.

### 3. TensorFlow must stay on the CPU

`analysis/ml_detect._import_tf()` calls `tf.config.set_visible_devices([], "GPU")` before the
first TF op because the Apple Metal pluggable device SIGABRTs (exit 134) on this mixed-float16
LSTM. The bundle takes the belt-and-braces route: `tensorflow-plugins/` (i.e.
`libmetal_plugin.dylib`) is pruned from the TOC entirely, so there is no pluggable device to
register in the first place, and the runtime hook sets `CUDA_VISIBLE_DEVICES=-1` before TF's C++
core loads. The self-test asserts `tf.config.get_visible_devices("GPU")` is empty and that no
plugin dylib shipped.

### 4. The `google3` protobuf stub

After TF loads, `ml_detect` registers empty stub packages for the `google3.*` ancestors protobuf
leaves out of `sys.modules`; without them the next `QObject.disconnect()` makes shiboken abort the
process. This survives freezing, but "it survives" is only worth asserting if you actually run the
sequence — so the self-test runs a real ML detection through `AnalysisController` and lets the
controller disconnect its worker, which is the exact call that used to kill the app.

### Also handled, less dramatically

* **matplotlib** — the macOS default backend is `macosx`, which builds an NSWindow per figure
  inside a process that already owns a Qt `NSApplication`. The runtime hook forces `MPLBACKEND=Agg`
  (the app only ever calls `savefig`) and points `MPLCONFIGDIR` at
  `~/Library/Caches/NanoScope/matplotlib` so the first-run font cache has somewhere writable.
* **The four readers** — `io/base.py` imports `abf`/`hdf5`/`tdms`/`text` through
  `importlib.import_module`, invisible to the module graph. They are explicit `hiddenimports`;
  without them the app starts fine and then claims it has no reader for `.abf`.
* **The 16 MB worker stack** — still applies in the bundle. The self-test runs a detection through
  the real `AnalysisController` QThread rather than calling `detect_events` directly, so the stack
  size is what is under test.
* **Anaconda's `pathlib` backport** — PyInstaller refuses to run when it sees that distribution.
  `run_pyinstaller.py` masks it for the preflight check only, rather than uninstalling a package
  out of a working analysis environment. (`import pathlib` already resolves to the stdlib here;
  the wrapper asserts that before masking anything. If you ever want it gone for real:
  `pip uninstall pathlib`.)

---

## Trimming: what was cut and what was measured

| Stage | Size |
|---|---|
| First working build | **1.7 GB** |
| − VTK/HoloViz chain, zarr, Qt module excludes | 1.1 GB |
| − huggingface_hub/pydantic/mypy/hf_xet/lxml, grpc | **1.1 GB (1125 MB)** |
| Compressed DMG | **346 MB** |

Qt went from ~1.2 GB installed to **62 MB** in the bundle. `QtWebEngineCore` alone is 589 MB and
this app imports exactly `QtCore`, `QtGui` and `QtWidgets`.

Two mechanisms do the work, and they are not interchangeable:

* `excludes` in the spec operates on the **import graph** — it stops modules being followed.
* `_prune_toc` operates on the **collected file list** — it removes what hooks add wholesale
  (Qt's `qml/`, `translations/`, unused plugin families, TensorFlow's 159 MB of C++ headers).

Pruning a Qt framework's payload leaves PyInstaller's mirror symlinks in `Contents/Frameworks` and
`Contents/Resources` pointing at nothing, and `codesign --deep` then fails the *entire bundle*
with a bare "No such file or directory". `verify_bundle.py` removes those symlinks — but only
after proving no Mach-O in the bundle references the name, matching the exact NUL-terminated load
command rather than a substring (`@rpath/QtQml` is a prefix of `@rpath/QtQmlModels`).

**One exclusion was tried and reverted:** `tensorflow.lite` (17 MB). `import tensorflow` pulls it
in eagerly, so excluding it makes TensorFlow unimportable — the bundle still builds, signs and
launches, and only the ML detector silently degrades to "TensorFlow is not available". The
self-test caught it. Do not re-add it.

---

## Verification performed

* `--self-test`: **18/18** from `dist/`, and 18/18 again from a copy installed out of the DMG at
  a different path, run with `env -i` (no `PATH`, no conda, no `PYTHONPATH`, no environment at
  all) — including a real `.keras` teacher model (`teacher_trained.keras`: W=2048, C=12;
  the student models are C=1) → prediction, an ML detection driven through `AnalysisController`
  and torn down, and a full `MainWindow` that opens a document.
* **All four readers on REAL recordings**, each through `--self-test FILE` in the installed
  bundle: ABF 2,993,798 samples → 1337 events; HDF5 9,800,724 → 273; TDMS 13,428,995 → 1;
  CSV 2,993,798 → 1337. The ABF and the CSV are the *same* recording exported two ways and
  agree exactly, which is a cross-check on the readers and not just on their imports.
* **The standalone analysis surfaces actually compute**, not merely import: Populations draws a
  histogram, the density window fills a 2-D histogram and an event ensemble, and the PSD window
  produces a finite Welch spectrum with a usable `total_rms` — each also written to a PNG.
  Building `MainWindow` does not cover these; they are constructed lazily on the menu action.
* **The export paths**: `save_figure` writing PDF + PNG + the source-data CSV sidecar, and the
  accepted-events CSV round-tripped back through pandas.
* Launched windowed from `dist/` with a sanitised environment (`env -i`, minimal `PATH`, cwd `/`)
  and confirmed with `lsof` that the process maps **nothing** outside the bundle but macOS's own
  frameworks — no `/opt/anaconda3`, no external `site-packages`.
* `open -a NanoScope.app recording.abf` → the log shows the open-document AppleEvent arriving and
  the file being opened, so the document types in `Info.plist` are not a lie.
* `codesign --verify --deep --strict`: valid, `Signature=adhoc`, `Mach-O thin (arm64)`.
* **Gatekeeper, for real**: quarantined DMG → copied app inherits `com.apple.quarantine` →
  launch is refused by the kernel (`AppleSystemPolicy: Security policy would not allow
  process`, then `appDeath`). `xattr -d -r com.apple.quarantine` (Route B in `INSTALL.txt`)
  clears it and the app opens. The documented workaround is the observed behaviour.

---

## Installing (for a labmate)

`INSTALL.txt` ships inside the DMG and says this in plain language. The short version:

The app is **ad-hoc signed only** — not signed with an Apple Developer ID and not notarised.
`spctl -a -t exec` reports `rejected`, and that is expected. The ad-hoc signature is not
decorative: arm64 refuses to execute an unsigned Mach-O at all, so without it the app would be
SIGKILLed instantly. What it does *not* do is satisfy Gatekeeper.

So: **do not promise anyone a clean double-click on a Mac that has never seen this app.** If the
DMG is downloaded through a browser, Slack, or email, macOS attaches `com.apple.quarantine` and
the first launch is blocked with a dialog offering only *Move to Trash* / *Done*.

Two ways through, both once-per-machine:

1. **System Settings** — let it get blocked, then System Settings → Privacy & Security → Security
   → **Open Anyway** → authenticate → open again and confirm.
2. **Terminal** — `xattr -d -r com.apple.quarantine /Applications/NanoScope.app`, then open
   normally.

**Right-click → Open no longer works.** Apple removed that Gatekeeper override in macOS 15
Sequoia, so on Sequoia and later (including macOS 26, which this was built on) it is not an
option. Older instructions on the internet still recommend it.

If the app is copied over on a USB stick or a network share instead of downloaded, macOS often
does not flag it and it simply opens.

To make this go away properly, the app would need to be signed with a paid Apple Developer ID
(99 USD/yr) and notarised. The spec has `codesign_identity=None` and signing is a single step in
`build_app.sh`, so that is a small change if a Developer ID ever appears.

---

## Runtime diagnostics

The app is windowed and has no terminal, so `nanoscope_app.py` redirects file descriptors 1 and 2
(not just `sys.stdout`, so TensorFlow's and Qt's C++ output is captured too) to:

```
~/Library/Logs/NanoScope/nanoscope.log
```

It rotates at 2 MB, keeping one previous file. An unhandled exception at startup produces a Qt
dialog naming that path, with the traceback in the "Show Details" pane — not a silently vanishing
Dock icon.
