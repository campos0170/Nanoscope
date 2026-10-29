# Release verification — 2026-10-28

Apple Silicon NanoScope 0.1.0 was rebuilt from `f9488d80632d` (HEAD of
`codex/beta-sweep-fixes`), which contains the fixes for the ten major defects of bug sweep 8
(SC3, NUM1, TU1, RD1, XS1, DATA1, DATA2, KEYS1, KEYS2 and THREAD2; see the
[sweep 8 findings](sweep8/FINDINGS.md), and the Unreleased section of
[CHANGELOG.md](../CHANGELOG.md) for each fix). A time base declared in **Inspector ▸ File Info ▸
Sample rate / time unit…** now reaches an open Signal Conditioning window and the Navigator
strip, and the exact-pass cache there is keyed by the rate it was computed at; the Results Fits
tab prints a fit only beside the table it was made on, and the Table selector takes the Event
inspector and gallery with it; a table marked stale by a rate declaration refuses hand-added
events, edge drags, merges and exports; the Results table ignores modifier chords on its
curation letters; the Event inspector and gallery leave Space / A / R to a focused button or an
open dropdown; and a Merge all, Re-measure or baseline comparison replaced by another analysis
no longer leaves the Manual and Acid study windows refusing to prepare a review. The 31 minor
defects and the four leads of that sweep are not addressed by this build, including the
intermittent crash under random open / close / detect storms recorded there as lead S8-LEAD1.
The display version remains 0.1.0; use the executable hash below to distinguish this build from
earlier 0.1.0 builds.

Unlike the 2026-10-27 build, this one carries **no change to the build environment**. The
embedded environment capture matches that build's for Python 3.11.7, the platform, the build
tools and all 85 recorded packages (SciPy 1.15.3, numpy 1.26.4, pandas 2.2.3, scikit-learn
1.7.0, TensorFlow 2.15.0, PySide6 6.11.1). Packaging changed no code: the frozen application
modules are the committed source, as compared below.

## Outputs

- Local app: `dist/NanoScope.app` (1,235 MB)
- Local DMG: `dist/NanoScope-0.1.0-arm64.dmg`
- DMG size: 401,538,339 bytes
- DMG SHA-256: `245cc1429730a591d634032a363557b9642ae1716109f15cd50d75327934138f`
- DMG CRC32 (`hdiutil verify`): `$13FD1FCC`
- App executable SHA-256: `4cdf45744b12b994101a48613c693f15d701e2888b08e7523292e55f7f5f3f5f`
- Embedded build id: `20261028T094844557887Z-f9488d80632d-dirty-148f005e58a8`

The DMG of the 2026-10-27 build (399,858,275 bytes, SHA-256
`3346220e7aad76e75b59938b424ccca547edb0636f072aac0afea7f4e26ebc0a`) was kept, byte for byte, at
`dist/previous/NanoScope-0.1.0-arm64.dmg`, because the build script removes the previous app and
DMG before it builds. `dist/` is not tracked. Only the local DMG was requested for this
release. Generated binaries, recordings and models are not part of the source commit.

## Build and checks

- Clean `packaging/build_app.sh` build (icon → PyInstaller 6.21.0 freeze → bundle sanitisation →
  ad-hoc signature → DMG), exit 0 in 3 min 42 s, with Python 3.11.7/arm64 on macOS 26.3.1;
  bundled Qt 6.11.1 (PySide6 6.11.1), TensorFlow 2.15.0, numpy 1.26.4, SciPy 1.15.3. Of the 80
  warning lines PyInstaller printed, 61 are import notices: 50 hidden `setuptools._distutils`
  imports, 3 other hidden imports it did not find (`scipy.special._cdflib`,
  `importlib_resources.trees`, `PySide6.support.deprecated`), 4 Windows and Linux ctypes
  libraries, and 4 submodule collections that failed on a missing optional package (OpenVINO
  twice, `jupyter_rfb`, a vendored `typeguard`); none names a missing `nanoscope` module. The
  other 19 are one notice that 17 SciPy binaries (14 extension modules and 3 gfortran runtime
  libraries) carry no valid macOS SDK version, which PyInstaller says can cause code-signing and
  hardened-runtime problems; the strict signature verification below passed.
- Before the build, the ten fixes were tested on the tree that became this commit: 14 new
  regression test files (260 cases, one or two per defect) pass in the working tree, and the
  whole suite, compared by failing test id with the untouched code of `d3af9ca`, ran 590 of 643
  test files (579 in a scratch copy of the tree, 11 that need the repository path in the working
  tree): 40 files are red on both, with the same failing tests, and none is new. The 53 files
  that read recordings by absolute path were not run. This is the record in the commit message
  of `f9488d8`; it was not repeated for this report.
- Bundle sanitisation passed: the Qt `cocoa` plugin resolves to the bundled QtGui 6.11.1, the
  binaries need macOS 15.0 and the plist declares 15.0, and the step removed 16 unreferenced Qt
  symlinks, one absolute build-host `LC_RPATH` entry (in numpy's bundled OpenBLAS) and 7
  `direct_url.json` files. Strict, deep ad-hoc signature verification passed in the build and
  again after it (`valid on disk`, `satisfies its Designated Requirement`, `Signature=adhoc`,
  identifier `org.nanoscope.NanoScope`, thin arm64).
- The standard `hdiutil create` DMG path succeeded directly; `hdiutil verify` reports a valid
  checksum. The image holds `NanoScope.app`, `INSTALL.txt` and an `Applications` alias. It was
  mounted read-only and the app copied out of it with `ditto`: the copy's signature is valid
  and it is identical to `dist/NanoScope.app` file for file (`diff -r`). The image was then
  unmounted.
- 37 packaging tests passed (`tests/packaging`, 86 s), run after the build.
- The rebuilt app's self-test (`NanoScope.app/Contents/MacOS/NanoScope --self-test`), run with
  an empty environment (`env -i`, no `PATH`, no conda, no `PYTHONPATH`), exited 0 twice, from
  `dist/` (25 s) and from the copy taken out of the DMG (23 s): **19 applicable checks passed,
  3 skipped** each time (22/22 in the tool's own summary, which counts skips as passed).
  Passing runtime checks include all four reader registrations, the *acid CUSUM and compiled
  clustering runtime* check (CUSUM core, HDBSCAN, dip test and UMAP), synthetic HDF5 and Parquet
  round-trips, detection on a worker thread, CPU-only TensorFlow import, the google3 protobuf
  stub surviving freezing, full `MainWindow` construction that opens a document and the unified
  incubation launcher, project save/close/restore with nested visible segments,
  Populations/density/PSD windows, Seaborn comparison figures with frozen-snapshot restoration
  and twelve PNG/PDF/SVG exports, and the figure + events CSV export paths.
- External-recording loading, Keras model prediction and ML-controller detection with a
  supplied model were skipped because no external data/model arguments were supplied; no
  recording was passed to the self-test.
- The frozen executable contains the new module `nanoscope.gui.curation_keys`, and its
  application code is the committed source. All 187 frozen `nanoscope.*` modules, the
  `nanoscope_selftest` module and the `nanoscope_app` entry-point script embedded in the
  executable (189 code objects) have the same code as the HEAD sources compiled with the build
  interpreter, comparing everything but the recorded file name and putting set constants in
  order; no frozen module lacks a source. The comparison was first checked on a known pair, the
  2026-10-27 executable against the `132b42b` sources it was built from: all 188 code objects
  identical. Against that executable, exactly 13 frozen modules differ, which are the 13 package
  files changed since `132b42b`: in `nanoscope.analysis` `conditioning_preview`, `events`,
  `modules`, `remeasure`; in `nanoscope.gui` `analysis_panel`, `analysis_worker`,
  `curation_keys` (new), `event_gallery`, `event_inspector`, `main_window`, `navigator`,
  `panels`, `signal_conditioning`. `nanoscope_selftest` and `nanoscope_app` are unchanged from
  that build. Two source modules, `nanoscope.__main__` and `nanoscope.incubation_cli`, are in
  neither bundle. This repeats the module-by-module comparison that the 2026-09-13 report
  recorded and the 2026-10-27 report did not.
- Embedded build metadata (`_build_info.json`) confirms `git_commit` matches this build's HEAD
  exactly, and `source_sha256`
  (`148f005e58a868232b0f9dcec134032cdef6dc92ccc522fee29baec3ff492e29`) equals the digest
  recomputed from the tree after the build. `git_dirty` reads `true`, for the same reason as on
  2026-10-27: the only modified tracked file at build time was
  `packaging/resources/NanoScope.icns`, regenerated by the build script's own icon-generation
  step — not uncommitted application code. (An untracked `examples/` folder was also present;
  it is not part of the package.)

Preferences, recovery files and synthetic test artifacts were isolated from user data: the
self-test redirects its settings, recovery and study-library paths to a scratch folder (172
files, 23.0 MB, removed at the end of each run), and both runs were made with writes to files
directly under the home folder denied. The only user-level files the checks wrote are
compiled-function (Numba) cache entries under `~/Library/Caches/NanoScope/numba`; the entries
made for the copy's scratch path were removed afterwards. The installed
`/Applications/NanoScope.app` was not touched: its folder date, executable hash and embedded
build id are the same before and after this verification.

Ad-hoc signing is not Apple Developer ID notarization. Build checks demonstrate packaging and
execution, not calibration validity or detector accuracy on difficult recordings. They do not
show the ten fixes working in the packaged app either: their regression tests ran against
source (which the comparison above shows is what was frozen), every GUI check is headless, and
no interactive session on this build was part of the verification.

Earlier verification is recorded in
[Release verification — 2026-10-27](RELEASE_VERIFICATION_2026-10-27.md),
[Release verification — 2026-09-13](RELEASE_VERIFICATION_2026-09-13.md) and
[Release verification — 2026-09-11](RELEASE_VERIFICATION_2026-09-11.md). Installation on an
individual Mac is a separate operation; the presence of this report does not prove that its
`/Applications` copy has been replaced. When this report was written that copy was still the
2026-10-19 build from `61cdea32986d` (embedded build id
`20261019T064014885500Z-61cdea32986d-dirty-9218089b961c`, executable SHA-256
`b39d403a7d50a92eb3519a405f4143d160dadc84084bef263ad8353ac187787e`).
