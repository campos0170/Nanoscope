# Release identity and environment capture

The marketing version remains **0.1.0**, defined once in
`nanoscope/_version.py`. Package metadata, build scripts and both PyInstaller
specs read that value. A rebuild is not automatically a new marketing release.

Every PyInstaller spec run generates `build/release-metadata/_build_info.json`
and embeds it under `nanoscope/` in the frozen application. It records:

- UTC build time with microseconds and a distinct build ID.
- Git commit, clean/modified/unavailable status, and a SHA-256 of application
  and packaging source inputs, including uncommitted source.
- Python, platform/architecture, exact app runtime dependency closure,
  separately listed build-tool versions, and metadata compatibility issues.

The macOS `CFBundleVersion` is a three-integer UTC-minute ordinal, separate
from `CFBundleShortVersionString`. It conservatively respects the positive
first component and **4/2/2 digit limits** in Apple's
[archived official build-version reference](https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/CoreFoundationKeys.html#//apple_ref/doc/uid/TP40009249-SW1).
Minutes since 2026-01-01 UTC, plus 10,000, are split into base-100 components;
the epoch starts at `1.0.0`. The number increases between UTC minutes if the
clock is correct. **Builds in the same minute can share this number**; use the
full microsecond build ID and source hash to distinguish them. Out-of-range
dates fail rather than wrapping. This is not a globally allocated release
sequence, and no uniqueness guarantee is made for simultaneous builds.
The immutable build ID and source SHA are also recorded in Info.plist. Windows
version resources are generated from the same marketing version and include
the build ID in their comments. These identify inputs; they do not promise
bit-for-bit binary reproducibility or code signing/notarization.

About NanoScope reads the **running application's embedded record**, not a DMG,
another installation, or today's repository checkout. An unbuilt source launch
says so. Copy diagnostics is explicit and local; the summary contains no
recording/project names, filesystem paths, logs, environment variables,
installer URLs or network upload.

## Captured environment, not an unverified lockfile

The checked-in `environments/macos-arm64-py311-captured.json` and companion
`.constraints.txt` capture the existing build environment's transitive **app
runtime** dependency versions. They deliberately exclude unrelated Anaconda
packages. Optional ML packages are included when present; build tools are a
separate section in JSON. No dependency has been installed, upgraded or removed
to create this inventory.

Refresh the inventory explicitly from the chosen build interpreter:

```sh
python packaging/build_identity.py --capture-environment packaging/environments/macos-arm64-py311-captured.json
```

The current capture identifies a pre-existing inconsistency: `tensorflow-macos`
2.15.0 declares Keras `>=2.15,<2.16`, but Keras 3.12.0 is installed. The constraints
are an audit/reproduction input, **not a claim that pip can solve this existing
environment**. Resolving it needs a separate ML compatibility decision and
validation with supported models; do not silently downgrade Keras or upgrade
TensorFlow. Dev-test tools are not part of runtime closure; the local pytest
version can also differ from `.[dev]` (currently pytest 7.4.0 vs declared >=8).

After compatibility issues are resolved, reproduce the approved environment in
an isolated interpreter on the matching OS/architecture/Python, compare the
manifest, and run scientific regressions and the packaged self-test before
distribution. A platform-specific capture should not be used as a universal
Windows/Linux lockfile.

## CI scope

Pushes and pull requests run the established core/I/O/packaging/recovery tests
plus a curated scientific and recording-reference workflow suite. CI preserves
its resolved dependency inventory as an artifact. The full analysis, GUI and
remaining regression shards remain available through workflow_dispatch;
historical binary-oracle expectations have not been rewritten or hidden.
