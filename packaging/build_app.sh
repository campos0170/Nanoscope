#!/usr/bin/env bash
#
# Build NanoScope.app (Apple Silicon, unsigned + ad-hoc signed) and a distributable DMG.
#
#   ./packaging/build_app.sh              full build: icon -> app -> ad-hoc sign -> DMG
#   ./packaging/build_app.sh --no-dmg     stop after the signed .app (fast iteration)
#   ./packaging/build_app.sh --keep-cache reuse PyInstaller's analysis cache (much faster rebuild)
#
# That is the whole procedure. Nothing else is needed and nothing is done by hand.
#
set -euo pipefail

APP_NAME="NanoScope"
PYTHON="${NANOSCOPE_PYTHON:-/opt/anaconda3/bin/python3}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
DIST="$ROOT/dist"
WORK="$ROOT/build"
APP="$DIST/$APP_NAME.app"

MAKE_DMG=1
CLEAN_FLAG="--clean"
for arg in "$@"; do
  case "$arg" in
    --no-dmg)     MAKE_DMG=0 ;;
    --keep-cache) CLEAN_FLAG="" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- preflight --------------------------------------------------------------------------------
say "Preflight"
[[ "$(uname -m)" == "arm64" ]] || { echo "This build is Apple Silicon only; you are on $(uname -m)." >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "Python not found at $PYTHON (override with NANOSCOPE_PYTHON=)" >&2; exit 1; }
APP_VERSION="$("$PYTHON" "$HERE/build_identity.py" --version)"
DMG="$DIST/$APP_NAME-$APP_VERSION-arm64.dmg"
"$PYTHON" -c "import PyInstaller" 2>/dev/null || { echo "PyInstaller missing: $PYTHON -m pip install 'pyinstaller>=6.11'" >&2; exit 1; }
# The spec calls copy_metadata("nanoscope"), which needs the package's dist-info to exist.
"$PYTHON" -c "import importlib.metadata as m; m.distribution('nanoscope')" 2>/dev/null \
  || { echo "nanoscope is not installed into $PYTHON. Run: $PYTHON -m pip install -e '$ROOT'" >&2; exit 1; }
echo "python:  $($PYTHON -c 'import sys;print(sys.version.split()[0])')  ($PYTHON)"
echo "arch:    $(uname -m)"
echo "macOS:   $(sw_vers -productVersion)"

# --- icon -------------------------------------------------------------------------------------
say "Generating icon"
"$PYTHON" "$HERE/make_icon.py"

# --- freeze -----------------------------------------------------------------------------------
say "Freezing with PyInstaller (several minutes; TensorFlow is ~1 GB of it)"
rm -rf "$APP" "$DIST/$APP_NAME" "$DMG"
# Not `python -m PyInstaller`: this conda env carries the obsolete `pathlib` backport's dist-info,
# which PyInstaller refuses to run alongside. run_pyinstaller.py masks it for the preflight check
# only, instead of uninstalling a package out of the owner's analysis environment. See that file.
"$PYTHON" "$HERE/run_pyinstaller.py" \
  --noconfirm $CLEAN_FLAG \
  --distpath "$DIST" \
  --workpath "$WORK" \
  "$HERE/$APP_NAME.spec"

[[ -d "$APP" ]] || { echo "build produced no $APP" >&2; exit 1; }

# --- sanitise + verify bundle integrity -------------------------------------------------------
# Two failures that are invisible until launch: a Qt5 platform plugin smuggled in by Anaconda's
# qt.conf, and dangling framework symlinks left behind by the spec's pruning (which make
# `codesign --deep` fail the whole bundle). This step also removes build-host LC_RPATH entries
# and direct_url.json installer provenance before the final signature. See verify_bundle.py.
say "Sanitising and verifying bundle integrity"
"$PYTHON" "$HERE/verify_bundle.py" "$APP"

# --- ad-hoc sign ------------------------------------------------------------------------------
# NOT optional. arm64 refuses to execute an unsigned Mach-O at all: without at least an ad-hoc
# signature the app is killed with SIGKILL ("code signature invalid") the moment it is launched.
# PyInstaller ad-hoc signs the individual binaries it writes, but rewriting the bundle (icon,
# Info.plist, our pruning) invalidates the outer seal, so re-sign the whole thing here.
say "Ad-hoc signing"
codesign --force --deep --sign - --timestamp=none "$APP"
codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/    /'
echo "    signature: $(codesign -dv "$APP" 2>&1 | grep -i 'Signature=' || echo 'ad-hoc')"

# --- report -----------------------------------------------------------------------------------
say "Size"
du -sh "$APP" | sed 's/^/    /'
echo "    largest components:"
du -sm "$APP/Contents/Frameworks"/* 2>/dev/null | sort -rn | head -8 | sed 's/^/      /'

# --- dmg --------------------------------------------------------------------------------------
if [[ "$MAKE_DMG" == "1" ]]; then
  say "Building DMG"
  DMG_TMP_ROOT="$(mktemp -d)"
  cleanup_dmg_workspace() {
    rm -rf -- "$DMG_TMP_ROOT"
  }
  trap cleanup_dmg_workspace EXIT

  STAGE="$DMG_TMP_ROOT/$APP_NAME"
  mkdir -p "$STAGE"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  cp "$HERE/INSTALL.txt" "$STAGE/" 2>/dev/null || true

  # `hdiutil create -srcfolder` can fail with "Device not configured" in
  # mount-restricted build environments. Prefer it because it is the standard
  # path, then fall back to the no-mount makehybrid + convert pipeline.
  if ! hdiutil create -volname "$APP_NAME $APP_VERSION" \
    -srcfolder "$STAGE" -ov -format UDZO -imagekey zlib-level=9 "$DMG" >/dev/null; then
    echo "    hdiutil create failed; retrying with no-mount hybrid fallback" >&2
    rm -f -- "$DMG"
    HYBRID_BASE="$DMG_TMP_ROOT/$APP_NAME-hybrid"
    hdiutil makehybrid -hfs -hfs-volume-name "$APP_NAME $APP_VERSION" \
      -o "$HYBRID_BASE" "$STAGE" >/dev/null
    hdiutil convert "$HYBRID_BASE.dmg" -format UDZO -imagekey zlib-level=9 \
      -o "$DMG" >/dev/null
  fi

  cleanup_dmg_workspace
  trap - EXIT
  du -sh "$DMG" | sed 's/^/    /'
fi

say "Done"
echo "    app: $APP"
[[ "$MAKE_DMG" == "1" ]] && echo "    dmg: $DMG"
echo "    log at runtime: ~/Library/Logs/$APP_NAME/nanoscope.log"
