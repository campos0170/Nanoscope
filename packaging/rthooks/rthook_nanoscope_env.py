"""PyInstaller runtime hook — runs INSIDE the frozen app before any app module is imported.

Everything here is environment that must be set before the libraries that read it are imported.
Order is why this is a runtime hook and not code in the launcher: PyInstaller executes runtime
hooks before the entry script, so `os.environ` is already correct when TensorFlow / matplotlib /
Qt first look at it.

Nothing here touches QT_PLUGIN_PATH — PyInstaller's own PySide6 runtime hook owns that, and
`nanoscope/bootstrap.py` deliberately stands down when frozen (see the docstring there).
"""
import os
import sys


def _cache_root() -> str:
    """The per-user cache directory this platform actually expects.

    Windows has no ~/Library; writing there would create a literal "Library\\Caches" folder in
    the user's profile, and worse, %LOCALAPPDATA% is the one location a locked-down corporate
    profile reliably permits writes to. macOS keeps its own convention.
    """
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches")
    return os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")


def _writable_cache_dir(name: str) -> str:
    base = os.path.join(_cache_root(), "NanoScope", name)
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:  # read-only home / sandbox weirdness: fall back to a temp dir
        import tempfile

        base = os.path.join(tempfile.gettempdir(), "NanoScope", name)
        os.makedirs(base, exist_ok=True)
    return base


# --- matplotlib -------------------------------------------------------------------------------
# nanoscope.plot.figures only ever writes files (fig.savefig). The platform default backend is
# a GUI one on every OS this ships to ("macosx" on macOS, TkAgg/QtAgg on Windows) — each builds
# a native window per figure inside a process that already owns a Qt event loop, which is
# pointless and a real crash risk. Agg is the whole job. Set, not defaulted:
# an inherited MPLBACKEND from someone's shell must not decide what the shipped app does.
os.environ["MPLBACKEND"] = "Agg"
# First run builds the font cache. The bundle is read-only, so give matplotlib its own writable
# home instead of letting it fall back or collide with the user's dev ~/.matplotlib. setdefault,
# not assignment: someone who deliberately exports MPLCONFIGDIR gets to keep it.
os.environ.setdefault("MPLCONFIGDIR", _writable_cache_dir("matplotlib"))
# THIS HOOK DOES NOT GET THE LAST WORD, so record the value that should win. PyInstaller runs
# custom runtime hooks (this one) FIRST and its own afterwards, and because matplotlib is
# collected it always pulls in pyi_rth_mplconfig, whose whole body is
#     os.environ['MPLCONFIGDIR'] = secure_mkdtemp()   +   atexit.register(shutil.rmtree, ...)
# — an empty throwaway directory, deleted on exit. That is why the shipped app rebuilt the font
# cache (~10 s) on EVERY launch while this line looked correct: matplotlib never saw it, so
# nothing was ever written to the persistent directory and nothing could be read back from it.
# nanoscope_app.py runs after every runtime hook and restores MPLCONFIGDIR from this variable
# before anything imports matplotlib; see _restore_mpl_cache_dir() there.
os.environ["NANOSCOPE_MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]

# UMAP and PyNNDescent cache compiled kernels. Keep their cache outside the
# signed, read-only bundle, and respect an explicitly supplied testing cache.
os.environ.setdefault("NUMBA_CACHE_DIR", _writable_cache_dir("numba"))

# --- TensorFlow -------------------------------------------------------------------------------
# CPU IS FORCED (see nanoscope/analysis/ml_detect._import_tf): the Apple Metal pluggable device
# SIGABRTs on this mixed_float16 LSTM. ml_detect does the decisive tf.config.set_visible_devices
# call itself; these two are the belt-and-suspenders half, hoisted to before TF's C++ core loads.
# The bundle also ships NO libmetal_plugin.dylib (excluded in the spec), so there is nothing for
# TF to auto-register even if this were missed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # drop the INFO spam, keep warnings + errors

# --- stdio ------------------------------------------------------------------------------------
# A windowed .app is launched by LaunchServices with no controlling terminal. Depending on how it
# was started, sys.stdout/sys.stderr can be None — and TensorFlow, Qt and numpy all write to
# stderr. `print(..., file=None)` and library code doing sys.stderr.write would blow up. The
# launcher redirects the real fds to a log file; this is only the floor under that, for the window
# between hook time and launcher time.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name, None) is None:
        setattr(sys, _name, open(os.devnull, "w"))
