"""Keep TensorFlow's Python extension and native dependents on one loaded image."""
from pathlib import Path


_EXTENSION = "tensorflow/python/_pywrap_tensorflow_internal.so"
_ALIAS = "_pywrap_tensorflow_internal.so"


def deduplicate_tensorflow_runtime(toc):
    """Replace the duplicate top-level dependency with a PyInstaller symlink.

    Dependency analysis can resolve a symlinked environment's extension through
    its real path and collect it again as a top-level BINARY. dyld then loads two
    copies and TensorFlow aborts import on duplicate op registration. Keep the
    extension at its Python import path and let native dependents share it.
    Refuse different source files instead of silently mixing TensorFlow builds.
    """
    entries = list(toc)
    canonical = [entry for entry in entries if entry[0] == _EXTENSION]
    aliases = [entry for entry in entries if entry[0] == _ALIAS]
    if not aliases:
        return entries
    if len(canonical) != 1 or canonical[0][2] != "EXTENSION" or len(aliases) != 1:
        raise ValueError("Ambiguous TensorFlow native extension in bundle analysis")
    alias = aliases[0]
    if alias[2] == "SYMLINK":
        if alias[1] != _EXTENSION:
            raise ValueError("TensorFlow native alias points to an unexpected target")
        return entries
    if alias[2] != "BINARY" or not Path(alias[1]).samefile(canonical[0][1]):
        raise ValueError("TensorFlow native dependency and Python extension differ")
    return [(entry[0], _EXTENSION, "SYMLINK") if entry == alias else entry
            for entry in entries]
