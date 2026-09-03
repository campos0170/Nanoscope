# Custom PyInstaller hooks

`NanoScope.spec` passes this directory as `hookspath`, so it must exist even though it is
currently empty — that is what this file is for.

Drop a `hook-<package>.py` here when a dependency needs collecting in a way PyInstaller's own
hooks get wrong. Nothing has needed one so far: the two build-time problems this project actually
hit are handled elsewhere, because neither is a per-package collection issue.

* **Anaconda's Qt5 plugins** — fixed by rewriting PyInstaller's Qt location table before
  `Analysis()` runs. See `../qt_prefix_fix.py`.
* **Over-collection** (VTK, HoloViz, huggingface_hub, unused Qt modules) — handled by `excludes`
  and `_prune_toc` in the spec, not by hooks.

A hook here would look like:

```python
# hook-somepackage.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("somepackage.plugins")
datas = collect_data_files("somepackage")
```

Verify any addition with `NanoScope.app/Contents/MacOS/NanoScope --self-test` — see `../README.md`.
