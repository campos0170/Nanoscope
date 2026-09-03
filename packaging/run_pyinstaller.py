"""Invoke PyInstaller with the obsolete `pathlib` backport masked from its preflight check.

WHY THIS EXISTS: this Anaconda install carries `pathlib 1.0.1` — the pre-3.4 backport of the
standard library module — as a leftover dependency. PyInstaller refuses to run at all when it sees
that distribution ("obsolete backport ... incompatible with PyInstaller") and tells you to
`conda remove pathlib`.

Doing that would be a permanent, unasked-for change to the owner's working analysis environment,
so instead we mask the *distribution metadata* for the duration of PyInstaller's preflight check
and nothing else. This is safe here for a specific reason: site-packages comes AFTER the standard
library on sys.path, so `import pathlib` in this interpreter already resolves to the stdlib module
— the backport's pathlib.py is dead code that nothing can reach, and only its .dist-info folder
is left to trip the check. The build verifies this (see `_assert_stdlib_pathlib`).

If the owner ever does want to clean it up for real:  pip uninstall pathlib
"""
from __future__ import annotations

import importlib.metadata as _md
import pathlib
import sys
import sysconfig

# Distributions that are obsolete stdlib backports, present in this env, and provably shadowed.
_MASKED = {"pathlib"}


def _assert_stdlib_pathlib() -> None:
    """Refuse to build if `import pathlib` is NOT the standard library one.

    If the backport ever did win the sys.path race, masking the check would hide a real problem
    and produce a bundle with a 2014-era pathlib in it. Fail loudly instead.
    """
    stdlib = pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()
    resolved = pathlib.Path(pathlib.__file__).resolve()
    if stdlib not in resolved.parents:
        raise SystemExit(
            f"REFUSING TO BUILD: `import pathlib` resolves to {resolved}, not the standard "
            f"library at {stdlib}. Remove the backport with `pip uninstall pathlib` and retry."
        )


def main(argv: list[str]) -> int:
    _assert_stdlib_pathlib()

    from PyInstaller import compat

    _real_check = compat.check_requirements

    def _check_with_mask() -> None:
        # compat.check_requirements does `from importlib.metadata import distribution` inside the
        # function body, so patching the module attribute here is enough and is undone right after.
        real_distribution = _md.distribution

        def distribution(name):  # noqa: ANN001, ANN202
            if name in _MASKED:
                raise _md.PackageNotFoundError(name)
            return real_distribution(name)

        _md.distribution = distribution
        try:
            _real_check()
        finally:
            _md.distribution = real_distribution

    compat.check_requirements = _check_with_mask

    from PyInstaller.__main__ import run

    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
