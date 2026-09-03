"""Compatibility shims must import at both ends of NanoScope's NumPy range."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_COMPAT = Path(__file__).parents[2] / "nanoscope" / "analysis" / "_compat.py"


def _load_with_numpy(monkeypatch, fake_numpy: types.ModuleType):
    """Execute a fresh copy so the test observes attribute access during import."""
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    spec = importlib.util.spec_from_file_location("_nanoscope_compat_probe", _COMPAT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_numpy_does_not_eagerly_read_the_removed_legacy_name(monkeypatch):
    """Having ``trapezoid`` must make the missing ``trapz`` branch unreachable.

    ``getattr(np, "trapezoid", np.trapz)`` fails this test because Python evaluates
    ``np.trapz`` before calling ``getattr``, even though the preferred name exists.
    """
    preferred = object()
    fake = types.ModuleType("numpy")
    fake.trapezoid = preferred

    def missing(name: str):
        if name == "trapz":
            raise AssertionError("the legacy fallback was evaluated eagerly")
        raise AttributeError(name)

    fake.__getattr__ = missing
    assert _load_with_numpy(monkeypatch, fake).trapezoid is preferred


def test_numpy_126_uses_the_legacy_name_when_needed(monkeypatch):
    legacy = object()
    fake = types.ModuleType("numpy")
    fake.trapz = legacy
    assert _load_with_numpy(monkeypatch, fake).trapezoid is legacy
