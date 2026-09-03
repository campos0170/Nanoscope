"""Signal-processing / DSP layer for nanoscope (ported from the standalone
``~/nanoscope`` package, re-seamed onto the neo-based data model).

This package is deliberately independent of Qt and of ``nanoscope.core.session``.
DSP kernels are signal-first: they take ``(data, fs, ...)`` numpy arrays, not a
``Recording`` object. The ``AnalysisModule`` adapters that plug into
``nanoscope.core.analysis`` live in :mod:`nanoscope.analysis.modules`.
"""

from __future__ import annotations

from ._compat import trapezoid

__all__ = ["trapezoid", "register_all"]


def register_all() -> None:
    """Register every ported :class:`AnalysisModule` into ``ANALYSIS_REGISTRY``.

    Idempotent-ish: calling twice raises via ``register_module`` (duplicate id is
    a programming error), so callers should invoke it exactly once at app start /
    once per test session. The concrete modules land in :mod:`.modules` in a later
    phase; guard the import so this package is usable before that module exists.
    """
    try:
        from . import modules  # noqa: F401  (Phase 5 — not present yet)
    except ImportError:
        return
    register = getattr(modules, "register_all", None)
    if callable(register):
        register()
