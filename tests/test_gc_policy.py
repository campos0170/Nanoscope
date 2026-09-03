"""Regression coverage for the suite-wide Qt wrapper collection policy."""

from __future__ import annotations

import gc


def test_ordinary_tests_run_with_cyclic_gc_disabled():
    assert not gc.isenabled(), (
        "tests/conftest.py must disable cyclic GC before importing Qt; otherwise PySide6 "
        "wrapper finalization can deadlock an ordinary full-suite run"
    )
