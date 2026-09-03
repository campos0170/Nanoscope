"""PC-C6: ONE definition of the conditioned trace, and a census that says so out loud.

``conditioning.conditioned_trace`` was landed as a *copy* of
``AnalysisPanel._conditioned_from_form``, not as its replacement: two function bodies, one
array, bit-identical today and free to drift tomorrow — which is the exact failure this
codebase organises against. The panel's half belongs to another change, so what is held
here is the half ``analysis/`` owns:

  - the CENSUS. Three functions in the package apply
    ``lowpass(preprocess_for_detection(...))`` and the rule is one builder per INPUT:
    ``conditioned_trace`` from the live form's params, ``modules.detection_signal`` from a
    finished detection's recorded provenance. ``_conditioned_from_form`` takes the same
    input as the first and is therefore the duplicate. A fourth builder fails here.
  - the duplicate is held to the definition over a RANDOMIZED chain matrix, not only the
    four chains the landing test hashed — drift usually starts in a stage nobody pinned.
  - the module docstring says which of those things is true TODAY, and this file fails on
    the day the delegation lands so the sentence gets rewritten instead of becoming a
    fossil. That failure is the handover, not a regression.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (the panel cross-check imports gui). --------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import ast
import hashlib
import inspect
import pathlib

import numpy as np
import pytest

from nanoscope.analysis import conditioning
from nanoscope.analysis.conditioning import conditioned_trace
from nanoscope.gui.analysis_panel import AnalysisPanel

FS = 50_000.0
N = 40_000

#: The census, as of 2026-09-02. ``qualname`` per module file.
DEFINITION = ("nanoscope/analysis/conditioning.py", "conditioned_trace")
FROM_PROVENANCE = ("nanoscope/analysis/modules.py", "detection_signal")
#: WAS the census's third entry until 2026-09-02, when the queued delegation landed:
#: ``AnalysisPanel._conditioned_from_form`` is now ``return conditioned_trace(raw, fs,
#: params)`` and builds no chain of its own. Kept as a NAME so the test below can say
#: which function must never grow a body again.
THE_FORMER_DUPLICATE = ("nanoscope/gui/analysis_panel.py",
                        "AnalysisPanel._conditioned_from_form")

_PKG = pathlib.Path(conditioning.__file__).resolve().parents[1]


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _called_names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def _chain_builders() -> set[tuple[str, str]]:
    """Every function in the package that applies the two-step chain ITSELF.

    Source-level, not runtime: a duplicate is a second BODY, and a body that has been
    replaced by a call to :func:`conditioned_trace` no longer shows up here — which is
    precisely the signal this file is watching for.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG.parent).as_posix()
        stack: list[tuple[ast.AST, str]] = [
            (ast.parse(path.read_text(encoding="utf-8")), "")]
        while stack:
            node, prefix = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = _called_names(child)
                    if {"preprocess_for_detection", "lowpass"} <= names:
                        found.add((rel, f"{prefix}{child.name}"))
                    stack.append((child, f"{prefix}{child.name}."))
                elif isinstance(child, ast.ClassDef):
                    stack.append((child, f"{prefix}{child.name}."))
    return found


# ================================================================================================
# 1. the census
# ================================================================================================
def test_the_only_builders_of_the_chain_are_the_two_inputs():
    """A FOURTH place that conditions a trace by hand is the failure this pins.

    The rule is one builder per input, not one builder full stop: the live form and a
    finished detection's provenance are genuinely different questions (the second has to
    tolerate a legacy field the first can refuse). What is NOT allowed is a second answer
    to the same question — which is what the panel's copy is."""
    assert _chain_builders() == {DEFINITION, FROM_PROVENANCE}, (
        "the census moved. Until 2026-09-02 it held a third entry, "
        f"{THE_FORMER_DUPLICATE[1]}, which was a bit-identical copy of the definition; it "
        "now delegates. Two inputs, two builders — a third is a second answer to one of "
        "them.")


def test_the_panel_delegates_rather_than_keeping_a_body_of_its_own():
    """The delegation, asserted on the source of the method that used to be the copy.

    This test used to be the OTHER way round: it read the panel's body, found its own
    two calls, and failed the day the delegation landed so that the caption describing
    the arrangement would be rewritten rather than left standing. That day is 2026-09-02
    and this is the rewrite — what it pins now is that the body does not come back.

    ONE SENTENCE IS STILL OWED, and it is not in a file this change may touch:
    ``conditioning.py``'s module docstring still calls the panel "STILL a second copy"
    and names the delegation as queued. That paragraph is now describing last month's
    arrangement; its own module's owner has to rewrite it. Asserting on it here would
    only add a red test to a tree where the code is right, so the debt is written down
    instead — see the wiring report for 2026-09-02.
    """
    panel_src = inspect.getsource(AnalysisPanel._conditioned_from_form)
    assert "conditioned_trace(" in panel_src, (
        "AnalysisPanel._conditioned_from_form grew a body again. It is the panel's own "
        "door and it may forward, but the array has ONE definition: conditioning."
        "conditioned_trace.")
    body = panel_src.split('"""')[-1]          # past the docstring, which QUOTES the chain
    assert "preprocess_for_detection(" not in body, \
        "the panel is building the chain itself beside the call that defines it"
    assert "verbatim move" not in (conditioning.__doc__ or ""), \
        "the panel's body was never moved here; it was copied, and then delegated"


# ================================================================================================
# 2. the duplicate, held to the definition
# ================================================================================================
def _random_chains(n: int = 24) -> list[dict]:
    """Chains sampled across every stage, so drift in an unpinned one is caught.

    The landing test hashed four hand-written chains. Four chains pin four chains; a
    stage nobody exercised is where two copies quietly stop agreeing."""
    rng = np.random.default_rng(20260902)
    out: list[dict] = [{}]
    for _ in range(n - 1):
        p: dict = {}
        if rng.random() < 0.5:
            p.update(notch_enabled=True, notch_freq_hz=float(rng.choice([50.0, 60.0])),
                     notch_q=float(rng.choice([20.0, 30.0])),
                     notch_harmonics=int(rng.integers(1, 4)))
        if rng.random() < 0.4:
            p.update(median_enabled=True, median_window_s=float(rng.choice([1e-3, 5e-3])))
        if rng.random() < 0.4:
            p.update(hp_enabled=True, hp_cutoff_hz=float(rng.choice([1.0, 20.0])))
        if rng.random() < 0.5:
            p.update(detrend_enabled=True,
                     detrend_method=str(rng.choice(["block_median", "block_mean"])),
                     detrend_window_s=float(rng.choice([0.02, 0.05, 0.2])))
        if rng.random() < 0.3:
            p.update(rmean_enabled=True, rmean_window_s=float(rng.choice([1e-3, 1e-2])))
        if rng.random() < 0.3:
            p.update(savgol_enabled=True, savgol_window_s=float(rng.choice([1e-3, 5e-3])))
        if rng.random() < 0.25:
            p.update(normalize_enabled=True,
                     normalize_mode=str(rng.choice(["zscore", "median_mad"])))
        if rng.random() < 0.7:
            p.update(filter_cutoff_hz=float(rng.choice([2000.0, 5000.0, 10000.0])),
                     filter_order=int(rng.choice([4, 8])),
                     filter_kind=str(rng.choice(["bessel", "butter"])))
        out.append(p)
    return out


def test_the_panels_copy_is_the_same_array_across_a_randomized_chain_matrix():
    """Identical BYTES, not merely close: a preview and a measurement "on what is on
    screen" that differ by one filter design are two traces nobody can tell apart by eye.

    Skipped, not failed, once the panel delegates — at that point there is one body and
    nothing left to cross-check."""
    if "conditioned_trace(" in inspect.getsource(AnalysisPanel._conditioned_from_form):
        pytest.skip("the panel delegates; there is no second body to compare")

    rng = np.random.default_rng(7)
    raw = (100.0 + 1.5 * rng.normal(size=N)).astype(np.float32)
    raw[15_000:15_500] -= np.float32(35.0)

    checked = 0
    for params in _random_chains():
        try:
            mine = conditioned_trace(raw, FS, dict(params))
        except Exception as exc:  # noqa: BLE001 — a kernel refusal must refuse both ways
            with pytest.raises(type(exc)):
                AnalysisPanel._conditioned_from_form(raw, FS, dict(params))
            continue
        theirs = AnalysisPanel._conditioned_from_form(raw, FS, dict(params))
        assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
        assert _digest(mine) == _digest(theirs), f"the two copies disagree on {params}"
        checked += 1
    assert checked >= 15, "the matrix degenerated; almost nothing was actually compared"
