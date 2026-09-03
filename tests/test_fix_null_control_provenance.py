"""H1 — the Null control must print a number that survives being pressed again.

THE DEFECT, measured through the real GUI on the owner's -700 mV apo recording
(``500 pM Apo ferritin_1 x PBS_22 nm_0005_Process_-700 mV.abf``, channel 2, 50 kHz),
one 1164-event detection, five clicks of *Null control…* with nothing else touched::

    run   readout                                        per-surrogate counts
    1     estimated false-positive rate: 17.0%           233, 266,  95,  65, 331
    2     estimated false-positive rate: 24.0%           377,  55, 643, 117, 206
    3     estimated false-positive rate: 20.3%           119, 258,  68, 363, 376
    4     estimated false-positive rate: 13.8%            56, 118,  83, 159, 385
    5     estimated false-positive rate: 19.0%            80, 386,  42, 466, 130

A 10.2-point swing on a number a reader takes for a property of the pore, and the
label's tooltip was empty on every one of them — the single path in this panel that
puts a percentage on screen was the single path that said nothing about where it came
from, while the *refusal* path beside it explained itself in full.

THE INVARIANT UNDER TEST — both halves, because either alone is a trap:

* **reproducible** — the run is seeded from what is being measured, so pressing the
  button again returns the same counts, and so does an app restart or a close/reopen;
* **honest** — the surrogates disagree with each other by ~15x, so a seeded mean
  printed bare is a *repeatable meaningless number*. The readout carries the range the
  surrogates actually spanned, and the tooltip carries the counts, the seed and the
  event-free count.

A result that carries no seed must be labelled NOT REPRODUCIBLE rather than printed as
if it were a measurement — that is the invariant, not the call site: it holds for any
caller, including one that forgets ``rng=`` tomorrow.

WHICH TESTS ARE REVERT DETECTORS. Every test below whose name starts with
``test_h1_`` fails against an out-of-tree revert of the three hunks (the seeded call in
``AnalysisPanel._on_null_requested``, the seed derivation ``AnalysisPanel._null_seed``,
and the spread + provenance in ``StatsPanel.set_fdr``). ``test_h1_refusal_branch_*``
is the exception and says so in its own docstring: it asserts what must NOT have moved
(the -600 mV / -500 mV refusal path), so it passes both before and after by design.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, must precede any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 60_000
_EVENTS = [(6_000, 6_400), (18_000, 18_350), (33_000, 33_500), (47_000, 47_300)]


# ================================================================================================
# WHERE DID `nanoscope` COME FROM?
#
# A revert stub with the wrong signature once manufactured four fake failures in this
# project and was believed. So before anything else: every module these tests make a
# claim about must be the one in the tree under test, imported from source, and the
# three files this fix touches must all resolve out of the SAME package directory --
# a half-reverted tree (panels.py from one place, analysis_panel.py from another) is
# exactly the shape that produces a convincing lie.
# ================================================================================================

def test_h1_the_modules_under_test_resolve_from_the_tree_under_test():
    from nanoscope.analysis import truth
    from nanoscope.gui import analysis_panel, panels

    pkg = Path(nanoscope.__file__).resolve().parent
    for mod in (truth, analysis_panel, panels):
        got = Path(mod.__file__).resolve()
        assert got.parent.parent == pkg or got.parent == pkg, (
            f"{mod.__name__} resolved from {got}, outside the nanoscope package at {pkg}")
        assert got.suffix == ".py", f"{mod.__name__} is not source: {got}"

    # The three hunks live in two files; name them so a partial revert cannot hide.
    assert hasattr(analysis_panel.AnalysisPanel, "_null_seed"), (
        "AnalysisPanel._null_seed is missing — hunk 2 (seed derivation) is not in this tree")
    assert hasattr(panels.StatsPanel, "_fdr_spread"), (
        "StatsPanel._fdr_spread is missing — hunk 3 (spread + provenance) is not in this tree")
    assert hasattr(panels.StatsPanel, "_fdr_provenance"), (
        "StatsPanel._fdr_provenance is missing — hunk 3 (spread + provenance) is not in this tree")


# ================================================================================================
# builders — a real MainWindow over a small synthetic recording
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _win_with_doc(qtbot, tmp_path, name="h1.csv", seed=0):
    win = MainWindow()
    qtbot.addWidget(win)
    arr = _trace(seed)
    doc = _doc(tmp_path / name, arr)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    qtbot.wait(10)
    return win, win._analysis_panel, doc_id


def _quit(win):
    win._analysis_controller.abort()
    win._controller.abort()


#: What a `null_control` run looks like from the panel's side. The counts are the
#: -700 mV run-1 counts, so the arithmetic the panel does here is the arithmetic it did
#: on the owner's recording.
_REAL_SHAPE = {
    "n_real": 1164, "n_event_free": 151,
    "per_surrogate": [233, 266, 95, 65, 331],
    "n_surrogate_mean": 198.0, "n_surrogate_std": 113.0,
    "estimated_fdr": 198.0 / 1164.0, "notes": (),
}


def _record_null_calls(monkeypatch) -> list[dict]:
    """Replace `truth.null_control` with a recorder, keeping its keyword contract.

    The panel's job does ``from ..analysis.truth import null_control`` at call time on
    the worker thread, so patching the module attribute is what the worker picks up.
    The stub takes ``rng`` and ``cancel`` by keyword exactly as the real one does: a
    stub with a laxer signature is how a fake failure gets manufactured, so this one
    would raise if the panel called it the old way with a positional surprise.
    """
    from nanoscope.analysis import truth

    calls: list[dict] = []

    def recorder(rec, *, detect_kwargs=None, n_surrogates=5, rng=None, cancel=None):
        calls.append({"rng": rng, "n_surrogates": n_surrogates,
                      "detect_kwargs": dict(detect_kwargs or {})})
        out = dict(_REAL_SHAPE)
        out["per_surrogate"] = list(_REAL_SHAPE["per_surrogate"])
        return out

    monkeypatch.setattr(truth, "null_control", recorder)
    return calls


def _click_null(win, panel, qtbot):
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.detection_panel.null_button.click()
    qtbot.wait(30)


# ================================================================================================
# hunk 1 + 2 — the run is seeded, and the seed travels with the answer
# ================================================================================================

def test_h1_the_null_control_run_is_seeded_and_the_seed_travels_with_the_answer(
        qapp, qtbot, tmp_path, monkeypatch):
    """The click must hand `null_control` a seed, and the answer must carry it.

    Pre-fix, ``analysis_panel.py`` called ``null_control(build(), detect_kwargs=kw)``
    with no ``rng=`` — so ``rng`` arrived as None, the Generator came from OS entropy,
    and nothing downstream could even say what draw produced the percentage.
    """
    calls = _record_null_calls(monkeypatch)
    win, panel, doc_id = _win_with_doc(qtbot, tmp_path)
    _click_null(win, panel, qtbot)

    assert len(calls) == 1, f"the click did not reach null_control: {calls!r}"
    rng = calls[0]["rng"]
    assert rng is not None, (
        "Null control was launched with rng=None — OS entropy, unrepeatable; this is the "
        "defect, five clicks printed 17.0/24.0/20.3/13.8/19.0 % on the owner's -700 mV file")
    assert isinstance(rng, (int, np.integer)), (
        f"the seed must be an integer a human can read and re-enter, got {rng!r}")

    stored = panel._fdr[(doc_id, 0)]
    assert stored["seed"] == int(rng), (
        "the seed did not travel with the answer, so the panel cannot say what produced "
        f"the number: {stored.get('seed')!r} vs {rng!r}")
    _quit(win)


def test_h1_pressing_the_button_again_reruns_the_same_null(qapp, qtbot, tmp_path,
                                                           monkeypatch):
    """Same recording, same settings, second press -> the same seed. This IS the claim."""
    calls = _record_null_calls(monkeypatch)
    win, panel, doc_id = _win_with_doc(qtbot, tmp_path)
    _click_null(win, panel, qtbot)
    _click_null(win, panel, qtbot)

    assert len(calls) == 2
    # ``None == None`` is not agreement, it is two unrepeatable draws that happen to be
    # described by the same word. Pin the seed's existence before comparing the two.
    assert calls[0]["rng"] is not None and calls[1]["rng"] is not None, (
        f"neither press named a draw: {calls[0]['rng']!r}, {calls[1]['rng']!r}")
    assert calls[0]["rng"] == calls[1]["rng"], (
        f"two presses drew two different nulls: {calls[0]['rng']} vs {calls[1]['rng']}")
    # And the panel prints the same thing both times, which is what the user checks.
    assert panel._fdr[(doc_id, 0)]["seed"] == calls[0]["rng"]
    _quit(win)


def test_h1_changing_a_detect_setting_draws_an_independent_null(qapp, qtbot, tmp_path,
                                                                monkeypatch):
    """Reproducibility must not become "one phase randomisation for the whole app".

    A constant seed would satisfy "press it again" and quietly replay the same draw on
    every recording and every setting. The seed is a function of what is measured, so a
    changed setting is a genuinely independent null.
    """
    calls = _record_null_calls(monkeypatch)
    win, panel, doc_id = _win_with_doc(qtbot, tmp_path)
    _click_null(win, panel, qtbot)
    panel.detection_panel.k_in.setValue(panel.detection_panel.k_in.value() + 1.0)
    qtbot.wait(10)
    _click_null(win, panel, qtbot)

    assert len(calls) == 2
    assert calls[0]["detect_kwargs"] != calls[1]["detect_kwargs"], "fixture broken"
    assert calls[0]["rng"] != calls[1]["rng"], (
        "a different threshold replayed the same surrogate draw")
    _quit(win)


def test_h1_the_seed_ignores_the_per_load_token_and_the_process(qapp):
    """Reproducible tomorrow, not just this afternoon.

    A ``doc_id`` carries a per-LOAD suffix, so seeding off it would re-seed the same
    file on every reopen. And :func:`hash` is salted per process, so seeding off it
    would re-seed on every app start. Both would print a different published number for
    the same recording and the same settings.
    """
    from nanoscope.gui.analysis_panel import AnalysisPanel

    kw = {"k_in": 5.0, "min_dwell_s": 5e-4}
    a = AnalysisPanel._null_seed(("/data/pore.abf#load-28259b34622c", 2), kw)
    b = AnalysisPanel._null_seed(("/data/pore.abf#load-727e359b5d5a", 2), kw)
    assert a == b, f"closing and reopening the file re-seeded the null: {a} vs {b}"

    c = AnalysisPanel._null_seed(("/data/pore.abf#load-28259b34622c", 3), kw)
    assert a != c, "two channels of one file share a draw"

    # Same seed out of a FRESH interpreter — PYTHONHASHSEED randomisation must not reach it.
    code = (
        "import os; os.environ.setdefault('QT_QPA_PLATFORM','offscreen');"
        "import nanoscope;"
        "from nanoscope.gui.analysis_panel import AnalysisPanel as P;"
        "print(P._null_seed(('/data/pore.abf#load-28259b34622c', 2),"
        " {'k_in': 5.0, 'min_dwell_s': 5e-4}))"
    )
    env = dict(os.environ, PYTHONHASHSEED="1")
    root = str(Path(nanoscope.__file__).resolve().parent.parent)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=root, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert int(out.stdout.strip().splitlines()[-1]) == a, (
        f"a restart changed the seed: {out.stdout.strip()} vs {a}")


# ================================================================================================
# hunk 3 — the branch that prints a number explains itself
# ================================================================================================

def test_h1_the_printed_rate_carries_the_spread_the_surrogates_actually_had(qapp):
    """A bare mean of 233/266/95/65/331 hides that the draws span 5x.

    Pre-fix the readout was ``estimated false-positive rate: 17.0% (AAFT, 5 surrogates)``
    and nothing else. 65/1164 = 5.6 % and 331/1164 = 28.4 % are the same experiment.
    """
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr(dict(_REAL_SHAPE, seed=2603058456))
    text = panel.fdr_label.text()

    assert "17.0%" in text, f"the point estimate is gone: {text!r}"
    assert "5.6" in text and "28.4" in text, (
        f"the readout does not say how far the surrogates disagreed: {text!r}")


def test_h1_the_printing_branch_has_a_tooltip_and_it_names_the_seed(qapp):
    """``panels.py`` cleared the tooltip on exactly the finite-FDR branch.

    The refusal branch beside it explains itself; the one branch that puts a number on
    screen said nothing about it. This is that hunk.
    """
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr(dict(_REAL_SHAPE, seed=2603058456))
    tip = panel.fdr_label.toolTip()

    assert tip, "the branch that prints a percentage still explains nothing"
    assert "2603058456" in tip, f"the seed is not on screen anywhere: {tip!r}"
    for count in _REAL_SHAPE["per_surrogate"]:
        assert str(count) in tip, f"per-surrogate count {count} missing from {tip!r}"
    assert "151" in tip, "the event-free count the refusal would have quoted is missing"
    assert "1164" in tip, "the denominator is missing"


def test_h1_a_result_with_no_seed_is_labelled_not_reproducible(qapp):
    """The invariant is enforced where the number is PRINTED, not only where it is run.

    Any caller can forget ``rng=`` tomorrow; a stored result from an older session
    already has. Such a number is still shown — and is marked, in the same place the
    seed would have been, as one that will not come back the same.
    """
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr(dict(_REAL_SHAPE))          # no "seed" key at all
    tip = panel.fdr_label.toolTip()

    assert "17.0%" in panel.fdr_label.text(), "the number was suppressed, not caveated"
    assert "NOT REPRODUCIBLE" in tip, (
        f"an unseeded percentage was printed with no warning: {tip!r}")
    assert "seed =" not in tip, "it claims a seed it does not have"


def test_h1_a_degenerate_spread_is_not_printed_twice(qapp):
    """When every surrogate lands on the same percentage the mean IS the range.

    Printing "5.0% — individual surrogates gave 5.0–5.0%" reads as a second, different
    claim. The clause is omitted, and the tooltip still carries the counts.
    """
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr({"n_real": 100, "n_event_free": 0, "per_surrogate": [5, 5, 5],
                   "n_surrogate_mean": 5.0, "estimated_fdr": 0.05, "notes": (),
                   "seed": 11})
    text = panel.fdr_label.text()
    assert "5.0%" in text
    assert "individual surrogates" not in text, f"redundant range printed: {text!r}"
    assert "seed = 11" in panel.fdr_label.toolTip()


def test_h1_a_legacy_result_without_per_surrogate_counts_still_prints(qapp):
    """`set_fdr` is fed stored dicts from older sessions; none of this may raise."""
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": 0.42, "n_surrogates": 10})
    assert "42.0%" in panel.fdr_label.text()
    assert "NOT REPRODUCIBLE" in panel.fdr_label.toolTip()

    panel.set_fdr({"estimated_fdr": 0.05, "n_real": 0, "per_surrogate": [1, 2, 3],
                   "n_event_free": 1, "notes": (), "seed": 3})
    assert "5.0%" in panel.fdr_label.text()          # no ZeroDivisionError on n_real=0
    assert "individual surrogates" not in panel.fdr_label.text()

    # A per_surrogate stored as a numpy array must render, not raise: `arr or ()` is
    # "truth value of an array ... is ambiguous", and this readout is reached from
    # `_sync_surrogate_readouts` on every selection change.
    panel.set_fdr({"estimated_fdr": 0.2, "n_real": 100, "n_event_free": 3,
                   "per_surrogate": np.array([10, 20, 30]), "notes": (), "seed": 4})
    assert "20.0%" in panel.fdr_label.text()
    assert "10.0–30.0%" in panel.fdr_label.text()


# ================================================================================================
# WHAT MUST NOT HAVE MOVED — the -600 mV / -500 mV refusal path
#
# This one passes before AND after on purpose: it is the guard on the branch this fix
# was told not to touch. On the owner's -600 mV and -500 mV recordings the AAFT
# surrogates are not a null and `null_control` returns nan; that path must still print
# the event-free lower bound and put the reason in the tooltip, verbatim.
# ================================================================================================

def test_h1_refusal_branch_is_untouched_and_still_fires_verbatim(qapp):
    from nanoscope.gui.panels import StatsPanel

    reason = ("the AAFT surrogates produced 4040 detections where this recording's own "
              "event-free noise produces 56, so the surrogates are not the same noise")
    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": float("nan"), "n_real": 315, "n_event_free": 56,
                   "per_surrogate": [4040, 4038, 4042], "notes": (reason,), "seed": 7})

    text = panel.fdr_label.text()
    assert "nan" not in text
    assert text == ("false-positive rate NOT estimated — 56 detection(s) on this "
                    "recording's own event-free noise (a lower bound, measured with no "
                    "surrogate).")
    assert panel.fdr_label.toolTip() == reason, (
        "the refusal's tooltip is no longer its own note")
    assert "seed" not in panel.fdr_label.toolTip()
    assert "individual surrogates" not in text


def test_h1_none_still_resets_the_readout(qapp):
    """Repointing at a recording with no null control of its own says so, as before."""
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr(dict(_REAL_SHAPE, seed=1))
    panel.set_fdr(None)
    assert panel.fdr_label.text() == StatsPanel._NO_FDR_TEXT
    assert "17.0" not in panel.fdr_label.text()


# ================================================================================================
# the premise underneath the fix (passes before and after — it is the mechanism the fix
# uses, not the fix): a seeded `null_control` really is bit-reproducible.
# ================================================================================================

@pytest.mark.parametrize("seed", [0, 7])
def test_h1_premise_a_seeded_null_control_repeats_exactly(seed):
    from nanoscope.analysis.truth import null_control, synth_trace

    rec, _ = synth_trace(duration_s=4.0, n_events=60, noise_pA=15.0,
                         depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3), seed=5)
    kw = {"k_in": 5.0, "min_dwell_s": 1e-4}
    a = null_control(rec, detect_kwargs=kw, n_surrogates=3, rng=seed)
    b = null_control(rec, detect_kwargs=kw, n_surrogates=3, rng=seed)
    assert a["per_surrogate"] == b["per_surrogate"]
    assert a["n_event_free"] == b["n_event_free"]
    assert a["estimated_fdr"] == b["estimated_fdr"] or (
        np.isnan(a["estimated_fdr"]) and np.isnan(b["estimated_fdr"]))
