"""H1b — THE FALSE-POSITIVE RATE'S PROVENANCE MUST BE OWNED BY THE NUMBER IT DESCRIBES.

H1 (2026-08-22) made the Null control reproducible and gave the printed percentage a rich
provenance tooltip. It shipped three things wrong, all of them in *what the app tells the
user*, and all three were invisible to a test that builds ``StatsPanel`` by itself.

WHAT WAS MEASURED, wired ``MainWindow``, real event loop, offscreen Qt
=====================================================================

**1. The tooltip outlived its label and followed the user onto another recording.**
Two recordings A and B; *Null control…* run on A; then B selected (B has no null control
of its own, so ``AnalysisPanel._sync_surrogate_readouts`` pushes ``set_fdr(None)``)::

    label   'false-positive rate: not measured (run Null control)'
    tooltip 'The mean of 5 independent AAFT surrogates, over 4 real detections.
             Per-surrogate counts: 0, 0, 0, 0, 0.  ...  seed = 2029049619 — pressing
             "Null control…" again on THIS RECORDING with these settings reproduces
             these counts exactly.'

Every "this recording" in that tooltip is recording A, which is no longer on screen.
``set_fdr(None)`` cleared the label and returned without touching the tooltip.

**2. The seed was a function of the FILE PATH.** The owner's -700 mV apo recording
(md5 ``1a4d0c62d443fd4fd235f72fd48c0356``) copied byte-for-byte to three names, each
opened through ``MainWindow.open_paths`` and channel 2 selected, nothing else touched::

    nameA.abf  seed 1592734707
    nameB.abf  seed 1091228298
    nameC.abf  seed 1722390092

Three "reproducible" numbers for one recording, so the published figure belonged to a
scratch copy's *name* and not to the pore.

**3. The seed ignored the 8 detrend and 15 filter controls** — which are stripped before
the DETECTOR (they are not its kwargs) but are NOT stripped before the SURROGATE, because
``_surrogate_builder`` conditions the trace with the whole form. Same file, same channel,
through the real click::

    default                                     seed 1592734707
    notch toggled                               seed 1592734707
    + median + high-pass + savgol + detrend     seed 1592734707

so a difference the reader takes for "the notch changed the false-positive rate" was
mostly the phase randomisation the two runs shared.

THE INVARIANT UNDER TEST
========================
A printed number's provenance string is owned by the same lifecycle as the number — so
``StatsPanel`` has exactly ONE writer of that readout and it always writes both halves —
and the seed is a function of WHAT WAS MEASURED: the recording's bytes and every setting
that reaches the surrogate, never where the file happens to sit and never a subset.

WHICH TESTS ARE REVERT DETECTORS. Every test named ``test_fup_*`` below fails against an
out-of-tree revert of the four hunks, EXCEPT ``test_fup_refusal_branch_still_renders_
verbatim``, which asserts what must NOT have moved and passes both before and after by
design (it says so in its own docstring).

WHAT THIS FIX MOVED, AND IT IS NOT NOTHING. Changing the seed changes every draw the run
makes. Measured through the real click on the owner's own three recordings, old seed
(derived from his own file path) against new::

    -700 mV   14.9 % (145,222,137,188,178)  ->  23.9 % (205,238,441,96,411)
    -600 mV   REFUSED, 31 event-free        ->  REFUSED, 31 event-free   (label identical)
    -500 mV   REFUSED, 139 event-free       ->  REFUSED, 159 event-free

``n_event_free`` is drawn from the same Generator as the surrogates
(``truth._null_control`` -> ``_event_free_trace(rec, real, kw, gen)``), so no seed change
can leave it fixed. The refusal branch's CODE and rendering are untouched; its numbers are
re-drawn. See the notes file for the full strings.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 60_000
_EVENTS = [(6_000, 6_400), (18_000, 18_350), (33_000, 33_500), (47_000, 47_300)]

#: The -700 mV run-1 shape, as the panel sees it. The counts are the owner's own
#: (172, 237, 94, 16, 301): 301/16 = 18.8125, the ratio whose printed form is hunk 4.
_REAL_SHAPE = {
    "n_real": 1164, "n_event_free": 151,
    "per_surrogate": [172, 237, 94, 16, 301],
    "n_surrogate_mean": 164.0, "n_surrogate_std": 113.0,
    "estimated_fdr": 164.0 / 1164.0, "notes": (),
}

_SEED_LINE = re.compile(r"seed = (\d+)")


# ================================================================================================
# WHERE DID `nanoscope` COME FROM, AND ARE ALL FOUR HUNKS HERE?
#
# A revert stub with a wrong signature once manufactured four fake failures in this
# project and was believed, and a test once passed by comparing `None == None`. So before
# any claim: both modules under test must be source, must resolve out of the SAME package
# directory (a half-reverted tree is exactly the shape that produces a convincing lie),
# and each hunk must be nameable.
# ================================================================================================

def test_fup_the_modules_under_test_resolve_from_the_tree_under_test():
    from nanoscope.gui import analysis_panel, panels

    pkg = Path(nanoscope.__file__).resolve().parent
    for mod in (analysis_panel, panels):
        got = Path(mod.__file__).resolve()
        assert got.suffix == ".py", f"{mod.__name__} is not source: {got}"
        assert got.parent.parent == pkg or got.parent == pkg, (
            f"{mod.__name__} resolved from {got}, outside the nanoscope package at {pkg}")
    # Both files, one package. Print it so a run against a revert tree says where it ran.
    print(f"\nnanoscope resolved from {pkg}")

    assert hasattr(panels.StatsPanel, "_set_fdr_readout"), (
        "StatsPanel._set_fdr_readout is missing — hunk 1 (the single readout writer) is "
        "not in this tree")
    assert hasattr(analysis_panel.AnalysisPanel, "_recording_identity"), (
        "AnalysisPanel._recording_identity is missing — hunk 2 (bytes, not path) is not "
        "in this tree")
    # Hunk 3 is a call-site change with no new name; assert the call site itself.
    src = inspect.getsource(analysis_panel.AnalysisPanel._on_null_requested)
    assert "_null_seed(self._sel, params)" in src, (
        "hunk 3 is not in this tree: _on_null_requested still seeds off the stripped "
        f"detect kwargs rather than the whole form:\n{src}")


# ================================================================================================
# builders — a real MainWindow over documents whose meta.path is a REAL FILE ON DISK
#
# The seed's identity is now the bytes at that path, so these tests need files that
# actually exist and whose contents they control. The trace the panel analyses is the
# synthetic array; the file on disk is what `fingerprint_source` reads.
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


def _add(win, path: Path, payload: bytes, arr: np.ndarray) -> str:
    """Write ``payload`` to ``path`` and add a document that names it."""
    path.write_bytes(payload)
    doc = _doc(path, arr)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


def _window(qtbot) -> MainWindow:
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _quit(win) -> None:
    win._analysis_controller.abort()
    win._controller.abort()


def _record_null_calls(monkeypatch) -> list[dict]:
    """Replace `truth.null_control` with a recorder that keeps its keyword contract.

    The panel's job does ``from ..analysis.truth import null_control`` at call time on
    the worker thread, so patching the module attribute is what the worker picks up. The
    stub's signature is the real one keyword for keyword — a laxer stub is how a fake
    failure gets manufactured, and this one raises if the panel ever calls it another way.
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


def _click_null(win, qtbot) -> tuple[str, str]:
    """Press *Null control…* and drain the loop. Returns what the user then reads."""
    panel = win._analysis_panel
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=120_000):
        panel.detection_panel.null_button.click()
    qtbot.wait(40)
    return (panel._stats.fdr_label.text(), panel._stats.fdr_label.toolTip())


def _printed_seed(tip: str) -> str:
    m = _SEED_LINE.search(tip)
    assert m, f"the tooltip names no seed, so the user cannot reproduce anything: {tip!r}"
    return m.group(1)


# ================================================================================================
# HUNK 1 — the provenance tooltip dies with the number it describes
# ================================================================================================

def test_fup_the_provenance_tooltip_does_not_survive_onto_another_recording(
        qapp, qtbot, tmp_path, monkeypatch):
    """The defect, through the wired app: select another recording, read what is there.

    This is the whole finding. The label is honest on its own ("not measured"), and the
    tooltip underneath it is a paragraph of numbers about a recording that is not on
    screen — including the sentence promising that pressing the button again "on this
    recording" reproduces them.
    """
    _record_null_calls(monkeypatch)
    win = _window(qtbot)
    id_a = _add(win, tmp_path / "recording_A.dat", b"A" * 4096, _trace(0))
    id_b = _add(win, tmp_path / "recording_B.dat", b"B" * 4096, _trace(1))

    win._on_selection(id_a, 0)
    qtbot.wait(20)
    text_a, tip_a = _click_null(win, qtbot)
    assert "14.1%" in text_a, f"fixture broken — no percentage on A: {text_a!r}"
    assert "1164 real detections" in tip_a, f"fixture broken — no provenance: {tip_a!r}"

    win._on_selection(id_b, 0)
    qtbot.wait(40)
    panel = win._analysis_panel
    text_b, tip_b = panel._stats.fdr_label.text(), panel._stats.fdr_label.toolTip()

    assert text_b == panel._stats._NO_FDR_TEXT, (
        f"B has no null control of its own; the label must say so: {text_b!r}")
    assert tip_b == "", (
        "the previous recording's provenance is still on screen under a label that says "
        f"the number was never measured:\n{tip_b!r}")
    # Say the quiet part out loud so a revert failure reads as the defect, not as a typo.
    assert "1164" not in tip_b and "this recording" not in tip_b, (
        f"recording A's numbers are being shown as recording B's: {tip_b!r}")
    _quit(win)


def test_fup_the_readout_has_exactly_one_writer(qapp):
    """The invariant, not the symptom line: no exit of ``set_fdr`` may write half of it.

    The leak was one branch out of four forgetting the tooltip. Fixing that branch fixes
    today; fixing the SHAPE means the fifth branch someone writes next month cannot
    reopen it. ``set_fdr`` must therefore touch ``fdr_label`` only through
    ``_set_fdr_readout``, which writes both halves and defaults the tooltip to empty.
    """
    from nanoscope.gui import panels
    from nanoscope.gui.panels import StatsPanel

    # The whole class, not just `set_fdr`: the invariant is "one writer", and a helper
    # somewhere else in StatsPanel that pokes the label is the same hole one method over.
    cls = next(n for n in ast.parse(Path(panels.__file__).read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "StatsPanel")
    offenders = [
        (fn.name, call.func.attr)
        for fn in cls.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for call in ast.walk(fn)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("setText", "setToolTip")
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "fdr_label"
        and fn.name != "_set_fdr_readout"
    ]
    assert offenders == [], (
        "the FDR readout has more than one writer, so a branch can again set a number "
        f"without its provenance (or provenance without its number): {offenders}")

    # ...and nothing OUTSIDE this file reaches in either. `AnalysisPanel` owns the
    # readout's lifecycle and drives it through `set_fdr`; a panel poking
    # `_stats.fdr_label` directly would reopen the hole one module over.
    gui = Path(panels.__file__).resolve().parent
    outside = sorted(
        f"{p.name}:{node.lineno}"
        for p in gui.glob("*.py") if p.name != "panels.py"
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, ast.Attribute) and node.attr == "fdr_label")
    assert outside == [], (
        f"the FDR readout is written from outside panels.py: {outside}")

    # And the one writer really does write both halves, including the empty default.
    panel = StatsPanel()
    panel._set_fdr_readout("x", "why")
    assert (panel.fdr_label.text(), panel.fdr_label.toolTip()) == ("x", "why")
    panel._set_fdr_readout("y")
    assert panel.fdr_label.toolTip() == "", (
        "a caller with no provenance left the previous caller's on screen")


# ================================================================================================
# HUNK 2 — the recording is its bytes, not its filename
# ================================================================================================

def test_fup_the_printed_seed_follows_the_bytes_not_the_filename(
        qapp, qtbot, tmp_path, monkeypatch):
    """Two names, one recording: the tooltip must promise the same draw for both.

    Pre-fix, the owner's own -700 mV file under three names seeded 1592734707 /
    1091228298 / 1722390092 — so the percentage he published off a scratch copy was not
    the percentage he would see opening his own file.
    """
    _record_null_calls(monkeypatch)
    win = _window(qtbot)
    same = b"\x01\x02\x03\x04" * 3000
    arr = _trace(0)
    id_1 = _add(win, tmp_path / "scratch_copy.dat", same, arr)
    id_2 = _add(win, tmp_path / "the owners own name.dat", same, arr)
    id_3 = _add(win, tmp_path / "different_bytes.dat", same + b"\x99", arr)

    seeds = {}
    for label, doc_id in (("copy", id_1), ("owner", id_2), ("other", id_3)):
        win._on_selection(doc_id, 0)
        qtbot.wait(20)
        _, tip = _click_null(win, qtbot)
        seeds[label] = _printed_seed(tip)

    assert seeds["copy"] == seeds["owner"], (
        "the same recording under two names promises two different draws — the printed "
        f"percentage is a property of the path: {seeds}")
    assert seeds["other"] != seeds["copy"], (
        "a file with different contents replayed the same phase randomisation, so the "
        f"seed is not a function of what was measured either: {seeds}")
    _quit(win)


def test_fup_an_unreadable_source_still_seeds_reproducibly(qapp, tmp_path):
    """UNKNOWN bytes are not "changed" bytes.

    A segment's persistent key is a ``segment_uid`` and not a file at all, and a source
    on a disconnected volume cannot be read. Both must still seed to something stable —
    a null that refuses to reproduce is the defect H1 existed to fix — so the identity
    falls back to the source key, and the per-load token is still ignored.
    """
    from nanoscope.gui.analysis_panel import AnalysisPanel

    missing = str(tmp_path / "not-a-file.abf")
    assert AnalysisPanel._recording_identity(missing) == f"key:{missing}"

    kw = {"k_in": 5.0, "min_dwell_s": 5e-4}
    a = AnalysisPanel._null_seed((missing + "#load-28259b34622c", 2), kw)
    b = AnalysisPanel._null_seed((missing + "#load-727e359b5d5a", 2), kw)
    assert a == b, f"closing and reopening re-seeded an unreadable source: {a} vs {b}"

    real = tmp_path / "real.abf"
    real.write_bytes(b"z" * 1024)
    ident = AnalysisPanel._recording_identity(str(real))
    assert ident.startswith("bytes:1024:sha256:"), (
        f"a readable file must be identified by its bytes, not its name: {ident!r}")
    assert str(real) not in ident, (
        f"the path leaked into the identity of a readable file: {ident!r}")


# ================================================================================================
# HUNK 3 — every setting that reaches the surrogate reaches the seed
# ================================================================================================

@pytest.mark.parametrize("control", ["notch_enabled", "median_enabled", "hp_enabled",
                                     "savgol_enabled", "detrend_enabled"])
def test_fup_a_preprocessing_toggle_draws_an_independent_null(
        qapp, qtbot, tmp_path, monkeypatch, control):
    """These five steer the trace the surrogates are built from. Pre-fix, all five were
    stripped before the seed and the same draw came back — under a tooltip claiming a
    changed setting draws a genuinely independent null.

    The second half matters as much as the first: toggling BACK must restore the original
    seed, or "reproducible" has been traded away rather than corrected.
    """
    _record_null_calls(monkeypatch)
    win = _window(qtbot)
    doc_id = _add(win, tmp_path / "one.dat", b"Q" * 4096, _trace(0))
    win._on_selection(doc_id, 0)
    qtbot.wait(20)
    box = getattr(win._analysis_panel.detection_panel, control)

    _, tip0 = _click_null(win, qtbot)
    box.setChecked(not box.isChecked())
    qtbot.wait(20)
    _, tip1 = _click_null(win, qtbot)
    box.setChecked(not box.isChecked())
    qtbot.wait(20)
    _, tip2 = _click_null(win, qtbot)

    s0, s1, s2 = (_printed_seed(t) for t in (tip0, tip1, tip2))
    assert s0 != s1, (
        f"toggling {control} replayed the same phase randomisation (seed {s0}); a "
        "difference read as 'the filter changed the false-positive rate' is the draw "
        "the two runs share")
    assert s0 == s2, (
        f"putting {control} back did not bring the number back: {s0} -> {s1} -> {s2}")
    _quit(win)


# ================================================================================================
# HUNK 4 — the one line whose job is dispersion may not round it away
# ================================================================================================

def test_fup_the_dispersion_ratio_is_not_rounded_into_an_understatement(
        qapp, qtbot, tmp_path, monkeypatch):
    """``{max/min:.0f}x`` rounded the caveat, in both directions.

    Measured on the revert tree with the owner's own counts (172, 237, 94, 16, 301):
    301/16 = 18.8125 printed as **19x**, an overstatement. Going the other way 2.49x
    prints "2x" and so does 2.5x (halves round to even), understating the spread on the
    one line that exists to say the headline is not to be trusted.
    """
    _record_null_calls(monkeypatch)
    win = _window(qtbot)
    doc_id = _add(win, tmp_path / "spread.dat", b"S" * 4096, _trace(0))
    win._on_selection(doc_id, 0)
    qtbot.wait(20)
    _, tip = _click_null(win, qtbot)

    assert "18.8x" in tip, f"the spread line understates the spread: {tip!r}"
    assert "18x" not in tip.replace("18.8x", ""), f"rounded ratio still printed: {tip!r}"
    _quit(win)


def test_fup_a_bare_ratio_keeps_reading_as_a_bare_ratio(qapp):
    """One decimal, with a bare ``.0`` dropped: 2.49 -> "2.5x", 19.0 -> "19x"."""
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": 0.1, "n_real": 1000, "n_event_free": 1,
                   "per_surrogate": [100, 249], "notes": (), "seed": 1})
    assert "2.5x" in panel.fdr_label.toolTip(), panel.fdr_label.toolTip()
    panel.set_fdr({"estimated_fdr": 0.1, "n_real": 1000, "n_event_free": 1,
                   "per_surrogate": [10, 190], "notes": (), "seed": 1})
    assert "19x" in panel.fdr_label.toolTip(), panel.fdr_label.toolTip()
    assert "19.0x" not in panel.fdr_label.toolTip()


# ================================================================================================
# WHAT MUST NOT HAVE MOVED — the -600 mV / -500 mV refusal RENDERING
#
# This one passes before AND after on purpose: it guards the branch this change was told
# not to touch. Its NUMBERS do move, because `n_event_free` is drawn from the same
# Generator the seed controls (see the module docstring); its rendering may not.
# ================================================================================================

def test_fup_refusal_branch_still_renders_verbatim(qapp):
    from nanoscope.gui.panels import StatsPanel

    reason = ("the AAFT surrogates produced 19261 detections where this recording's own "
              "event-free noise produces 31, so the surrogates are not the same noise")
    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": float("nan"), "n_real": 276, "n_event_free": 31,
                   "per_surrogate": [19261, 19184, 19200], "notes": (reason,), "seed": 7})

    text = panel.fdr_label.text()
    assert "nan" not in text
    assert text == ("false-positive rate NOT estimated — 31 detection(s) on this "
                    "recording's own event-free noise (a lower bound, measured with no "
                    "surrogate).")
    assert panel.fdr_label.toolTip() == reason, (
        "the refusal's tooltip is no longer its own note")
    assert "seed" not in panel.fdr_label.toolTip()
    assert "individual surrogates" not in text
