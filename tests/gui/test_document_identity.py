"""WHICH LOAD OF WHICH FILE — three defects, one guarantee.

Every "before" below was measured through a real ``MainWindow`` on the tree as it stood
on 2026-08-02, with the app's own open / detect / save / restore paths:

* **A result computed on one load installed itself onto another.** ``doc_id`` WAS the
  resolved path, so the ~50 ``session.has_document(res.doc_id)`` guards could not tell a
  reopened recording from the one that was analysed. Close a 100,000-sample recording
  while a detection is in flight, reopen 4,000 samples of it, and the answer landed:
  ``result installed onto the NEW load? True``, ``events past EOF = 5 of 5``,
  ``analyzed_duration_s = 1.0`` against a true 0.04 s — a capture rate wrong by 25x —
  and ``staleness banner? False``, because nothing about the calibration had changed.
* **A project restore checked only ``path.exists()``.** Re-export from Clampex over the
  same filename and reopen the project: ``reattached 5 events to a 5000-sample trace;
  5 entirely/partly past EOF``, ``event 0 depth_pA = 30.17`` in a signal whose standard
  deviation is 1.08 pA, ``info bar text: ''``.
* **After a restore the app drew one recording and analysed another.** ``drawn =
  ['short_b.csv']``, ``analysis target = long_a.csv``, ``'Visible view' resolves to
  (0, 4000) of 40000 samples (10.0% of the target)``, ``measure_bounds() = None``,
  ``add_current() = False`` — the Measure tool's *Add to list* a silent no-op.

The fix is :mod:`nanoscope.core.identity`, which splits the one string those three shared
into the three identities they each needed: the LOAD (``doc_id``), the RECORDING
(``persistent_key`` / ``source_key``) and the BYTES (``SourceFingerprint``). These tests
drive the real window, because the split being correct in the value objects is exactly
what would not have caught any of the three.

Repro harness, the reverted-copy builder these tests were checked against, and the
discriminating-power probe for the ones a whole-feature revert cannot move:
``/private/tmp/claude-501/-Users-abrahamcampos-Desktop/
0faceeb7-78b4-42dd-94ae-f2dd7dffdcaa/scratchpad/P1/identity/``
(``repro_before.py``, ``make_revert.py``, ``probe.py``, ``bitid.py``, ``oracle_guard.py``).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import live_doc_id
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.identity import (
    LOAD_SEP,
    SourceFingerprint,
    describe_source_change,
    fingerprint_source,
    make_doc_id,
    new_load_uid,
    source_of,
)
from nanoscope.core.segment import persistent_key
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_EVENTS = [(10_000, 10_400), (30_000, 30_500), (50_000, 50_300),
           (70_000, 70_600), (90_000, 90_200)]


# ================================================================================================
# builders
# ================================================================================================

def _trace(n: int = N, seed: int = 0, events=_EVENTS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in events:
        if e <= n:
            x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(win: MainWindow, doc: TraceDocument) -> str:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    return doc_id


def _win(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _result(win: MainWindow, doc_id: str, arr: np.ndarray) -> AnalysisResult:
    """A real Schmitt detection, stamped with ``doc_id`` — what the worker hands back."""
    df, prov = detect_events(np.asarray(arr, dtype=np.float64), FS, min_dwell_s=5e-4)
    return AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov))


def _detect(win: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _banner(win: MainWindow) -> str:
    return win._info_bar._label.text()


# ================================================================================================
# 1. the identity split itself
# ================================================================================================

def test_two_loads_of_one_file_are_two_documents_with_one_persistent_key(tmp_path):
    """The whole change in four lines. Same file, same source, same persistent key —
    different loads, therefore different ids."""
    csv = tmp_path / "same.csv"
    arr = _trace()
    _write_csv(csv, arr)
    a, b = _doc(csv, arr), _doc(csv, arr)

    assert a.source_key == b.source_key == str(csv.resolve())
    assert persistent_key(a) == persistent_key(b) == a.source_key
    assert a.doc_id != b.doc_id
    assert source_of(a.doc_id) == source_of(b.doc_id) == a.source_key
    # ...and the load token is real, not an empty string that would collapse them again.
    assert a.meta.load_uid and b.meta.load_uid and a.meta.load_uid != b.meta.load_uid


def test_rebuilding_a_document_keeps_its_identity(tmp_path):
    """LOAD-BEARING, not tidiness. ``Session.add_document`` folds the sidecar's issues into
    a frozen doc with ``dataclasses.replace``, which re-enters ``__post_init__`` — AFTER the
    session has keyed the document under its id. Minting a second token there would orphan
    every layer, calibration, result and undo stack of that recording in one line."""
    from dataclasses import replace

    csv = tmp_path / "rebuild.csv"
    arr = _trace()
    _write_csv(csv, arr)
    doc = _doc(csv, arr)
    again = replace(doc, meta=replace(doc.meta, issues=("something the loader noticed",)))
    assert again.doc_id == doc.doc_id
    assert again.meta.load_uid == doc.meta.load_uid


def test_a_doc_id_survives_a_source_containing_the_separator(tmp_path):
    """``source_of`` splits on the LAST separator, so a path that happens to contain the
    literal is recovered exactly. A defensive case with a real cost if wrong: the id would
    be truncated and the export provenance would name a file that does not exist."""
    weird = f"/data/run{LOAD_SEP}2/pore.abf"
    doc_id = make_doc_id(weird, new_load_uid())
    assert source_of(doc_id) == weird
    # And an id with no token at all (a meta that never became a document) reads back whole —
    # including this one, whose tail is not a token, so the split refuses to guess.
    assert source_of(weird) == weird
    assert make_doc_id(weird, "") == weird
    assert source_of(f"/data/pore.abf{LOAD_SEP}notatoken") == \
        f"/data/pore.abf{LOAD_SEP}notatoken"


def test_only_identity_knows_the_shape_of_a_doc_id():
    """AST GUARD. ``doc_id`` is opaque: exactly one module may build or take it apart.

    The alternative is what this replaced — ``str(ctx.meta.path.resolve())`` written out
    twice in the analysis layer and assumed to be an identity in fifty places. A third
    detector copying that line would be wrong in precisely the way the whole tier exists to
    fix, and it would look right.
    """
    import nanoscope

    root = pathlib.Path(nanoscope.__file__).parent
    # identity.py builds and reads the id; document.py is where `source_key` is DEFINED, and
    # is the one place a path legitimately becomes an identity. Everything else must go
    # through one of them.
    allowed = {"identity.py", "document.py"}
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name in allowed:
            continue
        src = path.read_text()
        rel = str(path.relative_to(root))
        if LOAD_SEP in src:
            offenders.append(f"{rel}: writes the load separator literally")
        for node in ast.walk(ast.parse(src)):
            # `<anything>.meta.path.resolve()` used as a document identity.
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (isinstance(fn, ast.Attribute) and fn.attr == "resolve"
                    and isinstance(fn.value, ast.Attribute) and fn.value.attr == "path"
                    and isinstance(fn.value.value, ast.Attribute)
                    and fn.value.value.attr == "meta"):
                offenders.append(f"{rel}:{node.lineno}: meta.path.resolve() as an identity "
                                 f"— use identity.doc_id_for_meta / document.source_key")
    assert offenders == [], "\n".join(offenders)


# ================================================================================================
# 2. C5 — a result may only land on the load it was computed from
# ================================================================================================

def test_a_detection_landing_after_a_reopen_does_not_install_on_the_new_load(
        qapp, qtbot, tmp_path):
    """THE C5 REPRO. Before: ``result installed onto the NEW load? True``, all five events
    past the end of a 4,000-sample recording, ``analyzed_duration_s`` 1.0 s against a true
    0.04 s, and no banner of any kind."""
    csv = tmp_path / "reopen.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    old_id = _install(win, _doc(csv, arr))
    in_flight = _result(win, old_id, arr)              # computed on the LONG load
    assert int(in_flight.payload.df["end_idx"].max()) > 4_000

    # ...while it runs, the user closes the recording and reopens a short range of it.
    win._session.close_document(old_id)
    short = arr[:4_000].copy()
    new_id = _install(win, _doc(csv, short))
    assert new_id != old_id, "a reopen must be a new load"

    said: list[str] = []
    win._analysis_panel.runRefused.connect(said.append)
    win._analysis_panel._on_controller_finished(in_flight)

    assert win._session.results.get(new_id, 0, "detect.schmitt") is None, \
        "a detection computed on a previous load installed itself onto the reopened one"
    assert win._session.results.get(old_id, 0, "detect.schmitt") is None
    # ...and it was not merely dropped in silence: an answer that vanishes without a word is
    # indistinguishable from a detector that found nothing.
    assert len(said) == 1
    assert "closed and reopened" in said[0]
    assert "run it again" in said[0]


def test_a_detection_landing_after_a_plain_close_still_says_so(qapp, qtbot, tmp_path):
    """The neighbouring case, which must read DIFFERENTLY: the recording is simply gone, so
    there is nothing to re-run it on and the message must not suggest otherwise."""
    csv = tmp_path / "closed.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    doc_id = _install(win, _doc(csv, arr))
    in_flight = _result(win, doc_id, arr)
    win._session.close_document(doc_id)

    said: list[str] = []
    win._analysis_panel.runRefused.connect(said.append)
    win._analysis_panel._on_controller_finished(in_flight)
    assert len(said) == 1
    assert "was closed while that run was in flight" in said[0]
    assert "reopened" not in said[0]


def test_the_batch_sink_drops_a_reopened_loads_answer_too(qapp, qtbot, tmp_path):
    """THE TWIN. Batch detect has its own result sink with its own copy of the same guard —
    the exact shape this project has watched come back three times. It is fixed here without
    being touched, because the IDENTITY carries the load: the guard it already had became
    true the moment ``doc_id`` stopped meaning "some document with this path"."""
    from nanoscope.gui.analysis_worker import BatchOutcome

    csv = tmp_path / "batch.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    old_id = _install(win, _doc(csv, arr))
    in_flight = _result(win, old_id, arr)
    win._session.close_document(old_id)
    new_id = _install(win, _doc(csv, arr[:4_000].copy()))

    win._on_batch_finished(BatchOutcome(results=(in_flight,)))
    assert win._session.results.get(new_id, 0, "detect.schmitt") is None
    msg = win.statusBar().currentMessage()
    assert "Dropped 1" in msg and "reopened" in msg, msg


def test_a_module_stamps_the_load_it_ran_on(tmp_path):
    """The stamp itself, at the source: two loads of one file produce results that cannot be
    confused, because the detector asks ``identity.doc_id_for_meta`` rather than re-deriving
    the path."""
    from nanoscope.analysis.context import SimpleAnalysisContext
    from nanoscope.analysis.modules import DetectSchmitt
    from nanoscope.core.document import CalibratedChannel

    csv = tmp_path / "stamp.csv"
    arr = _trace()
    _write_csv(csv, arr)
    a, b = _doc(csv, arr), _doc(csv, arr)

    ids = []
    for doc in (a, b):
        ch = CalibratedChannel(info=doc.meta.channels[0], data=doc.channels[0], units="pA",
                               sample_rate_hz=FS, t0_s=0.0)
        res = DetectSchmitt().run(SimpleAnalysisContext(
            channel=ch, meta=doc.meta, params={"min_dwell_s": 5e-4}))
        ids.append(res.doc_id)
    assert ids[0] == a.doc_id and ids[1] == b.doc_id
    assert ids[0] != ids[1]


def test_opening_a_file_that_is_already_open_still_focuses_it(qapp, qtbot, tmp_path):
    """The policy the split could have broken silently: one load per file. Asking
    ``has_document`` about a freshly built doc_id is now always False, so the "already
    open?" question had to move to the SOURCE — otherwise a second File ▸ Open would load
    the same 180 MB twice and leave two documents claiming the same recording."""
    csv = tmp_path / "twice.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    doc_id = _install(win, _doc(csv, arr))
    win.open_paths([str(csv)])
    # Drain the open pipeline before judging: a guard that FAILED here would not add the
    # second document until the (asynchronous) load finished, so a fixed wait would report
    # success for the wrong reason.
    qtbot.waitUntil(lambda: win._active_open is None and not win._open_queue, timeout=30_000)
    qtbot.wait(30)
    assert [d.doc_id for d in win._session.documents()] == [doc_id]
    assert win._session.doc_id_for_source(str(csv.resolve())) == doc_id
    win._controller.abort()


# ================================================================================================
# 3. C3 — a path is a name, not a recording
# ================================================================================================

def _project_with_detection(win, qtbot, csv, arr, tmp_path, name="p.nanoscope"):
    _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    proj = tmp_path / name
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()
    return proj


def _restore(qtbot, proj) -> MainWindow:
    w = _win(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    return w


def test_a_re_export_over_the_same_filename_is_named_counted_and_explained(
        qapp, qtbot, tmp_path):
    """THE C3 REPRO. Before: 5 events reattached to a 5,000-sample flat trace, 5 of them past
    its end, event 0 claiming a 30.17 pA blockade in a signal with a 1.08 pA standard
    deviation — and ``info bar text: ''``."""
    csv = tmp_path / "reexport.csv"
    arr = _trace()
    _write_csv(csv, arr)
    proj = _project_with_detection(_win(qtbot), qtbot, csv, arr, tmp_path)

    # Re-export from Clampex over the same filename: shorter, and flat.
    flat = np.full(5_000, -49.94, dtype=np.float32)
    flat += np.random.default_rng(3).normal(0.0, 1.07, 5_000).astype(np.float32)
    _write_csv(csv, flat)

    w2 = _restore(qtbot, proj)
    live = live_doc_id(w2._session, csv)
    assert live is not None, "the project must still open — this warns, it never refuses"

    assert not w2._info_bar.isHidden(), "the warning was written but never raised"
    text = _banner(w2)
    assert "is not the file this project was saved from" in text
    assert "5,000 samples where the project recorded 100,000" in text   # names WHAT differs
    assert "5 restored event(s)" in text                                 # ...and the harm
    assert "past the end of this recording" in text
    assert "Re-detect" in text                                           # ...and the remedy
    w2._controller.abort()


def test_an_unchanged_source_says_nothing_at_all(qapp, qtbot, tmp_path):
    """THE CONTROL, and the reason the warning is worth reading. A check that cries wolf on
    every ordinary reopen is one the user learns to click past — so the same project,
    reopened over the same untouched file, must be silent."""
    csv = tmp_path / "same.csv"
    arr = _trace()
    _write_csv(csv, arr)
    proj = _project_with_detection(_win(qtbot), qtbot, csv, arr, tmp_path)

    w2 = _restore(qtbot, proj)
    assert "not the file this project was saved from" not in _banner(w2)
    assert w2._info_bar.isHidden() or _banner(w2) == ""
    w2._controller.abort()


def test_touching_a_file_without_changing_it_is_not_a_mismatch(tmp_path):
    """A modification time that moved is NOT evidence. Copying a file preserves mtime and
    re-saving one in a sync folder changes it, so mtime alone can only produce false alarms;
    it is carried for the SENTENCE ("last modified …"), never for the verdict."""
    p = tmp_path / "touched.bin"
    p.write_bytes(b"x" * 4096)
    before = fingerprint_source(p, n_samples=100)
    os.utime(p, ns=(before.mtime_ns + 10**9, before.mtime_ns + 10**9))
    after = fingerprint_source(p, n_samples=100)

    assert after.mtime_ns != before.mtime_ns
    assert describe_source_change(before, after, loaded_n_samples=100) == ""
    # ...but a real content change, at identical size, IS caught by the digest.
    p.write_bytes(b"y" * 4096)
    changed = fingerprint_source(p, n_samples=100)
    assert changed.size == before.size
    assert "contents differ" in describe_source_change(before, changed, loaded_n_samples=100)


def test_an_unknown_fingerprint_is_not_a_mismatch():
    """Silence must mean NO EVIDENCE, never "verified identical". A v1–v3 project carries no
    stamp and a source on a disconnected volume cannot be read; neither is a reason to tell
    the user their data changed."""
    known = SourceFingerprint(n_samples=10, size=5, digest="sha256:aa", mtime_ns=1)
    assert describe_source_change(None, known, loaded_n_samples=999) == ""
    assert describe_source_change(known, None, loaded_n_samples=10) == ""
    assert SourceFingerprint.from_json(None) is None
    assert SourceFingerprint.from_json({"n_samples": "not a number"}) is None
    assert fingerprint_source("/no/such/file/anywhere") is None


def test_a_v3_project_still_opens_and_says_nothing(qapp, qtbot, tmp_path):
    """MIN_READ_VERSION. The stamp is additive: an older project simply lacks the key, loads
    unchanged, and raises no warning it has no evidence for."""
    import json
    import zipfile

    from nanoscope.core.project import MIN_READ_VERSION, load_project

    csv = tmp_path / "old.csv"
    arr = _trace()
    _write_csv(csv, arr)
    proj = _project_with_detection(_win(qtbot), qtbot, csv, arr, tmp_path, "v3.nanoscope")

    # Rewrite the archive as a version-3 project with no `source` key anywhere.
    downgraded = tmp_path / "v3_downgraded.nanoscope"
    with zipfile.ZipFile(proj) as src, zipfile.ZipFile(downgraded, "w") as dst:
        for item in src.namelist():
            blob = src.read(item)
            if item == "manifest.json":
                man = json.loads(blob)
                man["version"] = 3
                for d in man["documents"]:
                    d.pop("source", None)
                blob = json.dumps(man).encode()
            dst.writestr(item, blob)

    assert MIN_READ_VERSION <= 3
    state = load_project(downgraded)
    assert all(d.source is None for d in state.documents)

    w2 = _restore(qtbot, downgraded)
    assert live_doc_id(w2._session, csv) is not None
    assert "not the file this project was saved from" not in _banner(w2)
    w2._controller.abort()


# ================================================================================================
# 4. C9 — the app never analyses a recording it is not drawing
# ================================================================================================

def test_a_restore_draws_the_recording_it_lands_on(qapp, qtbot, tmp_path):
    """THE C9 REPRO. Before: ``drawn = ['short_b.csv']``, ``analysis target = long_a.csv``,
    ``'Visible view' resolves to (0, 4000) of 40000 samples`` — the first 10 % of a recording
    nobody was looking at, and ``measure_bounds() = None`` so *Add to list* did nothing."""
    long_a, short_b = tmp_path / "long_a.csv", tmp_path / "short_b.csv"
    arr_a, arr_b = _trace(40_000, seed=0, events=[(10_000, 10_400), (20_000, 20_500)]), \
        _trace(4_000, seed=1, events=[])
    _write_csv(long_a, arr_a)
    _write_csv(short_b, arr_b)

    win = _win(qtbot)
    id_a = _install(win, _doc(long_a, arr_a))
    id_b = _install(win, _doc(short_b, arr_b))
    win._on_selection(id_a, 0)
    _detect(win, qtbot)
    # Save with ONLY short_b drawn: the detection is on a recording the saved view omits.
    win._session.set_layer_visible(f"{id_a}#0", False)
    win._session.set_layer_visible(f"{id_b}#0", True)
    proj = tmp_path / "two.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _restore(qtbot, proj)
    a2, b2 = live_doc_id(w2._session, long_a), live_doc_id(w2._session, short_b)
    target = w2._current_selection
    assert target[0] == a2, "the restore should land on the recording carrying the detection"
    assert a2 in w2._session.shown_doc_ids(), "the analysed recording must be on screen"
    # ADDITIVE: the saved view is not overruled, it is added to.
    assert b2 in w2._session.shown_doc_ids()
    assert "added to the view" in _banner(w2)

    # ...and the two things that were silently wrong now work.
    n = w2._session.document(a2).meta.n_samples
    rng = w2._analysis_panel._visible_sample_range(a2, 0, n)
    assert rng is not None and rng[1] - rng[0] > n // 2, \
        "'Visible view' resolved to a stretch of the target it is not showing"
    mp = w2._analysis_panel.measure_panel
    mp.set_active(True)
    qtbot.wait(20)
    assert w2._trace_view.measure_bounds() is not None
    assert mp.add_current() is True
    w2._controller.abort()


def test_a_restore_prefers_a_target_that_is_already_drawn(qapp, qtbot, tmp_path):
    """The other half, and the reason "show what you analyse" costs the user nothing: when a
    DRAWN recording carries a detection, that is the one to land on, and the saved view is
    left exactly as it was."""
    a_csv, b_csv = tmp_path / "pa.csv", tmp_path / "pb.csv"
    arr = _trace()
    _write_csv(a_csv, arr)
    _write_csv(b_csv, _trace(seed=7))

    win = _win(qtbot)
    id_a = _install(win, _doc(a_csv, arr))
    id_b = _install(win, _doc(b_csv, _trace(seed=7)))
    for did in (id_a, id_b):
        win._on_selection(did, 0)
        _detect(win, qtbot)
    win._session.set_layer_visible(f"{id_a}#0", False)
    win._session.set_layer_visible(f"{id_b}#0", True)
    proj = tmp_path / "both.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _restore(qtbot, proj)
    b2 = live_doc_id(w2._session, b_csv)
    assert w2._current_selection[0] == b2
    assert w2._session.shown_doc_ids() == {b2}, "the saved view was disturbed for no reason"
    assert "added to the view" not in _banner(w2)
    w2._controller.abort()


def test_visible_view_refuses_rather_than_substituting_the_whole_recording(
        qapp, qtbot, tmp_path):
    """The guard that would have caught C8, the measure-panel offset defect and C9 alike.

    ``_visible_sample_range`` used to fall back to ``(0, n)`` — THE WHOLE RECORDING — whenever
    it could not make sense of the window, so "Visible view" over a channel that is not on
    screen ran a slow ML detection over ten times the data the user asked for and stamped the
    result with a range they never chose. It now answers None, and the caller says so.
    """
    csv = tmp_path / "vis.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = _win(qtbot)
    doc_id = _install(win, _doc(csv, arr))
    panel = win._analysis_panel

    drawn = panel._visible_sample_range(doc_id, 0, N)
    assert drawn is not None                       # the ordinary case is unchanged

    # Untick the channel: layer membership is the user's authoritative include/exclude
    # decision, so the app must not overrule it — it must decline to answer about it.
    win._session.set_layer_visible(f"{doc_id}#0", False)
    win._trace_view.rebuild() if hasattr(win._trace_view, "rebuild") else None
    qtbot.wait(20)
    assert panel._visible_sample_range(doc_id, 0, N) is None
    assert win._trace_view.visible_x_range_for(doc_id, 0) is None

    said: list[str] = []
    panel.runRefused.connect(said.append)
    # Past the runnable gate (TensorFlow + a loaded model are not what is under test here);
    # what matters is that the range resolution refuses instead of silently substituting.
    panel.can_run_ml = lambda: True
    panel._on_ml_detect_requested({"range_source": "visible", "model_path": "/nope.keras"})
    assert said and "not on screen" in said[0]
    assert "Whole recording" in said[0], "a refusal must name the way out"
    assert not win._analysis_controller.is_busy(), "it ran anyway"


def test_selecting_a_recording_that_draws_nothing_brings_it_on_screen(qapp, qtbot, tmp_path):
    """The invariant behind C9, exercised without a restore at all — because C9 was only the
    path that happened to expose it. Anything that retargets the app must leave the target
    visible."""
    csv = tmp_path / "hidden.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = _win(qtbot)
    other = tmp_path / "other.csv"
    _write_csv(other, arr)
    id_a = _install(win, _doc(csv, arr))
    id_b = _install(win, _doc(other, arr))

    # Hide everything of A, then target A the way a restore or a navigation would.
    win._session.set_layer_visible(f"{id_a}#0", False)
    assert id_a not in win._session.shown_doc_ids()
    win._restoring = True          # the state in which _show_document is deliberately a no-op
    try:
        win._on_selection(id_a, 0)
    finally:
        win._restoring = False
    assert id_a in win._session.shown_doc_ids()
    assert id_b in win._session.shown_doc_ids(), "nothing else may be hidden to make room"


def test_a_deliberately_unticked_channel_is_not_re_ticked(qapp, qtbot, tmp_path):
    """The limit of that invariant, and it matters: layer membership is the authoritative
    include/exclude decision for a CHANNEL (``core/layer.py``). Selecting a recording puts the
    RECORDING on screen; it does not overrule a channel the user switched off."""
    csv = tmp_path / "twoch.csv"
    arr = _trace()
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr.astype(np.float64),
                  "voltage": np.full(arr.size, -600.0)}).to_csv(csv, index=False)
    info_c = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    info_v = ChannelInfo(index=1, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(path=csv, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info_c, info_v))
    doc = TraceDocument(meta=meta,
                        channels=(arr, np.full(arr.size, -600.0, dtype=np.float32)))

    win = _win(qtbot)
    doc_id = _install(win, doc)
    # Unticking a channel in the Files dock REMOVES its layer (file_dock: `remove_layer`) —
    # membership, not visibility, is the include/exclude decision.
    win._session.remove_layer(f"{doc_id}#1")
    win._on_selection(doc_id, 0)
    assert win._session.layer(f"{doc_id}#1") is None, \
        "selecting the recording re-added a channel the user had switched off"
    assert doc_id in win._session.shown_doc_ids()

    # ...and "Visible view" over that channel declines to answer rather than inventing one.
    assert win._analysis_panel._visible_sample_range(doc_id, 1, arr.size) is None


# ================================================================================================
# 5. the split does not leak into anything persisted or exported
# ================================================================================================

def test_nothing_persisted_or_exported_carries_a_load_token(qapp, qtbot, tmp_path):
    """A per-load identity must never become a per-load PERSISTENCE key, and must never reach
    a human. Three surfaces: the project manifest, the events export's ``source_file``
    provenance, and the manual-measurement table's ``doc_key``."""
    import json
    import zipfile

    csv = tmp_path / "leak.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    doc_id = _install(win, _doc(csv, arr))
    _detect(win, qtbot)

    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    vals = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(vals[10_000]), float(vals[10_400]))
    qtbot.wait(10)
    assert mp.add_current() is True

    seg_id = win._session.create_segment(doc_id, 1_000, 20_000, name="cut")
    proj = tmp_path / "leak.nanoscope"
    assert win._write_project(str(proj)) is True

    with zipfile.ZipFile(proj) as zf:
        raw = zf.read("manifest.json").decode()
    assert LOAD_SEP not in raw, "a per-load token was written into a project"
    man = json.loads(raw)
    assert {d["path"] for d in man["documents"]} == {str(csv.resolve())}

    assert LOAD_SEP not in str(mp.measurements_df["doc_key"].iloc[0])
    assert mp.measurements_df["doc_key"].iloc[0] == str(csv.resolve())
    assert persistent_key(win._session.document(seg_id)) == \
        win._session.document(seg_id).origin.segment_uid

    # ...and the export's provenance names the FILE even once the recording is closed.
    res = win._session.results.get(doc_id, 0, "detect.schmitt")
    win._session.close_document(seg_id)
    win._session.close_document(doc_id)
    assert source_of(res.doc_id) == str(csv.resolve())
    win._analysis_controller.abort()
    win._controller.abort()


def test_a_project_round_trip_reattaches_everything_on_the_real_recording(
        qapp, qtbot, tmp_path):
    """The round trip the split had to keep working, end to end: a detection, a curation, a
    segment, a calibration and a manual measurement, saved and reopened, all re-attaching by
    persistent key onto brand-new loads."""
    from nanoscope.core.calibration import ChannelCalibration

    csv = tmp_path / "rt.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _win(qtbot)
    doc_id = _install(win, _doc(csv, arr))
    win._session.set_calibration(
        doc_id, 0, ChannelCalibration(0.5, 3.0, "nA", Provenance.UNKNOWN),
        write_sidecar=False)
    _detect(win, qtbot)
    win._analysis_panel.set_event_accepted(1, False)
    df_before = win._session.results.get(doc_id, 0, "detect.schmitt").payload.df.copy()

    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    vals = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(vals[30_000]), float(vals[30_500]))
    qtbot.wait(10)
    assert mp.add_current() is True

    seg_id = win._session.create_segment(doc_id, 1_000, 20_000, name="cut")
    seg_uid = win._session.document(seg_id).origin.segment_uid
    proj = tmp_path / "rt.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _restore(qtbot, proj)
    live = live_doc_id(w2._session, csv)
    assert live is not None and live != doc_id
    cal = w2._session.calibration(live, 0)
    assert cal.gain == pytest.approx(0.5) and cal.offset == pytest.approx(3.0)
    res = w2._session.results.get(live, 0, "detect.schmitt")
    assert res is not None
    pd.testing.assert_frame_equal(res.payload.df, df_before)
    segs = w2._session.segment_documents()
    assert len(segs) == 1 and segs[0].origin.segment_uid == seg_uid
    assert segs[0].doc_id != seg_id, "a re-derived segment is a new load"
    assert len(w2._analysis_panel.measure_panel.measurements_df) == 1
    assert "not the file this project was saved from" not in _banner(w2)
    w2._controller.abort()
