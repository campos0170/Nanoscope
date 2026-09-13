"""C-3 — CTRL+Z AFTER A RE-DETECTION SILENTLY REVERTED THE CAPTURE RATE BY 33%.

``EventHistory.checkpoint`` compared three things: ``payload.df``,
``baseline_fingerprint`` and ``warnings``. It was blind to **every other field of**
:class:`~nanoscope.core.analysis.AnalysisResult` — ``provenance`` above all, which carries
the analysed span and ``analyzed_duration_s``, the denominator of ``EventSet.rate_hz``.

Measured, 5 events over 60 000 samples at 50 kHz, through a real ``MainWindow``:

1. Detect over the whole file → span ``(0, 60000)``, ``analyzed_duration_s`` 1.2 s,
   **rate 4.1667 Hz**.
2. Re-run the detector over the VISIBLE VIEW — a sub-range still containing every event.
   ``gui/panels.py`` makes "Visible view" combo index 0, and ``ml_modules`` estimates the
   baseline over the WHOLE recording and offsets indices back to absolute coordinates, so
   the frame comes back byte-identical while the span and duration move:
   **rate 5.5556 Hz**. ``checkpoint()`` returned **False** — no step pushed, and
   ``_pristine`` still holding run 1.
3. Reject one event. The step that gets pushed carries the STALE snapshot. The menu reads
   "Undo Curate event."
4. Ctrl+Z → span ``(0, 60000)``, 1.2 s, **rate 4.1667 Hz**.

A Ctrl+Z aimed at one checkbox swallowed a whole detection run, and the number reaches
``voltage_dep._metric_value(…, "capture_rate_hz")`` — plotted and exported — with
``warnings == ()`` and a status line reading only "Undone: Curate event."

THE FIX IS NOT FOUR MORE FIELDS. ``params``, ``scalars``, ``calibration_fingerprint``,
``module_version`` and ``overlays`` were blind in exactly the same way, and an enumeration
would be wrong again the next time a field is added — which, with a frozen 20-column event
schema, is the ONLY way new information can arrive. So the comparison walks
``dataclasses.fields(AnalysisResult)``, and the test below walks it too: a field added
tomorrow is covered without this file being edited.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import fields as dc_fields, replace

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core import history as hist
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, signal, name: str = "rate.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(signal.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, tmp_path):
    """A window holding one detection stamped as a WHOLE-FILE run: span (0, 60000),
    analyzed_duration_s 1.2 s, rate 4.1667 Hz."""
    signal = _signal()
    session = window._session
    doc_id = session.add_document(_doc(tmp_path, signal))
    window._create_default_layers(session.document(doc_id))
    window._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    prov = dict(prov)
    prov.update(analyzed_duration_s=1.2, span_start=0, span_stop=N)
    res = AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov))
    session.add_result(res)
    return window, doc_id, res


def _rate(window, doc_id) -> float:
    return window._session.results.get(doc_id, 0, "detect.schmitt").payload.rate_hz()


def _span(window, doc_id) -> tuple:
    prov = window._session.results.get(doc_id, 0, "detect.schmitt").payload.provenance
    return (prov.get("span_start"), prov.get("span_stop"),
            prov.get("analyzed_duration_s"))


# ================================================================================================
# the measured defect
# ================================================================================================

def test_an_undo_aimed_at_one_checkbox_does_not_swallow_a_detection_run(detected, qtbot):
    """THE headline, step for step. 5.5556 Hz must still be 5.5556 Hz after Ctrl+Z."""
    window, doc_id, res1 = detected
    key = window._undo_key()
    assert _rate(window, doc_id) == pytest.approx(4.1666667, rel=1e-6)

    # (2) Re-run over the visible view: SAME rows, sub-range provenance.
    df = res1.payload.df
    prov2 = dict(res1.payload.provenance)
    prov2.update(analyzed_duration_s=0.9, span_start=0, span_stop=45_000)
    window._session.add_result(replace(res1, payload=EventSet(df.copy(deep=True), prov2)))
    qtbot.wait(10)
    assert _rate(window, doc_id) == pytest.approx(5.5555556, rel=1e-6)

    # (3) Reject one event — the ordinary click.
    live = window._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    window._analysis_panel.set_event_accepted(int(live["event_id"].iloc[0]), False)
    qtbot.wait(5)
    window._checkpoint_current()

    # (4) Ctrl+Z. It must undo the CHECKBOX and nothing else.
    assert window._history.undo_label(key) == "Curate event"
    window._history.undo(key)
    qtbot.wait(5)

    assert _rate(window, doc_id) == pytest.approx(5.5555556, rel=1e-6), (
        "undoing one checkbox silently reinstated the previous detection's analysed "
        "duration — a 33% error in a plotted, exported capture rate")
    assert _span(window, doc_id) == (0, 45_000, 0.9), (
        "the analysed span travels in provenance and was silently reverted with it")
    live = window._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    assert bool(live["accepted"].iloc[0]), "the checkbox itself was not undone"


def test_the_provenance_only_re_detection_is_itself_an_undoable_step(detected, qtbot):
    """The other half: the run that used to record NOTHING must now be on the stack, under
    a name that does not lie about it. Same rows, replaced EventSet, moved provenance —
    that is a re-run, not an edit."""
    window, doc_id, res1 = detected
    key = window._undo_key()
    depth_before = window._history.depth()

    df = res1.payload.df
    prov2 = dict(res1.payload.provenance)
    prov2.update(analyzed_duration_s=0.9, span_start=0, span_stop=45_000)
    window._session.add_result(replace(res1, payload=EventSet(df.copy(deep=True), prov2)))
    qtbot.wait(10)

    assert window._history.depth() == depth_before + 1, (
        "a re-detection whose frame happened to be identical recorded no step at all")
    assert window._history.undo_label(key) == "Re-run detection"
    window._history.undo(key)
    qtbot.wait(5)
    assert _span(window, doc_id) == (0, N, 1.2), "the undo did not return the first run"


def test_both_maps_are_refreshed_not_just_the_snapshot(detected, qtbot):
    """``_seen`` decides "is this a NEW EventSet or the edited one", which is what separates
    a curation from a detection replacing the table. It was left stale by the same early
    return, so ``_was_replaced`` answered against the wrong generation."""
    window, doc_id, res1 = detected
    key = window._undo_key()

    df = res1.payload.df
    prov2 = dict(res1.payload.provenance)
    prov2["analyzed_duration_s"] = 0.9
    res2 = replace(res1, payload=EventSet(df.copy(deep=True), prov2))
    window._session.add_result(res2)
    qtbot.wait(10)

    live = window._session.results.get(doc_id, 0, "detect.schmitt")
    assert not window._history._was_replaced(key, live), (
        "_seen still points at the previous EventSet, so the next in-place curation would "
        "be judged a table replacement")
    assert window._history._pristine[key].payload.provenance["analyzed_duration_s"] == 0.9


# ================================================================================================
# the guard — the comparison is over the DATACLASS, not over a list somebody maintains
# ================================================================================================

def _schema_frame(n: int = 3) -> pd.DataFrame:
    """A frame carrying the frozen 20-column schema, in order, with the right dtypes."""
    from nanoscope.analysis.events import SCHEMA

    cols = {}
    for name, spec in SCHEMA.items():
        dtype = spec[0] if isinstance(spec, (tuple, list)) else spec
        if str(dtype).startswith(("int", "uint")):
            cols[name] = np.arange(10, 10 + n, dtype="int64")
        elif str(dtype).startswith("float"):
            cols[name] = np.full(n, 1e-4, dtype="float64")
        elif str(dtype) == "bool":
            cols[name] = np.ones(n, dtype=bool)
        else:
            cols[name] = pd.Series([""] * n, dtype=object)
    df = pd.DataFrame(cols)
    df["event_id"] = np.arange(n, dtype="int64")
    df["start_idx"] = np.arange(10, 10 + 10 * n, 10, dtype="int64")
    df["end_idx"] = df["start_idx"] + 5
    return df


def _base_result() -> AnalysisResult:
    df = _schema_frame()
    return AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id="d", channel=0,
        params={"min_dwell_s": 1e-4}, calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(df, {"analyzed_duration_s": 1.2, "span_start": 0}),
        overlays=(), scalars={"n": 3.0}, warnings=(), baseline_fingerprint="abc")


def _mutate(res: AnalysisResult, name: str) -> AnalysisResult:
    """The same result with exactly ONE field moved, whatever that field's type is."""
    cur = getattr(res, name)
    if name == "payload":
        df = cur.df.copy(deep=True)
        df.iat[0, df.columns.get_loc("dwell_s")] = 9.9
        return replace(res, payload=EventSet(df, dict(cur.provenance)))
    if isinstance(cur, str):
        return replace(res, **{name: cur + "-moved"})
    if isinstance(cur, bool) or isinstance(cur, (int, float)):
        return replace(res, **{name: type(cur)(cur) + 1})
    if isinstance(cur, dict):
        return replace(res, **{name: {**cur, "a_new_key": 1.0}})
    if isinstance(cur, tuple):
        return replace(res, **{name: cur + ("moved",)})
    if cur is None:
        return replace(res, **{name: "now-set"})
    raise AssertionError(f"no mutation defined for field {name!r} of type {type(cur)}")


def test_every_field_of_the_result_is_compared():
    """THE GUARD. One field moved, whichever field it is, must be seen.

    This enumerates ``dataclasses.fields`` rather than naming fields, so adding a field to
    ``AnalysisResult`` extends this test the same day it extends the comparison. That is the
    whole difference between the fix and the same fix again next quarter.
    """
    base = _base_result()
    assert hist._results_equal(base, base)
    assert hist._results_equal(base, replace(base))          # a fresh object, same values

    blind = []
    for f in dc_fields(AnalysisResult):
        if hist._results_equal(base, _mutate(base, f.name)):
            blind.append(f.name)
    assert not blind, (
        f"EventHistory.checkpoint cannot see a change to {blind}. Every field of "
        "AnalysisResult must be compared — see core/history.py::_results_equal.")


def test_a_provenance_only_change_is_a_change():
    """The specific blindness that cost the capture rate."""
    base = _base_result()
    prov = dict(base.payload.provenance)
    prov["analyzed_duration_s"] = 0.9
    moved = replace(base, payload=EventSet(base.payload.df.copy(deep=True), prov))
    assert not hist._results_equal(base, moved)


def test_a_deep_copied_result_still_compares_equal():
    """The other direction, and the one that would cost the user their undo stack.

    ``_snapshot`` deep-copies the frame AND the provenance, so a comparison that reported
    a copy as different would push a phantom step on every single ``checkpoint_all()`` —
    re-dirtying a just-saved project and evicting real curation at MAX_DEPTH, which is the
    exact failure ``_OPTIONAL_COLUMNS`` already exists to prevent.
    """
    base = _base_result()
    base.payload.provenance["exclude_regions"] = [[0, 10], [20, 30]]
    base.payload.provenance["baseline_warnings"] = ["short window"]
    base.payload.provenance["polarity_evidence"] = np.array([1.0, -1.0, np.nan])
    for _ in range(3):
        assert hist._results_equal(base, hist._snapshot(base)), (
            "a snapshot of an untouched result compares as changed — every checkpoint "
            "would push a phantom step")


def test_an_uncomparable_provenance_value_is_treated_as_equal():
    """The conservative direction, chosen deliberately: a value whose ``__eq__`` is identity
    would compare unequal against its own deep copy forever. A missed difference costs a
    coarser undo label; a phantom difference costs the stack."""
    class _Opaque:
        pass

    base = _base_result()
    base.payload.provenance["model"] = _Opaque()
    assert hist._results_equal(base, hist._snapshot(base))


# ================================================================================================
# the three differences this DELIBERATELY does not see
# ================================================================================================

def test_a_dtype_only_change_is_not_a_change():
    """int64 -> float64 holding the same values. Invisible ON PURPOSE: the question is
    "did the user's data change", and pandas dtype churn (a column round-tripping through
    parquet, ``added`` materialised late) is not that. Making it visible re-opens the
    phantom-step failure ``_OPTIONAL_COLUMNS`` was written to close."""
    base = _base_result()
    df = base.payload.df.copy(deep=True)
    df["start_idx"] = df["start_idx"].astype("float64")
    other = replace(base, payload=EventSet(df, dict(base.payload.provenance)))
    assert hist._results_equal(base, other)


def test_a_pure_index_relabelling_is_not_a_change():
    """Values are compared positionally (``to_numpy``). An EventSet's row identity is
    ``event_id``, which IS compared; the pandas index is not data."""
    base = _base_result()
    df = base.payload.df.copy(deep=True)
    df.index = pd.RangeIndex(start=100, stop=100 + len(df))
    other = replace(base, payload=EventSet(df, dict(base.payload.provenance)))
    assert hist._results_equal(base, other)


def test_negative_zero_is_not_a_change():
    base = _base_result()
    df = base.payload.df.copy(deep=True)
    df.iat[0, df.columns.get_loc("dwell_s")] = 0.0
    a = replace(base, payload=EventSet(df, dict(base.payload.provenance)))
    df2 = df.copy(deep=True)
    df2.iat[0, df2.columns.get_loc("dwell_s")] = -0.0
    b = replace(base, payload=EventSet(df2, dict(base.payload.provenance)))
    assert hist._results_equal(a, b)


def test_row_order_and_row_count_are_still_changes():
    """The complement — what the positional comparison DOES see, so the three exemptions
    above cannot be read as "the frame is barely compared"."""
    base = _base_result()
    df = base.payload.df.iloc[::-1].reset_index(drop=True)
    assert not hist._results_equal(
        base, replace(base, payload=EventSet(df, dict(base.payload.provenance))))
    df2 = base.payload.df.iloc[:2].reset_index(drop=True)
    assert not hist._results_equal(
        base, replace(base, payload=EventSet(df2, dict(base.payload.provenance))))


def test_checkpoint_all_on_an_untouched_session_pushes_nothing(detected, qtbot):
    """End to end: the Edit menu opening (which calls ``checkpoint_all``) must not invent
    steps now that ten more fields are compared.

    THIS ONE PASSES BEFORE AND AFTER, deliberately, and it is the only test here that
    does. It is not a repro — it is the non-regression side of the change, and it is the
    failure the widened comparison could plausibly have caused: phantom "Edit events"
    steps on every menu open, re-dirtying a just-saved project and evicting real curation
    at MAX_DEPTH. Every other test in this file fails against the pre-fix package.
    """
    window, doc_id, _res = detected
    qtbot.wait(5)
    depth = window._history.depth()
    for _ in range(4):
        assert window._history.checkpoint_all() is False
    assert window._history.depth() == depth


def test_baseline_only_detection_with_fewer_events_is_not_named_merge(detected):
    window, doc_id, original = detected
    key = window._undo_key()
    frame = original.payload.df.iloc[:2].copy(deep=True)
    params = dict(original.params)
    params['baseline_open_pore_reference'] = {'mode': 'constant', 'value_pA': I0}
    rerun = replace(original, params=params, payload=EventSet(frame, dict(original.payload.provenance)))
    window._session.add_result(rerun)
    window._history.checkpoint(key)
    assert window._history.undo_label(key) == 'Re-run detection'
    window._history.undo(key)
    pd.testing.assert_frame_equal(window._session.results.get(doc_id, 0, 'detect.schmitt').payload.df,
                                  original.payload.df)
