"""C21 / C10 / C1 through the REAL widgets and the REAL analysis modules (pytest-qt).

The pure-layer half of this invariant is ``tests/analysis/test_flag_invariant.py``. This
file drives the three surfaces a user actually touches, because the whole lesson of the
defect class is that a guarantee proved in one module does not automatically arrive at the
call site:

* **the Event Inspector's mask resize** (C21) -- a drag used to blank ``flag`` while keeping
  ``accepted=False``, leaving a rejected row with no reason;
* **the ML detector on a sub-range** (C10 + C2) -- the boundary-clip rule never reached it,
  and its depth moved with the user's zoom;
* **"Mark selection as missed event"** -- the third private copy of the same rebuild, which
  asserted ``accepted=True`` unconditionally.

The two real-data cases run against the owner's -600 mV apo-ferritin recording and skip when
it is not on this machine.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
from pathlib import Path
from types import SimpleNamespace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import ml_detect, ml_modules
from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import FLAG_CLIPPED, EventSet, has_flag
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import CalibratedChannel, TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_inspector import EventInspectorWindow
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 50_000
I0 = -1000.0
DEPTH = 300.0

ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(not ABF.exists(), reason="owner's -600 mV recording not present")


# ================================================================================================
# builders
# ================================================================================================
def _signal(spans, *, n=N, depth=DEPTH, polarity=-1, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] += polarity * depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name="clip.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _session_with(tmp_path, signal) -> tuple[Session, str]:
    session = Session()
    return session, session.add_document(_doc(tmp_path, signal))


def _inspector(qtbot, session):
    insp = EventInspectorWindow(session)
    qtbot.addWidget(insp)
    return insp


def _set_region_to_indices(insp, session, doc_id, start_idx, end_idx) -> None:
    """Drive the region exactly as a finished user drag would."""
    xvals = session.x_axis(doc_id).values
    insp._region.setRegion((float(xvals[int(start_idx)]), float(xvals[int(end_idx)])))
    insp._region.sigRegionChangeFinished.emit(insp._region)


def _clipped_detection(tmp_path):
    """A detection whose FIRST row runs off the start of the recording, so it is clipped."""
    signal = _signal([(0, 300), (20_000, 20_400)])
    session, doc_id = _session_with(tmp_path, signal)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    assert has_flag(df["flag"], FLAG_CLIPPED).any(), "fixture must produce a clipped row"
    return session, doc_id, signal, EventSet(df, prov)


# ================================================================================================
# C21 -- the Event Inspector's mask resize
# ================================================================================================
def test_c21_resizing_a_clipped_event_never_leaves_it_rejected_with_no_reason(
    qapp, qtbot, tmp_path
):
    """Before the fix the resize wrote ``flag=''`` and left ``accepted=False``: the row was
    still out of every fit, but nothing in the table, the filter or the count could say why
    -- and ``n_clipped_boundary`` went on claiming a clipped event no row admitted to."""
    session, doc_id, _signal_, es = _clipped_detection(tmp_path)
    df = es.df
    row = 0
    assert has_flag(df["flag"].iloc[row], FLAG_CLIPPED)
    assert bool(df["accepted"].iloc[row]) is False

    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))
    insp.show_event(int(df["event_id"].iloc[row]))

    # Shrink the trailing edge; the row STILL starts at sample 0, so it is still clipped.
    _set_region_to_indices(insp, session, doc_id, 0, int(df["end_idx"].iloc[row]) - 50)
    assert int(df["start_idx"].iloc[row]) == 0
    assert has_flag(df["flag"].iloc[row], FLAG_CLIPPED), (
        "a still-clipped row must keep saying so; blanking the flag is C21"
    )
    assert bool(df["accepted"].iloc[row]) is False


def test_c21_dragging_an_event_off_the_boundary_clears_the_reason_and_the_rejection(
    qapp, qtbot, tmp_path
):
    """The other half. An automatic rejection is re-judged, not inherited: once the event no
    longer touches the edge, the reason is gone and so is the rejection. Inheriting it would
    reproduce C21 from the opposite direction -- rejected forever, reason erased."""
    session, doc_id, _signal_, es = _clipped_detection(tmp_path)
    df = es.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))
    insp.show_event(int(df["event_id"].iloc[0]))

    _set_region_to_indices(insp, session, doc_id, 40, int(df["end_idx"].iloc[0]))
    assert int(df["start_idx"].iloc[0]) == 40
    assert str(df["flag"].iloc[0]) == ""
    assert bool(df["accepted"].iloc[0]) is True


def test_a_resize_leaves_no_stale_clipped_count_behind(qapp, qtbot, tmp_path):
    """The provenance/row disagreement, through the widget that creates it. The Inspector
    edits the EventSet's frame IN PLACE and never rebuilds the object, so a count corrected
    only where the object is CONSTRUCTED stays at 1 with zero rows flagged -- and that
    number is what a project save writes to disk. The count is derived where it is READ, so
    the disagreement is not representable rather than merely not produced."""
    session, doc_id, _signal_, es = _clipped_detection(tmp_path)
    df = es.df
    assert int(es.provenance["n_clipped_boundary"]) == 1
    assert int(has_flag(df["flag"], FLAG_CLIPPED).sum()) == 1

    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))
    insp.show_event(int(df["event_id"].iloc[0]))
    _set_region_to_indices(insp, session, doc_id, 50, int(df["end_idx"].iloc[0]))

    assert not has_flag(df["flag"].iloc[0], FLAG_CLIPPED)
    assert int(has_flag(df["flag"], FLAG_CLIPPED).sum()) == 0
    assert int(es.provenance["n_clipped_boundary"]) == 0


def test_a_hand_reject_still_survives_a_resize(qapp, qtbot, tmp_path):
    """The guard on the fix: ``accepted`` is now WRITTEN by the rebuild, so it has to be
    shown that a curator's own decision is not what gets overwritten."""
    signal = _signal([(10_000, 10_400), (20_000, 20_400)])
    session, doc_id = _session_with(tmp_path, signal)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(prov))
    insp.show_event(int(df["event_id"].iloc[0]))
    insp.set_accepted_current(False)                     # a bare reject == by hand

    _set_region_to_indices(insp, session, doc_id,
                           int(df["start_idx"].iloc[0]) + 20, int(df["end_idx"].iloc[0]))
    assert bool(df["accepted"].iloc[0]) is False
    assert str(df["flag"].iloc[0]) == ""


@_needs_abf
def test_c21_on_the_owners_recording(qapp, qtbot):
    """The same edit on real data: a real translocation cut by the start of a segment."""
    from nanoscope.io.base import open_document

    doc = open_document(ABF)
    ci = next(i for i, c in enumerate(doc.meta.channels) if c.kind is ChannelKind.CURRENT)
    cur = np.asarray(doc.channels[ci], dtype=np.float64)
    fs = float(doc.meta.sample_rate_hz)
    # A global open-pore level, so the segment's own baseline cannot dive into the
    # half-event sitting at its first sample and hide it from the detector.
    i0 = float(np.median(cur))

    # Detect once to find a real event, then cut a segment straight through it.
    df0, _ = detect_events(cur, fs, min_dwell_s=2e-5, baseline=i0)
    ev = df0[(df0["dwell_s"] > 5.5e-3) & (df0["dwell_s"] < 7e-3)].iloc[0]
    s, e = int(ev["start_idx"]), int(ev["end_idx"])
    seg = np.ascontiguousarray(cur[(s + e) // 2: e + 200_000], dtype=np.float32)

    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=Path("/tmp/seg600.dat"), format_name="synthetic",
                         n_samples=seg.size, sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(seg,)))

    df, prov = detect_events(seg, fs, min_dwell_s=2e-5, baseline=i0)
    assert int(prov["n_clipped_boundary"]) >= 1
    # the halved event: 3.18 ms against the whole event's 6.38 ms, i.e. the dwell in that
    # row is a lower bound and nothing more -- which is what the flag exists to say.
    assert float(df["dwell_s"].iloc[0]) < 0.75 * float(ev["dwell_s"])
    row = int(np.flatnonzero(has_flag(df["flag"], FLAG_CLIPPED))[0])
    assert bool(df["accepted"].iloc[row]) is False

    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(prov))
    insp.show_event(int(df["event_id"].iloc[row]))
    _set_region_to_indices(insp, session, doc_id, 0, int(df["end_idx"].iloc[row]) - 10)

    assert has_flag(df["flag"].iloc[row], FLAG_CLIPPED)
    assert bool(df["accepted"].iloc[row]) is False


# ================================================================================================
# C10 + C2 -- the ML detector, through the real module
# ================================================================================================
class _Ctx:
    def __init__(self, ch, meta, params):
        self.channel, self.meta, self.params = ch, meta, params

    def report_progress(self, *a, **k):
        pass

    def cancelled(self):
        return False


def _run_ml(monkeypatch, signal, regions, i0, i1, *, fs=FS):
    """Run the REAL ``MLDetect.run`` over ``[i0, i1)`` with ``regions`` (ABSOLUTE, inclusive)
    pinned, so the numbers come out of the module the GUI drives -- no TensorFlow involved."""
    arr = np.ascontiguousarray(signal, dtype=np.float32)
    i0 = max(0, min(int(i0), arr.size))      # the panel clamps before the run; mirror it
    i1 = max(0, min(int(i1), arr.size))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=Path("/tmp/ml.dat"), format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz=fs, t0_s=0.0)

    monkeypatch.setattr(
        ml_detect, "ml_detect_regions",
        lambda sig, f, model, **kw: ([(s - i0, e - i0) for s, e in regions], None, {}))
    monkeypatch.setattr(
        ml_detect, "load_keras_model",
        lambda _p: SimpleNamespace(window=3072, input_shape=(None, 3072, 1),
                                   output_shape=(None, 3072, 1)))
    return ml_modules.MLDetect().run(_Ctx(ch, meta, {
        "model_path": "/fake.keras", "range_start": i0, "range_stop": i1,
        "range_source": "visible"}))


def test_c10_an_ml_region_cut_by_the_zoom_rectangle_is_flagged_and_rejected(monkeypatch):
    """The boundary-clip rule used to live in ``detect_events``, so ``MLDetect`` -- which
    calls ``characterize`` directly -- admitted a dwell that is a pure artifact of where the
    user zoomed as a fully-measured, accepted event."""
    signal = _signal([(10_000, 10_600)], n=40_000)
    mid = 10_300
    res = _run_ml(monkeypatch, signal, [(mid, 10_599)], mid, 20_000)

    row = res.payload.df.iloc[0]
    assert int(row["start_idx"]) == mid
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False
    assert int(res.payload.provenance["n_clipped_boundary"]) == 1
    assert any("cut off by the edge of the span" in w for w in res.warnings), res.warnings


def test_c10_a_region_ending_at_the_range_stop_is_flagged_too(monkeypatch):
    signal = _signal([(10_000, 10_600)], n=40_000)
    res = _run_ml(monkeypatch, signal, [(10_000, 10_299)], 2_000, 10_300)
    row = res.payload.df.iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False


def test_an_interior_ml_region_is_untouched(monkeypatch):
    """The control. Over-flagging would be its own defect."""
    signal = _signal([(10_000, 10_600)], n=40_000)
    res = _run_ml(monkeypatch, signal, [(10_000, 10_599)], 2_000, 20_000)
    row = res.payload.df.iloc[0]
    assert str(row["flag"]) == ""
    assert bool(row["accepted"]) is True
    assert int(res.payload.provenance["n_clipped_boundary"]) == 0
    assert res.warnings == () or all("cut off" not in w for w in res.warnings)


def test_c2_ml_depth_does_not_move_when_the_user_zooms(monkeypatch):
    """The characterisation baseline is measured on the WHOLE recording now, so the same
    event reports the same depth from every range. Before the fix the 10 ms block grid was
    anchored at ``range_start``, so the depth had a period of exactly one block."""
    signal = _signal([(10_000, 10_600)], n=200_000)
    depths = []
    for shift in range(0, 600, 100):
        res = _run_ml(monkeypatch, signal, [(10_000, 10_599)],
                      5_000 - shift, 20_000)
        depths.append(float(res.payload.df.iloc[0]["depth_pA"]))
    assert max(depths) - min(depths) < 1e-9, depths

    widths = []
    for pad in (2_000, 20_000, 80_000):
        res = _run_ml(monkeypatch, signal, [(10_000, 10_599)],
                      10_000 - pad, min(10_600 + pad, 200_000))
        widths.append(float(res.payload.df.iloc[0]["depth_pA"]))
    assert max(widths) - min(widths) < 1e-9, widths
    assert abs(depths[0] - DEPTH) < 0.15 * DEPTH, depths[0]


def test_the_pre_fix_recipe_really_did_move_with_the_zoom():
    """Proves the fixture above is capable of catching the defect: the OLD recipe --
    ``estimate_baseline`` on the SLICE at its default 10 ms window -- run on the very same
    trace and ranges, swings wildly. Without this, a fixture that happened not to exercise
    the block grid would make ``test_c2_...`` pass vacuously."""
    from nanoscope.analysis.baseline import estimate_baseline
    from nanoscope.analysis.characterize import characterize
    from nanoscope.analysis.filters import noise_rms

    signal = _signal([(10_000, 10_600)], n=200_000).astype(np.float64)
    depths = []
    for shift in range(0, 600, 100):
        i0, i1 = 5_000 - shift, 20_000
        sl = signal[i0:i1]
        base = estimate_baseline(sl, FS, polarity=None)          # the old default window
        df = characterize(sl, base, np.array([10_000 - i0]), np.array([10_600 - i0]),
                          FS, -1, noise_rms=float(noise_rms(sl - base)),
                          clip_bounds=(-1, sl.size + 1))         # the old: no clip rule
        depths.append(float(df.iloc[0]["depth_pA"]))
    assert max(depths) - min(depths) > 1.0, depths


def test_the_ml_provenance_says_which_baseline_the_depths_came_from(monkeypatch):
    signal = _signal([(10_000, 10_600)], n=40_000)
    prov = _run_ml(monkeypatch, signal, [(10_000, 10_599)], 2_000, 20_000).payload.provenance
    assert prov["characterize_baseline_scope"] == "recording"

    # The window is no longer a constant of `ml_modules` (it was
    # ``_CHAR_BASELINE_WINDOW_S = 0.20``, against the Schmitt path's 10 ms, which is
    # exactly how one event came to report two depths). It is whichever tracking
    # window the SHARED `BaselineConfig` names, and the load-bearing assertion is
    # that both detectors name the same one -- checkable, unlike a number.
    from nanoscope.analysis.baseline import DEFAULT_BASELINE

    assert float(prov["characterize_baseline_window_s"]) == pytest.approx(
        DEFAULT_BASELINE.tracking_window_s)
    assert prov["baseline_fingerprint"] == DEFAULT_BASELINE.fingerprint(FS)
    df, schmitt_prov = detect_events(signal, FS, min_dwell_s=1e-4)
    assert prov["baseline_fingerprint"] == schmitt_prov["baseline_fingerprint"]


@_needs_abf
def test_c2_on_the_owners_recording(monkeypatch):
    """The measured case from the hunt: one real 6.1 ms translocation, the analysed range
    slid by 100 samples at a time. Depth used to swing 8.0 -> 38.9 pA (and negative in
    between) against a true 67.7 pA; it is now constant and correct.

    THE SPAN IS NAMED, NOT SELECTED BY DWELL. It used to be picked out with
    ``5.5 ms < dwell < 7 ms``, and under the event-masked baseline no such row
    exists: the mask is dilated 1 ms each side, so this file's bursts (10.7% duty)
    are detected as single ~17 ms events rather than as strings of fragments. That
    is a change in what the detector GROUPS, and it has nothing to do with what this
    test measures -- whether the reported depth of one fixed span moves with the
    user's zoom rectangle. Naming the span makes the test immune to it.

    ``[88841, 89148)`` is the event the whole investigation is written around: true
    depth 66.3 pA against the modal open pore (-34.4 pA), reported as 32.04 pA by the
    Schmitt path, 37.54 pA by the row rebuild and 67.69 pA by this one.
    """
    from nanoscope.analysis.baseline import DEFAULT_BASELINE, characterisation_baseline
    from nanoscope.analysis.characterize import characterize
    from nanoscope.analysis.filters import noise_rms
    from nanoscope.io.base import open_document

    doc = open_document(ABF)
    ci = next(i for i, c in enumerate(doc.meta.channels) if c.kind is ChannelKind.CURRENT)
    cur = np.asarray(doc.channels[ci], dtype=np.float64)
    fs = float(doc.meta.sample_rate_hz)

    s, e = 88_841, 89_148

    base = characterisation_baseline(cur, fs, config=DEFAULT_BASELINE,
                                     health=False).baseline
    truth = float(characterize(cur, base, np.array([s]), np.array([e]), fs, -1,
                               noise_rms=float(noise_rms(cur - base))).iloc[0]["depth_pA"])
    # ...and that is the OPEN-PORE depth, not a number read off a baseline sitting
    # inside the blockade: the modal current of this file is -34.43 pA and the event
    # level is -100.71 pA.
    assert truth == pytest.approx(66.3, abs=1.0), truth

    depths = []
    for k in range(6):
        res = _run_ml(monkeypatch, cur, [(s, e - 1)], s - 5_000 - 100 * k, e + 5_000, fs=fs)
        depths.append(float(res.payload.df.iloc[0]["depth_pA"]))
    assert max(depths) - min(depths) < 1e-9, depths
    assert depths[0] == pytest.approx(truth, rel=1e-6)


# ================================================================================================
# the hand-added missed event -- the third copy of the same rebuild
# ================================================================================================
def test_a_hand_added_event_running_off_the_end_is_not_asserted_to_be_fine(qapp, qtbot,
                                                                          tmp_path):
    """The path set ``accepted = True`` unconditionally ("a hand-added event is, by
    definition, real"). It is real; its DWELL is still a lower bound when the selection runs
    off the recording, and saying otherwise is the same lie the merge used to tell."""
    signal = _signal([(10_000, 10_400)])
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))

    panel = win._analysis_panel
    panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region((N - 300) / FS, N / FS)   # runs to the last sample
    new_id = panel.add_current_selection_as_event()
    assert new_id is not None

    out = win._analysis_panel.current_detection().payload.df.set_index("event_id").loc[new_id]
    assert has_flag(out["flag"], FLAG_CLIPPED)
    assert bool(out["accepted"]) is False


def test_an_interior_hand_added_event_is_accepted_as_before(qapp, qtbot, tmp_path):
    """The control: the ordinary case must not have changed."""
    signal = _signal([(10_000, 10_400), (20_000, 20_400)])
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))

    panel = win._analysis_panel
    panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(30_000 / FS, 30_400 / FS)
    new_id = panel.add_current_selection_as_event()
    assert new_id is not None

    out = win._analysis_panel.current_detection().payload.df.set_index("event_id").loc[new_id]
    assert str(out["flag"]) == ""
    assert bool(out["accepted"]) is True


# ================================================================================================
# the ANALYSED SPAN -- an ML sub-range row must keep its boundary judgement
# ================================================================================================
# The ML detector flagged the row correctly (above); nothing carried the RANGE, so the
# first rebuild re-judged it against the whole recording, found itself nowhere near an
# edge and -- ``clipped_by_boundary`` being derived -- dropped it and un-rejected itself.
# No edit was required: a plain gap merge did it. The pure-layer half of these is in
# ``tests/analysis/test_span_invariant.py``; these drive the REAL MLDetect module.
def test_the_ml_detector_stamps_the_range_it_analysed(monkeypatch):
    from nanoscope.analysis.events import analysed_span

    signal = _signal([(10_000, 10_600)], n=40_000)
    res = _run_ml(monkeypatch, signal, [(10_000, 10_599)], 2_000, 20_000)
    assert analysed_span(res.payload.provenance, -1) == (2_000, 20_000)

    whole = _run_ml(monkeypatch, signal, [(10_000, 10_599)], 0, 40_000)
    assert analysed_span(whole.payload.provenance, -1) == (0, 40_000)


def test_a_merge_of_ml_rows_does_not_un_reject_what_the_zoom_rectangle_cut(monkeypatch):
    """THE §2.2 headline, through the module the GUI drives. Two fragments, the first cut
    by ``range_start``; the merged row used to come back accepted with ``flag=''`` and a
    dwell that is a pure artifact of where the user drew the rectangle."""
    from nanoscope.analysis.merge import merge_by_gap

    signal = _signal([(10_000, 10_600)], n=40_000)
    mid = 10_300
    res = _run_ml(monkeypatch, signal, [(mid, 10_449), (10_470, 10_599)], mid, 20_000)
    es = res.payload
    assert has_flag(es.df["flag"].iloc[0], FLAG_CLIPPED)

    merged, n = merge_by_gap(es, np.asarray(signal, dtype=np.float64), FS, max_gap_s=1e-3)
    assert n == 1 and len(merged.df) == 1
    row = merged.df.iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED), row["flag"]
    assert bool(row["accepted"]) is False
    assert int(merged.provenance["n_clipped_boundary"]) == 1
    assert len(merged.accepted.df) == 0


def test_an_inspector_resize_of_an_ml_row_keeps_the_range_clip(monkeypatch, qapp, qtbot,
                                                               tmp_path):
    """The mask resize is the other rebuild path. Nudging the mask by one sample used to
    be enough to lose the flag AND the rejection."""
    signal = _signal([(10_000, 10_600)], n=40_000)
    mid = 10_300
    es = _run_ml(monkeypatch, signal, [(mid, 10_599)], mid, 20_000).payload
    df = es.df
    assert has_flag(df["flag"].iloc[0], FLAG_CLIPPED)

    session, doc_id = _session_with(tmp_path, signal)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))
    insp.show_event(int(df["event_id"].iloc[0]))
    _set_region_to_indices(insp, session, doc_id, mid, 10_599)

    assert has_flag(df["flag"].iloc[0], FLAG_CLIPPED)
    assert bool(df["accepted"].iloc[0]) is False


def test_an_ml_run_inside_a_segment_gets_the_segments_own_coordinates(monkeypatch,
                                                                     tmp_path):
    """A segment is a derived recording with its own doc_id and SEGMENT-LOCAL indices, so
    an ML range inside it is segment-local too -- and the segment's own edges must not be
    mistaken for anything else, nor the parent's."""
    from nanoscope.analysis.events import analysed_span

    parent = _signal([(9_900, 10_500), (20_000, 20_400)], n=40_000)
    seg = np.ascontiguousarray(parent[10_000:30_000])
    # a range that starts inside the segment, in the SEGMENT's coordinates
    res = _run_ml(monkeypatch, seg, [(200, 799), (10_000, 10_399)], 200, 15_000)
    assert analysed_span(res.payload.provenance, -1) == (200, 15_000)

    df = res.payload.df
    assert int(df["start_idx"].iloc[0]) == 200
    assert has_flag(df["flag"].iloc[0], FLAG_CLIPPED)      # cut by the RANGE start
    assert str(df["flag"].iloc[1]) == ""                   # interior: untouched

    row = recharacterize_span(
        np.asarray(seg, dtype=np.float64), FS, 200, 800,
        provenance=res.payload.provenance, prior=df.iloc[0]).iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False


def test_the_span_survives_a_project_save_and_reopen(monkeypatch, tmp_path):
    """If provenance is the carrier, it has to come back out of the file: a restored
    project's first merge must still judge the row against the range it was detected in."""
    from nanoscope.analysis.events import analysed_span
    from nanoscope.analysis.merge import merge_by_gap
    from nanoscope.core.project import ProjectState, ResultRec, load_project, save_project

    signal = _signal([(10_000, 10_600)], n=40_000)
    mid = 10_300
    es = _run_ml(monkeypatch, signal, [(mid, 10_449), (10_470, 10_599)], mid, 20_000).payload

    state = ProjectState(results=[ResultRec(
        doc_id="d", channel=0, module_id="detect.keras",
        events=es.df, provenance=dict(es.provenance))])
    back = load_project(save_project(tmp_path / "p.nanoscope", state))
    rec = back.results[0]
    assert analysed_span(rec.provenance, -1) == (mid, 20_000)

    restored = EventSet(rec.events, dict(rec.provenance))
    assert has_flag(restored.df["flag"].iloc[0], FLAG_CLIPPED)
    merged, _ = merge_by_gap(restored, np.asarray(signal, dtype=np.float64), FS,
                             max_gap_s=1e-3)
    assert has_flag(merged.df["flag"].iloc[0], FLAG_CLIPPED)
    assert bool(merged.df["accepted"].iloc[0]) is False


def test_a_legacy_ml_result_with_only_range_start_stop_is_still_judged_right(monkeypatch):
    """A result saved by an older build carries ``range_start``/``range_stop`` and no
    canonical span. It must not be re-judged against the whole recording on reopen."""
    from nanoscope.analysis.events import (
        ANALYSED_START_KEY, ANALYSED_STOP_KEY, analysed_span,
    )

    signal = _signal([(10_000, 10_600)], n=40_000)
    mid = 10_300
    es = _run_ml(monkeypatch, signal, [(mid, 10_599)], mid, 20_000).payload
    legacy = {k: v for k, v in es.provenance.items()
              if k not in (ANALYSED_START_KEY, ANALYSED_STOP_KEY)}
    assert analysed_span(legacy, 40_000) == (mid, 20_000)

    row = recharacterize_span(np.asarray(signal, dtype=np.float64), FS, mid, 10_600,
                              provenance=legacy, prior=es.df.iloc[0]).iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False
