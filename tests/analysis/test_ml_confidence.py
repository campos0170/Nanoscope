"""The ML detector KEEPS its probability (`core.confidence.ConfidenceTrack`).

`ml_modules.MLDetect.run` used to bind the probability `ml_detect_regions` returns to
``_prob`` and drop it, so no surface could show what the detector's events were decoded
from. These tests pin what is kept, on which samples, at what precision, and that it
survives every way a result is rebuilt -- with a FAKE model (the `_FakeEventModel` pattern
of the ML GUI tests): ``load_keras_model`` is monkeypatched, so no model file is read and
TensorFlow is never imported.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import nanoscope  # noqa: F401
from nanoscope.analysis import ml_detect
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.ml_modules import MLDetect
from nanoscope.analysis.ml_preprocess import PREPROCESS_PRESETS, MLDecodeConfig
from nanoscope.analysis.modules import reconcile_result
from nanoscope.analysis.scoring import ensure_added_column
from nanoscope.core import history as H
from nanoscope.core.confidence import (
    QUANT_STEP,
    ConfidenceTrack,
    quantize_probability,
)
from nanoscope.core.document import CalibratedChannel, TraceDocument
from nanoscope.core.history import EventHistory, _results_equal, _snapshot
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
W = 1024
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400), (70_000, 70_200)]


def _trace(n: int = 90_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return x.astype(np.float32)


def _context(data: np.ndarray, params: dict) -> SimpleAnalysisContext:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=np.ascontiguousarray(data, dtype=np.float32),
                           units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(path=Path("synthetic_ml.dat"), format_name="synthetic",
                         n_samples=int(ch.data.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,), t0_s=0.0)
    return SimpleAnalysisContext(channel=ch, meta=meta, params=params)


class _FakeEventModel:
    """Fires on the deep, normalised event samples; a GRADED output (not 0/1) so the
    quantisation is actually exercised: ``|x|`` clipped to [0, 1]."""

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)
        return np.clip(np.abs(x), 0.0, 1.0).astype(np.float16)


class _WildModel:
    """A broken head: NaN, negative and > 1 outputs (a linear or clipped activation)."""

    def predict(self, arr, verbose=0):
        out = np.full(np.asarray(arr).shape, 0.5, dtype=np.float32)
        out[:, 0, 0] = np.nan
        out[:, 1, 0] = -0.5
        out[:, 2, 0] = 3.0
        return out


def _wrapper(model=None, window: int = W) -> ml_detect.KerasEventModel:
    return ml_detect.KerasEventModel(
        model=model or _FakeEventModel(), window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))


@pytest.fixture
def fake_model(monkeypatch):
    wrapper = _wrapper()
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    return wrapper


def _run(params: dict, data: np.ndarray | None = None, *, thresholds: bool = True):
    base = {"model_path": "/fake/model.keras"}
    if thresholds:
        base.update(enter=0.5, exit=0.3)
    base.update(params)
    return MLDetect().run(_context(_trace() if data is None else data, base))


# ------------------------------------------------------------------------------------------
# the window-start rule, now a function the inference loop and the track share
# ------------------------------------------------------------------------------------------

def _old_loop_starts(n, W, anchor_tail, last_start_inclusive):
    """VERBATIM the start rule `predict_probability` carried inline before it was moved
    into `ml_detect.window_starts` -- the oracle the move must agree with."""
    step = max(1, W // 2)
    if anchor_tail:
        starts = list(range(0, max(1, n - W + 1), step))
        if n >= W and starts[-1] != n - W:
            starts.append(n - W)
    elif last_start_inclusive:
        starts = list(range(0, max(1, n - W + 1), step))
    else:
        starts = list(range(0, max(0, n - W), step)) or [0]
    return starts


@pytest.mark.parametrize("anchor_tail,last_inclusive",
                         [(True, False), (True, True), (False, False), (False, True)])
@pytest.mark.parametrize("W_", [1, 2, 7, 64, 1024])
def test_window_starts_is_the_loop_rule_for_every_policy(anchor_tail, last_inclusive, W_):
    for n in [1, 2, 3, W_ - 1, W_, W_ + 1, 2 * W_, 2 * W_ + 1, 5 * W_ + 3, 10_007]:
        if n < 1:
            continue
        assert ml_detect.window_starts(
            n, W_, anchor_tail=anchor_tail, last_start_inclusive=last_inclusive
        ) == _old_loop_starts(n, W_, anchor_tail, last_inclusive)


class _Ones:
    def predict(self, arr, verbose=0):
        return np.ones(np.asarray(arr).shape, dtype=np.float32)


@pytest.mark.parametrize("anchor_tail,last_inclusive", [(True, False), (False, False),
                                                        (False, True)])
@pytest.mark.parametrize("n", [700, 1024, 1536, 2048, 2049, 5000])
def test_window_starts_says_exactly_which_samples_the_loop_covered(anchor_tail,
                                                                   last_inclusive, n):
    """The coverage the track records IS the loop's: an all-ones model leaves exactly the
    uncovered samples at 0, and they start where ``max(starts) + W`` says."""
    prob = ml_detect.predict_probability(_wrapper(_Ones()), np.zeros(n, np.float32),
                                         anchor_tail=anchor_tail,
                                         last_start_inclusive=last_inclusive)
    starts = ml_detect.window_starts(n, W, anchor_tail=anchor_tail,
                                     last_start_inclusive=last_inclusive)
    covered = min(n, max(starts) + W)
    assert np.all(prob[:covered] == 1.0)
    assert np.all(prob[covered:] == 0.0)


# ------------------------------------------------------------------------------------------
# what MLDetect keeps
# ------------------------------------------------------------------------------------------

def test_ml_result_keeps_the_probability_as_one_byte_per_analysed_sample(fake_model):
    data = _trace()
    res = _run({}, data)
    track = res.confidence
    assert isinstance(track, ConfidenceTrack)
    assert track.codes.dtype == np.uint8
    assert track.n == data.size and track.start == 0 and track.stop == data.size
    assert track.nbytes == data.size                       # never float64 per sample
    assert not track.codes.flags.writeable
    # The time base and the decode it was compared against, from THIS run.
    assert track.fs == FS and track.window == W and track.hop == W // 2
    assert track.enter == pytest.approx(0.5) and track.exit == pytest.approx(0.3)
    assert track.smoothing_kernel == res.payload.provenance["smooth_kernel"]
    assert track.model_file == "model.keras"
    assert track.n_windows == len(ml_detect.window_starts(data.size, W))
    assert track.covered == data.size                      # the app default anchors the tail

    # Equal to the model's own probability within half a quantisation step.
    _regions, prob, _prov = ml_detect.ml_detect_regions(
        data, FS, fake_model, enter=0.5, exit=0.3)
    assert np.max(np.abs(track.probability() - prob)) <= QUANT_STEP / 2 + 1e-6
    # ...and it is high on the events the table holds, low on the open pore.
    for s, e in _EVENTS:
        assert track.probability(s + 50, e - 50).mean() > 0.5
    assert track.probability(20_000, 25_000).max() < 0.3


def test_a_sub_range_run_keeps_its_track_on_the_documents_samples(fake_model):
    i0, i1 = 25_000, 60_000
    res = _run({"range_source": "visible", "range_start": i0, "range_stop": i1})
    track = res.confidence
    assert track.start == i0 and track.stop == i1 and track.n == i1 - i0
    # Document sample 30,100 is inside an event; the track reads it there, not at 5,100.
    assert track.at(30_100) is not None and track.at(30_100) > 0.5
    assert track.at(i0 - 1) is None and track.at(i1) is None
    assert track.probability(0, i0).size == 0


def test_a_notebook_tail_policy_records_the_samples_no_window_saw(fake_model):
    """The ferritin loop leaves the last stride with no window; `predict_probability`
    hands it back as 0. The track says how far the windows reached instead."""
    pre = PREPROCESS_PRESETS["apo_raw_global"].replace(anchor_tail=False,
                                                        last_start_inclusive=False)
    n = 90_000
    res = _run({"ml_preprocess": pre.to_dict()}, _trace(n))
    track = res.confidence
    starts = ml_detect.window_starts(n, W, anchor_tail=False, last_start_inclusive=False)
    assert track.covered == max(starts) + W < n
    assert track.uncovered_s == pytest.approx((n - track.covered) / FS)
    assert track.at(track.covered_stop) is None             # "never asked", not "p = 0"
    assert track.at(track.covered_stop - 1) is not None


def test_out_of_range_model_output_is_clipped_and_counted(monkeypatch):
    wrapper = _wrapper(_WildModel())
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    track = _run({}).confidence
    assert track.n_out_of_range > 0
    p = track.probability()
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0


def test_quantize_probability_rounds_to_the_nearest_code():
    codes, bad = quantize_probability(np.array([0.0, 0.5, 1.0, 0.002, np.nan, -1, 2],
                                               dtype=np.float32))
    assert codes.tolist() == [0, 128, 255, 1, 0, 0, 255]
    assert bad == 3


def test_a_cancelled_run_keeps_nothing(fake_model):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    ctx = _context(_trace(), {"model_path": "/fake/model.keras"})
    ctx.cancel_flag = cancel
    res = MLDetect().run(ctx)
    assert res.confidence is None


# ------------------------------------------------------------------------------------------
# it rides through every rebuild of the result, by reference
# ------------------------------------------------------------------------------------------

def test_the_track_is_shared_not_copied_through_edits_and_undo(fake_model):
    res = _run({})
    track = res.confidence
    assert copy.copy(track) is track and copy.deepcopy(track) is track
    assert replace(res, warnings=()).confidence is track
    assert _snapshot(res).confidence is track              # an undo snapshot
    rebuilt = reconcile_result(res, res.payload)          # a merge / re-measure install
    assert rebuilt.confidence is track
    # ...and an undo checkpoint does not see a phantom difference in it.
    assert _results_equal(res, _snapshot(res))


def test_results_without_a_track_still_compare_and_default_to_none(fake_model):
    res = _run({})
    bare = replace(res, confidence=None)
    assert bare.confidence is None
    assert not _results_equal(res, bare)                   # gaining a track IS a change


def test_thresholds_adopted_from_a_decode_config_are_the_ones_kept(fake_model):
    """With an ``ml_decode`` config and no top-level enter / exit, the run decodes with the
    config's thresholds (`MLDetect.run` adopts them); the track must carry THOSE, the same
    numbers the provenance reports, not the 0.5 / 0.3 parameter fallbacks."""
    dec = MLDecodeConfig(enter_threshold=0.7, exit_threshold=0.25, smoothing_kernel=5)
    res = _run({"ml_decode": dec.to_dict()}, thresholds=False)
    track, prov = res.confidence, res.payload.provenance
    assert (track.enter, track.exit) == (pytest.approx(0.7), pytest.approx(0.25))
    assert (prov["enter"], prov["exit"]) == (pytest.approx(track.enter),
                                             pytest.approx(track.exit))
    assert track.smoothing_kernel == prov["smooth_kernel"] == 5


# ------------------------------------------------------------------------------------------
# the undo stack COUNTS the tracks it keeps: each re-run step holds its own
# ------------------------------------------------------------------------------------------

def _history_on(res):
    """A Session holding one document, an EventHistory over it, and ``res`` re-keyed to
    that document and installed as its first (baseline) state."""
    sig = np.zeros(4096, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=Path("synthetic_ml_undo.dat"), format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(sig,)))
    history = EventHistory(session)
    key = (doc_id, 0, res.module_id)
    first = replace(res, doc_id=doc_id)
    _install(session, first)
    history.checkpoint(key)
    return session, history, key, first


def _install(session, res):
    ensure_added_column(res.payload.df)        # as the panel does on display
    session.add_result(res)


def _rerun(res, k: int, n: int | None = None):
    """``res`` as a genuine re-run: a moved setting and a NEW track of ``n`` samples."""
    n = res.confidence.n if n is None else n
    track = ConfidenceTrack.from_probability(
        np.full(n, (k % 200) / 255.0, dtype=np.float32), start=0, fs=FS, window=W,
        hop=W // 2, n_windows=1, covered=n, enter=0.5 + 0.01 * k, exit=0.3)
    events = type(res.payload)(res.payload.df.copy(deep=True), dict(res.payload.provenance))
    return replace(res, payload=events, params={**res.params, "enter": 0.5 + 0.01 * k},
                   confidence=track)


def _frames(history) -> int:
    return sum(e.nbytes for e in history._undo) + sum(e.nbytes for e in history._redo)


def test_each_rerun_step_counts_its_own_track_against_the_byte_budget(fake_model):
    session, history, key, first = _history_on(_run({}))
    n = first.confidence.n
    for k in range(1, 5):
        _install(session, _rerun(first, k))
        assert history.checkpoint(key)
    # Five runs: the four earlier ones on the stack, each with its own track; the live run's
    # track is the table's, which dropping a step would not free.
    assert len(history._undo) == 4
    assert history.nbytes() == _frames(history) + 4 * n


def test_curation_steps_on_one_run_count_its_track_once(fake_model):
    session, history, key, first = _history_on(_run({}))
    n = first.confidence.n
    live = session.results.get(*key)
    for row in range(3):
        live.payload.df.loc[row, "accepted"] = False
        assert history.checkpoint(key)
    # Three curation steps, all sharing the LIVE run's one track: nothing extra held.
    assert len(history._undo) == 3
    assert history.nbytes() == _frames(history)
    # A re-run makes that track a superseded one, held by four steps: counted ONCE.
    _install(session, _rerun(first, 1))
    assert history.checkpoint(key)
    assert history.nbytes() == _frames(history) + n


def test_the_byte_ceiling_evicts_old_reruns_for_their_tracks(fake_model, monkeypatch):
    session, history, key, first = _history_on(_run({}))
    n = first.confidence.n
    frame = H._sizeof(first)
    # Room for two earlier runs and their tracks, not three: before the tracks were
    # counted, the frames alone (a few kB) never reached any ceiling.
    monkeypatch.setattr(H, "MAX_BYTES", 2 * (n + frame) + n // 2)
    for k in range(1, 6):
        _install(session, _rerun(first, k))
        history.checkpoint(key)
    assert history.nbytes() <= H.MAX_BYTES
    assert len(history._undo) == 2                         # five steps pushed, three dropped
    assert history.evicted(key) == 3


def test_an_identical_rerun_commits_the_live_runs_track(fake_model):
    """Same settings, same rows, new track: no step (the track compares by identity), but
    the committed state must hold the LIVE track, not keep the superseded one alive."""
    session, history, key, first = _history_on(_run({}))
    again = replace(first, payload=type(first.payload)(first.payload.df.copy(deep=True),
                                                       dict(first.payload.provenance)),
                    confidence=_rerun(first, 7).confidence)
    _install(session, again)
    assert not history.checkpoint(key)
    assert history._pristine[key].confidence is again.confidence
    # ...so the step a later curation click pushes restores the live run's track.
    live = session.results.get(*key)
    live.payload.df.loc[0, "accepted"] = False
    assert history.checkpoint(key)
    assert history._undo[-1].result.confidence is again.confidence
