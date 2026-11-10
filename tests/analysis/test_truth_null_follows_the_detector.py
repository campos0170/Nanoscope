"""`truth.null_control` runs the detector it is given, not always Schmitt.

The Null control used to call `detect.detect_events` whatever table it was asked
about, so a CUSUM table was judged by a Schmitt null. `null_control` now takes the
detector function (``detector=``; ``None`` is `detect_events`, as before) and, WHEN
A DETECTOR IS GIVEN, for a recording whose polarity is not known in advance
(``SynthRecording(polarity=None)``), lets the detector measure the sign once on the
real run and holds it for the event-free and surrogate runs.

The default call is unchanged for every input: the same seven keys, the same notes,
the same numbers, and -- for ``polarity=None`` too -- the same polarity handed to
every run (``None``: each run measures its own, as it always did). A detector that
counted its own opposite direction (CUSUM) adds two keys, ``n_opposite`` and the
``polarity`` it was counted against; for a given detector the event-free refill
uses the detect kwargs' own ``baseline_config``; and a refused surrogate sends a
CUSUM reader to the opposite-direction count, not to ``n_event_free``.

``polarity=None`` is for `null_control` only: `sweep_threshold` and `recovery_map`
refuse it, and so does `null_control` when the given detector stamps no polarity to
hold.

Synthetic traces only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nanoscope.analysis import modules, truth
from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import (
    DEFAULT_CUSUM_H,
    DEFAULT_CUSUM_K,
    detect_events_cusum,
)
from nanoscope.analysis.truth import SynthRecording, null_control, synth_trace

SEVEN_KEYS = {
    "n_real", "n_event_free", "n_surrogate_mean", "n_surrogate_std",
    "estimated_fdr", "per_surrogate", "notes",
}

CUSUM_KW = dict(cusum_k=DEFAULT_CUSUM_K, cusum_h=DEFAULT_CUSUM_H)


def _clean_long_events(seed: int = 0):
    """3 s of white noise with 30 events of 2 to 6 ms, 3.3 to 5.3 sigma deep.

    Long enough for CUSUM's 1 ms minimum dwell and shallow enough that the frozen
    recovery level is estimated over several samples, which is where the detector
    is cleanest.
    """
    return synth_trace(
        duration_s=3.0, n_events=30, pink_frac=0.0, seed=seed,
        dwell_range_s=(2e-3, 6e-3), depth_frac_range=(0.05, 0.08),
    )


def test_null_control_runs_the_detector_it_is_given(monkeypatch):
    rec, _truth = _clean_long_events()
    calls: list[dict] = []

    def recorder(signal, fs, **kwargs):
        calls.append(dict(kwargs))
        return detect_events_cusum(signal, fs, **kwargs)

    # The default detector must not be reached at all when another one is given.
    def schmitt_must_not_run(*args, **kwargs):
        raise AssertionError("null_control ran detect_events although a detector was given")

    monkeypatch.setattr("nanoscope.analysis.detect.detect_events", schmitt_must_not_run)

    n_surrogates = 2
    out = null_control(
        rec, detect_kwargs=dict(cusum_k=0.9, cusum_h=30.0, merge_gap_s=None),
        n_surrogates=n_surrogates, rng=3, detector=recorder,
    )
    assert len(calls) == n_surrogates + 2      # real, event-free, one per surrogate
    for kwargs in calls:
        assert kwargs["cusum_k"] == 0.9 and kwargs["cusum_h"] == 30.0
        # "merge off" reaches the detector as None instead of being dropped, which
        # would have turned it into CUSUM's 0.5 ms default.
        assert "merge_gap_s" in kwargs and kwargs["merge_gap_s"] is None
        assert kwargs["polarity"] == rec.polarity
        assert kwargs["recording"] == rec.name
        assert "k_in" not in kwargs and "check_polarity" not in kwargs
    assert out["n_real"] > 0


def test_cusum_null_on_clean_long_events():
    rec, planted = _clean_long_events()
    out = null_control(
        rec, detect_kwargs=CUSUM_KW, n_surrogates=2, rng=11, detector=detect_events_cusum,
    )
    assert set(out) == SEVEN_KEYS | {"n_opposite", "polarity"}
    assert abs(out["n_real"] - len(planted)) <= 2
    assert out["n_event_free"] == 0
    assert out["per_surrogate"] == [0, 0]
    assert out["estimated_fdr"] == 0.0
    assert out["notes"] == ()
    assert out["n_opposite"] == 0 and type(out["n_opposite"]) is int
    # The direction the count was taken against: the recording's own, a plain int.
    assert out["polarity"] == rec.polarity == 1 and type(out["polarity"]) is int


def test_the_schmitt_result_shape_is_unchanged():
    rec, _truth = synth_trace(duration_s=3.0, n_events=30, pink_frac=0.0, seed=1)
    default = null_control(rec, n_surrogates=2, rng=1)
    # No n_opposite and no polarity: Schmitt does not count its opposite direction.
    assert set(default) == SEVEN_KEYS
    assert default["n_real"] > 0

    # ``detector=None`` IS detect_events: naming it explicitly changes nothing here.
    named = null_control(rec, n_surrogates=2, rng=1, detector=detect_events)
    np.testing.assert_equal(named, default)


def test_an_unset_polarity_is_measured_once_and_then_held(monkeypatch):
    rec, _truth = _clean_long_events()
    assert rec.polarity == 1
    unset = replace(rec, polarity=None)
    assert isinstance(unset, SynthRecording) and unset.polarity is None

    seen: list[object] = []

    def recorder(signal, fs, **kwargs):
        seen.append(kwargs["polarity"])
        return detect_events_cusum(signal, fs, **kwargs)

    # The event-free trace is built against a baseline that needs the sign too.
    baseline_of = truth._baseline_of
    baseline_polarity: list[object] = []

    def spy(recording, *args, **kwargs):
        baseline_polarity.append(recording.polarity)
        return baseline_of(recording, *args, **kwargs)

    monkeypatch.setattr(truth, "_baseline_of", spy)

    out = null_control(unset, detect_kwargs=CUSUM_KW, n_surrogates=2, rng=5, detector=recorder)

    # Measured by the detector on the real run, where the events are...
    assert seen[0] is None
    # ...and held for the event-free and surrogate runs, which have nothing to vote with.
    assert seen[1:] == [1, 1, 1]
    assert baseline_polarity == [1]

    # The real run is exactly a Detect run on Auto polarity.
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=None, recording=rec.name, **CUSUM_KW)
    assert prov["polarity_source"] == "measured" and prov["polarity"] == 1
    assert out["n_real"] == len(df)
    assert out["n_opposite"] == prov["n_opposite_isolated"]
    # The result names the sign that was measured and then held.
    assert out["polarity"] == 1 and type(out["polarity"]) is int

    # The caller's recording is not modified (it is frozen; the hold is a copy).
    assert unset.polarity is None


def test_the_default_call_leaves_an_unset_polarity_to_every_run(monkeypatch):
    """The hold is for a GIVEN detector only. The default call (no ``detector=``)
    has always handed ``polarity=None`` to each of its ``n_surrogates + 2`` runs and
    to the refill baseline, and let each one measure its own sign; it must keep
    doing exactly that, so its numbers are the ones it always returned for this
    input too. (Holding the first run's sign here changed them: measured on a pink
    trace, 21 event-free detections and 61 / 59 / 67 per surrogate became 24 and
    16 / 22 / 19.)
    """
    rec, _truth = synth_trace(duration_s=3.0, n_events=30, pink_frac=0.0, seed=1)
    unset = replace(rec, polarity=None)
    n_surrogates = 3

    seen: list[object] = []

    def recorder(signal, fs, **kwargs):
        seen.append(kwargs["polarity"])
        return detect_events(signal, fs, **kwargs)

    baseline_of = truth._baseline_of
    baseline_polarity: list[object] = []

    def spy(recording, *args, **kwargs):
        baseline_polarity.append(recording.polarity)
        return baseline_of(recording, *args, **kwargs)

    monkeypatch.setattr("nanoscope.analysis.detect.detect_events", recorder)
    monkeypatch.setattr(truth, "_baseline_of", spy)

    out = null_control(unset, n_surrogates=n_surrogates, rng=3)

    # Every run was handed None, and so was the refill baseline: nothing was held.
    assert seen == [None] * (n_surrogates + 2)
    assert baseline_polarity == [None]
    # The Schmitt result shape, with no direction reported.
    assert set(out) == SEVEN_KEYS
    assert out["n_real"] > 0

    # The same recording with the detector NAMED is the other documented case: the
    # sign measured on the real run is held for the rest.
    seen.clear()
    baseline_polarity.clear()
    named = null_control(unset, n_surrogates=n_surrogates, rng=3, detector=recorder)
    assert seen[0] is None and seen[1:] == [rec.polarity] * (n_surrogates + 1)
    assert baseline_polarity == [rec.polarity]
    assert set(named) == SEVEN_KEYS            # Schmitt counts no opposite direction

    # And a recording that knows its polarity passes it to every run either way.
    seen.clear()
    null_control(rec, n_surrogates=1, rng=3)
    assert seen == [rec.polarity] * 3


_NEEDS_POLARITY = (
    "this study needs a known polarity; polarity=None is accepted by null_control only"
)


@pytest.mark.parametrize("study", ["sweep_threshold", "recovery_map"])
def test_sweep_and_recovery_refuse_an_unset_polarity(monkeypatch, study):
    """``polarity=None`` is a value `null_control` understands. The other two studies
    used to run to the end with it, every detection taking its own vote on the
    direction; they refuse it now, before anything is detected."""
    rec, _truth = synth_trace(duration_s=2.0, n_events=20, pink_frac=0.0, seed=1)
    unset = replace(rec, polarity=None)
    small = {
        "sweep_threshold": dict(k_values=np.array([4.0, 5.0, 6.0]), rng=1),
        "recovery_map": dict(
            dwell_grid=np.array([5e-4, 2e-3]), depth_grid=np.array([0.1, 0.3]),
            n_per_cell=3, seed=1),
    }[study]
    run = getattr(truth, study)

    calls: list[object] = []
    real_detect = detect_events

    def recorder(signal, fs, **kwargs):
        calls.append(kwargs["polarity"])
        return real_detect(signal, fs, **kwargs)

    monkeypatch.setattr("nanoscope.analysis.detect.detect_events", recorder)

    with pytest.raises(ValueError) as refused:
        run(unset, **small)
    assert str(refused.value) == _NEEDS_POLARITY
    # The refusal comes first, even when another argument is also wrong.
    with pytest.raises(ValueError) as refused:
        run(unset, **{**small, **(dict(k_values=np.array([5.0]))
                                  if study == "sweep_threshold" else dict(n_per_cell=0))})
    assert str(refused.value) == _NEEDS_POLARITY
    assert calls == []                     # nothing was detected on the way

    # With a known polarity the study runs as it always did.
    out = run(rec, **small)
    assert len(out) == (3 if study == "sweep_threshold" else 4)
    assert calls and set(calls) <= {rec.polarity, -rec.polarity}
    assert None not in calls


def test_a_detector_with_no_polarity_to_hold_is_refused_by_name():
    """With ``polarity=None`` the sign held for the study is the one the detector
    stamped on the real run. A detector that stamps none used to end in a bare
    ``KeyError: 'polarity'``; it is refused with a sentence that names it."""
    rec, _truth = _clean_long_events()
    unset = replace(rec, polarity=None)

    def detector_without_a_polarity_stamp(signal, fs, **kwargs):
        df, prov = detect_events_cusum(signal, fs, **kwargs)
        return df, {k: v for k, v in prov.items() if k != "polarity"}

    def detector_with_an_empty_polarity_stamp(signal, fs, **kwargs):
        df, prov = detect_events_cusum(signal, fs, **kwargs)
        return df, {**prov, "polarity": None}

    for detector in (detector_without_a_polarity_stamp, detector_with_an_empty_polarity_stamp):
        with pytest.raises(ValueError) as refused:
            null_control(unset, detect_kwargs=CUSUM_KW, n_surrogates=1, rng=1,
                         detector=detector)
        message = str(refused.value)
        assert detector.__name__ in message
        assert "polarity=None" in message and "known polarity" in message

    # The same detector is fine with a recording that knows its polarity: nothing
    # has to be read back from its provenance then.
    out = null_control(rec, detect_kwargs=CUSUM_KW, n_surrogates=1, rng=1,
                       detector=detector_without_a_polarity_stamp)
    assert out["polarity"] == rec.polarity and out["n_real"] > 0


def test_the_refill_uses_the_detectors_baseline_config(monkeypatch):
    """One study, one baseline: with a detector given, the event-free refill is
    drawn against the ``baseline_config`` in the detect kwargs. The default call
    keeps the refill it always had, whatever the kwargs carry."""
    rec, _truth = _clean_long_events()
    config = replace(DEFAULT_BASELINE, base_s=0.2)
    assert config != DEFAULT_BASELINE

    baseline_of = truth._baseline_of
    refill_configs: list[object] = []

    def spy(recording, config=None):
        refill_configs.append(config)
        return baseline_of(recording, config)

    monkeypatch.setattr(truth, "_baseline_of", spy)

    seen: list[object] = []

    def recorder(signal, fs, **kwargs):
        seen.append(kwargs.get("baseline_config"))
        return detect_events_cusum(signal, fs, **kwargs)

    out = null_control(
        rec, detect_kwargs=dict(CUSUM_KW, baseline_config=config),
        n_surrogates=1, rng=5, detector=recorder,
    )
    assert out["n_real"] > 0
    assert seen == [config, config, config]          # real, event-free, surrogate
    assert len(refill_configs) == 1 and refill_configs[0] is config

    # A given detector whose kwargs carry no config refills at the default, as before.
    refill_configs.clear()
    null_control(rec, detect_kwargs=CUSUM_KW, n_surrogates=1, rng=5, detector=recorder)
    assert refill_configs == [None]

    # The default (Schmitt) call is not touched: same refill as ever.
    refill_configs.clear()
    schmitt, _t = synth_trace(duration_s=3.0, n_events=30, pink_frac=0.0, seed=1)
    out = null_control(
        schmitt, detect_kwargs=dict(baseline_config=config), n_surrogates=1, rng=5)
    assert out["n_real"] > 0
    assert refill_configs == [None]


_REFUSED_PREFIX = (
    "the AAFT surrogates produced {mean:.0f} detections where this recording's own "
    "event-free noise produces {n_event_free}, so the surrogates are not the same "
    "noise and a false-positive RATE cannot be read off them. Phase randomisation is "
    "a null only for a trace that IS a stationary spectrum plus Gaussian noise; a "
    "recording with slow open-pore drift or heavy tails is not, and the surrogate's "
    "power comes back inflated (measured on this machine's recordings: every band, "
    "by up to 14x). "
)


def _refuse_the_surrogate(monkeypatch, seed: int):
    """Make every "surrogate" a trace full of events, so the premise check fails.

    The control's own event-free run finds nothing on these clean recordings; a
    surrogate that holds 40 long events is plainly not the same noise.
    """
    busy, _planted = synth_trace(
        duration_s=3.0, n_events=40, pink_frac=0.0, seed=seed,
        dwell_range_s=(2e-3, 6e-3), depth_frac_range=(0.1, 0.2),
    )
    monkeypatch.setattr(
        truth, "aaft_surrogate", lambda noise, rng=None: busy.data.astype(np.float64))


def test_a_refused_cusum_null_points_at_the_opposite_count(monkeypatch):
    rec, _truth = _clean_long_events()
    _refuse_the_surrogate(monkeypatch, seed=77)
    out = null_control(
        rec, detect_kwargs=CUSUM_KW, n_surrogates=2, rng=3, detector=detect_events_cusum)

    assert np.isnan(out["estimated_fdr"])
    assert out["n_event_free"] == 0 and out["n_surrogate_mean"] > 10
    [note] = out["notes"]
    assert "Use n_event_free instead" not in note
    assert note == _REFUSED_PREFIX.format(
        mean=out["n_surrogate_mean"], n_event_free=out["n_event_free"]) + (
        "For CUSUM the event-free count is a weak lower bound, because the stretches "
        "cut out include its own noise firings. Read the opposite-direction count."
    )
    # The count the note sends the reader to is in the same result.
    assert "n_opposite" in out and "polarity" in out


def test_a_refused_schmitt_null_reads_as_it_always_did(monkeypatch):
    rec, _truth = synth_trace(duration_s=3.0, n_events=30, pink_frac=0.0, seed=1)
    _refuse_the_surrogate(monkeypatch, seed=77)
    expected_tail = (
        "Use n_event_free instead -- it is the false-positive count on this "
        "recording's own noise, measured with no surrogate at all, and it is a "
        "lower bound."
    )
    for named in (None, detect_events):
        out = null_control(rec, n_surrogates=2, rng=3, detector=named)
        assert set(out) == SEVEN_KEYS
        assert np.isnan(out["estimated_fdr"])
        [note] = out["notes"]
        assert note == _REFUSED_PREFIX.format(
            mean=out["n_surrogate_mean"], n_event_free=out["n_event_free"]) + expected_tail


def test_a_seeded_cusum_null_repeats_and_cancels():
    rec, _truth = synth_trace(duration_s=3.0, n_events=20, pink_frac=0.3, seed=9,
                              dwell_range_s=(2e-3, 6e-3), depth_frac_range=(0.05, 0.1))
    kw = dict(detect_kwargs=dict(CUSUM_KW, filter_cutoff_hz=5000.0), n_surrogates=2,
              detector=detect_events_cusum)
    first = null_control(rec, rng=7, **kw)
    again = null_control(rec, rng=7, **kw)
    np.testing.assert_equal(again, first)          # NaN-safe, key for key
    assert first["n_real"] > 0

    with pytest.raises(OperationCancelled):
        null_control(rec, rng=7, cancel=lambda: True, **kw)


def test_the_public_kwargs_helper_is_the_private_one():
    """`modules.detector_kwargs` is what a caller outside the module uses to build
    the call a Detect run makes; it is `_detect_kwargs` under a public name."""
    assert modules.detector_kwargs is modules._detect_kwargs
    form = dict(
        detector="detect.cusum", polarity=None, k_in=5.0, k_out=2.0,
        noise_policy="auto", noise_window_s=0.25, cusum_k=0.75, cusum_h=28.0,
        min_dwell_s=1e-3, max_dwell_s=None, merge_gap_s=None,
        filter_cutoff_hz=None, filter_order=8, filter_kind="bessel",
        hp_enabled=False, baseline_method="masked_median", _private=1,
    )
    kwargs = modules.detector_kwargs(form, "detect.cusum")
    assert kwargs == dict(
        polarity=None, cusum_k=0.75, cusum_h=28.0, min_dwell_s=1e-3,
        max_dwell_s=None, merge_gap_s=None, filter_cutoff_hz=None,
        filter_order=8, filter_kind="bessel",
    )
