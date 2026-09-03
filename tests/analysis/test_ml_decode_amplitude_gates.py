"""The three student presets' REAL decode contract, and the amplitude gates it needs.

Until this landed, ``hg_student`` / ``zn_student`` / ``cu_student`` shipped a
deliberately neutral placeholder decode and said so in their combo label. The real
``DETECTOR_CONTRACT`` has now been read out of cell 1 of each StudentInference
notebook, so what is pinned here is that contract, verbatim:

  HG  ~/Desktop/HostGuest_StudentInference_corrected.ipynb
  Zn  ~/Desktop/Zn2IonCondition/Zn2Insulin_StudentInference.ipynb
  Cu  ~/Desktop/Cu2Data/Cu2Insulin_StudentInference.ipynb

Why pin it this hard. Cell 1 of all three notebooks exists BECAUSE these numbers
drifted: they had been repeated at each call site, and the single-recording event
summary ran ``enter=0.5`` while the cross-recording comparison ran ``enter=0.4`` --
so the summary described different detections from everything built on it. A single
definition is only worth anything if something checks it is still the right one.

What these tests cover:

  1. all nine contract values per preset, exactly;
  2. the gates default to 0.0 == OFF, so every stored config, saved run and
     QSettings-persisted user preset made before they existed still decodes
     identically -- checked by construction AND by round-trip through a dict that
     does not have the keys;
  3. each gate rejects what it should and passes what it should, on a synthetic
     trace built so exactly one gate can be responsible;
  4. the ORDER: the gates run AFTER ``filter_short_events`` and measure the FINAL,
     MERGED extent. That is what the notebook does -- ``postprocess_predictions``
     (cell 4) ends with the duration filter and returns; cells 8 and 15 gate the
     regions it RETURNED -- and it is observable, so it is pinned.

Pure numpy/scipy: no Qt, no TensorFlow.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from nanoscope.analysis import event_features as ef
from nanoscope.analysis import ml_preprocess as mp

FS = 50_000.0


# ===========================================================================
# 1. the contract, verbatim from cell 1
# ===========================================================================
#: Read out of ``DETECTOR_CONTRACT`` (and, for Cu, the two constants defined
#: immediately below it) in cell 1 of each notebook. ``0.0`` where the notebook
#: does not define the field at all -- hg and zn carry ``min_depth_sigma`` alone,
#: which is a real difference between the projects, not an omission.
NOTEBOOK_CONTRACT = {
    "hg_student": dict(
        enter_threshold=0.4, exit_threshold=0.15, merge_radius_ms=0.05,
        min_event_ms=0.1, smoothing_kernel=3, min_depth_sigma=4.0,
        min_depth_pa=0.0, min_fracblock=0.0,
        onset_sigma=0.0, offset_sigma=0.0,
    ),
    "zn_student": dict(
        enter_threshold=0.4, exit_threshold=0.15, merge_radius_ms=0.05,
        min_event_ms=0.1, smoothing_kernel=3, min_depth_sigma=4.0,
        min_depth_pa=0.0, min_fracblock=0.0,
        onset_sigma=0.0, offset_sigma=0.0,
    ),
    "cu_student": dict(
        enter_threshold=0.5, exit_threshold=0.15, merge_radius_ms=0.1,
        min_event_ms=0.05, smoothing_kernel=5, min_depth_sigma=4.0,
        min_depth_pa=10.0, min_fracblock=0.05,
        onset_sigma=0.0, offset_sigma=0.0,
    ),
}


@pytest.mark.parametrize("name", list(NOTEBOOK_CONTRACT))
def test_each_student_preset_is_the_notebooks_detector_contract_exactly(name):
    cfg = mp.DECODE_PRESETS[name]
    assert cfg.preset == name
    for key, want in NOTEBOOK_CONTRACT[name].items():
        assert getattr(cfg, key) == pytest.approx(want), f"{name}: {key}"


def test_cu_alone_raises_enter_and_carries_the_two_absolute_gates():
    """Cu's differences from its two siblings are DELIBERATE, so they are asserted
    as differences rather than as three independent numbers that happen to differ.

    ``enter`` is 0.5 where hg/zn use 0.4 because the Zn-trained student's response to
    PURE NOISE peaks at 0.49 (measured) -- right on a 0.4 gate -- while deep-event
    recall on cu_30h is flat across 0.4 / 0.5 / 0.6 / 0.7, so the raise costs nothing.
    Precision is carried by the sigma gate, not by the probability gate; hence the
    two absolute gates, which the σ gate cannot substitute for (it scales with the
    recording's own noise, so it widens wherever the baseline wanders).
    """
    hg, zn, cu = (mp.DECODE_PRESETS[k] for k in
                  ("hg_student", "zn_student", "cu_student"))
    assert hg.enter_threshold == zn.enter_threshold == 0.4
    assert cu.enter_threshold == 0.5
    assert hg.min_depth_pa == zn.min_depth_pa == 0.0
    assert hg.min_fracblock == zn.min_fracblock == 0.0
    assert (cu.min_depth_pa, cu.min_fracblock) == (10.0, 0.05)
    # ...and the σ gate is shared: it is the one clause all three projects agree on.
    assert hg.min_depth_sigma == zn.min_depth_sigma == cu.min_depth_sigma == 4.0


@pytest.mark.parametrize("name", list(NOTEBOOK_CONTRACT))
def test_the_snaps_are_off_because_every_real_call_site_disables_them(name):
    """``postprocess_predictions``'s SIGNATURE defaults are ``offset_sigma=3.0,
    onset_sigma=2.0`` -- and they are not what ran. Cells 8 and 15 of all three
    notebooks (six call sites) pass ``offset_sigma=0, onset_sigma=0`` explicitly.
    Shipping the signature defaults would ship a decode never executed on this data.
    """
    cfg = mp.DECODE_PRESETS[name]
    assert cfg.onset_sigma == 0.0
    assert cfg.offset_sigma == 0.0


@pytest.mark.parametrize("name", list(NOTEBOOK_CONTRACT))
def test_the_uncalibrated_warning_is_gone_from_the_label(name):
    """The suffix warned that the decode was a placeholder. It is not one any more,
    and a warning left standing after its cause is fixed teaches the user to ignore
    the next one. Nothing else in the preset surface may still claim otherwise."""
    label = mp.PRESET_LABELS[name]
    assert "UNCALIBRATED" not in label.upper()
    assert "uncalibrated" not in mp.DECODE_PRESETS[name].describe().lower()


def test_no_shipped_preset_label_still_warns_about_an_uncalibrated_decode():
    for name, label in mp.PRESET_LABELS.items():
        assert "UNCALIBRATED" not in label.upper(), name


# ===========================================================================
# 2. the gates default OFF, so nothing that already exists moves
# ===========================================================================
def test_the_three_gates_default_to_off():
    cfg = mp.MLDecodeConfig()
    assert cfg.min_depth_sigma == 0.0
    assert cfg.min_depth_pa == 0.0
    assert cfg.min_fracblock == 0.0


@pytest.mark.parametrize(
    "name",
    ["notebook1", "notebook2", "notebook3", "apo_nb_500mV", "apo_nb_600mV",
     "apo_nb_700mV", "apo_raw_global"],
)
def test_every_pre_existing_preset_leaves_all_three_gates_off(name):
    """These seven were verified against their own notebooks BEFORE the gates
    existed. Switching one on here would silently re-gate every stored result that
    names it."""
    cfg = mp.DECODE_PRESETS[name]
    assert (cfg.min_depth_sigma, cfg.min_depth_pa, cfg.min_fracblock) == (0.0, 0.0, 0.0)


def test_a_stored_config_without_the_new_keys_round_trips_to_gates_off():
    """The exact shape a run saved before this landed has on disk: ten keys, none of
    them a gate. ``from_dict`` must reconstruct the SAME config it always did."""
    legacy = {
        "preset": "notebook1", "smoothing_kernel": 9, "enter_threshold": 0.5,
        "exit_threshold": 0.15, "merge_radius_ms": 0.5, "min_event_ms": 1.0,
        "onset_sigma": 0.0, "onset_max_ms": 5.0, "onset_gap_tol": 5,
        "offset_sigma": 0.0,
    }
    assert mp.MLDecodeConfig.from_dict(legacy) == mp.DECODE_PRESETS["notebook1"]


@pytest.mark.parametrize("name", list(mp.DECODE_PRESETS))
def test_every_preset_survives_the_full_to_dict_from_dict_round_trip(name):
    """A silently dropped field is worse than a missing one: the config would come
    back looking valid with a gate quietly switched off."""
    cfg = mp.DECODE_PRESETS[name]
    d = cfg.to_dict()
    for key in ("min_depth_sigma", "min_depth_pa", "min_fracblock"):
        assert key in d, f"{name}: {key} never reached the dict"
    assert mp.MLDecodeConfig.from_dict(d) == cfg


def test_the_gates_are_validated_like_every_other_field():
    for key in ("min_depth_sigma", "min_depth_pa", "min_fracblock"):
        with pytest.raises(ValueError, match=key):
            mp.MLDecodeConfig(**{key: -1.0})
        with pytest.raises(ValueError, match=key):
            mp.MLDecodeConfig(**{key: float("nan")})


def test_a_gate_appears_in_deviations_when_the_user_changes_it():
    base = mp.DECODE_PRESETS["cu_student"]
    assert "min_fracblock" in base.replace(min_fracblock=0.2).deviations_from(base)
    assert base.deviations_from(base) == ()


# ===========================================================================
# 3. each gate rejects what it should
# ===========================================================================
def _prob(n: int, spans) -> np.ndarray:
    p = np.zeros(n, np.float32)
    for s, e in spans:
        p[s:e] = 0.9
    return p


@dataclasses.dataclass(frozen=True)
class _Trace:
    """A synthetic recording with three events chosen so each gate has exactly one
    victim, and the deep one clears all three.

    Noise sigma is 2 pA, so the 4-sigma gate sits at ~8 pA. ``I0`` is 150 pA, so the
    0.05 fracblock gate sits at 7.5 pA. The three depths -- 40 / 5 / 9 pA -- then
    land one on each side of each gate:

    ======  =====  ==============  ==============  =================
    depth   dwell  sigma (>= 8pA)  pA (>= 10 pA)   dI/I0 (>= 0.05)
    ======  =====  ==============  ==============  =================
    40 pA   12 ms  pass            pass            0.267  pass
     5 pA   12 ms  FAIL            FAIL            0.033  FAIL
     9 pA   12 ms  pass            FAIL            0.060  pass
    ======  =====  ==============  ==============  =================

    So the 9 pA event is killed by the pA gate ALONE -- which is the case that gate
    was added for, since a sigma gate cannot express "10 pA" -- and the 5 pA one is
    killed by all three. Every event is 12 ms long, far past any duration filter, so
    nothing here can be attributed to the dwell test.
    """

    detr: np.ndarray
    raw: np.ndarray
    prob: np.ndarray
    spans: tuple[tuple[int, int], ...]


def _trace(depths_pA=(40.0, 5.0, 9.0), i0_pA=150.0, width=600) -> _Trace:
    rng = np.random.default_rng(7)
    n = 600_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    spans = []
    for k, depth in enumerate(depths_pA):
        s = 100_000 * (k + 1)
        detr[s:s + width] -= np.float32(depth)
        spans.append((s, s + width))
    # RAW = the detrended trace sitting on a real open-pore level. Detrending is
    # what removes exactly this, which is why the fracblock gate cannot be answered
    # from ``detr`` alone.
    raw = (detr - np.float32(i0_pA)).astype(np.float32)
    return _Trace(detr, raw, _prob(n, spans), tuple(spans))


_OPEN = mp.MLDecodeConfig(smoothing_kernel=1, enter_threshold=0.5,
                          exit_threshold=0.5, merge_radius_ms=0.0,
                          min_event_ms=0.0)


def _decode(cfg, tr: _Trace):
    return mp.decode_ml_regions(
        tr.prob, FS, cfg, signal=tr.detr, gate_signal=tr.detr, raw=tr.raw
    )


def test_with_every_gate_off_all_three_events_survive():
    tr = _trace()
    assert len(_decode(_OPEN, tr)) == 3


def test_a_shallow_but_long_event_fails_the_sigma_gate():
    """The 5 pA event is 600 samples (12 ms) long -- far past any duration filter --
    and still fails, because the gate is on DEPTH. Duration cannot buy its way past
    an amplitude gate, which is the whole point of having both."""
    tr = _trace()
    kept = _decode(_OPEN.replace(min_depth_sigma=4.0), tr)          # ~8 pA threshold
    assert [s for s, _ in kept] == [100_000, 300_000]               # 5 pA one gone
    # ...and it is not the duration filter doing it: the same event survives with the
    # gate off and a min duration that its 12 ms clears easily.
    assert len(_decode(_OPEN.replace(min_event_ms=5.0), tr)) == 3


def test_a_sub_10_pA_event_fails_cus_absolute_pA_gate():
    """The 9 pA event clears a 4-sigma gate (~8 pA) and is killed only by the
    absolute floor. That is exactly the case the pA gate was added for: a σ gate
    scales with the recording's own noise, so it cannot express "10 pA"."""
    tr = _trace()
    sigma_only = _decode(_OPEN.replace(min_depth_sigma=4.0), tr)
    assert [s for s, _ in sigma_only] == [100_000, 300_000]         # 9 pA survives σ
    both = _decode(_OPEN.replace(min_depth_sigma=4.0, min_depth_pa=10.0), tr)
    assert [s for s, _ in both] == [100_000]                        # ...not the floor


def test_a_small_dI_over_I0_event_fails_the_fracblock_gate():
    """Same three events, a MUCH larger pore. Nothing about the detrended trace
    changes -- same depths, same noise, so the σ gate keeps exactly what it kept --
    but 9 pA against a 2000 pA open pore is ΔI/I0 = 0.0045, well under 0.05.
    """
    big = _trace(i0_pA=2_000.0)
    on = _OPEN.replace(min_fracblock=0.05)
    assert [s for s, _ in _decode(_OPEN.replace(min_depth_sigma=4.0), big)] \
        == [100_000, 300_000]
    # 40/2000 = 0.02 -- even the DEEP event is under the floor on a pore this big.
    assert _decode(on, big) == []
    # On the 150 pA pore the same events are ΔI/I0 = 0.267 / 0.033 / 0.060, so the
    # gate keeps the two above 0.05 and drops the one below. Same trace, same
    # depths, same noise -- only I0 changed.
    small = _trace()
    assert [s for s, _ in _decode(on, small)] == [100_000, 300_000]


def test_the_fracblock_gate_follows_the_time_local_I0_not_a_global_one():
    """I0 drift is why the notebook computes a 5 s rolling median instead of one
    global number. Two identical 20 pA events, one on a 200 pA stretch of pore and
    one on a 1000 pA stretch: a global I0 would score them the same, the local one
    keeps the first (0.10) and drops the second (0.02) against a 0.05 floor."""
    rng = np.random.default_rng(11)
    n = 600_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    for s in (100_000, 400_000):
        detr[s:s + 600] -= np.float32(20.0)
    # A ramping open-pore level: 200 pA at the first event, 1000 pA at the second.
    i0 = np.linspace(150.0, 1_100.0, n).astype(np.float32)
    raw = (detr - i0).astype(np.float32)
    prob = _prob(n, [(100_000, 100_600), (400_000, 400_600)])

    cfg = _OPEN.replace(min_fracblock=0.05)
    kept = mp.decode_ml_regions(prob, FS, cfg, signal=detr, gate_signal=detr, raw=raw)
    assert [s for s, _ in kept] == [100_000]


def test_all_three_gates_are_ANDed_not_ORed():
    tr = _trace()
    cfg = _OPEN.replace(min_depth_sigma=4.0, min_depth_pa=10.0, min_fracblock=0.05)
    assert [s for s, _ in _decode(cfg, tr)] == [100_000]


def test_a_gate_of_zero_is_off_not_a_threshold_of_zero():
    """A threshold of literally zero would still reject an upward excursion, whose
    sustained depth is negative. ``0.0`` must mean the clause did not run at all,
    because that is what every pre-existing stored config relies on."""
    rng = np.random.default_rng(3)
    n = 200_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    detr[100_000:100_600] += np.float32(40.0)          # UPWARD -- negative depth
    prob = _prob(n, [(100_000, 100_600)])
    raw = (detr - np.float32(200.0)).astype(np.float32)
    assert mp.decode_ml_regions(
        prob, FS, _OPEN, signal=detr, gate_signal=detr, raw=raw) == [(100_000, 100_599)]
    assert mp.decode_ml_regions(
        prob, FS, _OPEN.replace(min_depth_sigma=4.0),
        signal=detr, gate_signal=detr, raw=raw) == []


# ===========================================================================
# 3b. it refuses LOUDLY rather than approximating
# ===========================================================================
def test_a_gate_without_a_signal_raises_instead_of_silently_not_running():
    tr = _trace()
    with pytest.raises(ValueError, match="gate_signal"):
        mp.decode_ml_regions(tr.prob, FS, _OPEN.replace(min_depth_sigma=4.0))


def test_the_fracblock_gate_without_a_raw_trace_raises():
    """Nothing but the RAW trace carries the open-pore level, and detrending is
    precisely what removes it. Guessing an I0 here would change which events exist
    while looking entirely plausible."""
    tr = _trace()
    with pytest.raises(ValueError, match="TIME-LOCAL"):
        mp.decode_ml_regions(
            tr.prob, FS, _OPEN.replace(min_fracblock=0.05),
            signal=tr.detr, gate_signal=tr.detr)


def test_gates_off_needs_neither_trace_so_nothing_pre_existing_starts_raising():
    tr = _trace()
    assert len(mp.decode_ml_regions(tr.prob, FS, _OPEN)) == 3


# ===========================================================================
# 4. ORDER: after the duration filter, on the merged extent
# ===========================================================================
def test_the_gates_run_after_the_min_duration_filter():
    """The notebook's structure fixes this: ``postprocess_predictions`` ends with
    ``filter_short_events`` and RETURNS; cells 8 and 15 gate what it returned.

    Observable because the two stages disagree here. The 4 pA event is 20 ms long,
    so it passes any duration test and fails the σ gate; the 40 pA event is 0.2 ms
    long, so it fails a 1 ms duration test and would pass the σ gate. With the gate
    after the filter, NOTHING survives -- the deep one is already gone by the time
    the gate looks. Were the gate first, the deep event would be gated in and then
    dropped by the filter: same answer here, so the discriminating case is next.
    """
    rng = np.random.default_rng(5)
    n = 300_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    detr[100_000:101_000] -= np.float32(4.0)      # 20 ms, shallow
    detr[200_000:200_010] -= np.float32(40.0)     # 0.2 ms, deep
    prob = _prob(n, [(100_000, 101_000), (200_000, 200_010)])
    raw = (detr - np.float32(200.0)).astype(np.float32)
    cfg = _OPEN.replace(min_event_ms=1.0, min_depth_sigma=4.0)
    assert mp.decode_ml_regions(
        prob, FS, cfg, signal=detr, gate_signal=detr, raw=raw) == []


def test_the_gates_measure_the_MERGED_extent_not_the_pre_merge_fragments():
    """THE discriminating case for the order, and the reason it matters.

    One real event, whose probability trace dips below ``exit`` in the middle, so
    hysteresis fires TWO regions that the merge then joins. The deep half is 30 pA;
    the shallow half is 4 pA. Gated on the MERGED extent the event's median depth is
    ~17 pA and it is kept whole. Gated on the pre-merge fragments the shallow half
    would be thrown away and the event would come back truncated.
    """
    rng = np.random.default_rng(9)
    n = 300_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    detr[100_000:100_500] -= np.float32(30.0)     # deep half
    detr[100_520:101_000] -= np.float32(4.0)      # shallow half -- 2 sigma, on its own
    prob = np.zeros(n, np.float32)
    prob[100_000:100_500] = 0.9
    prob[100_520:101_000] = 0.9                   # 20-sample gap: below exit
    raw = (detr - np.float32(200.0)).astype(np.float32)

    # merge 0.5 ms == 25 samples > the 20-sample gap, so the two fragments join.
    cfg = _OPEN.replace(merge_radius_ms=0.5, min_depth_sigma=4.0)
    kept = mp.decode_ml_regions(prob, FS, cfg, signal=detr, gate_signal=detr, raw=raw)
    assert kept == [(100_000, 100_999)], "the gate must see the whole merged event"

    # Without the merge, the shallow fragment is its own region and the gate -- the
    # same gate, same threshold -- correctly drops it. That is the difference the
    # ordering makes, and it is a difference in the EVENT, not just in the count.
    unmerged = mp.decode_ml_regions(
        prob, FS, cfg.replace(merge_radius_ms=0.0),
        signal=detr, gate_signal=detr, raw=raw)
    assert unmerged == [(100_000, 100_499)]


def test_the_gate_measures_the_trimmed_extent_when_the_offset_trim_is_on():
    """The trim runs inside ``postprocess_predictions`` and the gate runs after it,
    so the depth is measured on what the trim left -- not on the raw hysteresis run
    with its shallow recovery tail dragging the median up."""
    rng = np.random.default_rng(13)
    n = 300_000
    detr = rng.normal(0.0, 2.0, n).astype(np.float32)
    # Assigned, not subtracted: a NOISELESS tail, so the 3-sigma deep level (~-6 pA)
    # is never crossed inside it and the trim lands deterministically. With the noise
    # left on, ~7% of an 800-sample tail dips past -6 pA and the "last deep sample"
    # is at the very end of the tail -- no trim at all, and a flaky test.
    detr[100_000:100_200] = np.float32(-30.0)     # deep part
    detr[100_200:101_000] = np.float32(-3.0)      # long shallow tail
    prob = _prob(n, [(100_000, 101_000)])
    raw = (detr - np.float32(200.0)).astype(np.float32)

    cfg = _OPEN.replace(min_depth_sigma=4.0)
    # Untrimmed, the tail dominates the median (800 shallow samples vs 200 deep) and
    # the whole event fails the gate.
    assert mp.decode_ml_regions(
        prob, FS, cfg, signal=detr, gate_signal=detr, raw=raw) == []
    # Trimmed to the deep part first, the same event passes.
    trimmed = mp.decode_ml_regions(
        prob, FS, cfg.replace(offset_sigma=3.0),
        signal=detr, gate_signal=detr, raw=raw)
    assert len(trimmed) == 1 and trimmed[0][0] == 100_000
    assert trimmed[0][1] < 100_400


# ===========================================================================
# 5. ONE implementation of the gate, not two
# ===========================================================================
def test_the_decode_gate_is_event_features_sustained_gate_itself():
    """A second copy of a gate is exactly how this codebase's own comments say
    things drift, and how the notebooks' thresholds drifted in the first place. So
    the decode must AGREE WITH ``event_features`` to the last bit, not merely
    resemble it -- and the check is against the module the rest of the app gates
    with, on half-open windows, which is the off-by-one this is really guarding.
    """
    tr = _trace()
    cfg = _OPEN.replace(min_depth_sigma=4.0, min_depth_pa=10.0, min_fracblock=0.05)
    kept = _decode(cfg, tr)

    windows = np.asarray([(s, e) for s, e in tr.spans], dtype=np.int64)
    direct = ef.amplitude_gate(
        tr.detr, windows, min_depth_sigma=4.0, min_depth_pa=10.0,
        min_fracblock=0.05, raw=tr.raw, fs=FS)
    assert [s for s, _ in kept] == [
        int(w[0]) for w, k in zip(windows, direct.keep) if k
    ]


def test_the_module_constants_are_the_notebook_constants():
    assert ef.MIN_DEPTH_SIGMA == 4.0
    assert ef.MIN_DEPTH_PA == 10.0
    assert ef.MIN_FRACBLOCK == 0.05
    assert ef.I0_WINDOW_S == 5.0


def test_the_per_gate_attrition_is_reported_because_cell_15_prints_it():
    """"So it is visible WHICH gate is doing the work and these can be tuned against
    your own recordings instead of taken on faith" -- cell 1. Collapsing the three
    clauses into one boolean throws that away."""
    tr = _trace()
    r = ef.amplitude_gate(
        tr.detr, np.asarray(tr.spans, dtype=np.int64),
        min_depth_sigma=4.0, min_depth_pa=10.0, min_fracblock=0.05,
        raw=tr.raw, fs=FS)
    assert r.attrition() == {"fired": 3, "sigma": 2, "pA": 1, "fracblock": 2, "all": 1}


def test_the_local_I0_curve_tracks_a_drifting_open_pore():
    """The reason a global I0 is not good enough: a 150 -> 1100 pA ramp is a 170%
    drift, far past the 0.15 the notebook warns at."""
    n = 600_000
    ramp = np.linspace(150.0, 1_100.0, n).astype(np.float32)
    quiet = np.ones(n, dtype=bool)
    at = np.asarray([50_000, 550_000])
    vals, _med, drift = ef.local_open_pore_pA(-ramp, quiet, FS, at=at)
    assert abs(vals[0]) == pytest.approx(ramp[50_000], rel=0.02)
    assert abs(vals[1]) == pytest.approx(ramp[550_000], rel=0.02)
    assert drift > 0.15


def test_a_run_stored_before_the_gates_existed_now_reads_as_a_DEVIATION():
    """The honest answer for an old ``cu_student`` result, and the one that keeps it
    reproducible: it replays with gates off (bit-for-bit what it ran), and it is
    FLAGGED as deviating from the preset — because it genuinely does. Silently
    upgrading it to the new contract would change its event count; silently calling
    it un-customized would hide that its numbers are not the shipped ones.
    """
    stored = {
        "preset": "cu_student", "smoothing_kernel": 1, "enter_threshold": 0.5,
        "exit_threshold": 0.5, "merge_radius_ms": 0.0, "min_event_ms": 0.1,
        "onset_sigma": 0.0, "onset_max_ms": 5.0, "onset_gap_tol": 5,
        "offset_sigma": 0.0,
    }
    old = mp.MLDecodeConfig.from_dict(stored)
    assert (old.min_depth_sigma, old.min_depth_pa, old.min_fracblock) == (0.0, 0.0, 0.0)
    devs = old.deviations_from(mp.DECODE_PRESETS["cu_student"])
    assert {"min_depth_sigma", "min_depth_pa", "min_fracblock"} <= set(devs)
