"""Sweep 7, S7-DET1's last seam (agent E's "Needs wiring elsewhere"): the ML detector stamps
the RESOLVED baseline param source exactly as the two threshold detectors do, and records
params without the per-load scope key.

``MLDetect.run`` used to record ``params=dict(ctx.params)`` untouched and resolve the
baseline config alone (``baseline_config_from_params``), so a future ML batch pass over
recordings the batch form's windows were not proposed for would have stamped
``'proposed'`` on every one of them — "a human chose these windows for this file" for a
file nobody measured — and written the batch form's ``doc_id`` scope into a project file.
The threshold detectors were fixed to go through ONE resolver
(``modules._resolve_run_baseline``); the ML detector now goes through the same one.

The model is a stand-in that fires on the deep normalised samples (the same stub
``test_detector_choice_and_model_input`` uses), so the run is real end to end without
TensorFlow. Lives under tests/gui because that is this fix batch's file ownership; it
needs no Qt.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import ml_detect, ml_modules
from nanoscope.analysis.ml_preprocess import PREPROCESS_PRESETS
from nanoscope.core.identity import doc_id_for_meta


class _FiringModel:
    """Fires on |x| > 0.5 of the normalised input — enough to produce events."""

    def __init__(self, window: int) -> None:
        self.window = window

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)
        return (np.abs(x) > 0.5).astype(np.float16)


@pytest.fixture
def stubbed_model(monkeypatch):
    window = 1024
    wrapper = ml_detect.KerasEventModel(
        model=_FiringModel(window), window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _p: wrapper)
    return wrapper


def _trace(seed: int = 7, n: int = 40_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (-1000.0 + 6.0 * rng.standard_normal(n)).astype(np.float32)
    for s in range(4_000, n, 8_000):
        x[s:s + 300] -= 260.0
    return x


def _params(**extra) -> dict:
    cfg = PREPROCESS_PRESETS["apo_nb_600mV"]
    p = {"model_path": "<fake>", "ml_preprocess": cfg.to_dict(),
         "baseline_rough_s": 0.2, "baseline_base_s": 0.05,
         "baseline_param_source": "proposed"}
    p.update(extra)
    return p


def test_a_foreign_batch_scope_is_stamped_batch_and_the_scope_key_is_not_recorded(
        stubbed_model, context_factory):
    ctx = context_factory(_trace(), fs=50_000.0,
                          params=_params(**{B.BATCH_SCOPE_KEY: ("some-other-load", 0)}),
                          units="pA", path="ml_batch.abf")
    res = ml_modules.MLDetect().run(ctx)
    assert len(res.payload.df) > 0, "the stub produced no events; nothing below is exercised"
    assert res.params["baseline_param_source"] == B.BATCH_SOURCE
    assert B.BATCH_SCOPE_KEY not in res.params, "a per-load doc_id would reach a project file"
    assert res.payload.provenance["baseline_param_source"] == B.BATCH_SOURCE
    assert any("COPIED FROM THE BATCH FORM" in w for w in res.warnings), res.warnings
    # Everything else the form said is recorded as it was said.
    for k, v in _params().items():
        if k != "baseline_param_source":
            assert res.params[k] == v


def test_the_run_s_own_recording_keeps_the_form_s_source(stubbed_model, context_factory):
    ctx = context_factory(_trace(), fs=50_000.0, params=_params(), units="pA",
                          path="ml_own.abf")
    own = (doc_id_for_meta(ctx.meta), int(ctx.channel.info.index))
    ctx = context_factory(_trace(), fs=50_000.0,
                          params=_params(**{B.BATCH_SCOPE_KEY: own}), units="pA",
                          path="ml_own.abf")
    res = ml_modules.MLDetect().run(ctx)
    assert res.params["baseline_param_source"] == "proposed"
    assert B.BATCH_SCOPE_KEY not in res.params
    assert res.payload.provenance["baseline_param_source"] == "proposed"
    assert not any("COPIED FROM THE BATCH FORM" in w for w in res.warnings)


def test_without_a_scope_the_interactive_run_is_untouched(stubbed_model, context_factory):
    """The Detect / Run-ML buttons hand over no scope key: params recorded verbatim."""
    ctx = context_factory(_trace(), fs=50_000.0, params=_params(), units="pA",
                          path="ml_plain.abf")
    res = ml_modules.MLDetect().run(ctx)
    assert dict(res.params) == _params()
    assert res.payload.provenance["baseline_param_source"] == "proposed"


def test_the_ml_detector_uses_the_one_resolver_by_name():
    """The seam itself, on the code with comments stripped: the ML run resolves through
    ``modules._resolve_run_baseline`` and no longer calls the form-only resolver."""
    import inspect

    src = inspect.getsource(ml_modules.MLDetect.run)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "_resolve_run_baseline(" in code
    assert "baseline_config_from_params(" not in code
