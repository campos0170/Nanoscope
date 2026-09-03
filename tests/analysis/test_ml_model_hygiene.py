"""P3 — what ``load_keras_model`` guarantees, and what it must not accumulate.

C36, C37, C38: the input contract was checked rigorously and the OUTPUT contract not at
all; the last-resort load path made a temporary copy of the model before anything could
fail and never removed it; and the cache was an unbounded dict on a workflow whose whole
shape is retrain, re-save over the same filename, re-run.

These exercise the pure parts — the contract check, the eviction, the temp-directory
lifetime — so they run with no TensorFlow and no ``.keras`` file at all. The end-to-end
loads are covered by ``tests/analysis/test_ml_detect.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanoscope.analysis import ml_detect


# ================================================================================================
# a TensorFlow stand-in — the checks under test are about SHAPES, not about inference
# ================================================================================================

class _FakeModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape


def _stub_tf(monkeypatch, input_shape, output_shape):
    """Make ``load_keras_model`` load a model of exactly these shapes, with no TF."""
    class _TF:
        class keras:
            class models:
                @staticmethod
                def load_model(path, compile=False, custom_objects=None):
                    return _FakeModel(input_shape, output_shape)

    monkeypatch.setattr(ml_detect, "_import_tf", lambda: _TF)


def _model_file(tmp_path: Path, name: str = "m.keras") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really a keras file; _import_tf is stubbed")
    return path


# ================================================================================================
# C36 — an output that is not one probability per sample is refused BY NAME
# ================================================================================================

def test_a_per_window_classifier_is_refused_instead_of_broadcast(tmp_path, monkeypatch):
    """``(None, 1)`` — a single score per window — is numpy-broadcast across all W samples
    by ``predict_probability``, so every window comes out uniformly event or not-event and
    EVERY DWELL TIME in the table is an artifact of the window length and the stride,
    reported to four decimals with ``verdict="measured"``.

    Driven through ``load_keras_model``, which is where the refusal has to happen: the
    input side was checked there rigorously and the output side was merely RECORDED."""
    _stub_tf(monkeypatch, (None, 2048, 1), (None, 1))
    with pytest.raises(ValueError) as exc:
        ml_detect.load_keras_model(_model_file(tmp_path))
    msg = str(exc.value)
    assert "m.keras" in msg
    assert "dwell time" in msg
    assert "(None, 2048)" in msg


def test_a_dense_2_head_is_refused(tmp_path, monkeypatch):
    """``~/Downloads/older_spike_model.keras`` ends in ``Dense(2)``. Before, that reached
    the inference loop and threw "operands could not be broadcast together" — from inside
    a batch, after the whole-recording preprocessing pass had already been paid for."""
    _stub_tf(monkeypatch, (None, 3072, 1), (None, 2))
    with pytest.raises(ValueError) as exc:
        ml_detect.load_keras_model(_model_file(tmp_path, "older.keras"))
    assert "2 value(s) for the whole window" in str(exc.value)


def test_a_multi_output_model_is_refused_by_name(tmp_path, monkeypatch):
    """A model with more than one output has no single probability trace to decode. It used
    to be recorded as ``output_shape=()`` and handed to ``predict``."""
    _stub_tf(monkeypatch, (None, 2048, 1), [(None, 2048), (None, 3)])
    with pytest.raises(ValueError) as exc:
        ml_detect.load_keras_model(_model_file(tmp_path, "two.keras"))
    assert "more than one output" in str(exc.value)


@pytest.mark.parametrize("shape", [(None, 2048), (None, 2048, 1)])
def test_a_real_per_sample_detector_is_accepted_unchanged(shape, tmp_path, monkeypatch):
    """Both shapes the shipped students emit. The check must not narrow what already
    works — this batch moves no number on a well-formed input."""
    _stub_tf(monkeypatch, (None, 2048, 1), shape)
    wrapper = ml_detect.load_keras_model(_model_file(tmp_path))
    assert wrapper.window == 2048
    assert wrapper.output_shape == shape


def test_the_check_reads_the_length_not_the_channel_count(tmp_path, monkeypatch):
    """``predict_probability`` documents that it takes ``[..., 0]`` of a rank-3 output, so
    the channel count is the model's business. The LENGTH is not: it is what dwell times
    are made of."""
    _stub_tf(monkeypatch, (None, 512, 1), (None, 512, 3))
    assert ml_detect.load_keras_model(_model_file(tmp_path, "ok.keras")).window == 512

    _stub_tf(monkeypatch, (None, 512, 1), (None, 256, 1))
    with pytest.raises(ValueError):
        ml_detect.load_keras_model(_model_file(tmp_path, "short.keras"))


# ================================================================================================
# C38 — the cache is bounded and least-recently-used
# ================================================================================================

@pytest.fixture(autouse=True)
def _clean_cache():
    ml_detect.clear_model_cache()
    yield
    ml_detect.clear_model_cache()


def test_the_model_cache_does_not_grow_without_bound(tmp_path, monkeypatch):
    """The owner's loop is retrain -> re-save over the SAME filename -> re-run, and the key
    includes the mtime, so every cycle pinned another full Keras model and released none
    (measured: 1 -> 5 entries over four re-saves, four of them unreachable by any key the
    app could still produce). Driven through ``load_keras_model`` with a real file whose
    mtime moves, which is the workflow itself."""
    _stub_tf(monkeypatch, (None, 2048, 1), (None, 2048))
    path = _model_file(tmp_path)
    for i in range(6):
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
        ml_detect.load_keras_model(path)
    assert len(ml_detect._MODEL_CACHE) == ml_detect.MODEL_CACHE_MAX


def test_the_cache_evicts_the_least_recently_used(tmp_path, monkeypatch):
    _stub_tf(monkeypatch, (None, 2048, 1), (None, 2048))
    a = _model_file(tmp_path, "a.keras")
    b = _model_file(tmp_path, "b.keras")
    c = _model_file(tmp_path, "c.keras")
    ml_detect.load_keras_model(a)
    ml_detect.load_keras_model(b)
    ml_detect.load_keras_model(a)                  # a is used again -> b is now the oldest
    ml_detect.load_keras_model(c)

    paths = {key[0] for key in ml_detect._MODEL_CACHE}
    assert str(a.resolve()) in paths or str(a) in paths
    assert str(c) in paths
    assert str(b) not in paths


def test_a_cached_model_is_still_returned_without_reloading(tmp_path, monkeypatch):
    """The bound must not cost the thing the cache exists for."""
    _stub_tf(monkeypatch, (None, 2048, 1), (None, 2048))
    path = _model_file(tmp_path)
    first = ml_detect.load_keras_model(path)
    assert ml_detect.load_keras_model(path) is first


def test_clear_model_cache_releases_everything(tmp_path, monkeypatch):
    _stub_tf(monkeypatch, (None, 2048, 1), (None, 2048))
    ml_detect.load_keras_model(_model_file(tmp_path))
    ml_detect.clear_model_cache()
    assert len(ml_detect._MODEL_CACHE) == 0


# ================================================================================================
# C37 — the last-resort load path leaves no temporary copy behind
# ================================================================================================

def test_the_quantization_strip_removes_its_temp_directory_on_success(tmp_path, monkeypatch):
    """It is the LAST-RESORT branch, so it runs on exactly the loads that are already going
    badly — and every one of them used to leave a ``nanoscope_ml_*`` directory holding a
    full copy of the model in the system temp directory (12 had accumulated)."""
    import json
    import tempfile
    import zipfile

    src = tmp_path / "m.keras"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("config.json", json.dumps({"quantization_config": {"x": 1}}))
        zf.writestr("weights.h5", b"\x00")

    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        made.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy)

    class _TF:
        class keras:
            class models:
                @staticmethod
                def load_model(path, compile=False, custom_objects=None):
                    assert os.path.exists(path)
                    return object()

    ml_detect._load_quantization_stripped(_TF, str(src), {})

    assert made and not any(Path(d).exists() for d in made)


def test_the_quantization_strip_removes_its_temp_directory_on_failure(tmp_path, monkeypatch):
    """...and especially when the retry ALSO fails, which is the case that produced them."""
    import json
    import tempfile
    import zipfile

    src = tmp_path / "m.keras"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("config.json", json.dumps({"quantization_config": {"x": 1}}))

    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        made.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy)

    class _TF:
        class keras:
            class models:
                @staticmethod
                def load_model(path, compile=False, custom_objects=None):
                    raise RuntimeError("still cannot read it")

    with pytest.raises(RuntimeError):
        ml_detect._load_quantization_stripped(_TF, str(src), {})

    assert made and not any(Path(d).exists() for d in made)


# ================================================================================================
# C37b — a bad model path costs nothing before it is reported
# ================================================================================================

def test_the_model_is_loaded_before_any_signal_is_touched():
    """``MLDetect.run`` did the whole-recording notch + SavGol pass BEFORE
    ``load_keras_model``, so a mistyped or moved model path charged the user the entire
    preprocessing of a 3 M-sample recording before saying "model file not found".

    Asserted on the source order rather than by timing a 3 M-sample run: the two calls are
    unconditional and adjacent to statements that name themselves, and the ordering is the
    whole content of the fix."""
    import inspect

    from nanoscope.analysis import ml_modules

    src = inspect.getsource(ml_modules.MLDetect.run)
    load_at = src.index("ml_detect.load_keras_model(model_path)")
    preprocess_at = src.index("apply_ml_preprocess(full, fs, pre_cfg)")
    assert load_at < preprocess_at
