"""Runtime-loaded Keras event detector (RAW / global-raw contract).

This is the signal-first DSP core for the ML detector. It is deliberately
independent of Qt and of ``nanoscope.core.session``: every public function takes
``(signal, fs, ...)`` numpy arrays and returns numpy / plain-Python. The Qt
worker and the :class:`AnalysisModule` adapter that plug this into the app live
elsewhere (later phases). The regions this module emits are the SAME
``[start, end]`` regions the built-in Schmitt detector produces, so they flow
into ``characterize`` and the rest of the pipeline unchanged.

TensorFlow is imported *lazily* and only when a model is actually run
(``_import_tf``). Merely importing this module never touches TF, so the app
starts fast and degrades gracefully when TF is absent (callers get
:class:`MLUnavailable`, never a crash).

CPU IS FORCED. The Apple Metal pluggable device SIGABRTs (exit 134) on this
mixed_float16 LSTM during ``predict``. ``_import_tf`` hides every GPU
(``tf.config.set_visible_devices([], "GPU")``) BEFORE the first TF op, once, and
also sets ``CUDA_VISIBLE_DEVICES=-1`` at module import as a belt-and-suspenders.
This is non-negotiable: without it the process hard-crashes.

TWO input contracts are supported, auto-detected from ``input_shape[-1]``:

  * RAW / global-raw -- ``(None, W, 1)`` (``channels == 1``):
      detrend = notch at 60/120/180 Hz (iirnotch Q=30, filtfilt) then subtract
                savgol_filter(500 ms window, polyorder=3);
      normalize = signal / (max(|signal|) + 1e-8)  -- NO mean subtraction;
      each window is fed as raw ``(W, 1)``.
  * FEATURE / teacher -- ``(None, W, 12)`` (``channels == 12``):
      detrend (same as above), then PER WINDOW build
      :func:`create_multiscale_features_A` -> ``(W, 12)`` self-standardized
      features. *** NO global-raw normalize on this path *** -- the 12 features
      standardize internally, so dividing by max|x| would be wrong. The feature
      (teacher) model uses a custom ``clip_activation`` output, registered via
      :data:`_CUSTOM_OBJECTS` at load time.

Both contracts share EVERYTHING else: output ``(None, W, 1)`` (float16 -> float32,
taken as ``[..., 0]``), sliding window length ``W = input_shape[1]``, step ``W//2``
(50 % overlap), overlap-AVERAGE of the per-sample probabilities, and the SAME
hysteresis decode (median-smooth -> enter/exit -> inclusive ``[start, end]``
regions -> optional merge / min-duration). Detrend applies to BOTH; only the RAW
path divides by max|x|.

Ported from the proven end-to-end reference ``scratch_ml_smoketest.py`` (RAW) and
``StudentModel.ipynb`` cells 5/13 (FEATURE: create_multiscale_features_A +
teacher_predict).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, iirnotch, medfilt, savgol_filter

from .cancel import OperationCancelled, check_cancelled
from .ml_preprocess import (
    PREPROCESS_PRESETS,
    MLDecodeConfig,
    MLPreprocessConfig,
    apply_ml_preprocess,
    decode_ml_regions,
    normalization_center,
    normalization_scale,
    normalize_global,
)

# Belt-and-suspenders: ask CUDA to hide its GPUs before TF is ever imported.
# (The decisive step for Apple Metal is tf.config.set_visible_devices in
# _import_tf; this env var handles the CUDA case and is harmless otherwise.)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

__all__ = [
    "TENSORFLOW_AVAILABLE",
    "MLUnavailable",
    "MLCancelled",
    "KerasEventModel",
    "load_keras_model",
    "peek_model_window",
    "detrend_for_model",
    "normalize_global_raw",
    "create_multiscale_features_A",
    "clip_activation",
    "predict_probability",
    "decode_regions",
    "ml_detect_regions",
]

# ---------------------------------------------------------------------------
# The fixed RAW / global-raw preprocessing contract (do NOT retune these).
# ---------------------------------------------------------------------------
NOTCH_HZ: tuple[float, ...] = (60.0, 120.0, 180.0)
NOTCH_Q: float = 30.0
SAVGOL_MS: float = 500.0
SAVGOL_POLY: int = 3

# Decode defaults (0.1 ms merge / min-duration == 1e-4 s, as in the reference).
DEFAULT_ENTER: float = 0.5
DEFAULT_EXIT: float = 0.3
DEFAULT_SMOOTH_KERNEL: int = 9
DEFAULT_MERGE_S: float = 1e-4
DEFAULT_MIN_DUR_S: float = 1e-4

# Whether TensorFlow *could* be imported. Probed WITHOUT importing TF (so this
# module stays cheap and never grabs a GPU at import time). The lazy loader
# below is the only thing that actually imports and configures TF.
try:
    TENSORFLOW_AVAILABLE: bool = importlib.util.find_spec("tensorflow") is not None
except (ImportError, ValueError):  # pragma: no cover - defensive
    TENSORFLOW_AVAILABLE = False


class MLUnavailable(RuntimeError):
    """Raised when the ML detector cannot run because TensorFlow is unavailable.

    The app must degrade gracefully on this: catch it and tell the user ML
    detection is unavailable, never let it crash the process.
    """


class MLCancelled(OperationCancelled):
    """Raised out of :func:`predict_probability` / :func:`ml_detect_regions` when
    the caller's ``cancel()`` hook asks for an abort. The worker catches this and
    treats it as a clean cancel (no partial results are published).

    It subclasses :class:`~nanoscope.analysis.cancel.OperationCancelled` so that there
    is exactly ONE "the user asked to stop" signal in the codebase. Two independent
    cancellation exceptions is the same shape of mistake as fixing an invariant at one
    call site: everything that already knows how to handle a cancel -- the Qt workers
    above all -- would handle one of them and report the other to the user as a crash.
    The narrower type is kept because ml_modules translates it specifically."""


# ---------------------------------------------------------------------------
# Lazy TF import that FORCES CPU before the first TF op (once, cached).
# ---------------------------------------------------------------------------
_TF: Any = None


def _stub_protobuf_google3_packages() -> None:
    """Work around a FATAL PySide6/shiboken <-> protobuf conflict (see below).

    TensorFlow's bundled C++ protobuf registers extension modules under a
    ``google3.*`` namespace (e.g. ``google3.net.proto2.python.internal.cpp._message``)
    but leaves the INTERMEDIATE package names (``google3``, ``google3.net``, ...) out
    of ``sys.modules``. shiboken's lazy signature introspection -- triggered on the Qt
    main thread by an ordinary ``QObject.disconnect()`` once TF is loaded (the analysis
    controller disconnects its worker at the END of every run) -- walks the type
    registry and does ``import google3``. That raises ModuleNotFoundError, which
    shiboken turns into a *fatal* ``libshiboken: ... update_mapping`` abort (exit 134),
    crashing the whole app the instant an ML run finishes.

    Registering empty stub packages for the missing ancestors makes that import succeed
    harmlessly. The real leaf extension is untouched (protobuf reaches its message
    classes through the C++ layer, not these package objects), so TF keeps working.
    Idempotent and cheap. Verified: without this, an ML run through the AnalysisController
    SIGABRTs; with it, the disconnect completes normally.
    """
    import sys
    import types

    for name in list(sys.modules):
        if name.split(".", 1)[0] != "google3":
            continue
        parts = name.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            if sys.modules.get(pkg) is None:
                sys.modules[pkg] = types.ModuleType(pkg)


def _import_tf() -> Any:
    """Import TensorFlow, hide every GPU, and return the module (cached).

    CPU is forced here -- before any TF op runs -- because the Apple Metal
    pluggable device aborts this mixed_float16 LSTM during ``predict``. Raises
    :class:`MLUnavailable` (never a bare ImportError) when TF is not installed,
    so callers can degrade gracefully.

    POLLS THE AMBIENT CANCEL TOKEN, before and after. The import itself is a C-level
    call that nothing can interrupt -- measured at 4.10 s on this machine, and it was
    the single worst offender behind the SIGABRT on quit, because a cancel arriving
    during it was not seen until it finished. The poll on the way IN is the one that
    earns its place: ``MLDetect.run`` does the whole-recording notch + SavGol BEFORE
    it ever reaches a model, so a user who cancels during preprocessing used to go on
    to pay the full 4.1 s import of a library the run had already been told to
    abandon. The poll on the way OUT stops a run that was cancelled DURING the import
    from continuing into a prediction pass.
    """
    global _TF
    if _TF is not None:
        return _TF
    check_cancelled()
    try:
        tf = importlib.import_module("tensorflow")
    except Exception as exc:  # ImportError, or a broken TF install
        raise MLUnavailable(
            "TensorFlow is not available, so the ML event detector cannot run. "
            "Install tensorflow (2.15) to enable ML detection."
        ) from exc
    # protobuf (pulled in by TF) leaves 'google3.*' parent packages out of
    # sys.modules, which FATALLY crashes PySide6/shiboken on the next
    # QObject.disconnect() (the controller disconnects its worker after every run).
    # Stub the missing ancestors so that import succeeds. Must run right after the
    # import, before any Qt introspection can trip on it.
    _stub_protobuf_google3_packages()
    # Hide GPUs BEFORE the first op. Must happen before anything touches a
    # device; at first import nothing has, so this succeeds. If devices were
    # somehow already initialized, set_visible_devices raises RuntimeError --
    # swallow it (we still tried) rather than crash the import.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:  # pragma: no cover - depends on device init timing
        pass
    _TF = tf
    check_cancelled()
    return tf


# ---------------------------------------------------------------------------
# Custom Keras objects for the FEATURE / teacher model.
#
# The teacher (12-feature) model's output layer uses a custom ``clip_activation``
# registered ``@register_keras_serializable(package="Custom")`` in the training
# notebook, so it is serialized under the name ``"Custom>clip_activation"``. To
# load it we must supply that callable via ``custom_objects`` -- keyed by the
# REGISTERED name (the bare name does NOT resolve a package-qualified reference;
# the bare key is kept only for models saved without the package). The RAW student
# has no custom objects and loads with an empty mapping -- see load_keras_model,
# which only reaches for _CUSTOM_OBJECTS on a first-try failure.
# ---------------------------------------------------------------------------
def clip_activation(x: Any) -> Any:
    """Clips output between 0 and 1.

    Verbatim from the teacher notebook (``tf.clip_by_value(x, 0.0, 1.0)``), but
    resolving ``tf`` lazily via :func:`_import_tf` so importing this module never
    imports TensorFlow. Keras only ever calls this during model build / predict,
    by which point :func:`_import_tf` has already run.
    """
    tf = _import_tf()
    return tf.clip_by_value(x, 0.0, 1.0)


# Consulted ONLY on the load retry (feature/teacher model). Both spellings map to
# the same callable: "Custom>clip_activation" is how the notebook's
# ``@register_keras_serializable(package="Custom")`` serialized it; the bare
# "clip_activation" covers a model saved without a package qualifier.
_CUSTOM_OBJECTS: dict[str, Any] = {
    "clip_activation": clip_activation,
    "Custom>clip_activation": clip_activation,
}


# ---------------------------------------------------------------------------
# Model wrapper + cache
# ---------------------------------------------------------------------------
@dataclass
class KerasEventModel:
    """A loaded Keras model plus the window length + channel count it expects.

    ``model`` is the raw Keras object (call ``.predict`` on it). ``window`` is
    ``input_shape[1]`` -- the fixed sample count each window must have.
    ``channels`` is ``input_shape[-1]`` -- ``1`` for the RAW / global-raw contract
    or ``12`` for the FEATURE / teacher contract; it selects the preprocessing +
    per-window input in :func:`predict_probability`. ``path`` and ``mtime``
    identify the file on disk (used as the cache key).
    """

    model: Any
    window: int
    path: str
    mtime: float
    channels: int = 1
    input_shape: tuple = field(default=())
    output_shape: tuple = field(default=())

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<KerasEventModel W={self.window} C={self.channels} "
            f"path={os.path.basename(self.path)!r}>"
        )


# key: (abspath, mtime) -> KerasEventModel. Re-runs of the same file don't reload.
#
# BOUNDED, and an ``OrderedDict`` so the bound can be least-recently-used rather than
# arbitrary. A Keras student is tens of MB of weights and the key includes the mtime, so
# the owner's own loop — retrain, re-save over the SAME filename, re-run — pinned a whole
# extra model on every cycle and never released one (measured: 1 -> 5 entries over four
# re-saves, with the first four unreachable by any key the app could still produce).
# Two is the useful working set: the model you are running and the one you are comparing
# it against. Anything older is a reload, which is exactly what it costs on the first run.
_MODEL_CACHE: "OrderedDict[tuple[str, float], KerasEventModel]" = OrderedDict()

#: How many loaded models the cache keeps. See :data:`_MODEL_CACHE`.
MODEL_CACHE_MAX = 2


def clear_model_cache() -> None:
    """Drop every cached model. For tests, and for a user who wants the memory back."""
    _MODEL_CACHE.clear()


def _cache_put(key: tuple[str, float], wrapper: "KerasEventModel") -> None:
    """Insert, evicting least-recently-used down to :data:`MODEL_CACHE_MAX`."""
    _MODEL_CACHE[key] = wrapper
    _MODEL_CACHE.move_to_end(key)
    while len(_MODEL_CACHE) > MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)


def _check_output_contract(output_shape: Any, window: int, path: str) -> tuple:
    """Refuse a model whose output is not ONE PROBABILITY PER SAMPLE of the window.

    The input side was checked rigorously (rank, fixed window, channel count) and the
    output side was not checked at all — it was merely recorded. That asymmetry is a hole
    with a number at the bottom of it: :func:`predict_probability` overlap-averages the
    prediction back onto a per-sample trace, and the decoder turns runs of that trace into
    events. Hand it a per-WINDOW classifier and one of two things happens, neither of them
    an error the user could read:

    * ``(None, 1)`` — the single score is numpy-broadcast across all ``W`` samples, so
      every window is uniformly "event" or "not event" and **every dwell time in the
      resulting table is an artifact of the window length and the stride**, reported to
      four decimal places with ``verdict="measured"``;
    * ``(None, k)``, ``k`` not 1 and not ``W`` — a broadcast error thrown from inside the
      inference loop, after the whole-recording preprocessing pass has been paid for, in
      terms ("operands could not be broadcast together") that name neither the model nor
      the problem.

    ``~/Downloads/older_spike_model.keras`` ends in ``Dense(2)``. So the contract is
    stated here, once, where it can be refused by name before anything is computed.

    Accepted: ``(None, W)`` and ``(None, W, C)``. Rank 3 keeps whatever channel count the
    model emits because :func:`predict_probability` documents that it takes ``[..., 0]``;
    what may not vary is the LENGTH, which is the thing dwell times are made of.
    """
    name = os.path.basename(path)
    if output_shape is None:
        raise ValueError(
            f"{name}: this model has more than one output, so there is no single "
            f"probability trace to decode into events. Export the detector head on its "
            f"own — a per-sample output of shape (None, {window}) or (None, {window}, 1)."
        )
    shape = tuple(output_shape)
    if len(shape) not in (2, 3) or shape[1] != window:
        got = "x".join("None" if d is None else str(d) for d in shape)
        raise ValueError(
            f"{name}: this is not a per-sample event detector. Its input takes {window} "
            f"samples per window but its output is {got}, i.e. "
            + (f"{shape[1]} value(s) for the whole window"
               if len(shape) == 2 and isinstance(shape[1], int)
               else "not one value per sample")
            + f". Decoding it would give every event a dwell time set by the window "
              f"length and the stride rather than by the signal. Expected "
              f"(None, {window}) or (None, {window}, 1)."
        )
    return shape


def _strip_config_key(obj: Any, key: str) -> None:
    """Recursively delete every ``key`` from a nested dict/list config in place."""
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_config_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            _strip_config_key(v, key)


def _load_quantization_stripped(tf, abspath: str, custom_objects: dict[str, Any]):
    """Load a ``.keras`` model whose ``config.json`` carries a newer-Keras
    ``quantization_config`` that the installed Keras rejects.

    A ``.keras`` file is a zip of ``config.json`` + weight shards. We rewrite a
    temporary copy with every ``quantization_config`` stripped from the config
    (the weights are untouched) and load that. Only meaningful for zip-format
    ``.keras`` files; a non-zip (e.g. legacy ``.h5``) raises and the caller keeps
    the original, more informative error.

    THE DIRECTORY IS ALWAYS REMOVED. This is the last-resort branch, so it runs on
    exactly the loads that are already going badly — and every one of them used to
    leave a ``nanoscope_ml_*`` directory holding a full copy of the model behind in
    the system temp directory, because the ``mkdtemp`` came before the first thing
    that could raise and nothing ever cleaned it (measured: 3 leaked directories from
    3 failed loads; 12 had accumulated on this machine). The loaded model is fully
    in memory by the time this returns, so the copy is dead either way.
    """
    import json
    import shutil
    import tempfile
    import zipfile

    tmp_dir = tempfile.mkdtemp(prefix="nanoscope_ml_")
    try:
        patched = os.path.join(tmp_dir, "patched.keras")
        with zipfile.ZipFile(abspath) as zin, \
                zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in zin.namelist():
                data = zin.read(name)
                if name == "config.json":
                    cfg = json.loads(data)
                    _strip_config_key(cfg, "quantization_config")
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(name, data)
        return tf.keras.models.load_model(
            patched, compile=False, custom_objects=custom_objects)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_keras_model(path: str | os.PathLike) -> KerasEventModel:
    """Load a ``.keras`` event detector at runtime (``compile=False``) and wrap it
    with its window length + channel count.

    Both input contracts are accepted, auto-detected from ``input_shape[-1]``:
    ``(None, W, 1)`` (RAW, ``channels == 1``) and ``(None, W, 12)`` (FEATURE /
    teacher, ``channels == 12``). Loading is attempted FIRST with NO custom
    objects -- the RAW student needs none and loads immediately -- and only on a
    failure retried with ``custom_objects=_CUSTOM_OBJECTS`` so the teacher model,
    whose output uses the custom ``clip_activation``, deserializes.

    Cached by ``(absolute path, mtime)`` so re-running the same model is free;
    editing/replacing the file on disk invalidates the entry automatically.

    Raises:
        MLUnavailable: TensorFlow is not installed.
        FileNotFoundError: ``path`` does not exist.
        ValueError: the model is not a ``(None, W, 1)`` or ``(None, W, 12)``
            detector (wrong rank, non-fixed window, or an unsupported channel
            count), or its OUTPUT is not one probability per sample of the window
            (see :func:`_check_output_contract`).
    """
    abspath = os.path.abspath(os.fspath(path))
    if not os.path.exists(abspath):
        raise FileNotFoundError(f"model file not found: {abspath}")
    mtime = os.path.getmtime(abspath)
    key = (abspath, mtime)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        _MODEL_CACHE.move_to_end(key)      # a hit is a use; keep it off the eviction end
        return cached

    tf = _import_tf()  # forces CPU
    try:
        # RAW student: zero custom objects (keep this path clean, as contracted).
        model = tf.keras.models.load_model(abspath, compile=False)
    except Exception:
        # FEATURE / teacher: its output ``clip_activation`` is an unregistered
        # custom function in this fresh process, so the clean load raises. Retry
        # with it registered.
        try:
            model = tf.keras.models.load_model(
                abspath, compile=False, custom_objects=_CUSTOM_OBJECTS
            )
        except Exception as exc:
            # Last resort: a model saved by a NEWER Keras carries a
            # ``quantization_config`` on its layers that this Keras can't parse.
            # Strip it and retry (weights unaffected). If that fails too, raise
            # the more informative custom-objects error, not the zip/strip one.
            try:
                model = _load_quantization_stripped(tf, abspath, _CUSTOM_OBJECTS)
            except Exception:
                raise exc

    ishape = tuple(model.input_shape) if not isinstance(model.input_shape, list) else None
    if ishape is None or len(ishape) != 3:
        raise ValueError(
            f"expected a single-input model with shape (None, W, C) where C is 1 "
            f"(RAW) or 12 (FEATURE), got input_shape={model.input_shape!r}."
        )
    window = ishape[1]
    if window is None or int(window) <= 0:
        raise ValueError(
            f"model input window length is not a fixed positive integer "
            f"(input_shape={ishape!r}); cannot window the signal."
        )
    channels = _resolve_channels(ishape[2], ishape)
    _reject_teacher(channels, ishape, abspath)
    # ...and the OUTPUT side, which used to be recorded and never checked. See
    # _check_output_contract: an unchecked output is a dwell-time defect, not a crash.
    oshape = _check_output_contract(
        None if isinstance(model.output_shape, list) else model.output_shape,
        int(window), abspath)

    wrapper = KerasEventModel(
        model=model,
        window=int(window),
        path=abspath,
        mtime=mtime,
        channels=channels,
        input_shape=ishape,
        output_shape=oshape,
    )
    _cache_put(key, wrapper)
    return wrapper


def _first_window_in_config(obj: Any) -> int | None:
    """Recursively find the first ``batch_shape`` / ``batch_input_shape`` of rank 3
    in a Keras config and return its middle (time) dimension."""
    if isinstance(obj, dict):
        for key in ("batch_shape", "batch_input_shape"):
            val = obj.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                w = val[1]
                if isinstance(w, int) and w > 0:
                    return int(w)
        for v in obj.values():
            got = _first_window_in_config(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _first_window_in_config(v)
            if got is not None:
                return got
    return None


def peek_model_window(path: str | os.PathLike) -> int | None:
    """The model's input window length, read WITHOUT importing TensorFlow.

    A ``.keras`` file is a zip whose ``config.json`` carries the input layer's
    ``batch_shape`` (Keras 3) or ``batch_input_shape`` (Keras 2). Reading it is
    cheap and safe on the GUI thread, which is the point: the ML panel can default
    its preprocessing preset from the model's width (2048 -> a ferritin notebook
    preset, 3072 -> an apo preset) without ever loading TF on the main thread. The
    width only narrows the choice -- see
    :func:`~nanoscope.analysis.ml_preprocess.preset_for_window`.

    Returns ``None`` for anything it cannot read (missing file, non-zip, no
    recognisable input shape) -- callers treat that as "unknown" and change
    nothing.
    """
    import json
    import zipfile

    try:
        with zipfile.ZipFile(os.fspath(path)) as z:
            with z.open("config.json") as fh:
                cfg = json.load(fh)
    except Exception:
        return None
    try:
        return _first_window_in_config(cfg)
    except Exception:  # pragma: no cover - defensive
        return None


def _first_channels_in_config(obj: Any) -> int | None:
    """The first rank-3 input shape's LAST (feature) dimension, mirroring
    :func:`_first_window_in_config`. Used to spot a teacher without importing TF."""
    if isinstance(obj, dict):
        for key in ("batch_shape", "batch_input_shape"):
            val = obj.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                c = val[2]
                if isinstance(c, int) and c > 0:
                    return int(c)
        for v in obj.values():
            got = _first_channels_in_config(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _first_channels_in_config(v)
            if got is not None:
                return got
    return None


def peek_model_channels(path: str | os.PathLike) -> int | None:
    """The model's input FEATURE width, read WITHOUT importing TensorFlow.

    Same zipfile+json trick as :func:`peek_model_window`, and for the same reason: the
    GUI records a model path on the main thread and must not import TF there. This lets
    the panel refuse a teacher the moment it is picked, instead of accepting it into the
    label and only failing after the user has chosen a range and waited for inference.

    ``None`` means "cannot tell" -- the caller must treat that as allowed, because the
    authoritative check still runs in :func:`load_keras_model` at run time.
    """
    import json
    import zipfile

    try:
        with zipfile.ZipFile(os.fspath(path)) as z:
            with z.open("config.json") as fh:
                cfg = json.load(fh)
    except Exception:
        return None
    try:
        return _first_channels_in_config(cfg)
    except Exception:  # pragma: no cover - defensive
        return None


#: Input width of the FEATURE contract, i.e. the teacher architecture.
TEACHER_CHANNELS = 12


def is_teacher_model(path: str | os.PathLike) -> bool:
    """True if ``path`` is a teacher (12-feature) export, judged WITHOUT TensorFlow.

    Conservative by construction: an unreadable or unrecognisable config returns False
    (allowed), because this is the cheap GUI-thread gate and
    :func:`load_keras_model` is the authoritative one.
    """
    return peek_model_channels(path) == TEACHER_CHANNELS


def _reject_teacher(channels: int, ishape: tuple, path: str) -> None:
    """Refuse a teacher (12-feature) model. THE app-level rule: students only.

    The FEATURE path below is intact and correct -- it is what produced the teacher
    comparison runs -- but a teacher is not a detector this app should run. It is the
    DISTILLATION SOURCE: the large model whose outputs label the data the student is
    trained on. The student is the artefact meant to be deployed, and the two are not
    interchangeable just because both happen to emit a per-sample probability.

    Enforced on the INPUT WIDTH, not the filename. Across the owner's model corpus the
    two are perfectly aligned -- every ``*student*`` is ``(None, W, 1)`` and all seven
    ``*teacher*`` files are ``(None, W, 12)``, 8 of 8 -- but a filename is a label a user
    can mistype and an input shape is what the architecture actually IS.

    Raised here rather than in the GUI so every entry point is covered by one rule: the
    menu action, a script calling :func:`load_keras_model`, and the analysis module all
    go through this door.
    """
    if channels != TEACHER_CHANNELS:
        return
    raise ValueError(
        f"{os.path.basename(path)}: this is a TEACHER model — its input is "
        f"{ishape!r}, the {TEACHER_CHANNELS}-feature contract. This app runs STUDENT "
        f"models only: a teacher is the distillation source that labels training data, "
        f"not the detector meant to be deployed. Load the matching student export "
        f"instead (a single-channel model, input (None, {ishape[1]}, 1))."
    )


def _resolve_channels(n_features: Any, ishape: tuple) -> int:
    """Map ``input_shape[-1]`` to a supported channel count (1 or 12).

    ``None`` (an unspecified last dim) is treated as the RAW single channel, so
    the pre-existing global-raw student -- which some exports leave unshaped in
    the feature axis -- keeps working. Any other fixed count is rejected with a
    clear message; the two supported contracts are the only preprocessing paths.
    """
    if n_features is None:
        return 1
    n = int(n_features)
    if n in (1, 12):
        return n
    raise ValueError(
        f"model expects {n} input features per window (input_shape={ishape!r}), "
        "but this detector supports only 1 (RAW: a single detrended, global-raw "
        "normalized current channel) or 12 (FEATURE: per-window "
        "create_multiscale_features_A)."
    )


# ---------------------------------------------------------------------------
# Preprocessing (PART A + PART B of the contract) -- pure, no TF.
# ---------------------------------------------------------------------------
def detrend_for_model(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detrend per the RAW contract: notch 60/120/180 Hz (Q=30, filtfilt) then
    subtract a 500 ms / poly-3 Savitzky-Golay envelope. Returns float32.

    This is a FIXED model contract -- it is intentionally NOT the app's
    user-facing notch/savgol filters, which use different parameters.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError("signal must be a non-empty 1-D array")
    if not np.isfinite(sig).all():
        raise ValueError("signal contains NaN/Inf; detrending is undefined")

    # Up-front length guard: the notch filtfilt pads by ~3*ntaps (9 for the
    # 2nd-order iirnotch), so anything shorter is undefined for the whole
    # contract. Fail here with one clear "too short" message rather than deep
    # inside scipy.
    if sig.size < 10:
        raise ValueError(
            "recording is too short for the ML detrending contract "
            f"(need >= 10 samples for the notch filter, got {sig.size})"
        )

    for h in NOTCH_HZ:
        if h < fs / 2.0:
            b, a = iirnotch(h, NOTCH_Q, fs)
            sig = filtfilt(b, a, sig)

    w = int(SAVGOL_MS * fs / 1000.0)
    w = w + 1 if w % 2 == 0 else w
    max_w = len(sig) - 1 if len(sig) % 2 == 0 else len(sig)
    w = min(w, max_w)
    if w <= SAVGOL_POLY:
        raise ValueError(
            "recording is too short for the 500 ms / poly-3 SavGol detrending "
            f"contract (need > {SAVGOL_POLY + 1} samples, got {len(sig)})"
        )
    envelope = savgol_filter(sig, window_length=w, polyorder=SAVGOL_POLY)
    return (sig - envelope).astype(np.float32)


def normalize_global_raw(signal: np.ndarray) -> np.ndarray:
    """Global-raw normalization: ``signal / (max(|signal|) + 1e-8)``, float32.

    NO mean subtraction -- this is a pure rescale, exactly as the model was
    trained (the "global-raw" teacher scale).
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    scale = float(np.max(np.abs(sig)) + 1e-8) if sig.size else 1e-8
    return (sig / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# FEATURE path: 12 multiscale features per window (teacher contract).
#
# Ported VERBATIM from StudentModel.ipynb cell 5. Each feature column is
# self-standardized (zero-mean / unit-variance), so the FEATURE path does NOT
# apply the global-raw ÷max|x| normalization -- doing so would be wrong. These
# are pure numpy + scipy (gaussian_filter1d / savgol_filter); no TensorFlow.
# ---------------------------------------------------------------------------
def _standardize(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    a = np.asarray(a, float)
    return np.nan_to_num((a - np.mean(a)) / (np.std(a) + eps))


def _tkeo_basic(x: np.ndarray) -> np.ndarray:
    """Basic Teager-Kaiser Energy Operator."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.zeros_like(x)
    y = np.zeros_like(x)
    y[0] = x[0] ** 2 - x[1] * x[0]
    y[1:-1] = x[1:-1] ** 2 - x[2:] * x[:-2]
    y[-1] = x[-1] ** 2 - x[-1] * x[-2]
    return y


def _hilbert_env_safe(x: np.ndarray) -> np.ndarray:
    """Approximate envelope without scipy overhead."""
    a = np.abs(x)
    return 0.5 * a + 0.5 * gaussian_filter1d(a, sigma=5)


def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Moving RMS."""
    win = max(3, int(win))
    pad = win // 2
    xp = np.pad(x, pad, mode='reflect')
    sq = gaussian_filter1d(xp * xp, sigma=win / 6.0, mode='nearest')[pad:-pad]
    return np.sqrt(np.maximum(sq, 0))


def _robust_range(x: np.ndarray, win: int) -> np.ndarray:
    """Approximate (P90 - P10) via smoothed absolute deviation."""
    win = max(5, int(win))
    med = gaussian_filter1d(x, sigma=win / 6.0)
    dev = np.abs(x - med)
    return gaussian_filter1d(dev, sigma=win / 8.0)


def _monotonicity(g1: np.ndarray, win: int) -> np.ndarray:
    """Fraction of recent derivative signs that are consistent."""
    win = max(5, int(win))
    s = np.sign(g1)
    s[s == 0] = 1
    sm = gaussian_filter1d(s.astype(float), sigma=win / 6.0)
    return np.abs(sm)


def _autocorr_short(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """Short-lag autocorrelation."""
    y = np.zeros_like(x, dtype=float)
    y[lag:] = x[lag:] * x[:-lag]
    return gaussian_filter1d(y, sigma=3)


def create_multiscale_features_A(
    signal: np.ndarray, window_len: int = 2048, fs: float = 50_000
) -> np.ndarray:
    """
    Generate 12 optimized multiscale features for event detection.

    Output shape: (window_len, 12)
    Features:
      1. raw signal (standardized)
      2. first derivative
      3. second derivative
      4-7. multi-scale TKEO at scales [3,10,25,50] (0.06-1.0 ms)
      8. HF/LF RMS ratio
      9. robust local range
      10. Hilbert-like envelope
      11. monotonicity
      12. short-lag autocorrelation
    """
    x = np.asarray(signal, np.float32)
    if len(x) != window_len:
        raise ValueError(f"Signal length {len(x)} != window_len {window_len}")

    feats = []
    fs_f = float(fs)

    # 1-3: base derivatives
    feats.append(_standardize(x))
    g1 = np.gradient(x)
    g2 = np.gradient(g1)
    feats.append(_standardize(g1))
    feats.append(_standardize(g2))

    # 4-7: multi-scale TKEO
    # Scales span 0.06-1.0 ms to match this dataset's dwell (median 0.86 ms).
    # Old [1,3,5,7] (0.02-0.14 ms) was tuned to the 15 nm 4-sample events.
    for scale in [3, 10, 25, 50]:
        w = max(5, 2 * scale + 1)
        if w % 2 == 0:
            w += 1
        sm = savgol_filter(x, w, 3, mode='interp')
        t = sm[1:-1] ** 2 - sm[2:] * sm[:-2]
        t = np.pad(t, (1, 1), mode='edge')
        # post-smoothing capped at sigma=5 to keep edges localized (offset ~0.54 ms);
        # decoupled from the pre-smoothing scale, which still sets the energy scale.
        feats.append(_standardize(gaussian_filter1d(t, sigma=min(scale, 5))))

    # 8: HF/LF RMS ratio
    hf = _moving_rms(x, win=int(0.2e-3 * fs_f))
    lf = _moving_rms(x, win=int(2.0e-3 * fs_f))
    feats.append(_standardize(hf / (lf + 1e-8)))

    # 9-12: local & spectral descriptors
    feats.append(_standardize(_robust_range(x, win=int(1.0e-3 * fs_f))))
    feats.append(_standardize(_hilbert_env_safe(x)))
    feats.append(_standardize(_monotonicity(g1, win=int(0.5e-3 * fs_f))))
    feats.append(_standardize(_autocorr_short(x, lag=1)))

    F = np.column_stack(feats)
    if F.shape != (window_len, 12):
        raise ValueError(f"Feature matrix shape mismatch: got {F.shape}, expected {(window_len, 12)}")

    return F.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference: sliding window, 50 % overlap, overlap-average. TF only inside.
# ---------------------------------------------------------------------------
def predict_probability(
    model_wrapper: KerasEventModel,
    signal: np.ndarray,
    *,
    fs: float | None = None,
    batch: int = 256,
    anchor_tail: bool = True,
    last_start_inclusive: bool = False,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Run the model over ``signal`` and return a full-length float32 probability
    trace in ``[0, 1]``.

    Windows of length ``W = model_wrapper.window`` are stepped by ``W // 2``
    (50 % overlap), predicted in batches, and overlap-AVERAGED back onto a
    per-sample trace. Model output (float16) is cast to float32 and taken as
    ``[..., 0]``.

    ``anchor_tail`` (default True, the app's legacy behaviour) adds a final window
    anchored at ``len(signal) - W`` so every sample is covered. Neither notebook
    family does this, so their presets set it False.

    With ``anchor_tail=False``, ``last_start_inclusive`` picks WHICH notebook loop
    to reproduce -- the two differ, and only here:

      * ``False`` -- ferritin's ``range(0, n - W, step)``;
      * ``True``  -- apo's ``np.arange(0, max(1, n - W + 1), step)``.

    They agree on every ``n`` except those with ``(n - W) % step == 0``, where the
    ferritin loop stops one window early and leaves a whole stride at probability 0
    while the apo loop lands exactly on ``n - W`` and covers the recording. Ignored
    when ``anchor_tail`` is set, which already reaches ``n - W``.

    Dispatch is on ``model_wrapper.channels``:

      * ``channels == 1`` (RAW): each window is fed as ``(W, 1)``. ``signal`` is
        expected to be already detrended + global-raw normalized (see
        :func:`detrend_for_model` / :func:`normalize_global_raw`).
      * ``channels == 12`` (FEATURE): each (padded) window is passed through
        :func:`create_multiscale_features_A` -> ``(W, 12)`` and fed as ``(W, 12)``.
        ``signal`` is expected to be the DETRENDED signal only -- the features
        self-standardize, so NO global-raw normalization is applied. ``fs`` is
        REQUIRED (the features are sampling-rate dependent).

    Args:
        fs: sample rate in Hz; required for the FEATURE path, ignored for RAW.
        batch: windows per ``model.predict`` call (speed knob).
        progress: optional ``progress(frac, msg)`` sink, called per batch with
            ``frac`` in ``[0, 1]``.
        cancel: optional ``cancel() -> bool``; when it returns True the run
            aborts by raising :class:`MLCancelled`.
    """
    x = np.asarray(signal, dtype=np.float32).ravel()
    n = x.size
    if n == 0:
        raise ValueError("signal is empty")
    W = int(model_wrapper.window)
    if W <= 0:
        raise ValueError(f"model window must be positive, got {W}")
    channels = int(model_wrapper.channels)
    if channels not in (1, 12):
        raise ValueError(
            f"unsupported channel count {channels}; expected 1 (RAW) or 12 (FEATURE)"
        )
    if channels == 12:
        if fs is None or not (np.isfinite(float(fs)) and float(fs) > 0):
            raise ValueError(
                "the FEATURE path (channels == 12) requires a positive fs to build "
                "the multiscale features per window"
            )
        fs = float(fs)
    step = max(1, W // 2)
    batch = max(1, int(batch))

    if anchor_tail:
        starts = list(range(0, max(1, n - W + 1), step))
        if n >= W and starts[-1] != n - W:  # anchor a final window; the tail is covered
            starts.append(n - W)
    elif last_start_inclusive:
        # Apo notebook contract: `np.arange(0, max(1, n - W + 1), step)`. The bound
        # is INCLUSIVE of n - W, so a recording whose length lands exactly on a
        # stride is fully covered; otherwise the tail past the last start stays at
        # probability 0, same as the ferritin loop. `max(1, ...)` is the notebook's
        # own guard and yields the single zero-padded window when n <= W.
        starts = list(range(0, max(1, n - W + 1), step))
    else:
        # Ferritin notebook contract: `for start in range(0, n - W, step)`. The tail
        # past the last full stride gets no window at all and stays at probability 0.
        # The `or [0]` only fires when n <= W, where the notebook loop would emit NO
        # windows and hand back an all-zero trace; one zero-padded window is strictly
        # more useful and cannot affect any n > W run.
        starts = list(range(0, max(0, n - W), step)) or [0]
    total = len(starts)

    out = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    buf: list[np.ndarray] = []
    idxs: list[int] = []
    done = 0

    def _to_model_input(win: np.ndarray) -> np.ndarray:
        """One (length-W, already padded) window -> the per-window model input.

        RAW keeps the raw window (the channel axis is added once per batch in
        ``_flush``); FEATURE expands it to the (W, 12) multiscale feature matrix.
        """
        if channels == 1:
            return win
        return create_multiscale_features_A(win, window_len=W, fs=fs)

    def _flush() -> None:
        nonlocal done
        if not buf:
            return
        if cancel is not None and cancel():
            raise MLCancelled("ML inference cancelled")
        arr = np.asarray(buf, dtype=np.float32)
        if channels == 1:
            arr = arr[..., None]  # (B, W) -> (B, W, 1)
        # channels == 12: arr is already (B, W, 12)
        preds = np.asarray(model_wrapper.model.predict(arr, verbose=0), dtype=np.float32)
        if preds.ndim == 3:  # (B, W, 1) -> (B, W)
            preds = preds[:, :, 0]
        for p, s in zip(preds, idxs):
            e = min(s + W, n)
            out[s:e] += p[: e - s]
            counts[s:e] += 1.0
        done += len(idxs)
        buf.clear()
        idxs.clear()
        if progress is not None:
            progress(min(1.0, done / total), f"ML inference: {done}/{total} windows")

    if progress is not None:
        progress(0.0, f"ML inference: 0/{total} windows")
    for s in starts:
        if cancel is not None and cancel():
            raise MLCancelled("ML inference cancelled")
        win = x[s : s + W]
        if win.size < W:
            win = np.pad(win, (0, W - win.size))
        buf.append(_to_model_input(win))
        idxs.append(s)
        if len(buf) >= batch:
            _flush()
    _flush()

    counts[counts == 0] = 1.0
    return (out / counts).astype(np.float32)


# ---------------------------------------------------------------------------
# Decode: smooth -> hysteresis -> merge -> min-duration.  Pure, no TF.
# ---------------------------------------------------------------------------
def _smooth(prob: np.ndarray, kernel: int) -> np.ndarray:
    kernel = int(kernel)
    if kernel <= 1:
        return prob
    if kernel % 2 == 0:
        kernel += 1
    if len(prob) < kernel:
        return prob
    return medfilt(prob, kernel)


def _hysteresis(prob: np.ndarray, enter: float, exit_thr: float) -> list[tuple[int, int]]:
    """Two-threshold region extraction. A run of samples ``>= exit_thr`` becomes a
    region only if it ever reaches ``>= enter``. Returns INCLUSIVE ``[start, end]``."""
    prob = np.asarray(prob)
    above_exit = (prob >= exit_thr).astype(np.int8)
    above_enter = prob >= enter
    diff = np.diff(above_exit, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    regions: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if above_enter[s:e].any():
            regions.append((int(s), int(e - 1)))  # inclusive end
    return regions


def _merge(regions: list[tuple[int, int]], fs: float, merge_s: float) -> list[tuple[int, int]]:
    if not regions:
        return []
    gap = int(fs * merge_s)
    merged: list[tuple[int, int]] = []
    cs, ce = regions[0]
    for s, e in regions[1:]:
        if s - ce <= gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def _min_dur(regions: list[tuple[int, int]], fs: float, min_dur_s: float) -> list[tuple[int, int]]:
    min_n = int(fs * min_dur_s)
    return [(s, e) for s, e in regions if (e - s + 1) >= min_n]


def decode_regions(
    prob: np.ndarray,
    fs: float,
    *,
    enter: float = DEFAULT_ENTER,
    exit: float = DEFAULT_EXIT,
    smooth_kernel: int = DEFAULT_SMOOTH_KERNEL,
    merge_s: float = DEFAULT_MERGE_S,
    min_dur_s: float = DEFAULT_MIN_DUR_S,
) -> list[tuple[int, int]]:
    """Turn a per-sample probability trace into INCLUSIVE ``[start, end]`` event
    regions (sample indices).

    Pipeline: median-smooth (``smooth_kernel``, odd) -> hysteresis
    (``enter``/``exit``) -> merge regions closer than ``merge_s`` -> drop regions
    shorter than ``min_dur_s``. ``enter``/``exit`` are the tunable thresholds;
    the defaults reproduce the reference (0.5 / 0.3).

    The returned ends are INCLUSIVE (``dwell = end - start + 1``). The pipeline
    adapter converts these to characterize's exclusive ``ends`` (``end + 1``).
    """
    if not (0.0 < enter <= 1.0):
        raise ValueError(f"enter must be in (0, 1], got {enter}")
    if not (0.0 <= exit <= enter):
        raise ValueError(f"exit must satisfy 0 <= exit <= enter, got exit={exit}, enter={enter}")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")

    prob = np.asarray(prob, dtype=np.float32).ravel()
    if prob.size == 0:
        return []
    sm = _smooth(prob, smooth_kernel)
    regions = _hysteresis(sm, enter, exit)
    regions = _merge(regions, fs, merge_s)
    regions = _min_dur(regions, fs, min_dur_s)
    return [(int(s), int(e)) for s, e in regions]


# ---------------------------------------------------------------------------
# End to end: detrend -> normalize -> predict -> decode (+ provenance).
# ---------------------------------------------------------------------------
def _tail_desc(cfg: MLPreprocessConfig) -> str:
    """One phrase naming which window-start policy actually ran, for provenance."""
    if cfg.anchor_tail:
        return "anchored"
    if cfg.last_start_inclusive:
        return "starts arange(0, n-W+1, step) (apo notebook)"
    return "uncovered (notebook)"


def ml_detect_regions(
    signal: np.ndarray,
    fs: float,
    model_wrapper: KerasEventModel,
    *,
    enter: float = DEFAULT_ENTER,
    exit: float = DEFAULT_EXIT,
    smooth_kernel: int = DEFAULT_SMOOTH_KERNEL,
    merge_s: float = DEFAULT_MERGE_S,
    min_dur_s: float = DEFAULT_MIN_DUR_S,
    preprocess: MLPreprocessConfig | None = None,
    decode: MLDecodeConfig | None = None,
    detrended: np.ndarray | None = None,
    gate_signal: np.ndarray | None = None,
    norm_scale: float | None = None,
    norm_center: float | None = None,
    norm_scope: str | None = None,
    batch: int = 256,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[str, Any]]:
    """Run the whole ML detector on one channel.

    preprocess (both contracts) -> [RAW: global normalize | FEATURE: nothing]
    -> windowed predict (per-window features on the FEATURE path; overlap-average)
    -> decode. The path is auto-selected from ``model_wrapper.channels``
    (1 == RAW, 12 == FEATURE). Returns ``(regions, prob, provenance)``:

      * ``regions``: INCLUSIVE ``[start, end]`` sample-index tuples (same shape
        the built-in Schmitt detector emits; feed them into ``characterize``).
      * ``prob``: the full-length float32 probability trace (for overlays/QC).
      * ``provenance``: every effective parameter that produced them, so a run is
        reproducible from the stamp alone.

    Args:
        preprocess: the model's OWN signal chain
            (:class:`~nanoscope.analysis.ml_preprocess.MLPreprocessConfig`). ``None``
            selects the ``"apo_raw_global"`` preset (formerly named ``"apo_legacy"``),
            which is bit-for-bit :func:`detrend_for_model` -- so an un-configured call
            behaves exactly as before. The general "Preprocess" tab chain is never
            used here.
        decode: the notebook decode contract
            (:class:`~nanoscope.analysis.ml_preprocess.MLDecodeConfig`). ``None``
            falls back to the legacy :func:`decode_regions` driven by the
            ``enter``/``exit``/``smooth_kernel``/``merge_s``/``min_dur_s`` arguments.
        detrended: an ALREADY-preprocessed version of ``signal`` (same length).
            Supplied by the adapter when the preprocessing had to be measured over
            the whole recording rather than the analysed slice (masked-median's MAD
            and the SavGol envelope are both span-dependent). Skips the preprocess
            stage; the config is still stamped.
        gate_signal: the detrended trace **in pA** the decode's amplitude gates
            measure their depths on. ``None`` uses ``detr`` -- correct wherever
            ``detr`` is still in picoamps, which is every path except the
            ``robust_sigma`` one, where the adapter hands ``detrended`` a finished,
            CLIPPED model input in sigma units. On that path the adapter must pass
            the pre-normalization detrended pA here, or ``min_depth_pa`` /
            ``min_fracblock`` would be compared against the wrong units entirely.
            Ignored when every gate is off (every preset but the three student ones).
            The RAW gate input -- what ``min_fracblock`` divides by -- is always
            ``signal`` itself, since that is the raw calibrated slice by contract.
        norm_scale: the ``max|x - center| + 1e-8`` divisor to use, when it must come
            from a wider span than ``signal`` (whole-recording normalization scope).
            ``None`` measures it on the preprocessed ``signal`` itself.
        norm_center: the mean to subtract before dividing, for the same reason and
            over the same span as ``norm_scale``. Only consulted when the config sets
            ``normalize_center`` (the apo notebook chain); ``None`` measures it on the
            preprocessed ``signal`` itself. Supply it together with ``norm_scale`` --
            a scale measured over the recording must have been measured about the
            recording's centre.
        norm_scope: what the caller ACHIEVED, for provenance -- ``"recording"`` or
            ``"range"``. Defaults to ``"recording"`` when ``norm_scale`` was supplied
            (the caller measured it over the whole channel), else ``"range"``. The
            adapter passes ``"recording"`` explicitly when the analysed range already
            IS the whole recording.

    ``progress(frac, msg)`` reports 0..1 (inference dominates the bar).
    ``cancel() -> True`` aborts by raising :class:`MLCancelled`.
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    channels = int(model_wrapper.channels)
    input_mode = "features" if channels == 12 else "raw"
    pre_cfg = (
        preprocess if preprocess is not None else PREPROCESS_PRESETS["apo_raw_global"]
    )

    if progress is not None:
        progress(
            0.0,
            "preprocessing (detrend + normalize)" if channels == 1
            else "preprocessing (detrend; per-window features)",
        )
    if detrended is not None:
        detr = np.asarray(detrended, dtype=np.float32).ravel()
        if detr.size != sig.size:
            raise ValueError(
                f"detrended override has {detr.size} samples but signal has {sig.size}"
            )
        preprocess_scope = "recording"
    else:
        detr = apply_ml_preprocess(sig, fs, pre_cfg)
        # We preprocessed exactly the signal we were handed. Whether that IS the whole
        # recording only the caller knows -- when the analysed range already spans the
        # file it says so via ``norm_scope``.
        preprocess_scope = str(norm_scope) if norm_scope is not None else "range"

    # RAW path divides by max|x|; FEATURE path must NOT -- create_multiscale_features_A
    # standardizes every column internally, so global-raw scaling would be wrong.
    normalized = channels == 1 and pre_cfg.normalize
    if normalized:
        # The apo notebook subtracts the mean BEFORE taking max|.|, so the centre has
        # to be resolved first and the scale measured about it. Ferritin's centre is
        # 0.0 and this collapses to the plain x/(max|x|+1e-8).
        if pre_cfg.normalize_center:
            center = (
                float(norm_center) if norm_center is not None
                else normalization_center(detr)
            )
        else:
            center = 0.0
        scale = (
            float(norm_scale) if norm_scale is not None
            else normalization_scale(detr, center)
        )
        scale_scope = str(
            norm_scope if norm_scope is not None
            else ("recording" if norm_scale is not None else "range")
        )
        model_input = normalize_global(detr, scale, center)
    else:
        scale = None
        center = None
        scale_scope = "n/a"
        model_input = detr

    def _inference_progress(frac: float, msg: str) -> None:
        if progress is not None:
            progress(0.03 + 0.94 * frac, msg)

    prob = predict_probability(
        model_wrapper,
        model_input,
        fs=fs,
        batch=batch,
        anchor_tail=pre_cfg.anchor_tail,
        last_start_inclusive=pre_cfg.last_start_inclusive,
        progress=(_inference_progress if progress is not None else None),
        cancel=cancel,
    )

    if progress is not None:
        progress(0.98, "decoding events")
    if decode is not None:
        # ``signal=detr`` is what the onset snap / offset trim read, unchanged.
        # ``gate_signal`` is what the AMPLITUDE gates read, and it is a separate
        # argument precisely because on the robust-sigma path those are not the
        # same trace: ``detr`` is then the clipped model input in sigma units,
        # while a pA or dI/I0 threshold has to see picoamps. ``raw=sig`` is the raw
        # calibrated slice, the only trace still carrying the open-pore level that
        # the time-local I0 IS -- detrending is exactly what removes it.
        gate_sig = detr if gate_signal is None else np.asarray(
            gate_signal, dtype=np.float32
        ).ravel()
        if gate_sig.size != detr.size:
            raise ValueError(
                f"gate_signal has {gate_sig.size} samples but the decoded trace has "
                f"{detr.size}; the gates index regions into it directly"
            )
        regions = decode_ml_regions(
            prob, fs, decode, signal=detr, gate_signal=gate_sig, raw=sig
        )
        enter = float(decode.enter_threshold)
        exit = float(decode.exit_threshold)
        smooth_kernel = int(decode.smoothing_kernel)
        merge_s = float(decode.merge_radius_ms) / 1000.0
        min_dur_s = float(decode.min_event_ms) / 1000.0
    else:
        regions = decode_regions(
            prob,
            fs,
            enter=enter,
            exit=exit,
            smooth_kernel=smooth_kernel,
            merge_s=merge_s,
            min_dur_s=min_dur_s,
        )
    if progress is not None:
        progress(1.0, f"done: {len(regions)} events")

    if channels == 1:
        preprocess_desc = pre_cfg.describe()
    else:
        # The FEATURE contract standardizes every column internally, so the global-raw
        # rescale is not applied at all -- say so, and do not let the config's
        # normalize flag imply otherwise.
        preprocess_desc = (
            f"{pre_cfg.replace(normalize=False).describe()}; per-window "
            "create_multiscale_features_A -> (W, 12) self-standardized features; "
            "NO global-raw normalize"
        )

    provenance: dict[str, Any] = {
        "detector": "ml_keras",
        "model_path": model_wrapper.path,
        "model_window": int(model_wrapper.window),
        "model_channels": channels,
        "channels": channels,
        "input_mode": input_mode,
        "model_input_shape": tuple(model_wrapper.input_shape),
        "model_output_shape": tuple(model_wrapper.output_shape),
        "fs": float(fs),
        "n_samples": int(sig.size),
        "enter": float(enter),
        "exit": float(exit),
        "smooth_kernel": int(smooth_kernel),
        "merge_s": float(merge_s),
        "min_dur_s": float(min_dur_s),
        # Stamped as TOP-LEVEL keys, not left to be dug out of ``ml_decode``,
        # because they are the stage that removed the most regions: on zn_30h the
        # sustained gate turns 11004 fired regions into 1076 events. A run whose
        # count cannot be explained from its own stamp is not reproducible.
        # 0.0 means the gate did not run (the legacy path never runs them).
        "min_depth_sigma": float(decode.min_depth_sigma) if decode is not None else 0.0,
        "min_depth_pa": float(decode.min_depth_pa) if decode is not None else 0.0,
        "min_fracblock": float(decode.min_fracblock) if decode is not None else 0.0,
        "batch": int(batch),
        "n_regions": int(len(regions)),
        # -- the ML detector's OWN preprocessing, fully stamped ----------------
        "ml_preset": str(pre_cfg.preset),
        "ml_preprocess": pre_cfg.to_dict(),
        "ml_preprocess_desc": pre_cfg.describe(),
        "ml_preprocess_scope": preprocess_scope,
        "ml_decode_preset": str(decode.preset) if decode is not None else "legacy",
        "ml_decode": decode.to_dict() if decode is not None else None,
        "ml_decode_desc": (
            decode.describe() if decode is not None
            else (f"median smooth k={smooth_kernel}; hysteresis enter {enter:g} / "
                  f"exit {exit:g}; merge <= {merge_s * 1000:g} ms; "
                  f"drop < {min_dur_s * 1000:g} ms")
        ),
        "normalized": bool(normalized),
        "norm_scope": scale_scope,
        "norm_scale": scale,
        "norm_center": center,
        "anchor_tail": bool(pre_cfg.anchor_tail),
        "last_start_inclusive": bool(pre_cfg.last_start_inclusive),
        "preprocess": preprocess_desc,
        "inference": (
            "sliding window W, step W//2, 50% overlap-average, CPU forced; "
            f"input_mode={input_mode} (channels={channels}); "
            f"tail={_tail_desc(pre_cfg)}"
        ),
    }
    return regions, prob, provenance
