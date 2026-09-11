"""Plain frozen-input job for the existing sub-state fitting kernels."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from ..analysis.baseline import baseline_config_from_params, characterisation_baseline
from ..analysis.cancel import check_cancelled
from ..analysis.cusum import DEFAULT_CUSUM
from ..analysis.modules import detection_signal, fit_levels
from .analysis_worker import FrozenChannelSource


@dataclass(frozen=True)
class LevelFitOutcome:
    """Not an AnalysisResult: the common detection sink must never install it."""
    token: object
    result: Any
    inputs: dict[str, Any]


def source_signature(value) -> str:
    """Stable snapshot of the JSON-like settings/provenance, not an array hash."""
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def run_level_fit(source: FrozenChannelSource, accepted, provenance, *,
                  doc_id: str, channel: int, token: object, report) -> LevelFitOutcome:
    """Condition, reconstruct the recorded baseline and fit without GUI access."""
    report(0.02, "Preparing calibrated current for sub-state fitting…")
    chan = source.materialize()
    fs = float(chan.sample_rate_hz)
    check_cancelled()
    report(0.12, "Rebuilding the detection's conditioning…")
    signal = detection_signal(np.asarray(chan.data), fs, provenance)
    check_cancelled()
    report(0.35, "Rebuilding the detection's open-pore reference…")
    cfg, origin = baseline_config_from_params(provenance)
    base = characterisation_baseline(
        np.asarray(signal, dtype=np.float64), fs, config=cfg,
        polarity=provenance.get("polarity"), param_source=origin)
    check_cancelled()
    report(0.65, f"Fitting sub-state levels in {len(accepted):,} accepted events…")
    result = fit_levels(
        accepted, np.asarray(signal, dtype=np.float64), fs,
        doc_id=doc_id, channel=channel,
        baseline=np.asarray(base.baseline, dtype=np.float64),
        polarity=int(provenance.get("polarity", 1)), provenance=provenance,
        calibration_fingerprint=source.calibration.fingerprint(),
        baseline_fingerprint=base.config.fingerprint(fs),
    )
    check_cancelled()
    inputs = {
        "doc_id": doc_id, "channel": int(channel), "fs": fs,
        "baseline": np.asarray(base.baseline, dtype=np.float64),
        "sigma": result.params.get("sigma_pA"), "tau": result.params.get("tau_samples"),
        "polarity": int(provenance.get("polarity", 1)), "provenance": provenance,
        "config": DEFAULT_CUSUM,
        "calibration_fingerprint": result.calibration_fingerprint,
        "baseline_fingerprint": result.baseline_fingerprint,
    }
    report(1.0, "Sub-state fit complete; checking its source before applying…")
    return LevelFitOutcome(token, result, inputs)
