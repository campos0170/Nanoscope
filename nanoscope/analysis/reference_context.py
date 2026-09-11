"""Physical-domain safeguards for the optional user open-pore reference.

These checks never change a setting or convert a unit. An unsupported combination
must be explicitly changed by the user, not silently interpreted as another run.
"""
from __future__ import annotations

from typing import Any, Mapping

from .baseline import BATCH_SCOPE_KEY, baseline_config_from_params


REFERENCE_KEY = "baseline_open_pore_reference"
SUPPORTED_DETECTORS = ("detect.schmitt", "detect.cusum")


def validate_open_pore_context(
    params: Mapping[str, Any], *, units: str | None,
    module_id: str | None = None, batch: bool = False,
) -> None:
    """Refuse a reference in a different signal domain or unvalidated workflow."""
    config, _ = baseline_config_from_params(params)
    if getattr(config, "open_pore_reference", None) is None:
        return
    if batch or BATCH_SCOPE_KEY in params or params.get("baseline_param_source") == "batch":
        raise ValueError(
            "An open-pore reference belongs to one recording and channel. "
            "Run detection on that recording; copying a reference through Batch detect "
            "is not supported. No automatic reference was substituted."
        )
    if module_id not in SUPPORTED_DETECTORS:
        raise ValueError(
            "User open-pore references are currently supported for Schmitt and CUSUM "
            "detection only. Choose one of these or explicitly return to Automatic reference."
        )
    if str(units or "").strip() != "pA":
        raise ValueError(
            "User open-pore current is in pA. Verify this channel's calibration and "
            "pA units before using the reference; changing a unit label alone does not convert data."
        )
    incompatible = [label for key, label in (
        ("detrend_enabled", "runtime baseline subtraction"),
        ("hp_enabled", "high-pass filtering"),
        ("normalize_enabled", "normalization"),
    ) if params.get(key)]
    if incompatible:
        raise ValueError(
            "A calibrated pA open-pore reference cannot be combined with "
            + ", ".join(incompatible)
            + " in this mode. Disable those stages explicitly or return to Automatic reference."
        )


def reference_change_requires_detection(old_config: Any, new_config: Any) -> bool:
    """A changed reference requires fresh noise/polarity and candidate detection."""
    old = getattr(old_config, "open_pore_reference", None)
    new = getattr(new_config, "open_pore_reference", None)
    return (old is not None or new is not None) and old != new
