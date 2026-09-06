"""The voltage-dependence WINDOW says what it pooled -- on screen and in the PNG.

REGRESSION FOR THE apo-run-2026-08-19 FINDINGS **High-1** AND **High-2**, at the
surface the owner was actually looking at.

``tests/analysis/test_voltage_dep_comparability.py`` pins the verdict in the pure
core. This file pins that the verdict REACHES A HUMAN, which is the half that
failed: on the owner's three apoferritin recordings the window fitted "median
dI/I0" across three different baseline configurations and drew it with, in the
report's words, no warning anywhere in the window; and it printed
``V0 = 0.653 V`` before a merge and ``V0 = 0.441 V`` after one with nothing in the
plot, the details pane or the point table recording which.

So four surfaces are asserted here:

* the **banner** above the plot -- present exactly when the pool is not comparable,
  naming the mismatched groups and what to do about it;
* the **plot title** -- the merge state always, the baseline fingerprint whenever
  the metric is measured against the baseline. This is the one that survives *Save
  image*, which is how these numbers left the app in the first place;
* the **per-recording table** -- a Baseline and a Merge column beside each value;
* the **details pane** -- a "Pooled provenance" block.

And the negative: for a rate or a dwell, and — for the BASELINE half — within one
fingerprint, the window must stay quiet. A check that always fires is not a check.

ONE OF THOSE NEGATIVES HAS SINCE MOVED, deliberately.
``test_window_within_one_fingerprint_is_quiet_except_about_the_gain`` used to assert
that ``Median depth (pA)`` pooled inside one baseline configuration showed NO banner at
all. Every recording in ``_ladder`` carries the identity calibration ``(1.0, 0.0)``, so
that picture was three uncalibrated pores' raw picoamp scales on one axis, fitted, in
silence — the M3 defect, which is now fixed at :func:`nanoscope.core.sources.
pool_comparability`. The baseline half of that test is unchanged; the gain half is now
asserted present as tightly as it used to be asserted absent.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.voltage_dep_panel import VDepWindow

FP_500 = "sha256:b64a75ebdf29"
FP_600 = "sha256:ce8ae31f10bd"
FP_700 = "sha256:1a29aabf5fb1"

RAW_RATES = {-500.0: 4.783, -600.0: 4.610, -700.0: 6.497}
MERGED_RATES = {-500.0: 3.451, -600.0: 3.992, -700.0: 5.433}

_THREE_FINGERPRINTS = {-500.0: FP_500, -600.0: FP_600, -700.0: FP_700}
_ONE_FINGERPRINT = {-500.0: FP_600, -600.0: FP_600, -700.0: FP_600}


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, *, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _constant_df(*, n: int, dwell_s: float, depth_pA: float, frac_depth: float) -> pd.DataFrame:
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": float(dwell_s),
            "baseline_pA": -30.0, "depth_pA": float(depth_pA),
            "frac_depth": float(frac_depth), "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _install(session, doc_id, df, *, duration_s, fingerprint, merged_window_s=None) -> None:
    prov: dict = {"analyzed_duration_s": float(duration_s), "recording": "rec"}
    if fingerprint is not None:
        prov["baseline_fingerprint"] = fingerprint
        prov["baseline_method"] = "masked_median"
    if merged_window_s is not None:
        prov["merged"] = {"rule": "gap", "window_s": float(merged_window_s),
                          "n_before": len(df) + 1, "n_after": len(df)}
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=EventSet(df, prov)))


def _ladder(tmp_path, fingerprints, *, rates=None, merged_window_s=None) -> Session:
    session = Session()
    rates = rates or RAW_RATES
    for mV, fp in fingerprints.items():
        doc_id = session.add_document(_doc(tmp_path, nominal_mV=mV, name=f"apo_{int(-mV)}.dat"))
        v = mV / 1000.0
        _install(session, doc_id,
                 _constant_df(n=40, dwell_s=(v + 1.2) * 1e-3, depth_pA=-100.0 * v,
                              frac_depth=-0.1 * v),
                 duration_s=40.0 / rates[mV], fingerprint=fp,
                 merged_window_s=merged_window_s)
    return session


def _window(qtbot, session, *, metric: str, fit: str = "none") -> VDepWindow:
    w = VDepWindow(session)
    qtbot.addWidget(w)
    w.set_metric(metric)
    w.set_fit(fit)
    return w


def _title(w: VDepWindow) -> str:
    label = w.plot_item().titleLabel
    return "" if label is None else str(label.text)


def _table_column(w: VDepWindow, header: str) -> list[str]:
    head = w._table.horizontalHeader()
    cols = [w._table.horizontalHeaderItem(c).text() for c in range(head.count())]
    c = cols.index(header)
    return [w._table.item(r, c).text() for r in range(w._table.rowCount())]


# ================================================================================================
# 1. High-1 -- the banner
# ================================================================================================

@pytest.mark.parametrize("metric", ["median_depth", "median_frac_depth"])
def test_window_warns_when_an_amplitude_is_pooled_across_baselines(qapp, qtbot, tmp_path, metric):
    """THE High-1 REGRESSION AT THE SURFACE. Screenshots
    ``cross-voltage/screenshots/11_vdep_median_depth.png`` and
    ``…_median_frac_depth.png`` show this plot with nothing on it but the fit."""
    w = _window(qtbot, _ladder(tmp_path, _THREE_FINGERPRINTS), metric=metric, fit="linear")
    banner = w.warning_text()
    assert banner, "the window pooled three baselines and showed no warning"
    for fp in (FP_500, FP_600, FP_700):
        assert fp in banner
    assert "Re-detect" in banner
    # Full caveats remain in Details and outside-canvas warnings, not plot titles.
    assert "NOT COMPARABLE" in w.details_text()
    assert "MIXED baselines" not in _title(w)
    assert "⚠" in w._headline.text()


@pytest.mark.parametrize("metric", ["median_depth", "median_frac_depth"])
def test_window_within_one_fingerprint_is_quiet_except_about_the_gain(
        qapp, qtbot, tmp_path, metric):
    """A comparison WITHIN one baseline configuration must not raise the BASELINE
    banner — and must still raise the GAIN one over an absolute pA.

    THE SURFACE HALF OF THE SAME UPDATE as
    ``tests/analysis/test_voltage_dep_comparability.py::
    test_pooling_an_amplitude_within_one_baseline_warns_only_where_the_gain_bites``.
    This test asserted ``warning_text() == ""`` for both metrics, which is the picture
    the owner was actually shown: ``Median depth (pA)`` against voltage over three
    UNCALIBRATED pores, fitted, with an empty banner. Every recording in ``_ladder``
    carries the identity calibration ``(1.0, 0.0)`` — nobody has calibrated anything, so
    those picoamps are three different raw scales on one axis.

    The banner is empty for the ratio and carries the gain caution for the absolute
    current. Full Details retains that caution, while plots carry ordinary method
    and result labels rather than automatic warning prose.
    """
    w = _window(qtbot, _ladder(tmp_path, _ONE_FINGERPRINT), metric=metric, fit="linear")

    # The BASELINE half stays quiet on every surface: one fingerprint, one group.
    assert "NOT COMPARABLE" not in w.details_text()
    assert FP_600 in _title(w)              # still SAYS which baseline it was

    if metric == "median_frac_depth":
        # A ratio divides the gain out. Nothing to say, anywhere.
        assert w.warning_text() == ""
        assert "NOT COMPARABLE" not in _title(w)
        assert "gain uncalibrated" not in _title(w)
        assert "⚠" not in w._headline.text()
        return

    banner = w.warning_text()
    assert banner.startswith("CAUTION — comparing median depth across recordings:"), banner
    assert "UNCALIBRATED" in banner
    assert "median dwell" in banner and "median dI/I0" in banner
    # Details retains the warning for reproducible data and figure-bundle exports.
    assert "CAUTION" in w.details_text()
    assert "gain uncalibrated" not in _title(w)
    assert "⚠" in w._headline.text()
    # The fit is still drawn: the verdict is advisory, never a refusal.
    assert w.fit() is not None


@pytest.mark.parametrize("metric", ["capture_rate_hz", "median_dwell_s", "n_events"])
def test_rate_and_dwell_are_quiet_across_three_baselines(qapp, qtbot, tmp_path, metric):
    """Per metric, not blanket -- the panel is not disabled, it is qualified."""
    w = _window(qtbot, _ladder(tmp_path, _THREE_FINGERPRINTS), metric=metric, fit="exponential")
    assert w.warning_text() == ""
    assert "MIXED baselines" not in _title(w)
    assert w.fit() is not None                    # the trend is still drawn and fitted


def test_switching_metric_raises_and_clears_the_banner(qapp, qtbot, tmp_path):
    """The banner tracks the CURRENT metric, so it cannot be left stale on screen."""
    w = _window(qtbot, _ladder(tmp_path, _THREE_FINGERPRINTS), metric="capture_rate_hz")
    assert w.warning_text() == ""
    w.set_metric("median_frac_depth")
    assert "NOT COMPARABLE" in w.warning_text()
    w.set_metric("median_dwell_s")
    assert w.warning_text() == ""


# ================================================================================================
# 2. High-2 -- the merge state reaches the plot, the table and the details
# ================================================================================================

def test_the_headline_v0_moves_with_the_merge_and_the_window_now_records_which(
        qapp, qtbot, tmp_path):
    """THE High-2 REGRESSION, on the run's own numbers.

    Same three recordings, same panel: V0 = 0.653 V raw, 0.441 V after a 10 ms
    merge. Before the fix the two windows were indistinguishable apart from the
    number itself -- so a screenshot of either was unattributable."""
    raw = _window(qtbot, _ladder(tmp_path, _ONE_FINGERPRINT, rates=RAW_RATES),
                  metric="capture_rate_hz", fit="exponential")
    raw.set_abs_voltage(True)
    merged = _window(qtbot, _ladder(tmp_path, _ONE_FINGERPRINT, rates=MERGED_RATES,
                                    merged_window_s=0.010),
                     metric="capture_rate_hz", fit="exponential")
    merged.set_abs_voltage(True)

    assert raw.fit()["params"]["V0"] == pytest.approx(0.6530, abs=5e-4)
    assert merged.fit()["params"]["V0"] == pytest.approx(0.4407, abs=5e-4)

    # The two headline numbers differ by 33 %; the two renderings now differ too.
    assert "unmerged" in _title(raw)
    assert "merged: gap 10 ms" in _title(merged)
    assert _title(raw) != _title(merged)
    assert "unmerged" in raw.details_text()
    assert "merged: gap 10 ms" in merged.details_text()
    assert set(_table_column(raw, "Merge")) == {"unmerged"}
    assert set(_table_column(merged, "Merge")) == {"merged: gap 10 ms"}


def test_a_merged_set_pooled_with_an_unmerged_one_warns(qapp, qtbot, tmp_path):
    session = Session()
    for mV, window in ((-500.0, 0.010), (-600.0, None), (-700.0, 0.010)):
        doc_id = session.add_document(_doc(tmp_path, nominal_mV=mV, name=f"apo_{int(-mV)}.dat"))
        v = mV / 1000.0
        _install(session, doc_id,
                 _constant_df(n=40, dwell_s=(v + 1.2) * 1e-3, depth_pA=-100.0 * v,
                              frac_depth=-0.1 * v),
                 duration_s=10.0, fingerprint=FP_600, merged_window_s=window)
    w = _window(qtbot, session, metric="capture_rate_hz", fit="exponential")
    assert "merge state" in w.warning_text()
    assert "MIXED merge states" not in _title(w)
    assert w.fit() is not None                    # still fitted, never withheld


# ================================================================================================
# 3. the table and the details pane carry it on every metric
# ================================================================================================

def test_the_point_table_carries_a_baseline_and_a_merge_column(qapp, qtbot, tmp_path):
    w = _window(qtbot, _ladder(tmp_path, _THREE_FINGERPRINTS), metric="median_depth")
    assert _table_column(w, "Recording") == ["apo_700.dat", "apo_600.dat", "apo_500.dat"]
    assert _table_column(w, "Baseline") == [FP_700, FP_600, FP_500]
    assert _table_column(w, "Merge") == ["unmerged"] * 3


def test_the_details_pane_carries_a_pooled_provenance_block(qapp, qtbot, tmp_path):
    w = _window(qtbot, _ladder(tmp_path, _ONE_FINGERPRINT), metric="median_depth",
                fit="linear")
    text = w.details_text()
    assert "Pooled provenance" in text
    assert FP_600 in text
    assert "unmerged" in text


def test_provenance_is_shown_even_with_no_fit(qapp, qtbot, tmp_path):
    """A bare scatter is exported too, so it carries the same statement."""
    w = _window(qtbot, _ladder(tmp_path, _ONE_FINGERPRINT), metric="median_frac_depth")
    assert w.fit() is None
    assert FP_600 in _title(w)
    assert "Pooled provenance" in w.details_text()


def test_the_window_never_calls_vdep_points_directly(qapp, qtbot, tmp_path, monkeypatch):
    """The structural half of the fix: the window gets its points from
    ``vdep_series``, which cannot hand them over without the verdict. A future edit
    that reaches for ``vdep_points`` here loses the warning again, silently."""
    import ast
    import inspect

    from nanoscope.gui import voltage_dep_panel

    tree = ast.parse(inspect.getsource(voltage_dep_panel))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "vdep_points" not in called, (
        "voltage_dep_panel calls vdep_points directly; the comparability verdict "
        "only arrives with vdep_series")
    assert "vdep_series" in called

    # ... and it really is the call that runs, not merely the one written down.
    calls: list[str] = []
    real_series = voltage_dep.vdep_series
    monkeypatch.setattr(
        voltage_dep, "vdep_series",
        lambda *a, **k: (calls.append("series"), real_series(*a, **k))[1])
    w = VDepWindow(_ladder(tmp_path, _THREE_FINGERPRINTS))
    qtbot.addWidget(w)
    w.set_metric("median_depth")
    assert calls, "VDepWindow did not gather through vdep_series"


# ================================================================================================
# 4. the empty state
# ================================================================================================

def test_no_banner_on_an_empty_session(qapp, qtbot):
    w = VDepWindow(Session())
    qtbot.addWidget(w)
    assert w.warning_text() == ""
    assert w.warnings() == []
