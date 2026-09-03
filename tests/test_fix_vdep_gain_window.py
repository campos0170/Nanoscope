"""The voltage-dependence WINDOW says the gain caveat, and the per-recording ones.

REGRESSION FOR apo-run-2026-08-22 ``cross/README.md`` §0 [NEW, Medium] / §7.A (M3),
the L4 unit item beside it, and the M12 item, at the surface the owner was actually
looking at. ``tests/test_fix_vdep_gain_rule.py`` pins the verdict in the pure core;
this file pins that it REACHES A HUMAN, which is the half that failed before.

Measured through the real GUI on the three apoferritin recordings, all re-detected
under ONE common baseline window -- so the baseline and merge halves are both
satisfied and only the GAIN is left to object:

    warning_text()  ''                (nothing at all)
    headline        'V₀ = -0.432 V'
    plot title      'V₀ = -0.432 V   ·   R² = 0.9157   ·   merged: gap 10 ms
                     ·   baseline sha256:ce8ae31f10bd'
    y axis          'Median depth'    (no unit, for a column of picoamps)
    table headers   Recording · V (mV) · Median depth · Baseline · Merge

while ``Analysis ▸ Plots`` on ``depth_pA`` over the identical three recordings, in
the same session, printed the gain CAUTION.

Four surfaces are asserted here, the same four the comparability work already
asserts for the baseline and merge halves: the banner, the plot TITLE (what a saved
PNG keeps), the per-recording table, and the details pane. Plus the negative --
switch to a ratio or a time and the window goes quiet again, because a check that
always fires is not a check.

Reverted by ``make_revert.py`` in the M3 scratch.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.voltage_dep_panel import VDepWindow

PKG = pathlib.Path(nanoscope.__file__).resolve().parent
WHERE = f"(nanoscope resolved from {PKG})"

FP_600 = "sha256:ce8ae31f10bd"
FP_500 = "sha256:b64a75ebdf29"
FP_700 = "sha256:1a29aabf5fb1"

# The -700 mV recording's own detection caveat from the run, verbatim -- the one
# that reached the Summary panel and the Plots details pane and no further.
CAVEAT_700 = (
    "the tracked baseline sits 4.20 pA INSIDE the states, measured against the open "
    "pore 5-55 ms either side of each of 982 of them. That is 4.2% of their own 98.4 "
    "pA excursion, and it comes off every depth this baseline measures: depth_pA and "
    "frac_depth read about 4% low, and a voltage series measured this way puts one "
    "recording out of line with the rest.")

# The three points the real window plotted under the common window (median depth_pA
# of the merged, curated sets).
DEPTHS = {-500.0: 62.81626510620117, -600.0: 89.41650390625, -700.0: 99.79248046875}


# ================================================================================================
# builders -- a Session shaped like the run
# ================================================================================================

def _doc(tmp_path, *, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _df(*, n: int, dwell_s: float, depth_pA: float, frac_depth: float) -> pd.DataFrame:
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


def _ladder(tmp_path, fingerprints, *, gains=None, warnings=None) -> Session:
    """Three recordings, one detection each; ``gains``/``warnings`` keyed by mV."""
    session = Session()
    for mV, fp in fingerprints.items():
        doc_id = session.add_document(_doc(tmp_path, nominal_mV=mV, name=f"apo_{int(-mV)}.dat"))
        gain = (gains or {}).get(mV)
        if gain is not None:
            # write_sidecar=False: a harness must never write a .npcal.json beside
            # the owner's raw data.
            session.set_calibration(doc_id, 0,
                                    ChannelCalibration(gain=gain[0], offset=gain[1]),
                                    write_sidecar=False)
        prov = {"analyzed_duration_s": 10.0, "recording": "rec",
                "baseline_fingerprint": fp, "baseline_method": "masked_median"}
        session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={}, calibration_fingerprint=(1.0, 0.0),
            payload=EventSet(_df(n=40, dwell_s=0.02, depth_pA=DEPTHS[mV],
                                 frac_depth=2.0 - mV / -1000.0), prov),
            warnings=tuple((warnings or {}).get(mV, ()))))
    return session


ONE_WINDOW = {-500.0: FP_600, -600.0: FP_600, -700.0: FP_600}
THREE_WINDOWS = {-500.0: FP_500, -600.0: FP_600, -700.0: FP_700}


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
    assert header in cols, f"no {header!r} column in {cols} {WHERE}"
    c = cols.index(header)
    return [w._table.item(r, c).text() for r in range(w._table.rowCount())]


# ================================================================================================
# 1. M3 -- the gain caveat reaches the window it qualifies
# ================================================================================================

def test_median_depth_over_uncalibrated_pores_is_no_longer_silent(qapp, qtbot, tmp_path):
    """THE M3 REGRESSION AT THE SURFACE. Three recordings, ONE baseline window, so
    nothing else objects -- and the fit is on absolute pA across three raw scales."""
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_depth", fit="exponential")
    banner = w.warning_text()
    assert banner, f"the window fitted absolute pA over 3 uncalibrated pores in silence {WHERE}"
    assert "UNCALIBRATED" in banner, f"{banner} {WHERE}"
    assert "median depth" in banner, WHERE
    # ... on every surface the baseline verdict already reaches.
    assert "UNCALIBRATED" in w.details_text(), WHERE
    assert "gain uncalibrated" in _title(w), f"{_title(w)} {WHERE}"
    assert "⚠" in w._headline.text(), WHERE
    # and the fit is still drawn: qualified, never withheld.
    assert w.fit() is not None, WHERE


def test_the_saved_title_carries_the_gain(qapp, qtbot, tmp_path):
    """The title is what leaves the app. ``V₀ = -0.432 V`` left it saying nothing
    about the units it was fitted in."""
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_depth", fit="exponential")
    out = tmp_path / "vdep.png"
    assert w.save_image(str(out)) is not None, WHERE
    assert out.is_file() and out.stat().st_size > 0, WHERE
    title = _title(w)
    assert "gain uncalibrated" in title and "baseline " in title, f"{title} {WHERE}"


@pytest.mark.parametrize("metric", ["median_frac_depth", "median_dwell_s",
                                    "capture_rate_hz", "n_events"])
def test_a_gain_free_metric_stays_quiet_across_the_same_three_pores(
        qapp, qtbot, tmp_path, metric):
    """PER METRIC. dI/I₀ is a ratio and a dwell is a time: the gain divides out."""
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric=metric, fit="linear")
    assert w.warning_text() == "", f"{w.warning_text()} {WHERE}"
    assert "gain" not in _title(w), f"{_title(w)} {WHERE}"
    assert "⚠" not in w._headline.text(), WHERE


def test_switching_metric_raises_and_clears_the_gain_banner(qapp, qtbot, tmp_path):
    """The banner tracks the CURRENT metric, so it cannot be left stale on screen."""
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_frac_depth")
    assert w.warning_text() == "", WHERE
    w.set_metric("median_depth")
    assert "UNCALIBRATED" in w.warning_text(), WHERE
    w.set_metric("median_dwell_s")
    assert w.warning_text() == "", WHERE


def test_one_shared_real_calibration_silences_it(qapp, qtbot, tmp_path):
    """Not 'always complain': calibrate all three the same way and the depth pools."""
    gains = {mV: (0.1, 0.0) for mV in ONE_WINDOW}
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW, gains=gains),
                metric="median_depth", fit="exponential")
    assert w.warning_text() == "", f"{w.warning_text()} {WHERE}"
    assert "gain 0.1, offset 0" in _title(w), f"{_title(w)} {WHERE}"
    assert set(_table_column(w, "Gain")) == {"gain 0.1, offset 0"}, WHERE


def test_two_different_calibrations_are_named_per_recording(qapp, qtbot, tmp_path):
    gains = {-500.0: (0.1, 0.0), -600.0: (0.2, 0.0), -700.0: (0.1, 0.0)}
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW, gains=gains), metric="median_depth")
    assert "do NOT share a current calibration" in w.warning_text(), WHERE
    assert "MIXED gains (2)" in _title(w), f"{_title(w)} {WHERE}"
    assert _table_column(w, "Gain") == [
        "gain 0.1, offset 0", "gain 0.2, offset 0", "gain 0.1, offset 0"], WHERE


def test_the_table_carries_the_gain_beside_the_number_it_produced(qapp, qtbot, tmp_path):
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_depth")
    assert _table_column(w, "Recording") == ["apo_700.dat", "apo_600.dat", "apo_500.dat"], WHERE
    assert _table_column(w, "Gain") == ["uncalibrated"] * 3, WHERE
    # the pre-existing columns are untouched
    assert _table_column(w, "Baseline") == [FP_600] * 3, WHERE
    assert _table_column(w, "Merge") == ["unmerged"] * 3, WHERE


def test_the_details_pane_names_the_pooled_gain(qapp, qtbot, tmp_path):
    text = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_depth").details_text()
    assert "Pooled provenance" in text, WHERE
    assert "Gain:" in text, WHERE
    assert "uncalibrated" in text, WHERE


def test_the_gain_block_says_when_the_metric_does_not_read_it(qapp, qtbot, tmp_path):
    """Reported either way, held against the metric only when it applies -- exactly
    how the Baseline block already behaves for a rate."""
    text = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="capture_rate_hz").details_text()
    gain_line = [ln for ln in text.splitlines() if ln.strip().startswith("Gain:")]
    assert gain_line and "[not read by this metric]" in gain_line[0], f"{text} {WHERE}"


# ================================================================================================
# 2. L4 -- the one metric with no unit anywhere
# ================================================================================================

def test_median_depth_carries_its_unit_on_every_surface(qapp, qtbot, tmp_path):
    """It was ``Median depth`` in the combo, on the axis and in the table header, for
    a column of picoamps, on the window whose job is pooling several pores' pA."""
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_depth", fit="linear")
    assert w._metric_combo.currentText() == "Median depth (pA)", WHERE
    assert "Median depth (pA)" in w.plot_item().getAxis("left").labelString(), WHERE
    assert "Median depth (pA)" in [w._table.horizontalHeaderItem(c).text()
                                   for c in range(w._table.columnCount())], WHERE
    assert "Median depth (pA)" in w.details_text(), WHERE


def test_the_other_metrics_keep_the_units_they_had(qapp, qtbot):
    w = VDepWindow(Session())
    qtbot.addWidget(w)
    labels = [w._metric_combo.itemText(i) for i in range(w._metric_combo.count())]
    assert labels == ["Capture rate (Hz)", "Median dwell (s)", "Median depth (pA)",
                      "Median dI/I₀", "Event count"], f"{labels} {WHERE}"


# ================================================================================================
# 3. M12 -- the per-recording detection caveats reach the window that fits them
# ================================================================================================

def test_a_detection_caveat_reaches_the_voltage_window(qapp, qtbot, tmp_path):
    """THE M12 REGRESSION. The -700 mV recording's 4.2 % baseline bias reached the
    Summary panel and the Plots details pane; the window that turns that recording
    into one term of a fitted law never mentioned it."""
    session = _ladder(tmp_path, ONE_WINDOW, warnings={-700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_dwell_s", fit="linear")
    banner = w.warning_text()
    assert "DETECTION CAVEATS" in banner, f"{banner!r} {WHERE}"
    assert "apo_700.dat" in banner, WHERE
    assert "out of line with the rest" in banner, WHERE
    assert "DETECTION CAVEATS" in w.details_text(), WHERE
    assert "1 detection caveat(s)" in _title(w), f"{_title(w)} {WHERE}"


def test_a_caveat_is_not_dressed_up_as_an_incomparable_pool(qapp, qtbot, tmp_path):
    """Different claims: one point is biased, versus these must not be pooled."""
    session = _ladder(tmp_path, ONE_WINDOW, warnings={-700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_dwell_s", fit="linear")
    assert w.warnings() == [], WHERE                    # the comparability verdict
    assert "NOT COMPARABLE" not in w.warning_text(), WHERE
    assert "⚠" not in w._headline.text(), WHERE
    assert w.series().comparable is True, WHERE


def test_a_clean_ladder_shows_no_caveat_block(qapp, qtbot, tmp_path):
    w = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_dwell_s")
    assert w.warning_text() == "", WHERE
    assert "DETECTION CAVEATS" not in w.details_text(), WHERE
    assert "detection caveat" not in _title(w), WHERE


def test_both_kinds_of_honesty_can_be_on_screen_at_once(qapp, qtbot, tmp_path):
    """Three windows AND a biased -700: the pool warning and the caveat block are
    two different statements and neither may overwrite the other."""
    session = _ladder(tmp_path, THREE_WINDOWS, warnings={-700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_depth", fit="exponential")
    banner = w.warning_text()
    assert "NOT COMPARABLE" in banner, WHERE           # baseline half
    assert "UNCALIBRATED" in banner, WHERE             # gain half
    assert "DETECTION CAVEATS" in banner, WHERE        # per-recording half
    # ... and the gain advice does NOT send the reader to dI/I0 while the baselines
    # disagree, on this surface as on Plots.
    caution = [w_ for w_ in w.warnings() if w_.startswith("CAUTION")]
    assert len(caution) == 1 and "dI/I0" not in caution[0], f"{caution} {WHERE}"


# ================================================================================================
# 4. the two defects the fix above introduced, found by looking at the rendered window
# ================================================================================================

def test_the_banner_cannot_starve_the_plot_it_is_warning_about(qapp, qtbot, tmp_path):
    """FOUND IN MY OWN FIX. The banner's length is bounded by nothing: a line per
    recording per disagreeing group, plus a PARAGRAPH per detection caveat. Measured
    at 880x700 on the owner's three recordings as probed -- three baselines, an
    uncalibrated gain, two caveats -- the free-height banner took 290 px, squeezed
    the plot to 280 and pushed the table and details pane off the bottom."""
    from nanoscope.gui.voltage_dep_panel import _WARN_MAX_H

    session = _ladder(tmp_path, THREE_WINDOWS,
                      warnings={-600.0: ("x " * 120,), -700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_depth", fit="exponential")
    w.resize(880, 700)
    w.show()
    qtbot.waitExposed(w)

    assert w.warning_text(), WHERE
    assert w._warning_scroll.isVisible(), WHERE
    assert w._warning_scroll.height() <= _WARN_MAX_H, (
        f"banner grew to {w._warning_scroll.height()} px {WHERE}")
    assert w._plot_widget.height() >= 320, (
        f"the plot was squeezed to {w._plot_widget.height()} px {WHERE}")
    assert w._table.height() > 0 and w._details.height() > 0, (
        f"table {w._table.height()} px, details {w._details.height()} px {WHERE}")
    # nothing is HIDDEN by the cap -- the whole text is still in the label and in
    # the details pane, one flick of the wheel or one glance away.
    assert CAVEAT_700 in w._warning_label.text(), WHERE
    assert CAVEAT_700 in w.details_text(), WHERE


def test_the_title_wraps_instead_of_dropping_its_last_segment(qapp, qtbot, tmp_path):
    """FOUND IN MY OWN FIX. The title is what the exported PNG keeps. Adding the
    gain and the caveat count took the as-probed title from 89 characters to 133,
    and pyqtgraph clips at about 95 at this window's default width -- so the PNG
    silently lost the end of the statement the title exists to preserve."""
    from nanoscope.gui.voltage_dep_panel import _TITLE_WRAP

    session = _ladder(tmp_path, THREE_WINDOWS, warnings={-700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_depth", fit="exponential")
    title = _title(w)
    lines = title.split("<br>")
    assert len(lines) > 1, f"the title did not wrap: {title!r} {WHERE}"
    for line in lines:
        assert len(line) <= _TITLE_WRAP + 32, f"{line!r} ({len(line)}) {WHERE}"
    # every segment survived the wrap -- wrapped, not shortened
    for segment in ("V₀ = ", "R² = ", "NOT COMPARABLE", "unmerged",
                    "MIXED baselines (3)", "gain uncalibrated", "1 detection caveat(s)"):
        assert segment in title, f"{segment!r} lost from {title!r} {WHERE}"
    # ... and a title with nothing to qualify stays on one line.
    clean = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_dwell_s",
                    fit="linear")
    assert "<br>" not in _title(clean), f"{_title(clean)!r} {WHERE}"


def test_a_wrapped_title_is_given_the_room_to_be_read(qapp, qtbot, tmp_path):
    """FOUND IN MY OWN FIX, TWICE OVER. ``PlotItem.setTitle`` hard-codes a 30 px
    title row, so wrapping the title to stop it clipping at the RIGHT edge made it
    clip at the TOP instead: the 3-line as-probed title lost its whole first line,
    V₀ and R² with it."""
    session = _ladder(tmp_path, THREE_WINDOWS, warnings={-700.0: (CAVEAT_700,)})
    w = _window(qtbot, session, metric="median_depth", fit="exponential")
    w.resize(880, 700)
    w.show()
    qtbot.waitExposed(w)

    lines = _title(w).count("<br>") + 1
    assert lines >= 3, f"{_title(w)!r} {WHERE}"
    row_h = w.plot_item().layout.rowMaximumHeight(0)
    text_h = w.plot_item().titleLabel.item.boundingRect().height()
    assert row_h >= text_h, (
        f"the title row is {row_h} px for {text_h} px of text ({lines} lines) {WHERE}")
    assert row_h > 30.0, f"still the hard-coded 30 px {WHERE}"

    # a one-line title must NOT hog the extra room
    clean = _window(qtbot, _ladder(tmp_path, ONE_WINDOW), metric="median_dwell_s",
                    fit="linear")
    assert clean.plot_item().layout.rowMaximumHeight(0) == 30.0, (
        f"a one-line title should keep pyqtgraph's own row height {WHERE}")


def test_the_exported_png_really_keeps_the_extra_title_lines(qapp, qtbot, tmp_path):
    """Measured in the PNG, not in the widget: the title is only worth wrapping if
    the wrap survives *Save image…*, which is how these numbers leave the app."""
    from PySide6 import QtGui

    def ink_in_top(path, rows=46):
        """Non-background pixels in the exported image's TITLE BAND.

        46 rows: the title plus a little of the plot frame below it, so the second
        and third title lines land inside the band and nothing else changes between
        the two exports.
        """
        img = QtGui.QImage(str(path))
        assert img.width() > 0, f"unreadable PNG {path} {WHERE}"
        return sum(1 for y in range(min(rows, img.height()))
                   for x in range(0, img.width(), 2)
                   if (img.pixel(x, y) & 0xFFFFFF) not in (0x000000, 0x0D0F12, 0xFFFFFF))

    wrapped = _window(qtbot, _ladder(tmp_path, THREE_WINDOWS,
                                     warnings={-700.0: (CAVEAT_700,)}),
                      metric="median_depth", fit="exponential")
    plain = _window(qtbot, _ladder(tmp_path, ONE_WINDOW),
                    metric="median_dwell_s", fit="linear")
    a, b = tmp_path / "wrapped.png", tmp_path / "plain.png"
    wrapped.save_image(str(a))
    plain.save_image(str(b))
    wrapped_ink, plain_ink = ink_in_top(a), ink_in_top(b)
    assert wrapped_ink > plain_ink * 1.2, (
        f"the exported title band holds {wrapped_ink} px against {plain_ink} for a "
        f"one-line title — the extra lines did not reach the PNG {WHERE}")


def test_the_empty_state_says_nothing(qapp, qtbot):
    w = VDepWindow(Session())
    qtbot.addWidget(w)
    assert w.warning_text() == "", WHERE
    assert w.warnings() == [], WHERE
