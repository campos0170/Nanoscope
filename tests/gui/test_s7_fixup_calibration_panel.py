"""Sweep-7 fix-up: what the skeptics found wrong with the Calibration panel's own fixes.

Four defects, one panel (docs/sweep7/FINDINGS.md, the sweep-7 fix batch):

* D1 / X-SURF-10 -- the partner boxes were re-derived only on a RESELECT. Wave 1 moved the
  Open-pore-conductance / Command-voltage partner lookup onto ``Session.effective_kind``,
  but only in ``_load_samples``, reached from ``set_selection`` alone; the real Apply
  reaches ``_on_calibration_changed`` -> ``_populate``, which showed the boxes from the
  stale partners. Measured through the real combo + Apply on the undeclared CSV: after
  "treat as current" the title read "current (file says unknown)" and the conductance box
  stayed hidden; after re-declaring the current channel a voltage the box kept printing
  "I0 = -122.8 mV" from the channel just declared a voltage, until a click away and back;
* D4 -- the combo's tooltip still said the override picks a pane, after sweep 7 made it
  the one resolver the whole pipeline follows (Run detection, Batch detect, PSD, I-V, the
  chip and the workflow strip: "treat as voltage" on IN15 greys out all four);
* X-SURF-5 -- the float32-overflow refusal was written into the un-wrapped formula line
  and pushed the panel's minimumSizeHint().width() from 402 px to 1,608 px (house rule
  1,100), so in the Inspector dock the remedy at the end of the sentence was off-screen;
* X-SURF-6 -- on a channel carrying an infinity the min / max / span rows read "--" while
  the coverage line and every tooltip still claimed "exact over all 5,000 samples", and
  the median row printed "nan".

Every test drives the real panel: the real combo, key clicks into the real fields, the
real Apply button, or the session door the panel's own commit ends in.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel

FS = 50_000.0
HOUSE_WIDTH_PX = 1100


def _doc(path, channels, arrays) -> TraceDocument:
    n = int(arrays[0].size)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    return TraceDocument(meta=meta, channels=tuple(arrays))


def _undeclared(tmp_path) -> TraceDocument:
    """Two channels the file does not declare -- the shape the override was built for.
    ch0 is a -700 mV open pore in pA (raw median -49.13), ch1 its command voltage."""
    cur = np.full(40_000, -49.13, dtype=np.float32)
    volt = np.full(40_000, -700.0, dtype=np.float32)
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)),
        (cur, volt))


@pytest.fixture
def panel(qapp, qtbot):
    s = Session()
    p = CalibrationPanel(s)
    qtbot.addWidget(p)
    p.show()
    return p


def _type(qtbot, field, text: str) -> None:
    field.clear()
    qtbot.keyClicks(field, text)


def _apply_kind(qtbot, panel, kind, units: str, gain: str) -> str:
    """The user's gesture: pick the kind in the combo, type the units and the gain, press
    the real Apply button. Returns the status line."""
    panel._treat_as.setCurrentIndex(panel._treat_as.findData(kind))
    _type(qtbot, panel._units, units)
    _type(qtbot, panel._gain, gain)
    panel._btn_apply.click()
    return panel.status_text()


# ================================================================================================
# D1 / X-SURF-10: the partner boxes follow the Apply that changed the kind, not a reselect
# ================================================================================================

def test_the_partner_boxes_follow_the_apply_that_changed_the_kind(panel, qtbot, tmp_path):
    s = panel._session
    did = s.add_document(_undeclared(tmp_path))
    panel.set_selection(did, 0)
    assert not panel._cond_box.isVisible() and not panel._recon_box.isVisible()

    # (a) "Treat this channel as: current" + pA + x2.5 on the SELECTED channel.
    status = _apply_kind(qtbot, panel, ChannelKind.CURRENT, "pA", "2.5")
    assert status.startswith("Applied"), status
    assert "current (file says unknown)" in panel._title.text()
    assert panel._cur_ch == 0, "the current partner is resolved on the Apply itself"
    assert panel._cond_box.isVisible(), "the conductance box appears on the same repaint"
    assert panel._cond_iopen.text() == "-122.8 pA", panel._cond_iopen.text()

    # (a') the sibling declared a voltage, on ITS Apply.
    panel.set_selection(did, 1)
    _apply_kind(qtbot, panel, ChannelKind.VOLTAGE, "mV", "1")
    assert panel._volt_ch == 1
    assert panel._recon_box.isVisible(), "the command-voltage check appears on the Apply"
    assert panel._recon_effective.text() == "-700 mV"
    assert panel._cond_voltage.text() == "-700 mV"
    assert panel._cond_g.text() == "0.1755 nS", "G = -122.825 pA / -700 mV"

    # (b) the reverse: the current channel re-declared a VOLTAGE. There is no current
    # channel left, so the conductance box must go -- not keep printing I0 / G from the
    # channel the user has just said is a command voltage.
    panel.set_selection(did, 0)
    assert panel._cond_box.isVisible() and panel._recon_box.isVisible()
    _apply_kind(qtbot, panel, ChannelKind.VOLTAGE, "mV", "2.5")
    assert s.resolve_current_channel(did, 0) is None
    assert panel._cur_ch is None, "no current partner once ch0 is a voltage"
    assert not panel._cond_box.isVisible(), "the box is gone on the same Apply"
    assert panel._cond_iopen.text() == "—", "and holds no stale current"
    assert panel._cond_g.text() == "—"
    assert panel._volt_ch == 0, "the first voltage in channel order is now ch0 itself"
    assert panel._recon_box.isVisible()


def test_a_sibling_declared_a_partner_shows_its_box_without_a_reselect_or_losing_the_edit(
        panel, qtbot, tmp_path):
    """The other door: a calibration of the SELECTED recording changes on a channel that
    is not the selected one (a project restore, a batch, another surface). The partner
    set can change on that too, so the boxes must follow it -- while the values the user
    is typing on the selected channel survive."""
    s = panel._session
    did = s.add_document(_undeclared(tmp_path))
    s.set_calibration(did, 0, ChannelCalibration(2.5, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    panel.set_selection(did, 0)
    assert panel._cond_box.isVisible() and not panel._recon_box.isVisible()
    assert panel._cond_voltage.text() == "—", "no voltage known yet"

    _type(qtbot, panel._gain, "3")                       # mid-edit, not committed
    s.set_calibration(did, 1, ChannelCalibration(1.0, 0.0, "mV", kind=ChannelKind.VOLTAGE),
                      write_sidecar=False)

    assert panel._volt_ch == 1
    assert panel._recon_box.isVisible(), "the command-voltage check appears at once"
    assert panel._gain.text() == "3", "the user's uncommitted edit is untouched"
    assert panel._cond_voltage.text() == "-700 mV"
    assert panel._cond_iopen.text() == "-147.4 pA", "I0 tracks the live edit: 3 x -49.13"
    assert panel._cond_g.text() == "0.2106 nS"


# ================================================================================================
# D4: the combo's tooltip says what the override reaches
# ================================================================================================

def test_the_treat_as_tooltip_says_the_override_decides_what_is_analysed(panel):
    tip = panel._treat_as.toolTip()
    low = tip.lower()
    assert "pane" in low, "the pane half is still true"
    assert "run detection" in low and "batch detect" in low, tip
    assert "analys" in low, "and it says the analysis follows it"
    assert "as the file declares" in low, "and when to leave it alone"


# ================================================================================================
# X-SURF-5: the overflow refusal wraps; the panel stays under the house width
# ================================================================================================

def test_the_overflow_refusal_keeps_the_panel_under_the_house_width(panel, qtbot, tmp_path):
    rng = np.random.default_rng(0)
    cur = rng.normal(-60.0, 5.0, 20_000).astype(np.float32)
    volt = np.full(20_000, -100.0, dtype=np.float32)
    s = panel._session
    did = s.add_document(_doc(tmp_path / "rec.csv", (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE)),
        (cur, volt)))
    panel.set_selection(did, 0)
    before = panel.minimumSizeHint().width()
    assert before <= HOUSE_WIDTH_PX

    _type(qtbot, panel._gain, "1e38")
    problem = panel._formula.text()
    assert "overflows float32" in problem and "Keep |gain| below" in problem, problem
    width = panel.minimumSizeHint().width()
    assert width <= HOUSE_WIDTH_PX, (
        f"the refusal sentence widened the panel to {width} px (was {before} px); in the "
        f"Inspector dock that puts its remedy off-screen behind a horizontal scrollbar")


# ================================================================================================
# X-SURF-6: a tainted channel says its extremes are not computable, and keeps a median
# ================================================================================================

def _tainted(tmp_path, *, with_inf: bool) -> TraceDocument:
    sig = np.full(5_000, -50.0, dtype=np.float32)
    sig[100] = np.nan
    if with_inf:
        sig[200] = np.inf
    volt = np.full(5_000, -100.0, dtype=np.float32)
    return _doc(tmp_path / "rec.csv", (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE)),
        (sig, volt))


def test_a_channel_holding_an_infinity_says_its_extremes_are_not_computable(
        panel, qtbot, tmp_path):
    s = panel._session
    did = s.add_document(_tainted(tmp_path, with_inf=True))
    panel.set_selection(did, 0)

    rows = {k: (r.text(), c.text()) for k, (r, c) in panel._stat_labels.items()}
    for key in ("min", "max", "span"):
        assert rows[key] == ("—", "—"), rows
    coverage = panel._coverage_row.text()
    assert "exact over all" not in coverage, coverage
    assert "not computable" in coverage and "non-finite" in coverage, coverage
    assert "File info" in coverage, "and points at the surface that names where"
    for key in ("min", "max", "span"):
        for tip in (panel._stat_names[key].toolTip(),
                    panel._stat_labels[key][0].toolTip(),
                    panel._stat_labels[key][1].toolTip()):
            assert "Exact over all" not in tip and "exact over all" not in tip, tip
            assert "non-finite" in tip, tip
    assert "exact over all" not in panel._stats_box.toolTip()

    # The median is the robust row the CAL3 fix promised -- of the finite samples, and
    # it says so -- not "nan".
    assert rows["median"][0] == "-50", rows
    assert "non-finite" in panel._stat_names["median"].toolTip()
    assert "non-finite samples ignored" in coverage, coverage

    _type(qtbot, panel._gain, "2")
    assert panel._stat_labels["median"][1].text() == "-100"
    assert panel._stat_labels["min"][1].text() == "—"
    assert panel._btn_apply.isEnabled(), "the taint is the load's report, not this gain's"
    # ...and the conductance read-out is the same median, not a NaN in pA.
    assert panel._cond_iopen.text() == "-100 pA", panel._cond_iopen.text()
    assert panel._cond_g.text() == "1 nS", "-100 pA / -100 mV"


def test_a_nan_on_the_stride_is_ignored_by_the_median_and_disclosed(panel, qtbot, tmp_path):
    s = panel._session
    did = s.add_document(_tainted(tmp_path, with_inf=False))   # extremes stay finite
    panel.set_selection(did, 0)

    rows = {k: (r.text(), c.text()) for k, (r, c) in panel._stat_labels.items()}
    assert rows["median"][0] == "-50", rows
    assert rows["min"][0] == "-50" and rows["max"][0] == "-50" and rows["span"][0] == "0"
    coverage = panel._coverage_row.text()
    assert "exact over all 5,000 samples" in coverage, "the extremes ARE exact (NaN ignored)"
    assert "1 non-finite" in coverage and "ignored" in coverage, coverage
    assert "ignored" in panel._stat_names["median"].toolTip()


def test_a_clean_channel_still_claims_exactness_plainly(panel, qtbot, tmp_path):
    """Control: nothing about a clean channel's coverage line or tooltips changed."""
    s = panel._session
    cur = np.full(5_000, -50.0, dtype=np.float32)
    did = s.add_document(_doc(tmp_path / "rec.csv", (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),),
        (cur,)))
    panel.set_selection(did, 0)
    assert panel._coverage_row.text() == (
        "min / max / span: exact over all 5,000 samples.   median: every sample "
        "(5,000 of 5,000).")
    assert "ignored" not in panel._stat_names["median"].toolTip()
    assert panel._stat_names["min"].toolTip().startswith("Exact over all 5,000 samples")
