"""Sweep 7: the Calibration panel tells the truth about the channel it is calibrating.

Three findings, one panel (docs/sweep7/FINDINGS.md):

* the skeptics' MISSED (minor) item — after "Treat this channel as: current" was committed
  the title still read "Untitled · unknown" two rows above the combo reading "current", and
  the Open-pore-conductance / Command-voltage boxes stayed HIDDEN because the partner
  channels were resolved by header kind. The one panel that exists to dial a current gain
  against a conductance would not show a conductance for exactly the recordings the
  override was built for;
* S7-CAL4 — the preview multiplied a float64 stride while ``apply()`` casts to float32, so
  gain 1e38 previewed "median -5.77e+39 pA", Apply was offered, and every sample became
  -inf. The preview now uses apply()'s own arithmetic, decides overflow from the exact raw
  extremes, and refuses BEFORE Apply with the per-channel bound in the sentence;
* S7-CAL3 — min / max / span were read from a <=20,000-sample stride and printed as plain
  numbers; on the owner's -500 mV file the max had the wrong sign (-7.63 vs +5.49 pA).
  They are exact now, and the median — the one row still strided — says so, with the
  stride and the counts (MeasurePanel's policy, analysis/measure.py);
* S7-CAL5 — Apply on a recording whose file had moved reported "saved to pore.abf.npcal.json"
  and wrote it into the folder the recording had left.

Every test drives the real panel: ``set_selection``, key clicks into the real fields, the
real Apply button.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import shutil

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration, sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel

FS = 50_000.0


def _doc(path, channels, arrays) -> TraceDocument:
    n = int(arrays[0].size)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    return TraceDocument(meta=meta, channels=tuple(arrays))


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


# ================================================================================================
# the title and the partner boxes follow the override
# ================================================================================================

def test_the_title_and_the_partner_boxes_follow_the_kind_override(panel, qtbot, tmp_path):
    s = panel._session
    cur = np.full(40_000, -49.13, dtype=np.float32)      # a -700 mV open pore, in pA
    volt = np.full(40_000, -700.0, dtype=np.float32)     # its command voltage
    doc = _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)),
        (cur, volt))
    did = s.add_document(doc)
    panel.set_selection(did, 0)
    assert "unknown" in panel._title.text()
    assert not panel._cond_box.isVisible() and not panel._recon_box.isVisible()

    # The corrections, as the panel commits them.
    s.set_calibration(did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    s.set_calibration(did, 1, ChannelCalibration(1.0, 0.0, "mV", kind=ChannelKind.VOLTAGE),
                      write_sidecar=False)
    panel.set_selection(did, 0)
    title = panel._title.text()
    assert "current (file says unknown)" in title, title
    assert panel._cond_box.isVisible(), "the conductance box exists for this recording"
    assert panel._recon_box.isVisible(), "and so does the command-voltage check"
    assert panel._cond_iopen.text().startswith("-49.13")
    assert panel._cond_g.text().endswith("nS"), panel._cond_g.text()
    assert panel._cond_g.text().startswith("0.0701"), "G = -49.13 pA / -700 mV = 0.0702 nS"


# ================================================================================================
# CAL4: the preview overflows where apply() overflows, and refuses before Apply
# ================================================================================================

def test_a_gain_that_would_overflow_float32_is_refused_before_apply(panel, qtbot, tmp_path):
    arr = np.linspace(-263.062, -29.9072, 40_000, dtype=np.float32)   # IN15 of the -1000 mV ABF
    s = panel._session
    did = s.add_document(_doc(tmp_path / "pore.abf", (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),), (arr,)))
    panel.set_selection(did, 0)

    _type(qtbot, panel._gain, "1e38")
    assert panel._pending() is None
    problem = panel._formula.text()
    assert "overflows float32" in problem and "1.29e+36" in problem, problem
    assert not panel._btn_apply.isEnabled()
    for key in ("median", "min", "max", "span"):
        assert panel._stat_labels[key][1].text() == "—", "no finite-looking preview"
    panel._btn_apply.click()
    assert s.calibration(did, 0).gain == 1.0, "nothing was committed"
    assert s.calibration_issues(did, 0) == ()

    # Just under the bound is honoured — the rule is overflow, not "suspiciously large".
    _type(qtbot, panel._gain, "1e36")
    assert panel._pending() == (1e36, 0.0)
    assert panel._btn_apply.isEnabled()
    assert panel._formula.text() == "calibrated = raw × 1e+36 + 0"
    assert float(panel._stat_labels["min"][1].text()) == pytest.approx(-2.63062e38, rel=1e-4)

    # A negative gain swaps the extremes; the preview keeps (min, max) in order.
    _type(qtbot, panel._gain, "-2")
    assert float(panel._stat_labels["min"][1].text()) == pytest.approx(59.8144, rel=1e-4)
    assert float(panel._stat_labels["max"][1].text()) == pytest.approx(526.124, rel=1e-4)


# ================================================================================================
# CAL3: exact extremes, a median that says it is sampled
# ================================================================================================

def test_min_max_span_are_exact_and_the_median_says_it_is_sampled(panel, qtbot, tmp_path):
    n = 200_000                                  # stride 10 -> 20,000 samples read
    arr = np.full(n, -50.0, dtype=np.float32)
    arr[7] = 5.49                                # not on the stride
    arr[13] = -281.677                           # not on the stride
    s = panel._session
    did = s.add_document(_doc(tmp_path / "pore.abf", (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),), (arr,)))
    panel.set_selection(did, 0)

    raw = {k: panel._stat_labels[k][0].text() for k in ("median", "min", "max", "span")}
    assert float(raw["max"]) == pytest.approx(5.49, rel=1e-5), raw
    assert float(raw["min"]) == pytest.approx(-281.677, rel=1e-5), raw
    assert float(raw["span"]) == pytest.approx(287.167, rel=1e-5), raw
    assert float(raw["median"]) == -50.0

    _type(qtbot, panel._gain, "2.5")
    cal = {k: panel._stat_labels[k][1].text() for k in ("median", "min", "max", "span")}
    assert float(cal["max"]) == pytest.approx(13.725, rel=1e-5), cal
    assert float(cal["min"]) == pytest.approx(-704.1925, rel=1e-5), cal

    # It SAYS what was read: the stride and the counts, on the title, the box tooltip, the
    # median row and the coverage line; the exact rows say they are exact.
    assert "exact" in panel._stats_box.title() and "median sampled" in panel._stats_box.title()
    assert panel._stat_names["median"].text() == "median (sampled)"
    median_tip = panel._stat_names["median"].toolTip()
    assert "every 10th sample" in median_tip and "20,000 of 200,000" in median_tip, median_tip
    for key in ("min", "max", "span"):
        assert "Exact over all 200,000 samples" in panel._stat_names[key].toolTip()
        assert "Exact over all 200,000 samples" in panel._stat_labels[key][0].toolTip()
    assert "every 10th sample (20,000 of 200,000)" in panel._coverage_row.text()
    assert "exact over all 200,000 samples" in panel._stats_box.toolTip()


# ================================================================================================
# CAL5: Apply on a moved recording says it was not saved beside the data
# ================================================================================================

def test_apply_on_a_moved_recording_reports_it_was_not_saved_beside_the_data(
        panel, qtbot, tmp_path):
    old, new = tmp_path / "cal5_old", tmp_path / "cal5_new"
    old.mkdir(); new.mkdir()
    data = old / "pore.abf"
    data.write_bytes(b"the samples are in memory")
    arr = np.full(40_000, -57.68, dtype=np.float32)
    s = panel._session
    did = s.add_document(_doc(data, (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),), (arr,)))
    panel.set_selection(did, 0)

    _type(qtbot, panel._gain, "0.75")
    panel._btn_apply.click()
    assert panel.status_text() == "Applied · saved to pore.abf.npcal.json"
    side = sidecar_path(data)
    assert json.loads(side.read_text())["channels"]["0"]["gain"] == 0.75

    shutil.move(str(data), str(new / "pore.abf"))       # the user tidies the file in Finder

    _type(qtbot, panel._gain, "0.9")
    panel._btn_apply.click()
    status = panel.status_text()
    assert status.startswith("Applied · not saved beside the recording"), status
    assert "pore.abf is no longer at" in status and str(old) in status, status
    assert "lives only in this session" in status
    assert s.calibration(did, 0).gain == 0.9, "the calibration is live"
    assert json.loads(side.read_text())["channels"]["0"]["gain"] == 0.75, (
        "the orphan sidecar in the folder the recording left is not rewritten")
    assert not sidecar_path(new / "pore.abf").exists()
