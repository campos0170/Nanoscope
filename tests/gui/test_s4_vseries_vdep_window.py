"""S4 voltage-series — what the user SEES in Analysis ▸ Voltage dependence….

The pure-layer half of these four defects is pinned in
tests/analysis/test_s4_vseries_disclosure.py. These are the shipped surfaces: the
plot TITLE (which is what a Save-image PNG rasterises), the warning banner, the
per-recording table and the Fit-details pane, driven through the real widgets and,
for the entry point, through the real ``Analysis ▸ Voltage dependence…`` QAction on a
real :class:`MainWindow`.

* VS-03 — a detected recording the series could not place was invisible on every one
  of those surfaces. It is now on the banner, in the details pane and in the title.
* VS-08 — the exponential fit discarded a plotted recording and printed "over 3
  recording(s)" above "n = 2 recording(s)" with nothing between them.
* VS-02 — an "Event count" trend across unequal analysed durations, with the word
  duration nowhere on the window.
* VS-05 — nothing said which x-coordinates were measured and which were read off a
  filename, although the I-V window prints exactly that for the same recordings.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

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
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

WHERE = "(S4 voltage-series: the window says what it lost, dropped and pooled)"


# ================================================================================================
# builders
# ================================================================================================

def _nominal_doc(tmp_path, *, mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _measured_doc(tmp_path, *, raw_mV: float, name: str, units: str = "",
                  n: int = 2048) -> TraceDocument:
    cur = np.ascontiguousarray(np.full(n, -60.0, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, raw_mV, dtype=np.float32))
    infos = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="command", raw_units=units, kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta, channels=(cur, volt))


def _df(n: int) -> pd.DataFrame:
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 10 * i, "end_idx": 10 * i + 5, "t_start_s": 1e-3 * i,
            "dwell_s": 1e-3, "baseline_pA": -1000.0, "depth_pA": 50.0,
            "frac_depth": 0.05, "ecd_pA_s": 0.05, "max_dev_pA": 60.0,
            "mean_dev_pA": 50.0, "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False, "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _detect(session: Session, doc_id: str, n: int, *, duration_s: float) -> None:
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(_df(n), {"analyzed_duration_s": float(duration_s),
                                  "recording": "rec"})))


def _title(w: VDepWindow) -> str:
    label = w.plot_item().titleLabel
    return "" if label is None else str(label.text)


def _column(w: VDepWindow, header: str) -> list[str]:
    cols = [w._table.horizontalHeaderItem(c).text()
            for c in range(w._table.columnCount())]
    assert header in cols, f"no {header!r} column in {cols} {WHERE}"
    c = cols.index(header)
    return [w._table.item(r, c).text() for r in range(w._table.rowCount())]


def _window(qtbot, session, *, metric="capture_rate_hz", fit="none") -> VDepWindow:
    w = VDepWindow(session)
    qtbot.addWidget(w)
    w.set_metric(metric)
    w.set_fit(fit)
    return w


# ================================================================================================
# 1. VS-03 -- the lost recording is on screen
# ================================================================================================

def _mixed_session(tmp_path) -> Session:
    """Two placeable recordings plus one whose voltage units the file never declared."""
    session = Session()
    for mV, n in ((-500.0, 12), (-700.0, 20)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, n, duration_s=2.0)
    h5 = session.add_document(_measured_doc(tmp_path, raw_mV=519.7, name="mystery.h5"))
    _detect(session, h5, 7, duration_s=2.0)
    return session


def test_the_window_names_the_recording_it_could_not_place(qapp, qtbot, tmp_path):
    """THE VS-03 REGRESSION AT THE SURFACE: banner, details pane and title."""
    w = _window(qtbot, _mixed_session(tmp_path), fit="exponential")
    assert len(w.points()) == 2, WHERE

    banner = w.warning_text()
    assert "NOT IN THIS SERIES" in banner, f"the window lost a recording silently {WHERE}"
    assert "mystery.h5" in banner, f"{banner} {WHERE}"
    assert "voltage units (unknown)" in banner and "Calibration" in banner, WHERE

    details = w.details_text()
    assert "mystery.h5" in details, WHERE
    assert "covers 2 of the 3 open recording(s)" in details, f"{details} {WHERE}"
    assert "519.7" in details, WHERE               # the value, in its honest form
    # ...and on the surface an exported PNG keeps.
    assert "NOT in this series" in _title(w), f"{_title(w)} {WHERE}"


def test_declaring_the_units_clears_the_disclosure_and_adds_the_point(qapp, qtbot, tmp_path):
    """A disclosure that cannot go away is decoration; this one tracks the remedy."""
    session = _mixed_session(tmp_path)
    w = _window(qtbot, session, fit="exponential")
    h5 = next(d.doc_id for d in session.documents() if d.display_name == "mystery.h5")
    session.set_calibration(h5, 1, ChannelCalibration(target_units="mV"),
                            write_sidecar=False)
    w.refresh()
    assert len(w.points()) == 3, WHERE
    assert "NOT IN THIS SERIES" not in w.warning_text(), f"{w.warning_text()} {WHERE}"
    assert "NOT in this series" not in _title(w), WHERE
    assert w._table.rowCount() == 3, WHERE


def test_all_recordings_refused_does_not_read_as_no_recordings(qapp, qtbot, tmp_path):
    """The empty state must not say "No usable recordings" when there ARE some."""
    session = Session()
    for i, raw in enumerate((519.7, -600.0)):
        doc_id = session.add_document(
            _measured_doc(tmp_path, raw_mV=raw, name=f"nounits_{i}.h5"))
        _detect(session, doc_id, 5, duration_s=2.0)
    w = _window(qtbot, session)
    text = w.details_text()
    assert "No usable recordings" not in text, f"{text} {WHERE}"
    assert "nounits_0.h5" in text and "nounits_1.h5" in text, f"{text} {WHERE}"
    assert "Calibration" in text, WHERE


def test_the_action_opens_a_window_that_discloses_the_loss(qapp, qtbot, tmp_path):
    """Through the REAL menu action the user clicks, on a real MainWindow."""
    win = MainWindow()
    qtbot.addWidget(win)
    session = win._session
    for mV, n in ((-500.0, 12), (-700.0, 20)):
        doc = _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat")
        doc_id = session.add_document(doc)
        win._create_default_layers(doc)
        _detect(session, doc_id, n, duration_s=2.0)
    doc = _measured_doc(tmp_path, raw_mV=519.7, name="mystery.h5")
    h5 = session.add_document(doc)
    win._create_default_layers(doc)
    _detect(session, h5, 7, duration_s=2.0)

    win._sync_vdep_action()
    assert win._act_vdep.isEnabled(), WHERE
    win._act_vdep.trigger()                      # the gesture itself
    w = win._vdep_window
    assert w is not None, WHERE
    assert "mystery.h5" in w.warning_text(), f"{w.warning_text()} {WHERE}"
    assert "mystery.h5" in w.details_text(), WHERE


# ================================================================================================
# 2. VS-08 -- a plotted recording that is not in the fit
# ================================================================================================

def _zero_rung_session(tmp_path) -> Session:
    """A real low-|V| rung that captured nothing: 0 Hz over a real 5 s."""
    session = Session()
    for mV, n in ((-200.0, 0), (-500.0, 50), (-700.0, 90)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"pore_{int(-mV)}.dat"))
        _detect(session, doc_id, n, duration_s=5.0)
    return session


def test_a_recording_dropped_by_the_exponential_fit_is_named(qapp, qtbot, tmp_path):
    """THE VS-08 REGRESSION AT THE SURFACE. 3 plotted, 2 fitted, and it says so."""
    w = _window(qtbot, _zero_rung_session(tmp_path), metric="capture_rate_hz",
                fit="exponential")
    assert len(w.points()) == 3, WHERE
    assert w.fit()["n"] == 2, WHERE

    details = w.details_text()
    assert "DROPPED FROM THE FIT" in details, f"the fit lost a point silently {WHERE}"
    assert "pore_200.dat" in details.split("DROPPED FROM THE FIT")[1], f"{details} {WHERE}"
    assert "no logarithm" in details, WHERE
    # THE TWO COUNTS ON ONE SCREEN NO LONGER DISAGREE WITHOUT A SENTENCE BETWEEN THEM.
    first = details.splitlines()[0]
    assert "over 3 recording(s)" in first and "2 of them in the fit" in first, (
        f"{first!r} {WHERE}")
    assert "n = 2 recording(s)" in details, WHERE
    # ...and the row is marked where the reader would look for it.
    marked = [r for r in _column(w, "Recording") if "not in the fit" in r]
    assert marked and marked[0].startswith("pore_200.dat"), f"{marked} {WHERE}"


def test_a_fit_that_drops_nothing_says_nothing(qapp, qtbot, tmp_path):
    """The mark is a signal: with every rate positive there is nothing to disclose."""
    session = Session()
    for mV, n in ((-500.0, 50), (-700.0, 90)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"pore_{int(-mV)}.dat"))
        _detect(session, doc_id, n, duration_s=5.0)
    w = _window(qtbot, session, fit="exponential")
    details = w.details_text()
    assert "DROPPED FROM THE FIT" not in details, f"{details} {WHERE}"
    assert "of them in the fit" not in details.splitlines()[0], WHERE
    assert not [r for r in _column(w, "Recording") if "not in the fit" in r], WHERE


def test_the_mark_clears_when_the_fit_changes(qapp, qtbot, tmp_path):
    """Stale state is re-derived: switching to a fit that takes the point unmarks it."""
    w = _window(qtbot, _zero_rung_session(tmp_path), fit="exponential")
    assert [r for r in _column(w, "Recording") if "not in the fit" in r], WHERE
    w.set_fit("linear")                 # a straight line has no logarithm to take
    assert w.fit()["n"] == 3, WHERE
    assert not [r for r in _column(w, "Recording") if "not in the fit" in r], WHERE
    assert "DROPPED FROM THE FIT" not in w.details_text(), WHERE


# ================================================================================================
# 3. VS-02 -- the analysed duration is on the window
# ================================================================================================

def _duration_ladder(tmp_path) -> Session:
    """The owner's own spans, with one identical true capture rate of 4.0 Hz."""
    session = Session()
    for mV, dur in ((-500.0, 250.0), (-600.0, 60.0), (-700.0, 180.0)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, int(4.0 * dur), duration_s=dur)
    return session


def test_the_event_count_trend_states_the_durations_it_pooled(qapp, qtbot, tmp_path):
    """THE VS-02 REGRESSION AT THE SURFACE: title, table, provenance and details."""
    w = _window(qtbot, _duration_ladder(tmp_path), metric="n_events", fit="linear")
    assert w.fit() is not None, WHERE                 # still fitted, never refused

    title = _title(w)
    assert "UNEQUAL analysed durations" in title, f"{title} {WHERE}"
    assert "4.17×" in title or "4.17x" in title, f"{title} {WHERE}"

    details = w.details_text()
    assert "COUNTS ACROSS UNEQUAL ANALYSED DURATIONS" in details, f"{details} {WHERE}"
    assert "capture rate" in details, WHERE           # the honest sibling, named
    assert "Duration:" in details, WHERE              # the pooled-provenance row
    for span in ("250.000", "60.000", "180.000"):
        assert span in details, f"{span} missing {WHERE}"
    # ...and beside every count, in the row that produced it.
    assert sorted(_column(w, "Duration (s)")) == ["180.000", "250.000", "60.000"], WHERE


def test_a_rate_over_the_same_recordings_shows_the_durations_and_does_not_warn(
        qapp, qtbot, tmp_path):
    """PER METRIC. A capture rate has already divided the duration out."""
    w = _window(qtbot, _duration_ladder(tmp_path), metric="capture_rate_hz", fit="linear")
    assert "UNEQUAL analysed durations" not in _title(w), f"{_title(w)} {WHERE}"
    assert "COUNTS ACROSS UNEQUAL" not in w.details_text(), WHERE
    # The spans are still SHOWN -- marked as unread by this metric, like Baseline/Gain.
    assert "not read by this metric" in w.details_text().split("Duration:")[1][:60], WHERE
    assert sorted(_column(w, "Duration (s)")) == ["180.000", "250.000", "60.000"], WHERE


def test_equal_durations_are_stated_positively_not_warned(qapp, qtbot, tmp_path):
    """The quiet case still says what it pooled -- silence would be the old defect."""
    session = Session()
    for mV in (-500.0, -700.0):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, 40, duration_s=40.0)
    w = _window(qtbot, session, metric="n_events", fit="linear")
    assert "UNEQUAL" not in _title(w), f"{_title(w)} {WHERE}"
    assert "over 40.000 s each" in _title(w), f"{_title(w)} {WHERE}"


def test_the_duration_disclosure_reaches_the_exported_png(qapp, qtbot, tmp_path):
    """The title is what Save image rasterises, so the caveat leaves the app with it."""
    w = _window(qtbot, _duration_ladder(tmp_path), metric="n_events", fit="linear")
    out = tmp_path / "vdep.png"
    assert w.save_image(str(out)) is not None, WHERE
    assert out.is_file() and out.stat().st_size > 0, WHERE
    assert "UNEQUAL analysed durations" in _title(w), WHERE


# ================================================================================================
# 4. VS-05 -- measured against read off a filename
# ================================================================================================

def test_the_table_and_title_say_where_each_voltage_came_from(qapp, qtbot, tmp_path):
    """THE VS-05 REGRESSION AT THE SURFACE."""
    session = Session()
    a = session.add_document(_measured_doc(tmp_path, raw_mV=-700.0, name="meas.abf",
                                           units="mV"))
    _detect(session, a, 20, duration_s=4.0)
    b = session.add_document(_nominal_doc(tmp_path, mV=-600.0, name="from_name.csv"))
    _detect(session, b, 12, duration_s=4.0)

    w = _window(qtbot, session, fit="linear")
    assert sorted(_column(w, "V source")) == ["measured", "nominal"], WHERE
    assert "MIXED voltage sources" in _title(w), f"{_title(w)} {WHERE}"
    details = w.details_text()
    assert "V source:" in details, WHERE
    assert "measured" in details and "nominal" in details, WHERE
    # The V column stays a bare number -- the source is its own column, not a suffix.
    assert sorted(float(v) for v in _column(w, "V (mV)")) == [-700.0, -600.0], WHERE


def test_one_shared_source_is_stated_without_a_mixed_warning(qapp, qtbot, tmp_path):
    session = Session()
    for mV in (-500.0, -700.0):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.csv"))
        _detect(session, doc_id, 10, duration_s=4.0)
    w = _window(qtbot, session, fit="linear")
    assert "MIXED voltage sources" not in _title(w), WHERE
    assert set(_column(w, "V source")) == {"nominal"}, WHERE
    assert "V source: nominal" in " ".join(w.details_text().split()), WHERE


# ================================================================================================
# 5. the columns the older tests read are still where they were
# ================================================================================================

@pytest.mark.parametrize("header", ["Recording", "V (mV)", "Baseline", "Merge", "Gain"])
def test_the_established_table_columns_survive(qapp, qtbot, tmp_path, header):
    w = _window(qtbot, _duration_ladder(tmp_path), metric="n_events")
    assert len(_column(w, header)) == 3, WHERE
    assert w._table.horizontalHeaderItem(1).text() == "V (mV)", WHERE
