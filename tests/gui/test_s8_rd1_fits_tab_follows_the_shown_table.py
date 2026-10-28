"""S8-RD1 — The Results dock's Fits tab shows a fit only beside the table it was made on.

MEASURED DEFECT (sweep 8, the owner's −600 mV recording, Results dock). The result store keeps
ONE dwell fit and ONE depth fit per (recording, channel) — ``fit.dwell`` / ``fit.depth`` — and a
fit carried no record of which table it was fitted to. ``AnalysisPanel._refresh_fit_text`` printed
whatever sat in the slot beside whichever table the Table selector named, whatever had happened
to that table since:

* Fit under Schmitt (n = 316), switch the selector to CUSUM (369 accepted, never fitted): the Fits
  tab still read Schmitt's n = 316 under a table of 370.
* Press Fit under CUSUM: the dwell fit refused (4 events below t_min) but the depth fit landed, and
  back on Schmitt the tab printed Schmitt's dwell fit beside CUSUM's depth fit as one readout.
* Merge all (246 accepted), a re-run with other settings (44 events), one rejection (315), and the
  Undo of any of them: the tab went on printing the n = 316 fit — which ``MainWindow._stored_fit``
  refuses for Export figure as "saved fit is stale (the accepted events changed since it ran)".
* A reopened project showed the same single slot under whichever table came up first.

Nothing on the tab, its tooltip or the amber stale line said the numbers belonged to another table,
and tau, mu and their 95 % intervals are the numbers that end up in a methods section.

THE FIX. A fit now records the table it was made on (``detection_module_id``, beside the
``n_input_events`` it already carried — an additive ``FitResult.meta`` key, so older builds read the
project unchanged). ``AnalysisPanel._refresh_fit_text`` asks ``modules.judge_stored_fit`` — the
export's own two guards (calibration stamp, accepted count) plus that lineage — and prints a fit
only when it belongs to the table on screen; otherwise it says why nothing is shown and to press
Fit. The refresh runs from the one place every route funnels through (a table is shown, a fit
lands, an accept/reject changes the accepted set). A Fit press now replaces the PAIR: a half that
fails retires its stored predecessor instead of leaving it beside the fresh other half. A fit
without a recorded table (saved by an older build, or filed by the Plots window) is kept in the
store and in the project, still serves the export exactly as before, and is NOT shown on the tab:
the tab cannot tell which table it describes and says so.

These tests drive the real MainWindow on a synthetic recording (40 blockades, ten of them
doublets, so Schmitt finds 50 candidates and CUSUM 40): the Table selector, the Fit button
(including a press whose dwell half fails), the Merge all button, the table's R key (one row, then
a whole-table selection), a re-run with a higher k_in, a gain change, Edit ▸ Undo / Redo, and Save /
Open Project — once with a project written by this build and once with one that predates the
stamp. They assert what the user reads in the Fits box, and pin the persisted key by its spelling.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from nanoscope.analysis.modules import fit_dwell
from nanoscope.analysis.reconcile import reconcile_detections
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 260_000

#: The first line of each fit's ``summary()`` — what tells the two fits apart in the box.
DWELL_MODEL = "exponential (left-truncated MLE)"
DEPTH_MODEL = "Gaussian mixture on dI/I0"

#: The persisted ``FitResult.meta`` key the fix adds. Pinned by its literal spelling on purpose:
#: a saved project is read by older builds, so the name is part of the file format.
LINEAGE_KEY = "detection_module_id"


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 3) -> np.ndarray:
    """40 blockades of 14–46 pA, every fourth a DOUBLET (two sub-events 0.4 ms apart).

    Schmitt reports each half of a doublet — 50 candidates; CUSUM bridges the 0.4 ms gap — 40;
    and "Merge all" at 10 ms takes Schmitt's 50 to 40. The depth spread is what lets a raised
    ``k_in`` drop the shallow events, so a re-run changes the count.
    """
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    pos = 1_000
    depths = np.linspace(14.0, 46.0, 40)
    rng.shuffle(depths)
    for i, depth in enumerate(depths):
        dwell = 120 + int(rng.integers(0, 140))
        if i % 4 == 0:
            x[pos:pos + 70] -= depth
            x[pos + 90:pos + 90 + 70] -= depth
        else:
            x[pos:pos + dwell] -= depth
        pos += 5_000
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=int(arr.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the text reader reopens as one CURRENT channel at FS (a project
    stores the recording as a path, so the round-trip tests need a file to reopen)."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


def _detect(window: MainWindow, qtbot) -> None:
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(30)      # let the queued add_result -> resultsChanged settle


def _press_fit(window: MainWindow, qtbot, *, dwell_k: int = 1, depth_k: int = 1) -> None:
    """Press the Fits tab's own Fit button, the way the user does."""
    stats = window._analysis_panel._stats
    stats.n_components_dwell.setValue(dwell_k)
    stats.n_components_depth.setValue(depth_k)
    stats.fit_button.click()
    qtbot.wait(30)


def _pick_table(window: MainWindow, qtbot, name: str) -> None:
    """Choose a table in the Results toolbar's Table selector."""
    sel = window._analysis_panel._result_selector
    idx = sel.findText(name, Qt.MatchStartsWith)
    assert idx >= 0, [sel.itemText(i) for i in range(sel.count())]
    sel.setCurrentIndex(idx)
    qtbot.wait(30)


def _fits_text(window: MainWindow) -> str:
    return window._analysis_panel._stats.results.toPlainText()


def _accepted(window: MainWindow) -> int:
    df = window._analysis_panel._table.dataframe
    return int(df["accepted"].to_numpy(dtype=bool).sum())


def _selector_text(window: MainWindow) -> str:
    return window._analysis_panel._result_selector.currentText()


def _stored(window: MainWindow, doc_id: str, channel: int, module_id: str):
    return window._session.results.get(doc_id, channel, module_id)


def _detect_schmitt(window: MainWindow, qtbot, tmp_path: Path) -> tuple[str, int]:
    """One recording, one Schmitt table (50 candidates), the Fits tab raised."""
    doc_id, channel = _install(window, _doc(tmp_path / "pore_a.dat", _trace()))
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    _detect(window, qtbot)
    panel.results_tabs.setCurrentIndex(2)
    assert _selector_text(window) == "Schmitt — 50 candidates", _selector_text(window)
    return doc_id, channel


def _two_tables(window: MainWindow, qtbot, tmp_path: Path) -> tuple[str, int]:
    """Schmitt (50 candidates) AND CUSUM (40) on one recording, Schmitt on screen."""
    doc_id, channel = _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    panel.detection_panel.set_detector("detect.cusum")
    qtbot.wait(30)
    _detect(window, qtbot)
    assert _selector_text(window) == "CUSUM — 40 candidates", _selector_text(window)
    _pick_table(window, qtbot, "Schmitt")
    assert _selector_text(window) == "Schmitt — 50 candidates", _selector_text(window)
    return doc_id, channel


def _heads(text: str) -> list[int]:
    """The n of every fit printed: the ``n = 50   loglik = ...`` line under each model name.
    (A goodness-of-fit sentence quotes n again, so counting the substring would double it.)"""
    return [int(line.split()[2]) for line in (raw.strip() for raw in text.splitlines())
            if line.startswith("n = ")]


def _says_press_fit(text: str) -> bool:
    return "press fit" in text.lower()


def _shows_no_fit(text: str) -> bool:
    """Neither fit's summary is on the box."""
    return DWELL_MODEL not in text and DEPTH_MODEL not in text and "n = " not in text


# ================================================================================================
# the Table selector
# ================================================================================================

def test_a_fit_made_on_one_table_is_not_shown_under_another(window, qtbot, tmp_path):
    """The finder's step 2: Fit under Schmitt, switch the selector to CUSUM. The box used to
    read Schmitt's n = 50 under a table of 40."""
    _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50], (
        f"setup: no fit on screen: {_fits_text(window)!r}")

    _pick_table(window, qtbot, "CUSUM")
    shown = _fits_text(window)

    assert _shows_no_fit(shown), (
        f"the Fits tab prints Schmitt's fit under the CUSUM table: {shown!r}")
    assert "Schmitt" in shown and "CUSUM" in shown, (
        f"nothing says the stored fit belongs to another table: {shown!r}")
    assert _says_press_fit(shown), f"nothing says what to do: {shown!r}"


def test_a_fit_follows_the_selector_back_to_its_own_table(window, qtbot, tmp_path):
    """The positive half: the fit is not lost, it is shown where it belongs. Away and back
    (a round trip through the selector) it is the same readout, character for character."""
    _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)
    before = _fits_text(window)
    assert DWELL_MODEL in before and DEPTH_MODEL in before

    _pick_table(window, qtbot, "CUSUM")
    _pick_table(window, qtbot, "Schmitt")

    assert _fits_text(window) == before


def test_every_table_shows_only_the_fit_made_on_it(window, qtbot, tmp_path):
    """Fit under each table in turn: what the box prints is always about the table beside
    it (its n is that table's accepted count), never the other one's."""
    _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)                       # on Schmitt: n = 50
    _pick_table(window, qtbot, "CUSUM")
    _press_fit(window, qtbot)                       # on CUSUM: n = 40
    on_cusum = _fits_text(window)
    assert _heads(on_cusum) == [40, 40], on_cusum

    _pick_table(window, qtbot, "Schmitt")
    on_schmitt = _fits_text(window)
    assert _heads(on_schmitt) == [], (
        f"CUSUM's fit is printed under the Schmitt table: {on_schmitt!r}")
    assert _says_press_fit(on_schmitt)

    _press_fit(window, qtbot)                       # on Schmitt again
    assert _heads(_fits_text(window)) == [50, 50]
    _pick_table(window, qtbot, "CUSUM")
    assert _heads(_fits_text(window)) == [], (
        "Schmitt's fit is printed under the CUSUM table")


def test_the_lineage_alone_refuses_a_fit_when_the_accepted_counts_agree(window, qtbot, tmp_path):
    """The export's count guard cannot tell two tables apart when they happen to hold the
    same number of accepted events — and a methods-section tau is exactly what that
    coincidence would corrupt. Reject ten of Schmitt's 50 (through the same door the Event
    Inspector uses), so Schmitt and CUSUM both hold 40 accepted events; a fit made on
    Schmitt must still not be shown under CUSUM."""
    _two_tables(window, qtbot, tmp_path)
    panel = window._analysis_panel
    ids = panel._table.dataframe["event_id"].to_numpy()[:10]
    for event_id in ids:
        panel.set_event_accepted(int(event_id), False)
    qtbot.wait(30)
    assert _accepted(window) == 40, "setup: Schmitt should now hold CUSUM's 40"
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [40, 40], _fits_text(window)

    _pick_table(window, qtbot, "CUSUM")
    assert _accepted(window) == 40, "setup: CUSUM holds 40 accepted events"
    shown = _fits_text(window)

    assert _shows_no_fit(shown), (
        f"a fit made on the Schmitt table is printed under CUSUM because the counts agree: "
        f"{shown!r}")
    assert "Schmitt" in shown


def test_the_reconciled_table_has_its_own_fit_and_lends_it_to_no_other(
        window, qtbot, tmp_path, monkeypatch):
    """The finder's last step: the Schmitt fit then showed under the Reconciled table, and a
    fit made on the Reconciled table showed under CUSUM."""
    _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)                                  # on Schmitt: n = 50
    outcome = reconcile_detections(window._reconcile_sources())

    class _Accepted:
        def __init__(self, outcome):
            self.outcome = outcome

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(window, "_make_reconcile_dialog", lambda _sources: _Accepted(outcome))
    window._reconcile_detections()
    qtbot.wait(50)
    assert _selector_text(window).startswith("Reconciled"), _selector_text(window)
    n_reconciled = _accepted(window)
    on_reconciled = _fits_text(window)
    assert _shows_no_fit(on_reconciled), (
        f"Schmitt's fit is printed under the Reconciled table: {on_reconciled!r}")
    assert "Schmitt" in on_reconciled and "Reconciled" in on_reconciled

    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [n_reconciled, n_reconciled], _fits_text(window)
    _pick_table(window, qtbot, "CUSUM")
    on_cusum = _fits_text(window)
    assert _shows_no_fit(on_cusum), (
        f"the Reconciled table's fit is printed under CUSUM: {on_cusum!r}")
    assert "Reconciled" in on_cusum and _says_press_fit(on_cusum)


# ================================================================================================
# a half-failed Fit press
# ================================================================================================

def test_a_failed_dwell_fit_leaves_no_older_dwell_fit_beside_the_new_depth_fit(
        window, qtbot, tmp_path):
    """Press Fit (both land), then press it again with a dwell mixture the sample is too small
    for: the dwell half FAILS, the depth half is replaced. The older dwell fit used to stay in
    the store and came back beside the fresh depth fit the next time the box was redrawn."""
    doc_id, channel = _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)                                  # both land on Schmitt
    assert _stored(window, doc_id, channel, "fit.dwell") is not None

    _press_fit(window, qtbot, dwell_k=4, depth_k=1)            # 50 events < 80 for k = 4
    just_after = _fits_text(window)
    assert "dwell fit failed" in just_after, just_after
    assert DEPTH_MODEL in just_after

    _pick_table(window, qtbot, "CUSUM")                        # redraw the box from the store
    _pick_table(window, qtbot, "Schmitt")
    redrawn = _fits_text(window)
    assert DWELL_MODEL not in redrawn, (
        f"the dwell fit from the PREVIOUS press is back beside the new depth fit: {redrawn!r}")
    assert DEPTH_MODEL in redrawn


def test_a_half_failed_fit_under_another_table_is_not_a_chimera(window, qtbot, tmp_path):
    """The finder's step 3. Schmitt is fitted; under CUSUM the dwell half fails and the depth
    half lands. Back on Schmitt the box used to print Schmitt's dwell fit (n = 50) and CUSUM's
    depth fit (n = 40) as one readout."""
    _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)
    _pick_table(window, qtbot, "CUSUM")
    _press_fit(window, qtbot, dwell_k=4, depth_k=1)
    on_cusum = _fits_text(window)
    assert "dwell fit failed" in on_cusum and _heads(on_cusum) == [40], on_cusum

    _pick_table(window, qtbot, "Schmitt")
    on_schmitt = _fits_text(window)

    assert len(set(_heads(on_schmitt))) <= 1, (
        f"two tables' fits under one readout (n = {_heads(on_schmitt)}): {on_schmitt!r}")
    assert 40 not in _heads(on_schmitt), (
        f"CUSUM's depth fit is printed under the Schmitt table: {on_schmitt!r}")
    assert 50 not in _heads(on_schmitt), (
        f"Schmitt's dwell fit survived a Fit press that replaced the pair: {on_schmitt!r}")
    assert _says_press_fit(on_schmitt)

    _pick_table(window, qtbot, "CUSUM")
    assert _heads(_fits_text(window)) == [40] and DEPTH_MODEL in _fits_text(window)


# ================================================================================================
# what happens to the table the fit sits beside
# ================================================================================================

def test_merge_all_marks_the_fit_as_predating_the_merge(window, qtbot, tmp_path):
    """Merge all at 10 ms takes Schmitt's 50 rows to 40: the fit of the 50 is not the fit of
    these 40, and the export already says so. The box printed n = 50 beside the merged table."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50]

    panel.merge_toggle.setChecked(True)
    panel._merge.window_ms.setValue(10.0)
    with qtbot.waitSignal(panel.mergeFinished, timeout=60_000):
        panel._merge.merge_all_btn.click()
    qtbot.wait(100)
    assert _accepted(window) == 40, "setup: Merge all should leave 40 accepted events"
    shown = _fits_text(window)

    assert _shows_no_fit(shown), f"the pre-merge fit is printed beside the merged table: {shown!r}"
    assert "50" in shown and "40" in shown, (
        f"the box does not say which accepted counts differ: {shown!r}")
    assert _says_press_fit(shown)

    window._act_undo.trigger()                      # Edit > Undo "Merge events (50 -> 40)"
    qtbot.wait(100)
    assert _accepted(window) == 50
    assert _heads(_fits_text(window)) == [50, 50], (
        "the fit is not shown again once its own table is back: " + _fits_text(window))

    window._act_redo.trigger()                      # ...and Redo puts the merged table back
    qtbot.wait(100)
    assert _accepted(window) == 40
    assert _shows_no_fit(_fits_text(window)), _fits_text(window)


def test_one_rejection_marks_the_fit_and_undo_gives_it_back(window, qtbot, tmp_path):
    """The skeptic's V1: press R on one accepted event. The box read n = 50 beside 49
    accepted until something else happened to redraw it."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50]

    window.show()
    panel._table.view.setFocus()
    panel._table.view.selectRow(3)
    qtbot.wait(30)
    QTest.keyClick(panel._table.view, Qt.Key_R)
    qtbot.wait(50)
    assert _accepted(window) == 49, "setup: R should reject the selected event"
    shown = _fits_text(window)

    assert _shows_no_fit(shown), f"the box still prints the 50-event fit under 49: {shown!r}"
    assert "50" in shown and "49" in shown
    assert _says_press_fit(shown)

    window._act_undo.trigger()                       # Edit > Undo "Curate event"
    qtbot.wait(100)
    assert _accepted(window) == 50
    assert _heads(_fits_text(window)) == [50, 50], _fits_text(window)


def test_a_bulk_rejection_marks_the_fit_too(window, qtbot, tmp_path):
    """Select every row and press R: ONE trailing acceptance announcement for fifty rows (the
    table's bulk path), the same door the Inspector's Reject all lands on afterwards."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50]

    window.show()
    panel._table.view.setFocus()
    panel._table.view.selectAll()
    qtbot.wait(30)
    QTest.keyClick(panel._table.view, Qt.Key_R)
    qtbot.wait(50)
    assert _accepted(window) == 0, "setup: R on a whole-table selection should reject every row"
    shown = _fits_text(window)

    assert _shows_no_fit(shown), f"the 50-event fit is printed under a table with none: {shown!r}"
    assert "50" in shown and " 0 " in shown and _says_press_fit(shown)


def test_a_rerun_with_other_settings_does_not_keep_the_old_fit_on_the_tab(window, qtbot, tmp_path):
    """The skeptic's V2: re-run Schmitt with a higher k_in (fewer rows). The selector says
    "Schmitt — N candidates" for the smaller table and the box went on printing the 50-event
    fit."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50]

    panel._detection.k_in.setValue(15.0)
    qtbot.wait(30)
    _detect(window, qtbot)
    kept = _accepted(window)
    assert 20 < kept < 50, f"setup: a higher k_in should drop the shallow events (kept {kept})"
    assert _selector_text(window) == f"Schmitt — {kept} candidates"
    shown = _fits_text(window)

    assert _shows_no_fit(shown), (
        f"the 50-event fit is printed under the {kept}-event re-run: {shown!r}")
    assert "50" in shown and str(kept) in shown and _says_press_fit(shown)


def test_undoing_a_rerun_does_not_put_the_rerun_s_fit_under_the_restored_table(
        window, qtbot, tmp_path):
    """The skeptic's V3: Fit under the re-run's smaller table, then Edit > Undo the re-run. The
    restored 50-row table used to sit under the smaller table's fit."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    panel._detection.k_in.setValue(15.0)
    qtbot.wait(30)
    _detect(window, qtbot)
    kept = _accepted(window)
    assert 20 < kept < 50, f"setup: a higher k_in should drop the shallow events (kept {kept})"
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [kept, kept], _fits_text(window)

    window._act_undo.trigger()                       # Edit > Undo "Re-run detection"
    qtbot.wait(100)
    assert _accepted(window) == 50, "setup: Undo should bring the 50-row table back"
    restored = _fits_text(window)

    assert kept not in _heads(restored), (
        f"Undo restores the 50-row table under the {kept}-event fit: {restored!r}")
    assert _says_press_fit(restored)


def test_a_fit_made_under_the_old_calibration_is_not_shown_beside_a_rerun_under_the_new_one(
        window, qtbot, tmp_path):
    """The export's first guard, on the tab: Fit at gain 1, re-gain, re-run detection. The fit
    describes depths measured against the old gain and says so; the table is fresh, so the way
    out is just Fit."""
    doc_id, channel = _detect_schmitt(window, qtbot, tmp_path)
    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [50, 50]

    window._session.set_calibration(
        doc_id, channel, ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN),
        write_sidecar=False)
    _detect(window, qtbot)                         # the table is now measured under gain 2
    shown = _fits_text(window)

    assert _shows_no_fit(shown), f"a fit of the gain-1 depths is printed: {shown!r}"
    assert "calibration" in shown and _says_press_fit(shown)
    assert "re-run detection" not in shown, "the table is fresh; only Fit is needed"

    _press_fit(window, qtbot)
    assert _heads(_fits_text(window)) == [_accepted(window)] * 2, _fits_text(window)


def test_pressing_fit_on_a_table_the_calibration_has_outdated_says_to_rerun_detection(
        window, qtbot, tmp_path):
    """Detect at gain 1, re-gain (the table is stale: its amber line says re-run), press Fit.
    The fit lands under the NEW calibration on events measured under the old one — which the
    export refuses for the same reason — so the box says so and names the real remedy rather
    than printing numbers a second press would not improve."""
    doc_id, channel = _detect_schmitt(window, qtbot, tmp_path)
    window._session.set_calibration(
        doc_id, channel, ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN),
        write_sidecar=False)
    qtbot.wait(30)

    _press_fit(window, qtbot)
    shown = _fits_text(window)

    assert _shows_no_fit(shown), (
        f"numbers printed beside a table measured under another gain: {shown!r}")
    assert "calibration" in shown and "re-run detection" in shown and _says_press_fit(shown)


def test_the_fits_tab_hides_a_fit_exactly_when_the_export_refuses_it(window, qtbot, tmp_path):
    """The two surfaces read one stored object. Wherever the figure export says "saved fit is
    stale", the Fits tab must not be printing it, and where the export uses it the tab does."""
    _detect_schmitt(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _press_fit(window, qtbot)

    def verdicts():
        res = panel.current_detection()
        return (window._stored_fit(res, "fit.dwell")[0] is not None,
                window._stored_fit(res, "fit.depth")[0] is not None,
                DWELL_MODEL in _fits_text(window), DEPTH_MODEL in _fits_text(window))

    assert verdicts() == (True, True, True, True)
    panel.set_event_accepted(int(panel._table.dataframe["event_id"].iloc[0]), False)
    qtbot.wait(30)
    export_dwell, export_depth, tab_dwell, tab_depth = verdicts()
    assert (export_dwell, export_depth) == (False, False), "setup: the export must refuse both"
    assert (tab_dwell, tab_depth) == (False, False), (
        f"the export refuses these fits as stale and the Fits tab still prints them: "
        f"{_fits_text(window)!r}")
    _, why = window._stored_fit(panel.current_detection(), "fit.dwell")
    assert "stale" in why


# ================================================================================================
# what the fit records, and what a project keeps
# ================================================================================================

def test_a_fit_records_the_table_it_was_made_on_beside_its_existing_keys(window, qtbot, tmp_path):
    """The stamp is ADDED to ``FitResult.meta``; nothing a saved project already holds is
    renamed or removed."""
    doc_id, channel = _two_tables(window, qtbot, tmp_path)
    _press_fit(window, qtbot)
    _pick_table(window, qtbot, "CUSUM")
    _press_fit(window, qtbot)

    dwell = _stored(window, doc_id, channel, "fit.dwell").payload.meta
    depth = _stored(window, doc_id, channel, "fit.depth").payload.meta
    for meta in (dwell, depth):
        assert meta.get(LINEAGE_KEY) == "detect.cusum", meta.keys()
        assert meta["n_input_events"] == 40
        for key in ("use_filter_limited", "n_excluded_filter_limited", "warnings"):
            assert key in meta, f"an existing meta key went missing: {key}"
    assert "t_min" in dwell and "t_min_source" in dwell


def _save_two_table_project(qtbot, tmp_path: Path) -> tuple[Path, Path]:
    """Schmitt AND CUSUM stored on a recording written to disk, the fit made on CUSUM, saved."""
    csv = tmp_path / "pore_a.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = MainWindow()
    qtbot.addWidget(win)
    _install(win, _doc(csv, arr))
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    _detect(win, qtbot)
    panel.detection_panel.set_detector("detect.cusum")
    qtbot.wait(30)
    _detect(win, qtbot)
    _press_fit(win, qtbot)
    assert _heads(_fits_text(win)) == [40, 40]

    proj = tmp_path / "fits.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()
    return proj, csv


def test_the_project_file_records_the_table_beside_the_keys_it_already_held(qtbot, tmp_path):
    """The format: ``fits[].curve.meta`` gains one key; nothing is renamed or removed."""
    proj, _csv = _save_two_table_project(qtbot, tmp_path)

    with zipfile.ZipFile(proj) as z:
        manifest = json.loads(z.read("manifest.json"))
    for module_id in ("fit.dwell", "fit.depth"):
        saved = next(f for f in manifest["fits"] if f["module_id"] == module_id)
        meta = saved["curve"]["meta"]
        assert meta[LINEAGE_KEY] == "detect.cusum"
        assert meta["n_input_events"] == 40
        assert "use_filter_limited" in meta and "n_excluded_filter_limited" in meta


def test_a_reopened_project_shows_each_fit_under_its_own_table(qtbot, tmp_path):
    """Open that project in a fresh window. The slot used to come back under whichever table
    was up first."""
    proj, _csv = _save_two_table_project(qtbot, tmp_path)

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    w2._analysis_panel.results_tabs.setCurrentIndex(2)

    _pick_table(w2, qtbot, "Schmitt")
    on_schmitt = _fits_text(w2)
    assert _shows_no_fit(on_schmitt), (
        f"the reopened project prints CUSUM's fit under the Schmitt table: {on_schmitt!r}")
    assert "CUSUM" in on_schmitt and _says_press_fit(on_schmitt)

    _pick_table(w2, qtbot, "CUSUM")
    on_cusum = _fits_text(w2)
    assert _heads(on_cusum) == [40, 40], on_cusum
    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_project_saved_before_fits_named_their_table_still_opens_and_is_not_trusted(
        qtbot, tmp_path):
    """A fit saved by an earlier build has no ``detection_module_id``. It must open without a
    crash, stay in the store (and in the project, and serve the export exactly as before), but
    the Fits tab cannot tell which table it describes — so it says so instead of printing it.
    Pressing Fit makes a fit that does record its table."""
    csv = tmp_path / "pore_a.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id, channel = _install(win, _doc(csv, arr))
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    _detect(win, qtbot)
    _press_fit(win, qtbot)
    for module_id in ("fit.dwell", "fit.depth"):
        # What an earlier build wrote: the same fit, no record of its table.
        _stored(win, doc_id, channel, module_id).payload.meta.pop(LINEAGE_KEY, None)

    proj = tmp_path / "legacy.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    w2._analysis_panel.results_tabs.setCurrentIndex(2)
    live = w2._session.doc_id_for_source(str(csv.resolve()))
    shown = _fits_text(w2)

    assert _shows_no_fit(shown), (
        f"a fit that records no table is printed beside a table as if it matched: {shown!r}")
    assert "table" in shown and "record" in shown and _says_press_fit(shown), (
        f"nothing says why the saved fit is not shown: {shown!r}")
    # Kept, not discarded: still in the store and still usable by the figure export.
    kept = _stored(w2, live, channel, "fit.dwell")
    assert kept is not None and LINEAGE_KEY not in kept.payload.meta
    fit, why = w2._stored_fit(w2._analysis_panel.current_detection(), "fit.dwell")
    assert fit is not None and why == "", f"the export no longer uses the saved fit: {why!r}"

    _press_fit(w2, qtbot)
    assert _heads(_fits_text(w2)) == [50, 50], _fits_text(w2)
    assert _stored(w2, live, channel, "fit.dwell").payload.meta[LINEAGE_KEY] == "detect.schmitt"
    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_fit_filed_without_a_table_is_not_shown_beside_one(window, qtbot, tmp_path):
    """The Plots window files its own fits into ``fit.dwell`` / ``fit.depth`` when it closes
    (``PlotsWindow._persist_results``, straight into the store, no signal) and records no table.
    Such a fit is no more trustworthy than one from an old project."""
    doc_id, channel = _two_tables(window, qtbot, tmp_path)
    panel = window._analysis_panel
    det = panel.current_detection()
    window._session.results.put(fit_dwell(
        det.payload, doc_id=doc_id, channel=channel, n_boot=20,
        calibration_fingerprint=det.calibration_fingerprint))
    assert isinstance(_stored(window, doc_id, channel, "fit.dwell"), AnalysisResult)
    _pick_table(window, qtbot, "CUSUM")
    _pick_table(window, qtbot, "Schmitt")
    shown = _fits_text(window)

    assert _shows_no_fit(shown), f"a fit with no recorded table is printed: {shown!r}"
    assert "record" in shown and _says_press_fit(shown)
