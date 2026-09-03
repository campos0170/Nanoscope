"""End-to-end Session SAVE / OPEN through the real app path (offscreen, pytest-qt).

A source :class:`MainWindow` is loaded with two synthetic documents written to tmp
CSV files (so they reopen by path), given per-channel calibrations, custom overlay
layers, a Schmitt detection on each — one of them curated (a reject + a hand-edited
event's bounds and a recomputed column) — a stacked view mode, and a couple of
manual measurements. It is saved to a ``.nanoscope`` project.

A FRESH window then opens that project and the restore is asserted on the live
Session + widgets:

  * both documents reopened from disk (as references, not stored signal);
  * each channel's calibration restored (isolated from the sidecar, which is
    deleted first, so the project is provably the source);
  * layers (colour / visibility / time offset) and the view mode restored;
  * both detection EventSets back in the ResultStore with the SAME curation +
    edited bounds + recomputed column (byte-identical frames);
  * the event table + the batched span overlay repopulated, and the result is NOT
    marked stale (its fingerprint matches the restored calibration);
  * the manual measurements restored.

A second test deletes one source file and asserts the restore keeps what it can,
skips the missing document AND its results, and raises a non-blocking warning banner.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import json
import zipfile

from pandas.testing import assert_frame_equal

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
N = 100_000
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    """An in-memory doc whose meta.path IS the tmp CSV, so its doc_id == the reopened one."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = win._session
    doc_id = session.add_document(doc)
    win._create_default_layers(doc)
    ch = next(c.index for c in doc.meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    win._on_selection(doc_id, ch)
    return doc_id, ch


def _detect(win: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _span_items(pane) -> list[SpanOverlayItem]:
    """The DETECTION span overlay on a pane.

    Filtered by the hint's label: the manual-measurement panel now shades its own saved
    measurements with a second (differently coloured) SpanOverlayItem on the same pane, so
    "the only SpanOverlayItem" is no longer the same statement as "the detection's spans".
    """
    return [it for it in pane.plot.items
            if isinstance(it, SpanOverlayItem) and it.label == "events"]


def _fresh_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _live(sess, path) -> str | None:
    """The RESTORED session's doc_id for the recording loaded from ``path``.

    A reopened project cannot be checked with the source window's doc_ids: a doc_id names a
    LOAD (``core/identity.py``), and reopening a file is by construction a different load.
    The identity that crosses a round-trip is the SOURCE — which is exactly what the project
    stores (``DocumentRef.path`` == ``persistent_key``) — so that is what these tests resolve
    through. This is not a workaround for the split; it is the split, asserted.
    """
    return sess.doc_id_for_source(str(Path(path).resolve()))


def _make_source_project(qtbot, tmp_path) -> dict:
    """Build + save a fully-populated source project; return the paths + expected state.

    Uses the in-memory install path for the source window (fast, deterministic) but
    writes the matching CSVs to disk so the RESTORE genuinely reopens by path."""
    csv_a, csv_b = tmp_path / "pore_a.csv", tmp_path / "pore_b.csv"
    arr_a, arr_b = _event_trace(seed=0), _event_trace(seed=1)
    _write_csv(csv_a, arr_a)
    _write_csv(csv_b, arr_b)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv_a, arr_a))
    doc_b, ch_b = _install(win, _doc(csv_b, arr_b))
    sess = win._session

    # Calibrations BEFORE detection, so each result's fingerprint matches its calibration
    # (a restore must come back NOT stale).
    sess.set_calibration(doc_a, ch_a, ChannelCalibration(0.5, 0.0, "pA", Provenance.UNKNOWN))
    sess.set_calibration(doc_b, ch_b, ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN))

    # Detection A + curation: reject event 1, hand-edit event 0's bounds and one recomputed column.
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot)
    panel = win._analysis_panel
    df = sess.results.get(doc_a, ch_a, "detect.schmitt").payload.df
    assert len(df) == len(_EVENTS)
    panel.set_event_accepted(1, False)
    i0 = df.index[df["event_id"].to_numpy() == 0][0]
    df.loc[i0, "start_idx"] = int(df.loc[i0, "start_idx"]) + 17
    df.loc[i0, "end_idx"] = int(df.loc[i0, "end_idx"]) - 9
    df.loc[i0, "depth_pA"] = 42.5          # stand in for the inspector's recompute
    panel.refresh_after_edit(0)
    df_a = df.copy()

    # Detection B (uncurated).
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot)
    df_b = sess.results.get(doc_b, ch_b, "detect.schmitt").payload.df.copy()
    assert len(df_b) == len(_EVENTS)

    # Custom layer view state + non-default view mode.
    sess.set_layer_color(f"{doc_b}#{ch_b}", "#E57373")
    sess.set_layer_visible(f"{doc_b}#{ch_b}", False)
    sess.set_layer_time_offset(f"{doc_b}#{ch_b}", 0.25)
    sess.set_layer_time_offset(f"{doc_a}#{ch_a}", 0.1)
    sess.set_view_mode(ViewMode.STACKED)

    # Two manual measurements on A, through the real cursor tool.
    win._on_selection(doc_a, ch_a)
    mp = panel.measure_panel
    mp.set_active(True)
    vals = sess.x_axis(doc_a).values
    for s, e in _EVENTS[:2]:
        win._trace_view.set_measure_region(float(vals[s]), float(vals[e]))
        qtbot.wait(5)
        assert mp.current_measurement() is not None
        assert mp.add_current() is True
    meas = mp.measurements_df.copy()
    assert len(meas) == 2

    proj = tmp_path / "session.nanoscope"
    win._write_project(str(proj))
    assert proj.exists()

    win._analysis_controller.abort()
    win._controller.abort()

    return {
        "proj": str(proj), "csv_a": str(csv_a), "csv_b": str(csv_b),
        "doc_a": doc_a, "doc_b": doc_b, "ch_a": ch_a, "ch_b": ch_b,
        "df_a": df_a, "df_b": df_b, "meas": meas,
        "n_acc_a": int(df_a["accepted"].to_numpy().sum()),
    }


# ================================================================================================
# the full round trip
# ================================================================================================

def test_save_and_restore_roundtrip(qapp, qtbot, tmp_path):
    ctx = _make_source_project(qtbot, tmp_path)
    ch_a, ch_b = ctx["ch_a"], ctx["ch_b"]

    # Delete the calibration sidecars so a restored calibration can ONLY have come from
    # the project file, not from the .npcal.json the source window wrote next to the data.
    for sidecar in tmp_path.glob("*.npcal.json"):
        sidecar.unlink()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(ctx["proj"])
    qtbot.wait(50)
    sess = w2._session

    # -- both documents reopened from disk ----------------------------------------------------
    doc_a, doc_b = _live(sess, ctx["csv_a"]), _live(sess, ctx["csv_b"])
    assert doc_a is not None and doc_b is not None
    # ...as NEW LOADS. The source window's ids belong to loads that no longer exist, and a
    # session that answered has_document() for them is exactly how an in-flight detection
    # used to land on the wrong recording (core/identity.py).
    assert not sess.has_document(ctx["doc_a"])
    assert not sess.has_document(ctx["doc_b"])
    assert doc_a != ctx["doc_a"] and doc_b != ctx["doc_b"]

    # -- calibrations restored (from the project alone) ---------------------------------------
    ca = sess.calibration(doc_a, ch_a)
    assert ca.gain == 0.5 and ca.target_units == "pA"
    assert sess.calibration(doc_b, ch_b).gain == 2.0

    # -- layers + view mode restored ----------------------------------------------------------
    assert sess.view_mode is ViewMode.STACKED
    lb = sess.layer(f"{doc_b}#{ch_b}")
    assert lb is not None
    assert lb.color == "#E57373"
    assert lb.visible is False
    assert lb.time_offset_s == pytest.approx(0.25)
    la = sess.layer(f"{doc_a}#{ch_a}")
    assert la is not None and la.time_offset_s == pytest.approx(0.1)

    # -- both EventSets back with identical curation + edited bounds + recomputed column ------
    ra = sess.results.get(doc_a, ch_a, "detect.schmitt")
    assert ra is not None
    assert_frame_equal(ra.payload.df, ctx["df_a"])
    assert list(ra.payload.df["accepted"]) == list(ctx["df_a"]["accepted"])
    assert ra.payload.df.loc[ctx["df_a"].index[0], "depth_pA"] == 42.5

    rb = sess.results.get(doc_b, ch_b, "detect.schmitt")
    assert rb is not None
    assert_frame_equal(rb.payload.df, ctx["df_b"])

    # -- the event table + span overlays repopulated (restore landed on A) --------------------
    panel = w2._analysis_panel
    assert panel._sel == (doc_a, ch_a)
    assert panel._table._proxy.rowCount() == len(_EVENTS)
    pane = w2._trace_view.pane_for_layer(f"{doc_a}#{ch_a}")
    assert pane is not None
    spans = _span_items(pane)
    assert len(spans) == 1
    assert spans[0].n_spans == ctx["n_acc_a"]      # only the accepted events are shaded

    # -- restored result is NOT stale (fingerprint matches the restored calibration) ----------
    assert panel._stale.isHidden()

    # -- manual measurements restored ---------------------------------------------------------
    assert_frame_equal(panel.measure_panel.measurements_df, ctx["meas"])

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_colour_registry_save_reopen_drives_trace_and_plot_source(qapp, qtbot, tmp_path):
    """The v10 colour section is wired through the real MainWindow, not only the codec.

    A hand-picked layer ink must enter the Session registry, survive Save/Open under the
    recording's persistent key, and drive both consumers after reopen.  Checking only
    ``Layer.color`` would miss the original defect: that legacy field already round-tripped
    while the Plots surface rebuilt a different colour from source position.
    """
    from nanoscope.core.layer import colour_key
    from nanoscope.core.segment import persistent_key
    from nanoscope.gui.trace_view import layer_ink

    csv = tmp_path / "coloured.csv"
    arr = _event_trace(seed=11)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    layer = win._session.layer(f"{doc_id}#{ch}")
    assert layer is not None
    win._session.set_layer_color(layer.layer_id, "#C13AF0")

    key = colour_key(persistent_key(win._session.document(doc_id)), ch)
    assert win._session.colours.ink(key) == "#C13AF0"

    project = tmp_path / "coloured.nanoscope"
    assert win._write_project(str(project)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    reopened = _fresh_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    qtbot.wait(50)

    live = _live(reopened._session, csv)
    assert live is not None, (
        [str(d.meta.path) for d in reopened._session.documents()],
        reopened._info_bar._label.text(), reopened.statusBar().currentMessage())
    restored = reopened._session.layer(f"{live}#{ch}")
    assert restored is not None
    restored_key = colour_key(persistent_key(reopened._session.document(live)), ch)
    assert restored_key == key
    assert reopened._session.colours.ink(restored_key) == "#C13AF0"
    assert restored.color == "#C13AF0"
    assert layer_ink(reopened._session, restored) == "#C13AF0"

    sources = reopened._plots_sources()
    source = next(s for s in sources if s.key == restored_key)
    assert source.colour == "#C13AF0"

    reopened._analysis_controller.abort()
    reopened._controller.abort()


def test_restore_keeps_concrete_trace_ink_independent_from_registry(qapp, qtbot, tmp_path):
    """A restore must not turn ``layers[].color`` into a registry user override.

    The two fields intentionally disagree here, as they can when a project was written
    with an older dark-pane palette. The trace preserves its exact saved ink while the
    Plots source preserves the independently saved identity slot and user ink.
    """
    from nanoscope.core.layer import ColourRegistry, colour_key
    from nanoscope.core.segment import persistent_key
    from nanoscope.gui.trace_view import layer_ink

    csv = tmp_path / "independent-colours.csv"
    arr = _event_trace(seed=12)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    layer = win._session.layer(f"{doc_id}#{ch}")
    assert layer is not None

    key = colour_key(persistent_key(win._session.document(doc_id)), ch)
    trace_ink = "#4FC3F7"   # a concrete ink from the previous dark-pane palette
    plot_ink = "#C13AF0"    # an independently persisted user ink
    layer.color = trace_ink
    win._session.colours = ColourRegistry(
        slots={key: 5}, inks={key: plot_ink})

    project = tmp_path / "independent-colours.nanoscope"
    assert win._write_project(str(project)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    reopened = _fresh_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    qtbot.wait(50)

    live = _live(reopened._session, csv)
    assert live is not None
    restored = reopened._session.layer(f"{live}#{ch}")
    assert restored is not None
    assert restored.color == trace_ink
    assert layer_ink(reopened._session, restored) == trace_ink
    assert reopened._session.colours.slot(key) == 5
    assert reopened._session.colours.ink(key) == plot_ink

    source = next(s for s in reopened._plots_sources() if s.key == key)
    assert source.colour == plot_ink

    reopened._analysis_controller.abort()
    reopened._controller.abort()


# ================================================================================================
# missing source file -> restore what it can, with a warning
# ================================================================================================

def test_restore_skips_missing_source_file_with_warning(qapp, qtbot, tmp_path):
    ctx = _make_source_project(qtbot, tmp_path)

    # The second document's data file is gone by the time the project is reopened.
    Path(ctx["csv_b"]).unlink()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(ctx["proj"])
    qtbot.wait(50)
    sess = w2._session

    # The reachable document (and its curated detection) is fully restored...
    doc_a = _live(sess, ctx["csv_a"])
    assert doc_a is not None
    assert sess.results.get(doc_a, ctx["ch_a"], "detect.schmitt") is not None
    assert w2._analysis_panel._table._proxy.rowCount() == len(_EVENTS)

    # ...the missing one — and its result — are skipped, not fatal.
    assert _live(sess, ctx["csv_b"]) is None
    assert sess.results.get(ctx["doc_b"], ctx["ch_b"], "detect.schmitt") is None

    # A non-blocking warning banner names the skipped file.
    assert not w2._info_bar.isHidden()
    assert "pore_b" in w2._info_bar._label.text()

    w2._analysis_controller.abort()
    w2._controller.abort()


# ================================================================================================
# v6 — derived fits survive save + reopen (THROUGH THE WIDGETS, not the serialiser)
# ================================================================================================
# MEASURED BEFORE: _project_state() collected only `isinstance(res.payload, EventSet)`, so a
# dwell fit, a depth fit and a CUSUM level table were all dropped on save without a word. The
# round-trip tests in tests/core/test_project.py cover the FORMAT; these cover the WIRING,
# which is where the equivalent density-export defect actually lived — the handler was
# correct and the menu path was not.


def _fittable_trace(seed: int = 7) -> np.ndarray:
    """A trace with enough events to FIT — the fitters refuse fewer than 20.

    The shared ``_event_trace`` carries 5, which is plenty to exercise detection and
    curation but is refused by fit_dwell_exponential/fit_depth_gaussians by design.
    """
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for i in range(40):
        s = 1_000 + i * 2_400
        x[s:s + 120 + int(rng.integers(0, 140))] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _run_fits(win: MainWindow, qtbot) -> None:
    """Press the Fits box's own Fit button, the way the user does."""
    stats = win._analysis_panel._stats
    stats.n_components_dwell.setValue(1)
    stats.n_components_depth.setValue(1)
    stats.fit_button.click()
    qtbot.wait(20)


def test_a_dwell_and_depth_fit_survive_save_and_reopen(qapp, qtbot, tmp_path):
    """The fits come back with their numbers AND a working density.

    The curve is the part that was supposed to be impossible: a FitResult's ``_pdf`` is a
    closure. It is rebuilt on load by stats.density_from(), the same function the fitter
    used, so the reopened curve is the one its own tau/AIC describes — asserted pointwise.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _fittable_trace()
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    _run_fits(win, qtbot)

    sess = win._session
    before_dwell = sess.results.get(doc_a, ch_a, "fit.dwell")
    before_depth = sess.results.get(doc_a, ch_a, "fit.depth")
    assert before_dwell is not None and before_depth is not None, "the Fit button ran no fit"

    proj = tmp_path / "fits.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    s2 = w2._session
    live = _live(s2, csv)
    assert live is not None

    after_dwell = s2.results.get(live, ch_a, "fit.dwell")
    after_depth = s2.results.get(live, ch_a, "fit.depth")
    assert after_dwell is not None, "the dwell fit was dropped on save/reopen"
    assert after_depth is not None, "the depth fit was dropped on save/reopen"

    assert after_dwell.payload.params == before_dwell.payload.params
    assert after_dwell.payload.aic == before_dwell.payload.aic
    assert after_dwell.payload.model == before_dwell.payload.model

    # ...and the CURVE, pointwise. This is the assertion the whole design exists to allow.
    t = np.geomspace(1e-4, 1e-1, 200)
    assert np.array_equal(after_dwell.payload.predict(t), before_dwell.payload.predict(t))
    v = np.linspace(0.01, 0.9, 200)
    assert np.array_equal(after_depth.payload.predict(v), before_depth.payload.predict(v))

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_cusum_level_table_survives_save_and_reopen(qapp, qtbot, tmp_path):
    """``fit.levels`` is a DataFrame payload; it must come back frame-for-frame."""
    from pandas.testing import assert_frame_equal

    csv = tmp_path / "pore_a.csv"
    arr = _event_trace(seed=0)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    win._act_fit_levels.trigger()          # THE MENU PATH
    qtbot.wait(50)

    sess = win._session
    before = sess.results.get(doc_a, ch_a, "fit.levels")
    assert before is not None, "Analysis > Fit sub-state levels produced no result"
    levels = before.payload.copy()

    proj = tmp_path / "levels.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    live = _live(w2._session, csv)
    after = w2._session.results.get(live, ch_a, "fit.levels")
    assert after is not None, "the level table was dropped on save/reopen"
    assert_frame_equal(after.payload, levels)

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_reopened_fit_still_reads_as_stale_when_the_calibration_moved(qapp, qtbot, tmp_path):
    """The staleness stamp must come from the FILE.

    A fit computed at gain 1 and reopened after a re-gain has to keep saying so. Re-deriving
    the stamp from the live calibration on load is what made every reopened result look
    freshly-computed — the defect C6 fixed for detections, which a new persistence path is
    in an excellent position to reintroduce.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _fittable_trace()
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    win._session.set_calibration(
        doc_a, ch_a, ChannelCalibration(1.0, 0.0, "pA", Provenance.UNKNOWN))
    _detect(win, qtbot)
    _run_fits(win, qtbot)
    stamp = win._session.results.get(doc_a, ch_a, "fit.dwell").calibration_fingerprint

    # The gain moves AFTER the fit — the stated workflow, not an edge case.
    win._session.set_calibration(
        doc_a, ch_a, ChannelCalibration(4.0, 0.0, "pA", Provenance.UNKNOWN))
    proj = tmp_path / "stale.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    live = _live(w2._session, csv)
    after = w2._session.results.get(live, ch_a, "fit.dwell")
    assert after is not None
    assert after.calibration_fingerprint == stamp, "the stamp was re-derived, not restored"
    assert after.calibration_fingerprint != w2._session.calibration(live, ch_a).fingerprint(), (
        "a fit reopened after a gain change must still read as stale")

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_restored_fit_is_VISIBLE_in_the_fits_readout(qapp, qtbot, tmp_path):
    """Persisted is not the same as shown.

    A fit that comes back into the session but never reaches the Fits box is, from the
    user's chair, still missing — and that gap (correct handler, silent UI) is exactly
    where the density-export defect lived. So this asserts the text, not the object.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _fittable_trace()
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    _run_fits(win, qtbot)
    model = win._session.results.get(doc_a, ch_a, "fit.dwell").payload.model

    proj = tmp_path / "visible.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)

    shown = w2._analysis_panel._stats.results.toPlainText()
    assert model in shown, (
        f"the reopened project's Fits box does not show the restored fit; it reads {shown!r}")

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_restored_level_table_is_REACHABLE_from_the_menu(qapp, qtbot, tmp_path):
    """Persisting the CUSUM levels only put them in the store; nothing re-attached them.

    MEASURED BEFORE: after reopening, session.results held the fit.levels frame in full, and
    Analysis > Export sub-state levels... was greyed out — invoking it answered "Run
    Analysis > Fit sub-state levels (CUSUM)... first". The user's only way out was to re-run
    the CUSUM pass, i.e. exactly the cost persisting it was supposed to remove. The level
    table is also the one derived result that is NOT re-derivable from the saved event table
    alone, so this was the most expensive thing to lose.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _event_trace(seed=0)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    win._act_fit_levels.trigger()
    qtbot.wait(50)
    assert win._act_export_levels.isEnabled()
    levels = win._session.results.get(doc_a, ch_a, "fit.levels").payload.copy()

    proj = tmp_path / "reach.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)

    assert w2._act_export_levels.isEnabled(), (
        "the restored level table is in the session but the export action is still greyed out")
    assert w2._level_fit is not None
    assert_frame_equal(w2._level_fit.payload, levels)

    # ...and the menu path actually writes, rather than refusing.
    out = tmp_path / "lv.csv"
    from nanoscope.gui import main_window as mw
    orig = mw.ask_export_path
    mw.ask_export_path = lambda *a, **k: out
    try:
        w2._act_export_levels.trigger()
        qtbot.wait(20)
    finally:
        mw.ask_export_path = orig
    assert out.exists(), "the menu path wrote no level export"

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_level_table_retired_by_a_re_detection_is_not_persisted_at_all(qapp, qtbot, tmp_path):
    """The safety property that makes the previous test sound.

    A level table belongs to the exact event table it was fitted on. Re-running detection
    does not supersede it, it RETIRES it — and there is no replacement to put in its place.
    Before this change _invalidate_level_fit dropped only this window's handle and left the
    dead entry in the ResultStore, which was harmless only for as long as nothing read the
    store. Project save reads the store. So the retired table was written into the file, and
    the moment restored levels became reachable it would have come back attached to a
    detection it was never measured on, its level rows naming event_ids that now mean
    different events — silently, and exported under that detection's provenance.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _event_trace(seed=0)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv, arr))
    _detect(win, qtbot, min_dwell_s=1e-5)
    win._act_fit_levels.trigger()
    qtbot.wait(50)
    assert win._session.results.get(doc_a, ch_a, "fit.levels") is not None

    # Re-detect with settings that change which events exist at all.
    _detect(win, qtbot, min_dwell_s=2e-3)
    qtbot.wait(20)
    assert win._level_fit is None, "the app must retire a level fit whose detection changed"
    assert win._session.results.get(doc_a, ch_a, "fit.levels") is None, (
        "a retired level fit must leave the RESULT STORE too, or save will persist it")

    proj = tmp_path / "retired.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()
    win._controller.abort()

    import zipfile as _zip
    with _zip.ZipFile(proj) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert not [f for f in manifest.get("fits", []) if f["module_id"] == "fit.levels"], (
        "a retired level table was written into the project file")

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    assert w2._level_fit is None
    assert not w2._act_export_levels.isEnabled()

    w2._analysis_controller.abort()
    w2._controller.abort()


# ================================================================================================
# a save may not decide that UNREADABLE means UNWANTED
# ================================================================================================

def test_saving_after_a_partial_restore_keeps_the_unreadable_recording(qapp, qtbot, tmp_path):
    """MEASURED BEFORE: docs 2 -> 1, results 2 -> 1, "Saved project (1 result(s))", no dialog.

    A project referencing two recordings, one of whose files is not readable at open time,
    restores one document and warns about the other — correctly. The old Ctrl+S then rewrote
    the file from the LIVE session, which holds only what loaded, and the second recording's
    reference, calibration, layer state and whole curated detection left the project. The
    commonest cause of the original failure is an unmounted drive or a file moved for a
    moment — exactly the case where nothing is wrong with the work. Recovery had been "put
    the file back and reopen"; after that save there was nothing left to reopen.
    """
    ctx = _make_source_project(qtbot, tmp_path)
    proj = Path(ctx["proj"])
    with zipfile.ZipFile(proj) as zf:
        before = json.loads(zf.read("manifest.json"))
    assert len(before["documents"]) == 2 and len(before["results"]) == 2

    Path(ctx["csv_b"]).unlink()          # the second recording is unreadable at open time

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    assert len(w2._session.documents()) == 1       # a genuinely partial restore...
    assert not w2._info_bar.isHidden()             # ...that says so

    assert w2._save_project() is True              # Ctrl+S
    qtbot.wait(20)

    with zipfile.ZipFile(proj) as zf:
        after = json.loads(zf.read("manifest.json"))
    assert len(after["documents"]) == 2, "the unreadable recording was dropped from the project"
    assert len(after["results"]) == 2, "its curated detection was dropped with it"
    assert {d["path"] for d in after["documents"]} == {d["path"] for d in before["documents"]}
    assert "kept in the project" in w2.statusBar().currentMessage()

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_the_kept_recording_restores_whole_once_its_file_is_back(qapp, qtbot, tmp_path):
    """The point of keeping it: put the file back, reopen, and the project is whole again —
    including the curated detection that was never in the live session while the file was
    away. Preserving the reference is only worth anything if it still resolves."""
    ctx = _make_source_project(qtbot, tmp_path)
    proj = Path(ctx["proj"])
    csv_b = Path(ctx["csv_b"])
    stashed = csv_b.read_bytes()
    csv_b.unlink()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    assert w2._save_project() is True
    w2._analysis_controller.abort()
    w2._controller.abort()

    csv_b.write_bytes(stashed)           # the drive is mounted again

    w3 = _fresh_window(qtbot)
    with qtbot.waitSignal(w3.projectRestored, timeout=30_000):
        w3._open_project(str(proj))
    qtbot.wait(50)
    live_b = _live(w3._session, csv_b)
    assert live_b is not None, "the recording did not come back"
    res = w3._session.results.get(live_b, ctx["ch_b"], "detect.schmitt")
    assert res is not None, "its detection did not come back"
    assert len(res.payload.df) == len(_EVENTS)

    w3._analysis_controller.abort()
    w3._controller.abort()


def test_a_recording_the_user_deliberately_closed_is_still_dropped(qapp, qtbot, tmp_path):
    """The other half, and the one that keeps this from becoming a leak: carrying through
    what could NOT BE LOADED must not resurrect what the user CLOSED on purpose. Only the
    unreadable set is kept, never merely "absent from the session"."""
    ctx = _make_source_project(qtbot, tmp_path)
    proj = Path(ctx["proj"])

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    assert len(w2._session.documents()) == 2       # both files present: nothing carried
    assert w2._carried_over is None

    live_b = _live(w2._session, ctx["csv_b"])
    w2._session.close_document(live_b)             # closed deliberately
    qtbot.wait(20)
    assert w2._save_project() is True
    qtbot.wait(20)

    with zipfile.ZipFile(proj) as zf:
        after = json.loads(zf.read("manifest.json"))
    assert len(after["documents"]) == 1, "a deliberately closed recording must not be kept"
    assert "kept in the project" not in w2.statusBar().currentMessage()

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_a_segment_whose_parent_file_is_away_is_kept_with_its_parent(qapp, qtbot, tmp_path):
    """The transitive half. A segment is stored as a SPEC and re-derived from its parent, so
    a parent that could not be loaded means the segment could not be re-derived either —
    neither is in the live session, and the old save dropped both plus every result keyed
    under them. Carrying the parent reference is also what keeps the segment RESTORABLE: the
    save-time drop check resolves a segment against the documents in the project, and with
    the parent kept it still resolves, so no "cannot be saved" warning fires either.
    """
    csv = tmp_path / "parent.csv"
    arr = _event_trace(seed=0)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    parent_id, ch = _install(win, _doc(csv, arr))
    seg_id = win._session.create_segment(parent_id, 1_000, 60_000, name="crop A")
    win._create_default_layers(win._session.document(seg_id))
    win._on_selection(seg_id, 0)
    _detect(win, qtbot)
    assert win._session.results.get(seg_id, 0, "detect.schmitt") is not None

    proj = tmp_path / "seg.nanoscope"
    win._write_project(str(proj))
    with zipfile.ZipFile(proj) as zf:
        before = json.loads(zf.read("manifest.json"))
    assert len(before["segments"]) == 1 and len(before["documents"]) == 1
    win._analysis_controller.abort()
    win._controller.abort()

    csv.unlink()                      # the parent recording's file goes away

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    assert len(w2._session.documents()) == 0     # neither parent nor segment could load

    # Saving must not silently empty the project — and must not need a drop warning either.
    asked: list = []
    w2._confirm_save_dropping_segments = lambda ns, nr: (asked.append((ns, nr)), True)[1]
    assert w2._save_project() is True
    qtbot.wait(20)
    assert asked == [], f"a kept segment must not be reported as unsaveable: {asked}"

    with zipfile.ZipFile(proj) as zf:
        after = json.loads(zf.read("manifest.json"))
    assert len(after["documents"]) == 1, "the parent reference was dropped"
    assert len(after["segments"]) == 1, "the segment spec was dropped with its parent"
    assert len(after["results"]) == 1, "the segment's detection was dropped"

    w2._analysis_controller.abort()
    w2._controller.abort()
