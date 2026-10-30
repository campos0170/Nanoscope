"""AREA C+D — the read-outs that worked and presented a number badly.

One file, one theme: **a number on screen must carry what a reader needs to check it.**

* **High-5** — ``frac_depth`` was shown as "median dI/I0" at 1.81-2.57 with no flag. A
  value above 1 means the current MAGNITUDE grew; it is not a blockade fraction.
* **Medium-15** — the detection Summary omitted all three things the app's own rules
  require to travel with a depth: the depth in pA, the open-pore current it was
  measured against, and the ``baseline_fingerprint``.
* **Medium-8** — two surfaces report different open-pore currents for one recording
  (I-V window vs the event table's stamped ``baseline_pA``; −49.13 vs −51.78 pA at
  −700 mV) with no cross-reference.
* **Medium-9** — the Baseline health block was populated only from a stored
  detection's provenance, so the six checks appeared only AFTER the run they were
  meant to justify. The confirmation click was made blind.
* **Medium-16** — the event table led with two constant full-width text columns, so
  at a realistic dock width no measured column was visible.
* **Medium-17** — precision / recall / F1 read 1.000 straight after detection, scored
  against the detector's own uncurated output.
* **Low-22** — "Merge all" ran 3.3-8.7 s on the GUI thread with no progress bar, no
  busy cursor and no status message.
* **Low-24** — a health check contradicted its own message: "overshoot 0.000
  (… healthy corpus values 0.09-0.11)" rendered green, because the check is one-sided
  and never said so.
* **Low-25** — event span overlays were invisible at full-recording zoom: at 250 s
  extent every 20 ms span is sub-pixel, so 0 events and 1196 events looked identical.
* **Low-26** — the Summary table showed 5.5 of its rows behind a scrollbar.
* **Low-27** — the Schmitt tab had no range control while the ML tab did, unexplained.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.scoring import ScoreResult, ensure_added_column
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import EventTable, ScorePanel, StatsPanel

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
SPANS = [(10_000, 10_600), (20_000, 20_800), (30_000, 30_500), (40_000, 40_700)]


# ================================================================================================
# builders
# ================================================================================================
def _signal(spans=SPANS, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in spans:
        x[s:e] -= DEPTH
    return x.astype(np.float32)


def _doc(tmp_path, signal, name="apo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _open(win, tmp_path, signal=None, name="apo.dat"):
    signal = _signal() if signal is None else signal
    doc_id = win._session.add_document(_doc(tmp_path, signal, name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4,
                              recording=name, pore_id="pore-1")
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id, signal


def _open_measured(win, tmp_path, name="apo.dat"):
    """Like :func:`_open`, but letting the characterisation baseline estimator run, so
    the detection carries a real ``baseline_fingerprint`` in its provenance."""
    signal = _signal()
    doc_id = win._session.add_document(_doc(tmp_path, signal, name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, min_dwell_s=1e-4,
                              recording=name, pore_id="pore-1")
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id, signal


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _summary(win, key: str) -> str:
    return win._analysis_panel._stats.summary_text(key)


# ================================================================================================
# High-5 — dI/I0 above 1 is not a blockade fraction
# ================================================================================================
@pytest.mark.parametrize("value,flagged", [
    (2.357, True),     # the -500 mV merged median, from the report
    (2.574, True),     # -600 mV
    (1.900, True),     # -700 mV
    (1.0001, True),
    (1.0, False),      # exactly 1 is not "grew"
    (0.42, False),
    (float("nan"), False),
    (None, False),
])
def test_the_enhancement_note_fires_exactly_above_one(value, flagged):
    note = StatsPanel.frac_depth_note(value)
    assert bool(note) is flagged
    if flagged:
        assert "not a blockade fraction" in note
        assert "GREW" in note


def test_the_summary_flags_a_di_i0_above_one_with_a_glyph_not_only_colour(win, tmp_path):
    """A SCREENSHOT IS HOW THIS NUMBER LEAVES THE APP. Colour alone survives neither a
    colour-blind reader nor a greyscale print, and it was the screenshot that was going
    to be read as "a 250% blockade"."""
    panel = win._analysis_panel
    _open(win, tmp_path)
    panel._stats.set_summary({"median_frac_depth": 2.357})
    assert "⚠ >1" in _summary(win, "median_frac_depth")
    assert "not a blockade fraction" in panel._stats.summary_tooltip("median_frac_depth")


def test_a_normal_blockade_fraction_gets_no_warning(win, tmp_path):
    _open(win, tmp_path)
    win._analysis_panel._stats.set_summary({"median_frac_depth": 0.42})
    assert "⚠" not in _summary(win, "median_frac_depth")


def test_the_event_table_column_says_what_a_value_above_one_means(win, tmp_path):
    _open(win, tmp_path)
    model = win._analysis_panel._table._model
    col = list(model.df.columns).index("frac_depth")
    tip = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole)
    assert "ABOVE 1" in tip and "GREW" in tip


def test_the_gallery_caption_is_clean_and_the_footer_explains_enhancement(win, tmp_path):
    """Artwork retains the value; its meaning is disclosed outside the tile."""
    from nanoscope.gui.event_gallery import EventGalleryWindow

    _open(win, tmp_path)
    win._open_event_gallery()
    gal = win._event_gallery
    assert isinstance(gal, EventGalleryWindow)
    gal.sort_box.setCurrentText("dI/I0 (deepest first)")

    row = pd.Series({"event_id": 412, "frac_depth": 2.36, "depth_pA": 90.0, "dwell_s": 0.02})
    assert gal.caption_for(row) == "412 · 2.36"
    row_ok = pd.Series({"event_id": 412, "frac_depth": 0.36, "depth_pA": 90.0, "dwell_s": 0.02})
    assert gal.caption_for(row_ok) == "412 · 0.36"
    page = gal._df.copy(deep=True)
    page.loc[page.index[0], "frac_depth"] = 2.36
    gal.set_events(gal._doc_id, gal._channel, page)
    assert "dI/I0 > 1" in gal.status.text()
    assert "not a blockade fraction" in gal.status.text()


def test_the_exported_sidecar_carries_the_same_warning_the_summary_shows(win, tmp_path):
    """THE OTHER HALF OF HIGH-5: the copy that leaves the app.

    The Summary flags a dI/I0 above 1 on screen; the CSV a co-author actually receives
    said nothing, and a ``frac_depth`` column of 2.36 under a header spelled
    "frac_depth" is exactly how that becomes "a 236% blockade" in someone else's notes.
    An ADDITIVE PROVENANCE KEY in ``<stem>.meta.json`` — the channel the frozen
    20-column schema leaves open — beside the ``baseline_fingerprint`` a reader is
    already there to find.
    """
    import json

    from nanoscope.gui import main_window as mw
    from nanoscope.gui.panels import StatsPanel

    _open(win, tmp_path)
    df = win._analysis_panel.current_detection().payload.df
    df["frac_depth"] = 2.36

    dest = tmp_path / "ev.csv"
    original, mw.ask_export_path = mw.ask_export_path, lambda *a, **k: dest
    try:
        win._export_events()
    finally:
        mw.ask_export_path = original

    meta = json.loads((tmp_path / "ev.meta.json").read_text())
    note = meta["provenance"]["frac_depth_note"]
    assert "ABOVE 1" in note and "GREW" in note
    # ONE OWNER OF THE WORDING. The file and the screen must not describe the same
    # number differently, so the sentence is the panel's, verbatim.
    assert note == StatsPanel.frac_depth_note(2.36)
    # ...and the schema is untouched: the note went in provenance, not into a column.
    assert "frac_depth_note" not in pd.read_csv(dest).columns


def test_an_empty_export_makes_no_claim_and_raises_no_warning(win, tmp_path):
    """Everything rejected is a path the user reaches on purpose. No rows, no median,
    no note — and no numpy "Mean of empty slice" warning from asking for one."""
    import warnings

    import pandas as _pd

    empty = _pd.DataFrame({"frac_depth": _pd.Series([], dtype="float64")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prov = MainWindow._export_provenance({"baseline_fingerprint": "sha256:x"}, empty)
    assert prov == {"baseline_fingerprint": "sha256:x"}


def test_a_normal_blockade_export_carries_no_warning(win, tmp_path):
    """The control. A note that fires on every export is a note nobody reads."""
    import json

    from nanoscope.gui import main_window as mw

    _open(win, tmp_path)
    win._analysis_panel.current_detection().payload.df["frac_depth"] = 0.42

    dest = tmp_path / "ev.csv"
    original, mw.ask_export_path = mw.ask_export_path, lambda *a, **k: dest
    try:
        win._export_events()
    finally:
        mw.ask_export_path = original

    meta = json.loads((tmp_path / "ev.meta.json").read_text())
    assert "frac_depth_note" not in meta["provenance"]


# ================================================================================================
# Medium-15 / Medium-8 — a depth does not travel alone
# ================================================================================================
def test_the_summary_carries_the_depth_the_open_pore_and_the_fingerprint(win, tmp_path):
    """All three used to be reachable only by scrolling the event table sideways or by
    exporting the provenance sidecar.

    Detected WITHOUT a hand-supplied baseline, so the characterisation estimator really
    runs and really stamps a ``baseline_fingerprint`` — the value the whole finding is
    about."""
    _open_measured(win, tmp_path)
    df = win._analysis_panel.current_detection().payload.df

    assert float(_summary(win, "median_depth_pA")) == pytest.approx(
        float(np.nanmedian(df["depth_pA"].to_numpy())), rel=1e-3)
    assert float(_summary(win, "median_baseline_pA")) == pytest.approx(
        float(np.nanmedian(df["baseline_pA"].to_numpy())), rel=1e-3)
    fp = win._analysis_panel.current_detection().payload.provenance["baseline_fingerprint"]
    assert _summary(win, "baseline_fingerprint") == str(fp)
    assert "rough" in _summary(win, "baseline_window")


def test_the_three_medians_are_taken_over_the_same_rows(win, tmp_path):
    """depth / |baseline| must reproduce the dI/I0 beside it, or the Summary is three
    numbers that cannot be checked against each other."""
    _open_measured(win, tmp_path)
    depth = float(_summary(win, "median_depth_pA"))
    base = float(_summary(win, "median_baseline_pA"))
    frac = float(_summary(win, "median_frac_depth"))
    assert depth / abs(base) == pytest.approx(frac, rel=0.05)


def test_the_fingerprint_comes_from_the_stored_run_not_from_the_live_form(win, tmp_path):
    """The form holds what the NEXT run would use. After a parameter edit those are
    different windows, and showing the form's numbers beside the stored table's depths
    would state a provenance the numbers do not have."""
    _open_measured(win, tmp_path)
    shown = _summary(win, "baseline_fingerprint")
    win._analysis_panel._baseline.rough_s.setValue(
        win._analysis_panel._baseline.rough_s.value() * 2.0)
    win._analysis_panel._fill_summary(win._analysis_panel.current_detection().payload)
    assert _summary(win, "baseline_fingerprint") == shown


def test_an_old_detection_with_no_baseline_keys_says_so_rather_than_nothing(win, tmp_path):
    _open_measured(win, tmp_path)
    res = win._analysis_panel.current_detection()
    stripped = EventSet(res.payload.df, {
        k: v for k, v in res.payload.provenance.items() if not str(k).startswith("baseline_")})
    win._analysis_panel._fill_summary(stripped)
    assert _summary(win, "baseline_fingerprint") == "not recorded"
    assert _summary(win, "baseline_window") == "not recorded"


def test_the_open_pore_row_cross_references_the_iv_window(win, tmp_path):
    """MEDIUM-8. Two surfaces call a number "the open pore" and they are not the same
    number (−49.13 vs −51.78 pA at −700 mV). Neither said so."""
    _open(win, tmp_path)
    tip = win._analysis_panel._stats.summary_tooltip("median_baseline_pA")
    assert "I-V" in tip and "not this number" in tip.lower()
    assert "5.1%" in tip


def test_the_baseline_column_cross_references_it_too(win, tmp_path):
    _open(win, tmp_path)
    model = win._analysis_panel._table._model
    col = list(model.df.columns).index("baseline_pA")
    tip = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole)
    assert "I-V" in tip


# ================================================================================================
# Low-26 — every Summary row visible
# ================================================================================================
def test_the_summary_table_shows_every_row_without_a_scrollbar():
    panel = StatsPanel()
    try:
        table = panel.summary_table
        assert table.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
        wanted = table.horizontalHeader().height() + 2 * table.frameWidth() + sum(
            table.rowHeight(i) for i in range(table.rowCount()))
        assert table.height() == wanted
        assert table.rowCount() == len(StatsPanel._SUMMARY_ROWS)
    finally:
        panel.deleteLater()


# ================================================================================================
# Medium-17 — 1.000 by construction is not a score
# ================================================================================================
def _score(**kw) -> ScoreResult:
    base = dict(n_detected=5, n_truth=5, tp=5, fp=0, fn=0,
                precision=1.0, recall=1.0, f1=1.0)
    base.update(kw)
    return ScoreResult(**base)


@pytest.mark.parametrize("kw,tautological", [
    ({}, True),                                              # fresh detection
    (dict(tp=3, fp=2, precision=0.6, f1=0.667), False),      # something rejected
    (dict(tp=5, fn=1, n_truth=6, recall=0.833, f1=0.909), False),   # something added
    (dict(n_detected=0, tp=0, precision=None, recall=None, f1=None), False),
])
def test_the_ratios_are_withheld_exactly_while_the_truth_is_the_detectors_output(
        kw, tautological):
    assert ScorePanel.is_tautological(_score(**kw)) is tautological


def test_the_readout_withholds_the_ratios_and_says_why(win, tmp_path):
    _open(win, tmp_path)
    score = win._analysis_panel._score
    for key in ("precision", "recall", "f1"):
        assert score.value_text(key) == "not scored yet"
    assert "by construction" in score.tautology_label.text()
    # ...and the COUNTS, which are real measurements, are still shown.
    assert score.value_text("n_detected") == str(len(SPANS))
    assert score.value_text("tp") == str(len(SPANS))


def test_rejecting_one_event_turns_the_ratios_back_on(win, tmp_path):
    _open(win, tmp_path)
    panel = win._analysis_panel
    ids = [int(i) for i in panel.current_detection().payload.df["event_id"].to_numpy()]
    panel.set_event_accepted(ids[0], False)
    score = panel._score
    assert score.value_text("precision") == "0.750"
    assert score.tautology_label.text() == ""


# ================================================================================================
# Medium-16 — a constant column carries no per-row information
# ================================================================================================
def _visual_order(table: EventTable) -> list[str]:
    header = table.view.horizontalHeader()
    cols = list(table._model.df.columns)
    return [cols[header.logicalIndex(v)] for v in range(header.count())]


def test_the_constant_identity_columns_are_demoted_and_stated_once(win, tmp_path):
    _open(win, tmp_path)
    table = win._analysis_panel._table
    order = _visual_order(table)
    # The scientific default order owns placement now, not a move to the far right:
    # both identity columns are painted after every scientific column, so neither of
    # them leads the table.
    last_scientific = max(order.index(n) for n in EventTable._SCIENTIFIC_ORDER if n in order)
    cols = list(table._model.df.columns)
    for name in EventTable._IDENTITY_COLUMNS:
        assert order.index(name) > last_scientific, order
        # Demoted, NOT hidden.
        assert not table.view.isColumnHidden(cols.index(name)), name
    # ...and the value they carried is stated, once, above the table.
    caption = table.constants_label.text()
    assert "recording = apo.dat" in caption and "pore_id = pore-1" in caption, caption
    assert caption.count("recording =") == 1 and caption.count("pore_id =") == 1, caption
    assert not table.constants_label.isHidden()


def test_a_measured_column_is_now_visible_at_a_realistic_dock_width(win, tmp_path):
    """THE POINT OF THE DEMOTION, IN PIXELS. Measured against the schema order it
    replaced: the two identity columns are full-width strings (a file stem and a pore
    id) in visual positions 1 and 2 of twenty, and every pixel they take is a pixel
    before the first measured column. The scientific default order does the demoting
    now, and it rearranges more than these two columns, so the saving is a floor."""
    _open(win, tmp_path, name="a_recording_with_a_realistic_file_name.dat")
    table = win._analysis_panel._table
    header = table.view.horizontalHeader()
    cols = list(table._model.df.columns)

    def _width_to_depth(order):
        upto = order.index("depth_pA")
        return sum(header.sectionSize(cols.index(n)) for n in order[:upto + 1])

    now = _width_to_depth(_visual_order(table))
    schema_order = _width_to_depth(cols)          # what the table used to paint
    assert now < schema_order
    # ...and the saving is at least the whole width of the two demoted columns, not a
    # rounding.
    assert schema_order - now >= sum(
        header.sectionSize(cols.index(n)) for n in EventTable._IDENTITY_COLUMNS)
    # No leading text-identity column survives ahead of the first measured one.
    order = _visual_order(table)
    assert not set(EventTable._IDENTITY_COLUMNS) & set(order[:order.index("depth_pA")])


def test_a_pore_id_that_actually_varies_stays_a_visible_column():
    """A table pooling two pores has a REAL pore_id column. Hiding it, or captioning it as
    a constant, would be hiding data, not tidying a constant. Where it is painted is the
    default scientific order's business since c86b119: after the measured columns, which
    the owner accepted (2026-10-30) over moving a varying identity column back in front."""
    table = EventTable()
    try:
        df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
        ensure_added_column(df)
        df["pore_id"] = pd.Series(
            ["A" if i % 2 else "B" for i in range(len(df))], dtype="string")
        table.set_events(df)
        cols = list(table._model.df.columns)
        assert not table.view.isColumnHidden(cols.index("pore_id"))
        assert "pore_id" not in table.constants_label.text()
    finally:
        table.deleteLater()


def test_the_demotion_is_visual_only_and_the_frame_is_untouched(win, tmp_path):
    """The 20-column schema is frozen — names, order, dtypes. Reordering what is
    PAINTED must not reorder anything else."""
    from nanoscope.analysis.events import SCHEMA

    _open(win, tmp_path)
    df = win._analysis_panel._table.dataframe
    assert list(df.columns)[:len(SCHEMA)] == list(SCHEMA)


# ================================================================================================
# Low-24 — a check that does not state its criterion cannot be read
# ================================================================================================
def _rendered(panel) -> str:
    """The health block AS QT DRAWS IT — through the same rich-text parser QLabel uses.

    NOT ``panel.health_label.text()``, which is the SOURCE. That distinction is the whole
    of Low-24's second life: the criterion was interpolated raw, ``<= 0.3)</span>`` parsed
    as a malformed tag, and Qt deleted the threshold on its way to the pixel while the
    source still contained it. A guard test written against the source went green over a
    defect a reader could see. Rendered here rather than via ``health_text()`` so this
    test measures the pixel no matter what that accessor happens to return.
    """
    from PySide6.QtGui import QTextDocument

    doc = QTextDocument()
    doc.setHtml(panel.health_label.text())
    return doc.toPlainText()


def _overshoot_health(value: float, level: str = "ok") -> dict:
    return {"checks": [{
        "name": "overshoot", "level": level, "label": "overshoot", "value": value,
        "threshold": "<= 0.3",
        "message": (f"overshoot {value:.3f} (241 samples above +4 sigma, 1,061,865 "
                    f"below; healthy corpus values 0.09-0.11)."),
    }]}


def test_a_green_check_states_the_measurement_and_what_would_have_failed_it():
    """THE CONTRADICTION. "overshoot 0.000 (… healthy corpus values 0.09-0.11)" was
    coloured green with nothing to reconcile the two, because the check flags only HIGH
    overshoot and never said so. The threshold was in ``BaselineCheck`` all along and
    the panel dropped it.

    ASSERTED ON WHAT IS RENDERED, NOT ON THE MARKUP. The first version of this test
    read the rich-text SOURCE and accepted ``"passes &lt;= 0.3" or "passes <= 0.3"``,
    so it went green while Qt was deleting the threshold on its way to the screen:
    the label is RichText, ``<= 0.3)</span>`` parses as a malformed tag, and the
    rendered line came out ``(measured 0.0002269, passes — overshoot 0.000 …``. The
    criterion never reached a reader's eye on the one check this finding is about.
    ``health_text()`` is now the rendered text; ``health_html()`` is the source.
    """
    panel = BaselinePanel()
    try:
        panel.set_health(_overshoot_health(0.0002269))
        text = _rendered(panel)
        assert "PASS" in text
        assert "passes <= 0.3" in text, text
        assert "measured 0.0002269" in text
        # ...and the panel's own accessor answers with what is rendered, so a future
        # assertion written against it cannot repeat the mistake this test made.
        assert panel.health_text() == text
    finally:
        panel.deleteLater()


@pytest.mark.parametrize("threshold", ["<= 0.3", "<= 2x", "0.025 .. 0.25", ">= 1.0 x 0.5 s"])
def test_every_threshold_form_survives_the_rich_text_parser(threshold):
    """The invariant, over the forms `analysis/baseline.py` actually emits.

    Three of the seven live thresholds begin with ``<``; four do not. Only the three
    were destroyed, which is why spot-checking the mask-fraction line ("passes
    0.025 .. 0.25", intact) looked like proof that the fix worked. A check states its
    criterion or it does not; that cannot depend on the first character.
    """
    panel = BaselinePanel()
    try:
        panel.set_health({"checks": [{
            "name": "x", "level": "ok", "label": "a check", "value": 0.5,
            "threshold": threshold, "message": "a message."}]})
        assert f"passes {threshold}" in _rendered(panel), _rendered(panel)
    finally:
        panel.deleteLater()


def test_a_failing_check_says_WARNING_in_words_not_only_in_orange():
    panel = BaselinePanel()
    try:
        panel.set_health(_overshoot_health(0.9, level="warn"))
        assert "WARNING" in panel.health_text()
    finally:
        panel.deleteLater()


# ================================================================================================
# Medium-9 — the checks are there BEFORE the confirmation click
# ================================================================================================
def test_the_health_block_says_whether_it_describes_a_proposal_or_a_detection():
    panel = BaselinePanel()
    try:
        panel.set_health(_overshoot_health(0.0002), measured_for="proposal")
        assert panel.health_scope() == "proposal"
        assert "PROPOSED" in panel.health_scope_label.text()
        panel.set_health(_overshoot_health(0.0002), measured_for="detection")
        assert panel.health_scope() == "detection"
        assert "detection on screen" in panel.health_scope_label.text()
    finally:
        panel.deleteLater()


def test_the_probe_measures_the_health_of_what_it_proposes(win, qtbot, tmp_path):
    """THE CONFIRMATION IS NO LONGER BLIND. The six checks used to be computed only by
    a DETECTION and read back out of its provenance — so they appeared after the run
    they were supposed to inform, on a tab the user had left."""
    _open(win, tmp_path)
    panel = win._analysis_panel
    panel._baseline.set_health(None)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel._baseline.probeRequested.emit()
    qtbot.wait(20)

    assert panel._baseline.proposal() is not None
    assert panel._baseline.health() is not None, "a proposal with no verdict is a blind click"
    assert panel._baseline.health_scope() == "proposal"
    names = {c[0] for c in BaselinePanel._health_checks(panel._baseline.health())}
    assert names <= {"ok", "warn", "info", "refuse"}
    assert len(BaselinePanel._health_checks(panel._baseline.health())) >= 4


def test_a_bare_proposal_from_an_older_session_still_lands(win, tmp_path):
    """The probe job's answer grew a second element. A stored/legacy answer that is
    still a bare BaselineProposal must not be dropped on the floor."""
    from nanoscope.analysis.baseline import DEFAULT_BASELINE
    from nanoscope.analysis.baseline_probe import BaselineProposal

    doc_id, _ = _open(win, tmp_path)
    panel = win._analysis_panel
    panel._pending, panel._pending_sel = "probe", (doc_id, 0)
    panel._on_controller_finished(
        BaselineProposal(config=DEFAULT_BASELINE, param_source="default", reason="legacy"))
    assert panel._baseline.proposal() is not None


# ================================================================================================
# Low-27 — the asymmetry with the ML tab, explained
# ================================================================================================
def test_the_schmitt_tab_states_its_range_and_how_to_get_a_smaller_one(win, tmp_path):
    _open(win, tmp_path)
    readout = win._analysis_panel._detection.range_readout
    assert readout.text() == "whole recording"
    tip = readout.toolTip()
    assert "0.24x realtime" in tip          # WHY the ML tab has a choice and this does not
    assert "Segment" in tip                 # ...and what to do instead
    assert "baseline_fingerprint" in tip    # ...and why that is the right answer


# ================================================================================================
# Low-22 — a job that costs seconds says it is running
# ================================================================================================
def test_merge_all_returns_immediately_and_reports_when_it_lands(win, qtbot, tmp_path):
    _open(win, tmp_path, _signal([(10_000, 10_600), (10_640, 11_200), (30_000, 30_500)]))
    panel = win._analysis_panel
    with qtbot.waitSignal(panel.mergeFinished, timeout=30_000) as landed:
        started = panel.merge_all_subevents(5e-3)
    assert started.status == "started"
    assert landed.args[0].status in {"ok", "noop"}


def test_the_click_puts_a_sentence_and_the_cancel_button_on_screen(win, qtbot, tmp_path):
    """A long job with a progress bar and no words is still a window that has stopped
    answering; a long job with neither is a hang."""
    _open(win, tmp_path, _signal([(10_000, 10_600), (10_640, 11_200), (30_000, 30_500)]))
    panel = win._analysis_panel
    panel.merge_controls.set_window_s(5e-3)
    with qtbot.waitSignal(panel.mergeFinished, timeout=30_000):
        panel.merge_controls.merge_all_btn.click()
        # ...said BEFORE the work, which is the whole point.
        assert "Merging" in win.statusBar().currentMessage()
        assert "Cancel" in win.statusBar().currentMessage()
        assert not win._progress.isHidden()
    qtbot.wait(20)
    assert "→" in win.statusBar().currentMessage()


def test_merge_all_runs_off_the_gui_thread():
    """Pinned structurally as well as behaviourally: a future edit that moved the work
    back inline would keep every behavioural test above green (they would simply see a
    signal emitted synchronously) while restoring the freeze."""
    import inspect

    from nanoscope.gui.analysis_panel import AnalysisPanel

    src = inspect.getsource(AnalysisPanel.merge_all_subevents)
    assert "run_callable" in src
    assert "frozen_for_dispatch" in src


def test_a_merge_whose_table_moved_under_it_is_dropped_out_loud(win, qtbot, tmp_path):
    """The merged table is built from the dispatch-time frame. Installing it over rows
    the user curated while it ran would silently undo the curation — and unlike a
    re-measure there is no surviving row to carry the decision onto, because a merge
    deletes rows."""
    _open(win, tmp_path, _signal([(10_000, 10_600), (10_640, 11_200), (30_000, 30_500)]))
    panel = win._analysis_panel
    with qtbot.waitSignal(panel.mergeFinished, timeout=30_000) as landed:
        assert panel.merge_all_subevents(5e-3).status == "started"
        ids = [int(i) for i in panel.current_detection().payload.df["event_id"].to_numpy()]
        panel.set_event_accepted(ids[-1], False)      # curated DURING the job
    qtbot.wait(20)
    assert landed.args[0].status == "dropped"
    assert "discarded" in win.statusBar().currentMessage()
    # ...and the curation the user made is still there.
    df = panel.current_detection().payload.df
    assert not bool(df.loc[df["event_id"] == ids[-1], "accepted"].iloc[0])


def test_merge_all_refuses_while_another_analysis_holds_the_controller(win, qtbot, tmp_path):
    _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.remeasure_active_detection() == "started"
    report = panel.merge_all_subevents(5e-3)
    assert report.status == "busy"
    win._post_merge_status(report)
    assert "still running" in win.statusBar().currentMessage()
    qtbot.waitUntil(lambda: panel._pending is None, timeout=60_000)


def test_the_region_and_inspector_merges_stay_synchronous(win, tmp_path):
    """Their cost is bounded by a selection or one forward run, not by the table. A
    controller job in front of them would buy nothing and cost the Event Inspector's
    'merge, then advance' its immediacy."""
    _open(win, tmp_path, _signal([(10_000, 10_600), (10_640, 11_200), (30_000, 30_500)]))
    panel = win._analysis_panel
    ids = [int(i) for i in panel.current_detection().payload.df["event_id"].to_numpy()]
    report = panel.merge_event_with_next(ids[0], 5e-3)
    assert report.status == "ok"       # the answer IS the return value here


# ================================================================================================
# Low-25 — an overlay drawn and invisible is not drawn
# ================================================================================================
@pytest.fixture
def plot(qapp):
    """A live 800x300 PlotItem. Kept alive for the whole test: pyqtgraph's ViewBox is a
    C++ object, and letting the widget fall out of scope mid-test raises from inside
    pyqtgraph rather than failing the assertion under test."""
    import pyqtgraph as pg

    widget = pg.PlotWidget()
    widget.resize(800, 300)
    widget.show()
    QtWidgets.QApplication.processEvents()
    yield widget.getPlotItem()
    widget.hide()
    widget.setParent(None)


def _painted_rects(item):
    """The rects ``paint`` actually issues, captured through a recording painter."""
    rects: list[QtCore.QRectF] = []

    class _Recorder:
        def setPen(self, *_a): pass
        def setBrush(self, *_a): pass
        def drawRect(self, rect): rects.append(QtCore.QRectF(rect))

    item.paint(_Recorder(), None, None)
    return rects


def test_a_sub_pixel_span_is_still_painted_one_pixel_wide(plot):
    """At the -500 mV recording's full 250 s extent a 20 ms event is 8e-5 of the width.
    Every one of the 1196 spans rounded away, and "detection finished" looked identical
    whether 0 events or 1196 had been found."""
    from nanoscope.gui.overlays import SpanOverlayItem

    item = SpanOverlayItem(np.array([100.0]), np.array([100.02]))   # 20 ms of 250 s
    plot.addItem(item, ignoreBounds=True)
    plot.vb.setXRange(0.0, 250.0, padding=0)
    plot.vb.setYRange(-1.0, 1.0, padding=0)
    QtWidgets.QApplication.processEvents()

    px = float(plot.vb.viewPixelSize()[0])
    assert px > 0.02, "fixture must actually be sub-pixel"
    rects = _painted_rects(item)
    assert len(rects) == 1
    assert rects[0].width() == pytest.approx(px, rel=1e-6)
    # ...centred on the event, not shifted off it: a comb of 1196 spans must stay where
    # the events are.
    assert rects[0].center().x() == pytest.approx(100.01, abs=px)


def test_a_span_wider_than_a_pixel_is_left_exactly_alone(plot):
    """Zoomed in, this must be a no-op: the geometry the user measures against is the
    geometry that is drawn."""
    from nanoscope.gui.overlays import SpanOverlayItem

    item = SpanOverlayItem(np.array([100.0]), np.array([100.02]))
    plot.addItem(item, ignoreBounds=True)
    plot.vb.setXRange(99.9, 100.1, padding=0)
    plot.vb.setYRange(-1.0, 1.0, padding=0)
    QtWidgets.QApplication.processEvents()

    px = float(plot.vb.viewPixelSize()[0])
    assert px < 0.02, "fixture must actually be wider than a pixel"
    rects = _painted_rects(item)
    assert len(rects) == 1
    assert rects[0].x() == pytest.approx(100.0)
    assert rects[0].width() == pytest.approx(0.02)


# ================================================================================================
# A RATE NAMES THE ROWS IT COUNTED
# ================================================================================================
# `EventSet.rate_hz()` divides ALL rows by the analysed duration. The Summary shows
# "n accepted" one row above the rate, so the two disagreed the moment anything was
# curated out. Measured on the −600 mV merged detection: 240 rows, 239 accepted (one
# `clipped_by_boundary`), Summary 4.008 Hz against the 3.992 Hz the report's master table
# and the Voltage-dependence panel both quote. Three surfaces, two denominators, and
# nothing on screen saying which was which.

def test_the_summary_rate_is_over_accepted_rows_and_says_so(win, tmp_path):
    _open(win, tmp_path)
    panel = win._analysis_panel._stats
    panel.set_summary({"n_events": 240, "n_accepted": 239,
                       "rate_hz": 240 / 59.876, "rate_accepted_hz": 239 / 59.876})
    labels = [lbl for lbl, _key in StatsPanel._SUMMARY_ROWS]
    assert "rate (Hz, accepted)" in labels, labels
    shown = panel.summary_text("rate_accepted_hz")
    assert "3.992" in shown, shown
    tip = panel.summary_tooltip("rate_accepted_hz")
    assert "NOT n events / duration" in tip and "3.992" in tip and "4.008" in tip, tip


def test_the_two_denominators_are_the_same_number_when_nothing_is_rejected(win, tmp_path):
    """The control: with every row accepted the distinction costs nothing, which is why
    it went unnoticed on the two recordings where it does not bite."""
    _open(win, tmp_path)
    panel = win._analysis_panel._stats
    panel.set_summary({"n_events": 1196, "n_accepted": 1196,
                       "rate_hz": 1196 / 250.053, "rate_accepted_hz": 1196 / 250.053})
    assert "4.783" in panel.summary_text("rate_accepted_hz")


# ================================================================================================
# High-5, THE OTHER SURFACES — a note about a COLUMN cannot live in one panel
# ================================================================================================
# `StatsPanel.frac_depth_note`'s own docstring said "the note belongs to the QUANTITY and
# not to this table". It then had exactly two consumers — the Summary row and the CSV
# sidecar — and could have no more: the cross-panel import guard means no sibling panel can
# reach a classmethod on this one. So the Event Inspector printed `dI/I0 2.36` bare on
# every event, the Measure panel's live read-out did the same, and `plot/figures.py`
# captioned its depth histogram "blockade depth ΔI/I₀" over a distribution centred at 2.4 —
# the figure a co-author is most likely to actually receive.
#
# The sentence now lives in `analysis.events`, beside the column's schema.

def test_the_sentence_has_one_home_and_the_panel_delegates_to_it():
    from nanoscope.analysis import events as events_mod

    assert StatsPanel.frac_depth_note(2.357) == events_mod.frac_depth_note(2.357)
    assert StatsPanel.frac_depth_note(0.42) == "" == events_mod.frac_depth_note(0.42)
    # The inline glyph shares the threshold with the sentence, so they cannot disagree.
    assert events_mod.frac_depth_flag(2.357) == " ⚠>1"
    assert events_mod.frac_depth_flag(1.0) == ""


def test_the_event_inspector_marks_enhancements_in_details_not_the_title(win, tmp_path):
    """Event/state readouts retain safeguards; artwork titles remain clean."""
    import inspect

    from nanoscope.gui import event_inspector

    cls = event_inspector.EventInspectorWindow
    assert "frac_depth_flag(" in inspect.getsource(cls._refresh_readout)
    assert "frac_depth_flag(" in inspect.getsource(cls._level_lines)
    assert "frac_depth_flag(" not in inspect.getsource(cls._event_title)


def test_the_measure_readout_marks_an_enhancement(win, tmp_path):
    _open(win, tmp_path)
    panel = win._analysis_panel.measure_panel
    rows = dict((key, i) for i, (_lbl, key) in enumerate(panel._READOUT_ROWS))
    panel._render_readout({"frac_amplitude": 2.357})
    item = panel._readout_table.item(rows["frac_amplitude"], 0)
    assert "⚠>1" in item.text(), item.text()
    assert "not a blockade fraction" in item.toolTip(), item.toolTip()

    panel._render_readout({"frac_amplitude": 0.42})
    item = panel._readout_table.item(rows["frac_amplitude"], 0)
    assert "⚠" not in item.text(), item.text()


def test_the_publication_depth_figure_does_not_call_an_enhancement_a_blockade():
    """A neutral quantity label stays on the figure; its caveat stays in details."""
    import numpy as np
    import pandas as pd

    from nanoscope.plot.figures import depth_histogram, figure_details

    enhanced = pd.DataFrame({"frac_depth": np.linspace(1.8, 3.0, 400)})
    _fig, ax = depth_histogram(enhanced, fit=False)
    assert ax.get_xlabel() == r"$\Delta I / I_0$"
    assert "ENHANCEMENT" not in ax.get_xlabel()
    assert any("ABOVE 1" in note for note in figure_details(_fig)["plots"][0]["details"])

    blockade = pd.DataFrame({"frac_depth": np.linspace(0.05, 0.6, 400)})
    _fig, ax = depth_histogram(blockade, fit=False)
    assert "blockade depth" in ax.get_xlabel(), ax.get_xlabel()
    assert "ENHANCEMENT" not in ax.get_xlabel()
