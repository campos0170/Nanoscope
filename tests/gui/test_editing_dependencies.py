"""Every input a table was made with is judged on its own, and the fits, the Plots window and
Export figure ask the same judge (docs/EDITING_FOUNDATION_DESIGN.md 7.1 and 7.2, decision 10).

THE GAPS (design 1.4), each measured on 27ec0eb:

* **Events → fits.** A fit was judged by its accepted count alone, so a re-measure that moved
  every ``frac_depth`` at an unchanged count left the depth fit printed as current on the Fits
  tab, its mu and 95 % interval those of values the table no longer holds. A fit made through
  the panel also recorded none of its table's stamps: it was handed the bare EventSet, and the
  open-pore estimator's stamp lives on the stored result.
* **One headline, one input.** The amber line names the first stamp that moved, in its order of
  precedence (rate, calibration, estimator). A table stale on two counts said one, and nothing
  on screen compared the form's chain and thresholds with the run beside it.
* **Plots window.** It marked a superseded calibration and nothing else -- its own docstring
  said so: a re-declared sample rate, and a changed estimator on the table on screen, drew the
  same numbers with no mark.
* **Export figure.** Only the fit guard ran. A table on a clock the recording no longer has was
  written as figures in seconds the trace does not have, and a table the line calls stale on
  its calibration was written under "Wrote 4 figure(s)" and nothing else.

THE FIX. A fit pressed in the panel is handed the stored result, so it records the table's
rate and estimator stamps beside the digest of what it fitted (``modules._stamp_fit_input``);
the Fits tab says "the fitted events changed since the fit" where the counts agree and the
digest does not. The amber line keeps its headline and its precedence, and its Details pane
lists every input -- rate, calibration, estimator, chain, thresholds -- as made → now, with the
action that repairs it (``analysis.lineage``). The Plots window and Export figure ask the
Results panel's own verdict (``AnalysisPanel.table_staleness``): the rate of every table, the
estimator of the table on screen, whose form it is judged against.

These tests drive the real MainWindow on synthetic recordings whose ``meta.path`` is under
``tmp_path``: the Fit button, Re-measure, a calibration, a rate declaration, the Baseline form,
the Details button, Analysis ▸ Plots and Export figure. They assert what the user reads.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
#: 4 s: room for every estimator window the tests set.
N = 200_000
I0 = -1_000.0
#: 24 blockades -- the fits refuse fewer than 20 -- of 4 to 22 ms and 250 to 480 pA.
SPANS = [(5_000 + 7_500 * i, 5_000 + 7_500 * i + 200 + 40 * i) for i in range(24)]

#: The first line of each fit's ``summary()``: what tells the two fits apart in the Fits box.
DWELL_MODEL = "exponential (left-truncated MLE)"
DEPTH_MODEL = "Gaussian mixture on dI/I0"
#: The five inputs, as the Details list names them.
TITLES = ("Sample rate:", "Calibration:", "Open-pore estimator:", "Signal chain:", "Thresholds:")


# ================================================================================================
# builders
# ================================================================================================

def _signal(seed: int = 0) -> np.ndarray:
    """The blockades on a slow 20 pA drift. On a flat open pore every estimator window finds
    the same baseline under each event, and a re-measure would move no value at all."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = I0 + 20.0 * np.sin(2.0 * np.pi * t / 3.0) + rng.normal(0.0, 8.0, N)
    for i, (s, e) in enumerate(SPANS):
        x[s:e] -= 250.0 + 10.0 * i
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(w: MainWindow, tmp_path: Path, name: str = "pore.dat", *, seed: int = 0,
          rate: float | None = FS, declare: float | None = None) -> str:
    """Open a synthetic recording at ``tmp_path / name``, put it on screen and detect on it
    through the detector module with the form's own parameters, so the table carries every
    stamp a run gives it (rate, calibration, estimator) and starts current. ``rate=None`` is a
    file that declares no rate; ``declare`` declares one before the run."""
    data = _signal(seed)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=rate,
                         rate_provenance=Provenance.FILE_META if rate else Provenance.UNKNOWN,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(data,))
    session = w._session
    doc_id = session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    if declare is not None:
        _declare(w, doc_id, declare)
    session.add_result(DetectSchmitt().run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta,
        w._analysis_panel._form_params())))
    QtWidgets.QApplication.processEvents()
    res = w._analysis_panel.current_detection()
    assert res is not None and len(res.payload.df) == len(SPANS), "precondition: every blockade"
    assert w._analysis_panel._stale.isHidden(), "precondition: the table starts current"
    return doc_id


def _declare(w: MainWindow, doc_id: str, hz: float | None) -> None:
    """Inspector ▸ File Info ▸ Sample rate / time unit… (``None`` hands it back to the file)."""
    w._session.set_sample_rate_declaration(
        doc_id, None if hz is None else SampleRateDeclaration("sample_rate", sample_rate_hz=hz))
    QtWidgets.QApplication.processEvents()


def _set_gain(w: MainWindow, doc_id: str, gain: float) -> None:
    w._session.set_calibration(doc_id, 0, ChannelCalibration(gain=gain, offset=0.0,
                                                             target_units="pA"),
                               write_sidecar=False)
    QtWidgets.QApplication.processEvents()


def _press_fit(w: MainWindow) -> None:
    """The Fits tab's own Fit button, the way the user presses it."""
    w._analysis_panel._stats.fit_button.click()
    from .fit_test_helpers import wait_for_fit
    wait_for_fit(w._analysis_panel._stats)
    QtWidgets.QApplication.processEvents()


def _fits_text(w: MainWindow) -> str:
    return w._analysis_panel._stats.results.toPlainText()


def _details(w: MainWindow) -> str:
    """Open the amber line's Details, as a click does, and read the list of inputs in it."""
    banner = w._analysis_panel._stale
    assert not banner.isHidden(), "precondition: the amber line is up"
    banner._details.click()          # the suite's popup net closes it again
    QtWidgets.QApplication.processEvents()
    assert banner._details_pane.isAncestorOf(banner._inputs), "the list is in the Details pane"
    return banner.inputs_text()


def _pump(qtbot, panel, timeout: int = 60_000) -> None:
    """Wait for the off-thread re-measure to land."""
    qtbot.waitUntil(lambda: panel._pending is None, timeout=timeout)
    QtWidgets.QApplication.processEvents()


def _accepted(w: MainWindow) -> int:
    return int(w._analysis_panel.current_detection().payload.df["accepted"].to_numpy(
        dtype=bool).sum())


# ================================================================================================
# events -> fits (decision 10)
# ================================================================================================

def test_a_re_measure_that_keeps_the_count_stales_the_depth_fit_it_moved(window, qtbot,
                                                                          tmp_path):
    """The fit is judged by the values it fitted, not by how many there were. A re-measure
    under another estimator window keeps every event and moves every ``frac_depth``: the depth
    fit no longer describes the table and says so; the dwell fit, whose ``dwell_s`` did not
    move, is still printed. On 27ec0eb the depth fit read current."""
    _open(window, tmp_path)
    panel = window._analysis_panel
    _press_fit(window)
    text = _fits_text(window)
    assert DWELL_MODEL in text and DEPTH_MODEL in text, f"precondition: both fits current\n{text}"
    before = panel.current_detection().payload.df["frac_depth"].to_numpy().copy()
    n_before = _accepted(window)

    panel.baseline_panel.base_s.setValue(0.2)
    QtWidgets.QApplication.processEvents()
    assert panel._stale.kind == "baseline", "precondition: the line offers Re-measure"
    assert panel.remeasure_active_detection() == "started"
    _pump(qtbot, panel)

    after = panel.current_detection().payload.df["frac_depth"].to_numpy()
    assert _accepted(window) == n_before, "precondition: the count is unchanged"
    assert not np.array_equal(before, after), "precondition: the re-measure moved the depths"
    text = _fits_text(window)
    assert DEPTH_MODEL not in text, (
        f"the depth fit still reads current after a re-measure moved every value it fitted:\n"
        f"{text}")
    assert ("The stored depth fit predates a change to this Schmitt table: the fitted events "
            "changed since the fit (swapped, resized or re-measured) — press Fit to fit it as "
            "it stands.") in text, text
    assert DWELL_MODEL in text, "the dwell fit's values did not move; it is still current"
    assert "accepted events and the table has" not in text, (
        "the count sentence would describe an unchanged count as the change")


def test_a_fit_pressed_in_the_panel_records_its_tables_estimator_and_rate(window, tmp_path):
    """Each fit states what its table was made from, as the table states it: the estimator
    stamp (on the stored result) and the rate (in its provenance), under the fit's
    ``FitResult.meta`` keys. The keys are pinned by their literal spelling: a project file
    carries them."""
    doc_id = _open(window, tmp_path)
    _press_fit(window)
    table = window._analysis_panel.current_detection()
    assert table.baseline_fingerprint, "precondition: the table carries its estimator stamp"
    fits = [window._session.results.get(doc_id, 0, mid) for mid in ("fit.dwell", "fit.depth")]
    for fit in fits:
        assert fit is not None, "a fit did not land"
        meta = fit.payload.meta
        assert meta.get("baseline_fingerprint") == table.baseline_fingerprint, (
            f"{fit.module_id} does not record its table's estimator: {sorted(meta)}")
        assert meta.get("fs") == pytest.approx(FS)

    from nanoscope.analysis.lineage import made_with

    for fit in fits:
        made = made_with(fit)
        assert (made.baseline_fingerprint, made.rate_hz) == (table.baseline_fingerprint, FS)


# ================================================================================================
# the amber line's Details: every input, as made -> now, with its action
# ================================================================================================

@pytest.mark.parametrize("change", ["calibration", "rate", "estimator"])
def test_details_lists_every_input_as_made_and_now_with_its_action(window, tmp_path, change):
    """The sentence the docs quote is kept, first; under it, the five inputs, each with its own
    verdict, the changed one as made → now with the button that repairs it."""
    doc_id = _open(window, tmp_path)
    panel = window._analysis_panel
    if change == "calibration":
        _set_gain(window, doc_id, 2.0)
        rows = ["• Calibration: gain 1, offset 0 → gain 2, offset 0 — Re-run"]
        kind = "calibration"
    elif change == "rate":
        _declare(window, doc_id, 25_000.0)
        rows = ["• Sample rate: 50 kHz → 25 kHz — Re-run"]
        kind = "sample_rate"
    else:
        panel.baseline_panel.rough_s.setValue(2.0)
        QtWidgets.QApplication.processEvents()
        rows = ["• Open-pore estimator: masked median, rough 500 ms / base 100 ms",
                "→ masked median, rough 2000 ms / base 100 ms",
                "— Re-measure depths"]
        kind = "baseline"
    banner = panel._stale
    assert banner.kind == kind, "precondition: the headline names the input that moved"
    sentence = banner.message()

    text = _details(window)

    assert text.startswith("Inputs, as made → now:"), text
    for title in TITLES:
        assert f"• {title}" in text, f"{title} is not listed:\n{text}"
    for row in rows:
        assert row in text, f"{row!r} is not in Details:\n{text}"
    if change != "calibration":
        assert "• Calibration: gain 1, offset 0 — unchanged" in text, text
    if change != "rate":
        assert "• Sample rate: 50 kHz — unchanged" in text, text
    assert "• Signal chain: unchanged" in text and "• Thresholds: unchanged" in text, text
    assert banner.message() == sentence == banner._label.text(), "the sentence is kept as is"
    assert banner._summary.toolTip() == sentence, "the line's tooltip is the sentence alone"


def test_a_table_stale_on_two_counts_lists_both_under_one_headline(window, tmp_path):
    """The headline keeps its precedence and names the calibration; Details names the
    estimator too, with its own repair, where the line used to stop at the first."""
    doc_id = _open(window, tmp_path)
    panel = window._analysis_panel
    _set_gain(window, doc_id, 2.0)
    panel.baseline_panel.rough_s.setValue(2.0)
    QtWidgets.QApplication.processEvents()
    assert panel._stale.kind == "calibration"
    assert panel._stale.action_text() == "Re-run"

    text = _details(window)

    assert "• Calibration: gain 1, offset 0 → gain 2, offset 0 — Re-run" in text, text
    estimator = next(line for line in text.splitlines() if "Open-pore estimator:" in line)
    assert "→" in estimator and estimator.endswith("— Re-measure depths"), estimator


def test_a_rate_the_recording_no_longer_knows_is_declared_before_the_re_run(window, tmp_path):
    """Handed back to a file that declares no rate, the recording has no clock to re-run on:
    the rate's row says to declare it first, as the line's own sentence ends."""
    doc_id = _open(window, tmp_path, "rateless.dat", rate=None, declare=FS)
    _declare(window, doc_id, None)
    assert window._session.document(doc_id).meta.sample_rate_hz is None
    assert window._analysis_panel._stale.kind == "sample_rate"

    text = _details(window)

    assert ("• Sample rate: 50 kHz → unknown — Declare the rate (Inspector ▸ File Info ▸ "
            "Sample rate / time unit…), then Re-run") in text, text


def test_a_changed_open_pore_reference_offers_a_fresh_detection(window, tmp_path):
    """A changed reference is not re-measured: noise, polarity and the candidates themselves
    came from the old one. The estimator's row offers what the line's button does."""
    _open(window, tmp_path)
    panel = window._analysis_panel
    editor = panel.baseline_panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText("-1000")
    QtWidgets.QApplication.processEvents()
    assert panel._stale.action_text() == "Run detection", "precondition: the line's own button"

    text = _details(window)

    estimator = next(line for line in text.splitlines() if "Open-pore estimator:" in line)
    assert estimator.endswith("— Run detection"), estimator


def test_a_manual_table_lists_rebuild_as_the_repair(window, tmp_path):
    """A Manual table is never re-run or re-measured; every row repairs with Rebuild, as its
    line does, and it has no chain of its own to compare with the form."""
    doc_id = _open(window, tmp_path)
    panel = window._analysis_panel
    selector = panel._result_selector
    selector.setCurrentIndex(selector.findData(MANUAL_MODULE_ID))     # created empty, shown
    QtWidgets.QApplication.processEvents()
    assert panel.current_detection().module_id == MANUAL_MODULE_ID
    _set_gain(window, doc_id, 2.0)
    assert panel._stale.action_text() == "Rebuild", "precondition: the line's own button"

    text = _details(window)

    assert "• Calibration: gain 1, offset 0 → gain 2, offset 0 — Rebuild" in text, text
    assert ("• Signal chain: not compared — the Schmitt/CUSUM chain did not make this "
            "result.") in text, text


def test_the_list_is_read_again_each_time_details_opens(window, tmp_path):
    """A threshold edit re-judges the chain notice, not the amber line; the list under the
    line must not describe the form as it stood before the edit."""
    doc_id = _open(window, tmp_path)
    panel = window._analysis_panel
    _set_gain(window, doc_id, 2.0)
    assert "• Thresholds: unchanged" in _details(window)

    k_in = panel.detection_panel.k_in
    k_in.setValue(k_in.value() + 1.0)
    QtWidgets.QApplication.processEvents()

    text = _details(window)
    thresholds = next(line for line in text.splitlines() if "Thresholds:" in line)
    assert "k_in" in thresholds and "→" in thresholds, thresholds
    assert thresholds.endswith("— Revert, or re-run to apply"), thresholds


# ================================================================================================
# the Plots window asks the same judge
# ================================================================================================

def _plots(w: MainWindow):
    """Analysis ▸ Plots, every source ticked, a histogram."""
    w._act_plots.trigger()
    QtWidgets.QApplication.processEvents()
    plots = w._plots_window
    plots.set_sources([s.label for s in plots.sources()])
    plots.set_plot_type("Histogram")
    return plots


def _headline(plots) -> str:
    return str(plots._title_presentation[2] or "")


def test_the_plots_window_marks_a_re_declared_rate_on_a_table_not_on_screen(window, tmp_path):
    """Recording B's rate is re-declared while A is on screen: B's time columns are marked in
    Details and in the title note, in the amber line's own sentence, and A's are not."""
    a = _open(window, tmp_path, "a.dat")
    b = _open(window, tmp_path, "b.dat", seed=1)
    window._on_selection(a, 0)
    QtWidgets.QApplication.processEvents()
    _declare(window, b, 25_000.0)
    plots = _plots(window)

    plots.set_histogram_column("dwell_s")

    text = plots.details_text()
    assert "b.dat: STALE: dwell_s on this plot" in text, text
    assert "The sample rate changed since this detection ran (50,000 Hz → 25,000 Hz)" in text
    assert "a.dat: STALE" not in text, text
    assert "⚠ STALE: dwell_s" in _headline(plots), _headline(plots)

    plots.set_histogram_column("depth_pA")      # a rate does not move a depth
    assert "STALE" not in plots.details_text()
    assert "STALE" not in _headline(plots)


def test_the_plots_window_marks_the_table_on_screen_after_a_baseline_change(window, tmp_path):
    """The estimator is judged against the form, and the form is the table on screen's: A's
    depths are marked, B's -- off screen, its own form not on screen -- are not."""
    a = _open(window, tmp_path, "a.dat")
    _open(window, tmp_path, "b.dat", seed=1)
    window._on_selection(a, 0)
    QtWidgets.QApplication.processEvents()
    panel = window._analysis_panel
    panel.baseline_panel.rough_s.setValue(2.0)
    QtWidgets.QApplication.processEvents()
    assert panel._stale.kind == "baseline", "precondition: the line says the estimator moved"
    plots = _plots(window)

    plots.set_histogram_column("depth_pA")

    text = plots.details_text()
    assert "a.dat: STALE: depth_pA on this plot" in text, text
    assert "The open-pore estimator changed since this detection ran" in text, text
    assert "b.dat: STALE" not in text, text
    assert "⚠ STALE: depth_pA" in _headline(plots), _headline(plots)
    assert "SUPERSEDED CALIBRATION" not in text, "the calibration did not move"


# ================================================================================================
# Export figure asks the same judge
# ================================================================================================

@pytest.fixture
def asked(monkeypatch, tmp_path):
    """Every export dialog the door raised; each is answered with ``tmp_path / 'figs.pdf'``."""
    titles: list[str] = []

    def ask(*_args, **kwargs):
        titles.append(kwargs.get("title", ""))
        return tmp_path / "figs.pdf"

    monkeypatch.setattr(mw, "ask_export_path", ask)
    return titles


def test_export_figure_refuses_a_table_on_another_clock(window, tmp_path, asked):
    """As the event exports refuse it: before any dialog, on the status line, nothing
    written."""
    doc_id = _open(window, tmp_path)
    _declare(window, doc_id, 25_000.0)
    assert window._analysis_panel._stale.kind == "sample_rate"

    window._export_figures()

    said = window.statusBar().currentMessage()
    assert said.startswith("Nothing was exported — re-run detection under the declared rate"), said
    assert "50,000 Hz" in said and "25,000 Hz" in said, said
    assert asked == [], "no dialog for figures that cannot be written"
    assert not list(tmp_path.glob("figs_*")), "a refused export writes nothing"


@pytest.mark.parametrize("change", ["calibration", "estimator"])
def test_export_figure_names_a_stale_table_in_its_status_line(window, tmp_path, asked, change):
    """A table stale on its calibration or its estimator stays exportable; the line that
    reports the export names it, in the amber line's sentence."""
    doc_id = _open(window, tmp_path)
    panel = window._analysis_panel
    if change == "calibration":
        _set_gain(window, doc_id, 2.0)
        sentence = "Calibration changed since this detection ran"
    else:
        panel.baseline_panel.rough_s.setValue(2.0)
        QtWidgets.QApplication.processEvents()
        sentence = "The open-pore estimator changed since this detection ran"
    assert not panel._stale.isHidden(), "precondition: the amber line is up"

    window._export_figures()

    said = window.statusBar().currentMessage()
    assert said.startswith("Wrote "), said
    assert list(tmp_path.glob("figs_*.pdf")), "the figures were written"
    assert "⚠ They draw the Schmitt table of “pore.dat”, which is stale: " + sentence in said, said


def test_export_figure_of_a_current_table_names_nothing(window, tmp_path, asked):
    _open(window, tmp_path)

    window._export_figures()

    said = window.statusBar().currentMessage()
    assert said.startswith("Wrote "), said
    assert "stale" not in said.lower(), said


def test_details_is_measured_again_each_time_it_opens(qapp, qtbot):
    """The Details popup is built once and kept, and a QMenu keeps the height it had the first
    time it opened. The Inputs list changes with every open (an input moved, an input put
    back), so a list that had grown ran past the bottom of the popup and its last rows were cut
    off. Every open measures the popup again."""
    from nanoscope.gui.analysis_panel import _StaleBanner

    holder = QtWidgets.QWidget()
    qtbot.addWidget(holder)
    banner = _StaleBanner(lambda kind: None)
    QtWidgets.QVBoxLayout(holder).addWidget(banner)
    holder.resize(700, 120)
    holder.show()
    rows = {"n": 2}
    banner.set_inputs_source(lambda: "\n".join(
        f"{TITLES[i % len(TITLES)]} as made {i} -> now {i + 1}, Re-run to apply"
        for i in range(rows["n"])))
    banner.show_message("Depths stale: the sample rate changed since this table was detected.")
    menu, pane = banner._details_menu, banner._details_pane

    for n in (2, 9, 4, 12):
        rows["n"] = n
        banner._open_details()
        assert pane.parentWidget() is menu, "the pane sits in the popup"
        bottom = (pane.mapTo(menu, QtCore.QPoint(0, 0)).y()
                  + banner._inputs.mapTo(pane, QtCore.QPoint(0, 0)).y() + banner._inputs.height())
        assert bottom <= menu.height(), (
            f"{n} input rows: the list ends at {bottom} px, the popup is {menu.height()} px tall")
        menu.close()
