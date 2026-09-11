"""The two FIGURES in the subpopulation report window — items 6-7 and item 13.

``test_subpop_gui.py`` covers the path from the menu item to a report object. What
is at stake HERE is the last step, which until this pass did not exist: the report
already carried the t-SNE coordinates ("EXPORTED and never printed") and the
per-subpopulation median waveforms ("as a table a plot can be drawn from"), and
nothing drew either. These tests are about the drawing, and they are written around
one rule:

    **NO NUMBER MOVES.** Both views are renderings of an existing report object. So
    the assertions are mostly EQUALITY against the report's own arrays — the points
    on screen are ``report.embeddings`` exactly, the curves are the columns of
    ``report.waveforms`` exactly, the colours come from the ONE shared map. A view
    that quietly re-derived, re-scaled, re-sorted or re-binned anything would fail
    here, which is the point: this window's whole claim is that it adds ink and not
    arithmetic.

The second theme is the one the window's docstring argues: this pipeline's numbers
come with caveats that are not decoration (uncalibrated bootstrap intervals with a
label-switching signature, BIC disagreeing with the CV about K), and four tidy
coloured blobs are exactly the figure that makes a reader forget them. So there are
tests that the caveats cannot be hidden — no tab widget, the strip on screen beside
the figures with every warning verbatim.

The third is colour. A second categorical palette is this tree's signature failure,
so ``gui.categorical`` is the only map and tests pin that BOTH callers use it, that
it is a leaf, and that a subpopulation is the same ink in the scatter and in the
waveform overlay.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import dataclasses
import pathlib
import re

import numpy as np
import pyqtgraph as pg
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet
from nanoscope.gui import categorical as CAT
from nanoscope.gui.subpop_window import (
    EmbeddingScatterView,
    SubpopReportWindow,
    WaveformOverlayView,
)

FS = 50_000.0
N_EVENTS = 220
N_SAMPLES = 400_000

_GMM = "GMM subpopulation (the mixture)"
_HDBSCAN = "HDBSCAN (exploratory)"
_KMEANS = "k-means, k=3 (exploratory)"


# ================================================================================================
# a real report, built once — the views are never handed a hand-written stub
# ================================================================================================

def _synthetic_events():
    """Two obvious event classes: shallow-and-brief, deep-and-long.

    The same builder ``test_subpop_gui`` uses. Real events through the real
    pipeline, so ``embeddings`` / ``labels`` / ``waveforms`` are whatever the
    pipeline actually produces rather than what a stub author assumed.
    """
    rng = np.random.default_rng(0)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans = []
    pos = 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    events = EventSet(
        pd.DataFrame(rows)[REQUIRED],
        {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1},
    )
    return np.ascontiguousarray(x, dtype=np.float32), events


@pytest.fixture(scope="module")
def report():
    """One real :class:`SubpopReport` for the whole module (about a second).

    The cheap grid, exactly as ``test_subpop_gui._Cheap`` — the SAME pipeline code,
    a smaller sweep. The full grid's numbers are pinned against the owner's own run
    in ``tests/analysis``; nothing here is about the numbers being right, only about
    the picture being the numbers.
    """
    arr, events = _synthetic_events()
    cfg = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    rep = SR.run_subpopulations(
        arr, arr, FS, events, recording="r", duration_s=N_SAMPLES / FS, config=cfg)
    # Guard the fixture itself: if the pipeline ever stops producing these, the
    # failures below would read as view bugs.
    assert rep.embeddings is not None and rep.labels is not None
    assert rep.waveforms is not None and rep.quantification is not None
    return rep


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """A namespace of this module's own, so nothing here can read or leave state.

    The window persists nothing (it is deliberately not a panel), but a test that
    constructs windows should not be the reason a later test reads a stale key.
    """
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("SubpopViewsPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


@pytest.fixture
def window(qapp, qtbot, report):
    w = SubpopReportWindow()
    qtbot.addWidget(w)
    w.set_report(report, subtitle="synthetic")
    return w


@pytest.fixture
def scatter(window) -> EmbeddingScatterView:
    return window.embedding_view()


@pytest.fixture
def waves(window) -> WaveformOverlayView:
    return window.waveform_view()


def _with_umap(rep):
    """A copy of ``rep`` whose embedding table also carries UMAP columns.

    The numbers are the t-SNE ones shifted — this is a FIXTURE, not a computation
    the app does: what is under test is that the view offers UMAP, never picks it
    by default, and says the coordinates are unstable whenever it draws them. The
    real UMAP is an opt-in at the analysis layer and is not run here precisely
    because it is not bit-reproducible.
    """
    emb = rep.embeddings.copy()
    emb["umap_x"] = emb["tsne_x"].to_numpy() * 0.5 + 3.0
    emb["umap_y"] = emb["tsne_y"].to_numpy() * 0.5 - 3.0
    return dataclasses.replace(rep, embeddings=emb)


# ================================================================================================
# (a) items 6-7 — the embedding scatter draws the report's own coordinates
# ================================================================================================

def test_the_scatter_offers_only_the_embeddings_this_run_computed(scatter):
    """UMAP is off by default, so it is not in the combo — not greyed, not there."""
    assert scatter.embeddings() == ("t-SNE",)
    assert scatter.embedding() == "t-SNE"


def test_the_scatter_offers_the_mixture_and_every_exploratory_labelling(scatter):
    """The whole argument of the view: ONE cloud, the two independent methods."""
    assert scatter.colour_modes() == (_GMM, _HDBSCAN, _KMEANS)
    assert scatter.colour_mode() == _GMM, "the reported model is what opens"


def test_the_points_are_the_reports_own_coordinates_bit_for_bit(scatter, report):
    """No jitter, no rescale, no re-sort. The view adds ink, not arithmetic."""
    x, y, _labels = scatter.drawn()
    np.testing.assert_array_equal(x, report.embeddings["tsne_x"].to_numpy(float))
    np.testing.assert_array_equal(y, report.embeddings["tsne_y"].to_numpy(float))


def test_the_colours_are_the_reports_own_labels_bit_for_bit(scatter, report):
    _x, _y, labels = scatter.drawn()
    expected = report.quantification.fit.labels.reindex(report.embeddings.index)
    np.testing.assert_array_equal(labels, expected.to_numpy(np.int64))


def test_recolouring_moves_no_point(scatter, report):
    """The two colourings must be the SAME cloud or they are not comparable."""
    x0, y0, gmm = scatter.drawn()
    scatter.set_colour_mode(_HDBSCAN)
    x1, y1, hdb = scatter.drawn()
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(y0, y1)
    np.testing.assert_array_equal(
        hdb, report.labels[_col_for(report, "hdbscan")].reindex(
            report.embeddings.index).to_numpy(np.int64))
    assert not np.array_equal(gmm, hdb), "the fixture must actually disagree"


def _col_for(report, base: str) -> str:
    for col in report.labels.columns:
        if str(col).partition("_")[0] == base:
            return str(col)
    raise AssertionError(f"no {base} column in {list(report.labels.columns)}")


def test_every_embedded_event_is_drawn_and_the_caption_says_how_many(scatter, report):
    x, _y, _l = scatter.drawn()
    assert x.size == len(report.embeddings)
    assert f"{x.size} of {len(report.embeddings)} event(s) drawn" in scatter.caption()


def test_the_legend_is_rebuilt_and_not_appended_to(scatter):
    """The bug this caught: pyqtgraph's ``PlotItem.clear()`` does not clear a legend
    whose samples are free-standing swatches, so the second colouring listed the
    mixture's classes AND HDBSCAN's at once — a picture claiming seven classes."""
    n_gmm = len(scatter.legend_entries())
    scatter.set_colour_mode(_KMEANS)
    n_km = len(scatter.legend_entries())
    scatter.set_colour_mode(_GMM)
    assert len(scatter.legend_entries()) == n_gmm
    assert n_km == 3, "k-means, k=3 has three classes and no noise"


def test_the_legend_names_the_classes_of_the_colouring_it_is_showing(scatter):
    """"subpop" and "cluster" are not synonyms and the legend does not pretend."""
    assert all(t.startswith("subpop ") for t, _c in scatter.legend_entries())
    scatter.set_colour_mode(_HDBSCAN)
    texts = [t for t, _c in scatter.legend_entries()]
    assert all(t.startswith("cluster ") or t.startswith("noise") for t in texts)
    assert any("noise (−1)" in t for t in texts), "HDBSCAN's −1 is a decision, and named"


def test_hdbscans_noise_is_grey_and_listed_last(scatter):
    """A noise cloud in a categorical colour reads as one more class."""
    scatter.set_colour_mode(_HDBSCAN)
    entries = scatter.legend_entries()
    text, colour = entries[-1]
    assert text.startswith("noise")
    assert colour.lower() == CAT.NOISE_HEX.lower()
    assert colour.lower() not in {c.lower() for c in CAT.CATEGORY_COLOURS}


def test_the_legend_counts_are_the_counts_actually_drawn(scatter):
    _x, _y, labels = scatter.drawn()
    for text, _colour in scatter.legend_entries():
        cid = int(re.search(r"(-?\d+|−1)", text.replace("−", "-")).group(0))
        n = int(re.search(r"n = (\d+)", text).group(1))
        assert n == int(np.count_nonzero(labels == cid)), text


def test_a_label_the_embedding_has_no_row_for_is_dropped_and_declared(report, qtbot):
    """NaN ("not in this stage's scope") and −1 ("in no cluster") are NOT the same,
    and collapsing them would repaint out-of-scope events as noise."""
    labels = report.labels.copy()
    col = _col_for(report, "hdbscan")
    labels.loc[labels.index[:7], col] = np.nan
    view = EmbeddingScatterView()
    qtbot.addWidget(view)
    view.set_report(dataclasses.replace(report, labels=labels))
    view.set_colour_mode(_HDBSCAN)
    _x, _y, drawn = view.drawn()
    assert drawn.size == len(report.embeddings) - 7
    assert "7 event(s) carry no" in view.caption()
    assert "not the same thing as noise" in view.caption()


def test_naming_an_embedding_or_a_colouring_that_is_not_there_raises(scatter):
    """A silent no-op leaves the reader looking at t-SNE under a UMAP heading."""
    with pytest.raises(ValueError, match="no embedding"):
        scatter.set_embedding("UMAP")
    with pytest.raises(ValueError, match="no colouring"):
        scatter.set_colour_mode("t-SNE by eye")


def test_the_axes_are_named_arbitrary_and_the_honesty_line_is_on_screen(scatter, window):
    pi = scatter._plot.getPlotItem()
    assert "arb. units" in pi.getAxis("bottom").labelText
    assert "arb. units" in pi.getAxis("left").labelText
    assert "nothing may be concluded from the coordinates" in scatter._honesty.text()
    assert scatter._honesty.isVisibleTo(window)


def test_the_honesty_line_still_matches_the_argument_in_the_analysis_layer():
    """The sentence is restated in the window because it does not import analysis.
    If the analysis layer ever stops arguing it, this string is a leftover."""
    src = pathlib.Path(SR.__file__).read_text()
    assert "sees a cluster label" in src
    assert "nothing may be concluded from the coordinates" in src
    assert "EXPORTED" in src and "never printed" in src


# ================================================================================================
# (b) UMAP — opt-in only, and never silently
# ================================================================================================

def test_umap_is_off_in_the_default_config():
    """The discipline this app is built on: a column that looks like data must be
    reproducible. UMAP is not, across umap-learn / numba / platform versions."""
    assert SR.SubpopConfig().with_umap is False
    assert SR.SubpopConfig().with_tsne is True


def test_with_no_umap_the_view_still_explains_that_it_exists_and_why_it_is_off(scatter):
    """Otherwise a reader concludes this pipeline has one embedding."""
    note = scatter._umap_note.text()
    assert "UMAP is OFF" in note
    assert "not bit-reproducible" in note
    assert "SubpopConfig(with_umap=True)" in note, "the note names the opt-in"


def test_when_umap_is_present_it_is_offered_but_is_not_the_default(report, qtbot):
    view = EmbeddingScatterView()
    qtbot.addWidget(view)
    view.set_report(_with_umap(report))
    assert view.embeddings() == ("t-SNE", "UMAP")
    assert view.embedding() == "t-SNE", "the reproducible one opens"
    assert "not stable" in view._umap_note.text()


def test_drawing_umap_says_in_the_view_that_the_coordinates_are_not_stable(report, qtbot):
    view = EmbeddingScatterView()
    qtbot.addWidget(view)
    rep = _with_umap(report)
    view.set_report(rep)
    view.set_embedding("UMAP")
    assert "NOT STABLE" in view._umap_note.text()
    assert view._umap_note.isVisible() or view._umap_note.text()
    x, y, _l = view.drawn()
    np.testing.assert_array_equal(x, rep.embeddings["umap_x"].to_numpy(float))
    np.testing.assert_array_equal(y, rep.embeddings["umap_y"].to_numpy(float))
    assert "UMAP 1 (arb. units)" == view._plot.getPlotItem().getAxis("bottom").labelText


# ================================================================================================
# (c) item 13 — the median waveforms, on one axis, in the scatter's colours
# ================================================================================================

def test_one_curve_per_subpopulation_column(waves, report):
    cols = [c for c in report.waveforms.columns if c != "T"]
    assert [name for name, _c, _n in waves.series()] == cols


def test_the_curves_are_the_tables_own_numbers_bit_for_bit(waves, report):
    import pyqtgraph as pg

    curves = [it for it in waves._plot.getPlotItem().items
              if isinstance(it, pg.PlotDataItem)]
    drawn = {}
    for c in curves:
        x, y = c.getData()
        drawn[round(float(np.nanmean(y)), 9)] = (x, y)
    T = report.waveforms["T"].to_numpy(float)
    for col in report.waveforms.columns:
        if col == "T":
            continue
        y_exp = report.waveforms[col].to_numpy(float)
        key = round(float(np.nanmean(y_exp)), 9)
        assert key in drawn, f"{col} is not on the axis"
        x_got, y_got = drawn[key]
        np.testing.assert_array_equal(x_got, T)
        np.testing.assert_array_equal(y_got, y_exp)


def test_the_event_counts_are_the_tables_own_attrs(waves, report):
    counts = report.waveforms.attrs["n_events"]
    assert {name: n for name, _c, n in waves.series()} == {
        str(k): int(v) for k, v in counts.items()}


def test_the_caption_says_median_raw_and_time_normalised(waves):
    """Three claims a reader would otherwise get wrong: it is a median not a mean,
    it is the RAW trace not the detrended one the features came from, and the x
    axis is not seconds."""
    cap = waves.caption()
    assert "MEDIAN" in cap
    assert "RAW" in cap
    assert "the x axis is not seconds" in cap


def test_a_column_that_could_not_be_averaged_is_named_not_drawn_flat(report, qtbot):
    """A flat line at zero is a waveform, and would be read as one."""
    import pyqtgraph as pg

    wf = report.waveforms.copy()
    wf.attrs["n_events"] = dict(report.waveforms.attrs["n_events"])
    dead = [c for c in wf.columns if c != "T"][0]
    wf[dead] = np.nan
    view = WaveformOverlayView()
    qtbot.addWidget(view)
    view.set_report(dataclasses.replace(report, waveforms=wf))
    curves = [it for it in view._plot.getPlotItem().items if isinstance(it, pg.PlotDataItem)]
    assert len(curves) == len([c for c in wf.columns if c != "T"]) - 1
    assert dead in view.caption()
    assert "not drawable" in view.caption()
    assert dead in [name for name, _c, _n in view.series()], "still listed, with its colour"


def test_a_run_with_no_waveforms_says_which_rather_than_going_blank(report, qtbot):
    view = WaveformOverlayView()
    qtbot.addWidget(view)
    view.set_report(dataclasses.replace(report, waveforms=None))
    assert "No per-subpopulation waveforms" in view.message()
    assert view.caption() == view.message()


# ================================================================================================
# (d) one subpopulation, one ink, everywhere in the window
# ================================================================================================

def test_a_subpopulation_is_the_same_colour_in_both_figures(scatter, waves):
    """The reason the palette is shared at all. If the scatter's subpop 1 and the
    overlay's subpop 1 were different inks, the two figures would read as two
    different findings."""
    from_scatter = {}
    for text, colour in scatter.legend_entries():
        from_scatter[int(re.search(r"subpop (\d+)", text).group(1))] = colour.lower()
    from_waves = {}
    for name, colour, _n in waves.series():
        from_waves[int(name.rsplit("_", 1)[-1])] = colour.lower()
    assert from_scatter == from_waves
    assert from_scatter, "the fixture must actually have subpopulations"


def test_the_mapping_is_by_value_and_not_by_position(scatter):
    """``category_colour(2)`` is one colour whoever asks. A palette walked in
    enumeration order recolours a class the moment an empty one is filtered out."""
    for text, colour in scatter.legend_entries():
        cid = int(re.search(r"subpop (\d+)", text).group(1))
        assert colour.lower() == CAT.category_colour(cid).name().lower()
    scatter.set_colour_mode(_HDBSCAN)
    for text, colour in scatter.legend_entries():
        if text.startswith("cluster "):
            cid = int(re.search(r"cluster (\d+)", text).group(1))
            assert colour.lower() == CAT.category_colour(cid).name().lower()


def test_the_palette_is_okabe_ito_and_not_a_local_list():
    from nanoscope.plot.style import OKABE_ITO

    assert CAT.CATEGORY_COLOURS is OKABE_ITO


# ================================================================================================
# (e) the caveats stay in the same eyeline as the figures
# ================================================================================================

def test_the_window_has_no_tab_widget(window):
    """A tab is a place to put the reasons not to believe the picture where the
    reader who opened the picture will not see them."""
    assert window.findChildren(QtWidgets.QTabWidget) == []


def test_every_warning_the_run_raised_is_in_the_strip_verbatim(window, report):
    strip = window.caveats()
    assert report.warnings, "the fixture must actually raise some"
    assert window.details_text() == window._caveat_summary.full_text()
    assert "Figure context (full captions)" in window.details_text()
    for w in report.warnings:
        assert w in strip, w[:60]
        assert w in window.details_text(), w[:60]


def test_the_strip_also_carries_the_caveat_that_is_about_the_picture(window):
    """The two colourings share a palette and not a meaning; the crosstab is the
    correspondence, not the colour."""
    strip = window.caveats()
    assert "colour is a class id WITHIN ONE COLOURING" in strip
    assert "crosstab" in strip
    assert strip.index("WITHIN ONE COLOURING") < strip.index(window.report().warnings[0])


def test_the_strip_is_on_screen_beside_the_figures_without_any_interaction(window, qtbot):
    window.show()
    qtbot.waitExposed(window)
    box = window._caveat_box
    assert window._standing_caveats.isVisible()
    assert window._caveat_summary.summary.isVisible()
    assert box.isHidden(), "full details are disclosed while their summary stays visible"
    figures_pane = window.splitter().widget(1)
    assert box.isAncestorOf(window._caveats)
    assert figures_pane.isAncestorOf(box), "the strip belongs to the figure column"
    assert figures_pane.isAncestorOf(window.embedding_view())
    window.close()


def test_the_strip_title_counts_the_run_s_caveats_and_the_view_s(window, report):
    assert window._caveat_box.title() == (
        f"Caveats — {len(report.warnings)} from the run, and one about these figures")


def test_the_header_still_carries_the_caveat_count_with_the_figures_present(window, report):
    assert f"{len(report.warnings)} caveat" in window.header()


def test_the_caveats_are_a_pane_of_the_figures_splitter_not_a_fixed_strip(window):
    """Measured defect, fixed: as a fixed-height widget UNDER the splitter the box
    took the same ~226 px at every window size, so every pixel lost to a smaller
    window came out of the pictures and none out of the warnings."""
    fig = window.figures_splitter()
    assert fig.count() == 3
    assert fig.widget(2) is window._caveat_box
    assert not fig.childrenCollapsible(), "the reader may resize it, never remove it"
    assert window._caveat_summary.summary.isVisibleTo(window)
    assert window._caveats.isHidden(), "full details are disclosed without losing the standing summary"
    assert window._caveats.maximumHeight() <= 260, "expanded details must remain bounded"
    window._caveat_summary.toggle.setChecked(True)
    assert window._caveats.toPlainText() == window.caveats()


@pytest.mark.parametrize("size", [(1440, 980), (1100, 760), (900, 700)])
def test_neither_figure_collapses_to_a_strip_at_any_usable_window_size(window, qtbot, size):
    """THE BUG THIS PINS, measured before the fix: dragged to 1100x700 the scatter
    drew a 535x12 data rectangle — a picture of nothing, still captioned "1076 of
    1076 event(s) drawn". A 2-D embedding at that aspect turns a round blob into a
    streak and two neighbouring classes into one arm.

    The assertion is on the DATA rectangle (the ViewBox), not the widget: a widget
    is tall enough long before the axis furniture leaves any room for the points.
    """
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    for name, view in (("scatter", window.embedding_view()),
                       ("waveforms", window.waveform_view())):
        rect = view._plot.getPlotItem().getViewBox().screenGeometry()
        assert rect.height() >= 60, f"{name} data rect is {rect.width()}x{rect.height()}"
        assert rect.width() / max(rect.height(), 1) < 9.0, (
            f"{name} is drawn at {rect.width() / max(rect.height(), 1):.1f}:1")
    assert window._standing_caveats.isVisible()
    assert window._standing_caveats.height() > 0
    window.close()


def test_the_window_refuses_to_be_dragged_smaller_than_its_figures_need(window):
    """The scrollable body keeps plot floors without forcing the whole page to
    grow beyond the Explorer. The surrounding controls still need a usable floor."""
    assert window.minimumHeight() >= 480
    window.resize(400, 300)
    assert window.height() >= window.minimumHeight()


def test_the_figures_share_the_window_with_the_report_text(window):
    """Not a replacement for the document: the numbers and the picture together."""
    assert window.splitter().count() == 2
    assert window.text().strip(), "the report document is still rendered"
    assert all(s > 0 for s in window.splitter().sizes())


# ================================================================================================
# (f) the empty state is a view too
# ================================================================================================

def test_a_window_with_no_report_says_what_it_is_waiting_for(qapp, qtbot):
    w = SubpopReportWindow()
    qtbot.addWidget(w)
    assert "Generate Subpopulations" in w.embedding_view().message()
    assert "Generate Subpopulations" in w.waveform_view().message()
    assert w.embedding_view().drawn()[0].size == 0
    assert w.waveform_view().series() == []
    assert not w.can_export()


def test_clearing_a_report_clears_both_figures(window):
    assert window.embedding_view().drawn()[0].size > 0
    window.set_report(None)
    assert window.embedding_view().drawn()[0].size == 0
    assert window.waveform_view().series() == []
    assert window.caveats() == ""


def test_installing_a_second_report_replaces_rather_than_accumulates(window, report):
    n_before = len(window.embedding_view().legend_entries())
    window.set_report(report, subtitle="again")
    assert len(window.embedding_view().legend_entries()) == n_before
    assert len(window.waveform_view().series()) == len(report.waveforms.columns) - 1


def test_a_report_whose_run_computed_no_embedding_explains_instead_of_blanking(report, qtbot):
    view = EmbeddingScatterView()
    qtbot.addWidget(view)
    view.set_report(dataclasses.replace(report, embeddings=None))
    assert view.embeddings() == ()
    assert "no embedding" in view.message().lower()
    assert "Everything else in the report stands without it" in view.message()


# ================================================================================================
# (g) the shared map is shared — BOTH callers, and it stays a leaf
# ================================================================================================

def test_the_plots_panel_uses_the_shared_map_and_not_a_copy_of_it():
    """Identity, not equality. A copied four-line function that happened to agree
    today is exactly how the two pictures drift apart."""
    from nanoscope.gui import plots_panel as PP

    assert PP._cluster_color is CAT.category_colour


def test_the_report_window_uses_the_shared_map_too():
    import nanoscope.gui.subpop_window as mod

    src = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(src)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "categorical"
        for alias in node.names
    }
    assert "category_colour" in imported
    assert "add_category_scatter" in imported


@pytest.mark.parametrize("module_name", [
    "nanoscope.gui.subpop_window",
    "nanoscope.gui.plots_panel",
])
def test_neither_caller_hard_codes_a_categorical_palette(module_name):
    """The failure this module exists to prevent, pinned at the source level: a
    palette colour written as a literal in a caller is a second copy of the map by
    another name, and it is how the two pictures drift apart one colour at a time.

    Caught a real one: ``subpop_window`` spelled ``#E69F00`` for the UMAP warning
    label, which is ``PALETTE["warn"]``. It asks by role now.

    BLACK AND WHITE ARE EXEMPT and that is not a fudge. Okabe-Ito's eighth entry is
    ``#000000``, but ``plots_panel`` uses black and white as INK ON A SWATCH — the
    readable text colour over a filled cell, chosen by luminance and not by class —
    and that is not a categorical decision. Every other palette entry is.

    It reads the AST and not the raw text, so it judges CODE. A comment or a
    docstring may name ``#E69F00`` while explaining why it is no longer written
    there — the first draft of this test failed on its own explanation.
    """
    import importlib

    from nanoscope.plot.style import OKABE_ITO

    tree = ast.parse(pathlib.Path(importlib.import_module(module_name).__file__).read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    ink_and_paper = {"#000000", "#ffffff"}
    palette = {h.lower() for h in OKABE_ITO} - ink_and_paper
    offenders: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            offenders |= {h.lower() for h in re.findall(r"#[0-9A-Fa-f]{6}", node.value)} & palette
    assert not offenders, (
        f"{module_name} spells palette colours itself: {sorted(offenders)} — ask "
        f"plot.style for them by role, or gui.categorical by class id")


def test_the_shared_module_imports_no_gui_module():
    """It is a leaf, and the report window's import allow-list depends on that."""
    tree = ast.parse(pathlib.Path(CAT.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "gui" not in (node.module or "").split(".")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("nanoscope.gui")


def test_the_scatter_helper_refuses_mismatched_arrays():
    """Truncating would draw a picture in which some points are the wrong colour,
    which is worse than not drawing it."""
    import pyqtgraph as pg

    plot = pg.PlotWidget().getPlotItem()
    with pytest.raises(ValueError):
        CAT.add_category_scatter(plot, [0.0, 1.0], [0.0, 1.0], [0])


def test_category_counts_puts_every_negative_id_last():
    assert CAT.category_counts([2, -1, 0, -1, 2]) == [(0, 1), (2, 2), (-1, 2)]


# ================================================================================================
# (h) the frozen contracts, restated where the figures could break them
# ================================================================================================

def test_drawing_the_figures_mutates_nothing_in_the_report(report, qtbot):
    """A view that sorted a table in place, or filled a NaN, would move a number in
    the object the export writes."""
    before = {
        "emb": report.embeddings.copy(),
        "wave": report.waveforms.copy(),
        "labels": report.labels.copy(),
        "fit": report.quantification.fit.labels.copy(),
    }
    w = SubpopReportWindow()
    qtbot.addWidget(w)
    w.set_report(report)
    for mode in w.embedding_view().colour_modes():
        w.embedding_view().set_colour_mode(mode)
    pd.testing.assert_frame_equal(report.embeddings, before["emb"])
    pd.testing.assert_frame_equal(report.waveforms, before["wave"])
    pd.testing.assert_frame_equal(report.labels, before["labels"])
    pd.testing.assert_series_equal(report.quantification.fit.labels, before["fit"])


def test_the_figures_add_no_thread_and_no_unbounded_wait():
    """Contract 5. A figure is drawn on the GUI thread or not at all."""
    import nanoscope.gui.categorical as cat_mod
    import nanoscope.gui.subpop_window as win_mod

    for mod in (win_mod, cat_mod):
        src = pathlib.Path(mod.__file__).read_text()
        assert "QThread" not in src, mod.__name__
        assert "terminate()" not in src, mod.__name__
        assert ".wait()" not in src, mod.__name__


def test_the_views_never_reach_for_the_analysis_layer():
    """Contract 4. If a view wants a number the report does not carry, it goes
    without it — it does not import the pipeline and recompute it in the GUI."""
    import nanoscope.gui.subpop_window as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        for name in names:
            assert "analysis" not in name.split("."), name
    src = pathlib.Path(mod.__file__).read_text()
    for banned in ("sklearn", "scipy", "TSNE", "GaussianMixture"):
        assert banned not in src, banned


# ================================================================================================
# the legend is BESIDE the data, not on top of it
# ================================================================================================

def _legend_occlusion(view) -> tuple[int, int]:
    """``(points under the legend, points drawn)`` — in DATA coordinates.

    The legend is a graphics item and the points are data, so the only honest test
    maps one into the other's frame: take the legend's bounding rect, map it to the
    ViewBox, convert those two corners to data coordinates, and count the drawn
    points inside. Asserting on the anchor constant instead would pass for a legend
    that was anchored outside and then drawn 300 px wide.
    """
    vb = view._plot.getPlotItem().getViewBox()
    # Taken from the ITEMS rather than from a view accessor, so the one helper
    # covers both figures: the scatter's points and the waveform curves' vertices
    # are equally "ink the caption says is on screen".
    xs, ys = [], []
    for item in vb.addedItems:
        if not hasattr(item, "getData"):
            continue
        data = item.getData()
        if data is None or data[0] is None or len(data[0]) == 0:
            continue
        xs.append(np.asarray(data[0], dtype=float))
        ys.append(np.asarray(data[1], dtype=float))
    if not xs:
        return 0, 0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    lr = view._legend.mapRectToItem(vb, view._legend.boundingRect())
    p1 = vb.mapToView(QtCore.QPointF(lr.left(), lr.top()))
    p2 = vb.mapToView(QtCore.QPointF(lr.right(), lr.bottom()))
    x0, x1 = sorted((p1.x(), p2.x()))
    y0, y1 = sorted((p1.y(), p2.y()))
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return int(np.count_nonzero(inside)), int(x.size)


@pytest.mark.parametrize("size", [(1600, 980), (1360, 860), (1100, 760), (900, 620)])
def test_the_legend_hides_no_point_it_claims_to_have_drawn(window, qtbot, size):
    """THE BUG THIS PINS, measured on the owner's zn_30h before the fix: the legend
    was anchored INSIDE the ViewBox on an opaque plate, and at the window's own
    default size it covered 62 of 1076 points while the caption underneath read
    "1076 of 1076 event(s) drawn".

    The loss was not even, which is what made it matter: 43 of subpop 2 (16 % of
    that class) and 19 of subpop 3 (15 %), against 0 of subpops 0 and 1. Subpop 3 is
    the class the crosstab shows is least well supported — the one a reader most
    needs to see whole — so the plate was hiding evidence about the weakest claim.

    The legend now occupies its own PlotItem layout column, past the right axis,
    where it cannot overlap the data at any size or aspect. Parametrized across the
    range because "outside" must not be a property of one window geometry.
    """
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    for name, view in (("scatter", window.embedding_view()),
                       ("waveforms", window.waveform_view())):
        hidden, drawn = _legend_occlusion(view)
        assert drawn > 0, f"{name} drew nothing, so this test proved nothing"
        assert hidden == 0, (
            f"{name} at {size}: the legend covers {hidden} of {drawn} drawn points, "
            f"while the caption claims all {drawn} are drawn")
    window.close()


def test_the_caption_may_claim_every_point_because_every_point_is_visible(window, qtbot):
    """The caption's count and the legend's geometry are one claim, so they are
    asserted together: the number in the text equals the number not hidden."""
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    view = window.embedding_view()
    hidden, drawn = _legend_occlusion(view)
    assert f"{drawn} of {drawn} event(s) drawn" in view.caption()
    assert hidden == 0
    window.close()


def test_the_legend_is_not_parented_into_the_data_area(scatter):
    """The structural half of the same claim. ``addLegend()`` parents the legend to
    the ViewBox; this one lives in the PlotItem's grid, in a column past the axis."""
    pi = scatter._plot.getPlotItem()
    assert scatter._legend.parentItem() is not pi.getViewBox(), (
        "a legend parented to the ViewBox is inside the data area by construction")
    assert pi.legend is scatter._legend, (
        "PlotItem.legend must still point at it, or a later addLegend() builds a second one")


# ================================================================================================
# more classes than the palette has inks
# ================================================================================================

def test_colours_repeat_is_false_for_any_labelling_the_palette_can_hold():
    assert not CAT.colours_repeat([0, 1, 2, 3])
    assert not CAT.colours_repeat([-1, 0, 7])
    assert not CAT.colours_repeat([5, 5, 5])
    assert not CAT.colours_repeat([])


def test_colours_repeat_is_true_exactly_when_two_classes_share_an_ink():
    n = len(CAT.CATEGORY_COLOURS)
    assert CAT.colours_repeat([0, n]), "0 and 8 are the same colour"
    assert CAT.colours_repeat(list(range(n + 1)))
    assert not CAT.colours_repeat(list(range(n)))
    assert CAT.category_colour(0).name() == CAT.category_colour(n).name()


def test_negative_ids_never_make_colours_repeat():
    """Every negative id is grey ON PURPOSE — that is one meaning, not a collision,
    so the noise ink must not trip the warning."""
    assert not CAT.colours_repeat([-1, -2, -5, 0, 1])


def test_the_scatter_says_so_when_two_classes_share_a_colour(scatter, report):
    """A legend with two identical swatches tells the reader the ambiguity exists
    but not which point is which, so the caption has to say it in words.

    HDBSCAN is the labelling that can get here: the mixture's K is capped at 8 by
    the CV grid, but HDBSCAN's cluster count is data-dependent and uncapped.
    """
    n = len(CAT.CATEGORY_COLOURS)
    labels = report.labels.copy()
    labels["hdbscan"] = np.tile(
        np.arange(n + 3, dtype=float), int(np.ceil(len(labels) / (n + 3))))[:len(labels)]
    scatter.set_report(dataclasses.replace(report, labels=labels))
    scatter.set_colour_mode(_HDBSCAN)
    cap = scatter.caption()
    assert "COLOURS REPEAT" in cap, f"caption is silent about the collision: {cap!r}"
    assert f"{n + 3} class(es)" in cap


def test_the_scatter_stays_quiet_when_no_colour_repeats(scatter):
    """The note must be a fact about the labelling, not decoration that is always on."""
    assert "COLOURS REPEAT" not in scatter.caption()
    _x, _y, ids = scatter.drawn()
    assert not CAT.colours_repeat(ids)


def test_the_wrap_rule_is_asked_for_and_not_re_spelled_in_the_view():
    """``% 8`` in a caller is a second copy of the palette's size — the class of
    duplication this module exists to end. The view must ASK."""
    import nanoscope.gui.subpop_window as SW
    src = pathlib.Path(SW.__file__).read_text(encoding="utf-8")
    assert "colours_repeat" in src
    assert "len(CATEGORY_COLOURS)" not in src, (
        "the view is re-deriving the palette length instead of asking categorical")


def test_every_categorical_swatch_in_the_app_comes_from_the_one_helper():
    """``legend_swatch`` had exactly ONE caller while ``categorical``'s docstring
    claimed it was "the pyqtgraph plumbing both windows would otherwise write
    twice": ``plots_panel._legend_clusters`` hand-wrote its identical body.

    That is the tree's signature failure in miniature. A pen or an alpha added to
    the shared helper would have changed the report window's swatches and silently
    left the PCA panel's alone — two pictures drifting apart one colour at a time.
    Pinned by AST so a comment cannot satisfy it: no caller may construct a
    ScatterPlotItem whose brush is a categorical colour.
    """
    import nanoscope.gui.plots_panel as pp
    import nanoscope.gui.subpop_window as sw

    for mod in (pp, sw):
        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "ScatterPlotItem":
                continue
            brushes = [k.value for k in node.keywords if k.arg == "brush"]
            for b in brushes:
                spelled = ast.dump(b)
                assert "category_colour" not in spelled and "_cluster_color" not in spelled, (
                    f"{mod.__name__} builds a categorical legend swatch by hand; "
                    f"call categorical.legend_swatch instead")
        assert "legend_swatch" in pathlib.Path(mod.__file__).read_text(), mod.__name__


def test_the_shared_swatch_really_is_the_object_both_callers_get():
    """Identity, not resemblance — the same function object, so the two windows
    cannot be given different swatch behaviour without one test failing."""
    import nanoscope.gui.plots_panel as pp
    import nanoscope.gui.subpop_window as sw

    assert pp.legend_swatch is CAT.legend_swatch
    assert sw.legend_swatch is CAT.legend_swatch
    assert pp._cluster_color is CAT.category_colour


@pytest.mark.parametrize("n_classes", [2, 4, 8, 11, 16, 25, 40])
def test_a_long_legend_never_crops_the_figure_it_labels(window, report, qtbot, n_classes):
    """THE REGRESSION THIS PINS, and it was introduced by the fix above.

    Moving the legend into the PlotItem's layout made its height a DEMAND on the
    plot rather than a decoration floating over it. Measured with 25 HDBSCAN
    clusters: the legend asked for 593 px inside a 238 px widget and the ViewBox
    grew to match — 595 px of plot rendered into a 238 px viewport, the bottom two
    thirds of the cloud cropped away, and the caption underneath still reading
    "1076 of 1076 event(s) drawn". At 40 classes the ViewBox reached 940 px.

    That is the same false claim the opaque plate made, rotated onto the other
    axis. :func:`_fit_legend` bounds it by wrapping into columns. The assertion is
    that the ViewBox never exceeds the widget that has to show it — cropping is
    exactly the case where ``boundingRect`` and the real widget disagree.
    """
    labels = report.labels.copy()
    labels["hdbscan"] = np.tile(
        np.arange(n_classes, dtype=float),
        int(np.ceil(len(labels) / n_classes)))[:len(labels)]
    window.set_report(dataclasses.replace(report, labels=labels))
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    view = window.embedding_view()
    view.set_colour_mode(_HDBSCAN)
    qtbot.wait(30)

    vb = view._plot.getPlotItem().getViewBox()
    assert vb.boundingRect().height() <= view._plot.height() + 1, (
        f"{n_classes} classes: the ViewBox is {vb.boundingRect().height():.0f} px inside a "
        f"{view._plot.height()} px widget, so the figure is cropped while the caption "
        f"claims every point is drawn")
    assert vb.boundingRect().width() > 0
    hidden, drawn = _legend_occlusion(view)
    assert hidden == 0
    assert drawn > 0
    window.close()


def test_the_mixture_never_costs_the_figure_any_width_to_its_legend():
    """The CV grid is K=1-8, so a mixture colouring must always fit ONE column —
    the width trade is reserved for HDBSCAN, which is the only uncapped labelling."""
    import nanoscope.gui.subpop_window as SW
    assert SW._LEGEND_ROWS >= 8, (
        "a smaller cap would wrap a legal mixture colouring into two columns and "
        "take width off the figure for a K the CV grid can actually choose")
