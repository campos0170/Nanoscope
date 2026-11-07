"""Phase 2 figure lifetime: plain history, source policy, and sampled outputs stay independent."""
from dataclasses import replace
from uuid import uuid4

import nanoscope  # noqa: F401 bootstrap before Qt
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest

from nanoscope.core.figure_codecs import decode_legacy, encode_legacy
from nanoscope.core.figure_history_state import HistoryCapsule
from nanoscope.core.figure_spec import FigureRecord, FigureValidationError, thaw_json
from nanoscope.gui.figure_adapter_pages import (
    adapter_for, apply_look_to_record, initial_history, move_history, NeedsResidentHistory)
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plots_panel import PlotsWindow, plot_config_to_json
from . import p1_hosting as H


def _record(adapter):
    decoded = decode_legacy(adapter.capture_native(), figure_id=str(uuid4()), name="Lifetime test")
    return adapter.capture_record(FigureRecord(decoded.spec, decoded.resources))


def test_appearance_capsule_keeps_redo_and_dropped_count_without_replaying_changes(qtbot):
    first = PlotAppearanceWidget()
    second = PlotAppearanceWidget()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.apply_appearance(PlotAppearance(title="A"))
    first.apply_appearance(PlotAppearance(title="B"))
    first.undo()
    first._history.dropped = 7
    capsule = first.capture_history_capsule()
    second.set_state(first.state())
    changed = []
    second.changed.connect(changed.append)
    second.restore_history_capsule(HistoryCapsule.from_dict(capsule.to_dict()))
    assert second.capture_history_capsule() == capsule
    assert changed == []
    assert second.redo() and second.state().title == "B"
    assert first.state().title == "A"


def test_mismatched_appearance_history_is_rejected_before_any_stack_write(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    widget.apply_appearance(PlotAppearance(title="kept"))
    before = widget.capture_history_capsule()
    wrong = replace(before, base=PlotAppearance(title="wrong").to_dict())
    with pytest.raises(FigureValidationError, match="does not match"):
        widget.restore_history_capsule(wrong)
    assert widget.capture_history_capsule() == before


def test_plots_capsule_keeps_import_origins_and_commits_an_open_text_run(qtbot):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    page._history.open_origin()
    origin = page._history.origin
    page._norm_refusal.record(" density ", origin)
    page._config_losses._records[origin] = ("original.json", ("style.future_field",))
    page._style = replace(page._style, title="Unfinished typing")
    page._history_timer.start()
    capsule = page.capture_history_capsule()
    assert not page._history_timer.isActive()
    assert thaw_json(capsule.base)["style"]["title"] == "Unfinished typing"
    page._history.reset(page.config())
    page._norm_refusal.forget()
    page._config_losses.forget()
    page.restore_history_capsule(capsule)
    assert page.capture_history_capsule() == capsule
    assert page._history.origin == origin
    assert page._norm_refusal._records[origin] == " density "
    assert page._config_losses._records[origin] == ("original.json", ("style.future_field",))


@pytest.mark.parametrize("kind", ["event_waveforms", "noise"])
def test_follow_selection_has_one_visible_control_and_no_callback_during_restore(qtbot, kind):
    page = H.page_builders()[kind]()
    explorer = H.hosted(qtbot, page, kind)
    old, current = [], []
    page.configure_figure_binding(True, old.append)
    page.configure_figure_binding(False, current.append)
    assert old == current == []
    explorer.reveal("sources")
    box = page._figure_follow_check
    assert box.isVisible() and not box.isChecked()
    QtTest.QTest.mouseClick(box, QtCore.Qt.LeftButton, pos=QtCore.QPoint(8, box.height() // 2))
    assert old == [] and current == [True]


def test_dormant_plots_style_and_undo_are_one_origin_preserving_step_without_widgets(qtbot, monkeypatch):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    adapter = adapter_for(page, "distributions")
    record = _record(adapter)
    history = adapter.capture_transient()
    before = page.config()
    monkeypatch.setattr(PlotsWindow, "__init__", lambda *a, **k: pytest.fail("constructed a dormant view"))
    styled, moved = apply_look_to_record(record, PlotAppearance(background="light", title_size=17), history)
    assert len(moved.past) == len(history.past) + 1
    assert moved.extra["origin"] == history.extra["origin"]
    restored, undone = move_history(styled, moved)
    assert thaw_json(undone.base) == plot_config_to_json(before)
    assert encode_legacy(restored.spec, restored.resources).payload == encode_legacy(record.spec, record.resources).payload
    assert page.config() == before


def test_dormant_configuration_undo_requests_live_semantics_without_moving_state(qtbot):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    adapter = adapter_for(page, "distributions")
    page.set_histogram_column("frac_depth")
    record, history = _record(adapter), adapter.capture_transient()
    with pytest.raises(NeedsResidentHistory):
        move_history(record, history)
    assert adapter.capture_transient() == history


@pytest.mark.parametrize("mode", ["histogram_kde", "scatter_fit", "contours"])
def test_sampled_estimates_restore_without_estimating_and_keep_pending_draft(qtbot, monkeypatch, mode):
    from nanoscope.gui import plots_panel
    builders = H.page_builders()
    page = builders["distributions"]()
    qtbot.addWidget(page)
    if mode == "histogram_kde":
        page.set_kde_overlay(True)
    elif mode == "scatter_fit":
        page.set_plot_type(plots_panel.SCATTER)
        page.set_scatter_columns("dwell_s", "frac_depth")
        page.set_scatter_fit(plots_panel._SFIT_LINEAR)
    else:
        page.set_plot_type(plots_panel.KDE)
        page.set_kde_columns("dwell_s", "frac_depth")
    adapter = adapter_for(page, "distributions")
    applied = page.figure_applied_config()
    draft = replace(page.config(), histogram=replace(page.config().histogram,
                                                     bins=page.config().histogram.bins + 3))
    page._set_config(draft, draw=False, own=True)
    page._stale, page._stale_reason, page._stale_reasons = True, "bins", ["bins"]
    page._history_commit()
    record, transient = _record(adapter), adapter.capture_transient()
    native = encode_legacy(record.spec, record.resources)
    assert native.payload["phase2_render"]["nodes"]
    clone = builders["distributions"]()
    qtbot.addWidget(clone)
    target = adapter_for(clone, "distributions")
    monkeypatch.setattr(plots_panel.populations, "linear_fit", lambda *a, **k: pytest.fail("refit"))
    monkeypatch.setattr(plots_panel.PlotsWindow, "_kde_for", lambda *a, **k: pytest.fail("KDE estimate"))
    import scipy.stats
    monkeypatch.setattr(scipy.stats, "gaussian_kde", lambda *a, **k: pytest.fail("KDE contour estimate"))
    report = target.restore_record(record, mode="resume")
    assert not report.skipped, report
    target.install_transient(transient)
    assert clone.config() == draft
    assert clone._stale and clone._stale_reasons == ["bins"]
    assert clone.figure_applied_config() == applied
    again = target.capture_native()
    assert len(again.payload["phase2_render"]["nodes"]) == len(native.payload["phase2_render"]["nodes"])
    for left, right in zip(native.payload["phase2_render"]["nodes"], again.payload["phase2_render"]["nodes"]):
        pd.testing.assert_frame_equal(native.tables[left["table"]], again.tables[right["table"]])


def test_managed_suspend_never_blanks_or_publishes(qtbot, monkeypatch):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    adapter = adapter_for(page, "distributions")
    state = page.config()
    monkeypatch.setattr(page, "_persist_results", lambda: pytest.fail("published on suspension"))
    monkeypatch.setattr(page, "reset_state", lambda: pytest.fail("blanked on suspension"))
    adapter.suspend()
    assert page.config() == state


def test_initial_history_is_detached_and_starts_at_saved_record(qtbot):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    record = _record(adapter_for(page, "distributions"))
    first = initial_history(record)
    assert first.past == first.future == ()
    changed = first.to_dict()
    changed["base"]["style"]["title"] = "external mutation"
    assert thaw_json(first.base)["style"]["title"] != "external mutation"


def test_frozen_manual_inputs_restore_into_an_empty_catalog(qtbot, monkeypatch):
    from nanoscope.core.session import Session
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    adapter = adapter_for(page, "distributions")
    record = _record(adapter)
    clone = PlotsWindow(Session())
    qtbot.addWidget(clone)
    assert not clone._sources
    target = adapter_for(clone, "distributions")
    monkeypatch.setattr(clone, "_armed_redraw", lambda: pytest.fail("implicit analysis run"))
    report = target.restore_record(record)
    assert not report.skipped, report
    assert clone.selected_source_keys() == page.selected_source_keys()
    for key in page.selected_source_keys():
        pd.testing.assert_frame_equal(clone._sources[key].df, page._sources[key].df)
    assert clone._plot_armed and clone._plot.listDataItems()


def test_catalog_refresh_preserves_committed_state_until_one_explicit_plot(qtbot, monkeypatch):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    assert page._plot_armed and page._plot.listDataItems()
    page._history_commit()
    before_config = page.config()
    before_history = page.capture_history_capsule()
    before_sources = dict(page._sources)
    before_items = tuple(page._plot.items)
    before_recipes = page.figure_source_recipes()
    cluster, pca = object(), object()
    page._cluster_result, page._pca_result = cluster, pca
    incoming = [replace(source, df=source.df.iloc[:max(1, len(source.df) // 2)].copy())
                for source in before_sources.values()]
    draws = []
    real_draw = page._armed_redraw

    def draw_once():
        draws.append(True)
        return real_draw()

    monkeypatch.setattr(page, "_armed_redraw", draw_once)
    page.set_data(incoming, catalog_only=True)
    assert draws == []
    assert page.config() == before_config
    assert page.capture_history_capsule() == before_history
    assert tuple(page._plot.items) == before_items
    assert page.figure_source_recipes() == before_recipes
    assert page._cluster_result is cluster and page._pca_result is pca
    assert all(page._sources[key] is source for key, source in before_sources.items())
    # A failed replacement cannot destroy either the committed result or the
    # previously prepared valid catalog.
    with pytest.raises(AttributeError):
        page.set_data([object()], catalog_only=True)
    assert tuple(page._plot.items) == before_items
    assert page._cluster_result is cluster and page._pca_result is pca
    page._run_plot()
    assert draws == [True]
    assert page._figure_pending_sources is None
    assert page._cluster_result is None and page._pca_result is None
    for source in incoming:
        pd.testing.assert_frame_equal(page._sources[source.key].df, source.df)


def test_fresh_catalog_install_does_not_draw_or_replace_the_empty_canvas(qtbot, monkeypatch):
    from nanoscope.core.session import Session
    page = PlotsWindow(Session())
    qtbot.addWidget(page)
    monkeypatch.setattr(page, "_armed_redraw", lambda: pytest.fail("catalog drew"))
    monkeypatch.setattr(page, "_show_plot_prompt", lambda: pytest.fail("catalog replaced canvas"))
    frame = pd.DataFrame({"dwell_s": [.001, .002], "accepted": [True, True]})
    page.set_data({"input": frame}, catalog_only=True)
    assert page.selected_source_keys() == ["input"]
    assert not page._plot_armed
