"""Real MainWindow factories are lazy and figures use UUID routes throughout the shell."""
import pytest

from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.analysis_catalog import PAGE_IDS


def test_existing_analysis_route_raises_recent_and_new_creates_another(qapp, qtbot):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window._open_iv()
    owner = window._figure_workspace
    first = owner.current_id()
    window._analysis_explorer.close()
    assert not window._analysis_explorer.isVisible()
    window._open_iv()
    assert window._analysis_explorer.isVisible()
    assert owner.current_id() == first and len(owner.records()) == 1
    second = owner.create("iv")
    assert first != second and len(owner.records()) == 2
    assert owner.page(first) is not owner.page(second)
    assert window._analysis_explorer.current_page_id() == second
    assert window._iv_window is owner.page(second)


@pytest.mark.parametrize("kind", PAGE_IDS)
def test_all_eight_kinds_have_independent_record_factory_without_running_update(qapp, qtbot, kind):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    key = owner.create(kind)
    assert owner.get(key).spec.kind == kind
    assert window._has_work_to_lose() and window._visible_project_dirty()
    assert owner.page(key) is not None
    adapter = window._analysis_explorer.adapter(key)
    assert adapter.key == key and adapter.kind == kind
    assert owner.history(key).mark == 0


@pytest.mark.parametrize("kind,view,forbidden", [
    ("noise", "spectrogram", ("_refresh_view", "_recompute")),
    ("event_waveforms", "ensemble", ("_redraw_density", "_redraw_ensemble")),
])
def test_new_view_variant_installs_empty_snapshot_without_calculation(qapp, qtbot, monkeypatch, kind, view, forbidden):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    factory = owner._factory
    calls = []
    def guarded_factory(kind, key):
        page = factory(kind, key)
        for method in forbidden:
            monkeypatch.setattr(page, method, lambda *args, name=method, **kwargs: calls.append(name))
        return page
    owner._factory = guarded_factory
    key = owner.create(kind, initial_recipe={"view": view})
    assert owner.page(key) is not None and not calls
    assert owner.page(key).current_view().lower() == view
    assert owner.get(key).spec.result.status == "empty"


def test_new_from_recipe_uses_analysis_settings_with_new_words_and_project_look(qapp, qtbot):
    from dataclasses import replace
    from nanoscope.gui.plots_panel import PlotConfig, plot_config_to_json
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    template = PlotConfig()
    template = replace(template, style=replace(template.style, title="Old title", font_size_pt=23),
                       axes=replace(template.axes, x_min="1", x_max="8", manual_x=(1, 8)))
    owner = window._ensure_figure_workspace()
    key = owner.create("distributions", initial_recipe={"plot_template": plot_config_to_json(template)})
    page = owner.page(key)
    assert page is not None
    config = page.config()
    assert config.style.title == "" and config.style.font_size_pt == 7
    assert config.style.font_family == "Arial"
    assert (config.style.title_font_size_pt, config.style.axis_label_font_size_pt,
            config.style.tick_font_size_pt, config.style.legend_font_size_pt) == (7, 7, 6, 6)
    assert not config.axes.x_min and config.axes.manual_x is None
    assert owner.history(key).mark == 0


def test_deferred_new_figure_style_undo_returns_to_project_look_without_materialization(qapp, qtbot):
    from nanoscope.core.figure_codecs import encode_legacy
    from nanoscope.gui.figure_adapter_plots import _native_config
    from nanoscope.gui.plot_appearance import PlotAppearance
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    owner.resident_limit = 1
    owner.create("iv")
    key = owner.create("distributions")
    assert owner.page(key) is None
    native = _native_config(encode_legacy(owner.get(key).spec, owner.get(key).resources))
    assert native.style.font_size_pt == 7
    assert native.style.title_font_size_pt == 7
    assert owner.history(key).mark == 0
    assert owner.apply_look_to_figure(key, PlotAppearance(title_size=18), keep_hand=False)
    assert owner.undo(key) and owner.page(key) is None
    restored = _native_config(encode_legacy(owner.get(key).spec, owner.get(key).resources))
    assert restored.style.font_size_pt == 7
    assert restored.style.title_font_size_pt == 7
    assert restored.style.font_family == "Arial"
