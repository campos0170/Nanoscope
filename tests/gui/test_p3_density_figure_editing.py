"""Density editing generates once, keeps exact display units, and retains the legacy export."""
from dataclasses import replace

import nanoscope  # noqa: F401 Qt bootstrap
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.gui.main_window import MainWindow
from .test_analysis_launcher import _doc, _eventset, _install, _install_detection


@pytest.fixture
def density_window(qtbot, tmp_path, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    rng = np.random.default_rng(112)
    for index, name in enumerate(("apo500.dat", "apo600.dat")):
        doc_id = _install(window, _doc(tmp_path, name=name))
        events = _eventset([True] * 30)
        events.df["dwell_s"] = np.exp(rng.normal(-6.7 + index * .1, .35, 30))
        events.df["frac_depth"] = rng.normal(.3 + index * .08, .04, 30)
        _install_detection(window._session, doc_id, events)
    window._sync_export_action()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
        lambda *args: pytest.fail(str(args[2])))
    yield window
    for editor in list(getattr(window, "_report_figure_editors", ())):
        editor.close()


def test_density_menu_opens_one_retained_figure_with_project_style_and_exact_data(density_window, monkeypatch, tmp_path):
    import nanoscope.gui.main_window as MW
    import nanoscope.plot.figures as figures
    from nanoscope.core.figure_style import builtin_styles
    window = density_window
    assert window._act_edit_density.isEnabled() and window._act_export_density.isEnabled()
    owner = window._ensure_figure_workspace()
    style = owner.save_style("Apo density", look={"title_size_pt": 9.5, "line_width_pt": .625})
    owner.set_default_style(style.ref)
    monkeypatch.setattr(window, "_choose_density_amplitude", lambda ycol=None: ("frac_depth", "Fractional blockade"))
    monkeypatch.setattr(MW, "ask_export_path", lambda *a, **k: pytest.fail("opening wrote an export"))
    real_builder = figures.joint_density
    built = []
    def generate(*args, **kwargs):
        figure, axis = real_builder(*args, **kwargs)
        built.append(figure)
        return figure, axis
    monkeypatch.setattr(figures, "joint_density", generate)
    window._act_edit_density.trigger()
    assert len(built) == 1
    editor = window._report_figure_editors[-1]
    assert editor._figure is built[0]
    assert editor.style()["style_binding"]["ref"] == style.ref.to_dict()
    expected = built[0]._nanoscope_source_data
    pd.testing.assert_frame_equal(editor._data, expected, check_exact=True)
    display = editor._data.attrs["display_columns"]["display_x"]
    np.testing.assert_array_equal(editor._data[display], editor._data.dwell_s * 1000)
    assert editor._provenance["column_units"] == {"dwell_s": "s", display: "ms", "frac_depth": "1"}
    assert editor._provenance["quantity_display"] == editor._data.attrs["quantity_display"]
    assert len(editor._provenance["source_catalog"]) == 2
    monkeypatch.setattr(figures, "joint_density", lambda *a, **k: pytest.fail("style regenerated density"))
    import scipy.stats
    monkeypatch.setattr(scipy.stats, "gaussian_kde", lambda *a, **k: pytest.fail("style recomputed KDE"))
    vertices = [(collection, [path.vertices.copy() for path in collection.get_paths()])
                for axis in editor._figure.axes for collection in axis.collections]
    assert editor.apply_style(builtin_styles()[1])
    title = next(c for c in editor._text_controllers if c.panel == "density.dwell_marginal")
    title.set_style_property("title", "text", "Apo density Ω")
    assert editor._figure.axes[1].get_title() == "Apo density Ω"
    assert editor.undo_style()
    assert editor._figure.axes[1].get_title() != "Apo density Ω"
    for collection, paths in vertices:
        for got, expected_path in zip(collection.get_paths(), paths):
            np.testing.assert_array_equal(got.vertices, expected_path)
    output = tmp_path / "density-edited.svg"
    assert editor.export_image(path=output) == output and output.exists()
    assert len(built) == 1


@pytest.mark.parametrize("change", ["epoch", "same_pixels_new_source", "rows"])
def test_density_chooser_cannot_apply_old_sources_to_changed_project(density_window, monkeypatch, change):
    import nanoscope.plot.figures as figures
    window = density_window
    def choose(ycol=None):
        result = window._session.results.all()[0]
        if change == "epoch":
            window._figure_project_epoch += 1
        elif change == "same_pixels_new_source":
            window._session.results.put(replace(result))
        else:
            result.payload.df.loc[0, "frac_depth"] += .01
        return "frac_depth", "Fractional blockade"
    monkeypatch.setattr(window, "_choose_density_amplitude", choose)
    monkeypatch.setattr(figures, "joint_density", lambda *a, **k: pytest.fail("stale source generated"))
    assert window._edit_density_figure() is None
    assert not getattr(window, "_report_figure_editors", ())
    assert "changed" in window.statusBar().currentMessage()


def test_density_cancel_does_not_generate(density_window, monkeypatch):
    import nanoscope.plot.figures as figures
    monkeypatch.setattr(density_window, "_choose_density_amplitude", lambda *a: None)
    monkeypatch.setattr(figures, "joint_density", lambda *a, **k: pytest.fail("cancel generated"))
    assert density_window._edit_density_figure() is None


def test_density_source_change_during_build_discards_unopened_artists(density_window, monkeypatch):
    import nanoscope.plot.figures as figures
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    window = density_window
    figure = Figure(); FigureCanvasAgg(figure)
    axis = figure.subplots()
    clears = []
    clear = figure.clear
    def clear_once(*args, **kwargs):
        clears.append(True)
        return clear(*args, **kwargs)
    monkeypatch.setattr(figure, "clear", clear_once)
    def generate(*args, **kwargs):
        window._figure_project_epoch += 1
        return figure, axis
    monkeypatch.setattr(figures, "joint_density", generate)
    assert window._edit_density_figure(ycol="frac_depth") is None
    assert clears == [True] and not getattr(window, "_report_figure_editors", ())


def test_legacy_density_export_still_uses_save_figure(density_window, monkeypatch, tmp_path):
    import nanoscope.gui.main_window as MW
    import nanoscope.plot.figures as figures
    from matplotlib.figure import Figure
    target = tmp_path / "density.pdf"
    figure = Figure()
    calls = []
    monkeypatch.setattr(MW, "ask_export_path", lambda *a, **k: target)
    monkeypatch.setattr(figures, "joint_density", lambda *a, **k: (figure, None))
    monkeypatch.setattr(figures, "save_figure", lambda fig, path: calls.append((fig, path)) or {"image": target})
    density_window._export_density(ycol="frac_depth")
    assert calls == [(figure, str(target))]
    assert not getattr(density_window, "_report_figure_editors", ())


def test_density_unknown_registry_preserves_native_artwork(density_window, monkeypatch):
    from nanoscope.core.project import AnalyticalWorkspace
    import nanoscope.gui.report_figure_window as reports
    window = density_window
    owner = window._ensure_figure_workspace()
    raw = {"version": 99, "unknown": [None, "preserve"]}
    owner.install_workspace(AnalyticalWorkspace(extra={"figure_styles_v1": raw}))
    opened = []
    def retain(owner, figure, data, **kwargs):
        opened.append((figure, data, kwargs))
        return "retained"
    monkeypatch.setattr(reports, "open_report_figure", retain)
    assert window._edit_density_figure(ycol="frac_depth") == "retained"
    assert opened[0][2]["style"] is None
    assert owner.capture_workspace().extra["figure_styles_v1"] == raw
    opened[0][0].clear()


def test_density_refuses_stale_sampling_rate_before_amplitude_or_generation(density_window, monkeypatch):
    import nanoscope.plot.figures as figures
    window = density_window
    result = window._density_results()[0]
    live = window._session.document(result.doc_id).meta.sample_rate_hz
    result.payload.provenance["fs"] = live / 2
    monkeypatch.setattr(window, "_choose_density_amplitude", lambda *a: pytest.fail("stale table asked a question"))
    monkeypatch.setattr(figures, "joint_density", lambda *a, **k: pytest.fail("stale table generated"))
    assert window._edit_density_figure() is None
    assert "re-run detection" in window.statusBar().currentMessage()
    assert not getattr(window, "_report_figure_editors", ())
