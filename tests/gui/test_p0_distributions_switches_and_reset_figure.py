"""Phase 0, group G3: Distributions' duplicate switches go, and Reset appearance is Reset figure.

* The Style column of Distributions no longer has **Fit values on plot**, **R² on figure**,
  **KS statistic on figure** or **Title weight**. Which numbers the fit label shows is its
  Numbers row, and the title's bold and italic are its Text style.
* Saved projects, templates and JSON sidecars written before that still carry the old keys
  (``fit_values_on_plot``, ``show_r2``, ``show_gof``, ``title_bold``, ``title_italic``).
  They open and the figure looks the same: the keys are folded into the label's numbers and
  the title's style on the way in, and nothing writes them again. The project format stays 12.
* **Reset appearance** is **Reset figure**, in the page toolbar of every Analysis Explorer
  page. It puts the words and the look back to their defaults in one Undo step. The study
  figure editor, which has no such toolbar, keeps it in its Appearance section.

Real routes: clicks go to the real widgets (the Numbers boxes, the toolbar button), the Undo
key goes through the window, a template is recalled from its combo, a sidecar is imported from
a file, and a project is saved with ``MainWindow._write_project`` and opened with
``_open_project`` until ``projectRestored``. The old-format payloads are written out by hand.
What is asserted is what is DRAWN: the fit label's plain text, the title's ``QFont``, the
axis label text. The "as the old build drew it" strings were captured from the build before
the switches went (same synthetic frame, same seed). Synthetic data only; offscreen.
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import fields, replace
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core import project as project_format
from nanoscope.core.analysis_workspace import PLOTS_ARTIFACT_ID
from nanoscope.core.session import Session
from nanoscope.gui import figure_text, plots_panel as pp
from nanoscope.gui.figure_text import AxisState, TextStyle, figure_text_for
from nanoscope.gui.fit_label import FIT_FIELDS, FitLabelState
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plots_panel import (
    PlotConfig, PlotsWindow, StyleSettings, plot_config_from_json, plot_config_to_json,
)

from .test_p0_reset_on_quantity_change import spy_on_comparison_renders
from .test_plot_appearance_history import _build as _host, _study_window
from .test_plot_window_routes import (
    _choose, _click, _click_tab, _edit_double_click, _legend_row, _names, _press_undo,
    _restored_main_window, _show, window_handle,
)
from .test_plots_reopen_figure import _main, _recordings, _settle

LEFT = QtCore.Qt.MouseButton.LeftButton
RIGHT = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier

#: The five keys the old Style carried. Written out here, not read from the package.
OLD_KEYS = ("fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic")

#: What the build BEFORE the switches went drew for this window's synthetic frame (seed 0, 300
#: events), captured from that build: its default fit label, the label with Fit values on plot
#: ticked, the same with KS statistic on figure unticked, and the scatter line with and
#: without R² on figure.
AS_THE_OLD_BUILD_DREW = {
    "hist": "τ = 1.903 ms · n = 300",
    "hist_values": "τ = 1.903 ms · n = 300 · BIC = -3152.82 · KS = 0.038",
    "hist_values_no_ks": "τ = 1.903 ms · n = 300 · BIC = -3152.82",
    "line": "y = 1.992·x + 5.027 · R² = 0.985 · n = 300",
    "line_no_r2": "y = 1.992·x + 5.027 · n = 300",
}

#: Wording of the user's own that names the two numbers an old switch could take off, and what
#: that same build drew for it, per fit kind and per the one switch that mattered for it (the
#: histogram has no R², the line no KS): captured from the build before the switches went.
TYPED_WORDING = "{equation} | {r2} | {ks} | {bic}"
TYPED_AS_THE_OLD_BUILD_DREW = {
    ("hist", True): "τ = 1.903 ms | — | KS = 0.038 | BIC = -3152.82",     # KS statistic on figure
    ("hist", False): "τ = 1.903 ms | — | — | BIC = -3152.82",
    ("line", True): "y = 1.992·x + 5.027 | R² = 0.985 | — | —",           # R² on figure
    ("line", False): "y = 1.992·x + 5.027 | — | — | —",
}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    QtCore.QSettings().remove(PlotsWindow._TPL_SETTINGS_KEY)
    yield
    figure_text.GATE.reset()
    QtCore.QSettings().remove(PlotsWindow._TPL_SETTINGS_KEY)


# -- builders ---------------------------------------------------------------------------------

def _df(seed: int = 0, n: int = 300, slope: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "xcol": x, "ycol": slope * x + 5.0 + rng.normal(0.0, 0.3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4,
        "depth_pA": rng.normal(50.0, 5.0, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


def _distributions(qtbot, kind: str = "hist", frame: pd.DataFrame | None = None) -> PlotsWindow:
    """Distributions, shown, with a fit drawn: the exponential on dwell times or the line."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    w.set_data({"probe": _df() if frame is None else frame})
    if kind == "hist":
        w.set_plot_type("Histogram")
        w.set_histogram_column("dwell_s")
        w.set_fit("exponential")
    else:
        w.set_plot_type("Scatter")
        w.set_scatter_columns("xcol", "ycol")
        w.set_scatter_fit("linear")
    _settle()
    return w


def _label(w: PlotsWindow) -> str:
    item = w._fit_label_item
    return item.plain_text() if item.isVisible() else "(hidden)"


def _title_font(w: PlotsWindow) -> tuple[bool, bool]:
    font = w._plot.titleLabel.item.font()
    return font.bold(), font.italic()


def _tick(w: PlotsWindow, name: str) -> None:
    """A real click on one box of the fit label's Numbers row, in the Appearance section."""
    w._show_control_section("Appearance")
    box = w._appearance_widget.fit_label_fields[name]
    assert box.isVisible(), name
    QTest.mouseClick(box, LEFT)
    QtWidgets.QApplication.processEvents()


def _type_wording(w: PlotsWindow, text: str) -> None:
    """Your own words in the Fit label group's Text box, typed key by key."""
    w._show_control_section("Appearance")
    box = w._appearance_widget.fit_label_template
    assert box.isVisible()
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box, text)
    QtWidgets.QApplication.processEvents()


# ==============================================================================================
# 1. the controls are gone, and the Numbers row says which numbers
# ==============================================================================================

def test_the_style_column_has_none_of_the_removed_controls(qtbot):
    w = _distributions(qtbot)
    w._show_control_section("Appearance")
    for attr in ("_fit_values_check", "_r2_check", "_gof_check", "_title_bold", "_title_italic"):
        assert not hasattr(w, attr), attr
    shown = {box.text() for box in w._style_group.findChildren(QtWidgets.QCheckBox)}
    shown |= {label.text() for label in w._style_group.findChildren(QtWidgets.QLabel)}
    for gone in ("Fit values on plot", "R² on figure", "KS statistic on figure",
                 "Title weight", "Bold", "Italic"):
        assert gone not in shown, gone
    # ...and every number they could show is one box of the Numbers row.
    assert tuple(w._appearance_widget.fit_label_fields) == FIT_FIELDS
    assert {"bic", "ks", "r2"} <= set(w._appearance_widget.fit_label_fields)
    assert StyleSettings().fit_label_state().fields == ("equation", "r2", "n")


def test_every_box_of_the_numbers_row_is_in_view_in_the_controls_column(qtbot):
    """With Fit values on plot gone, BIC and KS are ticked here only. Six boxes on one line
    were wider than the 344 px column (BIC and KS off its right edge, a horizontal scroll)."""
    w = _distributions(qtbot)
    w._show_control_section("Appearance")
    QtWidgets.QApplication.processEvents()
    viewport = w._section_area.viewport()
    for name, box in w._appearance_widget.fit_label_fields.items():
        right = box.mapTo(viewport, box.rect().bottomRight()).x()
        assert box.isVisible() and 0 <= right <= viewport.width(), (name, right, viewport.width())
    assert w._section_area.horizontalScrollBar().maximum() == 0


def test_the_numbers_row_gives_exactly_what_fit_values_on_plot_and_ks_gave(qtbot):
    w = _distributions(qtbot)
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist"]
    _tick(w, "bic")
    _tick(w, "ks")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"]
    _tick(w, "ks")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values_no_ks"]
    _tick(w, "bic")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist"]


def test_the_numbers_row_gives_exactly_what_r2_on_figure_gave(qtbot):
    w = _distributions(qtbot, "line")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["line"]
    _tick(w, "r2")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["line_no_r2"]
    _tick(w, "r2")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["line"]


def test_the_refusal_follows_the_r2_box_the_way_it_does_in_every_other_window(qtbot):
    """R² on figure used to drop the number and keep the refusal. The Numbers row says whether
    R² is on the figure now, and its rule is fit_label's: a ticked R² prints the refusal that
    says an R² cannot be judged, an unticked one (with no {r2} in the words) makes no R² claim.
    Details still says."""
    flat = _df()
    flat["ycol"] = 7.0                                   # y does not vary: R² is undefined
    w = _distributions(qtbot, "line", flat)
    assert _label(w).endswith("R² undefined (y does not vary)")
    _tick(w, "r2")
    assert "R²" not in _label(w) and _label(w).startswith("y = ")
    assert "R² is undefined" in w.details_text()


def test_the_r2_boxs_tip_says_what_the_label_does_with_it(qtbot):
    """Each claim of the tip, driven: ticked, a refusal is printed (after the other numbers);
    unticked, the automatic text says nothing about R² and Details still does; wording of your
    own that names {r2} prints it either way."""
    flat = _df()
    flat["ycol"] = 7.0                                   # y does not vary: R² is undefined
    w = _distributions(qtbot, "line", flat)
    tip = w._appearance_widget.fit_label_fields["r2"].toolTip()
    for claim in ("undefined", "residual degrees of freedom", "Details still says", "{r2}"):
        assert claim in tip, claim
    refusal = "R² undefined (y does not vary)"
    assert _label(w).endswith(f"n = 300 · {refusal}")
    _tick(w, "r2")
    assert "R²" not in _label(w) and "R² is undefined" in w.details_text()
    _type_wording(w, "{equation} | {r2}")                # the words name it: printed either way
    assert _label(w).endswith(f" | {refusal}"), _label(w)
    # The other refusal the tip names: a line through two points has no residual degrees of freedom.
    two = _distributions(qtbot, "line", _df().iloc[:2].copy())
    assert _label(two).endswith("R² = 1.000 (exact by construction)")
    _tick(two, "r2")
    assert "R²" not in _label(two) and "exact by construction" in two.details_text()


@pytest.mark.parametrize("make", ["iv", "vdep"])
def test_unticking_r2_leaves_the_refusal_to_details_in_the_other_windows_with_an_r2(
        qapp, qtbot, tmp_path, make):
    """The R² box's tip is the same text in every window that has the Numbers row, and says
    Details still carries the refusal: a line through two recordings has no residual degrees
    of freedom, so its R² is exact by construction, in the label while ticked and in Details
    either way."""
    from nanoscope.core.session import Session as _Session
    from nanoscope.gui.iv_panel import IVWindow
    from nanoscope.gui.voltage_dep_panel import VDepWindow
    from . import test_iv, test_voltage_dep as vd

    session = _Session()
    if make == "iv":
        for mV in (-200.0, -500.0):
            session.add_document(test_iv._current_doc(
                tmp_path, i_open_pA=100.0 * mV, nominal_mV=mV, name=f"p{int(-mV)}.dat"))
        w = IVWindow(session)
    else:
        for mV in vd._LADDER_MV[:2]:
            v = mV / 1000.0
            doc_id = session.add_document(vd._current_doc(
                tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
            vd._install_detection(
                session, doc_id,
                vd._constant_df(n=vd._N_PER_DOC, dwell_s=(v + 1.2) * 1e-3,
                                depth_pA=-100.0 * v, frac_depth=-0.1 * v),
                duration_s=vd._N_PER_DOC / (vd._EXP_A * np.exp(v / vd._EXP_V0)))
        w = VDepWindow(session)
        w.set_fit("exponential")
    qtbot.addWidget(w)
    w.resize(900, 760)
    w.show()
    qtbot.waitExposed(w)
    box = w._appearance_widget.fit_label_fields["r2"]
    assert "Details still says" in box.toolTip()
    assert "exact by construction" in w.fit_label_text()
    w._workflow.set_section("Appearance")
    assert box.isVisible()
    # On the indicator: in a wide strip the box is wider than its click area, so its middle misses.
    QTest.mouseClick(box, LEFT, NOMOD, QtCore.QPoint(8, box.height() // 2))
    QtWidgets.QApplication.processEvents()
    assert "R²" not in w.fit_label_text() and "exact by construction" not in w.fit_label_text()
    assert "exact by construction" in w.details_text().lower(), w.details_text()


def test_the_residual_strips_tooltip_sends_the_chi_square_to_the_fit_label(qtbot):
    w = _distributions(qtbot)
    tip = w._residual_check.toolTip()
    assert "fit label" in tip and "values box" not in tip
    numbers = w.config().style.fit_label_state()
    w._residual_check.click()                           # the strip's number follows the equation
    text = _label(w)
    assert text.index("τ =") < text.index("χ²") < text.index("n = 300"), text
    assert "χ²/dof" in text
    # It writes into the equation's text and leaves the Numbers row, the label's own state, as it was.
    assert w.config().style.fit_label_state() == numbers
    assert [name for name, box in w._appearance_widget.fit_label_fields.items()
            if box.isChecked()] == ["equation", "r2", "n"]


def test_the_figures_own_menu_edits_the_label_starting_from_the_numbers_row(qtbot):
    w = _distributions(qtbot)
    _tick(w, "bic")
    _tick(w, "ks")
    controller = figure_text_for(w._plot)
    controller.last_menu = None
    _click(w._plot, w._fit_label_item.sceneBoundingRect().center(), button=RIGHT)
    menu = controller.last_menu
    assert menu is not None and "Edit fit label…" in _names(menu), menu and _names(menu)
    _choose(menu, "Edit fit label…")
    assert controller.editor is not None
    assert controller.editor.text() == "{equation} · {r2} · {n} · {bic} · {ks}"
    controller.editor.cancel()


# ==============================================================================================
# 2. old payloads: the five keys fold into the label's numbers and the title's style
# ==============================================================================================

def _fields_the_old_build_drew(saved, values_on: bool, show_r2: bool, show_gof: bool):
    """The old rule, written out: a label nobody saved had BIC and KS when Fit values on plot
    was on; a saved one had its own fields; then R² on figure and KS statistic on figure took
    their number off."""
    fields = (list(saved) if saved is not None
              else ["equation", "r2", "n"] + (["bic", "ks"] if values_on else []))
    return tuple(name for name in FIT_FIELDS if name in fields
                 and not (name == "r2" and not show_r2) and not (name == "ks" and not show_gof))


@pytest.mark.parametrize("saved", [None, ["equation", "n"], ["equation", "r2", "n", "ci", "bic", "ks"]],
                         ids=["no saved label", "saved: equation, n", "saved: all six"])
@pytest.mark.parametrize("show_gof", [True, False])
@pytest.mark.parametrize("show_r2", [True, False])
@pytest.mark.parametrize("values_on", [False, True])
def test_every_setting_of_the_old_switches_folds_to_the_fields_the_old_build_drew(
        values_on, show_r2, show_gof, saved):
    payload = plot_config_to_json(PlotConfig())
    payload["style"].update(fit_values_on_plot=values_on, show_r2=show_r2, show_gof=show_gof)
    payload["style"]["fit_label"] = {} if saved is None else {"fields": saved}
    state = plot_config_from_json(payload).style.fit_label_state()
    assert state.fields == _fields_the_old_build_drew(saved, values_on, show_r2, show_gof)
    assert state.visible and state.template == "" and state.placement == "top-left"


def test_title_weight_folds_into_the_titles_text_style_where_it_has_none():
    payload = plot_config_to_json(PlotConfig())
    payload["style"].update(title="Dwell", title_bold=True, title_italic=False)
    style = plot_config_from_json(payload).style
    assert (style.text_style("title").bold, style.text_style("title").italic) == (True, None)
    assert style.title == "Dwell"
    # A title style that already says "not bold" wins, as it did over the old box.
    payload["style"]["text_styles"] = {"title": {"bold": False, "size_pt": 18.0}}
    style = plot_config_from_json(payload).style
    assert style.text_style("title").bold is False and style.text_style("title").size_pt == 18.0
    payload["style"].update(title_bold=True, title_italic=True)
    style = plot_config_from_json(payload).style
    assert (style.text_style("title").bold, style.text_style("title").italic) == (False, True)


def test_nothing_writes_the_old_keys_any_more_in_the_current_project_format():
    style = plot_config_to_json(PlotConfig())["style"]
    assert not set(OLD_KEYS) & set(style)
    assert not set(OLD_KEYS) & {field.name for field in fields(StyleSettings)}
    assert project_format.VERSION == 13
    # A config that went through a fold writes none of them either, and reads back equal.
    payload = plot_config_to_json(PlotConfig())
    payload["style"].update(fit_values_on_plot=True, show_r2=False, title_bold=True)
    folded = plot_config_from_json(payload)
    written = plot_config_to_json(folded)
    assert not set(OLD_KEYS) & set(written["style"])
    assert plot_config_from_json(written) == folded


def _old_style(**changes) -> dict:
    """The Style mapping an older build wrote: today's keys with the five old ones back, and
    the label and element styles empty because nothing had set them."""
    style = plot_config_to_json(PlotConfig())["style"]
    style.update(fit_values_on_plot=False, show_r2=True, show_gof=True,
                 title_bold=False, title_italic=False, fit_label={}, text_styles={})
    style.update(changes)
    return style


def _in_a_fresh_window(qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    w.set_data({"probe": _df()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    return w


def test_an_old_template_recalled_from_its_combo_draws_the_old_numbers_and_title(qtbot):
    old = plot_config_to_json(PlotConfig(
        plot_type=pp.HISTOGRAM, histogram=pp.HistogramConfig(column="dwell_s"),
        fit=pp.FitConfig(kind="exponential")))
    old["style"] = _old_style(title="Dwell, pore A", fit_values_on_plot=True,
                              title_bold=True, title_italic=True)
    PlotsWindow._store_templates({"An older template": old})
    w = _in_a_fresh_window(qtbot)
    w._load_templates_into_combo()
    w._template_combo.setCurrentText("An older template")        # the real recall route
    QtWidgets.QApplication.processEvents()
    w.set_fit("exponential")                                     # a template waits for Apply
    _settle()
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"]
    assert w._plot.titleLabel.text.endswith("Dwell, pore A")
    assert _title_font(w) == (True, True)


def test_an_old_sidecar_imports_unchanged_and_its_old_keys_are_not_reported_as_not_taken(
        qtbot, tmp_path):
    source = _distributions(qtbot)
    sidecar = tmp_path / "older.json"
    assert source.export_config_json(sidecar) is not None
    old = json.loads(sidecar.read_text(encoding="utf-8"))
    assert not set(OLD_KEYS) & set(old["style"])           # this build wrote it; now age it
    old["style"].update(_old_style(title="Dwell, pore A", fit_values_on_plot=True,
                                   show_gof=False, title_bold=True))
    sidecar.write_text(json.dumps(old), encoding="utf-8")

    w = _in_a_fresh_window(qtbot)
    assert w.import_config_json(sidecar)
    _settle()
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values_no_ks"]
    assert _title_font(w) == (True, False)
    # Every key the file named was taken (the same file's own export reports nothing lost).
    assert w._config_losses.names == (), w._config_losses.names
    # Saved again, the figure is the same and the old keys are gone.
    again = tmp_path / "again.json"
    assert w.export_config_json(again) is not None
    written = json.loads(again.read_text(encoding="utf-8"))
    assert not set(OLD_KEYS) & set(written["style"])
    assert written["style"]["fit_label"]["fields"] == ["equation", "r2", "n", "bic"]


@pytest.mark.parametrize("kind", ["hist", "line"])
def test_wording_that_names_r2_or_ks_draws_a_dash_where_its_switch_was_off(qtbot, tmp_path, kind):
    """The old switches took R² and KS out of the readout, so a ``{r2}`` or ``{ks}`` in the
    user's own words printed a dash while its box was unticked. Every setting of the three
    boxes, through the real import of a sidecar file, against what that build drew."""
    w = _distributions(qtbot, kind)
    sidecar = tmp_path / "older.json"
    assert w.export_config_json(sidecar) is not None
    base = json.loads(sidecar.read_text(encoding="utf-8"))
    for values_on in (False, True):
        for show_r2 in (True, False):
            for show_gof in (True, False):
                old = json.loads(json.dumps(base))
                old["style"].update(_old_style(
                    fit_values_on_plot=values_on, show_r2=show_r2, show_gof=show_gof,
                    fit_label={"template": TYPED_WORDING}))
                sidecar.write_text(json.dumps(old), encoding="utf-8")
                assert w.import_config_json(sidecar)
                _settle()
                switch = show_gof if kind == "hist" else show_r2
                setting = (values_on, show_r2, show_gof)
                assert _label(w) == TYPED_AS_THE_OLD_BUILD_DREW[(kind, switch)], setting
                assert w._config_losses.names == (), (setting, w._config_losses.names)
    # The last file had both boxes unticked. Saved again, the words that stand are the ones
    # drawn: no token is left that names a number the figure does not show.
    again = tmp_path / "again.json"
    assert w.export_config_json(again) is not None
    assert json.loads(again.read_text(encoding="utf-8"))["style"]["fit_label"]["template"] == (
        "{equation} | — | — | {bic}")


def _older_plots_project(src: Path, dst: Path, style: dict, *, replace_style: bool) -> int:
    """Encode an actual v12 native artifact, including its tables and blobs."""
    from nanoscope.core.figure_codecs import record_from_artifact, encode_legacy
    from nanoscope.core.figure_spec import FIGURE_ARTIFACT_KIND
    state = project_format.load_project(src)
    rewritten = 0
    artifacts = []
    for artifact in state.workspace.artifacts:
        if artifact.kind == FIGURE_ARTIFACT_KIND:
            record = record_from_artifact(artifact)
            if record.spec.kind == "distributions":
                artifact = encode_legacy(record.spec, record.resources)
        if artifact.artifact_id == PLOTS_ARTIFACT_ID:
            tree = artifact.payload["config"]["fields"]["style"]["fields"]
            if replace_style:
                tree.clear()
            tree.update(style)
            rewritten += 1
        artifacts.append(artifact)
    state.workspace.artifacts = artifacts
    project_format.save_project(dst, state)
    with zipfile.ZipFile(dst) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members[project_format.MANIFEST_NAME])
    manifest["version"] = 12
    members[project_format.MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return rewritten


def _rewritten_as_an_older_build_wrote_it(src: Path, dst: Path, style: dict) -> int:
    return _older_plots_project(src, dst, style, replace_style=True)


def _manifest_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read(project_format.MANIFEST_NAME).decode("utf-8")


def test_an_old_project_opens_with_the_same_label_and_title_and_is_saved_without_the_old_keys(
        qapp, qtbot, tmp_path):
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plots.set_title("Dwell, pore A")
    _settle()
    for name in ("bic", "ks"):                        # the figure an old Fit values on plot made
        plots._appearance_widget.fit_label_fields[name].click()
    _settle()
    reference = plots._fit_label_item.plain_text()
    assert "BIC = " in reference and "KS = " in reference
    for name in ("bic", "ks"):
        plots._appearance_widget.fit_label_fields[name].click()
    _settle()
    saved = tmp_path / "now.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    for key in OLD_KEYS:                              # this build writes none of them
        assert f'"{key}"' not in _manifest_text(saved), key

    # The same project as the old build wrote it: no saved label or element styles, Fit
    # values on plot ticked, R² on figure off, Title weight bold and italic. The Style tree
    # is written out by hand.
    current = plots.config().style
    old_tree = {
        "title": "Dwell, pore A", "x_label": "", "y_label": "", "show_legend": True,
        "font_family": "", "font_size_pt": 10, "data_color": current.data_color,
        "fit_color": current.fit_color, "point_size": 7, "line_width": 1, "background": "dark",
        "export_w": 0, "export_h": 0, "fit_values_on_plot": True, "label_overrides": {},
        "legend_pos": None, "legend_font_size_pt": 0, "legend_counts": True,
        "legend_include": {}, "show_grid": True, "title_font_size_pt": 0,
        "axis_label_font_size_pt": 0, "tick_font_size_pt": 0, "title_bold": True,
        "title_italic": True, "show_r2": False, "show_gof": True, "show_ci_band": False,
        "show_provenance": False, "show_residuals": False, "heatmap_colormap": "viridis",
        "text_styles": {}, "fit_label": {}}
    older = tmp_path / "older.nanoscope"
    assert _rewritten_as_an_older_build_wrote_it(saved, older, old_tree) == 1
    assert '"fit_values_on_plot": true' in _manifest_text(older)

    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(older))
    again._open_plots()
    _settle()
    reopened = again._plots_window
    assert reopened._fit_label_item.plain_text() == reference
    assert reopened._plot.titleLabel.text.endswith("Dwell, pore A")
    assert _title_font(reopened) == (True, True)
    assert reopened.config().style.fit_label_state().fields == ("equation", "n", "bic", "ks")
    assert reopened.config().style.text_style("title").bold is True

    # An edit makes the page save itself again: the project it writes now has none of the five
    # keys, uses current version 13, and opens to the same label and the (edited) bold title.
    box = reopened._appearance_widget.title
    box.setFocus()
    box.end(False)
    QTest.keyClicks(box, "!")
    QTest.keyClick(box, QtCore.Qt.Key.Key_Return)
    qtbot.wait(800)                                   # past the coalescing timer
    resaved = tmp_path / "resaved.nanoscope"
    assert again._write_project(str(resaved))
    again._dirty = False
    text = _manifest_text(resaved)
    for key in OLD_KEYS:
        assert f'"{key}"' not in text, key
    assert json.loads(text)["version"] == project_format.VERSION == 13
    with qtbot.waitSignal(window.projectRestored, timeout=20000):
        window._open_project(str(resaved))
    window._open_plots()
    _settle()
    last = window._plots_window
    assert last._fit_label_item.plain_text() == reference
    assert last._plot.titleLabel.text.endswith("Dwell, pore A!")
    assert _title_font(last) == (True, True)
    window._dirty = False


def _with_old_keys_added(src: Path, dst: Path, **old_keys) -> int:
    return _older_plots_project(src, dst, old_keys, replace_style=False)


@pytest.mark.parametrize("case", ["histogram, own words, KS statistic on figure off",
                                  "line, automatic text, R² on figure off"])
def test_an_old_project_with_an_unticked_r2_or_ks_switch_opens_to_what_the_old_build_drew(
        qapp, qtbot, tmp_path, case):
    """The same two projects, as the old build drew them on reopening (captured from it): the
    histogram's own words with their ``{ks}`` as a dash, and the line without its R²."""
    window, plots = _recordings(qtbot, tmp_path)
    if case.startswith("histogram"):
        plots.set_plot_type("Histogram")
        plots.set_histogram_column("dwell_s")
        plots.set_fit("exponential")
        # Fixture setup uses the public config boundary; physical label editing is
        # exercised above on the visible standalone controls and by text-route tests.
        plots.apply_config(replace(plots.config(), style=plots.config().style.with_fit_label(
            FitLabelState(template=TYPED_WORDING))))
        old, expected = {"show_gof": False}, "τ = 2.304 ms | — | — | BIC = -1212.74"
    else:
        plots.set_plot_type("Scatter")
        plots.set_scatter_columns("frac_depth", "dwell_s")
        plots.set_scatter_fit("linear")
        old, expected = {"show_r2": False}, "y = -5.54·x + 4.633 (y in ms) · n = 120"
    _settle()
    saved = tmp_path / "now.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    older = tmp_path / "older.nanoscope"
    assert _with_old_keys_added(saved, older, **old) == 1
    assert all(f'"{key}": ' in _manifest_text(older) for key in old)

    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(older))
    again._open_plots()
    _settle()
    reopened = again._plots_window
    if case.startswith("line"):       # an open recomputes no fit: the line waits for Plot
        reopened._run_plot()
        _settle()
    assert _label(reopened) == expected
    window._dirty = again._dirty = False


def test_an_older_template_stored_here_is_carried_into_a_project_as_it_was_and_the_page_is_clean(
        qapp, qtbot, tmp_path):
    """The guide says: a page this build saves after an edit holds none of the old settings, and
    an older Distributions template stored on this computer is carried into the project as it
    was stored (and recalled the way the template test above recalls it)."""
    from nanoscope.core.plot_preset_state import WORKSPACE_KEY as PRESETS_KEY

    older = plot_config_to_json(PlotConfig())
    older["style"].update(_old_style(fit_values_on_plot=True, show_r2=False, show_gof=False,
                                     title_bold=True, title_italic=True))
    PlotsWindow._store_templates({"Older template": older})
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plots.set_title("Dwell")                           # an edit: the page saves itself again
    _settle()
    saved = tmp_path / "carried.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    workspace = json.loads(_manifest_text(saved))["workspace"]
    page = next(a for a in workspace["artifacts"] if a["payload"].get("kind") == "distributions")
    carried = json.dumps(workspace[PRESETS_KEY])
    for key in OLD_KEYS:
        assert f'"{key}"' not in json.dumps(page), key             # the page: none of them
        assert f'"{key}": ' in carried, key                         # the template: as stored
    # Nothing but that template holds them: with it gone from this computer, no key is left.
    QtCore.QSettings().remove(PlotsWindow._TPL_SETTINGS_KEY)
    plots.set_title("Dwell, pore A")
    _settle()
    again = tmp_path / "carried-again.nanoscope"
    assert window._write_project(str(again))
    window._dirty = False
    text = _manifest_text(again)
    for key in OLD_KEYS:
        assert f'"{key}"' not in text, key


def test_an_installed_label_and_title_style_survive_the_next_appearance_edit(qtbot):
    """The Appearance widget is told the style an install put on the figure (a template, an
    import, a project). Before, its next edit was built on what it was last told, and wiped
    the label's numbers and the title's bold from the figure."""
    w = _distributions(qtbot)
    style = w.config().style.with_fit_label(FitLabelState(fields=("equation", "n", "bic", "ks")))
    style = style.with_text_style("title", replace(style.text_style("title"), bold=True))
    w.apply_config(replace(w.config(), style=style))
    _settle()
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"].replace(" · R² = ", " · ")
    box = w._appearance_widget.title
    box.setFocus()
    box.end(False)
    QTest.keyClicks(box, "Dwell")
    QTest.keyClick(box, QtCore.Qt.Key.Key_Return)
    qtbot.wait(800)
    assert w._plot.titleLabel.text.endswith("Dwell")
    assert w.config().style.fit_label_state().fields == ("equation", "n", "bic", "ks")
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"]
    assert _title_font(w) == (True, False)


# ==============================================================================================
# 3. Reset figure: in the page toolbar, words and look, one Undo step
# ==============================================================================================

PAGES = ("distributions", "iv", "vdep", "psd", "event", "summary", "comparison", "subpop")


def _page(qtbot, kind: str):
    if kind == "distributions":
        return _distributions(qtbot)
    page = _host(kind)
    qtbot.addWidget(page)
    return _show(page, qtbot, size=(1100, 820))


def _toolbar(kind: str, page):
    """The page toolbar: Distributions' own QToolBar, the shared bar elsewhere."""
    return page._workflow_toolbar if kind == "distributions" else page._workflow


def _controls(kind: str, page) -> list:
    """The toolbar's widgets, left to right."""
    bar = _toolbar(kind, page)
    if kind == "distributions":
        return [w for w in (bar.widgetForAction(a) for a in bar.actions()) if w is not None]
    layout = bar.layout()
    return [layout.itemAt(i).widget() for i in range(layout.count())
            if layout.itemAt(i).widget() is not None]


def _depth(kind: str, page) -> tuple[int, int]:
    return (page._history if kind == "distributions" else page._appearance_widget._history).depth()


def _plot_of(kind: str, page):
    return {"distributions": lambda: page._plot, "iv": lambda: page.plot_item(),
            "vdep": lambda: page.plot_item(), "psd": lambda: page._plot,
            "event": lambda: page._density_main, "summary": lambda: page._plot,
            "comparison": lambda: None,
            "subpop": lambda: page._embed_view._plot.getPlotItem()}[kind]()


def _fit_label_of(kind: str, page):
    return {"distributions": lambda: page._fit_label_item, "iv": lambda: page._fit_label,
            "vdep": lambda: page._fit_label}.get(kind, lambda: None)()


def _click_button(button) -> None:
    """A real mouse click on ``button``, through the window it is in."""
    handle = window_handle(button)
    QTest.mouseClick(handle, LEFT, NOMOD, button.mapTo(button.window(), button.rect().center()))
    QtWidgets.QApplication.processEvents()


def _words_and_look(kind: str, page, drawn=None) -> dict:
    """What is drawn: the title, the x label, the title's size, the fit label's text. Comparison
    figures is drawn by matplotlib: ``drawn`` is the record of what its renders drew
    (``spy_on_comparison_renders``) and its last entry is the figure on screen."""
    if kind == "comparison":
        last = drawn[-1]
        return {"title": last["title"], "x label": ", ".join(last["x"]),
                "title size": last["size"], "fit label": None}
    plot = _plot_of(kind, page)
    label = _fit_label_of(kind, page)
    return {
        "title": None if plot is None else plot.titleLabel.item.toPlainText(),
        "x label": None if plot is None else plot.getAxis("bottom").labelText,
        "title size": (plot.titleLabel.opts.get("size")
                       if plot is not None and plot.titleLabel.isVisible() else None),
        "fit label": None if label is None or not label.isVisible() else label.plain_text(),
    }


def _edit_words_and_look(kind: str, page) -> None:
    widget = page._appearance_widget
    state = replace(widget.state(), title="My title", x_label="Bias", title_size=18.0,
                    line_width=3.0)
    if _fit_label_of(kind, page) is not None:
        state = state.with_text_style("annotation.fit", TextStyle(text="Fit: {equation}"))
    widget.apply_appearance(state)
    QtWidgets.QApplication.processEvents()


#: Pages that keep a range typed in Axis properties with the look (Comparison figures has no
#: Axis properties; Distributions keeps its range in the Axis range boxes).
PINNING_PAGES = ("iv", "vdep", "psd", "event", "summary", "subpop")


@pytest.mark.parametrize("kind", PAGES)
def test_reset_figure_is_in_the_page_toolbar_and_puts_words_and_look_back_in_one_undo_step(
        qapp, qtbot, kind, monkeypatch):
    # Comparison figures is a matplotlib figure: what it DRAWS is read off its renders.
    drawn = spy_on_comparison_renders(monkeypatch) if kind == "comparison" else None
    page = _page(qtbot, kind)
    widget = page._appearance_widget
    button = widget.reset_button

    # Where it is: in the page toolbar, just before Export ▾, from every Section, and no
    # longer in the Appearance section.
    assert button.text() == "Reset figure" and button.accessibleName() == "Reset figure"
    assert button.isVisible() and _toolbar(kind, page).isAncestorOf(button)
    assert not widget.isAncestorOf(button)
    texts = [b.text() for b in widget.findChildren(QtWidgets.QPushButton)]
    assert "Reset figure" not in texts and "Reset appearance" not in texts, texts
    assert "Undo" in button.toolTip() and "defaults" in button.toolTip()
    assert button.isFlat() == (kind == "distributions"), "flat among the toolbar's text, only there"
    if kind != "distributions":
        for section in ("Data", "Analysis", "Appearance", "Export"):
            page._workflow.set_section(section)
            assert button.isVisible(), section
    controls = _controls(kind, page)
    export = page._export_menu_btn if kind == "distributions" else page._workflow.export_button
    assert controls.index(button) == controls.index(export) - 1 == len(controls) - 2

    # What it does: the words and the look go back, in one gesture, one Undo step.
    start = _words_and_look(kind, page, drawn)
    _edit_words_and_look(kind, page)
    edited = _words_and_look(kind, page, drawn)
    if _plot_of(kind, page) is not None or kind == "comparison":
        assert edited["title"] == "My title" and edited["x label"] == "Bias", edited
        assert edited["title size"] != start["title size"]
    if _fit_label_of(kind, page) is not None:
        assert edited["fit label"] is None or edited["fit label"].startswith("Fit: "), edited
    steps = _depth(kind, page)

    _click_button(button)

    assert _words_and_look(kind, page, drawn) == start
    assert _depth(kind, page) == (steps[0] + 1, 0), "one gesture, one Undo step"
    if kind == "distributions":
        style = page.config().style
        assert (style.title, style.x_label, style.title_font_size_pt) == ("", "", 0)
        assert not style.fit_label and not style.text_styles
        # Its Undo button counts steps; it does not name them.
        assert page._undo_btn.toolTip().startswith("Undo the last change to this plot's configuration")
        assert "reset figure" not in page._undo_btn.toolTip()
    else:
        assert not {k: v for k, v in page.appearance_state().items()
                    if k not in ("quantity_display", "source_appearance")}
        assert not page.appearance_state().get("source_appearance", {})
        assert widget.undo_button.toolTip().startswith("Undo:"), widget.undo_button.toolTip()
    _press_undo(page)       # from the Section where the click was made, on every page
    assert _words_and_look(kind, page, drawn) == edited
    assert _depth(kind, page) == (steps[0], 1)

    # A range typed in Axis properties goes with the look on the pages that keep it there.
    if kind in PINNING_PAGES:
        host = figure_text_for(_plot_of(kind, page)).host
        lo, hi = host.axis_state("bottom").shown_range
        host.apply_axis("bottom", AxisState(range=(lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo)),
                                            tick_major=(hi - lo) / 7.0))
        (pinned,) = page.appearance_state()["axes"].values()    # the page keeps it with the look
        assert {"min", "max", "tick_major"} <= set(pinned), pinned
        _click_button(button)
        assert "axes" not in page.appearance_state()


def test_reset_figure_in_distributions_also_returns_the_numbers_and_the_title_weight(qtbot):
    w = _distributions(qtbot)
    _tick(w, "bic")
    _tick(w, "ks")
    figure_text_for(w._plot).host.apply_text(
        "title", replace(w._style.text_style("title"), text="Dwell", bold=True, italic=True))
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"] and _title_font(w) == (True, True)
    steps = w._history.depth()
    _click_button(w._appearance_widget.reset_button)
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist"], "numbers back to equation, R², n"
    title = w.config().style.text_style("title")
    assert (title.bold, title.italic) == (None, None), "and the title's weight"
    assert w._history.depth() == (steps[0] + 1, 0)
    _press_undo(w)
    assert _label(w) == AS_THE_OLD_BUILD_DREW["hist_values"]
    assert w._history.depth() == (steps[0], 1)
    w.set_title("Dwell")
    assert _title_font(w) == (True, True)


#: What Distributions' Reset figure leaves as it is, in the words its tool tip uses, and where
#: each lives: Style fields, or the config section ``axes``.
KEPT_BY_DISTRIBUTIONS_RESET = {
    "Axis range boxes": ("axes",),
    "Font size (body)": ("font_size_pt",),
    "the legend's position, entries and (n = …) switch": (
        "legend_pos", "legend_include", "legend_counts"),
    "source names": ("label_overrides",),
    "Confidence band": ("show_ci_band",),
    "Residual strip": ("show_residuals",),
    "Provenance caption": ("show_provenance",),
    "Heatmap colormap": ("heatmap_colormap",),
}


def test_reset_figure_in_distributions_leaves_what_is_not_the_words_or_the_look(
        qtbot, monkeypatch):
    w = _distributions(qtbot)
    config = w.config()
    style = replace(
        config.style, font_size_pt=13, legend_counts=False, legend_pos=(0.3, 0.2),
        legend_include=pp.FrozenMap({"series:probe": False}), show_ci_band=True,
        show_residuals=True, show_provenance=True, heatmap_colormap="magma",
        label_overrides=pp.FrozenMap({"probe": "Apo"}), title="Dwell", x_label="Time",
        title_font_size_pt=18)
    w.apply_config(replace(config, style=style))
    w.set_axis_range(xmin=0.5, xmax=8.0)
    kept = w.config()
    assert kept.axes.manual_x is not None
    for name in ("_redraw", "_drawn_frames", "_draw_histogram", "_overlay_fit"):
        monkeypatch.setattr(w, name, lambda *a, _name=name, **k: pytest.fail(
            f"Reset figure called {_name}"))
    _click_button(w._appearance_widget.reset_button)
    monkeypatch.undo()                       # closing the window redraws it
    after = w.config()
    assert (after.style.title, after.style.x_label, after.style.title_font_size_pt) == ("", "", 0)
    for field in ("font_size_pt", "legend_counts", "legend_pos", "legend_include",
                  "show_ci_band", "show_residuals", "show_provenance", "heatmap_colormap",
                  "label_overrides"):
        assert getattr(after.style, field) == getattr(kept.style, field), field
    assert after.axes == kept.axes and after.fit == kept.fit
    assert after.histogram == kept.histogram and after.selection == kept.selection

    # The button's tip: the shared words about "its look", then which rows of THIS window are
    # not in that look. Every row it names was set above and is as it was.
    tip = w._appearance_widget.reset_button.toolTip()
    other = PlotAppearanceWidget()
    qtbot.addWidget(other)
    shared = other.reset_button.toolTip()
    assert tip.startswith(shared) and len(tip) > len(shared), "the shared tip, then what it keeps"
    for phrase, places in KEPT_BY_DISTRIBUTIONS_RESET.items():
        assert phrase in tip, phrase
        for place in places:
            here, then = (after, kept) if place == "axes" else (after.style, kept.style)
            assert getattr(here, place) == getattr(then, place), (phrase, place)


def test_a_summaries_legend_name_survives_reset_figure_in_the_toolbar(qtbot):
    page = _page(qtbot, "summary")
    _edit_double_click(page._plot, _legend_row(page), "Apo")
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"
    steps = _depth("summary", page)
    _click_button(page._appearance_widget.reset_button)
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"
    assert _depth("summary", page) == steps, "a name kept in Source names is no step"


def test_in_the_explorer_the_toolbars_reset_figure_resets_the_page_in_front(
        qapp, qtbot, tmp_path, monkeypatch):
    """Phase 1: Reset figure is the Explorer toolbar's, just before Export ▾, one button for
    whichever page is in front. The page's own button is the object that does the work: it
    stays (hidden, the toolbar clicks it) so ``reset_button.click()`` keeps working."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    window._open_iv()
    iv = window._iv_window
    window._open_psd()
    noise = window._psd_window
    window._open_plots()
    plots = window._plots_window
    explorer = window._analysis_explorer
    bar = explorer.toolbar()
    for page in (iv, noise, plots):
        assert explorer.isAncestorOf(page._appearance_widget.reset_button)
        assert not page._appearance_widget.reset_button.isVisible()
    _click_tab(explorer, iv)
    button = bar.reset_figure_button
    action = bar._reset_from_menu
    assert action.isVisible() and action.isEnabled() and action.text() == "Reset figure"
    assert not button.isVisible()
    assert "Undo" in action.toolTip() and "defaults" in action.toolTip()
    assert button.toolTip() == iv._appearance_widget.reset_button.toolTip()
    # A figure made in a project is bound to the project's figure style; Reset figure returns it
    # to that look, as a new figure has it, with the words gone (and not to a figure with no style).
    new_look = iv.appearance_state()
    assert new_look["style_binding"] and "title" not in new_look
    iv._appearance_widget.apply_appearance(
        replace(iv._appearance_widget.state(), title="Pore 3", x_label="Bias"))
    steps = _depth("iv", iv)
    assert iv._plot.titleLabel.item.toPlainText() == "Pore 3"
    bar._update_menu.popup(bar.mapToGlobal(QtCore.QPoint(20, 20)))
    _choose(bar._update_menu, "Reset figure")
    assert iv.appearance_state() == new_look and iv._plot.titleLabel.item.toPlainText() == ""
    assert _depth("iv", iv) == (steps[0] + 1, 0)
    _press_undo(iv)
    assert iv._plot.titleLabel.item.toPlainText() == "Pore 3"
    _click_tab(explorer, plots)
    assert button.toolTip() == plots._appearance_widget.reset_button.toolTip()
    assert button.toolTip() != iv._appearance_widget.reset_button.toolTip(), \
        "Distributions says which rows it keeps"
    window._dirty = False


@pytest.mark.parametrize("history", [True, False], ids=["with Undo and Redo", "without a history"])
def test_taking_the_reset_button_hands_over_the_same_button_and_leaves_undo_and_redo_together(
        qtbot, history):
    widget = PlotAppearanceWidget(history=history)
    qtbot.addWidget(widget)
    widget.resize(420, 500)
    widget.show()
    qtbot.waitExposed(widget)
    button = widget.reset_button
    holder = button.parentWidget().layout()     # the row it shares with Undo and Redo, or the grid
    assert widget.isAncestorOf(button) and holder.indexOf(button) >= 0
    taken = widget.take_reset_button()
    QtWidgets.QApplication.processEvents()
    assert taken is button and not widget.isAncestorOf(button)
    assert holder.indexOf(button) == -1, "its layout let go of it: no ghost cell"
    row = widget._reset_row
    held = None if row is None else row.count()
    assert widget.take_reset_button() is button, "taking it again changes nothing"
    assert (None if row is None else row.count()) == held, "and adds nothing to the row"
    if history:
        gap = widget.redo_button.x() - widget.undo_button.geometry().right()
        assert 0 < gap < 24, f"Undo and Redo sit side by side, not spread over the cell: {gap}px"
    widget.apply_appearance(replace(widget.state(), title="Study"))
    button.click()                                      # still the widget's own Reset
    assert widget.state() == PlotAppearance()


def test_a_window_without_a_toolbar_keeps_the_button_in_its_appearance_section(qtbot):
    editor = _study_window()
    qtbot.addWidget(editor)
    editor.show()
    widget = editor._appearance_widget
    button = widget.reset_button
    assert button.text() == "Reset figure" and widget.isAncestorOf(button)
    assert not editor.findChildren(QtWidgets.QToolBar) or all(
        not bar.isAncestorOf(button) for bar in editor.findChildren(QtWidgets.QToolBar))
    # The editor draws the figure when Apply appearance is pressed (an edit is a draft until
    # then), so what is DRAWN is read off the matplotlib figure after that real button.
    apply_button = next(b for b in editor.findChildren(QtWidgets.QPushButton)
                        if b.text() == "Apply appearance")
    figure = editor._figure
    assert figure._suptitle is None and figure.axes[0].get_title() == "Accepted events"
    own_size = figure.axes[0].title.get_fontsize()
    widget.apply_appearance(replace(widget.state(), title="Study", title_size=16.0))
    QTest.mouseClick(apply_button, LEFT)
    assert figure.get_suptitle() == "Study" and figure._suptitle.get_fontsize() == 16.0
    assert figure.axes[0].title.get_fontsize() == 16.0
    steps = editor.history_mark()
    QTest.mouseClick(button, LEFT)
    assert widget.state() == PlotAppearance()
    assert editor.history_mark() == steps + 1
    QTest.mouseClick(apply_button, LEFT)
    # ...and once drawn again, Reset figure shows the study's own figure: no title over it, the
    # panel's title at the size it had.
    assert figure.get_suptitle() == "" and figure.axes[0].get_title() == "Accepted events"
    assert figure.axes[0].title.get_fontsize() == own_size
    assert widget.undo() and widget.state().title == "Study"
    QTest.mouseClick(apply_button, LEFT)
    assert figure.get_suptitle() == "Study" and figure._suptitle.get_fontsize() == 16.0
    editor.close()


def test_the_control_inventory_lists_the_new_controls_and_not_the_removed_ones():
    root = Path(nanoscope.__file__).resolve().parents[1]
    inventory = next(path for path in (root / "docs" / "inventory.json", root / "inventory.json")
                     if path.exists())
    controls = [control for area in json.loads(inventory.read_text(encoding="utf-8"))
                for control in area["controls"]]
    labels = [control["label"] for control in controls]
    for gone in ("Fit values on plot", "R² on figure", "KS statistic on figure",
                 "Title size / Title weight", "Reset appearance"):
        assert gone not in labels, gone
    assert {"Reset figure", "Fit label · Numbers", "Title size"} <= set(labels)
    notes = {control["label"]: control.get("notes", "") for control in controls}
    # What the entries say Reset figure leaves in Distributions is what its tip says.
    for phrase in KEPT_BY_DISTRIBUTIONS_RESET:
        assert phrase in notes["Reset figure"], phrase
    assert "one place" not in notes["Fit label · Numbers"]
    assert "Residual strip" in notes["Fit label · Numbers"]


def _guide_bullet(start: str) -> str:
    """The USER_GUIDE bullet that begins with ``start``, on one line."""
    root = Path(nanoscope.__file__).resolve().parents[1]
    text = (root / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    rest = text[text.index(start):]
    end = rest.find("\n- **", len(start))
    return " ".join((rest if end < 0 else rest[:end]).split())


def test_the_guide_says_what_the_build_does_with_the_numbers_and_the_old_settings():
    """The sentences a reviewer checked against the code, kept in step with it: the Numbers
    row is not the only thing that writes into the label (Residual strip), a refusal sits at
    the end of the label, wording naming {r2}/{ks} draws a dash, and a stored older template is
    carried into a project as it was (the tests above drive each one)."""
    appearance = _guide_bullet("- **Format** — choose Title or an axis Label")
    # Phase 1 moves Numbers to Format and Residual strip to Fit display; the same numerical
    # behavior is still documented, independently of the element style controls.
    assert "except that **Residual strip** in **Data & analysis ▸ Fit display** also adds the strip's χ²/dof after the equation" in appearance
    assert "not in the Style group" not in appearance
    assert "only switch" not in appearance and "no other switch" not in appearance
    label = _guide_bullet("- **Fit label** — when a fit is applied")
    assert "at the end of the label or where `{r2}` stands" in label
    assert "where your own wording names it a dash is drawn" in label
    assert "the one place" not in label
    earlier = _guide_bullet("- **Files from earlier builds**")
    for stated in ("`{r2}` or `{ks}` for a switch that was unticked, a dash is drawn there",
                   "shows no R² refusal tag on the label",
                   "summary line of a scatter fit always carries R²",
                   "an older Distributions template stored on this computer, are carried into "
                   "the project as they were",
                   "New project saves use version 13"):
        assert stated in earlier, stated


def test_no_surface_still_calls_it_reset_appearance():
    root = Path(nanoscope.__file__).resolve().parents[1]
    sources = [*sorted((root / "nanoscope").rglob("*.py")), root / "docs" / "USER_GUIDE.md",
               root / "docs" / "inventory.json"]
    found = [path.relative_to(root).as_posix() for path in sources
             if "Reset appearance" in path.read_text(encoding="utf-8")]
    assert found == []
