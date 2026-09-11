"""The two invariants this batch found implemented once per window instead of once.

Both were "fixed" in the first pass and both came back, in the file the fix had not
reached. That is the project's signature failure and these are the guards against it.

**Low-21 — a setter that cannot honour its argument must not return quietly.**
``QComboBox.setCurrentText`` on a non-editable combo does nothing at all for unmatched
text: no exception, no signal, no change. The first fix wrote a raising helper inside
``plots_panel.py`` and wired it into 2 of that window's 7 combo setters and 0 sibling
windows. The verification pass of this batch then hit the identical defect in
``VDepWindow.set_metric`` — capture-rate numbers reported under a "median depth"
heading — and caught it only because it had added a read-back assertion by hand.

**Medium-13 / High-7 — a log axis whose labels can be read.**
Fixed twice, differently, in two files. ``psd_panel`` dropped only labels that would
COLLIDE; ``plots_panel`` blanked every non-decade label. Measured on the −600 mV merged
dwells through a real window, blanking painted each of the five decade labels TWICE at
identical coordinates and drew ZERO labels when zoomed to 15–25 ms — inside one decade,
astride the median dwell of all three recordings.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import axes as axes_mod
from nanoscope.gui import plots_panel as plots_mod
from nanoscope.gui import psd_panel as psd_mod
from nanoscope.gui import voltage_dep_panel as vdep_mod
from nanoscope.gui.combos import choose_combo_item


def test_the_tree_under_test_is_the_real_one():
    """pytest's rootdir insertion has twice swapped the tree under a revert run on this
    project. Assert where the package came from before believing anything else here."""
    here = Path(nanoscope.__file__).resolve()
    assert here.parent.parent == Path("/Users/abrahamcampos/Desktop/nanoscope").resolve(), here


# ================================================================================================
# Low-21 — the raising combo setter, in every window
# ================================================================================================
def _combo(items):
    c = QtWidgets.QComboBox()
    c.addItems(items)
    return c


def test_an_unmatched_name_raises_and_names_the_options(qapp):
    c = _combo(["exponential", "gamma (peaked)", "gaussian-mixture"])
    with pytest.raises(ValueError) as exc:
        choose_combo_item(c, "not-a-model", "fit")
    assert "gamma (peaked)" in str(exc.value)
    assert c.currentText() == "exponential", "the selection moved on a failed set"


def test_an_ambiguous_prefix_raises_rather_than_picking_one(qapp):
    c = _combo(["gamma (peaked)", "gaussian-mixture"])
    with pytest.raises(ValueError, match="ambiguous"):
        choose_combo_item(c, "g", "fit")


def test_a_unique_prefix_resolves(qapp):
    c = _combo(["exponential", "gamma (peaked)"])
    assert choose_combo_item(c, "gamma", "fit") == "gamma (peaked)"


#: Every setter in the three overlay windows that lands a string in a QComboBox. The
#: list is the point: two of these were fixed and five were not, in the window the fix
#: was written in, and three more live in windows it never reached.
_SETTERS = [
    ("plots", "set_plot_type"), ("plots", "set_time_unit"),
    ("plots", "set_histogram_column"), ("plots", "set_colour_by"),
    ("plots", "set_normalization"), ("plots", "set_fit"), ("plots", "set_scatter_fit"),
    ("vdep", "set_metric"), ("vdep", "set_fit"),
    ("psd", "set_window"),
]


@pytest.mark.parametrize("window,setter", _SETTERS,
                         ids=[f"{w}.{s}" for w, s in _SETTERS])
def test_every_combo_setter_refuses_a_name_it_cannot_honour(qapp, qtbot, window, setter):
    """Measured before: ``set_histogram_column("frac")`` left ``depth_pA`` selected and
    the window reported picoamps; ``set_time_unit("milliseconds")`` left the axis in
    seconds; ``VDepWindow.set_metric("median depth (pA)")`` reported capture rates."""
    session = Session()
    win = {"plots": plots_mod.PlotsWindow,
           "vdep": vdep_mod.VDepWindow,
           "psd": psd_mod.PSDWindow}[window](session)
    qtbot.addWidget(win)
    with pytest.raises(ValueError):
        getattr(win, setter)("definitely-not-an-option-¶")


def test_no_setter_in_these_windows_reaches_setCurrentText_directly(qapp):
    """The structural half: the rule holds by construction, not by having been applied.

    A ``self._x.setCurrentText(...)`` inside a ``set_*`` method is the defect. The
    helper's own read-back call is the one legitimate use and it lives in
    ``gui/combos.py``, which is not scanned here.
    """
    offenders: list[str] = []
    for label, mod in (("plots_panel", plots_mod), ("psd_panel", psd_mod),
                       ("voltage_dep_panel", vdep_mod)):
        tree = ast.parse(Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("set_"):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "setCurrentText"):
                    offenders.append(f"{label}.{node.name}")
    assert offenders == [], offenders


# ================================================================================================
# Medium-13 / High-7 — one readable log axis, subclassed by both windows
# ================================================================================================
def test_both_windows_axes_descend_from_the_one_implementation():
    assert issubclass(plots_mod._DecadeLabelAxis, axes_mod.ReadableLogAxis)
    assert issubclass(psd_mod._ReadableAxis, axes_mod.ReadableLogAxis)


def _drawn(ax) -> list[str]:
    img = QtGui.QImage(1400, 900, QtGui.QImage.Format_ARGB32)
    p = QtGui.QPainter(img)
    try:
        specs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    return [] if specs is None else [t for _r, _f, t in specs[2] if str(t).strip()]


@pytest.mark.parametrize("factory", [
    pytest.param(plots_mod._DecadeLabelAxis, id="plots"),
    pytest.param(psd_mod._ReadableAxis, id="psd"),
])
def test_neither_axis_goes_blank_inside_a_single_decade(qapp, factory):
    """THE DEFECT BLANKING INTRODUCED. 15-25 ms contains no decade."""
    ax = factory(orientation="bottom")
    values = list(np.log10([0.015, 0.018, 0.020, 0.022, 0.025]))
    labels = [s for s in ax.logTickStrings(values, 1.0, None) if s]
    assert labels, "an axis zoomed inside one decade offered no labels at all"


@pytest.mark.parametrize("factory", [
    pytest.param(plots_mod._DecadeLabelAxis, id="plots"),
    pytest.param(psd_mod._ReadableAxis, id="psd"),
])
def test_neither_axis_mixes_two_notations(qapp, factory):
    """``0.01, 0.1, 1, 10¹, 10²`` — one and the same axis, two kinds of number."""
    ax = factory(orientation="bottom")
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    labels = [s for s in ax.logTickStrings(values, 1.0, None) if s]
    supers = [s for s in labels if any(c in s for c in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")]
    assert len(supers) in (0, len(labels)), labels


@pytest.mark.parametrize("factory", [
    pytest.param(plots_mod._DecadeLabelAxis, id="plots"),
    pytest.param(psd_mod._ReadableAxis, id="psd"),
])
def test_neither_axis_carries_an_invented_multiplier(qapp, factory):
    ax = factory(orientation="bottom")
    assert ax.autoSIPrefix is False
    ax.setLabel("dwell (ms, log)")
    assert "(x" not in ax.labelString(), ax.labelString()


@pytest.mark.parametrize("factory", [axes_mod.ReadableLinearAxis, plots_mod._DecadeLabelAxis,
                                    psd_mod._ReadableAxis])
def test_disabling_si_prefix_clears_scale_on_an_already_labelled_axis(qapp, factory):
    ax = factory(orientation="bottom")
    ax.setLabel("Recorded measurement")
    ax.enableAutoSIPrefix(True)
    ax.setRange(0, .01)
    assert ax.autoSIPrefixScale != 1
    ax.enableAutoSIPrefix(False)
    assert ax.autoSIPrefixScale == 1
    assert ax.labelUnitPrefix == ""
    assert "(x" not in ax.labelString()


def test_a_decade_is_never_dropped_in_favour_of_a_minor_label(qapp):
    """Priority is by label FORM, not emission order.

    Over ~3.6 decades pyqtgraph emits one flat level of all 31 ticks, so left-to-right
    thinning keeps ``9·10¹`` and drops ``10²``, ``10³``, ``10⁴``.
    """
    rank = axes_mod.ReadableLogAxis.label_rank
    assert rank("10²") == 0 and rank("10⁻⁵") == 0
    assert rank("9·10¹") == 1 and rank("2.5·10⁻³") == 1
    # ...and in the decimal notation the other window uses.
    assert rank("0.01") == 0 and rank("1") == 0 and rank("100") == 0
    assert rank("0.02") == 1 and rank("20") == 1


def test_two_labels_at_the_same_place_do_not_both_survive(qapp):
    """The duplicate-painting half: 10 text specs at 5 positions became 5."""
    from PySide6.QtCore import QRectF

    ax = plots_mod._DecadeLabelAxis(orientation="bottom")
    rect = QRectF(100.0, 0.0, 30.0, 12.0)
    specs = [(QRectF(rect), 0, "1"), (QRectF(rect), 0, "1")]
    assert len(ax.without_collisions(specs)) == 1
