"""Phase 1: AST guards over the new shell modules.

``analysis_inspector.py``, ``analysis_toolbar.py``, ``plot_styles.py``, ``format_inspector.py``,
``format_toolbar.py`` and ``widget_transfer.py`` may not install or define an event filter,
use ``WA_DeleteOnClose``,
connect ``itemChanged``, override ``keyPressEvent``, register a shortcut (the one exception is
Update plot and Figures in the toolbar, and the CLEARING of another page's with an empty sequence),
scroll anything outside ``InspectorHost.reveal``, walk a layout with ``itemAt``, call
``windowHandle()``, or import a page module. Each rule is also run against a source that breaks
it, so the guard cannot go blind.
"""
from __future__ import annotations

import ast
import os
import pathlib

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

GUI = pathlib.Path(__import__("nanoscope.gui", fromlist=["x"]).__file__).parent
MODULES = ("analysis_inspector", "analysis_toolbar", "plot_styles", "format_inspector",
           "format_toolbar", "widget_transfer")
PAGE_MODULES = {"iv_panel", "voltage_dep_panel", "distribution_summary_panel",
                "comparison_figures_panel", "event_plot_panel", "psd_panel", "subpop_window",
                "plots_panel", "main_window", "analysis_explorer", "analysis_home"}
SCROLLING = {"ensureWidgetVisible", "scrollContentsBy"}
#: (module, shortcut calls allowed there)
# Update plot (Return/Enter) and the explicit Figures-list toggle (Ctrl/Command
# Shift L) are window actions. The extra toggle does not install a global filter.
SHORTCUT_ALLOWED = {"analysis_toolbar": {"setShortcuts": 1, "setShortcut": 1,
                                         "setShortcutContext": 2}}


def _is_empty_sequence(node: ast.AST) -> bool:
    """``QKeySequence()`` or ``[]``: a shortcut being taken away, not given."""
    if isinstance(node, ast.List):
        return not node.elts
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        return name == "QKeySequence"
    return False


def scan(module: str, source: str) -> list[str]:
    """Every rule broken in ``source`` (a module named ``module``), as sentences."""
    tree = ast.parse(source)
    found: list[str] = []
    shortcuts: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        if named:
            stack.append(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                "eventFilter", "keyPressEvent"):
            found.append(f"defines {node.name}")
        if isinstance(node, ast.ImportFrom):
            name = (node.module or "").split(".")[-1]
            names = {a.name for a in node.names}
            if name in PAGE_MODULES or (node.level and not name and names & PAGE_MODULES):
                found.append(f"imports a page module: {name or names}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in PAGE_MODULES:
                    found.append(f"imports a page module: {alias.name}")
        if isinstance(node, ast.Attribute):
            if node.attr in ("installEventFilter", "removeEventFilter"):
                found.append(f"uses {node.attr}")
            if node.attr == "WA_DeleteOnClose":
                found.append("uses WA_DeleteOnClose")
            if node.attr == "itemChanged":
                found.append("uses itemChanged")
            if node.attr == "QShortcut":
                found.append("builds a QShortcut")
        if isinstance(node, ast.Name) and node.id == "QShortcut":
            found.append("builds a QShortcut")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "itemAt":
                found.append("walks a layout with itemAt")
            if attr == "windowHandle":
                found.append("calls windowHandle()")
            if attr in ("setShortcut", "setShortcuts", "setShortcutContext"):
                empty = node.args and _is_empty_sequence(node.args[0])
                if not empty:
                    shortcuts[attr] = shortcuts.get(attr, 0) + 1
            inside_reveal = stack[-2:] == ["InspectorHost", "reveal"]
            scrolls = attr in SCROLLING or (
                attr == "setValue" and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr in ("verticalScrollBar", "horizontalScrollBar"))
            if scrolls and not inside_reveal:
                found.append(f"scrolls outside InspectorHost.reveal ({attr})")
        for child in ast.iter_child_nodes(node):
            visit(child)
        if named:
            stack.pop()

    visit(tree)
    allowed = SHORTCUT_ALLOWED.get(module, {})
    for attr, count in shortcuts.items():
        if count > allowed.get(attr, 0):
            found.append(f"registers a shortcut ({attr} x{count})")
    return found


@pytest.mark.parametrize("module", MODULES)
def test_the_shell_modules_break_no_rule(module):
    source = (GUI / f"{module}.py").read_text()
    assert scan(module, source) == []


@pytest.mark.parametrize("snippet, expected", [
    ("class A:\n  def eventFilter(self, w, e):\n    return False\n", "defines eventFilter"),
    ("v.installEventFilter(self)\n", "uses installEventFilter"),
    ("w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)\n", "uses WA_DeleteOnClose"),
    ("tree.itemChanged.connect(f)\n", "uses itemChanged"),
    ("class A:\n  def keyPressEvent(self, e):\n    pass\n", "defines keyPressEvent"),
    ("s = QtGui.QShortcut(k, w)\n", "builds a QShortcut"),
    ("a.setShortcut(k)\n", "registers a shortcut (setShortcut x1)"),
    ("a.setShortcuts([k])\n", "registers a shortcut (setShortcuts x1)"),
    ("area.ensureWidgetVisible(w)\n", "scrolls outside InspectorHost.reveal (ensureWidgetVisible)"),
    ("area.verticalScrollBar().setValue(0)\n", "scrolls outside InspectorHost.reveal (setValue)"),
    ("class InspectorHost:\n  def other(self):\n    self.a.ensureWidgetVisible(w)\n",
     "scrolls outside InspectorHost.reveal (ensureWidgetVisible)"),
    ("item = layout.itemAt(0)\n", "walks a layout with itemAt"),
    ("h = w.windowHandle()\n", "calls windowHandle()"),
    ("from .iv_panel import IVWindow\n", "imports a page module: iv_panel"),
    ("from . import plots_panel\n", "imports a page module: {'plots_panel'}"),
    ("import nanoscope.gui.psd_panel\n", "imports a page module: nanoscope.gui.psd_panel"),
])
def test_each_rule_catches_its_violation(snippet, expected):
    assert expected in scan("analysis_inspector", snippet)


def test_the_allowed_forms_pass():
    ok = ("class InspectorHost:\n  def reveal(self):\n    a.ensureWidgetVisible(f)\n"
          "    a.verticalScrollBar().setValue(1)\n"
          "x.setShortcut(QtGui.QKeySequence())\nx.setShortcuts([])\n")
    assert scan("analysis_inspector", ok) == []
    toolbar = "act.setShortcuts([a, b])\nact.setShortcutContext(c)\n"
    assert scan("analysis_toolbar", toolbar) == []
    assert scan("analysis_toolbar", toolbar + "other.setShortcuts([a])\n") != []
    figures = "figures.setShortcut(k)\nfigures.setShortcutContext(c)\n"
    assert scan("analysis_toolbar", toolbar + figures) == []
    assert scan("analysis_toolbar", toolbar + figures + "third.setShortcut(k)\n") != []


# -- the files the Foundation EDITS (not only the five new modules) ---------------------------------
#: ``{module: {rule: how many the file had at ba5a214}}``. The Foundation may not add one; a
#: later group that must adds it here WITH a sentence saying why. Counted over the AST, so a
#: ``getattr(page, "installEventFilter")``, a string constant naming the call and an
#: ``eventFilter`` assigned as a class attribute are caught too.
EDITED_BASELINE = {
    "iv_panel": {},
    # BoundedDetails watches only its own summary's Resize/FontChange events so
    # it can fit whole text lines while retaining every word in Full details.
    "plot_workflow": {"itemAt": 2, "eventFilter": 2, "installEventFilter": 1},
    "plot_appearance": {},
    # One exact itemAt(indexOf(page)) retires the removed tab's Python wrapper
    # before native removeTab deletes its item; it does not walk the layout.
    "analysis_explorer": {"installEventFilter": 1, "eventFilter": 2, "itemAt": 1},
    "plot_templates": {}, "source_appearance": {},
    # The final native page adapters add none. Baselines checked against the recovered
    # Foundation 94b98f6: Subpopulations' three pre-existing filters remain unchanged.
    "voltage_dep_panel": {}, "distribution_summary_panel": {},
    "comparison_figures_panel": {}, "event_plot_panel": {}, "psd_panel": {},
    "subpop_window": {"eventFilter": 6, "installEventFilter": 3},
    "plots_panel": {}, "widget_transfer": {},
}
_NAMES = {"installEventFilter", "eventFilter", "WA_DeleteOnClose", "windowHandle", "itemAt"}


def count_rules(source: str) -> dict[str, int]:
    """How many times ``source`` names each of the rules' calls, however it spells them."""
    counts: dict[str, int] = {}

    def bump(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in _NAMES:
            bump(node.attr)
        elif isinstance(node, ast.Name) and node.id in _NAMES:
            bump(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _NAMES:
            bump(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _NAMES:
            bump(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _NAMES:
                    bump(target.id)
    return counts


@pytest.mark.parametrize("module", sorted(EDITED_BASELINE))
def test_the_edited_modules_add_no_event_filter_delete_on_close_itemat_or_window_handle(module):
    now = count_rules((GUI / f"{module}.py").read_text())
    base = EDITED_BASELINE[module]
    over = {k: (v, base.get(k, 0)) for k, v in now.items() if v > base.get(k, 0)}
    assert over == {}, f"{module}: more than at ba5a214 (now, base): {over}"


@pytest.mark.parametrize("snippet, name", [
    ("self._plot.viewport().installEventFilter(self)\n", "installEventFilter"),
    ("getattr(page, 'installEventFilter')(self)\n", "installEventFilter"),
    ("class A:\n  eventFilter = other\n", "eventFilter"),
    ("class A:\n  def eventFilter(self, w, e):\n    return False\n", "eventFilter"),
    ("w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)\n", "WA_DeleteOnClose"),
    ("h = w.windowHandle()\n", "windowHandle"),
])
def test_the_edited_module_counter_sees_each_spelling(snippet, name):
    assert count_rules(snippet).get(name, 0) >= 1
