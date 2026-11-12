"""The rules ``plots_panel.py`` is split under (``plots_constants.py`` and the ``plots_<part>_mixin.py`` mixins).

``PlotsWindow`` keeps its constructor, its signals and its one ``@Slot``, its Qt overrides, the saved-figure types and
their JSON codec, ``_ConfigHistory`` and the five draw methods in ``plots_panel.py``. Other methods were cut out of it
and pasted, unchanged, into plain-Python mixin classes listed before ``QMainWindow``, and its constants and pure
helpers into the ``plots_constants`` leaf, which ``plots_panel`` imports back name by name. These tests hold what that
move depends on (``test_main_window_parts.py`` holds the same for ``MainWindow``, and the rule functions the two
share are defined there and tried on violating input there):

* no part declares a ``@Slot``: from a mixin it registers on the window's meta-object all the same, but
  ``signal.connect(self.<slot>)`` then goes through PySide's global receiver, and Qt does not remove that connection
  when the window's C++ object is deleted (measured on ``MainWindow``). The Plots window's one slot,
  ``_on_notes_ready``, is connected to a worker's ``done`` signal. Two rules hold it: one reads the decorators and
  names the file, and one reads the meta-object, which does not care how ``Slot`` is spelled;
* a part is one plain ``*Mixin`` class: no base, no ``__init__``, no ``super()``, no signal, and no member whose name
  ``QMainWindow`` has (a Qt override, or a Qt method it would shadow);
* a part may import ``plots_constants`` and never ``plots_panel`` or another part, in any spelling and at any depth;
  the leaf imports none of them;
* a name the tests patch on the ``plots_panel`` module is read only in ``plots_panel.py``: a part that bound it would
  read its own copy, and the patch would silently stop reaching it;
* every name the leaf defines is the SAME object on ``plots_panel`` (one state, not two);
* every part file is a base of ``PlotsWindow`` ahead of ``QMainWindow``, and no name is defined in two of the
  window's classes (which copy runs would depend on the order of the bases);
* the parts ``test_measure_navigation.py`` names by hand (its allowlist and its per-part reach) are the files that
  exist. The other lists of parts in the tests are read from the directory.
"""
from __future__ import annotations

import ast
import os
import pathlib

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from .test_main_window_parts import (
    base_problems,
    leaf_definitions,
    leaf_name_problems,
    mixin_names_qmainwindow_has,
    names_defined_twice,
    signals_held_by_mixins,
    slot_owners_outside_the_class,
    slots_declared,
)

GUI = pathlib.Path(__import__("nanoscope.gui", fromlist=["x"]).__file__).parent
TESTS = pathlib.Path(__file__).parent
LEAF = "plots_constants"
PARTS = sorted(p.stem for p in GUI.glob("plots_*_mixin.py"))
#: Every name ``QMainWindow`` has: a mixin that defined one would shadow it or override it.
QT_NAMES = frozenset(n for n in dir(QtWidgets.QMainWindow) if not (n.startswith("__") and n.endswith("__")))
#: The names tests rebind on the ``plots_panel`` module (``monkeypatch.setattr(plots_panel, "<name>", ...)``, also
#: through ``importlib.import_module(type(panel).__module__)``). A test that starts to patch another name on
#: ``plots_panel`` adds it here: the rule below then keeps every part from reading it.
PATCHED_ON_PLOTS_PANEL = {"confirm_large_svg", "ask_export_path", "choose_live_plot_export", "ImageExporter",
                          "kde_curve", "PLOT_CONFIG_VERSION", "_LEGACY_NORMS", "write_live_plot_image"}


def _tree(stem: str) -> ast.Module:
    return ast.parse((GUI / f"{stem}.py").read_text(encoding="utf-8"))


def window_files_imported(tree: ast.AST) -> set[str]:
    """The window's own files (``plots_panel`` and every part) ``tree`` imports, by any spelling and at any depth:
    ``from .x import y``, ``from . import x``, ``from nanoscope.gui import x``, ``import nanoscope.gui.x``."""
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            named |= {alias.name.split(".")[-1] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            named |= {(node.module or "").split(".")[-1], *(alias.name for alias in node.names)}
    return named & {"plots_panel", *PARTS}


def _class_members(cls: ast.ClassDef):
    """``(name, value)`` for every name a class body binds: ``def``, nested ``class``, assignment, annotation."""
    for member in cls.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield member.name, None
        elif isinstance(member, ast.Assign):
            for target in member.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        yield node.id, member.value
        elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            yield member.target.id, member.value


def part_problems(tree: ast.Module) -> list[str]:
    """Every rule a part module breaks, as sentences."""
    found = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1 or not classes[0].name.endswith("Mixin"):
        found.append("does not define exactly one *Mixin class")
    for cls in classes:
        if cls.bases or cls.keywords or cls.decorator_list:
            found.append(f"{cls.name} has a base class, a metaclass or a decorator")
        for name, value in _class_members(cls):
            if name == "__init__":
                found.append(f"{cls.name} defines __init__")
            elif name in QT_NAMES:
                found.append(f"{cls.name} defines {name}, a name QMainWindow has")
            if isinstance(value, ast.Call) and ast.unparse(value.func).split(".")[-1] == "Signal":
                found.append(f"{cls.name} declares the signal {name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "super":
            found.append("calls super()")
        if isinstance(node, ast.Name) and node.id == "__class__":
            found.append("names __class__")
    reached = window_files_imported(tree)
    if reached:
        found.append(f"imports {sorted(reached)}")
    return found


def patched_names_used(tree: ast.AST) -> set[str]:
    """The patched names ``tree`` binds by import or reads (or assigns) as a bare name."""
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            used |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return used & PATCHED_ON_PLOTS_PANEL


def parts_named_by_hand(tree: ast.AST) -> set[str]:
    """The strings in ``tree`` that name the leaf or a part: ``"plots_fit_mixin"``, ``"nanoscope.gui.plots_fit_mixin"``."""
    named = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            last = node.value.rsplit(".", 1)[-1]
            if last == LEAF or (last.startswith("plots_") and last.endswith("_mixin")):
                named.add(last)
    return named


# -- the parts are found ----------------------------------------------------------------------------------
def test_the_parts_are_found():
    assert PARTS and (GUI / f"{LEAF}.py").is_file(), "no plots_*_mixin.py found: the glob is broken, not the code"


# -- no slot ----------------------------------------------------------------------------------------------
def test_no_part_declares_a_slot():
    offenders = {stem: slots_declared(_tree(stem)) for stem in [*PARTS, LEAF]}
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, (
        f"@Slot methods in {offenders}: leave them in plots_panel.py, where signal.connect(self.<slot>) "
        f"is torn down with the window")


def test_the_slot_scan_would_catch_each_spelling():
    for source in ("@Slot\ndef f(self): pass\n", "@QtCore.Slot()\ndef f(self): pass\n",
                   "class _AMixin:\n    @QtCore.Slot(object, name='x')\n    def f(self, n): pass\n"):
        assert slots_declared(ast.parse(source)) == ["f"], source
    assert slots_declared(ast.parse("@staticmethod\ndef f(): pass\n")) == []


def test_every_registered_slot_is_defined_in_plots_window_itself():
    """The meta-object's view of the rule above: it does not depend on how ``Slot`` was spelled or where."""
    from nanoscope.gui.plots_panel import PlotsWindow

    assert slot_owners_outside_the_class(PlotsWindow) == {}


# -- a part is a plain mixin ------------------------------------------------------------------------------
@pytest.mark.parametrize("stem", PARTS)
def test_a_part_is_a_plain_mixin_that_imports_only_the_leaf(stem):
    assert part_problems(_tree(stem)) == []


def test_the_leaf_imports_no_plots_panel_and_no_part():
    assert window_files_imported(_tree(LEAF)) == set()


def test_the_part_rules_would_catch_each_violation():
    cases = {
        "class _AMixin(QtWidgets.QMainWindow):\n    pass\n": "has a base class",
        "class _AMixin(metaclass=Meta):\n    pass\n": "has a base class, a metaclass or a decorator",
        "@dataclass\nclass _AMixin:\n    pass\n": "has a base class, a metaclass or a decorator",
        "class _AMixin:\n    def __init__(self):\n        pass\n": "defines __init__",
        "class _AMixin:\n    def closeEvent(self, e):\n        pass\n": "defines closeEvent, a name QMainWindow has",
        "class _AMixin:\n    def update(self):\n        pass\n": "defines update, a name QMainWindow has",
        "class _AMixin:\n    style = 1\n": "defines style, a name QMainWindow has",
        "class _AMixin:\n    show: bool = True\n": "defines show, a name QMainWindow has",
        "class _AMixin:\n    changed = QtCore.Signal()\n": "declares the signal changed",
        "class _AMixin:\n    changed = Signal(int)\n": "declares the signal changed",
        "class _AMixin:\n    changed: object = QtCore.Signal(int)\n": "declares the signal changed",
        "class _AMixin:\n    def f(self):\n        super().f()\n": "calls super()",
        "class _AMixin:\n    def f(self):\n        return __class__\n": "names __class__",
        "from .plots_panel import PlotConfig\nclass _AMixin:\n    pass\n": "imports ['plots_panel']",
        "class _AMixin:\n    def f(self):\n        from .plots_panel import PlotConfig\n": "imports ['plots_panel']",
        "from . import plots_fit_mixin\nclass _AMixin:\n    pass\n": "imports ['plots_fit_mixin']",
        "from nanoscope.gui import plots_views_mixin\nclass _AMixin:\n    pass\n": "imports ['plots_views_mixin']",
        "import nanoscope.gui.plots_style_mixin\nclass _AMixin:\n    pass\n": "imports ['plots_style_mixin']",
        "import nanoscope.gui.plots_panel as pp\nclass _AMixin:\n    pass\n": "imports ['plots_panel']",
        "x = 1\n": "does not define exactly one *Mixin class",
        "class _AMixin:\n    pass\nclass _BMixin:\n    pass\n": "does not define exactly one *Mixin class",
    }
    for source, expected in cases.items():
        problems = part_problems(ast.parse(source))
        assert any(expected in p for p in problems), (source, problems)
    allowed = ("from .plots_constants import HISTOGRAM\nfrom .plots_series import descriptors\n"
               "from .fit_label import FitReadout\n"
               "class _AMixin:\n    _LIMIT = 3\n    @staticmethod\n    def f():\n        return HISTOGRAM\n")
    assert part_problems(ast.parse(allowed)) == []


def test_no_mixin_defines_a_name_qmainwindow_has():
    """The runtime view of the plain-mixin rule: every name in the mixins' own dicts, not only the ones a scan of
    the class body reads."""
    from nanoscope.gui.plots_panel import PlotsWindow

    assert mixin_names_qmainwindow_has(PlotsWindow) == {}


def test_no_part_holds_a_signal_whatever_it_is_called():
    from nanoscope.gui.plots_panel import PlotsWindow

    assert signals_held_by_mixins(PlotsWindow) == {}


# -- the names tests patch on plots_panel -----------------------------------------------------------------
@pytest.mark.parametrize("stem", [LEAF, *PARTS])
def test_a_name_patched_on_plots_panel_is_read_only_in_plots_panel(stem):
    used = patched_names_used(_tree(stem))
    assert used == set(), (
        f"{stem} uses {sorted(used)}: tests patch these on nanoscope.gui.plots_panel, and a body in "
        f"{stem}.py reads {stem}'s own copy, so the patch would silently stop reaching it")


def test_the_patched_name_scan_would_catch_a_use():
    assert patched_names_used(ast.parse("from .export_paths import ask_export_path\n")) == {"ask_export_path"}
    assert patched_names_used(ast.parse("import nanoscope.gui.plot_export as kde_curve\n")) == {"kde_curve"}
    assert patched_names_used(ast.parse("def f():\n    return kde_curve(1)\n")) == {"kde_curve"}
    assert patched_names_used(ast.parse("def f(self):\n    return self.ask_export_path\n")) == set()
    assert patched_names_used(_tree("plots_panel")) == PATCHED_ON_PLOTS_PANEL, (
        "premise: plots_panel.py still reads every one of them, so the list is not stale")


# -- one state, not two -----------------------------------------------------------------------------------
def test_every_leaf_name_is_the_same_object_on_plots_panel():
    """``plots_panel`` imports the leaf's names back, and nothing rebinds one afterwards (``_TIME_UNITS =
    dict(_TIME_UNITS)`` in ``plots_panel.py`` would be a second copy a part never sees)."""
    from nanoscope.gui import plots_constants, plots_panel

    leaf, owner = vars(plots_constants), vars(plots_panel)
    defined, imported = leaf_definitions(_tree(LEAF))
    assert defined, "the scan found no definition in the leaf: it is broken, not the code"
    # the scan sees every name the leaf binds at run time, or a name it misses would be skipped silently
    assert {n for n in leaf if not (n.startswith("__") and n.endswith("__"))} - imported == defined
    assert leaf_name_problems(defined, leaf, owner) == []
    # and the names the leaf only imports are the same objects on plots_panel when it has them too
    assert sorted(n for n in imported if n in owner and n in leaf and owner[n] is not leaf[n]) == []


# -- the window's bases -----------------------------------------------------------------------------------
def test_every_part_is_a_base_of_plots_window_ahead_of_qmainwindow():
    from nanoscope.gui.plots_panel import PlotsWindow

    assert base_problems(PlotsWindow, PARTS) == []


def test_no_name_is_defined_in_two_of_the_windows_classes():
    from nanoscope.gui.plots_panel import PlotsWindow

    assert names_defined_twice(PlotsWindow) == {}


# -- the lists of parts the tests keep by hand ------------------------------------------------------------
def test_the_parts_test_measure_navigation_names_by_hand_are_the_files_that_exist():
    """Its allowlist and its per-part reach pins name the parts one by one: a part added later must reach both.
    (The lists in test_subpop_views, test_p1_guards and test_p0_final_seams are read from the directory.)"""
    tree = ast.parse((TESTS / "test_measure_navigation.py").read_text(encoding="utf-8"))
    assert parts_named_by_hand(tree) == {LEAF, *PARTS}


def test_the_hand_list_scan_would_see_a_missing_or_a_stale_name():
    source = ast.parse('names = {"plots_constants", "nanoscope.gui.plots_fit_mixin", "plots_series", "x"}\n')
    assert parts_named_by_hand(source) == {"plots_constants", "plots_fit_mixin"}
    assert parts_named_by_hand(source) != {LEAF, *PARTS}
