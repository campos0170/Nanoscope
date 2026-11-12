"""The rules ``main_window.py`` is split under (``main_window_<part>.py`` mixins, a shared leaf, a seams module).

``MainWindow`` keeps its constructor, its Qt overrides, ``_wire``, the menus, every ``@Slot`` and
``_finalize_restore``, and takes 263 of its 579 callables (functions, static and class methods and properties;
the other 316 stay in the class: counted on 2026-11-11) from plain-Python mixin classes listed before
``QMainWindow``. A method moves by being cut out of the class and pasted, byte for byte, into a mixin. These
tests hold what that move depends on (``test_plots_window_parts.py`` holds the same for ``PlotsWindow``):

* no mixin declares a ``@Slot``: from a mixin it registers on ``MainWindow``'s meta-object all the same, but
  ``signal.connect(self.<slot>)`` then goes through PySide's global receiver, and Qt does not remove that
  connection when the window's C++ object is deleted (measured: ``QGuiApplication.applicationStateChanged``
  kept a receiver and called the dead window's slot on its next emission). Two rules hold it: one reads the
  decorators (so it names the file), and one reads the meta-object, which does not care how ``Slot`` is spelled
  (``from PySide6.QtCore import Slot as S``): every registered slot has its function in ``MainWindow`` itself;
* a part is a plain class with no ``__init__``, no Qt override and no ``super()``, and never imports
  ``main_window`` or another part at module scope (it may import ``main_window_shared`` and ``main_window_seams``);
  no mixin defines any name ``QMainWindow`` has, whether or not it is on the short list of Qt overrides below,
  and none holds a ``Signal`` (however it was imported);
* the seams forward, at call time, to the three names tests patch on ``main_window``, and a part that binds one
  of them binds that forwarder (and holds ``main_window``, ``MainWindow`` or another part as no module-level name);
* every name ``main_window_shared`` defines is the same object on ``main_window``; every part file is a base of
  ``MainWindow``, before ``QMainWindow``; and no name is defined in two of the window's classes (the copy that
  ran would depend on the order of the bases).

The rule functions that are not specific to ``MainWindow`` (``slots_declared``, ``slot_owners_outside_the_class``,
``mixin_names_qmainwindow_has``, ``signals_held_by_mixins``, ``names_defined_twice``, ``base_problems``,
``leaf_definitions``, ``leaf_name_problems``) are used by ``test_plots_window_parts.py`` as well, and are tried on
violating input here.
"""
from __future__ import annotations

import ast
import os
import pathlib

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

GUI = pathlib.Path(__import__("nanoscope.gui", fromlist=["x"]).__file__).parent
SUPPORT = {"main_window_shared", "main_window_seams"}
PARTS = sorted(p.stem for p in GUI.glob("main_window_*.py") if p.stem not in SUPPORT)
#: Qt overrides: a mixin cannot supply one, because ``super()`` inside it would resolve differently.
QT_OVERRIDES = {"event", "eventFilter", "closeEvent", "showEvent", "hideEvent", "resizeEvent", "moveEvent",
                "changeEvent", "focusNextPrevChild", "dragEnterEvent", "dragMoveEvent", "dragLeaveEvent",
                "dropEvent", "keyPressEvent", "keyReleaseEvent", "mousePressEvent", "mouseReleaseEvent",
                "paintEvent", "contextMenuEvent", "minimumSizeHint", "sizeHint"}


def _tree(stem: str) -> ast.AST:
    return ast.parse((GUI / f"{stem}.py").read_text())


def slots_declared(tree: ast.AST) -> list[str]:
    """Names of the functions in ``tree`` decorated with ``Slot`` (``@Slot``, ``@QtCore.Slot``, with or without a call)."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name == "Slot":
                out.append(node.name)
    return out


def part_problems(tree: ast.AST) -> list[str]:
    """Every rule a part module breaks, as sentences."""
    found = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1 or not classes[0].name.endswith("Mixin"):
        found.append("does not define exactly one *Mixin class")
    for cls in classes:
        if cls.bases or cls.keywords:
            found.append(f"{cls.name} has a base class")
        for member in cls.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if member.name == "__init__":
                    found.append(f"{cls.name} defines __init__")
                if member.name in QT_OVERRIDES:
                    found.append(f"{cls.name} overrides {member.name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "super":
            found.append("calls super()")
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            name = node.module or ""
            if name == "main_window" or (name.startswith("main_window_") and name not in SUPPORT):
                found.append(f"imports {name} at module scope")
            if not name and any(a.name == "main_window" or (a.name.startswith("main_window_") and a.name not in SUPPORT)
                                for a in node.names):
                found.append("imports a main_window module at module scope")
    return found


#: The three names tests patch on ``main_window``; the parts reach them through ``main_window_seams``.
SEAM_NAMES = ("ask_export_path", "locked_project_read", "_on_screen")


def slot_owners_outside_the_class(window_cls) -> dict[str, str | None]:
    """The registered slots of ``window_cls``'s meta-object whose function is not in the class's own ``__dict__``,
    as ``{signature: name of the class that defines it}``.

    ``slots_declared`` reads decorators, so a slot written under an alias (``from PySide6.QtCore import Slot as S``)
    passes it; the meta-object lists every registered slot whatever it was called when it was written.
    """
    from PySide6 import QtCore

    meta = window_cls.staticMetaObject
    stray = {}
    for index in range(meta.methodOffset(), meta.methodCount()):
        method = meta.method(index)
        if method.methodType() == QtCore.QMetaMethod.Slot:
            name = bytes(method.name()).decode()
            owner = next((c for c in window_cls.__mro__ if name in vars(c)), None)
            if owner is not window_cls:
                stray[bytes(method.methodSignature()).decode()] = getattr(owner, "__name__", None)
    return stray


def window_with_a_slot_in_a_mixin():
    """A ``QMainWindow`` subclass whose one slot was written in a mixin, under an alias of ``Slot``."""
    from PySide6 import QtWidgets
    from PySide6.QtCore import Slot as QSlot

    class _SavedMixin:
        @QSlot(int, object)
        def _on_saved(self, count, payload):
            pass

    class _Window(_SavedMixin, QtWidgets.QMainWindow):
        pass

    return _Window


def mixin_names_qmainwindow_has(window_cls) -> dict[str, list[str]]:
    """``{mixin: names}`` for every ``*Mixin`` in ``window_cls``'s MRO that defines a name ``QMainWindow`` (or one
    of its bases) also has: a Qt method it would shadow, or a Qt override it cannot supply (``super()`` inside a
    mixin resolves differently). ``QT_OVERRIDES`` is a short list of the usual ones; this is every name."""
    from PySide6 import QtWidgets

    qt_names = set(dir(QtWidgets.QMainWindow))
    out = {}
    for cls in window_cls.__mro__:
        if cls.__name__.endswith("Mixin"):
            hit = sorted(n for n in vars(cls) if n in qt_names and not (n.startswith("__") and n.endswith("__")))
            if hit:
                out[cls.__name__] = hit
    return out


def mixins_of(window_cls) -> list[type]:
    """The ``*Mixin`` classes ``window_cls`` lists as bases, in order."""
    return [c for c in window_cls.__bases__ if c.__name__.endswith("Mixin")]


def base_problems(window_cls, parts: list[str]) -> list[str]:
    """What is wrong with the mixins ``window_cls`` lists as bases, against the part files that exist: a part file
    that is not a base (or a base with no file), and a mixin that comes after ``QMainWindow`` in the MRO."""
    from PySide6 import QtWidgets

    found = []
    mixins = mixins_of(window_cls)
    named = sorted(c.__module__.rsplit(".", 1)[-1] for c in mixins)
    if named != sorted(parts):
        found.append(f"the mixins in the bases come from {named}, the part files are {sorted(parts)}")
    mro = window_cls.__mro__
    late = [c.__name__ for c in mixins if mro.index(c) > mro.index(QtWidgets.QMainWindow)]
    if late:
        found.append(f"{late} come after QMainWindow in the MRO")
    return found


def signals_held_by_mixins(window_cls) -> dict[str, list[str]]:
    """``{mixin: names}`` of the ``Signal`` objects a mixin holds, however ``Signal`` was imported. A signal
    declared in a plain mixin is not a signal of the window's own class body, where the window keeps them."""
    from PySide6 import QtCore

    held = {c.__name__: sorted(n for n, v in vars(c).items() if isinstance(v, QtCore.Signal))
            for c in mixins_of(window_cls)}
    return {name: signals for name, signals in held.items() if signals}


def names_defined_twice(window_cls) -> dict[str, list[str]]:
    """``{name: classes}`` for every name defined in more than one of the window's own classes (its class body and
    its mixins): the copy that runs would depend on the order of the bases, and a copy left behind by a move
    would be dead code that looks live."""
    owners: dict[str, list[str]] = {}
    for cls in (window_cls, *mixins_of(window_cls)):
        for name in vars(cls):
            if not (name.startswith("__") and name.endswith("__")):
                owners.setdefault(name, []).append(cls.__name__)
    return {name: where for name, where in owners.items() if len(where) > 1}


def leaf_definitions(tree: ast.Module) -> tuple[set[str], set[str]]:
    """``(defined, imported)``: the names a leaf module's body defines and the names it imports."""
    defined: set[str] = set()
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined |= {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.Import):
            imported |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {alias.asname or alias.name for alias in node.names}
    return defined, imported


def leaf_name_problems(defined: set[str], leaf: dict, owner: dict) -> list[str]:
    """The leaf's definitions the owner module does not import back, or holds a different object under."""
    return sorted(n for n in defined if n not in owner or owner[n] is not leaf[n])


def part_global_problems(stem: str, namespace: dict) -> list[str]:
    """What a part's module-level names must not be: a seam name bound from anywhere but ``main_window_seams`` (a
    patch on ``main_window`` would no longer reach the call), or ``main_window``, ``MainWindow``, another part or
    another part's mixin (the cycle the module-scope import rule exists to prevent, spelled another way)."""
    import inspect

    from nanoscope.gui import main_window, main_window_seams

    found = []
    for name in SEAM_NAMES:
        if name in namespace and namespace[name] is not getattr(main_window_seams, name):
            found.append(f"{stem}.{name} is not the main_window_seams forwarder")
    for name, value in namespace.items():
        if value is main_window or value is main_window.MainWindow:
            found.append(f"{stem} holds {name}, the main window, at module scope")
        elif inspect.ismodule(value) and value.__name__.rsplit(".", 1)[-1] in PARTS:
            found.append(f"{stem} holds {name}, another part, at module scope")
        elif (inspect.isclass(value) and value.__name__.endswith("Mixin")
              and value.__module__ != namespace.get("__name__")):
            found.append(f"{stem} holds {name}, another part's mixin, at module scope")
    return found


def test_no_part_declares_a_slot():
    assert PARTS, "no main_window_<part>.py files found: the glob is broken, not the code"
    offenders = {stem: slots_declared(_tree(stem)) for stem in [*PARTS, *sorted(SUPPORT)]}
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, (
        f"@Slot methods in {offenders}: leave them in main_window.py, where signal.connect(self.<slot>) "
        f"is torn down with the window")


def test_the_slot_scan_would_catch_each_spelling():
    for source in ("@Slot\ndef f(self): pass\n", "@QtCore.Slot()\ndef f(self): pass\n",
                   "@QtCore.Slot(int, name='x')\ndef f(self, n): pass\n"):
        assert slots_declared(ast.parse(source)) == ["f"], source
    assert slots_declared(ast.parse("@staticmethod\ndef f(): pass\n")) == []


@pytest.mark.parametrize("stem", PARTS)
def test_a_part_is_a_plain_mixin_that_imports_no_part_or_main_window(stem):
    assert part_problems(_tree(stem)) == []


def test_the_part_rules_would_catch_each_violation():
    cases = {
        "class _AMixin(QtWidgets.QMainWindow):\n    pass\n": "has a base class",
        "class _AMixin:\n    def __init__(self):\n        pass\n": "defines __init__",
        "class _AMixin:\n    def closeEvent(self, e):\n        pass\n": "overrides closeEvent",
        "class _AMixin:\n    def f(self):\n        super().f()\n": "calls super()",
        "from .main_window import MainWindow\nclass _AMixin:\n    pass\n": "imports main_window at module scope",
        "from .main_window_exports import _ExportsMixin\nclass _AMixin:\n    pass\n":
            "imports main_window_exports at module scope",
        "from . import main_window\nclass _AMixin:\n    pass\n": "imports a main_window module at module scope",
        "x = 1\n": "does not define exactly one *Mixin class",
    }
    for source, expected in cases.items():
        problems = part_problems(ast.parse(source))
        assert any(expected in p for p in problems), (source, problems)
    allowed = "from .main_window_shared import _CLOSE_SAVE\nfrom .main_window_seams import _on_screen\nclass _AMixin:\n    def f(self):\n        pass\n"
    assert part_problems(ast.parse(allowed)) == []


def test_the_shared_leaf_and_the_seams_import_main_window_only_where_allowed():
    shared = _tree("main_window_shared")
    for node in ast.walk(shared):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            assert (node.module or "").split(".")[0] not in {"main_window"} | set(PARTS), node.module
    seams = _tree("main_window_seams")
    # at module scope the seams import nothing from main_window (that would be a cycle) ...
    assert [n for n in seams.body if isinstance(n, (ast.Import, ast.ImportFrom))
            and (getattr(n, "module", None) or "").startswith("main_window")] == []
    # ... only inside the forwarders, at call time
    for fn in (n for n in seams.body if isinstance(n, ast.FunctionDef)):
        assert any(isinstance(n, ast.ImportFrom) and n.level == 1 and any(a.name == "main_window" for a in n.names)
                   for n in ast.walk(fn)), fn.name


@pytest.mark.parametrize("name", ["ask_export_path", "locked_project_read", "_on_screen"])
def test_a_seam_forwards_to_main_window_when_it_is_called(name, monkeypatch):
    from nanoscope.gui import main_window, main_window_seams

    seen = []

    def stand_in(*args, **kwargs):
        seen.append((args, kwargs))
        return ("answer", name)

    monkeypatch.setattr(main_window, name, stand_in)
    assert getattr(main_window_seams, name)(1, 2, key="v") == ("answer", name)
    assert seen == [((1, 2), {"key": "v"})]


def test_every_part_is_a_base_of_main_window_ahead_of_qmainwindow():
    from nanoscope.gui.main_window import MainWindow

    assert base_problems(MainWindow, PARTS) == []


def test_the_base_rule_would_catch_each_violation():
    from PySide6 import QtWidgets

    def mixin(name: str, stem: str):
        return type(name, (), {"__module__": f"nanoscope.gui.{stem}"})

    a, b = mixin("_AMixin", "main_window_a"), mixin("_BMixin", "main_window_b")

    class _Good(a, b, QtWidgets.QMainWindow):
        pass

    class _Late(QtWidgets.QMainWindow, a, b):
        pass

    assert base_problems(_Good, ["main_window_a", "main_window_b"]) == []
    assert base_problems(_Good, ["main_window_a", "main_window_b", "main_window_c"]) != []   # a part that is not a base
    assert base_problems(_Good, ["main_window_a"]) != []                                      # a base with no file
    assert base_problems(_Late, ["main_window_a", "main_window_b"]) != []                     # after QMainWindow


def test_every_registered_slot_is_defined_in_main_window_itself():
    """``test_no_part_declares_a_slot`` reads decorators, so it depends on how ``Slot`` is spelled and on which
    file it is looking in. This reads what Qt registered: every slot on ``MainWindow``'s meta-object has its
    function in ``vars(MainWindow)``, wherever and however it was written."""
    from nanoscope.gui.main_window import MainWindow

    assert slot_owners_outside_the_class(MainWindow) == {}


def test_the_slot_owner_rule_would_catch_a_slot_in_a_mixin_written_under_an_alias():
    window = window_with_a_slot_in_a_mixin()
    assert slot_owners_outside_the_class(window) == {"_on_saved(int,PyObject)": "_SavedMixin"}


def test_no_mixin_defines_a_name_qmainwindow_has():
    from nanoscope.gui.main_window import MainWindow

    assert mixin_names_qmainwindow_has(MainWindow) == {}


def test_the_qmainwindow_name_rule_would_catch_a_shadowed_qt_method():
    from PySide6 import QtWidgets

    class _QtMixin:
        _fine = 1

        def update(self):
            pass

    class _Window(_QtMixin, QtWidgets.QMainWindow):
        pass

    assert mixin_names_qmainwindow_has(_Window) == {"_QtMixin": ["update"]}


@pytest.mark.parametrize("stem", [*PARTS, "main_window_shared"])
def test_a_part_binds_the_patched_names_only_from_the_seams_and_holds_no_window_or_part(stem):
    import importlib

    module = importlib.import_module(f"nanoscope.gui.{stem}")
    assert part_global_problems(stem, vars(module)) == []


def test_the_module_global_rule_would_catch_each_violation():
    from nanoscope.gui import main_window, main_window_exports, main_window_seams

    own = "nanoscope.gui.main_window_x"
    cases = {
        "is not the main_window_seams forwarder": {"ask_export_path": lambda *args, **kwargs: None},
        # the real function, bound from where it is defined: a patch on main_window would not reach this part
        "is not the main_window_seams forwarder (the real home)": {"locked_project_read": main_window.locked_project_read},
        "the main window": {"mw": main_window},
        "the main window (the class)": {"MainWindow": main_window.MainWindow},
        "another part, at module scope": {"exports": main_window_exports},
        "another part's mixin": {"_ExportsMixin": main_window_exports._ExportsMixin},
    }
    for label, namespace in cases.items():
        problems = part_global_problems("main_window_x", {"__name__": own, **namespace})
        assert problems, label
    allowed = {"__name__": own, "seams": main_window_seams,
               "ask_export_path": main_window_seams.ask_export_path, "_on_screen": main_window_seams._on_screen,
               "_XMixin": type("_XMixin", (), {"__module__": own})}
    assert part_global_problems("main_window_x", allowed) == []


def test_no_part_holds_a_signal_whatever_it_is_called():
    from nanoscope.gui.main_window import MainWindow

    assert signals_held_by_mixins(MainWindow) == {}


def test_no_name_is_defined_in_two_of_the_windows_classes():
    from nanoscope.gui.main_window import MainWindow

    assert names_defined_twice(MainWindow) == {}


def test_every_shared_name_is_the_same_object_on_main_window():
    """``main_window`` imports the leaf's names back, and nothing rebinds one afterwards (``_DETECTOR_LABELS =
    dict(_DETECTOR_LABELS)`` in ``main_window.py`` would be a second copy no part ever sees)."""
    from nanoscope.gui import main_window, main_window_shared

    leaf, owner = vars(main_window_shared), vars(main_window)
    defined, imported = leaf_definitions(_tree("main_window_shared"))
    assert defined, "the scan found no definition in the leaf: it is broken, not the code"
    # the scan sees every name the leaf binds at run time, or a name it misses would be skipped silently
    assert {n for n in leaf if not (n.startswith("__") and n.endswith("__"))} - imported == defined
    assert leaf_name_problems(defined, leaf, owner) == []
    # and the names the leaf only imports are the same objects on main_window when it has them too
    assert sorted(n for n in imported if n in owner and n in leaf and owner[n] is not leaf[n]) == []


def test_the_shared_rules_would_catch_each_violation():
    """The rule functions ``test_plots_window_parts.py`` uses too, each on a violating window."""
    from PySide6 import QtWidgets
    from PySide6.QtCore import Signal as QSignal

    class _SignalMixin:
        changed = QSignal(int)

    class _Window(_SignalMixin, QtWidgets.QMainWindow):
        pass

    assert signals_held_by_mixins(_Window) == {"_SignalMixin": ["changed"]}

    def mixin(name: str, **members):
        return type(name, (), {"__module__": "nanoscope.gui.x", **members})

    a, b = mixin("_AMixin", f=lambda self: 1), mixin("_BMixin", g=lambda self: 1)

    class _Twice(a, b, QtWidgets.QMainWindow):
        def f(self):    # also defined in _AMixin: which copy runs depends on the order of the bases
            return 2

    class _TwoMixins(a, b, mixin("_CMixin", g=lambda self: 2), QtWidgets.QMainWindow):
        pass

    class _Good(a, b, QtWidgets.QMainWindow):
        pass

    assert names_defined_twice(_Good) == {}
    assert names_defined_twice(_Twice) == {"f": ["_Twice", "_AMixin"]}
    assert names_defined_twice(_TwoMixins) == {"g": ["_BMixin", "_CMixin"]}

    one, two = object(), object()
    assert leaf_name_problems({"A", "B"}, {"A": one, "B": two}, {"A": one, "B": two}) == []
    assert leaf_name_problems({"A", "B"}, {"A": one, "B": two}, {"A": one, "B": dict()}) == ["B"]   # forked
    assert leaf_name_problems({"A", "B"}, {"A": one, "B": two}, {"A": one}) == ["B"]                # not imported back
    defined, imported = leaf_definitions(ast.parse(
        "import numpy as np\nfrom .x import Y as Z\nA = 1\nB: int = 2\nC, D = 1, 2\ndef f(): pass\nclass K: pass\n"))
    assert defined == {"A", "B", "C", "D", "f", "K"} and imported == {"np", "Z"}


def test_a_deleted_window_leaves_no_connection_behind_on_the_application(qapp, qtbot, tmp_path):
    """The failure a ``@Slot`` in a mixin caused: the window's connection to the application's state signal
    outlived it, and the next emission called the dead window's slot."""
    import shiboken6
    from PySide6 import QtCore, QtGui

    from nanoscope.gui.main_window import MainWindow

    signal = QtCore.SIGNAL("applicationStateChanged(Qt::ApplicationState)")
    app = QtGui.QGuiApplication.instance()
    before = app.receivers(signal)
    window = MainWindow(recovery_root=tmp_path / "R")
    try:
        assert app.receivers(signal) == before + 1, "the window no longer listens for the application's state"
        window._dirty = False
        if window._recovery_controller is not None:
            window._recovery_controller.abort()
        window.close()
        window.deleteLater()
        qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=5000)
        assert app.receivers(signal) == before, "a deleted window is still connected to applicationStateChanged"
    finally:
        if shiboken6.isValid(window):
            window.close()
            window.deleteLater()
