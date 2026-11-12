"""Both PyInstaller specs bundle ``docs/USER_GUIDE.md`` where the app looks for it, and the two
release checks look for the same file there.

Help > User guide… reads the guide through ``importlib.resources`` from the ``nanoscope`` package
folder of the bundle (``nanoscope/gui/user_guide_window.py``), the folder the build record already
goes to. PyInstaller is not run here; the specs are parsed, not executed. A bundle without the
guide still builds and starts (both Help entries are greyed out), so the self-test and
``verify_bundle.py`` each carry a check for it.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

from nanoscope.gui.user_guide_window import GUIDE_NAME

ROOT = Path(__file__).resolve().parents[2]
SPECS = ("NanoScope.spec", "NanoScope-win.spec")


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    (node,) = [n for n in tree.body
               if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets)]
    return node.value


@pytest.mark.parametrize("filename", SPECS)
def test_the_spec_adds_the_guide_to_the_package_folder(filename):
    tree = ast.parse((ROOT / "packaging" / filename).read_text())
    datas = _assignment(tree, "datas")
    pairs = [tuple(ast.unparse(e) for e in item.elts)
             for item in datas.elts if isinstance(item, ast.Tuple) and len(item.elts) == 2]
    assert ("str(ROOT / 'docs' / 'USER_GUIDE.md')", "'nanoscope'") in pairs, pairs
    assert (ROOT / "docs" / GUIDE_NAME).is_file(), "the file the spec names does not exist"


@pytest.mark.parametrize("filename", SPECS)
def test_no_prune_fragment_of_the_spec_can_drop_the_bundled_guide(filename):
    """Both specs filter every collected file against a list of path fragments."""
    tree = ast.parse((ROOT / "packaging" / filename).read_text())
    fragments = [n.value for n in ast.walk(_assignment(tree, "_PRUNE_FRAGMENTS"))
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert len(fragments) > 20, "the prune list was not read"
    destination = f"nanoscope/{GUIDE_NAME}"
    assert [f for f in fragments if f in destination] == []


def _packaging_module(name: str):
    spec = importlib.util.spec_from_file_location(name + "_under_test", ROOT / "packaging" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _folders_the_specs_add_the_guide_to() -> set[str]:
    folders = set()
    for filename in SPECS:
        datas = _assignment(ast.parse((ROOT / "packaging" / filename).read_text()), "datas")
        for item in datas.elts:
            if isinstance(item, ast.Tuple) and len(item.elts) == 2 \
                    and f"'{GUIDE_NAME}'" in ast.unparse(item.elts[0]):
                folders.add(ast.literal_eval(item.elts[1]))
    return folders


def test_verify_bundle_refuses_a_bundle_without_the_guide_where_the_specs_put_it(tmp_path):
    verifier = _packaging_module("verify_bundle")
    (folder,) = _folders_the_specs_add_the_guide_to()           # both specs, one place
    app = tmp_path / "NanoScope.app"
    in_the_bundle = app / "Contents" / "Frameworks" / folder
    in_the_bundle.mkdir(parents=True)
    with pytest.raises(SystemExit, match="user guide is not in the bundle"):
        verifier.check_user_guide(app)
    (in_the_bundle / GUIDE_NAME).write_text("# NanoScope — User Guide\n", encoding="utf-8")
    verifier.check_user_guide(app)


def test_verify_bundle_runs_that_check_with_the_others():
    tree = ast.parse((ROOT / "packaging" / "verify_bundle.py").read_text(encoding="utf-8"))
    (main,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    called = [c.func.id for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert "check_user_guide" in called, "build_app.sh would not run the guide check"


def test_the_self_test_reads_the_guide_through_the_apps_own_reader():
    """The self-test runs inside the built bundle, so it cannot be called from here: what is
    checked is that it holds a check, named for the guide, that asks the same reader Help does
    (which looks for ``GUIDE_NAME`` in the ``nanoscope`` folder the specs add it to) and that
    insists on the section Help > Keyboard shortcuts opens at."""
    tree = ast.parse((ROOT / "packaging" / "nanoscope_selftest.py").read_text(encoding="utf-8"))
    checks = [fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
              and any(isinstance(d, ast.Call) and getattr(d.func, "id", "") == "check"
                      and d.args and isinstance(d.args[0], ast.Constant)
                      and "user guide" in d.args[0].value for d in fn.decorator_list)]
    assert len(checks) == 1, "the self-test needs exactly one check for the user guide"
    body = ast.unparse(checks[0])
    assert "from nanoscope.gui.user_guide_window import" in body
    assert "read_guide()" in body and "SHORTCUTS_HEADING" in body
