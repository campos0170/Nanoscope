"""C11 + C18 — an export must confirm the files it ACTUALLY writes.

C11: *Analysis ▸ Export figure…* proposed ``figures.pdf`` — a file the command never writes
— and then wrote ``{stem}_{dwell,depth,scatter,trace}.{pdf,png,csv}``: up to **twelve** files,
none of them named in the dialog, all silently overwritten. The ``.csv`` source-data files are
the ones a user is most likely to have hand-edited.

C18: *Export events for training…* proposed ``events_training.csv`` and also wrote
``events_training.meta.json`` beside it, equally unmentioned.

The fix is one seam (``nanoscope.gui.export_paths.ask_export_path``) plus a writer-owned
enumeration of outputs on each side (``figures.figure_output_paths``,
``scoring.training_export_paths``). The last test in this file is the AST guard that keeps
every future export inside the seam — the reason this is not another one-call-site patch.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import json
import os
import pathlib
import textwrap

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.scoring import training_export_paths, write_training_export
from nanoscope.gui import export_paths as EP
from nanoscope.plot import figures

GUI_DIR = pathlib.Path(nanoscope.__file__).parent / "gui"


# ================================================================================================
# helpers — drive the seam without a live dialog
# ================================================================================================
class _Dialog:
    """Stands in for QFileDialog.getSaveFileName + the overwrite confirmation."""

    def __init__(self, chosen, *, answer=True):
        self.chosen = str(chosen)
        self.answer = answer
        self.asked_about: list[pathlib.Path] | None = None
        self.n_asks = 0

    def get_save_file_name(self, *a, **kw):
        return (self.chosen, "")

    def confirm(self, parent, title, existing):
        self.n_asks += 1
        self.asked_about = list(existing)
        return self.answer


@pytest.fixture
def dialog(monkeypatch):
    def install(chosen, *, answer=True):
        d = _Dialog(chosen, answer=answer)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            staticmethod(d.get_save_file_name))
        monkeypatch.setattr(EP, "confirm_overwrite", d.confirm)
        return d
    return install


def _touch(paths, text="PRECIOUS HAND-EDITED SOURCE DATA"):
    for p in paths:
        pathlib.Path(p).write_text(text)


# ================================================================================================
# the writers declare what they write
# ================================================================================================
def test_figure_output_paths_names_all_three_siblings():
    """The single ``.pdf`` a caller passes stands for a whole family."""
    got = [p.name for p in figures.figure_output_paths("/tmp/out/fig_dwell.pdf")]
    assert got == ["fig_dwell.pdf", "fig_dwell.png", "fig_dwell.csv"]


def test_figure_output_paths_ignores_the_extension_it_is_handed():
    """``save_figure`` replaces the extension, so the declaration must too — otherwise
    choosing ``figures.png`` would confirm a different set than the write touches."""
    for given in ("fig.pdf", "fig.png", "fig.csv", "fig"):
        assert [p.name for p in figures.figure_output_paths(f"/tmp/{given}")] == \
            ["fig.pdf", "fig.png", "fig.csv"]


def test_save_figure_writes_only_what_it_declared(tmp_path):
    """The declaration is checked against reality, not trusted."""
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0, 1, 2], [1.0, 2.0, 3.0], label="x")
    out = figures.save_figure(fig, tmp_path / "fig.pdf")
    matplotlib.pyplot.close(fig)

    declared = set(figures.figure_output_paths(tmp_path / "fig.pdf"))
    assert set(out.values()) <= declared
    assert set(tmp_path.iterdir()) <= declared


def test_training_export_paths_names_the_meta_sidecar():
    """C18: the sidecar is half the write and was never mentioned."""
    got = [p.name for p in training_export_paths("/tmp/t/events_training.csv")]
    assert got == ["events_training.csv", "events_training.meta.json"]


def test_training_export_paths_applies_the_writers_own_suffix_rule():
    """A bare name becomes ``.csv`` in the writer; the declaration must agree, or the
    confirmation is about a file that is not the one replaced."""
    assert [p.name for p in training_export_paths("/tmp/t/events")] == \
        ["events.csv", "events.meta.json"]
    assert [p.name for p in training_export_paths("/tmp/t/events.parquet")] == \
        ["events.parquet", "events.meta.json"]


def test_events_export_paths_names_the_meta_sidecar():
    """``Analysis > Export events...`` used to write a bare ``to_csv`` with nothing
    beside it, so an exported table carried NO record of which open-pore estimator
    produced its depths -- and that choice moves ``depth_pA`` by up to 15x on the
    owner's own recordings. It now writes the same two files the training export
    does, and both must be declared or the confirmation is about the wrong set."""
    from nanoscope.analysis.scoring import events_export_paths

    assert [p.name for p in events_export_paths("/tmp/t/events.csv")] == \
        ["events.csv", "events.meta.json"]
    assert [p.name for p in events_export_paths("/tmp/t/events")] == \
        ["events.csv", "events.meta.json"]


def test_write_events_export_writes_exactly_its_declared_paths(tmp_path):
    from nanoscope.analysis.scoring import events_export_paths, write_events_export

    df = pd.DataFrame({"start_idx": [0, 10], "end_idx": [5, 15],
                       "depth_pA": [50.0, 60.0], "accepted": [True, True]})
    out = write_events_export(df, tmp_path / "ev",
                              provenance={"fs": 50000.0, "polarity": -1,
                                          "baseline_fingerprint": "sha256:deadbeef00"})
    assert set(out.values()) == set(events_export_paths(tmp_path / "ev"))
    assert set(tmp_path.iterdir()) == set(events_export_paths(tmp_path / "ev"))
    meta = json.loads(out["meta"].read_text())
    assert meta["provenance"]["baseline_fingerprint"] == "sha256:deadbeef00"


def test_main_window_events_export_declares_the_sidecar():
    """The AST-level half: `_export_events` must hand `ask_export_path` the enumerator,
    not the single name it was given. It used to hand it nothing."""
    import ast
    import inspect

    from nanoscope.gui import main_window as MW

    src = inspect.getsource(MW.MainWindow._export_events)
    tree = ast.parse(textwrap.dedent(src))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ask_export_path"]
    assert len(calls) == 1
    outputs = [kw for kw in calls[0].keywords if kw.arg == "outputs"]
    assert outputs and getattr(outputs[0].value, "id", None) == "events_export_paths"


def test_write_training_export_writes_exactly_its_declared_paths(tmp_path):
    df = pd.DataFrame({
        "start_idx": [0, 10], "end_idx": [5, 15], "dwell_s": [1e-4, 2e-4],
        "depth_pA": [50.0, 60.0], "accepted": [True, True],
    })
    out = write_training_export(df, tmp_path / "t")          # bare name -> .csv
    assert set(out.values()) == set(training_export_paths(tmp_path / "t"))
    assert set(tmp_path.iterdir()) == set(training_export_paths(tmp_path / "t"))


# ================================================================================================
# the seam
# ================================================================================================
def test_ask_export_path_confirms_every_existing_output_not_just_the_chosen_name(
        tmp_path, dialog, qapp):
    """THE C11 assertion, at the seam.

    The user picks ``figures``; twelve real outputs already exist; all twelve must be named.
    """
    names = ["dwell", "depth", "scatter", "trace"]
    real = [p for n in names for p in figures.figure_output_paths(tmp_path / f"figures_{n}.pdf")]
    _touch(real)
    assert len(real) == 12

    d = dialog(tmp_path / "figures")
    got = EP.ask_export_path(
        None, title="Export figures", suggestion="figures", filters="*",
        outputs=lambda p: [q for n in names
                           for q in figures.figure_output_paths(f"{p.with_suffix('')}_{n}.pdf")])

    assert got == tmp_path / "figures"
    assert d.n_asks == 1
    assert sorted(p.name for p in d.asked_about) == sorted(p.name for p in real)
    # ...and the file the DIALOG named is not among them, because it is never written.
    assert "figures.pdf" not in {p.name for p in d.asked_about}


def test_declining_the_confirmation_aborts_the_export(tmp_path, dialog, qapp):
    _touch(figures.figure_output_paths(tmp_path / "fig.pdf"))
    d = dialog(tmp_path / "fig", answer=False)
    assert EP.ask_export_path(None, title="t", suggestion="fig", filters="*",
                              outputs=figures.figure_output_paths) is None
    assert d.n_asks == 1


def test_nothing_is_confirmed_when_nothing_would_be_replaced(tmp_path, dialog, qapp):
    """A first-time export must not nag."""
    d = dialog(tmp_path / "fig")
    assert EP.ask_export_path(None, title="t", suggestion="fig", filters="*",
                              outputs=figures.figure_output_paths) == tmp_path / "fig"
    assert d.asked_about == []


def test_a_cancelled_dialog_never_reaches_the_confirmation(tmp_path, dialog, qapp):
    d = dialog("")
    assert EP.ask_export_path(None, title="t", suggestion="x", filters="*") is None
    assert d.n_asks == 0


def test_existing_of_reports_only_real_files(tmp_path, qapp):
    f = tmp_path / "there.csv"
    f.write_text("x")
    d = tmp_path / "a_directory"
    d.mkdir()
    got = EP.existing_of([f, tmp_path / "absent.csv", d, f])   # note the duplicate
    assert got == [f], "a directory is not an overwrite victim, and duplicates must collapse"


def test_the_native_overwrite_prompt_is_suppressed(tmp_path, monkeypatch, qapp):
    """Two prompts, the first of them wrong, is worse than one correct prompt.

    The native confirmation asks about the single chosen name; ours asks about the real
    set. Leaving both on would show the misleading one first.
    """
    seen = {}

    def fake(parent, title, suggestion, filters, options=None, **kw):
        seen["options"] = options
        return (str(tmp_path / "x.csv"), "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake))
    EP.ask_export_path(None, title="t", suggestion="x.csv", filters="*")
    assert seen["options"] == QtWidgets.QFileDialog.Option.DontConfirmOverwrite


# ================================================================================================
# THE GUARD — this is what stops the next export reintroducing C11
# ================================================================================================
def _save_dialog_calls(src: str) -> list[str]:
    """Every ``...getSaveFileName(...)`` call in ``src``, by attribute name."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "getSaveFileName":
                out.append(node.func.attr)
    return out


def test_no_gui_module_opens_a_save_dialog_outside_the_export_seam():
    """A save dialog anywhere else is a save dialog whose overwrite prompt nobody checked.

    This is the structural half of the fix: C11 and C18 were the same defect in two
    commands, and patching those two would have left the next one free to repeat it.
    """
    offenders = []
    for path in sorted(GUI_DIR.glob("*.py")):
        if path.name == "export_paths.py":
            continue
        if _save_dialog_calls(path.read_text()):
            offenders.append(path.name)
    assert offenders == [], (
        f"{offenders} call QFileDialog.getSaveFileName directly; route them through "
        f"nanoscope.gui.export_paths.ask_export_path so the overwrite prompt names the "
        f"files the write will actually touch")


def test_the_guard_would_actually_catch_a_violation():
    """A guard nobody has seen fail is a guard nobody has tested."""
    violation = (
        "class P:\n"
        "    def save(self):\n"
        "        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 't', 'x.csv', '*')\n"
    )
    assert _save_dialog_calls(violation) == ["getSaveFileName"]
    assert _save_dialog_calls("path = ask_export_path(self, title='t')\n") == []


def test_export_paths_is_the_only_gui_module_that_needs_the_seam():
    """Sanity: the seam module itself really does own the dialog."""
    assert _save_dialog_calls((GUI_DIR / "export_paths.py").read_text()) == ["getSaveFileName"]


# ================================================================================================
# main_window's two commands, end to end
# ================================================================================================
def test_main_window_figure_export_enumerates_all_twelve_paths(tmp_path):
    """``MainWindow._figure_export_paths`` is what the seam is handed; it must expand to the
    same names the write loop produces."""
    from nanoscope.gui.main_window import MainWindow

    got = MainWindow._figure_export_paths(tmp_path / "figures",
                                          ["dwell", "depth", "scatter", "trace"])
    assert len(got) == 12
    assert sorted(p.name for p in got) == sorted(
        f"figures_{n}.{e}" for n in ("dwell", "depth", "scatter", "trace")
        for e in ("pdf", "png", "csv"))


def test_main_window_figure_export_paths_match_what_save_figure_writes(tmp_path):
    """Belt and braces: the enumeration is derived from the writer, so a change to
    ``save_figure``'s outputs cannot leave the confirmation behind."""
    from nanoscope.gui.main_window import MainWindow

    names = ["dwell", "scatter"]
    declared = set(MainWindow._figure_export_paths(tmp_path / "f", names))
    from_writer = {p for n in names
                   for p in figures.figure_output_paths(tmp_path / f"f_{n}.pdf")}
    assert declared == from_writer


def test_project_save_as_confirms_the_suffixed_name(tmp_path, dialog, qapp):
    """Typing ``study`` over an existing ``study.nanoscope`` must warn.

    The write appends ``.nanoscope``; if the confirmation used the raw typed name it would
    look at a file that does not exist and sail straight past the one it is about to replace.
    """
    from nanoscope.gui.main_window import _with_project_suffix

    existing = tmp_path / "study.nanoscope"
    existing.write_text("an earlier project")
    d = dialog(tmp_path / "study")
    got = EP.ask_export_path(
        None, title="Save Project", suggestion="study", filters="*",
        outputs=lambda p: (pathlib.Path(_with_project_suffix(str(p))),))
    assert got == tmp_path / "study"
    assert [p.name for p in d.asked_about] == ["study.nanoscope"]


# ================================================================================================
# Return must not overwrite  (found by the bundle verification)
# ================================================================================================
#
# `setDefaultButton(box.buttons()[-1])` was written to mean "Cancel" -- Cancel is added second.
# But QMessageBox.buttons() returns LAYOUT order, not insertion order, and macOS lays the
# destructive button out last, so the Enter key was bound to REPLACE on the one prompt whose
# entire job is to stop an accidental overwrite. Measured in the shipped bundle:
# buttons ['Cancel','Replace'], defaultButton 'Replace', escapeButton None. Pressing Return on
# the figure-export prompt destroyed the .pdf, the .png AND the .csv source data -- the last of
# which is the file most likely to have been hand-edited.
#
# The fix is to name the buttons instead of indexing them. These tests assert the binding, not
# the layout, so they hold whatever order a future Qt or platform style chooses.

def test_return_and_escape_both_keep_the_users_files(tmp_path, qapp):
    """Both keyboard defaults must resolve to Cancel, by identity -- never by position."""
    from PySide6 import QtWidgets

    import nanoscope.gui.export_paths as EP

    captured = {}

    real_exec = QtWidgets.QMessageBox.exec

    def _capture(self):
        # Read everything to plain Python HERE: the QMessageBox is destroyed when
        # confirm_overwrite returns, and a QPushButton captured out of it is a dangling
        # C++ object by the time the assertions run ("Internal C++ object already deleted").
        default, escape = self.defaultButton(), self.escapeButton()
        captured["default"] = None if default is None else default.text()
        captured["escape"] = None if escape is None else escape.text()
        captured["texts"] = [b.text() for b in self.buttons()]
        captured["roles"] = [self.buttonRole(b) for b in self.buttons()]
        return 0                      # do not block; clickedButton() stays None -> declined

    QtWidgets.QMessageBox.exec = _capture
    try:
        victim = tmp_path / "figures_dwell.pdf"
        victim.write_bytes(b"hand-edited")
        assert EP.confirm_overwrite(None, title="Export figures", existing=[victim]) is False
    finally:
        QtWidgets.QMessageBox.exec = real_exec

    assert captured["default"] is not None, "no default button: Return would do nothing"
    assert captured["default"] == "Cancel"
    assert captured["escape"] is not None, "no escape button: Esc would do nothing"
    assert captured["escape"] == "Cancel"
    # the destructive button is still present and still named
    assert "Replace" in captured["texts"]
    assert QtWidgets.QMessageBox.DestructiveRole in captured["roles"]
    # and the file the user was warned about is untouched by the refusal
    assert victim.read_bytes() == b"hand-edited"


def test_the_default_is_cancel_whatever_order_qt_lays_the_buttons_out(tmp_path, qapp):
    """The defect was positional, so pin it positionally: whichever end 'Replace' lands on,
    it must not be the default. This is the assertion that would have failed before the fix."""
    from PySide6 import QtWidgets

    import nanoscope.gui.export_paths as EP

    seen = {}
    real_exec = QtWidgets.QMessageBox.exec

    def _capture(self):
        seen["default_is_last"] = self.defaultButton() is self.buttons()[-1]
        seen["default_text"] = self.defaultButton().text()
        seen["last_text"] = self.buttons()[-1].text()
        return 0

    QtWidgets.QMessageBox.exec = _capture
    try:
        EP.confirm_overwrite(None, title="t", existing=[tmp_path / "x.pdf"])
    finally:
        QtWidgets.QMessageBox.exec = real_exec

    assert seen["default_text"] == "Cancel"
    if seen["last_text"] == "Replace":
        assert not seen["default_is_last"], "buttons()[-1] is Replace and it is the default"
