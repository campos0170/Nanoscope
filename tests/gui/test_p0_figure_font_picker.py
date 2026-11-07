"""Phase 0, group G2: the figure font list, Arial as the default, a fallback for a missing family.

Owner decision 4 (2 November 2026): the font list is Arial (default), Helvetica, Times New
Roman and Courier New; every other installed font is behind "Other font…", off the main list;
an empty font resolves to Arial, with a fallback when a family is missing. This file drives it
the way a person does and reads what is DRAWN, not a state field:

* every picker (the Appearance section, Distributions' Style group, the Text style… popover,
  the composer) lists exactly the main list, and Other font… picks through a dialog (patched
  ``exec``, as the repo's other tests patch modal dialogs) or cancels back;
* an untouched figure in each window type draws its title, axis labels, tick numbers, legend
  and fit label in Arial -- the raw font of every glyph run Qt lays out (``glyphRuns``) and the
  tick ``QFont``, not ``PlotAppearance.font_family``;
* a family the computer lacks falls back to its substitute instead of the system UI font;
* no text in a figure -- a scene walk over every text item, a character the font lacks (the
  subscript zero of I₀ in Times New Roman), the level labels of Noise's histogram -- is drawn in
  the application font, and the PDF made from it embeds no dot-prefixed (private) font;
* Text style… reopened on a text with a font of its own shows it, and OK stores it as it was;
* a family an old project stored (Menlo, Times New Roman, an empty one, a missing Calibri) comes
  back from a real save and ``_open_project`` as the same value, on the right row;
* choosing a font through each picker repaints and is ONE Undo step; ``set_value`` says nothing;
* the user guide and the control inventory name what the picker has.

Synthetic data only; every recording is an in-memory document or a CSV written to ``tmp_path``.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text, font_picker
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_text import TextStyle, figure_text_for
from nanoscope.gui.font_picker import (DEFAULT_ROW_LABEL, FontListDialog, FontPicker, figure_family,
                                       figure_qfont, html_font_family)
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_tick_family
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.plot import figure_fonts as ff

from .test_figure_composer import entries  # noqa: F401 -- real frozen snapshots
from .test_plot_ux_quick_wins import _figure
from .test_plot_window_routes import (_background, _build, _choose, _click, _main_window,
                                      _names, _restored_main_window, _show)
from .test_plots_presentation_controls import _events
from .test_plots_reopen_figure import _install_events

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
Accepted = QtWidgets.QDialog.DialogCode.Accepted
Rejected = QtWidgets.QDialog.DialogCode.Rejected

MAIN_ROWS = ["Arial (default)", "Helvetica", "Times New Roman", "Courier New", "", "Other font…"]
KINDS = ("iv", "vdep", "noise", "events", "summaries", "subpop", "distributions")
#: The off-list family these tests pick and store: installed with macOS, so its name is real.
OTHER = "Menlo"


@pytest.fixture(autouse=True)
def _fresh_gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture
def arial(qapp):
    """These checks read the face Qt really draws, so they need Arial on this computer. (It asks
    the font database, which aborts the process when it runs before an application exists, so it
    takes ``qapp`` first.)"""
    if "arial" not in {f.casefold() for f in font_picker.installed_families()}:
        pytest.skip("this computer has no Arial")


# ================================================================================================
# reading what is DRAWN
# ================================================================================================

def drawn_families(label) -> set[str]:
    """The raw font of every glyph run Qt lays out for a pyqtgraph text item.

    A LabelItem keeps its ``QGraphicsTextItem`` as ``.item`` and an AxisItem's label *is* one.
    ``doc.size()`` lays the text out as painting does; ``QTextFragment.glyphRuns`` then names
    the face each run is shaped in -- the answer to "what font is this drawn in", whichever
    of CSS, ``setFont`` or the application font decided it.
    """
    item = getattr(label, "item", getattr(label, "textItem", label))
    doc = item.document()
    doc.size()
    out: set[str] = set()
    block = doc.firstBlock()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            out.update(run.rawFont().familyName() for run in it.fragment().glyphRuns())
            it += 1
        block = block.next()
    return out


def pdf_font_names(path) -> set[str]:
    """The PostScript names of the fonts a PDF embeds (its /BaseFont entries). A private
    system face has a dot-prefixed one (``.SFNS-Regular`` on macOS)."""
    import re

    return {name.decode() for name in re.findall(rb"/BaseFont\s*/([^\s/>\]]+)", path.read_bytes())}


def tick_family(axis) -> str:
    """The family the tick numbers are drawn in: the axis' tick QFont as Qt resolves it, or
    the application font when the axis has none (what an untouched axis used to draw in)."""
    font = axis.style.get("tickFont")
    return QtGui.QFontInfo(font if font is not None else QtWidgets.QApplication.font()).family()


def app_family() -> str:
    return QtGui.QFontInfo(QtWidgets.QApplication.font()).family()


def in_family(fams: set[str], family: str) -> bool:
    """``family`` is what the text is drawn in. A glyph the face lacks (a subscript digit, α…)
    is shaped by whichever face Qt falls back to, so other names may sit beside it -- but never
    the application font, which is what an unset family used to draw in (and, on macOS, a
    private face that no journal has)."""
    return family in fams and (family == app_family() or app_family() not in fams)


def pick(combo: QtWidgets.QComboBox, text: str) -> None:
    """Choose a row the way a person does: open the drop-down and click the row."""
    row = next(i for i in range(combo.count()) if combo.itemText(i) == text)
    combo.showPopup()
    QtWidgets.QApplication.processEvents()
    view = combo.view()
    rect = view.visualRect(view.model().index(row, 0))
    QtTest.QTest.mouseClick(view.viewport(), L, NOMOD, rect.center())
    QtWidgets.QApplication.processEvents()


def picking(monkeypatch, family: str | None):
    """Patch the Other font… dialog: ``family`` is picked (OK), or None cancels. Returns the
    list the patched ``exec`` records its calls in."""
    calls: list[str | None] = []

    def fake_exec(dialog):
        calls.append(dialog.family())
        if family is None:
            return Rejected
        assert dialog.select_family(family), f"{family} is not in the installed list"
        return Accepted

    monkeypatch.setattr(FontListDialog, "exec", fake_exec)
    return calls


def settle(qtbot, picker: FontPicker) -> None:
    """Other font… opens its dialog after the pick has finished; wait for it to be gone."""
    qtbot.waitUntil(lambda: not picker._asking, timeout=3000)
    QtWidgets.QApplication.processEvents()


# ================================================================================================
# the picker on its own
# ================================================================================================

def test_the_picker_lists_exactly_the_main_list_and_other_font(qapp, qtbot):
    picker = FontPicker()
    qtbot.addWidget(picker)
    assert [picker.itemText(i) for i in range(picker.count())] == MAIN_ROWS
    assert [picker.itemData(i) for i in range(4)] == ["", "Helvetica", "Times New Roman", "Courier New"]
    assert picker.itemText(0) == DEFAULT_ROW_LABEL == "Arial (default)"
    assert picker.value() == "" and not picker.has_extra_row()
    # Each row is drawn in the font it names (the first in Arial, the default's own).
    assert [picker.itemData(i, QtCore.Qt.ItemDataRole.FontRole).family() for i in range(4)] == list(
        ff.FIGURE_FAMILIES)
    # The separator is a separator, not a row that can be picked.
    assert picker.itemData(4, QtCore.Qt.ItemDataRole.AccessibleDescriptionRole) == "separator"


@pytest.mark.parametrize("value", [None, "", "Default", "Arial", "arial", "Times New Roman",
                                   "courier new", OTHER, "Calibri", "No Such Font 4711"])
@pytest.mark.parametrize("blocked", [False, True])
def test_set_value_says_nothing_and_leaves_the_block_it_found(qapp, qtbot, value, blocked):
    picker = FontPicker()
    qtbot.addWidget(picker)
    heard: list[tuple] = []
    picker.changed.connect(lambda v: heard.append(("changed", v)))
    picker.currentIndexChanged.connect(lambda i: heard.append(("index", i)))
    picker.currentTextChanged.connect(lambda t: heard.append(("text", t)))
    picker.activated.connect(lambda i: heard.append(("activated", i)))
    picker.blockSignals(blocked)
    picker.set_value(value)
    assert picker.signalsBlocked() is blocked, "the block the caller set is the block it finds"
    picker.blockSignals(False)
    qapp.processEvents()
    assert heard == [], f"set_value({value!r}) said {heard}"
    # ...and what it showed is what the picker now reports, as stored.
    assert picker.value() == ("" if value in (None, "", "Default") else value)


def test_a_listed_family_in_any_case_shows_its_row_and_keeps_its_spelling(qapp, qtbot):
    picker = FontPicker()
    qtbot.addWidget(picker)
    picker.set_value("times new roman")
    assert picker.currentText() == "Times New Roman" and not picker.has_extra_row()
    assert picker.value() == "times new roman", "showing a stored value never rewrites it"
    picker.set_value("Arial")           # an explicit Arial is the first row, and stays "Arial"
    assert picker.currentIndex() == 0 and picker.value() == "Arial"
    heard = []
    picker.changed.connect(heard.append)
    picker.setCurrentIndex(1)            # a pick is a new value, with the row's own spelling
    assert heard == ["Helvetica"] and picker.value() == "Helvetica"
    picker.setCurrentIndex(0)
    assert heard == ["Helvetica", ""], "picking the first row stores the unset value"


def test_a_family_the_list_lacks_gets_one_extra_row_and_never_more_than_one(qapp, qtbot):
    picker = FontPicker()
    qtbot.addWidget(picker)
    heard = []
    picker.changed.connect(heard.append)
    picker.set_value(OTHER)
    rows = [picker.itemText(i) for i in range(picker.count())]
    assert rows == MAIN_ROWS[:4] + [OTHER] + MAIN_ROWS[4:], rows
    assert picker.currentText() == OTHER and picker.currentData() == OTHER
    # A second off-list family REPLACES the row: one extra row, never two.
    picker.set_value("Calibri")
    assert picker.count() == len(MAIN_ROWS) + 1 and picker.currentData() == "Calibri"
    assert picker.value() == "Calibri", "the family is kept whether or not it is installed"
    if not font_picker.is_installed("Calibri"):
        assert picker.currentText() == "Calibri — not on this computer"
        # ...and a family this computer lacks does not outlive the pick that replaces it: its
        # row would advertise a font that cannot be drawn.
        picker.setCurrentIndex(2)
        assert heard == ["Times New Roman"]
        assert not picker.has_extra_row() and picker.count() == len(MAIN_ROWS)
        assert picker.findData("Calibri") == -1
    assert heard in ([], ["Times New Roman"])


def test_an_installed_familys_row_stays_so_it_can_be_picked_again(qapp, qtbot):
    picker = FontPicker()
    qtbot.addWidget(picker)
    heard = []
    picker.changed.connect(heard.append)
    picker.set_value(OTHER)
    picker.setCurrentIndex(1)                               # Helvetica, picked from the list
    assert heard == ["Helvetica"]
    assert picker.has_extra_row() and picker.findData(OTHER) == 4, "Menlo's row stayed"
    heard.clear()
    picker.setCurrentIndex(picker.findData(OTHER))          # ...and is picked again
    assert heard == [OTHER] and picker.value() == OTHER and picker.currentText() == OTHER
    # A silent set_value to a listed family (an Undo) leaves the row too.
    picker.set_value("Courier New")
    assert picker.has_extra_row() and picker.count() == len(MAIN_ROWS) + 1
    # The next off-list family takes the row over: still one extra row.
    second = next(f for f in font_picker.listable_families()
                  if f not in (OTHER, *ff.FIGURE_FAMILIES))
    picker.set_value(second)
    assert picker.count() == len(MAIN_ROWS) + 1 and picker.findData(OTHER) == -1
    assert picker.currentData() == second


def test_the_wheel_over_the_picker_never_walks_onto_other_font(qapp, qtbot, monkeypatch):
    """A wheel turned over the picker on its way through a scrolled panel must not step it onto
    Other font… and open a dialog. The application-wide wheel guard (``gui.app._WheelGuard``)
    refuses it for every combo, and this pins that the picker is one."""
    calls = picking(monkeypatch, OTHER)
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    layout = QtWidgets.QVBoxLayout(host)
    picker = FontPicker()
    layout.addWidget(picker)
    host.show()
    qtbot.waitExposed(host)
    picker.setCurrentIndex(3)
    picker.setFocus()
    heard = []
    picker.changed.connect(heard.append)
    for _ in range(3):
        QtWidgets.QApplication.sendEvent(picker, QtGui.QWheelEvent(
            QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
            QtCore.Qt.MouseButton.NoButton, NOMOD, QtCore.Qt.ScrollPhase.NoScrollPhase, False))
    qapp.processEvents()
    assert picker.currentIndex() == 3 and heard == [] and calls == []


def test_other_font_picks_through_the_dialog_and_cancel_keeps_the_previous_row(
        qapp, qtbot, monkeypatch):
    picker = FontPicker()
    qtbot.addWidget(picker)
    picker.show()
    picker.set_value("Helvetica")
    heard = []
    picker.changed.connect(heard.append)
    # Cancel: the dialog opened (on the previous pick), nothing was reported, the row is back.
    calls = picking(monkeypatch, None)
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert calls == [None] or calls == ["Helvetica"], calls
    assert picker.currentText() == "Helvetica" and picker.value() == "Helvetica" and heard == []
    assert not picker.has_extra_row()
    # OK on a family off the list: one report, one extra row, and the row is the pick.
    picking(monkeypatch, OTHER)
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert heard == [OTHER] and picker.value() == OTHER and picker.currentText() == OTHER
    assert picker.has_extra_row()
    # OK on a family that IS on the list picks that row (no extra row), and says so once.
    heard.clear()
    picking(monkeypatch, "Courier New")
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert heard == ["Courier New"] and picker.currentText() == "Courier New"
    assert picker.currentIndex() == 3, "a listed family is its own row, not the extra one"
    # OK on Arial is the first row: the unset value.
    heard.clear()
    picking(monkeypatch, "Arial")
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert heard == [""] and picker.currentIndex() == 0
    # Picking what is already there reports nothing.
    heard.clear()
    picking(monkeypatch, "Arial")
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert heard == []


def test_a_second_request_while_the_list_is_open_opens_no_second_list(qapp, qtbot, monkeypatch):
    """The request is queued, so it can arrive twice (a key held on the row); the list that
    is open answers for both."""
    picker = FontPicker()
    qtbot.addWidget(picker)
    opened: list[int] = []

    def fake_exec(dialog):
        opened.append(1)
        if len(opened) == 1:
            picker._choose_other()          # the second request, inside the first list's loop
        return Rejected

    monkeypatch.setattr(FontListDialog, "exec", fake_exec)
    picker.setCurrentIndex(picker.count() - 1)
    settle(qtbot, picker)
    assert opened == [1]
    assert picker.currentIndex() == 0 and picker.value() == ""


def test_other_font_never_opens_its_dialog_from_inside_the_combos_own_signal(
        qapp, qtbot, monkeypatch):
    """A modal event loop started from within the combo's ``currentIndexChanged`` would run
    while the combo is still mid-emission. The dialog is requested instead and opened from
    the event loop, so ``setCurrentIndex`` has returned by the time ``exec`` runs."""
    picker = FontPicker()
    qtbot.addWidget(picker)
    calls = picking(monkeypatch, None)
    picker.setCurrentIndex(picker.count() - 1)
    assert calls == [], "exec ran inside setCurrentIndex"
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    settle(qtbot, picker)
    assert picker.currentIndex() == 0, "Cancel put the first row back"


def test_other_font_opens_its_dialog_after_the_drop_down_has_closed(qapp, qtbot, monkeypatch):
    """A modal dialog opened while the combo's own list is still up misbehaves on macOS, so the
    dialog waits for the pick to finish: the list is gone when ``exec`` runs."""
    picker = FontPicker()
    qtbot.addWidget(picker)
    picker.show()
    qtbot.waitExposed(picker)
    seen: list[bool] = []

    def fake_exec(dialog):
        seen.append(picker.view().isVisible())
        return Rejected

    monkeypatch.setattr(FontListDialog, "exec", fake_exec)
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert seen == [False]


# ================================================================================================
# the Other font… list
# ================================================================================================

def test_the_dialog_lists_every_installed_family_but_no_private_one(qapp, qtbot):
    dialog = FontListDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    shown = dialog.families()
    installed = [f for f in QtGui.QFontDatabase.families()
                 if not f.startswith(".") and not QtGui.QFontDatabase.isPrivateFamily(f)
                 and not ff.is_symbol_family(f)]          # the faces of symbols are no text font
    assert shown == list(dict.fromkeys(installed))
    assert not any(f.startswith(".") for f in shown)
    assert {"Arial", "Helvetica", "Times New Roman", "Courier New", OTHER} <= set(shown)
    assert dialog.family() is None and not dialog._ok.isEnabled(), "nothing is picked yet"
    assert not dialog.testAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)


def test_the_dialog_filters_steps_with_the_arrow_keys_and_takes_the_pick(qapp, qtbot):
    dialog = FontListDialog(current="Courier New")
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.family() == "Courier New" and dialog._ok.isEnabled()
    QtTest.QTest.keyClicks(dialog._filter, "cour")
    shown = dialog.families()
    assert "Courier New" in shown and all("cour" in f.casefold() for f in shown)
    assert dialog.family() in shown, "the highlight moves to a match that is still shown"
    QtTest.QTest.keyClicks(dialog._filter, "zzzz")
    assert dialog.families() == [] and dialog.family() is None and not dialog._ok.isEnabled()
    dialog._filter.clear()
    dialog._filter.setText("hel")
    first = dialog.family()
    QtTest.QTest.keyClick(dialog._filter, QtCore.Qt.Key.Key_Down)
    if len(dialog.families()) > 1:
        assert dialog.family() != first
    QtTest.QTest.keyClick(dialog._filter, QtCore.Qt.Key.Key_Up)
    assert dialog.family() == first
    # The sample line is drawn in the highlighted family, Greek and Å in it.
    assert dialog._sample.font().family() == dialog.family()
    assert "τ" in dialog._sample.text() and "Å" in dialog._sample.text()
    # Return in the filter box is OK; Escape is Cancel.
    dialog._filter.setText("Courier New")
    QtTest.QTest.keyClick(dialog._filter, QtCore.Qt.Key.Key_Return)
    assert dialog.result() == int(Accepted) and dialog.family() == "Courier New"
    again = FontListDialog(current="Courier New")
    qtbot.addWidget(again)
    again.show()
    QtTest.QTest.keyClick(again, QtCore.Qt.Key.Key_Escape)
    assert again.result() == int(Rejected)


def test_a_double_click_on_a_row_picks_it(qapp, qtbot):
    dialog = FontListDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.select_family("Helvetica")
    row = dialog._list.currentItem()
    # The order the platform delivers a double click in: press, release, press, double, release.
    viewport, pos = dialog._list.viewport(), dialog._list.visualItemRect(row).center()
    QtTest.QTest.mousePress(viewport, L, NOMOD, pos)
    QtTest.QTest.mouseRelease(viewport, L, NOMOD, pos)
    QtTest.QTest.mousePress(viewport, L, NOMOD, pos)
    QtWidgets.QApplication.sendEvent(viewport, QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QPointF(pos),
        QtCore.QPointF(viewport.mapToGlobal(pos)), L, L, NOMOD))
    QtTest.QTest.mouseRelease(viewport, L, NOMOD, pos)
    assert dialog.result() == int(Accepted) and dialog.family() == "Helvetica"


# ================================================================================================
# the drawing policy: Arial when unset, a substitute when missing
# ================================================================================================

def test_the_stack_walks_the_installed_list_in_the_installed_spelling(monkeypatch):
    monkeypatch.setattr(font_picker, "installed_families",
                        lambda: ["Liberation Sans", "Tinos", "Cousine", "Menlo"])
    assert figure_family(None) == figure_family("") == "Liberation Sans"      # Arial missing
    assert figure_family("Arial") == "Liberation Sans"
    assert figure_family("Helvetica") == "Liberation Sans"
    assert figure_family("Times New Roman") == "Tinos"
    assert figure_family("Courier New") == "Cousine"
    assert figure_family("Calibri") == "Liberation Sans", "an unknown family falls back to Arial's stack"
    assert figure_family(OTHER) == OTHER, "an installed family is itself"
    monkeypatch.setattr(font_picker, "installed_families", lambda: ["Menlo"])
    assert figure_family("Times New Roman") == "Times New Roman", "nothing to fall back to: its own name"


def test_a_missing_family_is_drawn_through_its_substitute_not_the_system_font(monkeypatch):
    monkeypatch.setattr(font_picker, "installed_families", lambda: ["Liberation Serif", "Cousine"])
    serif = figure_qfont("Times New Roman")
    assert serif.family() == "Liberation Serif"
    mono = figure_qfont("Courier New")
    assert mono.family() == "Cousine"
    # Nothing in the stack installed: the family's own kind, not the application font.
    monkeypatch.setattr(font_picker, "installed_families", lambda: ["Menlo"])
    assert figure_qfont("Times New Roman").styleHint() == QtGui.QFont.StyleHint.Serif
    assert figure_qfont("Courier New").styleHint() == QtGui.QFont.StyleHint.Monospace
    assert figure_qfont(None).styleHint() == QtGui.QFont.StyleHint.SansSerif
    # The size and weight are the caller's.
    base = QtGui.QFont()
    base.setPointSizeF(13.0)
    base.setBold(True)
    out = figure_qfont("Courier New", base)
    assert out.pointSizeF() == 13.0 and out.bold()


def test_html_font_family_is_the_installed_part_of_the_stack_in_double_quotes(qapp, monkeypatch):
    """What pyqtgraph's label HTML gets: the stack's names this computer has, in order, with
    the names that contain a space in DOUBLE quotes (that HTML is written style='…'), and no
    generic keyword -- Qt reads one, like a name it cannot find, as the application font."""
    monkeypatch.setattr(font_picker, "installed_families", lambda: [
        "arial", "Helvetica", "Liberation Sans", "Times New Roman", "Tinos", "Cousine", "Menlo"])
    assert html_font_family(None) == html_font_family("") == 'Arial, Helvetica, "Liberation Sans"'
    assert html_font_family("Helvetica") == 'Helvetica, Arial, "Liberation Sans"'
    assert html_font_family("Times New Roman") == '"Times New Roman", Tinos'
    assert html_font_family("Courier New") == "Cousine"
    assert html_font_family(OTHER) == 'Menlo, Arial, Helvetica, "Liberation Sans"'
    assert html_font_family("No Such Font 4711") == 'Arial, Helvetica, "Liberation Sans"'
    for family in (None, "Times New Roman", "Courier New", OTHER):
        css = html_font_family(family)
        assert "'" not in css and not {"serif", "sans-serif", "monospace"} & {
            part.strip() for part in css.split(",")}, css
    # A name that cannot be quoted is not used; Arial's list stands in.
    assert html_font_family("A'B") == html_font_family(None)
    assert html_font_family('A";color:red') == html_font_family(None)
    # The foundation's full stack is still what SVG and matplotlib are given.
    assert ff.css_font_family("Times New Roman", quote='"').startswith('"Times New Roman", Times, ')


def test_html_font_family_reaches_the_glyphs(qapp, arial):
    """pyqtgraph writes label HTML as style='…': the list must survive that and draw."""
    import pyqtgraph as pg

    label = pg.LabelItem("")
    for family, expect in (("Times New Roman", "Times New Roman"), ("Courier New", "Courier New"),
                           (None, "Arial"), ("No Such Font 4711", "Arial")):
        label.setText("τ (ms)", family=html_font_family(family))
        assert drawn_families(label) == {expect}, (family, drawn_families(label))


#: A character of each kind the app writes into a label and a face may lack: the subscript
#: and superscript digits of ``I₀`` and ``10⁻³``, the Ångström sign, Greek, operators.
LABEL_CHARS = ("I\u2080\u2081\u2082\u2083\u2089\u2070\u00b9\u00b2\u00b3\u2074\u207a\u207b "
               "\u221a\u221e\u2248\u2265\u2264\u2260\u2192\u2190\u2194\u00d7\u00f7\u00b7 "
               "\u2022\u2026\u2013\u2014\u2030\u03a9\u2126\u212b\u00b5\u03bc\u03b1\u03b2 "
               "\u03b3\u03b4\u03b5\u03c3\u03c4\u03c6\u03c7\u03c8\u03c9\u0394\u03a3\u03a0 "
               "\u039b\u0393\u0398\u039e\u2211\u2202\u222b\u00b0\u2032\u2033\u00bd\u2212 "
               "\u00b1\u00c5")


@pytest.mark.parametrize("kind, family", [
    ("iv", None), ("iv", "Helvetica"), ("iv", "Times New Roman"), ("iv", "Courier New"),
    ("iv", OTHER), ("iv", "No Such Font 4711"),
    *[(kind, "Times New Roman") for kind in KINDS if kind != "iv"]])
def test_a_character_the_face_lacks_is_never_drawn_in_the_application_font(
        kind, family, qapp, qtbot, tmp_path, arial):
    """The subscript zero of I₀ is not in Times New Roman, nor the Ångström sign: Qt draws them
    in the next listed family that has them, else in the font it picks. With a name this
    computer lacks, or a generic keyword, in the list that pick was the application font,
    which on macOS is a private face the exported PDF then embedded (.SFNS-Regular)."""
    rig = make_rig(kind, qtbot, tmp_path)
    rig.page.set_appearance_state({"font_family": family or "", "title": LABEL_CHARS,
                                   "x_label": LABEL_CHARS, "y_label": LABEL_CHARS})
    QtWidgets.QApplication.processEvents()
    plot = rig.plots[0]
    want = family if family and font_picker.is_installed(family) else "Arial"
    for name, label in (("title", plot.titleLabel), ("y label", plot.getAxis("left").label),
                        ("x label", plot.getAxis("bottom").label)):
        fams = drawn_families(label)
        assert want in fams, (kind, family, name, fams)
        assert app_family() not in fams, f"{kind} {family}: the {name} has a glyph in {fams}"
    # ...and the PDF made from it embeds no private face.
    from nanoscope.gui.plot_export import write_live_plot_image

    target = tmp_path / "labels.pdf"
    write_live_plot_image(target, plot, width=800, height=600)
    names = pdf_font_names(target)
    assert names and not [n for n in names if n.startswith(".")], (kind, family, names)


@pytest.mark.parametrize("generic, default", [("sans-serif", "Zz Missing Sans"),
                                              ("serif", "Zz Missing Serif"),
                                              ("monospace", "Zz Missing Mono")])
def test_with_none_of_the_stack_installed_labels_and_tick_numbers_take_the_same_family(
        generic, default, qapp, monkeypatch):
    """A figure default and substitutes this computer really lacks: the labels are given the
    one family Qt picks for the figure's style hint -- a sans-serif, a serif, a fixed-width
    face -- which is the family the tick numbers are drawn in, and not the application font."""
    import pyqtgraph as pg

    monkeypatch.setattr(ff, "DEFAULT_FAMILY", default)
    monkeypatch.setitem(ff._STACKS, default.casefold(), ((default + " Sub",), generic))
    axis = pg.AxisItem("left")
    apply_tick_family(axis, None)
    tick = tick_family(axis)
    if tick == app_family():
        pytest.skip("the application font is what Qt picks here; nothing to tell apart")
    css = html_font_family(None)
    assert css.strip('"') == tick and "Zz" not in css, css
    label = pg.LabelItem("")
    label.setText("Aa τ (ms)", family=css)
    assert drawn_families(label) == {tick}


def test_the_foundation_quote_argument_changes_only_the_quotes():
    assert ff.css_font_family("Times New Roman") == (
        "'Times New Roman', Times, 'Liberation Serif', Tinos, 'TeX Gyre Termes', "
        "'Nimbus Roman', serif")
    assert ff.css_font_family("Times New Roman", quote='"').replace('"', "'") == (
        ff.css_font_family("Times New Roman"))
    assert ff.css_font_family("Arial", quote='"').startswith("Arial, Helvetica, ")


# ================================================================================================
# an untouched figure in each window type draws in Arial
# ================================================================================================

class Rig:
    """One plot window with synthetic data, and the plots whose text is checked."""

    def __init__(self, kind, page, plots):
        self.kind, self.page, self.plots = kind, page, plots

    @property
    def widget(self) -> PlotAppearanceWidget:
        return self.page._appearance_widget

    @property
    def picker(self) -> FontPicker:
        return self.widget.font_family

    def state(self) -> dict:
        return self.page.appearance_state()

    def depth(self) -> int:
        history = self.page._history if self.kind == "distributions" else self.widget._history
        return history.depth()[0]

    def undo(self) -> bool:
        return self.page.undo_config() if self.kind == "distributions" else self.widget.undo()

    def family_state(self):
        return self.state().get("font_family")

    def show_appearance(self) -> None:
        workflow = getattr(self.page, "_workflow", None)
        if workflow is not None:
            workflow.set_section("Appearance")
        QtWidgets.QApplication.processEvents()


def _subpop_page(qtbot):
    from nanoscope.analysis import subpop_report
    from nanoscope.gui.subpop_window import SubpopReportWindow

    from .test_lay_subpop_window_fits_the_screen import FS, N_SAMPLES, _synthetic_events

    arr, events = _synthetic_events()
    config = subpop_report.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    report = subpop_report.run_subpopulations(arr, arr, FS, events, recording="r",
                                              duration_s=N_SAMPLES / FS, config=config)
    page = SubpopReportWindow()
    qtbot.addWidget(page)
    _show(page, qtbot, size=(1400, 880))
    page.set_report(report, subtitle="synthetic")
    QtWidgets.QApplication.processEvents()
    return page


def make_rig(kind, qtbot, tmp_path) -> Rig:
    if kind == "distributions":
        page = PlotsWindow(Session())
        qtbot.addWidget(page)
        page.resize(1200, 800)
        page.show()
        qtbot.waitExposed(page)
        page.set_data({"Detected events": _events()})
        page.set_plot_type("Histogram")
        page.set_histogram_column("dwell_s")
        page.set_fit("exponential")
        QtWidgets.QApplication.processEvents()
        return Rig(kind, page, [page.plot_item()])
    if kind == "subpop":
        page = _subpop_page(qtbot)
        return Rig(kind, page, [page._embed_view._plot.getPlotItem(),
                                page._wave_view._plot.getPlotItem()])
    page, plot, _edit, _shown = _build(kind, qtbot, tmp_path)
    return Rig(kind, page, [plot])


def commit_inline(editor, text):
    """Type ``text`` into the inline editor and press Return. Greek goes in as an input-method
    commit, which is what a keyboard layout with τ sends on macOS: QTest's key clicks only
    know ASCII keys and abort on anything else."""
    editor.selectAll()
    event = QtGui.QInputMethodEvent()
    event.setCommitString(text)
    QtWidgets.QApplication.sendEvent(editor, event)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()


def add_title(plot, text):
    """Type a title on the figure, by the gestures a person makes: double-click one that
    draws, else right-click the plot and choose Add title…."""
    controller = figure_text_for(plot)
    figure_text.GATE.reset()
    if plot.titleLabel.isVisible():
        _click(plot, plot.titleLabel.sceneBoundingRect().center(), button=L, double=True)
        editor = controller.editor
        assert editor is not None, "a double-click opened no editor"
    else:
        controller.last_menu = None
        _click(plot, _background(plot))
        menu = controller.last_menu
        assert menu is not None, "a right-click on the plot raised no menu"
        entry = next((n for n in _names(menu) if n.startswith("Add title")), None)
        assert entry is not None, _names(menu)
        _choose(menu, entry)
        editor = controller.editor
        assert editor is not None and editor.text() == ""
    commit_inline(editor, text)
    assert plot.titleLabel.isVisible()


def text_of(rig: Rig, plot) -> dict[str, set[str] | str]:
    """Every piece of text this plot draws that has a font, as the font it is drawn in."""
    out: dict = {}
    if plot.titleLabel.isVisible():
        out["title"] = drawn_families(plot.titleLabel)
    for side in ("bottom", "left"):
        axis = plot.getAxis(side)
        if axis.label.isVisible():
            out[f"{side} label"] = drawn_families(axis.label)
        out[f"{side} ticks"] = tick_family(axis)
    legend = getattr(plot, "legend", None)
    if legend is not None and legend.isVisible():
        for i, (_sample, label) in enumerate(legend.items):
            out[f"legend row {i}"] = drawn_families(label)
    return out


def assert_drawn_in(rig: Rig, family: str, *, strict=()):
    """Every text of every plot is in ``family``; for the keys in ``strict`` it is that and
    nothing else (typed text with no glyph a face lacks)."""
    seen = 0
    for plot in rig.plots:
        for name, fams in text_of(rig, plot).items():
            seen += 1
            if isinstance(fams, str):
                assert fams == family, f"{rig.kind} {name}: ticks drawn in {fams}"
                continue
            assert in_family(fams, family), f"{rig.kind} {name}: drawn in {fams}, not {family}"
            if name in strict:
                assert fams == {family}, f"{rig.kind} {name}: drawn in {fams}"
    assert seen >= 4, "the figure drew too little text to prove anything"


@pytest.mark.parametrize("kind", KINDS)
def test_an_untouched_figure_draws_in_arial(kind, qapp, qtbot, tmp_path, arial):
    if app_family() == "Arial":
        pytest.skip("the application font is Arial here; nothing to tell apart")
    rig = make_rig(kind, qtbot, tmp_path)
    assert rig.family_state() is None or rig.family_state() == "", "premise: no font chosen"
    assert_drawn_in(rig, "Arial")
    for plot in rig.plots:
        for side in ("bottom", "left"):
            # The CSS an axis label's HTML carries is Arial's whole stack, so a glyph Arial
            # lacks falls to Helvetica and a viewer without Arial to the next substitute.
            assert plot.getAxis(side).labelStyle["font-family"] == html_font_family(None), (kind, side)
    # The Greek the user types is drawn in the figure font, not a symbol face.
    for plot in rig.plots[:1]:
        add_title(plot, "τ μ Δ (ms)")
        assert drawn_families(plot.titleLabel) == {"Arial"}
    assert rig.family_state() is None or rig.family_state() == "", "typing a title chose no font"
    assert_drawn_in(rig, "Arial", strict=("title",))


@pytest.mark.parametrize("kind", ["iv", "vdep", "noise", "distributions"])
def test_the_fit_label_and_caption_draw_in_arial_too(kind, qapp, qtbot, tmp_path, arial):
    rig = make_rig(kind, qtbot, tmp_path)
    page = rig.page
    if kind == "vdep":
        page.set_fit("exponential")
        QtWidgets.QApplication.processEvents()
    label = page._fit_label_item if kind == "distributions" else page._fit_label
    if not label.isVisible():
        pytest.skip("the synthetic data gives no fit label here")
    assert in_family(drawn_families(label), "Arial"), "the fit label is figure text like any other"
    if kind == "vdep":
        assert page._caption.isVisible() and in_family(drawn_families(page._caption), "Arial")
    # The figure's font reaches it: a family chosen for the figure is the label's font, and an
    # element override (Text style… on the label itself) wins over it.
    state = PlotAppearance.from_mapping({**rig.state(), "font_family": "Times New Roman"})
    page.set_appearance_state(state.to_dict())
    QtWidgets.QApplication.processEvents()
    assert in_family(drawn_families(label), "Times New Roman")
    if kind == "vdep":
        assert in_family(drawn_families(page._caption), "Times New Roman")
    own = state.with_text_style("annotation.fit", TextStyle(family="Courier New"))
    page.set_appearance_state(own.to_dict())
    QtWidgets.QApplication.processEvents()
    assert in_family(drawn_families(label), "Courier New")


def test_event_waveforms_colour_bar_and_the_noise_spectrogram_bar_draw_in_arial(
        qapp, qtbot, tmp_path, arial):
    rig = make_rig("events", qtbot, tmp_path)
    page = rig.page
    bar_axis = page._colorbar.axis
    assert tick_family(bar_axis) == "Arial"
    # The event color scale sits horizontally below the heatmap. Inspect the
    # active axis, including the glyphs that are actually drawn.
    label = bar_axis.label
    assert label.isVisible() and in_family(drawn_families(label), "Arial")
    marginal = page._density_top.getAxis("left").label
    assert in_family(drawn_families(marginal), "Arial")
    page.set_appearance_state({"font_family": "Times New Roman"})
    QtWidgets.QApplication.processEvents()
    assert tick_family(bar_axis) == "Times New Roman"
    assert in_family(drawn_families(label), "Times New Roman")
    # A marginal's count label follows the figure font too.
    assert in_family(drawn_families(marginal), "Times New Roman")
    # The ensemble view's figure, with its legend, draws in the figure font as well.
    page.set_view("Ensemble")
    QtWidgets.QApplication.processEvents()
    ensemble = Rig("events", page, [page._ensemble_plot])
    assert_drawn_in(ensemble, "Times New Roman")
    page.set_appearance_state({})
    QtWidgets.QApplication.processEvents()
    assert_drawn_in(ensemble, "Arial")

    noise = make_rig("noise", qtbot, tmp_path)
    page = noise.page
    page.set_view("spectrogram")
    QtWidgets.QApplication.processEvents()
    if page._spec_bar is None:
        pytest.skip("the synthetic recording gives no spectrogram here")
    axis = page._spec_bar.getAxis("right")
    assert tick_family(axis) == "Arial" and in_family(drawn_families(axis.label), "Arial")
    page.set_appearance_state({"font_family": "Times New Roman"})
    QtWidgets.QApplication.processEvents()
    assert tick_family(axis) == "Times New Roman"
    assert in_family(drawn_families(axis.label), "Times New Roman")


def test_noise_level_labels_draw_in_the_figure_font_and_the_pdf_embeds_no_private_face(
        qapp, qtbot, tmp_path, arial):
    """The all-points histogram marks each level with pyqtgraph's InfLineLabel, a plain text
    item that the pass over the axes and the title never reached: it was drawn in the
    application font, and the exported PDF embedded it (.SFNS-Regular on macOS)."""
    from nanoscope.gui.plot_export import write_live_plot_image

    rig = make_rig("noise", qtbot, tmp_path)
    page = rig.page
    page.set_view("allpoints")
    QtWidgets.QApplication.processEvents()
    labels = [line.label for line in page._hist_marks]
    assert labels and all(label.toPlainText() for label in labels), "premise: levels are marked"
    size = labels[0].textItem.font().pointSizeF()
    for label in labels:
        assert drawn_families(label) == {"Arial"}
    target = tmp_path / "histogram.pdf"
    write_live_plot_image(target, page._hist_plot, width=800, height=600)
    names = pdf_font_names(target)
    assert names and not [n for n in names if n.startswith(".")], names
    # The figure's font reaches them, at the size they had.
    page.set_appearance_state({"font_family": "Times New Roman"})
    QtWidgets.QApplication.processEvents()
    for label in labels:
        assert drawn_families(label) == {"Times New Roman"}
        assert label.textItem.font().pointSizeF() == size
    # A redraw makes new lines; their labels start from the figure's font, not the default.
    page._hist_bins_combo.setCurrentText("256")
    QtWidgets.QApplication.processEvents()
    fresh = [line.label for line in page._hist_marks]
    assert fresh and not set(map(id, fresh)) & set(map(id, labels)), "premise: the lines were redrawn"
    for label in fresh:
        assert drawn_families(label) == {"Times New Roman"}
    page.set_appearance_state({})
    QtWidgets.QApplication.processEvents()
    for label in [line.label for line in page._hist_marks]:
        assert drawn_families(label) == {"Arial"}


def text_items(page) -> list:
    """Every text item showing in a plot widget the page shows: titles, axis labels, legend
    rows, fit labels, level labels -- whatever is a ``QGraphicsTextItem`` in the scene. The
    grey placeholder a plot shows instead of data is not figure text and is left out."""
    import pyqtgraph as pg

    placeholders = {id(getattr(page, name).textItem) for name in ("_message_item", "_spec_msg")
                    if getattr(page, name, None) is not None}
    found = []
    for view in page.findChildren(pg.GraphicsView):
        scene = view.scene()
        if scene is None or not view.isVisibleTo(page):
            continue
        found.extend(item for item in scene.items()
                     if isinstance(item, QtWidgets.QGraphicsTextItem) and item.isVisible()
                     and item.toPlainText().strip() and id(item) not in placeholders)
    return found


@pytest.mark.parametrize("kind", KINDS)
def test_no_text_item_of_an_untouched_figure_is_left_in_the_application_font(
        kind, qapp, qtbot, tmp_path, arial):
    """The checks above name the texts they know; this one walks the scene, so a text item
    nobody thought of (the Noise level labels were one) is found too."""
    if app_family() == "Arial":
        pytest.skip("the application font is Arial here; nothing to tell apart")
    rig = make_rig(kind, qtbot, tmp_path)
    items = text_items(rig.page)
    assert len(items) >= 3, "the figure drew too little text to prove anything"
    leaks = [(item.toPlainText()[:40], drawn_families(item)) for item in items
             if app_family() in drawn_families(item)]
    assert not leaks, f"{kind}: still in the application font: {leaks}"


@pytest.mark.parametrize("kind, view", [("noise", "fft"), ("noise", "spectrogram"),
                                        ("noise", "allpoints"), ("events", "Ensemble")])
def test_no_text_item_of_the_other_views_is_left_in_the_application_font(
        kind, view, qapp, qtbot, tmp_path, arial):
    if app_family() == "Arial":
        pytest.skip("the application font is Arial here; nothing to tell apart")
    rig = make_rig(kind, qtbot, tmp_path)
    rig.page.set_view(view)
    QtWidgets.QApplication.processEvents()
    items = text_items(rig.page)
    assert len(items) >= 3, "the figure drew too little text to prove anything"
    leaks = [(item.toPlainText()[:40], drawn_families(item)) for item in items
             if app_family() in drawn_families(item)]
    assert not leaks, f"{kind} {view}: still in the application font: {leaks}"


def test_apply_text_style_draws_each_kind_of_item_in_the_figure_family(qapp, arial):
    """The helper a window uses for the labels outside its PlotItem: a colour bar's label (an
    axis), a LabelItem and a TextItem take the style's family, else the figure's, else Arial."""
    import pyqtgraph as pg

    axis, label, text = pg.AxisItem("left"), pg.LabelItem(""), pg.TextItem("")
    shapes = (axis.label, label, text)
    for family, own, expect in ((None, None, "Arial"), ("Times New Roman", None, "Times New Roman"),
                                ("Times New Roman", "Courier New", "Courier New")):
        for item in (axis, label, text):
            figure_text.apply_text_style(item, TextStyle(family=own), "τ (ms)", family)
        for shape in shapes:
            assert in_family(drawn_families(shape), expect), (family, own, drawn_families(shape))


def test_a_missing_family_is_drawn_in_arial_not_in_the_system_font(qapp, qtbot, tmp_path, arial):
    """A family this computer has not got -- a Calibri from a colleague's project -- falls to
    Arial's stack in the glyphs Qt shapes and in the tick font, in both draw paths."""
    assert not font_picker.is_installed("No Such Font 4711")
    for kind in ("iv", "distributions"):
        rig = make_rig(kind, qtbot, tmp_path)
        rig.page.set_appearance_state({"font_family": "No Such Font 4711"})
        QtWidgets.QApplication.processEvents()
        assert_drawn_in(rig, "Arial")
        add_title(rig.plots[0], "τ (ms)")       # the path a figure with typed text takes
        assert rig.family_state() == "No Such Font 4711", "the missing family is kept as chosen"
        assert_drawn_in(rig, "Arial", strict=("title",))


def test_a_text_style_family_this_computer_lacks_is_drawn_in_arial_too(
        qapp, qtbot, tmp_path, arial):
    """The element override (Text style…) takes the same fallback as the figure's font: on the
    title, an axis label, the tick numbers and the fit label, in both draw paths."""
    gone = "No Such Font 4711"
    for kind in ("iv", "distributions"):
        rig = make_rig(kind, qtbot, tmp_path)
        add_title(rig.plots[0], "τ (ms)")
        state = PlotAppearance.from_mapping(rig.state())
        for key in ("title", "axis.left.label", "axis.bottom.ticks", "annotation.fit"):
            state = state.with_text_style(key, replace(state.text_style(key), family=gone))
        rig.page.set_appearance_state(state.to_dict())
        QtWidgets.QApplication.processEvents()
        plot = rig.plots[0]
        assert in_family(drawn_families(plot.titleLabel), "Arial"), kind
        assert in_family(drawn_families(plot.getAxis("left").label), "Arial"), kind
        assert tick_family(plot.getAxis("bottom")) == "Arial", kind
        label = rig.page._fit_label_item if kind == "distributions" else rig.page._fit_label
        if label.isVisible():
            assert in_family(drawn_families(label), "Arial"), kind
        assert plot.getAxis("bottom").style["tickFont"].family() == "Arial"


def test_a_styled_legend_row_keeps_the_figure_font(qapp, qtbot, tmp_path, arial):
    """A legend row given its own style (bold here) is rebuilt by the shared funnel; without a
    family of its own it stays in the figure's, not in whatever the rebuild leaves."""
    rig = make_rig("iv", qtbot, tmp_path)
    plot = rig.plots[0]
    state = {"legend_visible": True,
             "text": {"legend.entry.Open-pore current": {"bold": True}}}
    rig.page.set_appearance_state(state)
    QtWidgets.QApplication.processEvents()
    rows = [label for _sample, label in plot.legend.items]
    styled = [row for row in rows if row.opts.get("bold")]
    assert len(rows) == 2 and len(styled) == 1, "one row carries the style"
    for row in rows:
        assert in_family(drawn_families(row), "Arial")
    rig.page.set_appearance_state({**state, "font_family": "Times New Roman"})
    QtWidgets.QApplication.processEvents()
    for row in [label for _sample, label in plot.legend.items]:
        assert in_family(drawn_families(row), "Times New Roman")


def test_the_legend_box_follows_its_rows_when_the_font_changes_their_width(
        qapp, qtbot, tmp_path, arial):
    """pyqtgraph sizes a legend when a row is added, from the rows' geometry; a row re-set in
    another family (Courier New is far wider than Arial) would leave the box at the old size
    until the next row came or went."""
    rig = make_rig("summaries", qtbot, tmp_path)
    legend = rig.plots[0].legend

    def box_vs_settled() -> tuple[float, float]:
        shown = legend.boundingRect().width()
        legend.layout.activate()
        legend.updateSize()
        return shown, legend.boundingRect().width()

    shown, settled = box_vs_settled()
    assert shown == pytest.approx(settled, abs=0.01), "premise: the untouched legend is settled"
    rig.page.set_appearance_state({**rig.state(), "font_family": "Courier New"})
    QtWidgets.QApplication.processEvents()
    shown, settled = box_vs_settled()
    assert shown == pytest.approx(settled, abs=0.01), "the box kept the Arial rows' width"


def test_the_distributions_caption_and_detection_limit_words_follow_the_figure_font(
        qapp, qtbot, tmp_path, arial):
    from dataclasses import replace as dc_replace

    rig = make_rig("distributions", qtbot, tmp_path)
    page = rig.page
    # Ticked after the draw, the caption is made by the restyle...
    assert page._prov_item is None
    page._prov_check.setChecked(True)
    QtWidgets.QApplication.processEvents()
    assert page._prov_item is not None and in_family(drawn_families(page._prov_item), "Arial")
    # ...and ticked before the draw, by the draw.
    page._prov_check.setChecked(False)
    page._style = dc_replace(page._style, show_provenance=True)
    page._redraw()
    QtWidgets.QApplication.processEvents()
    assert page._prov_item is not None and page._prov_item.isVisible()
    assert page._limit_line is not None and page._limit_line.label.isVisible()
    for item in (page._prov_item, page._limit_line.label):
        assert in_family(drawn_families(item), "Arial")
    page.set_font_family("Times New Roman")
    QtWidgets.QApplication.processEvents()
    for item in (page._prov_item, page._limit_line.label):
        assert in_family(drawn_families(item), "Times New Roman")


@pytest.mark.parametrize("kind", ["distributions", "iv"])
def test_the_svg_export_names_the_figure_font_on_every_text(kind, qapp, qtbot, tmp_path, arial):
    """What leaves the app follows the live items: every <text> of an untouched figure names
    Arial first (the label HTML's fallback stack starts with it), and a chosen font names it."""
    import html
    import re

    from nanoscope.plot.pg_svg import export_svg

    rig = make_rig(kind, qtbot, tmp_path)

    def first_families() -> set[str]:
        out = tmp_path / f"{kind}.svg"
        export_svg(rig.plots[0], str(out))
        found = [html.unescape(f) for f in re.findall(r'<text[^>]*?font-family="([^"]*)"',
                                                       out.read_text())]
        assert len(found) >= 6, "too little text exported to prove anything"
        return {f.split(",")[0].strip().strip("'\"") for f in found}

    assert first_families() == {"Arial"}
    rig.page.set_appearance_state({"font_family": "Times New Roman"})
    QtWidgets.QApplication.processEvents()
    assert first_families() == {"Times New Roman"}


def test_the_untouched_tick_font_is_the_substitute_when_arial_is_missing(
        qapp, qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(font_picker, "installed_families", lambda: ["Liberation Sans", "Menlo"])
    for kind in ("iv", "distributions"):
        rig = make_rig(kind, qtbot, tmp_path)
        for plot in rig.plots:
            for side in ("bottom", "left"):
                assert plot.getAxis(side).style["tickFont"].family() == "Liberation Sans", kind
            css = plot.getAxis("left").labelStyle["font-family"]
            assert css == html_font_family(None) and "Liberation Sans" in css


# ================================================================================================
# every picker lists the main list; Other font… and Cancel work through each of them
# ================================================================================================

def _text_style_popover(rig: Rig, text="τ title"):
    """Type a title, then right-click it ▸ Text style…."""
    add_title(rig.plots[0], text)
    return _open_text_style(rig)


def _open_text_style(rig: Rig):
    """Right-click the title ▸ Text style…, the way a person opens it."""
    plot = rig.plots[0]
    controller = figure_text_for(plot)
    controller.last_menu = None
    _click(plot, plot.titleLabel.sceneBoundingRect().center(), button=R)
    menu = controller.last_menu
    assert menu is not None and "Text style…" in _names(menu), menu and _names(menu)
    _choose(menu, "Text style…")
    popover = controller.popover
    assert isinstance(popover, figure_text.TextStylePopover)
    return controller, popover


def test_each_picker_lists_exactly_the_main_list(qapp, qtbot, tmp_path):
    iv = make_rig("iv", qtbot, tmp_path)
    dist = make_rig("distributions", qtbot, tmp_path)
    _controller, popover = _text_style_popover(iv)
    composer = FigureComposerDialog([_figure("a", "x"), _figure("b", "y")])
    qtbot.addWidget(composer)
    pickers = {
        "Appearance (I-V)": iv.picker,
        "Appearance (Distributions)": dist.picker,
        "Distributions Style group": dist.page._font_combo,
        "Text style…": popover.family,
        "composer panel labels": composer._label_font,
    }
    for name, picker in pickers.items():
        assert isinstance(picker, FontPicker), name
        rows = [picker.itemText(i) for i in range(picker.count())]
        expected = list(MAIN_ROWS)
        if name == "Text style…":
            expected[0] = "Arial"           # unset there is the Custom font box, so Arial is a family
        assert rows == expected, f"{name}: {rows}"
        assert picker.count() < 8, "no full family list is left in any picker"
    assert not [box for root in (iv.widget, dist.widget, dist.page, popover, composer)
                for box in root.findChildren(QtWidgets.QFontComboBox)], "a full family list is left"
    popover.close()


def test_each_pickers_tool_tip_says_what_other_font_leaves_out(
        qapp, qtbot, tmp_path, monkeypatch):
    """The list behind Other font… leaves out the system's private families and the faces of
    symbols (the guide and the inventory's notes say so), so no tool tip may promise "every font"
    without that exception. All five pickers say it, in the same words, and the inventory's
    quotation of the first is pinned to it by the test of the documents."""
    with monkeypatch.context() as patch:
        patch.setattr(font_picker, "installed_families",
                      lambda: [".Private UI Face", "Wingdings", "Zapf Dingbats", "Menlo", "Arial"])
        dialog = FontListDialog()
        qtbot.addWidget(dialog)
        assert dialog.families() == ["Menlo", "Arial"], "premise: what the tool tips describe"
    said = ("Other font… lists every font installed on this computer except the system's "
            "private fonts and symbol fonts such as Wingdings.")
    iv = make_rig("iv", qtbot, tmp_path)
    dist = make_rig("distributions", qtbot, tmp_path)
    _controller, popover = _text_style_popover(iv)
    composer = FigureComposerDialog([_figure("a", "x"), _figure("b", "y")])
    qtbot.addWidget(composer)
    for name, picker in {
        "Appearance (I-V)": iv.picker,
        "Appearance (Distributions)": dist.picker,
        "Distributions Style group": dist.page._font_combo,
        "Text style…": popover.family,
        "composer panel labels": composer._label_font,
    }.items():
        assert picker.toolTip().endswith(said), f"{name}: {picker.toolTip()!r}"
    popover.close()


def test_other_font_in_the_appearance_section_picks_cancels_and_is_one_undo_step(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    for kind in ("iv", "events", "distributions"):
        rig = make_rig(kind, qtbot, tmp_path)
        rig.show_appearance()
        picker, depth = rig.picker, rig.depth()
        # Cancel: nothing moves, no step.
        calls = picking(monkeypatch, None)
        pick(picker, "Other font…")
        settle(qtbot, picker)
        assert calls and rig.family_state() in (None, "") and rig.depth() == depth
        assert picker.currentIndex() == 0 and picker.value() == ""
        # OK: the state, the drawing, the row and one step.
        picking(monkeypatch, OTHER)
        pick(picker, "Other font…")
        settle(qtbot, picker)
        qtbot.wait(900)                               # past the coalescing pause
        assert rig.family_state() == OTHER, kind
        assert rig.depth() == depth + 1, f"{kind}: one pick is one Undo step"
        assert picker.currentText() == OTHER and picker.has_extra_row()
        assert_drawn_in(rig, OTHER)
        assert rig.undo()
        assert rig.family_state() in (None, "")
        assert picker.currentIndex() == 0, "Undo puts the first row back"
        assert picker.has_extra_row() and picker.findData(OTHER) == 4, (
            "...and the installed font's row stays, to be picked again")
        assert_drawn_in(rig, "Arial")


@pytest.mark.parametrize("kind", KINDS)
def test_choosing_a_listed_font_repaints_and_is_one_undo_step(kind, qapp, qtbot, tmp_path, arial):
    rig = make_rig(kind, qtbot, tmp_path)
    rig.show_appearance()
    depth = rig.depth()
    pick(rig.picker, "Times New Roman")
    qtbot.wait(900)
    assert rig.family_state() == "Times New Roman"
    assert rig.depth() == depth + 1, f"{kind}: one pick is one Undo step"
    assert_drawn_in(rig, "Times New Roman")
    assert rig.undo()
    assert rig.family_state() in (None, "")
    assert rig.picker.currentIndex() == 0, "Undo puts the first row back"
    assert_drawn_in(rig, "Arial")
    # Choosing the first row again is the unset value, not a stored "Arial".
    pick(rig.picker, "Courier New")
    pick(rig.picker, "Arial (default)")
    qtbot.wait(900)
    assert rig.family_state() in (None, "")
    assert_drawn_in(rig, "Arial")


def test_distributions_style_group_font_row_is_one_undo_step_too(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    rig = make_rig("distributions", qtbot, tmp_path)
    page, combo = rig.page, rig.page._font_combo
    depth = rig.depth()
    pick_index = combo.findData("Helvetica")
    combo.setCurrentIndex(pick_index)
    assert page._style.font_family == "Helvetica" and rig.depth() == depth + 1
    assert_drawn_in(rig, "Helvetica")
    picking(monkeypatch, OTHER)
    combo.setCurrentIndex(combo.count() - 1)
    settle(qtbot, combo)
    assert page._style.font_family == OTHER and rig.depth() == depth + 2
    assert page.config().style.font_family == OTHER
    assert page.undo_config() and page._style.font_family == "Helvetica"
    assert combo.currentText() == "Helvetica" and combo.findData(OTHER) == 4
    assert page.undo_config() and page._style.font_family == ""
    assert combo.currentText() == DEFAULT_ROW_LABEL


def test_text_style_family_is_the_same_list_and_picks_one_undo_step(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    rig = make_rig("iv", qtbot, tmp_path)
    controller, popover = _text_style_popover(rig)
    depth = rig.depth()
    assert not popover.use_family.isChecked()
    # Unticked, the list changes nothing: the text follows the figure's font.
    pick(popover.family, "Times New Roman")
    assert popover.value().family is None
    QtTest.QTest.mouseClick(popover.use_family, L)
    assert popover.value().family == "Times New Roman"
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()
    plot = rig.plots[0]
    assert rig.state()["text"]["title"]["family"] == "Times New Roman"
    assert rig.depth() == depth + 1
    assert drawn_families(plot.titleLabel) == {"Times New Roman"}
    assert "Arial" in drawn_families(plot.getAxis("left").label), "only the title took the font"
    assert rig.undo()
    assert "family" not in rig.state().get("text", {}).get("title", {})
    assert drawn_families(plot.titleLabel) == {"Arial"}

    # Ticked with the first row left alone: that text is Arial, whatever the figure's font is.
    rig.page.set_appearance_state({**rig.state(), "font_family": "Courier New"})
    QtWidgets.QApplication.processEvents()
    assert drawn_families(plot.titleLabel) == {"Courier New"}
    controller, popover = _text_style_popover(rig, text="τ again")
    QtTest.QTest.mouseClick(popover.use_family, L)
    assert popover.value().family == "Arial"
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()
    assert drawn_families(plot.titleLabel) == {"Arial"}


@pytest.mark.parametrize("stored", ["Times New Roman", "times new roman", "Arial", "arial",
                                    "Courier New", OTHER, "No Such Font 4711"])
def test_text_style_reopened_on_a_text_with_its_own_font_shows_it_and_ok_keeps_it(
        stored, qapp, qtbot, tmp_path, arial):
    """Text style… on a text that already has a font of its own opens with Custom font ticked
    and that font in the box, and OK without a touch stores the family exactly as it was --
    not the row's words ("No Such Font 4711 — not on this computer"), not a spelling the row
    changed ("times new roman" is the Times New Roman row), not Arial (the first row)."""
    rig = make_rig("iv", qtbot, tmp_path)
    plot = rig.plots[0]
    add_title(plot, "τ title")
    state = PlotAppearance.from_mapping(rig.state())
    rig.page.set_appearance_state(
        state.with_text_style("title", replace(state.text_style("title"), family=stored)).to_dict())
    QtWidgets.QApplication.processEvents()
    controller, popover = _open_text_style(rig)
    assert popover.use_family.isChecked()
    assert popover.family.value() == stored and popover.value().family == stored
    row = ff.canonical_family(stored) if ff.is_listed(stored) else (
        stored if font_picker.is_installed(stored) else f"{stored} — not on this computer")
    assert popover.family.currentText() == row
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()
    assert rig.state()["text"]["title"]["family"] == stored


def test_other_font_in_text_style_keeps_the_popover_and_cancel_keeps_the_font(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    """A modal window closes the popup it is opened from, so the picker hides the Text style
    box while its list is open and shows the same box again afterwards."""
    rig = make_rig("iv", qtbot, tmp_path)
    controller, popover = _text_style_popover(rig)
    assert popover.windowType() == QtCore.Qt.WindowType.Popup and popover.isVisible()
    QtTest.QTest.mouseClick(popover.use_family, L)
    seen: list[bool] = []

    def fake_exec(dialog):
        seen.append(popover.isVisible())
        assert dialog.select_family(OTHER)
        return Accepted

    monkeypatch.setattr(FontListDialog, "exec", fake_exec)
    pick(popover.family, "Other font…")
    settle(qtbot, popover.family)
    assert seen == [False], "the popover is out of the way while the list is open"
    assert popover.isVisible() and controller.popover is popover, "...and the same one is back"
    assert popover.family.value() == OTHER and popover.value().family == OTHER
    # Cancel keeps the previous pick.
    picking(monkeypatch, None)
    pick(popover.family, "Other font…")
    settle(qtbot, popover.family)
    assert popover.isVisible() and popover.family.value() == OTHER
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()
    assert rig.state()["text"]["title"]["family"] == OTHER
    assert drawn_families(rig.plots[0].titleLabel) == {OTHER}


def test_the_composer_label_font_is_the_same_list_and_a_pick_invalidates_the_preview(
        entries, qapp, qtbot, monkeypatch):  # noqa: F811
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.update_preview()
    assert dialog.snapshot is not None and dialog.label_family() == "Arial"
    picker = dialog._label_font
    pick(picker, "Times New Roman")
    assert dialog.snapshot is None and dialog.label_family() == "Times New Roman"
    dialog.update_preview()
    assert dialog.snapshot.config()["label_style"]["family"] == "Times New Roman"
    # Other font…: the pick is the label family; Cancel keeps it and the preview.
    picking(monkeypatch, OTHER)
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert dialog.label_family() == OTHER and dialog.snapshot is None
    dialog.update_preview()
    assert dialog.snapshot.config()["label_style"]["family"] == OTHER
    picking(monkeypatch, None)
    pick(picker, "Other font…")
    settle(qtbot, picker)
    assert dialog.label_family() == OTHER and dialog.snapshot is not None, (
        "Cancel changed nothing, so the preview still stands")
    # The first row is the unset value: label_family() is Arial.
    pick(picker, "Arial (default)")
    assert dialog.label_family() == "Arial"


def test_composition_labels_name_the_stack_in_the_svg_and_a_missing_family_still_composes(
        entries, qapp, qtbot):  # noqa: F811
    from xml.etree import ElementTree as ET

    from nanoscope.gui.figure_composition import DEFAULT_LABEL_STYLE

    for family in ("Arial", "No Such Family 4711"):
        dialog = FigureComposerDialog(
            entries, initial_layout={"columns": 2, "share_x": False, "share_y": False,
                                     "panel_labels": ["A", "B", "C"], "spacing": 20,
                                     "label_style": {**DEFAULT_LABEL_STYLE, "family": family}})
        qtbot.addWidget(dialog)
        assert dialog.label_family() == family
        dialog.update_preview()
        assert dialog.snapshot is not None, dialog._message.text()
        assert dialog.snapshot.config()["label_style"]["family"] == family
        svg = ET.fromstring(dialog.snapshot.svg_bytes)
        labels = [e for e in svg.iter() if (e.get("id") or "").endswith("_label")]
        assert len(labels) == 3
        assert all(e.get("font-family") == ff.css_font_family(family) for e in labels), (
            [e.get("font-family") for e in labels])


def test_composition_draws_its_labels_through_the_figure_font_resolver(
        entries, qapp, qtbot, monkeypatch):  # noqa: F811
    """The panel letters are laid out and painted in the resolved face (a family this computer
    lacks gets its substitute); the composition still stores the family as chosen."""
    from nanoscope.gui import figure_composition

    asked: list[str] = []
    real = figure_composition.figure_qfont

    def spy(family, *args, **kwargs):
        asked.append(family)
        return real(family, *args, **kwargs)

    monkeypatch.setattr(figure_composition, "figure_qfont", spy)
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    picking(monkeypatch, "Menlo")
    dialog.show()
    pick(dialog._label_font, "Other font…")
    settle(qtbot, dialog._label_font)
    dialog.update_preview()
    assert dialog.snapshot is not None, dialog._message.text()
    assert asked == ["Menlo"]
    assert dialog.snapshot.config()["label_style"]["family"] == "Menlo"


# ================================================================================================
# an old stored value comes back from a real save and open exactly as it was
# ================================================================================================

STORED = ["", "Times New Roman", OTHER, "Calibri"]


@pytest.mark.parametrize("stored", STORED)
def test_an_old_stored_family_round_trips_through_save_and_open(
        stored, qapp, qtbot, tmp_path, monkeypatch):
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    for seed, document in enumerate(window._session.documents()):
        _install_events(window, document.doc_id, seed=seed)
    window._open_iv()
    window._open_psd()
    window._open_plots()
    QtWidgets.QApplication.processEvents()
    iv, noise, dist = window._iv_window, window._psd_window, window._plots_window
    for page in (iv, noise):
        # The way an older build stored it: announced as the user's change, so the project
        # takes it (a silent set_appearance_state would leave the workspace unmarked).
        page._appearance_widget.apply_appearance(
            PlotAppearance.from_mapping({"font_family": stored} if stored else {}))
    dist.set_font_family(stored)
    QtWidgets.QApplication.processEvents()
    saved = {"iv": iv.appearance_state(), "noise": noise.appearance_state(),
             "dist": dist.config().style.font_family}
    assert saved["dist"] == stored
    assert (saved["iv"].get("font_family", "") == stored
            and saved["noise"].get("font_family", "") == stored)
    project = tmp_path / "fonts.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    reopened._open_iv()
    reopened._open_psd()
    reopened._open_plots()
    QtWidgets.QApplication.processEvents()
    iv2, noise2, dist2 = reopened._iv_window, reopened._psd_window, reopened._plots_window
    assert iv2.appearance_state().get("font_family", "") == stored
    assert noise2.appearance_state().get("font_family", "") == stored
    assert dist2.config().style.font_family == stored
    # (Distributions' embedded Appearance widget is not refreshed when a project opens -- it
    # shows blank for the title and sizes too, in the build before this one as well -- so the
    # row read here is the Style group's, which the window does sync.)
    for picker in (iv2._appearance_widget.font_family, noise2._appearance_widget.font_family,
                   dist2._font_combo):
        assert picker.value() == stored, "the picker shows the stored value, as stored"
        if stored in ("", "Times New Roman"):
            assert not picker.has_extra_row()
            assert picker.currentText() == (DEFAULT_ROW_LABEL if not stored else stored)
        else:
            assert picker.has_extra_row() and picker.currentData() == stored
            if stored == "Calibri" and not font_picker.is_installed("Calibri"):
                assert picker.currentText() == "Calibri — not on this computer"
    reopened._dirty = False
    window._dirty = False


def test_an_old_project_that_names_no_font_draws_in_arial_after_reopening(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    from nanoscope.core import project as project_format
    from nanoscope.core.analysis_workspace import IV_ARTIFACT_ID
    from nanoscope.core.figure_codecs import record_from_artifact, encode_legacy
    from nanoscope.core.figure_spec import FIGURE_ARTIFACT_KIND

    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    window._open_iv()
    QtWidgets.QApplication.processEvents()
    current = tmp_path / "current-fonts.nanoscope"
    window._iv_window._refresh_from_user()
    assert len(window._iv_window.points()) == 3, "the legacy fixture contains a plotted result"
    assert window._write_project(str(current))
    state = project_format.load_project(current)
    native = []
    for artifact in state.workspace.artifacts:
        if artifact.kind == FIGURE_ARTIFACT_KIND:
            record = record_from_artifact(artifact)
            artifact = encode_legacy(record.spec, record.resources)
        if artifact.artifact_id == IV_ARTIFACT_ID:
            # A v12 native IV artifact predates the explicit default style binding.
            # Its empty appearance has no chosen font; retain its scientific tables.
            artifact.payload['appearance'] = {}
        native.append(artifact)
    state.workspace.artifacts = native
    state.workspace.extra.pop('figure_workspace_v1', None)
    assert any(a.artifact_id == IV_ARTIFACT_ID and a.payload['appearance'] == {} for a in native)
    project = tmp_path / "nofont.nanoscope"
    with monkeypatch.context() as historical:
        historical.setattr(project_format, 'VERSION', 12)
        project_format.save_project(project, state)
    assert project_format.read_project_container_version(project) == 12
    window._dirty = False
    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    reopened._open_iv()
    QtWidgets.QApplication.processEvents()
    page = reopened._iv_window
    assert "font_family" not in page.appearance_state()
    plot = page.plot_item()
    for side in ("bottom", "left"):
        assert "Arial" in drawn_families(plot.getAxis(side).label)
        assert tick_family(plot.getAxis(side)) == "Arial"
    reopened._dirty = False
    window._dirty = False


# ================================================================================================
# the documents say what the picker has
# ================================================================================================

def test_the_user_guide_and_the_control_inventory_describe_the_font_list(qapp):
    """Read the app's own words off a real picker, then check the documents against them: the
    guide names every row, and the inventory quotes the tooltip of the Style group's font row
    byte for byte (the manual renders that column as a quotation)."""
    import json
    from pathlib import Path

    docs = Path(nanoscope.__file__).resolve().parents[1] / "docs"
    if not (docs / "USER_GUIDE.md").exists() or not (docs / "inventory.json").exists():
        pytest.skip("the documents are not beside this checkout")
    guide = (docs / "USER_GUIDE.md").read_text(encoding="utf-8")
    start = guide.index("### Figure fonts")
    section = guide[start:guide.index("\n### ", start + 10)]
    picker = FontPicker()
    rows = [picker.itemText(i) for i in range(picker.count()) if picker.itemText(i)]
    for row in rows:
        assert row in section.replace("\n", " "), f"the guide's Figure fonts never names {row!r}"
    inventory = json.loads((docs / "inventory.json").read_text(encoding="utf-8"))
    controls = [c for area in inventory for c in area["controls"]]
    style = [c for c in controls if c["label"] == "Font" and c.get("group") == "Style"]
    assert len(style) == 1
    assert style[0]["what_it_does"] == f'"{picker.toolTip()}"'
    for family in (*ff.FIGURE_FAMILIES, ff.OTHER_FONT_LABEL):
        assert family in style[0]["notes"], family
