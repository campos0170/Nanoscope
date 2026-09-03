"""chaos3-002, the two surfaces the first pass could not reach — the Analysis and
Calibration panel titles.

``tests/gui/test_s3_chaos3_label_text_format.py`` established the rule and pinned it on
the labels that pass owned: ``FileInfoPanel._title`` / ``["Path"]`` / ``["Format"]``,
``CalibrationChip._label`` and ``EventGalleryWindow.status``. The two labels the defect
report actually named live in files that pass did NOT own, and were left at
``Qt::AutoText``:

  * ``AnalysisPanel._title``     (``analysis_panel._ElidedLabel``)
  * ``CalibrationPanel._title``  (a plain ``QLabel``)

Both interpolate ``ChannelInfo.name`` — which ``io/text.py`` takes verbatim from a CSV
column header, and every binary reader takes verbatim from the file's channel table —
straight into a label Qt then parses as HTML. Measured in pixels through a wired window:
a channel really named ``current Ch <A>`` painted ``current Ch``, and ``I<sub>0</sub>``
painted ``I0``, in the same instant the File dock tree beside them showed the true name.
The Calibration title is the channel identification the user reads before pressing Apply
to write a gain/offset (and a sidecar file) for that channel.

These tests assert on what is PAINTED, not on ``label.text()`` — the helper below is a
copy of the one in the sweep-3 file, deliberately: a guard test written against the
source is the mistake ``tests/gui/test_readout_honesty.py`` documents, and it is why
``test_window_layout.py::test_the_analysis_title_elides_rather_than_widening_the_dock``
passed over this defect for the whole of its life.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0

#: Name forms Qt 6's ``mightBeRichText()`` accepts as markup. ``V<cmd>`` and ``I<pA>`` do
#: NOT trigger it (Qt validates the tag name), which is what made the defect look narrow
#: and made it worse: two channels differing only after the '<' can collapse onto one
#: painted string while a third renders verbatim.
SUBSCRIPT = "I<sub>0</sub>"      # painted "I0" — a real pClamp/LabVIEW export shape
SWALLOWING = "Ch <A>"            # painted "Ch " — everything from the '<' swallowed
BOLD = "<b>ch</b>"               # painted "ch", restyled


def painted(label: QtWidgets.QLabel) -> str:
    """The characters the label actually puts on screen.

    Not ``label.text()``: that is the SOURCE, and the source was never wrong.
    """
    text = label.text()
    fmt = label.textFormat()
    rich = (fmt == Qt.TextFormat.RichText
            or (fmt == Qt.TextFormat.AutoText and QtGui.Qt.mightBeRichText(text)))
    if not rich:
        return text
    doc = QtGui.QTextDocument()
    doc.setHtml(text)
    return doc.toPlainText()


def _doc(tmp_path, channel_name: str, units: str = "pA") -> TraceDocument:
    arr = np.full(4_000, 100.0, dtype=np.float32)
    info = ChannelInfo(index=0, name=channel_name, raw_units=units,
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "pore9.csv", format_name="text", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _select(window, tmp_path, channel_name: str, units: str = "pA"):
    """Install a recording and select its channel through the app's own selection path —
    the same call ``FileDock.selectionChanged`` lands on when the user clicks the row."""
    doc = _doc(tmp_path, channel_name, units)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    window._calibration_panel.set_selection(doc_id, 0)
    return doc_id


# ================================================================================================
# the two titles the defect named
# ================================================================================================

@pytest.mark.parametrize("name", [SUBSCRIPT, SWALLOWING, BOLD])
def test_the_analysis_title_paints_the_channel_name_in_the_file(window, tmp_path, name):
    _select(window, tmp_path, name)
    shown = painted(window._analysis_panel._title)
    assert name in shown, f"the name in the file is not on screen: {shown!r}"
    assert shown == window._analysis_panel._title.text(), (
        "the painted title and its source must be the same string")


@pytest.mark.parametrize("name", [SUBSCRIPT, SWALLOWING, BOLD])
def test_the_calibration_title_paints_the_channel_name_in_the_file(window, tmp_path, name):
    _select(window, tmp_path, name)
    shown = painted(window._calibration_panel._title)
    assert name in shown, f"the name in the file is not on screen: {shown!r}"
    assert shown == window._calibration_panel._title.text()


def test_two_channels_differing_only_after_the_bracket_stay_distinguishable(
        window, tmp_path):
    """The consequence, stated as itself: this title is what the user matches against
    their acquisition software, and what names the channel Apply is about to calibrate."""
    _select(window, tmp_path, "Ch <A>")
    a_cal = painted(window._calibration_panel._title)
    a_an = painted(window._analysis_panel._title)
    _select(window, tmp_path, "Ch <B>")
    assert painted(window._calibration_panel._title) != a_cal
    assert painted(window._analysis_panel._title) != a_an


def test_the_file_dock_tree_and_the_two_titles_agree(window, tmp_path):
    """The inconsistency that made the defect visible: the tree drew the true name at the
    same moment the two titles drew a shortened one."""
    _select(window, tmp_path, SWALLOWING)
    tree = window._file_dock._tree
    rows: list[str] = []
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        it = stack.pop()
        rows.append(it.text(0))
        stack += [it.child(i) for i in range(it.childCount())]
    assert SWALLOWING in rows, rows
    assert SWALLOWING in painted(window._calibration_panel._title)
    assert SWALLOWING in painted(window._analysis_panel._title)


# ================================================================================================
# the RULE, not the three instances of it
# ================================================================================================

def test_both_titles_declare_their_text_format(window, tmp_path):
    _select(window, tmp_path, "current")
    assert window._analysis_panel._title.textFormat() == Qt.TextFormat.PlainText
    assert window._calibration_panel._title.textFormat() == Qt.TextFormat.PlainText


def test_the_units_the_file_declared_are_painted_verbatim(window, tmp_path):
    """``raw_units`` rides in both titles and in the calibration panel's units row, and it
    is as much a header string as the name is."""
    _select(window, tmp_path, "current", units="p<A>")
    assert "p<A>" in painted(window._calibration_panel._title)
    assert "p<A>" in painted(window._analysis_panel._title)
    assert "p<A>" in painted(window._calibration_panel._units_row)


def test_every_calibration_read_out_that_carries_a_file_string_is_declared(window, tmp_path):
    """The audit, pinned. Each of these labels prints a string this app did not author —
    a unit off the header, a sidecar file name, the OS's own error text, a sidecar note —
    and a row added later inherits the discipline rather than the Qt default."""
    _select(window, tmp_path, "current")
    panel = window._calibration_panel
    plain = [panel._title, panel._units_row, panel._status, panel._cond_note,
             panel._cond_iopen, panel._cond_voltage, panel._cond_g,
             panel._recon_measured, panel._recon_gain, panel._recon_effective,
             panel._recon_nominal]
    for lbl in plain:
        assert lbl.textFormat() == Qt.TextFormat.PlainText, lbl.text()
    # The one label that IS markup declares itself markup, and escapes what it interpolates.
    assert panel._badge.textFormat() == Qt.TextFormat.RichText


def test_a_sidecar_note_containing_a_bracket_survives_onto_the_provenance_badge(
        window, tmp_path):
    """``cal.note`` is read out of the ``.npcal.json`` next to the user's data. The badge is
    rich text on purpose (a coloured span), so the note must be ESCAPED — the other half of
    the same rule."""
    from nanoscope.core.calibration import ChannelCalibration

    doc_id = _select(window, tmp_path, "current")
    panel = window._calibration_panel
    panel._set_badge(ChannelCalibration(gain=1.0, offset=0.0, target_units="pA",
                                        provenance=Provenance.FILENAME,
                                        note="10x head stage <I> assumed"))
    assert "10x head stage <I> assumed" in painted(panel._badge)
    assert doc_id  # the selection really happened


def test_the_sidecar_confirmations_do_not_parse_the_users_file_name(window, tmp_path,
                                                                   monkeypatch):
    """The delete/erase confirmations name the file they are about to touch and quote the
    notes they are about to destroy. A dialog that mis-names either is asking for consent
    to something other than what it says."""
    _select(window, tmp_path, "current")
    panel = window._calibration_panel
    seen: list[QtWidgets.QMessageBox] = []

    def fake_exec(self):
        seen.append(self)
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec, raising=False)
    panel._confirm_sidecar_removal(tmp_path / "pore <9>.npcal.json", action="Reset")
    panel._confirm_note_loss(tmp_path / "pore <9>.npcal.json",
                             ((0, "gain from head stage <I>"),), action="Apply")
    assert len(seen) == 2
    for box in seen:
        assert box.textFormat() == Qt.TextFormat.PlainText
    assert "pore <9>.npcal.json" in seen[0].text()
    assert "head stage <I>" in seen[1].informativeText()
