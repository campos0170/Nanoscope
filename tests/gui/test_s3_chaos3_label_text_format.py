"""chaos3-002 — a name out of the FILE is not markup, and a QLabel must be told so.

``QLabel``'s default text format is ``Qt::AutoText``: Qt runs ``mightBeRichText()`` over
whatever it is handed and, whenever a '<' is followed by a recognised tag name, PARSES IT AS
HTML. Channel names, unit strings, file names and format names all come verbatim out of the
file (or out of the path the user chose), so a recording called ``run<b>3.abf`` or a channel
called ``I<sub>0</sub>`` is painted under a name that is not its name — the '<' and what
follows it swallowed or restyled, silently, with the SOURCE string still intact so a test
written against ``label.text()`` sees nothing wrong.

The codebase already states the rule for itself: ``baseline_panel._check_html`` escapes every
interpolated value with the note that Qt "silently swallows a '<' and everything up to the
next '>' … there it deleted a whole criterion on its way to the screen".

These tests assert on what is PAINTED, not on the source, for every label in the owned
surfaces that interpolates a string this app did not author:

  * ``FileInfoPanel._title``            — the recording's own file name
  * ``FileInfoPanel._fields["Path"]``   — the full path the user chose
  * ``FileInfoPanel._fields["Format"]`` — the reader's format name
  * ``CalibrationChip._label``          — "✓ Gain set: …", ending in the channel's units,
                                          which are ``info.raw_units`` from the header
  * ``EventGalleryWindow.status``       — the trace name and the subpopulation report source

The channel table in ``FileInfoPanel`` is a ``QTableWidget`` and already paints plain, which
is the behaviour the File-dock tree has and the two panel titles did not; it is asserted here
so that it stays that way.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.file_info_panel import FileInfoPanel
from nanoscope.gui.status_chip import CalibrationChip
from nanoscope.gui.workflow_status import CalibrationState, CalibrationStatus

FS = 100_000.0

#: Name forms a real acquisition file can carry that Qt 6's mightBeRichText() accepts as
#: markup (verified in the reporter's names_probe: 'V<cmd>' and 'I<pA>' do NOT trigger it,
#: because Qt validates the tag name — these do).
HTML_LOOKING = "I<sub>0</sub>"
SWALLOWING = "Ch <A>"


def painted(label: QtWidgets.QLabel) -> str:
    """The characters the label actually puts on screen.

    Not ``label.text()``: that is the SOURCE, and a guard test written against the source is
    the precise mistake ``tests/gui/test_readout_honesty.py`` documents — it goes green over
    a defect a reader can see.
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


# ================================================================================================
# FileInfoPanel — "the FACTS the reader reported, verbatim"
# ================================================================================================

def _doc(path: Path, channel_name: str = "current", units: str = "pA",
         format_name: str = "synthetic") -> TraceDocument:
    arr = np.full(2_000, 100.0, dtype=np.float32)
    info = ChannelInfo(index=0, name=channel_name, raw_units=units,
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name=format_name, n_samples=arr.size, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def info_panel(qapp, qtbot, tmp_path):
    session = Session()
    panel = FileInfoPanel(session)
    qtbot.addWidget(panel)
    return session, panel, tmp_path


def test_a_file_name_that_looks_like_markup_is_painted_verbatim(info_panel):
    session, panel, tmp_path = info_panel
    path = tmp_path / f"{SWALLOWING} recording.csv"
    path.write_text("time,current\n0,1\n")
    doc_id = session.add_document(_doc(path, channel_name=HTML_LOOKING))
    panel.set_document(doc_id)

    assert painted(panel._title) == path.name
    assert painted(panel._fields["Path"]) == str(path)


def test_a_format_name_that_looks_like_markup_is_painted_verbatim(info_panel):
    session, panel, tmp_path = info_panel
    path = tmp_path / "rec.csv"
    path.write_text("time,current\n0,1\n")
    doc_id = session.add_document(_doc(path, format_name="text <b>delimited</b>"))
    panel.set_document(doc_id)

    assert painted(panel._fields["Format"]) == "text <b>delimited</b>"


def test_the_channel_table_still_shows_the_name_in_the_file(info_panel):
    """The tree in the File dock shows the true name; this table must agree with it. It is a
    QTableWidget, so it already paints plain — pinned so a future "richer" cell cannot
    silently reintroduce the split reading."""
    session, panel, tmp_path = info_panel
    path = tmp_path / "rec.csv"
    path.write_text("time,current\n0,1\n")
    doc_id = session.add_document(_doc(path, channel_name=HTML_LOOKING, units="p<A>"))
    panel.set_document(doc_id)

    assert panel._table.item(0, 0).text() == HTML_LOOKING
    assert panel._table.item(0, 1).text() == "p<A>"


def test_every_file_info_field_declares_its_text_format(info_panel):
    """The rule, not the three instances of it: a row added later inherits the discipline."""
    _session, panel, _tmp = info_panel
    assert panel._title.textFormat() == Qt.TextFormat.PlainText
    for key, label in panel._fields.items():
        assert label.textFormat() == Qt.TextFormat.PlainText, key


# ================================================================================================
# CalibrationChip — the one on-screen claim that magnitudes are physical
# ================================================================================================

def test_the_calibration_chip_paints_the_units_from_the_file_verbatim(qapp, qtbot):
    chip = CalibrationChip()
    qtbot.addWidget(chip)
    chip.set_status(CalibrationStatus(state=CalibrationState.CALIBRATED,
                                      label="Calibrated", detail="×2500 → I<sub>0</sub>"))
    assert painted(chip._label) == "✓ Gain set: ×2500 → I<sub>0</sub>"
    assert chip.text() == painted(chip._label), (
        "the introspection helper and the screen must not disagree")


def test_the_calibration_chip_declares_its_text_format(qapp, qtbot):
    chip = CalibrationChip()
    qtbot.addWidget(chip)
    assert chip._label.textFormat() == Qt.TextFormat.PlainText


# ================================================================================================
# EventGalleryWindow.status — the trace name and the report source
# ================================================================================================

def _events(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "start_s": np.linspace(0.0, 0.5, n),
        "end_s": np.linspace(0.001, 0.501, n),
        "dwell_s": np.full(n, 1e-3),
        "depth_pA": np.full(n, 25.0),
        "accepted": np.ones(n, dtype=bool),
        "flag": [""] * n,
    })


@pytest.fixture
def gallery(qapp, qtbot, tmp_path):
    session = Session()
    path = tmp_path / "pore9.csv"
    path.write_text("time,current\n0,1\n")
    doc_id = session.add_document(_doc(path))
    gal = EventGalleryWindow(session, None)
    qtbot.addWidget(gal)
    gal.set_events(doc_id, 0, _events(), "detect.schmitt", {})
    return gal, doc_id


def test_the_gallery_status_declares_its_text_format(gallery):
    gal, _doc_id = gallery
    assert gal.status.textFormat() == Qt.TextFormat.PlainText


def test_a_subpopulation_source_that_looks_like_markup_is_painted_verbatim(gallery):
    """``set_subpopulations(source=…)`` is documented as "what produced it (e.g. the
    recording name)" — a string out of the file system, printed on the status line."""
    gal, _doc_id = gallery
    gal.set_subpopulations({int(i): 0 for i in range(6)},
                           source=f"{SWALLOWING} report.json", caveats=())
    text = painted(gal.status)
    assert f"from: {SWALLOWING} report.json" in text, text
