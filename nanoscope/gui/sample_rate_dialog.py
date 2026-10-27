"""Declare a recording's time base — the unit of its time column, or its sample rate.

The sibling of :mod:`nanoscope.gui.acquisition_dialog`, with one difference it must say out
loud: this is NOT metadata only. The declaration moves every seconds-based number the app
shows for the recording, and stored detections are marked stale against it rather than
re-scaled. The file's own reading stays on screen for as long as the dialog is open, and
"Use the file's own time base" is always the first choice.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.sample_rate import TIME_UNITS, SampleRateDeclaration
from ..core.segment import SegmentDocument
from ..core.types import Provenance
from .numeric_text import parse_strict_number

_SCALE_WORD = {scale: name for name, scale in TIME_UNITS.values()}


class SampleRateDialog(QtWidgets.QDialog):
    def __init__(self, session, doc_id: str, parent=None):
        super().__init__(parent)
        self._session, self._doc_id = session, doc_id
        self.setWindowTitle("Sample rate / time unit — recording time base")
        self.resize(660, 480)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Declare this recording's time base when the file does not say, or says it wrongly: "
            "the unit its time column is written in (a CSV in milliseconds reads 1,000x slow), "
            "or the sample rate it was acquired at (a legacy TDMS declares none). Every "
            "seconds-based number — the x-axis, durations, dwell times, spectra — follows the "
            "declaration. Save the project to keep it; the raw file is never modified.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(QtCore.Qt.PlainText)
        self.header.setWordWrap(True)
        form.addRow("Time base as read", self.header)
        self.mode = QtWidgets.QComboBox()
        for label, value in (("Use the file's own time base", "file"),
                             ("The time column is in…", "time_unit"),
                             ("Declare the sample rate (Hz)", "sample_rate")):
            self.mode.addItem(label, value)
        form.addRow("Declaration", self.mode)
        self.unit = QtWidgets.QComboBox()
        for key, (name, _scale) in TIME_UNITS.items():
            self.unit.addItem(f"{name} ({key})", key)
        form.addRow("Time column unit", self.unit)
        self.rate = QtWidgets.QLineEdit()
        self.rate.setPlaceholderText("e.g. 100000")
        form.addRow("Sample rate (Hz)", self.rate)
        self.note = QtWidgets.QPlainTextEdit()
        self.note.setMaximumHeight(70)
        self.note.setPlaceholderText(
            "Where you know this from — the exporter's units, the amplifier's settings record")
        form.addRow("Provenance note", self.note)
        layout.addLayout(form)
        self.preview = QtWidgets.QLabel()
        self.preview.setTextFormat(QtCore.Qt.PlainText)
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        warning = QtWidgets.QLabel(
            "Detections already made on this recording keep the rate they were measured at "
            "and are marked stale, never silently re-scaled: re-run them under the declared "
            "rate. Segments cut from this recording follow it.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.error = QtWidgets.QLabel()
        self.error.setTextFormat(QtCore.Qt.PlainText)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.mode.currentIndexChanged.connect(self._enabled)
        self.unit.currentIndexChanged.connect(self._preview)
        self.rate.textChanged.connect(self._preview)
        self._load()

    def _load(self) -> None:
        read = self._session.time_base_as_read(self._doc_id)
        doc = self._session.document(self._doc_id)
        self._segment = isinstance(doc, SegmentDocument)
        noun = "/time" if "HDF5" in doc.meta.format_name else "the time column"
        rate_prov = (f"{read.sample_rate_hz:,.6g} Hz · {read.rate_provenance.value} · "
                     f"t₀ = {read.t0_s:.6g} s") if read.sample_rate_hz is not None else ""
        if read.sample_rate_hz is None and read.refused_rate_hz is not None:
            # The floor refused the time base as a clock (io.base.MIN_PLAUSIBLE_RATE_HZ); the
            # interval it measured is kept so the unit — the one fact missing — can be
            # declared here, which is the remedy the Issues box names.
            self.header.setText(
                f"refused — {noun} steps by {1.0 / read.refused_rate_hz:,.6g} per sample as "
                f"written, which cannot be seconds ({read.refused_rate_hz:,.6g} Hz as read); "
                f"declare the unit it is written in")
        elif read.sample_rate_hz is None:
            self.header.setText("unknown — the file declares no usable time base; "
                                "the x-axis is in samples")
        elif read.rate_provenance is Provenance.USER:
            # Only a segment reads 'user' AS READ: it was cut from a declared parent.
            self.header.setText(f"{rate_prov} (the declaration made on the recording this "
                                f"was cut from)")
        elif read.rate_provenance is Provenance.MEASURED:
            word = _SCALE_WORD.get(read.time_scale_s)
            how = (f"{noun} read as {word}" if word
                   else f"{noun} scaled by {read.time_scale_s:g} s per unit")
            self.header.setText(f"{rate_prov} ({how})")
        else:
            self.header.setText(f"{rate_prov} (declared in the file's header; there is no "
                                f"time column to read in another unit)")
        if self._segment:
            # A window of its parent: the clock is the parent's, declared there and followed
            # here (Session.set_sample_rate_declaration), so nothing on this dialog applies.
            self.header.setText(self.header.text() + " — a segment, on its parent's clock")
            for widget in (self.mode, self.unit, self.rate, self.note):
                widget.setEnabled(False)
            self.preview.setText("")
            self.error.setText("This is a segment: it follows the time base of the recording "
                               "it was cut from. Declare the time base on that recording.")
            return
        declaration = self._session.sample_rate_declaration(self._doc_id)
        self.mode.setCurrentIndex(self.mode.findData(declaration.mode if declaration else "file"))
        unit = (declaration.time_unit if declaration and declaration.time_unit
                else next((k for k, (_n, s) in TIME_UNITS.items() if s == read.time_scale_s), "s"))
        self.unit.setCurrentIndex(self.unit.findData(unit))
        self.rate.setText(format(declaration.sample_rate_hz, ".17g")
                          if declaration and declaration.sample_rate_hz else "")
        self.note.setPlainText(declaration.note if declaration else "")
        # A unit can only be declared for a column that was read: a file with no usable
        # time base has nothing to re-read, so the choice is shown but cannot be taken.
        item = self.mode.model().item(self.mode.findData("time_unit"))
        if isinstance(item, QtGui.QStandardItem):
            item.setEnabled(read.can_declare_unit)
        self.error.clear()
        self._enabled()

    def _enabled(self) -> None:
        mode = self.mode.currentData()
        self.unit.setEnabled(mode == "time_unit")
        self.rate.setEnabled(mode == "sample_rate")
        self.note.setEnabled(mode != "file")
        self._preview()

    def _declaration(self) -> SampleRateDeclaration | None:
        mode = self.mode.currentData()
        if mode == "file":
            return None
        note = self.note.toPlainText().strip()
        if mode == "time_unit":
            return SampleRateDeclaration("time_unit", time_unit=self.unit.currentData(), note=note)
        # The shared strict ASCII parser (see acquisition_dialog.py for why).
        hz = parse_strict_number(self.rate.text())
        if hz is None or not hz > 0:
            raise ValueError("the sample rate must be a finite positive number in Hz")
        return SampleRateDeclaration("sample_rate", sample_rate_hz=hz, note=note)

    def _preview(self) -> None:
        try:
            value = self._declaration()
            if value is None:
                self.preview.setText("The file's own time base will be used.")
                return
            rate, t0, _scale = value.resolve(self._session.time_base_as_read(self._doc_id))
            self.preview.setText(
                f"Would read as {rate:,.6g} Hz, t₀ = {t0:.6g} s — {value.describe()}.")
        except (KeyError, TypeError, ValueError) as exc:
            self.preview.setText(f"Cannot apply: {exc}")

    def apply(self) -> None:
        if self._segment:
            self.error.setText("Not applied: a segment follows the time base of the recording "
                               "it was cut from; declare it on that recording.")
            return
        try:
            value = self._declaration()
            self._session.set_sample_rate_declaration(self._doc_id, value)
        except (KeyError, TypeError, ValueError) as exc:
            self.error.setText(f"Not applied: {exc}")
            return
        if value is None:
            self.error.setText("The file's own time base is in use again. Save the project to "
                               "retain this.")
            return
        meta = self._session.document(self._doc_id).meta
        self.error.setText(f"Declared: {meta.sample_rate_hz:,.6g} Hz. Save the project to retain "
                           f"it; detections already made on this recording are marked stale.")
