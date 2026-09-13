"""Explicit, metadata-only acquisition declaration for specialists."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core.acquisition import AcquisitionBandwidth, describe_bandwidth
from .numeric_text import parse_strict_number


class AcquisitionBandwidthDialog(QtWidgets.QDialog):
    def __init__(self, session, doc_id: str, parent=None):
        super().__init__(parent)
        self._session, self._doc_id = session, doc_id
        self.setWindowTitle("Acquisition bandwidth — recording metadata")
        self.resize(630, 510)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Declare the acquisition low-pass setting for this recording and channel. "
            "This does not apply a filter, rerun analysis, or change event acceptance. "
            "Save the project to keep overrides; raw files are never modified.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.channel = QtWidgets.QComboBox()
        for info in session.document(doc_id).meta.channels:
            self.channel.addItem(f"{info.name} (channel {info.index})", info.index)
        form.addRow("Channel", self.channel)
        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(QtCore.Qt.PlainText)
        self.header.setWordWrap(True)
        form.addRow("Original file declaration", self.header)
        self.mode = QtWidgets.QComboBox()
        for label, value in (("Use original file declaration", "file"),
                             ("User-provided cutoff", "reported"),
                             ("Unknown / not verified", "unknown"),
                             ("Reported stage bypassed", "disabled")):
            self.mode.addItem(label, value)
        form.addRow("Metadata source", self.mode)
        self.cutoff = QtWidgets.QLineEdit()
        self.cutoff.setPlaceholderText("e.g. 10000")
        form.addRow("Acquisition cutoff (Hz)", self.cutoff)
        self.filter_type = QtWidgets.QLineEdit()
        self.filter_type.setPlaceholderText("e.g. Bessel; leave blank if unknown")
        form.addRow("Filter type", self.filter_type)
        self.order = QtWidgets.QSpinBox()
        self.order.setRange(0, 100)
        self.order.setSpecialValueText("Unknown")
        form.addRow("Filter order", self.order)
        self.note = QtWidgets.QPlainTextEdit()
        self.note.setMaximumHeight(90)
        self.note.setPlaceholderText("Amplifier/settings record, verification source, or uncertainty")
        form.addRow("Provenance note", self.note)
        layout.addLayout(form)
        warning = QtWidgets.QLabel(
            "A cutoff alone does not determine the full acquisition response or rise time. "
            "Bypassing one reported stage does not establish that all hardware was unfiltered. "
            "Existing results retain the acquisition metadata recorded at detection time.")
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
        self.channel.currentIndexChanged.connect(self._load)
        self._load()

    def _load(self):
        ch = self.channel.currentData()
        original = self._session.document(self._doc_id).meta.channels[ch].acquisition_bandwidth
        self.header.setText(describe_bandwidth(original))
        self.header.setToolTip(original.note + "\n" + "\n".join(f"{k}: {v}" for k, v in original.header_fields))
        override = self._session.acquisition_bandwidth_override(self._doc_id, ch)
        value = override or original
        self.mode.setCurrentIndex(self.mode.findData(override.status if override else "file"))
        self.cutoff.setText(format(value.cutoff_hz, ".17g") if value.cutoff_hz is not None else "")
        self.filter_type.setText(value.filter_type)
        self.order.setMaximum(max(100, value.filter_order or 0))
        self.order.setValue(value.filter_order or 0)
        self.note.setPlainText(value.note)
        self.error.clear()
        self._enabled()

    def _enabled(self):
        mode = self.mode.currentData()
        for widget in (self.cutoff, self.filter_type, self.order):
            widget.setEnabled(mode == "reported")
        self.note.setEnabled(mode != "file")

    def apply(self):
        try:
            mode = self.mode.currentData()
            cutoff_hz = None
            if mode == "reported":
                # Shared strict ASCII parser (beta-sweep B27, 2026-09-12): the previous
                # ad hoc "_" check caught PEP 515 grouping but not a non-ASCII digit
                # script (e.g. Arabic-Indic), which bare float() also accepts.
                cutoff_hz = parse_strict_number(self.cutoff.text())
                if cutoff_hz is None or not cutoff_hz > 0:
                    raise ValueError(
                        "acquisition cutoff must be a finite positive number in Hz")
            value = None if mode == "file" else AcquisitionBandwidth(
                cutoff_hz=cutoff_hz,
                status=mode, source="user",
                filter_type=self.filter_type.text().strip() if mode == "reported" else "",
                filter_order=(self.order.value() or None) if mode == "reported" else None,
                note=self.note.toPlainText().strip())
            self._session.set_acquisition_bandwidth(self._doc_id, self.channel.currentData(), value)
        except (KeyError, TypeError, ValueError) as exc:
            self.error.setText(f"Not applied: {exc}")
            return
        self.error.setText("Metadata applied. Save the project to retain it. Existing results are unchanged.")
