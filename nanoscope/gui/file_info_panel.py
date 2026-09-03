from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.session import Session
from ..core.types import ChannelInfo, Provenance, RecordingMeta

__all__ = ["FileInfoPanel"]


def _prov(p: Provenance) -> str:
    return p.value


class FileInfoPanel(QtWidgets.QWidget):
    """Read-only view of a document's RecordingMeta: the FACTS the reader reported, verbatim.
    Unknown sample rate is shown plainly as "unknown" — NEVER a fabricated 50 kHz. Every notes/
    issues entry is surfaced here rather than in a modal.

    Two cells are LIVE rather than as-loaded, and both keep the file's own word on screen:
    the Issues box merges what the current calibration did to each channel, and the
    Channels table's Kind column reads what the channel is TREATED AS (``Session.
    effective_kind``) with the header's word in parentheses when the user's override
    differs from it -- ``current (file says unknown)``. See :meth:`_kind_label` for why.
    """

    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._doc_id: str | None = None

        self._build_ui()

        session.documentOpened.connect(self._on_document_opened)
        session.documentClosed.connect(self._on_document_closed)
        session.calibrationChanged.connect(lambda d, _c: self._refresh_if(d))
        self._show_empty()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._title = QtWidgets.QLabel("No document")
        # PLAIN TEXT, EXPLICITLY -- for the reason spelled out at the field labels below.
        # The title is the recording's own file NAME, straight off the path.
        self._title.setTextFormat(QtCore.Qt.PlainText)
        self._title.setStyleSheet("font-weight:600; font-size:13px;")
        self._title.setWordWrap(True)
        root.addWidget(self._title)

        self._form = QtWidgets.QFormLayout()
        self._form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self._fields: dict[str, QtWidgets.QLabel] = {}
        for key in ("Path", "Format", "Samples", "Duration", "Sample rate", "t₀",
                    "Loaded range", "Nominal voltage"):
            lbl = QtWidgets.QLabel("—")
            # PLAIN TEXT, EXPLICITLY. THIS PANEL'S WHOLE JOB is "the FACTS the reader
            # reported, verbatim", and half of these rows carry a string this app did not
            # author -- the path the user chose, the format name the reader supplied. A
            # QLabel left at Qt::AutoText runs mightBeRichText() over what it is handed and
            # parses it as HTML, so a recording saved as `run<b>3.abf` would be shown under
            # a name that is not its name, with the '<b>' and everything after it swallowed
            # or restyled -- silently, and in the one panel whose contract is verbatim.
            # Declared once here rather than escaped at each of the seven setText calls,
            # which is the version that cannot go stale when an eighth row is added.
            lbl.setTextFormat(QtCore.Qt.PlainText)
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self._fields[key] = lbl
            self._form.addRow(key, lbl)
        root.addLayout(self._form)

        root.addWidget(self._section("Channels"))
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "Units", "Kind", "State"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._table.setMaximumHeight(160)
        root.addWidget(self._table)

        self._notes_head = self._section("Notes")
        root.addWidget(self._notes_head)
        self._notes = QtWidgets.QPlainTextEdit()
        self._notes.setReadOnly(True)
        self._notes.setMaximumHeight(90)
        root.addWidget(self._notes)

        self._issues_head = self._section("Issues")
        root.addWidget(self._issues_head)
        self._issues = QtWidgets.QPlainTextEdit()
        self._issues.setReadOnly(True)
        self._issues.setMaximumHeight(90)
        root.addWidget(self._issues)

        root.addStretch(1)

    @staticmethod
    def _section(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color:#8b9199; font-weight:600; margin-top:4px;")
        return lbl

    # -- external API -----------------------------------------------------------------------------
    def set_document(self, doc_id: str) -> None:
        if not self._session.has_document(doc_id):
            self._show_empty()
            return
        self._doc_id = doc_id
        self._render(self._session.document(doc_id).meta)

    # -- rendering --------------------------------------------------------------------------------
    def _render(self, meta: RecordingMeta) -> None:
        self._title.setText(meta.path.name)
        self._fields["Path"].setText(str(meta.path))
        self._fields["Format"].setText(meta.format_name)
        self._fields["Samples"].setText(f"{meta.n_samples:,}")

        if meta.duration_s is None:
            self._fields["Duration"].setText("unknown (no sample rate)")
        else:
            self._fields["Duration"].setText(f"{meta.duration_s:.6g} s")

        if meta.sample_rate_hz is None:
            self._fields["Sample rate"].setText("unknown  ·  x-axis is in samples")
        else:
            self._fields["Sample rate"].setText(
                f"{meta.sample_rate_hz:,.6g} Hz  ·  {_prov(meta.rate_provenance)}")

        self._fields["t₀"].setText(f"{meta.t0_s:.6g} s")

        if meta.loaded_range is None:
            self._fields["Loaded range"].setText("whole file")
        else:
            self._fields["Loaded range"].setText(
                f"samples {meta.loaded_range.start:,}–{meta.loaded_range.stop:,}")

        if meta.nominal_voltage_mV is None:
            self._fields["Nominal voltage"].setText("—")
        else:
            self._fields["Nominal voltage"].setText(
                f"{meta.nominal_voltage_mV:.6g} mV  ·  {_prov(meta.voltage_provenance)}")
        # Filename-derived voltage is unreliable — flag it.
        self._fields["Nominal voltage"].setStyleSheet(
            "color:#FFB74D;" if meta.voltage_provenance is Provenance.FILENAME else "")

        self._table.setRowCount(len(meta.channels))
        for r, info in enumerate(meta.channels):
            state = "live"
            if info.is_dead:
                state = f"dead — {info.dead_reason}" if info.dead_reason else "dead"
            kind_text, kind_tip = self._kind_label(info)
            cells = [info.name, info.raw_units or "—", kind_text, state]
            for c, val in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(val)
                if info.is_dead:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#7a828b")))
                if c == 2:
                    item.setToolTip(kind_tip)
                if c == 3 and info.is_dead:
                    item.setToolTip(info.dead_reason or "")
                self._table.setItem(r, c, item)

        self._notes.setPlainText("\n".join(meta.notes) if meta.notes else "(none)")
        # meta.issues describes the recording AS LOADED. Everything that reads a channel
        # reads the CALIBRATED array, and a calibration can ruin one — a gain that overflows
        # float32 leaves 2,420 non-finite samples in the owner's real ABF with meta.issues
        # empty. Two panels used to send the user here for exactly that case and this box
        # said "(none)". It is the same box; it now holds both halves.
        issues = tuple(meta.issues) + self._calibration_issues()
        has_issues = bool(issues)
        self._issues.setPlainText("\n".join(issues) if has_issues else "(none)")
        self._issues.setStyleSheet("color:#FFB74D;" if has_issues else "")

    def _kind_label(self, info: ChannelInfo) -> tuple[str, str]:
        """The Kind cell and its tooltip: what the channel is TREATED AS, everywhere
        (``Session.effective_kind`` -- the user's Calibration-panel override when there is
        one, else the header), with the header's own word kept in parentheses when the two
        differ: ``current (file says unknown)``.

        This panel's contract is "the facts the reader reported, verbatim", and the header's
        word IS kept -- but printing it ALONE made this the one surface out of step with
        every other on screen. Measured on a 2 s CSV of the owner's -700 mV recording with
        neutral column headers (both channels ``unknown``), after "Treat this channel as:
        current" + pA + gain 2.5 was committed through the real Apply button: the
        Calibration panel's title read ``Untitled · current (file says unknown)``, the
        Analysis panel's title the same, the chip ``✓ Gain set: ×2.5 → pA``, the rail
        offered DETECT -- and this table read Kind = ``unknown`` for both channels, before
        and after the detection, with no mention of the override. It is the panel a user
        opens to check what a channel IS, and it was the one that still said the header
        alone. Same shape as the two sibling titles (``calibration_panel._populate``,
        ``analysis_panel._kind_label``) so the three read alike; the header's word is
        always on screen, so the override is visibly an override.
        """
        header = info.kind
        if self._doc_id is None or not self._session.has_document(self._doc_id):
            return header.value, ""
        kind = self._session.effective_kind(self._doc_id, info.index)
        if kind is header:
            return header.value, ""
        return (f"{kind.value} (file says {header.value})",
                f"Your Calibration-panel override: the file declares this channel "
                f"'{header.value}'; it is treated as '{kind.value}' everywhere "
                f"(panes, detection, the workflow strip).")

    def _calibration_issues(self) -> tuple[str, ...]:
        """What the current calibration did to each channel of the shown document."""
        if self._doc_id is None or not self._session.has_document(self._doc_id):
            return ()
        out: list[str] = []
        for info in self._session.document(self._doc_id).meta.channels:
            try:
                out.extend(self._session.calibration_issues(self._doc_id, info.index))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(out)

    def _on_document_opened(self, doc_id: str) -> None:
        # Follow the newly opened document (it is about to become the selection anyway).
        self.set_document(doc_id)

    def _on_document_closed(self, doc_id: str) -> None:
        if self._doc_id == doc_id:
            self._show_empty()

    def _refresh_if(self, doc_id: str) -> None:
        if self._doc_id == doc_id:
            self._render(self._session.document(doc_id).meta)

    def _show_empty(self) -> None:
        self._doc_id = None
        self._title.setText("No document")
        for lbl in self._fields.values():
            lbl.setText("—")
        self._table.setRowCount(0)
        self._notes.setPlainText("")
        self._issues.setPlainText("")
