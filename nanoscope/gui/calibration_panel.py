from __future__ import annotations

import html

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis import iv
from ..core.calibration import (
    KIND_OVERRIDES,
    SIDECAR_DELETE_FAILED,
    SIDECAR_FAILED,
    SIDECAR_NONE,
    SIDECAR_RECORDING_MOVED,
    SIDECAR_REMOVED,
    SIDECAR_WRITTEN,
    CalibrationSet,
    ChannelCalibration,
    sidecar_path,
)
from ..core.document import TraceDocument
from ..core.session import Session
from ..core.types import ChannelKind, Provenance

__all__ = ["CalibrationPanel"]

# Strided sample size for the live MEDIAN (and the conductance / reconciliation medians).
# The median is robust to a stride on a stationary trace -- exact to 6 s.f. on all three of
# the owner's recordings -- and is the only row still read from it: min, max and span come
# from ``Session.raw_extremes`` (exact over every sample), because a tail order statistic
# taken from 0.16 % of a 12.5 M-sample channel is biased inward by construction and on the
# owner's -500 mV file printed a maximum of -7.63 pA where the channel reaches +5.49 pA.
_PROBE = 20_000


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.6g}"


def _median(arr: np.ndarray) -> float | None:
    """THE median this panel prints -- of the FINITE samples of a stride, in float64;
    ``None`` when it has none.

    One writer for all four medians on this screen (this channel's raw and calibrated
    rows, the voltage partner's, the current partner's I0), because ``np.median`` over a
    stride that carries one NaN is NaN, and on a recording the File info panel already
    reports as "a localised taint" the box printed "median (sampled)  nan  nan" under a
    title promising a robust median, and the conductance read "nan pA". A NaN/Inf is not
    a sample; the sentence beside the number says it was left out. For a stride with no
    non-finite sample this is ``np.median`` to the bit.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return None
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _count_nonfinite(arr: np.ndarray | None) -> int:
    return 0 if arr is None else int(np.count_nonzero(~np.isfinite(arr)))


def _fmt_voltage(value: float, units: str) -> str:
    """A voltage read-out with the units it actually has. ``""`` means the file never
    declared them and the user has not set any — say so, instead of asserting mV on the
    same screen whose units row reads "raw = (unknown)"."""
    return f"{value:.4g} {units}" if units else f"{value:.4g} (units unknown)"


class CalibrationPanel(QtWidgets.QWidget):
    """Edit gain / offset / target units for one (document, channel), previewing RAW vs CALIBRATED
    statistics live so the user can dial in the deferred current gain and SEE the effect before
    committing. Apply commits to Session (which re-renders the affected curves in ~8.5 ms without a
    reload); Reset restores CalibrationSet.default_for().

    Selection is pushed in by MainWindow via set_selection(); this panel never talks to another
    panel.
    """

    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._sel: tuple[str, int] | None = None
        # FLOAT32 sample of the selected channel -- the dtype ``ChannelCalibration.apply()``
        # computes in, so the preview overflows exactly where the commit will. A float64
        # preview showed "median -5.77e+39 pA" for gain 1e38 and offered Apply; apply()
        # then made every one of 749,651 samples -inf.
        self._raw_sample: np.ndarray | None = None
        self._raw_extremes: tuple[float, float] | None = None  # EXACT (min, max), nan-ignoring
        self._raw_n: int = 0                                   # samples in the channel
        self._raw_stride: int = 1                              # stride the median is read at
        self._volt_ch: int | None = None                 # doc's VOLTAGE channel index, if any
        self._volt_median_raw: float | None = None
        self._volt_units_raw: str = ""                   # VOLTAGE channel raw units ("" == undeclared)
        self._cur_ch: int | None = None                  # doc's CURRENT channel index, if any
        self._cur_sample: np.ndarray | None = None       # float64 sample of the CURRENT channel
        self._cur_units_raw: str = ""                    # CURRENT channel raw units (pA/nA)
        self._loading = False

        self._build_ui()

        session.calibrationChanged.connect(self._on_calibration_changed)
        session.documentClosed.connect(self._on_document_closed)
        self._show_empty()

    # -- UI ---------------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._title = QtWidgets.QLabel("No channel selected")
        # PLAIN TEXT, EXPLICITLY. This title is "<channel name> · <kind> · <raw units>", and
        # the name and the units come verbatim out of the FILE header (io/text.py takes a
        # CSV column header verbatim as ChannelInfo.name). A QLabel left at Qt::AutoText
        # runs mightBeRichText() over what it is handed and parses it as HTML the moment a
        # '<' is followed by a recognised tag name, so a channel really named ``Ch <A>`` was
        # painted ``Ch`` and ``I<sub>0</sub>`` was painted ``I0`` — while the File dock tree
        # (plain-drawn) showed the true name at the same moment, so the two readings of one
        # channel contradicted each other on one screen. This title is the channel
        # identification the user reads before pressing Apply to write a gain/offset (and a
        # sidecar) for that channel; two channels differing only after the '<' must not
        # become indistinguishable here. Same rule and same remedy as file_info_panel /
        # status_chip / event_gallery; the reasoning is spelled out at length in
        # ``baseline_panel._check_html``.
        self._title.setTextFormat(QtCore.Qt.PlainText)
        self._title.setStyleSheet("font-weight:600; font-size:13px;")
        root.addWidget(self._title)

        self._badge = QtWidgets.QLabel()
        # RICH TEXT, EXPLICITLY — this one really is markup (a coloured <span>), so it is
        # declared rather than left to mightBeRichText() to guess, and every value
        # interpolated into it is escaped at the setText (see :meth:`_set_badge`): the
        # provenance NOTE comes out of the sidecar file next to the user's data.
        self._badge.setTextFormat(QtCore.Qt.RichText)
        self._badge.setWordWrap(True)
        root.addWidget(self._badge)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self._gain = QtWidgets.QLineEdit()
        self._gain.setValidator(QtGui.QDoubleValidator(self._gain))
        self._gain.validator().setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self._gain.textChanged.connect(self._update_preview)
        self._gain.returnPressed.connect(self._apply)
        form.addRow("Gain", self._gain)

        self._offset = QtWidgets.QLineEdit()
        self._offset.setValidator(QtGui.QDoubleValidator(self._offset))
        self._offset.validator().setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self._offset.textChanged.connect(self._update_preview)
        self._offset.returnPressed.connect(self._apply)
        form.addRow("Offset", self._offset)

        self._units = QtWidgets.QLineEdit()
        self._units.setPlaceholderText("raw units unchanged")
        self._units.textChanged.connect(self._update_preview)
        self._units.returnPressed.connect(self._apply)
        form.addRow("Target units", self._units)

        # WHAT THE CHANNEL IS, next to what its numbers mean — the two halves of the same
        # correction, in one place. Answering only the units half relabelled the y-axis and left
        # the pane ordered as whatever the file declared, which is the owner's opening question
        # about this app: a channel he had corrected to 'pA' still sorted below the voltage,
        # so two recordings of one experiment opened with their panes in opposite orders. It is
        # also the only way to separate a TDMS whose channels are BOTH raw Volts: in overlay
        # they are one (kind, units) group, so they share one pane and one y-axis until
        # something says which is the current. See ``trace_view._kind``.
        self._treat_as = QtWidgets.QComboBox()
        self._treat_as.addItem("as the file declares", None)
        for kind in KIND_OVERRIDES:
            self._treat_as.addItem(kind.value, kind)
        self._treat_as.setToolTip(
            "What this channel IS, to the whole app: which pane it draws in and how it is "
            "grouped with the others, AND which channel the analysis acts on — Run "
            "detection, Batch detect, Power spectrum and I-V, the calibration chip and the "
            "workflow strip all follow it. 'current' makes an undeclared channel analysable; "
            "'voltage' on a channel the file calls current switches detection off for it. "
            "Leave it at 'as the file declares' unless the file declared the channel wrongly "
            "or not at all.")
        # No live preview hook: this changes no number in the stats table (it is not part of
        # ChannelCalibration.fingerprint and reaches no sample). What it changes, once Apply
        # commits it, is what the channel IS everywhere -- Session.effective_kind is the one
        # resolver: the pane and overlay group (trace_view) AND whether the pipeline acts on
        # the channel at all. Measured after sweep 7: "treat as voltage" on IN15 greys out
        # Analysis > Run detection, Batch detect, Power spectrum and I-V and locks
        # CALIBRATE / FILTER / DETECT (Session.resolve_current_channel 1 -> None). A tooltip
        # that said "which pane" over that reach sent the user looking for a layout change
        # when the analysis had been switched off.
        form.addRow("Treat this channel as", self._treat_as)
        root.addLayout(form)

        self._formula = QtWidgets.QLabel()
        # WRAPPED, like _status and _coverage_row: this line carries every refusal
        # _pending_problem words, and the float32-overflow one is a 250-character sentence
        # whose remedy ("Keep |gain| below about 1.29e+36 here") is its tail. Un-wrapped, it
        # took the panel's minimumSizeHint().width() from 402 px to 1,608 px (house limit
        # 1,100) and, inside the Inspector dock's scroll area, put that tail behind a
        # horizontal scrollbar at every dock width the owner uses.
        self._formula.setWordWrap(True)
        self._formula.setStyleSheet("color:#8b9199;")
        root.addWidget(self._formula)

        # RAW vs CALIBRATED live statistics -------------------------------------------------------
        # The title says WHICH rows are exact and which are sampled, because the box used to
        # say "(sampled)" over four rows of which one was robust and three were not, and
        # nothing on screen told them apart. Same policy as MeasurePanel (analysis/measure.py):
        # a strided preview names its stride and its counts; it never prints a tail order
        # statistic as if it were exact.
        self._stats_box = QtWidgets.QGroupBox(
            "Raw vs calibrated (min / max / span exact · median sampled)")
        grid = QtWidgets.QGridLayout(self._stats_box)
        grid.setHorizontalSpacing(16)
        headers = ["", "Raw", "Calibrated"]
        for c, h in enumerate(headers):
            lbl = QtWidgets.QLabel(h)
            lbl.setStyleSheet("color:#8b9199; font-weight:600;")
            grid.addWidget(lbl, 0, c)
        self._stat_labels: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        self._stat_names: dict[str, QtWidgets.QLabel] = {}
        for r, key in enumerate(("median", "min", "max", "span"), start=1):
            name_l = QtWidgets.QLabel("median (sampled)" if key == "median" else key)
            grid.addWidget(name_l, r, 0)
            self._stat_names[key] = name_l
            raw_l = QtWidgets.QLabel("—")
            cal_l = QtWidgets.QLabel("—")
            raw_l.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            cal_l.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            grid.addWidget(raw_l, r, 1)
            grid.addWidget(cal_l, r, 2)
            self._stat_labels[key] = (raw_l, cal_l)
        self._units_row = QtWidgets.QLabel("")
        # PLAIN TEXT: "units: raw = <raw_units>   calibrated = <target units>" — the raw
        # units are the file header's own string and the target units are whatever the user
        # typed. Neither is markup this app authored (see the title above).
        self._units_row.setTextFormat(QtCore.Qt.PlainText)
        self._units_row.setStyleSheet("color:#8b9199;")
        grid.addWidget(self._units_row, 5, 0, 1, 3)
        # WHAT WAS READ, in numbers: the stride and the sample count behind the median, and
        # the whole-channel count behind the extremes. Filled per selection.
        self._coverage_row = QtWidgets.QLabel("")
        self._coverage_row.setTextFormat(QtCore.Qt.PlainText)
        self._coverage_row.setWordWrap(True)
        self._coverage_row.setStyleSheet("color:#8b9199;")
        grid.addWidget(self._coverage_row, 6, 0, 1, 3)
        root.addWidget(self._stats_box)

        # Command-voltage reconciliation ----------------------------------------------------------
        self._recon_box = QtWidgets.QGroupBox("Command-voltage check")
        rlay = QtWidgets.QFormLayout(self._recon_box)
        self._recon_measured = QtWidgets.QLabel("—")
        self._recon_gain = QtWidgets.QLabel("—")
        self._recon_effective = QtWidgets.QLabel("—")
        self._recon_nominal = QtWidgets.QLabel("—")
        # PLAIN TEXT: "= effective" is formatted with the VOLTAGE channel's units string
        # off the file header (:func:`_fmt_voltage`); the other three are numbers, and are
        # declared with it so a row added later inherits the discipline rather than the
        # default.
        for lbl in (self._recon_measured, self._recon_gain,
                    self._recon_effective, self._recon_nominal):
            lbl.setTextFormat(QtCore.Qt.PlainText)
        rlay.addRow("Measured median (raw)", self._recon_measured)
        rlay.addRow("× gain + offset", self._recon_gain)
        rlay.addRow("= effective", self._recon_effective)
        rlay.addRow("Nominal (grouping)", self._recon_nominal)
        root.addWidget(self._recon_box)

        # Open-pore conductance ------------------------------------------------------------------
        # The whole point of the deferred current gain: G = I_open / V. Recomputed live as the
        # user edits THIS channel's gain, so they can dial it until G is physical (a 22 nm pore in
        # 1x PBS is ~100 nS). Shown whenever the document has a current channel, whichever channel
        # is selected — the current gain lives on the current channel, but its effect is the point.
        self._cond_box = QtWidgets.QGroupBox("Open-pore conductance")
        clay = QtWidgets.QFormLayout(self._cond_box)
        self._cond_iopen = QtWidgets.QLabel("—")
        self._cond_voltage = QtWidgets.QLabel("—")
        self._cond_g = QtWidgets.QLabel("—")
        self._cond_g.setStyleSheet("font-weight:600; font-size:14px;")
        for lbl in (self._cond_iopen, self._cond_voltage, self._cond_g):
            lbl.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            # PLAIN TEXT: I₀ and V are printed WITH their units, and those unit strings come
            # off the file header (or "(units unknown)" when it declared none).
            lbl.setTextFormat(QtCore.Qt.PlainText)
        clay.addRow("Open-pore current I₀", self._cond_iopen)
        clay.addRow("Applied voltage V", self._cond_voltage)
        clay.addRow("G = I₀ / V", self._cond_g)
        self._cond_note = QtWidgets.QLabel("")
        # PLAIN TEXT: ``iv.conductance_note`` names the units it could not convert, and it
        # names them by quoting the file's own string back at the reader.
        self._cond_note.setTextFormat(QtCore.Qt.PlainText)
        self._cond_note.setWordWrap(True)
        self._cond_note.setStyleSheet("color:#8b9199;")
        clay.addRow(self._cond_note)
        root.addWidget(self._cond_box)

        root.addStretch(1)

        btns = QtWidgets.QHBoxLayout()
        self._btn_reset = QtWidgets.QPushButton("Reset to default")
        self._btn_reset.clicked.connect(self._reset)
        self._btn_apply = QtWidgets.QPushButton("Apply")
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self._apply)
        btns.addWidget(self._btn_reset)
        btns.addStretch(1)
        btns.addWidget(self._btn_apply)
        root.addLayout(btns)

        # What the last commit did to <datafile>.npcal.json. A calibration commit can WRITE
        # or DELETE a file next to the user's raw data; doing either with no visible trace is
        # what made C12 invisible.
        self._status = QtWidgets.QLabel("")
        # PLAIN TEXT: this line names the sidecar FILE (derived from the recording's own
        # path) and, on a failure, quotes THE OS'S OWN reason string verbatim. Both are
        # strings this app did not author, and this is the only visible trace that a file
        # next to the user's raw data was written or deleted.
        self._status.setTextFormat(QtCore.Qt.PlainText)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#8b9199;")
        root.addWidget(self._status)

    # -- selection --------------------------------------------------------------------------------
    def set_selection(self, doc_id: str, channel: int) -> None:
        if not self._session.has_document(doc_id):
            self._show_empty()
            return
        self._sel = (doc_id, channel)
        self._load_samples(doc_id, channel)
        self._populate()

    def _load_samples(self, doc_id: str, channel: int) -> None:
        self._raw_sample = None
        self._raw_extremes = None
        self._raw_n = 0
        self._raw_stride = 1
        doc = self._session.document(doc_id)
        if not isinstance(doc, TraceDocument):
            self._resolve_partners(doc_id)
            return
        raw = doc.raw(channel)
        self._raw_n = int(len(raw))
        self._raw_stride = max(1, self._raw_n // _PROBE)
        self._raw_sample = np.asarray(raw[::self._raw_stride], dtype=np.float32)
        # Exact, cached on the session for the document's lifetime, and the SAME two
        # numbers Session.calibration_issues decides float32 overflow from -- so the
        # preview's refusal and the post-commit damage report cannot disagree.
        self._raw_extremes = self._session.raw_extremes(doc_id, channel)
        self._resolve_partners(doc_id)

    def _resolve_partners(self, doc_id: str) -> None:
        """THE ONE RESOLVER of the two partner boxes: which channel of the recording is
        the voltage the Command-voltage check reads, and which is the live current whose
        gain sets the Open-pore conductance.

        THE PARTNERS ARE RESOLVED BY WHAT THE CHANNELS ARE TREATED AS -- the user's
        Calibration-panel override when there is one (Session.effective_kind), not the
        header alone. Reading the header here meant the Open-pore-conductance and
        Command-voltage boxes stayed HIDDEN on exactly the recordings the override was
        built for: an undeclared CSV corrected to 'current' + pA showed
        _cond_box.isVisible() == False, I0 / V / G all '—', in the one panel that exists
        to dial a current gain against a conductance.

        AND IT RUNS ON EVERY PATH THAT REPAINTS THE BOXES, not only on a reselect. Its
        first home was ``_load_samples`` (reached from ``set_selection`` alone), so the
        Apply that committed the override -- ``_on_calibration_changed`` -> ``_populate``
        -- showed the boxes from the partners of the PREVIOUS resolution: the title
        flipped to "current (file says unknown)" while the conductance box stayed hidden
        until the user clicked another channel and back; and in the reverse direction,
        re-declaring the current channel a voltage left the box printing "I0 = -122.8 mV"
        (G '—') from the channel just declared a voltage. Cheap by construction: a
        <= 20,000-sample strided median per partner.
        """
        self._volt_ch = None
        self._volt_median_raw = None
        self._volt_units_raw = ""
        self._cur_ch = None
        self._cur_sample = None
        self._cur_units_raw = ""
        doc = self._session.document(doc_id)
        if not isinstance(doc, TraceDocument):
            return
        for info in doc.meta.channels:
            kind = self._session.effective_kind(doc_id, info.index)
            if kind is ChannelKind.VOLTAGE and self._volt_ch is None:
                self._volt_ch = info.index
                self._volt_median_raw = _median(self._sample(doc.raw(info.index)))
                self._volt_units_raw = info.raw_units
            # First LIVE current channel — the one whose gain sets the conductance.
            if kind is ChannelKind.CURRENT and not info.is_dead and self._cur_ch is None:
                self._cur_ch = info.index
                self._cur_sample = self._sample(doc.raw(info.index))
                self._cur_units_raw = info.raw_units

    @staticmethod
    def _sample(arr: np.ndarray) -> np.ndarray:
        """A <= ``_PROBE``-sample stride in float64 -- for the MEDIANS only (this channel's,
        the voltage partner's, the current partner's). Never for an extreme."""
        step = max(1, len(arr) // _PROBE)
        return np.asarray(arr[::step], dtype=np.float64)

    def _finite_extremes(self) -> tuple[float, float] | None:
        """The exact raw ``(min, max)`` when both are finite, else ``None`` -- a channel with
        an infinity in it (already reported by the load) has no extremes to reason from."""
        ext = self._raw_extremes
        if ext is None or not (np.isfinite(ext[0]) and np.isfinite(ext[1])):
            return None
        return float(ext[0]), float(ext[1])

    def _calibrated_extremes(self, gain: float, offset: float) -> tuple[float, float] | None:
        """Exact calibrated ``(min, max)`` from the exact raw extremes, computed IN FLOAT32
        WITH THE SAME ARITHMETIC AS ``ChannelCalibration.apply()`` -- so a gain that
        overflows the real channel overflows here, and a negative gain that swaps the
        extremes swaps them here. The output is sorted, so (min, max) stays (min, max)."""
        ext = self._finite_extremes()
        if ext is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            out = (np.asarray(ext, dtype=np.float32) * np.float32(gain)
                   + np.float32(offset))
        lo, hi = sorted((float(out[0]), float(out[1])))
        return lo, hi

    def _overflows_float32(self, gain: float, offset: float) -> bool:
        """Would ``apply()`` make a finite sample of this channel non-finite? Decided from
        the exact extremes -- affine + monotone, so the two extremes bound every output --
        and only when the extremes are themselves finite (an infinity the recording already
        carries is the load's report, not this calibration's doing)."""
        ext = self._calibrated_extremes(gain, offset)
        return ext is not None and not (np.isfinite(ext[0]) and np.isfinite(ext[1]))

    def _populate(self) -> None:
        assert self._sel is not None
        doc_id, ch = self._sel
        doc = self._session.document(doc_id)
        info = doc.meta.channels[ch]
        cal = self._session.calibration(doc_id, ch)

        # THE KIND IN THE TITLE IS WHAT THE CHANNEL IS TREATED AS. After the user committed
        # "Treat this channel as: current" the title still read "Untitled · unknown" two
        # rows above the combo reading "current" -- the panel contradicting itself about the
        # one thing it had just been told. The header's word is kept, in parentheses, so
        # the override is visibly an override.
        eff_kind = self._session.effective_kind(doc_id, ch)
        kind_text = (eff_kind.value if eff_kind is info.kind
                     else f"{eff_kind.value} (file says {info.kind.value})")

        self._loading = True
        try:
            self._title.setText(f"{info.name}   ·   {kind_text}"
                                + (f"   ·   {info.raw_units}" if info.raw_units else ""))
            self._gain.setText(f"{cal.gain:g}")
            self._offset.setText(f"{cal.offset:g}")
            self._units.setText(cal.target_units)
            self._set_treat_as(cal.kind)
            self._set_badge(cal)
            enabled = isinstance(doc, TraceDocument)
            for w in (self._gain, self._offset, self._units, self._treat_as,
                      self._btn_apply, self._btn_reset):
                w.setEnabled(enabled)
            self._recon_box.setVisible(self._volt_ch is not None)
            self._cond_box.setVisible(self._cur_ch is not None)
        finally:
            self._loading = False
        self._update_preview()

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def status_text(self) -> str:
        """What the panel last reported about the sidecar (public for the suite)."""
        return self._status.text()

    def _set_treat_as(self, kind) -> None:
        """Point the combo at ``kind`` (``None`` == "as the file declares").

        By DATA and not by index: the entries are built from ``KIND_OVERRIDES``, and a lookup
        by position would silently select the wrong kind the day that tuple grows an entry.
        A kind the combo does not offer — which a hand-edited sidecar can perfectly well store,
        since :class:`ChannelCalibration` accepts any real :class:`ChannelKind` — falls back to
        "as the file declares" rather than to entry 0 by accident; :meth:`_apply` would then
        drop the override, which is the honest outcome for a value this panel cannot show.
        """
        index = self._treat_as.findData(kind)
        self._treat_as.setCurrentIndex(index if index >= 0 else 0)

    def _set_badge(self, cal: ChannelCalibration) -> None:
        colors = {
            Provenance.MEASURED: "#81C784",
            Provenance.FILE_META: "#4FC3F7",
            Provenance.FILENAME: "#FFB74D",
            Provenance.UNKNOWN: "#8b9199",
        }
        color = colors.get(cal.provenance, "#8b9199")
        # EVERY INTERPOLATED VALUE IS ESCAPED, BECAUSE THIS LABEL IS RICH TEXT (it carries a
        # coloured <span>). ``cal.note`` is read out of the .npcal.json sidecar next to the
        # user's data — a string this app did not author — and an unescaped '<' in it would
        # be swallowed together with everything up to the next '>', exactly as
        # ``baseline_panel._check_html`` records happening to a whole criterion there.
        note = f"  ·  {html.escape(cal.note)}" if cal.note else ""
        self._badge.setText(
            f'<span style="color:{color}; font-weight:600;">provenance: '
            f'{html.escape(cal.provenance.value)}</span>{note}')

    # -- live preview -----------------------------------------------------------------------------
    def _pending(self) -> tuple[float, float] | None:
        """The typed (gain, offset), or None when they are not usable numbers.

        The finiteness check is NOT redundant with the QDoubleValidator: the validator
        accepts ``1e999`` (it is a well-formed number in scientific notation, and the default
        range is unbounded), and ``float("1e999")`` is ``inf``. Committing that would make
        every sample of the recording infinite. Same rule as the sidecar and the project —
        it lives on ChannelCalibration; this is the panel declining to build one it knows
        would be refused, so the preview says "enter a number" instead of raising.
        """
        try:
            gain = float(self._gain.text())
            offset = float(self._offset.text())
        except ValueError:
            return None
        if not (np.isfinite(gain) and np.isfinite(offset)):
            return None
        if float(np.float32(gain)) == 0.0:
            return None
        # A FINITE gain that overflows float32 on THIS channel ruins it exactly as an
        # infinite one does, and was accepted because the preview was float64. The bound is
        # per channel (3.4e38 / the largest raw magnitude: 1.37e36 on the owner's -1000 mV
        # IN15), so it is decided from the exact extremes, not from a fixed number.
        if self._overflows_float32(gain, offset):
            return None
        return gain, offset

    def _pending_problem(self) -> str:
        """Why the typed values cannot be committed — the FIRST real reason, in words.

        The preview said "gain/offset: enter a number" for a perfectly well-formed ``0``,
        which is not what is wrong with it. Every refusal :class:`ChannelCalibration` makes
        has a sentence here, so a disabled-looking form always explains itself.
        """
        try:
            gain = float(self._gain.text())
            offset = float(self._offset.text())
        except ValueError:
            return "gain/offset: enter a number"
        if not (np.isfinite(gain) and np.isfinite(offset)):
            return ("gain/offset: enter a finite number — an infinite one makes every "
                    "sample infinite")
        if float(np.float32(gain)) == 0.0:
            return ("gain 0 would make every sample exactly the offset — that deletes the "
                    "recording rather than scaling it. Use a small gain, not zero.")
        if self._overflows_float32(gain, offset):
            lo, hi = self._finite_extremes()   # type: ignore[misc]  (overflow implies finite)
            reach = max(abs(lo), abs(hi))
            bound = float(np.finfo(np.float32).max) / reach if reach > 0 else float("inf")
            return (f"gain {gain:g} overflows float32 on this channel: its raw samples reach "
                    f"{lo:.6g} … {hi:.6g}, so calibrated values would exceed ±3.4e+38 and "
                    f"every sample would become ±inf — the same ruin as an infinite gain. "
                    f"Keep |gain| below about {bound:.3g} here (or reduce the offset).")
        return "gain/offset: enter a number"

    def _update_preview(self, *_a) -> None:
        if self._loading or self._sel is None:
            return
        doc_id, ch = self._sel
        info = self._session.document(doc_id).meta.channels[ch]
        pend = self._pending()

        eff_units = self._units.text().strip() or info.raw_units
        self._units_row.setText(f"units: raw = {info.raw_units or '(unknown)'}"
                                + (f"    calibrated = {eff_units}" if eff_units else ""))

        # Apply is OFFERED only for values that will be committed. The button used to stay
        # enabled over a refused value and silently do nothing; the refusal is in the
        # formula line, and a disabled button says the same thing without a click.
        self._btn_apply.setEnabled(pend is not None and self._raw_sample is not None)
        if pend is None or self._raw_sample is None:
            self._formula.setText(self._pending_problem())
            for raw_l, cal_l in self._stat_labels.values():
                cal_l.setText("—")
            self._fill_raw_stats()
            self._update_conductance(pend)
            return
        gain, offset = pend
        self._formula.setText(f"calibrated = raw × {gain:g} + {offset:g}")
        self._fill_raw_stats()
        # THE SAME ARITHMETIC AS apply(): float32 sample × float32 gain + float32 offset.
        # Not "gain * sample" in float64 -- that is how 1e38 previewed as -5.77e+39.
        with np.errstate(over="ignore", invalid="ignore"):
            cal = self._raw_sample * np.float32(gain) + np.float32(offset)
        self._stat_labels["median"][1].setText(_fmt(_median(cal)))
        ext = self._calibrated_extremes(gain, offset)
        if ext is None:
            for key in ("min", "max", "span"):
                self._stat_labels[key][1].setText("—")
        else:
            lo, hi = ext
            self._set_stat("min", None, lo)
            self._set_stat("max", None, hi)
            self._set_stat("span", None, hi - lo)
        self._update_reconciliation(pend)
        self._update_conductance(pend)

    def _fill_raw_stats(self) -> None:
        """The RAW column: the median from the stride, the extremes and span EXACT.

        And say so, in the row labels' tooltips and the coverage line, with the stride and
        the counts -- MeasurePanel's policy (analysis/measure.py). Measured before this: on
        the owner's -500 mV recording (12,502,631 samples, every 625th read) the strided
        maximum was -7.63 pA against a true +5.49 pA, the minimum -215.5 against -281.7,
        the span 207.8 against 287.2 -- all printed as plain numbers under a title that
        said only "(sampled)". The median was exact to 6 s.f. on the same stride.
        """
        s = self._raw_sample
        if s is None or s.size == 0:
            for raw_l, _ in self._stat_labels.values():
                raw_l.setText("—")
            self._coverage_row.setText("")
            for key, name_l in self._stat_names.items():
                name_l.setToolTip("")
            return
        n, stride, m = self._raw_n, self._raw_stride, int(s.size)
        self._stat_labels["median"][0].setText(_fmt(_median(s)))
        ext = self._finite_extremes()
        if ext is None:
            for key in ("min", "max", "span"):
                self._stat_labels[key][0].setText("—")
        else:
            lo, hi = ext
            self._set_stat("min", lo, None)
            self._set_stat("max", hi, None)
            self._set_stat("span", hi - lo, None)
        every = "every sample" if stride == 1 else f"every {stride}th sample"
        # WHAT THE SENTENCES CLAIM FOLLOWS WHAT THE ROWS COULD COMPUTE. On a channel that
        # carries an infinity Session.raw_extremes is non-finite, the three rows above are
        # '—' -- and this line read "min / max / span: exact over all 5,000 samples" with
        # every tooltip saying "Exact over all 5,000 samples of the channel" over the three
        # blanks. A claim of exactness over a dash is the one thing the CAL3 fix existed to
        # stop. The taint itself is the load's report (File info names the sample); here it
        # is named as the reason the rows are empty, and the median says what it left out.
        bad = _count_nonfinite(s)
        left_out = (", non-finite samples ignored" if ext is None else
                    f", {bad:,} non-finite ignored" if bad else "")
        median_tip = (f"Median of {every} — {m:,} of {n:,} samples{left_out}. Robust to the "
                      f"stride on a stationary trace; the exact rows are min, max and span.")
        if ext is None:
            exact_tip = ("Not computable: the channel holds a non-finite sample (NaN/Inf), "
                         "so its extremes are not numbers — see File info for where. "
                         "Nothing in this row is exact over the channel.")
            extremes_line = ("min / max / span: not computable — the channel holds a "
                             "non-finite sample (NaN/Inf); see File info.")
            box_tip = (f"min, max and span are not computable (the channel holds a "
                       f"non-finite sample; see File info); the median is read from "
                       f"{every} ({m:,} of {n:,}){left_out}.")
        else:
            nan_note = " (NaN ignored)" if bad else ""
            exact_tip = (f"Exact over all {n:,} samples of the channel{nan_note}. The "
                         f"calibrated value is the raw extreme through the same float32 "
                         f"arithmetic Apply uses.")
            extremes_line = f"min / max / span: exact over all {n:,} samples{nan_note}."
            box_tip = (f"min, max and span are exact over all {n:,} samples{nan_note}; the "
                       f"median is read from {every} ({m:,} of {n:,}){left_out}.")
        self._stat_names["median"].setToolTip(median_tip)
        for key in ("min", "max", "span"):
            self._stat_names[key].setToolTip(exact_tip)
        for key, (raw_l, cal_l) in self._stat_labels.items():
            tip = median_tip if key == "median" else exact_tip
            raw_l.setToolTip(tip)
            cal_l.setToolTip(tip)
        self._coverage_row.setText(
            f"{extremes_line}   median: {every} ({m:,} of {n:,}){left_out}.")
        self._stats_box.setToolTip(box_tip)

    def _set_stat(self, key: str, raw: float | None, cal: float | None) -> None:
        raw_l, cal_l = self._stat_labels[key]
        if raw is not None:
            raw_l.setText(_fmt(raw))
        if cal is not None:
            cal_l.setText(_fmt(cal))

    def _volt_effective(self, pend: tuple[float, float] | None) -> tuple[float, float, str]:
        """The VOLTAGE channel's effective (gain, offset, units).

        The values being TYPED when that channel is the one selected (so the read-outs
        track the live edit), the committed calibration otherwise — the same rule as
        :meth:`_readout_open_pore_current` for the current channel. ONE resolver, shared
        by the command-voltage check and the conductance's applied voltage, so the two
        boxes cannot disagree about the same channel. Units ``""`` == the file never
        declared them and the user has not set any."""
        doc_id, ch = self._sel  # type: ignore[misc]
        if ch == self._volt_ch and pend is not None:
            gain, offset = pend
            units = self._units.text().strip() or self._volt_units_raw
        else:
            cal = self._session.calibration(doc_id, self._volt_ch)
            gain, offset = cal.gain, cal.offset
            units = cal.target_units or self._volt_units_raw
        return gain, offset, units

    def _update_reconciliation(self, pending: tuple[float, float]) -> None:
        if self._volt_ch is None or self._volt_median_raw is None:
            # No voltage partner (any more): the box is hidden, and it must hold nothing a
            # later show could paint before the next resolution fills it.
            for lbl in (self._recon_measured, self._recon_gain,
                        self._recon_effective, self._recon_nominal):
                lbl.setText("—")
            return
        # The full calibration — gain AND offset — reaches the effective voltage, exactly as
        # the stats table applies it. Using the gain alone made the two disagree on the same
        # screen by exactly the offset, and fed conductance the uncorrected voltage.
        gain, offset, units = self._volt_effective(pending)
        effective = self._volt_median_raw * gain + offset
        doc_id, _ch = self._sel  # type: ignore[misc]
        nominal = self._session.document(doc_id).meta.nominal_voltage_mV
        self._recon_measured.setText(_fmt(self._volt_median_raw))
        self._recon_gain.setText(f"{gain:g} + {offset:g}")
        self._recon_effective.setText(_fmt_voltage(effective, units))
        self._recon_nominal.setText("—" if nominal is None else f"{nominal:.4g} mV")

    # -- conductance read-out ---------------------------------------------------------------------
    def _readout_open_pore_current(
        self, pend: tuple[float, float] | None
    ) -> tuple[float | None, str]:
        """Open-pore current of the doc's CURRENT channel + its (calibrated) units.

        Uses the gain/offset the user is DIALING when the current channel is the one selected (so
        the read-out tracks the live edit), and the committed calibration otherwise (so it is still
        shown while the user is on the voltage channel)."""
        if self._cur_ch is None or self._cur_sample is None or self._sel is None:
            return None, ""
        doc_id, ch = self._sel
        if ch == self._cur_ch and pend is not None:
            gain, offset = pend
            units = self._units.text().strip() or self._cur_units_raw
        else:
            cal = self._session.calibration(doc_id, self._cur_ch)
            gain, offset = cal.gain, cal.offset
            units = cal.target_units or self._cur_units_raw
        return _median(gain * self._cur_sample + offset), units

    def _readout_applied_voltage(
        self, pend: tuple[float, float] | None
    ) -> tuple[float | None, str]:
        """Applied command voltage + its (calibrated) units.

        The measured VOLTAGE channel median through the FULL calibration — gain and offset,
        the same numbers the reconciliation box shows — or ``nominal_voltage_mV`` (mV by
        that field's contract) when there is no voltage channel. Units ``""`` == undeclared:
        the caller must not label the number mV or divide it into a nS conductance."""
        if self._sel is None:
            return None, ""
        doc_id, _ch = self._sel
        if self._volt_ch is not None and self._volt_median_raw is not None:
            gain, offset, units = self._volt_effective(pend)
            return self._volt_median_raw * gain + offset, units
        nominal = self._session.document(doc_id).meta.nominal_voltage_mV
        return (None, "") if nominal is None else (float(nominal), "mV")

    def _update_conductance(self, pend: tuple[float, float] | None) -> None:
        """Recompute the open-pore conductance read-out. Called on every gain/offset/units edit,
        so the headline nS number moves live as the current gain is dialed."""
        if self._cur_ch is None or self._cur_sample is None:
            # No current partner (any more): the box is hidden. BLANKED, not left: after
            # the current channel was re-declared a voltage the hidden box still held
            # "I0 = -122.8 mV", and the next resolution that showed it painted that first.
            for lbl in (self._cond_iopen, self._cond_voltage, self._cond_g):
                lbl.setText("—")
            self._cond_note.setText("")
            return
        i_open, cur_units = self._readout_open_pore_current(pend)
        v_val, v_units = self._readout_applied_voltage(pend)
        self._cond_iopen.setText(
            "—" if i_open is None else f"{i_open:.4g} {cur_units}".rstrip())
        self._cond_voltage.setText("—" if v_val is None else _fmt_voltage(v_val, v_units))
        # G in nS only when BOTH units are known — to_nS refuses an unknown voltage unit the
        # same way it refuses an unknown current unit, and the note below says which one.
        ratio = (iv.conductance(i_open, v_val)
                 if (i_open is not None and v_val is not None) else None)
        g_nS = iv.to_nS(ratio, cur_units, v_units) if ratio is not None else None
        self._cond_g.setText("—" if g_nS is None else f"{g_nS:.4g} nS")

        note = iv.conductance_note(cur_units, v_units)
        if v_val is None:
            note = "no applied voltage known for this recording"
        elif self._volt_ch is None:
            note += "   ·   V from filename/header (nominal)"
        else:
            note += "   ·   V measured on the command channel"
        self._cond_note.setText(note)

    # -- commit -----------------------------------------------------------------------------------
    # THE single commit path. Apply and Reset both land here and NOTHING else in this panel
    # may call session.set_calibration (tests/gui/test_sidecar_safety.py has an AST guard).
    #
    # The reason it is single: committing a calibration can DELETE
    # ``<datafile>.npcal.json`` — a user-owned file next to their raw data, carrying a `note`
    # a project cannot round-trip — and that happens whenever every channel ends up at its
    # default. *Reset to default* does that by definition; *Apply* does it too, the moment
    # the typed values happen to be the defaults. Fixing only the button that made it obvious
    # would leave the identical silent deletion one keystroke away.
    def _commit(self, cal: ChannelCalibration, *, action: str = "Apply") -> bool:
        """Install ``cal`` on the selected channel. False == the user declined.

        Asks BEFORE the write whenever it would destroy something the app cannot put back,
        and afterwards states what actually happened to the file, so a loss is never
        something the user has to discover later on disk. There are TWO such things and the
        rule is asked about at the level the damage happens:

        * the sidecar FILE is deleted (every channel back at its default) — the unlink is
          not undoable;
        * the file survives but a channel's ``note`` goes out of it. Guarding only the first
          left this one silent: reset one channel of a recording whose other channel is
          still calibrated and the note was dropped from a file that stayed exactly where it
          was, under a status line reading "Applied · saved to …". The file is not the thing
          that cannot be recreated; the note is.

        One prompt per commit — a removal already says the note goes with it, so the second
        question is only asked when the first was not.

        ``action`` is the button the user actually pressed. It exists because the one
        confirmation guarding the sidecar was titled "Reset calibration" unconditionally —
        including when *Apply* raised it, which it does whenever the typed values happen to
        equal the defaults. A destructive prompt that names a button the user did not press
        reads as the app talking about something else, and the natural response is to accept
        it without reading.
        """
        if self._sel is None:
            return False
        doc_id, ch = self._sel
        effect = self._session.sidecar_effect_of(doc_id, ch, cal)
        path = sidecar_path(self._session.document(doc_id).meta.path)
        if effect == SIDECAR_REMOVED:
            if not self._confirm_sidecar_removal(path, action=action):
                self._set_status("Nothing changed.")
                return False
        else:
            lost = self._session.sidecar_notes_lost_by(doc_id, ch, cal)
            if lost and not self._confirm_note_loss(path, lost, action=action):
                self._set_status("Nothing changed.")
                return False
        reason: list[str] = []
        outcome = self._session.set_calibration(doc_id, ch, cal, reason=reason)
        self._set_status(self._outcome_text(outcome, doc_id, reason))
        return True

    def _confirm_sidecar_removal(self, path, *, action: str = "Apply") -> bool:
        """Ask before deleting the user's calibration sidecar. True to proceed.

        A separate method so a test can answer it without a live modal — the deletion is
        the thing under test, and a suite that cannot reach it cannot pin it.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        # PLAIN TEXT: both strings below carry the user's own FILE NAME and full PATH.
        # QMessageBox has the same Qt::AutoText default a QLabel has, and a confirmation
        # that mis-names the file it is about to delete is asking for consent to something
        # other than what it says.
        box.setTextFormat(QtCore.Qt.PlainText)
        box.setWindowTitle(f"{action} calibration")
        # The ACTION belongs in the body, not only in the title: macOS discards a
        # QMessageBox window title entirely (the platform guidelines forbid it), so on the
        # machine this app is developed and used on, a title is a string nobody reads — the
        # same shape as the tooltip this project already lost on a native menu item. The
        # button label carries it too, and both of those are on screen.
        box.setText(f"{action} will delete {path.name}.")
        box.setInformativeText(
            f"Every channel of this recording would be back at its default, so the saved "
            f"calibration file next to your data is removed:\n\n{path}\n\n"
            f"Any note stored in it is lost — a project file cannot carry that note. "
            f"The calibration itself is still reset either way.")
        delete = box.addButton(f"{action} and delete the file",
                               QtWidgets.QMessageBox.DestructiveRole)
        keep = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is delete

    def _confirm_note_loss(self, path, lost: tuple[tuple[int, str], ...], *,
                           action: str = "Apply") -> bool:
        """Ask before a write drops a note out of a sidecar that survives. True to proceed.

        Shows the note IN FULL, because the only remedy is to keep a copy of it, and a
        confirmation that summarises the thing it is about to delete is asking the user to
        approve something they can no longer read. A separate method, for the same reason as
        :meth:`_confirm_sidecar_removal`: a suite that cannot answer it cannot pin it.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        # PLAIN TEXT, and this one matters most of all: the informative text prints the
        # NOTES themselves, read out of the sidecar file, because the only remedy is to keep
        # a copy of them. A note containing a '<' would be shown with a piece missing in the
        # one dialog whose entire purpose is letting the user read it before it is erased.
        box.setTextFormat(QtCore.Qt.PlainText)
        box.setWindowTitle(f"{action} calibration")
        what = "note" if len(lost) == 1 else "notes"
        box.setText(f"{action} will erase {len(lost)} calibration {what} from {path.name}.")
        box.setInformativeText(
            "Clearing a channel's calibration removes its entry from the file next to your "
            "data, and the note goes with it. A project file cannot carry it back, and "
            "nothing else on this machine has a copy:\n\n"
            + "\n\n".join(f"channel {ch}: “{note}”" for ch, note in lost)
            + f"\n\nThe rest of {path.name} is left as it is.")
        go = box.addButton(f"{action} and erase the {what}",
                           QtWidgets.QMessageBox.DestructiveRole)
        keep = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is go

    def _outcome_text(self, outcome: str, doc_id: str, reason: list[str] | None = None) -> str:
        """What the last commit did to the sidecar, in the user's words.

        The failure branch reports THE OS'S OWN REASON. It used to say "(the folder is not
        writable)" for every ``OSError`` — one of at least five things that error means
        here (a full disk, a read-only file, a disconnected network volume, a name too long,
        a permission denied), and the one a user cannot act on when their folder is in fact
        writable. A guess in a failure message costs more than no explanation, because it
        sends the user to check something that is already correct.

        The two failures get OPPOSITE consequence sentences because their consequences are
        opposite. A failed WRITE (``SIDECAR_FAILED``) leaves nothing next to the data: the
        calibration really is gone on reopen. A failed DELETE (``SIDECAR_DELETE_FAILED``)
        leaves the OLD file in place: its values silently re-apply on reopen. One shared
        string used to tell a user who had just confirmed "Reset and delete the file" that
        the calibration "will be gone when you reopen" — the exact inverse of what the
        surviving sidecar was about to do to them.
        """
        data_path = self._session.document(doc_id).meta.path
        name = sidecar_path(data_path).name
        why = f" — {reason[0]}" if reason else ""
        return {
            SIDECAR_WRITTEN: f"Applied · saved to {name}",
            # The recording's file was beside that path on open and is not there now. The
            # old text said "saved to pore.abf.npcal.json" and wrote the file into the
            # folder the recording had LEFT; reopening it where it now lived gave gain 1.0.
            SIDECAR_RECORDING_MOVED: (
                f"Applied · not saved beside the recording: {data_path.name} is no longer "
                f"at {data_path.parent}; the calibration lives only in this session (and "
                f"in a saved project). Move the recording back, or save a project, to keep it."),
            SIDECAR_REMOVED: f"Applied · {name} deleted (all channels are back at default)",
            SIDECAR_NONE: "Applied",
            SIDECAR_FAILED: (f"Applied to this session only · could NOT update {name}{why}. "
                             f"The calibration is NOT stored next to your data and will be "
                             f"gone when you reopen the recording."),
            SIDECAR_DELETE_FAILED: (
                f"Applied to this session only · could NOT delete {name}{why}. "
                f"The OLD calibration file is still next to your data and WILL re-apply "
                f"its values (and any note) when you reopen the recording."),
        }.get(outcome, "Applied")

    def _apply(self) -> None:
        if self._sel is None:
            return
        pend = self._pending()
        if pend is None:
            return
        doc_id, ch = self._sel
        gain, offset = pend
        existing = self._session.calibration(doc_id, ch)
        target_units = self._units.text().strip()
        # A user edit carries no measurement/header evidence -> provenance UNKNOWN, but keep the
        # advisory note (e.g. the IN14 10x explanation) so the reason it existed survives.
        cal = ChannelCalibration(gain=gain, offset=offset, target_units=target_units,
                                 provenance=Provenance.UNKNOWN, note=existing.note,
                                 kind=self._treat_as.currentData())
        self._commit(cal, action="Apply")

    def _reset(self) -> None:
        if self._sel is None:
            return
        doc_id, ch = self._sel
        info = self._session.document(doc_id).meta.channels[ch]
        self._commit(CalibrationSet.default_for(info), action="Reset")

    # -- reactions --------------------------------------------------------------------------------
    def _on_calibration_changed(self, doc_id: str, ch: int) -> None:
        if self._sel is None or self._sel[0] != doc_id:
            return
        # A CALIBRATION OF THIS RECORDING CHANGED, ON ANY OF ITS CHANNELS: the partners are
        # re-resolved first, whichever channel it was, because a calibration carries the
        # kind override and a kind change on ANY channel can make or unmake a partner. The
        # boxes are then shown from the fresh resolution. Repopulating from the stale one
        # is how the title said "current (file says unknown)" over a hidden conductance box
        # (see _resolve_partners).
        self._resolve_partners(doc_id)
        if self._sel[1] == ch:
            self._populate()
            return
        # A sibling's calibration changed while a different channel is selected: the
        # user's typed values on the selected channel stay as they are; what follows the
        # sibling's committed calibration -- the boxes' presence and the reconciliation /
        # conductance read-outs -- is refreshed.
        self._recon_box.setVisible(self._volt_ch is not None)
        self._cond_box.setVisible(self._cur_ch is not None)
        pend = self._pending()
        if pend is not None:
            self._update_reconciliation(pend)
        self._update_conductance(pend)

    def _on_document_closed(self, doc_id: str) -> None:
        if self._sel is not None and self._sel[0] == doc_id:
            self._show_empty()

    def _show_empty(self) -> None:
        self._sel = None
        self._raw_sample = None
        self._raw_extremes = None
        self._raw_n = 0
        self._raw_stride = 1
        self._cur_ch = None
        self._cur_sample = None
        self._loading = True
        try:
            self._title.setText("No channel selected")
            self._badge.setText("")
            self._formula.setText("")
            self._gain.clear()
            self._offset.clear()
            self._units.clear()
            self._set_treat_as(None)
            for raw_l, cal_l in self._stat_labels.values():
                raw_l.setText("—")
                cal_l.setText("—")
                raw_l.setToolTip("")
                cal_l.setToolTip("")
            for name_l in self._stat_names.values():
                name_l.setToolTip("")
            self._coverage_row.setText("")
            self._stats_box.setToolTip("")
            self._units_row.setText("")
            self._recon_box.setVisible(False)
            self._cond_box.setVisible(False)
            for lbl in (self._cond_iopen, self._cond_voltage, self._cond_g):
                lbl.setText("—")
            self._cond_note.setText("")
            self._status.setText("")
            for w in (self._gain, self._offset, self._units, self._treat_as,
                      self._btn_apply, self._btn_reset):
                w.setEnabled(False)
        finally:
            self._loading = False
