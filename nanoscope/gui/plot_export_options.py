"""Presentation-only size and format choices for the current live figure.

This dialog neither chooses a destination nor writes a file. Callers bypass it for
programmatic exports. Zero primitive dimensions retain the native-size PNG/PDF
contract; physical SVG dimensions travel separately so its viewBox stays unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6 import QtCore, QtWidgets


MAX_RASTER_EDGE = 32_768
MAX_RASTER_PIXELS = 100_000_000
MAX_PHYSICAL_MM = 2_000.0
DEFAULT_DPI = 300


@dataclass(frozen=True)
class ExportOptions:
    """Validated presentation choices, with PNG pixels / PDF points in width/height.

    Native exports and all SVG exports use zero primitive dimensions. Physical SVG
    sizes must be applied using the mm metadata, preserving the existing viewBox.
    Whole pixel/point rounding can cause a small difference from requested mm size.
    """

    extension: str
    width: int = 0
    height: int = 0
    physical_width_mm: float | None = None
    physical_height_mm: float | None = None
    dpi: int = DEFAULT_DPI

    @property
    def native(self) -> bool:
        return self.physical_width_mm is None and self.physical_height_mm is None


def _native_dimensions(native_size) -> tuple[float, float]:
    if hasattr(native_size, "width") and hasattr(native_size, "height"):
        width, height = native_size.width(), native_size.height()
    else:
        try:
            width, height = native_size
        except (TypeError, ValueError) as exc:
            raise ValueError("The plot size must contain a width and height.") from exc
    try:
        width, height = float(width), float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("The plot size must be numeric.") from exc
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("The plot must have a finite, positive width and height.")
    return width, height


def _format_extension(value: str) -> str:
    extension = "." + str(value).lower().lstrip(".")
    if extension not in {".png", ".pdf", ".svg"}:
        raise ValueError("Choose PNG, PDF, or SVG.")
    return extension


def _validate_raster(width: int, height: int) -> None:
    if width > MAX_RASTER_EDGE or height > MAX_RASTER_EDGE:
        raise ValueError(
            f"PNG dimensions must not exceed {MAX_RASTER_EDGE:,} pixels per side. "
            "Reduce the physical size or DPI.")
    if width * height > MAX_RASTER_PIXELS:
        raise ValueError(
            f"PNG export is limited to {MAX_RASTER_PIXELS // 1_000_000} megapixels. "
            "Reduce the physical size or DPI.")


def resolve_export_options(native_size, *, extension: str = ".png",
                           native: bool = True, width: float = 85.0,
                           height: float = 60.0, units: str = "mm",
                           dpi: int = DEFAULT_DPI) -> ExportOptions:
    """Translate and validate choices without touching the plot or scientific data."""
    native_width, native_height = _native_dimensions(native_size)
    extension = _format_extension(extension)
    try:
        dpi_value = float(dpi)
    except (TypeError, ValueError) as exc:
        raise ValueError("DPI must be a whole number between 36 and 2400.") from exc
    if (not math.isfinite(dpi_value) or not dpi_value.is_integer()
            or not 36 <= dpi_value <= 2400):
        raise ValueError("DPI must be a whole number between 36 and 2400.")
    dpi = int(dpi_value)
    if native:
        if extension == ".png":
            _validate_raster(max(1, round(native_width)), max(1, round(native_height)))
        return ExportOptions(extension=extension, dpi=dpi)
    if units not in {"mm", "in"}:
        raise ValueError("Physical units must be mm or in.")
    try:
        width_mm = float(width) * (25.4 if units == "in" else 1.0)
        height_mm = float(height) * (25.4 if units == "in" else 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Physical dimensions must be numeric.") from exc
    if not all(math.isfinite(value) and 0.1 <= value <= MAX_PHYSICAL_MM
               for value in (width_mm, height_mm)):
        raise ValueError(
            f"Each physical dimension must be between 0.1 and {MAX_PHYSICAL_MM:g} mm.")
    if extension == ".svg":
        return ExportOptions(extension, 0, 0, width_mm, height_mm, dpi)
    resolution = dpi if extension == ".png" else 72
    result_width = max(1, round(width_mm / 25.4 * resolution))
    result_height = max(1, round(height_mm / 25.4 * resolution))
    if extension == ".png":
        _validate_raster(result_width, result_height)
    return ExportOptions(extension, result_width, result_height,
                         width_mm, height_mm, dpi)


class ExportOptionsDialog(QtWidgets.QDialog):
    """Compact, side-effect-free publication sizing dialog for the current figure."""

    def __init__(self, parent=None, native_size=(800, 600), initial_format="png",
                 initial_width=0, initial_height=0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export current figure")
        self.setMinimumWidth(420)
        self._native_size = _native_dimensions(native_size)
        self._aspect = self._native_size[0] / self._native_size[1]
        if not math.isfinite(self._aspect) or self._aspect <= 0:
            raise ValueError("The plot aspect ratio must be finite and positive.")
        self._updating = False
        self._units = "mm"
        self._selected_options: ExportOptions | None = None
        self._width_mm = 85.0
        self._height_mm = self._width_mm / self._aspect
        extension = _format_extension(initial_format)

        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Export the current plot with its visible styling and view. "
            "These choices do not change the analysis or its numerical results.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QtWidgets.QFormLayout()
        self.format_combo = QtWidgets.QComboBox()
        for label, value in (("PNG · raster image", ".png"),
                             ("PDF · vector document", ".pdf"),
                             ("SVG · vector image", ".svg")):
            self.format_combo.addItem(label, value)
        self.format_combo.setCurrentIndex(self.format_combo.findData(extension))
        form.addRow("Format", self.format_combo)
        self.size_combo = QtWidgets.QComboBox()
        self.size_combo.addItem("Native · current plot size", True)
        self.size_combo.addItem("Physical dimensions", False)
        form.addRow("Size", self.size_combo)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom", None)
        self.preset_combo.addItem("Single column · 85 mm", 85.0)
        self.preset_combo.addItem("Double column · 180 mm", 180.0)
        self.preset_combo.setToolTip(
            "Convenient starting widths, not journal requirements. "
            "Check the target publication's figure specifications.")
        form.addRow("Width preset", self.preset_combo)
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.addItem("Millimetres (mm)", "mm")
        self.units_combo.addItem("Inches (in)", "in")
        form.addRow("Units", self.units_combo)

        dimensions = QtWidgets.QWidget()
        dimension_layout = QtWidgets.QHBoxLayout(dimensions)
        dimension_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setAccessibleName("Figure width")
        self.height_spin.setAccessibleName("Figure height")
        for spin in (self.width_spin, self.height_spin):
            spin.setDecimals(4)
            # Show out-of-budget aspect-linked values honestly; validation below
            # disables export instead of silently clamping a requested dimension.
            spin.setRange(0.0001, 1_000_000_000)
            spin.setKeyboardTracking(False)
        dimension_layout.addWidget(self.width_spin)
        dimension_layout.addWidget(QtWidgets.QLabel("×"))
        dimension_layout.addWidget(self.height_spin)
        form.addRow("Width × height", dimensions)
        self.aspect_check = QtWidgets.QCheckBox("Keep original plot aspect ratio")
        self.aspect_check.setChecked(True)
        form.addRow("", self.aspect_check)

        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(36, 2400)
        self.dpi_spin.setValue(DEFAULT_DPI)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setToolTip(
            "Converts physical size to PNG pixel dimensions and embeds resolution metadata. "
            "It does not resample scientific data or affect vector output.")
        form.addRow("PNG resolution (DPI)", self.dpi_spin)
        layout.addLayout(form)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.summary_label.setMinimumHeight(64)
        layout.addWidget(self.summary_label)
        sizing_note = QtWidgets.QLabel(
            "Sizing scales the whole canvas, including text and line widths. "
            "Check readability at the final printed size.")
        sizing_note.setWordWrap(True)
        layout.addWidget(sizing_note)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Choose destination…")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        initial_width, initial_height = float(initial_width), float(initial_height)
        if not all(math.isfinite(value) and value >= 0
                   for value in (initial_width, initial_height)):
            raise ValueError("Initial export dimensions must be finite and nonnegative.")
        if extension != ".svg" and (initial_width > 0 or initial_height > 0):
            if initial_width <= 0:
                initial_width = initial_height * self._aspect
            if initial_height <= 0:
                initial_height = initial_width / self._aspect
            resolution = DEFAULT_DPI if extension == ".png" else 72
            self._width_mm = initial_width / resolution * 25.4
            self._height_mm = initial_height / resolution * 25.4
            self.aspect_check.setChecked(math.isclose(
                initial_width / initial_height, self._aspect, rel_tol=0.001))
            self.size_combo.setCurrentIndex(1)

        self.format_combo.currentIndexChanged.connect(self._refresh)
        self.size_combo.currentIndexChanged.connect(self._refresh)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.units_combo.currentIndexChanged.connect(self._change_units)
        self.width_spin.valueChanged.connect(self._change_width)
        self.height_spin.valueChanged.connect(self._change_height)
        self.aspect_check.toggled.connect(self._change_aspect)
        self.dpi_spin.valueChanged.connect(self._refresh)
        self._set_dimension_controls()
        self._refresh()

    def _set_dimension_controls(self) -> None:
        self._updating = True
        scale = 25.4 if self._units == "in" else 1.0
        for spin, value in ((self.width_spin, self._width_mm),
                            (self.height_spin, self._height_mm)):
            spin.setSuffix(" " + self._units)
            spin.setValue(value / scale)
        self._updating = False

    def _change_units(self, *_args) -> None:
        self._units = self.units_combo.currentData()
        self._set_dimension_controls()
        self._refresh()

    def _set_custom_preset(self) -> None:
        blocker = QtCore.QSignalBlocker(self.preset_combo)
        self.preset_combo.setCurrentIndex(0)
        del blocker

    def _change_width(self, value: float) -> None:
        if self._updating:
            return
        self._width_mm = value * (25.4 if self._units == "in" else 1.0)
        if self.aspect_check.isChecked():
            self._height_mm = self._width_mm / self._aspect
        self._set_custom_preset()
        self._set_dimension_controls()
        self._refresh()

    def _change_height(self, value: float) -> None:
        if self._updating:
            return
        self._height_mm = value * (25.4 if self._units == "in" else 1.0)
        if self.aspect_check.isChecked():
            self._width_mm = self._height_mm * self._aspect
        self._set_custom_preset()
        self._set_dimension_controls()
        self._refresh()

    def _change_aspect(self, checked: bool) -> None:
        if checked:
            self._height_mm = self._width_mm / self._aspect
            self._set_dimension_controls()
        self._refresh()

    def _apply_preset(self, *_args) -> None:
        width_mm = self.preset_combo.currentData()
        if width_mm is None:
            return
        self._width_mm = width_mm
        if self.aspect_check.isChecked():
            self._height_mm = width_mm / self._aspect
        self.size_combo.setCurrentIndex(1)
        self._set_dimension_controls()
        self._refresh()

    def options(self) -> ExportOptions:
        """Return validated choices without accepting the dialog or writing a file."""
        return resolve_export_options(
            self._native_size, extension=self.format_combo.currentData(),
            native=self.size_combo.currentData(), width=self._width_mm,
            height=self._height_mm, units="mm", dpi=self.dpi_spin.value())

    def _refresh(self, *_args) -> None:
        native = self.size_combo.currentData()
        # Presets are usable directly from Native and switch to physical sizing.
        for widget in (self.units_combo, self.width_spin, self.height_spin,
                       self.aspect_check):
            widget.setEnabled(not native)
        self.dpi_spin.setEnabled(not native and self.format_combo.currentData() == ".png")
        try:
            options = self.options()
        except ValueError as exc:
            self.summary_label.setText("Cannot export: " + str(exc))
            self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
            return
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        if options.native:
            width, height = self._native_size
            if options.extension == ".svg":
                summary = "Native SVG viewport; vector geometry remains scalable. DPI is not applied."
            else:
                unit = "pixels" if options.extension == ".png" else "points"
                summary = f"Native output: approximately {width:g} × {height:g} {unit}."
                if options.extension == ".png":
                    summary += " DPI does not resize native output."
                else:
                    summary += " Vector geometry remains scalable; DPI is not applied."
        else:
            summary = (f"Requested: {options.physical_width_mm:.2f} × "
                       f"{options.physical_height_mm:.2f} mm.\n")
            if options.extension == ".svg":
                summary += "SVG physical dimensions with the original viewBox. DPI is not applied."
            elif options.extension == ".png":
                summary += (f"Output: {options.width:,} × {options.height:,} pixels. "
                            f"Sized at {options.dpi} DPI.")
            else:
                summary += (f"Output: {options.width:,} × {options.height:,} points (72 pt/in). "
                            "Vector geometry remains scalable; DPI is not applied.")
        self.summary_label.setText(summary)

    def accept(self) -> None:
        try:
            self._selected_options = self.options()
        except ValueError:
            self._refresh()
            return
        super().accept()


def choose_export_options(parent, native_size, initial_format="png",
                          initial_width=0, initial_height=0) -> ExportOptions | None:
    """Ask for figure presentation options, or return ``None`` when cancelled."""
    dialog = ExportOptionsDialog(parent, native_size, initial_format,
                                 initial_width, initial_height)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    return dialog._selected_options
