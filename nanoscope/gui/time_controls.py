"""Millisecond text over a seconds-valued Qt editing API.

Range, step, value(), setValue() and valueChanged remain in seconds. Existing
configuration, analysis, sample-index and history consumers therefore keep their
meaning. Only the human-facing text, parser and validator use milliseconds.
"""
from __future__ import annotations

from PySide6 import QtGui, QtWidgets


class SecondsSpinBox(QtWidgets.QDoubleSpinBox):
    _factor = 1000.
    _decimal_shift = 3

    def __init__(self, parent=None):
        self._display_validator = None
        super().__init__(parent)
        self._display_validator = QtGui.QDoubleValidator(self)
        self._display_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.setSuffix(' ms')

    def _display_decimals(self):
        return max(0, self.decimals() - self._decimal_shift)

    def _numeric_text(self, text):
        text = text.strip()
        prefix, suffix = self.prefix().strip(), self.suffix().strip()
        if prefix and text.startswith(prefix): text = text[len(prefix):].strip()
        if suffix and text.endswith(suffix): text = text[:-len(suffix)].strip()
        return text.replace('−', '-')

    def textFromValue(self, value):  # noqa: N802 - Qt virtual API
        return self.locale().toString(value * self._factor, 'f', self._display_decimals())

    def valueFromText(self, text):  # noqa: N802 - Qt virtual API
        value, valid = self.locale().toDouble(self._numeric_text(text))
        # Invalid/intermediate text is refused by validate; a direct parser call
        # must not turn a blank draft into a fabricated zero.
        return value / self._factor if valid else self.value()

    def validate(self, text, position):
        validator = self._display_validator
        if validator is None:
            validator = QtGui.QDoubleValidator()
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setLocale(self.locale())
        validator.setRange(self.minimum() * self._factor, self.maximum() * self._factor, self._display_decimals())
        if self.specialValueText() and text.strip() == self.specialValueText():
            return QtGui.QValidator.Acceptable, text, position
        numeric = self._numeric_text(text)
        state, _, _ = validator.validate(numeric, min(position, len(numeric)))
        return state, text, position


class HoursSpinBox(SecondsSpinBox):
    """Millisecond presentation over an explicitly hours-valued incubation API."""
    _factor = 3600000.
    _decimal_shift = 5


def rescale_time_text(text, factor, *, intervals=False):
    """Scale finite numeric tokens while retaining blank and invalid draft tokens.

    Interval separators are syntax, never scientific values. An invalid token stays
    invalid; valid neighbours still have their declared unit converted. Applying the
    inverse gives the same draft meaning without executing expressions or inventing 0.
    """
    import re
    import math
    from .numeric_text import parse_strict_number
    parts = re.split(r'([:;])', str(text)) if intervals else [str(text)]
    for index, part in enumerate(parts):
        value = parse_strict_number(part, allow_unicode_minus=True)
        if value is None:
            continue
        leading = part[:len(part) - len(part.lstrip())]
        trailing = part[len(part.rstrip()):]
        scaled = value * factor
        if not math.isfinite(scaled):
            raise ValueError('Time draft is outside the finite display range')
        parts[index] = leading + format(scaled, '.17g') + trailing
    return ''.join(parts)
