"""The new displayed unit never changes any seconds-valued caller contract."""
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui.time_controls import SecondsSpinBox


def test_seconds_spin_text_entry_paste_and_signal_are_inverse(qtbot):
    spin = SecondsSpinBox()
    qtbot.addWidget(spin)
    spin.setDecimals(6)
    spin.setRange(-2, 2)
    spin.setSingleStep(.001)
    spin.setValue(.05)
    assert spin.value() == .05 and spin.text().strip() == '50.000 ms'
    values = []
    spin.valueChanged.connect(values.append)
    spin.lineEdit().setText('125.250 ms')
    spin.interpretText()
    assert spin.value() == .12525 and values[-1] == .12525
    spin.stepUp()
    assert spin.value() == .12625
    spin.lineEdit().setText('−5.000 ms')
    spin.interpretText()
    assert spin.value() == -.005
    assert spin.valueFromText('') == -.005


def test_seconds_spin_bounds_precision_sentinels_and_locale(qtbot):
    spin = SecondsSpinBox()
    qtbot.addWidget(spin)
    spin.setDecimals(7)
    spin.setRange(0, .1)
    spin.setValue(.0000001)
    assert spin.text().strip() == '0.0001 ms'
    assert spin.validate('100.0000 ms', 8)[0] == QtGui.QValidator.Acceptable
    assert spin.validate('1000.0000 ms', 9)[0] != QtGui.QValidator.Acceptable
    spin.setSpecialValueText('Off')
    spin.setValue(0)
    assert spin.text() == 'Off'
    spin.setSpecialValueText('')
    spin.setLocale(QtCore.QLocale(QtCore.QLocale.German))
    spin.lineEdit().setText('12,5000 ms')
    spin.interpretText()
    assert spin.value() == .0125
    assert spin.text().endswith(' ms')
    assert isinstance(spin, QtWidgets.QDoubleSpinBox)
