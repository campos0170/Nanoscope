"""Runtime bindings must preserve graphics parents and actually render source data."""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtGui


def test_python_graphics_item_keeps_parent_and_exports_colored_data(qapp, qtbot):
    widget = pg.PlotWidget(background="w")
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    curve = widget.plot([0., 1., 2.], [1., 2., 1.],
                        pen=pg.mkPen("#124a90", width=3),
                        symbol="o", symbolBrush="#124a90", symbolPen="#124a90")
    view = widget.getPlotItem().getViewBox()
    # PySide6 6.9.1 lost the QGraphicsItem* returned by the Python itemChange
    # override. It silently left a top-level item and exported an empty plot.
    assert curve.parentItem() is view.childGroup
    assert curve.getViewBox() is view
    assert curve.curve.parentItem() is curve
    assert curve.scatter.parentItem() is curve
    widget.show()
    qtbot.waitExposed(widget)
    qtbot.wait(20)
    image = ImageExporter(widget.getPlotItem()).export(toBytes=True)
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
    blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
    assert blue.sum() > 50, "The export lost the blue source curve and observations."
