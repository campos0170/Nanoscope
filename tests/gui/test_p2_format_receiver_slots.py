"""The canvas receivers have static Qt signatures and retain sender/retire semantics."""
from types import SimpleNamespace

from PySide6 import QtCore, QtWidgets

from nanoscope.gui.figure_text import TextStyle, TextTarget
from nanoscope.gui.format_inspector import ElementRef, FormatPane, SelectionHub
from nanoscope.gui.plot_appearance import PlotAppearanceWidget
from nanoscope.gui.plot_workflow import FormatSpec


class Controller(QtCore.QObject):
    claimed = QtCore.Signal(str)
    cleared = QtCore.Signal()
    editingChanged = QtCore.Signal()
    plot = None

    def __init__(self, panel, parent):
        super().__init__(parent)
        self.host = SimpleNamespace(pin_view=panel)

    def targets(self):
        return {"title": TextTarget(None, "Automatic", TextStyle(), "title")}

    def target(self, key):
        return self.targets().get(key)

    def view(self):
        return None


def test_canvas_receivers_exist_in_static_metaobject_before_connection():
    # An undecorated Python receiver is added dynamically by connect(); that is the
    # lifetime mechanism this candidate removes, not a suppression of canvas events.
    meta = FormatPane.staticMetaObject
    for signature in ("_on_claimed(QString)", "_on_cleared()", "_on_editing_changed()", "_on_series_selected(QString)"):
        index = meta.indexOfSlot(signature)
        assert index >= meta.methodOffset(), signature
        assert meta.method(index).methodType() == QtCore.QMetaMethod.MethodType.Slot


def test_declared_receivers_keep_sender_identity_and_disconnect_on_retire(qtbot):
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    appearance = PlotAppearanceWidget(parent=owner)
    first, second = Controller("first", owner), Controller("second", owner)
    hub = SelectionHub(owner)
    pane = FormatPane(FormatSpec(appearance, lambda: [first, second]), hub, owner)
    first.claimed.emit("title")
    assert hub.current() == ElementRef("first", "title")
    second.claimed.emit("title")
    assert hub.current() == ElementRef("second", "title")
    qtbot.waitUntil(lambda: not pane._canvas_claim_pending)
    second.cleared.emit()
    assert hub.current() is None

    repositioned = []
    class Toolbar(QtWidgets.QWidget):
        def reposition(self):
            repositioned.append(True)
    second._nanoscope_minibar = Toolbar(owner)
    second.editingChanged.emit()
    assert repositioned == [True]

    pane.retire()
    pane.retire()  # retirement remains idempotent
    # Reconnect only a recorder to prove signals still fire; the retired pane does not.
    received = []
    second.claimed.connect(received.append)
    second.claimed.emit("title")
    second.editingChanged.emit()
    assert received == ["title"]
    assert hub.current() is None
    assert repositioned == [True]


def test_external_renderer_series_receiver_disconnects_when_pane_retires(qtbot):
    from nanoscope.core.figure_series import SeriesDescriptor
    from nanoscope.gui.figure_series import SeriesSelection
    owner=QtWidgets.QWidget();qtbot.addWidget(owner)
    emitter=SeriesSelection(owner);appearance=PlotAppearanceWidget(parent=owner)
    controller=Controller('first',owner);hub=SelectionHub(owner);selected=[]
    pane=FormatPane(FormatSpec(appearance,lambda:[controller],
        series_provider=lambda:[SeriesDescriptor('series.test','Test')],
        series_selected=emitter.selected,series_select=selected.append),hub,owner)
    assert emitter.receivers(QtCore.SIGNAL('selected(QString)'))==1
    emitter.selected.emit('series.test');assert hub.current()==ElementRef('','series.test')
    pane.retire();pane.retire()
    assert emitter.receivers(QtCore.SIGNAL('selected(QString)'))==0
    assert selected[-1] is None
