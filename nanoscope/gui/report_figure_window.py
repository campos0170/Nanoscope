"""Edit one already-generated report Figure before explicitly exporting its artwork."""
from __future__ import annotations

from copy import deepcopy
import weakref

from PySide6 import QtCore

from ..plot.matplotlib_targets import bind_panel
from .study_figure_window import StudyFigureWindow


class ReportFigureWindow(StudyFigureWindow):
    """Own retained report artists; no builder, fitter or source callback is accepted."""
    def __init__(self, figure, data, resources=None, *, name, provenance, style=None,
                 library_owner=None, parent=None, available_styles=None):
        for index, axis in enumerate(figure.axes):
            if not hasattr(axis, '_nanoscope_panel_id') and not hasattr(axis, '_nanoscope_event_panel_identity'):
                bind_panel(axis, axis.get_gid() or f'report:{name}:panel:{index}',
                           label=f'Panel {index + 1}')
        super().__init__(figure, data, resources or {}, name=name,
            provenance=deepcopy(provenance), style=style, library_owner=library_owner,
            parent=parent, available_styles=available_styles, owner_kind='report')


def open_report_figure(owner, figure, data, *, name, provenance, resources=None, style=None):
    """Transfer a generated Figure to an editor. The caller must not clear it afterward.

    Generation is deliberately outside this API. Data/provenance/resources are detached
    on entry, and the retained artist is cleared once on explicit close. The existing
    batch report export remains a separate MainWindow action.
    """
    for old in owner.findChildren(ReportFigureWindow, options=QtCore.Qt.FindDirectChildrenOnly):
        if old.isHidden() and old._figure is None: old.deleteLater()
    editor = ReportFigureWindow(figure, data.copy(deep=True), deepcopy(resources or {}),
        name=name, provenance=deepcopy(provenance), style=deepcopy(style),
        library_owner=owner, parent=owner)
    # Builders may use pyplot.subplots. The retained editor now owns the Figure;
    # release pyplot's global manager without clearing any retained artists.
    from matplotlib import pyplot as plt
    plt.close(figure)
    held = getattr(owner, '_report_figure_editors', None)
    if held is None:
        held = []; owner._report_figure_editors = held
    held.append(editor)
    # Parent ownership remains after close; only this live-editor index is released.
    ref = weakref.ref(editor)
    editor.closed.connect(lambda: held.remove(ref()) if ref() in held else None)
    editor.show()
    return editor
