"""A multi-panel renderer exposes every independent title and legend in Format."""
import nanoscope  # noqa: F401 bootstrap before Qt
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.plot_appearance import PlotAppearance
from . import p1_format_rig as F


def test_independent_panel_titles_and_legends_are_offered_with_human_names(qtbot):
    window = F.bare(qtbot, views=("panel_a", "panel_b"))
    window.widget.apply_appearance(PlotAppearance(title="Drawn title"))
    for controller, label in zip(window.controllers, ("Dwell", "Depth")):
        controller.host.independent_title = True
        controller.host.independent_legend = True
        controller.host.panel_label = label
    rows = window.pane._plan()
    labels = {ref: label for ref, label, _parent in rows}
    for panel, label in (("panel_a", "Dwell"), ("panel_b", "Depth")):
        assert labels[ElementRef(panel, "title")] == f"Title ({label})"
        assert labels[ElementRef(panel, "legend")] == f"Legend ({label})"
        assert ElementRef(panel, "axis.bottom") in labels


def test_shared_view_titles_and_legends_remain_single_controls(qtbot):
    window = F.bare(qtbot, views=("a", "b"))
    window.widget.apply_appearance(PlotAppearance(title="Shared title"))
    refs = window.pane.elements()
    assert sum(ref.key == "title" for ref in refs) == 1
    assert sum(ref.key == "legend" for ref in refs) == 1


def test_independent_panel_words_use_the_target_host_not_the_figure_title_box(qtbot):
    window = F.bare(qtbot, views=("panel_a", "panel_b"))
    window.widget.apply_appearance(PlotAppearance(title="Figure words"))
    for controller in window.controllers:
        controller.host.independent_title = True
        controller.host.independent_text = True
    window.pane.show_element(ElementRef("panel_b", "title"))
    panel = window.pane.current_panel()
    assert panel.words.isVisibleTo(panel)
    assert not window.widget.title.isVisibleTo(panel)
