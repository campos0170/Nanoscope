"""Native layout items must be invalidated at explicit ownership boundaries.

No allocation lotteries: hold the old item, exercise the actual action, and check
that its wrapper no longer claims a C++ item removed by Qt is alive.
"""
import nanoscope  # noqa: F401
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets
from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from . import p1_hosting as H


@pytest.mark.parametrize('history', [False, True])
@pytest.mark.parametrize('compact', [False, True])
def test_taking_reset_invalidates_only_old_item_and_preserves_real_action(qtbot, history, compact):
    appearance = PlotAppearanceWidget(PlotAppearance(tick_size=19), history=history,
                                     compact=compact)
    qtbot.addWidget(appearance)
    button = appearance.reset_button
    old = button.parentWidget().layout()
    held = old.itemAt(old.indexOf(button))
    other = next(old.itemAt(i) for i in range(old.count())
                 if old.itemAt(i).widget() is not None and old.itemAt(i).widget() is not button)
    other_widget = other.widget()
    destination = QtWidgets.QWidget()
    qtbot.addWidget(destination)
    target = QtWidgets.QVBoxLayout(destination)
    changed = []
    appearance.changed.connect(changed.append)

    try:
        returned = appearance.take_reset_button()
        assert returned is button and shiboken6.isValid(button)
        assert not shiboken6.isValid(held)
        assert old.indexOf(button) == -1
        assert shiboken6.isValid(other) and other.widget() is other_widget
        target.addWidget(button)
        assert appearance.take_reset_button() is button and target.indexOf(button) == 0
        button.click()
        assert appearance.state().tick_size is None and len(changed) == 1
        if history:
            appearance.undo()
            assert appearance.state().tick_size == 19
    finally:
        # Do not let a deliberately held regression witness contaminate later tests.
        for item in (held, other):
            if shiboken6.isValid(item):
                shiboken6.invalidate(item)



@pytest.mark.parametrize('removed_index', [0, 1, 2])
@pytest.mark.parametrize('remove_current', [False, True])
def test_removing_a_tab_invalidates_exact_item_preserving_remaining_order_and_actions(
        qtbot, removed_index, remove_current):
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    keys = ['iv', 'voltage_series', 'noise']
    pages = [H.SyntheticPlotPage(key) for key in keys]
    activations = []
    for key, page in zip(keys, pages):
        explorer.add_page(key, page, activate=lambda key=key: activations.append(key))
    selected_index = removed_index if remove_current else (removed_index + 1) % len(pages)
    explorer.raise_page(keys[selected_index])
    activations.clear()
    page = pages[removed_index]
    stack = page.parentWidget()
    layout = stack.layout()
    held = layout.itemAt(layout.indexOf(page))
    others = [(other, layout.itemAt(layout.indexOf(other))) for other in pages if other is not page]
    remaining = [key for key in keys if key != keys[removed_index]]
    labels = {key: explorer._tabs.tabText(explorer._tabs.indexOf(candidate))
              for key, candidate in zip(keys, pages)}
    synchronous_validity = []
    explorer._tabs.currentChanged.connect(
        lambda _index: synchronous_validity.append(shiboken6.isValid(held)))

    try:
        assert explorer.remove_page(keys[removed_index]) is page
        assert not shiboken6.isValid(held)
        assert shiboken6.isValid(page) and layout.indexOf(page) == -1
        assert list(explorer.page_ids()) == remaining and explorer._tabs.count() == 3
        assert all(shiboken6.isValid(item) and item.widget() is other for other, item in others)
        assert all(shiboken6.isValid(candidate._plot_widget)
                   and shiboken6.isValid(candidate._appearance_widget.reset_button)
                   for candidate in pages)
        assert [explorer._tabs.tabText(explorer._tabs.indexOf(explorer.page(key)))
                for key in remaining] == [labels[key] for key in remaining]
        assert not any(synchronous_validity), 'forget wrapper before native currentChanged'
        assert not activations, 'removal must not recompute the newly exposed page'
        if remove_current:
            assert synchronous_validity, 'removing the selected page must expose another tab'
            successor = remaining[min(removed_index, len(remaining) - 1)]
            assert explorer.current_page_id() == successor
        else:
            assert explorer.current_page_id() == keys[selected_index]
        assert explorer.remove_page(keys[removed_index]) is None
        explorer.show_home()
        activations.clear()
        for key in remaining:
            assert explorer.raise_page(key)
            assert explorer.current_page_id() == key and explorer.toolbar().bound_key() == key
        assert activations == remaining
        page.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert not shiboken6.isValid(page)
    finally:
        for item in [held, *(item for _, item in others)]:
            if shiboken6.isValid(item):
                shiboken6.invalidate(item)
