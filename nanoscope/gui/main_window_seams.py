"""Call-time forwarders for the three names that tests patch on ``nanoscope.gui.main_window``.

A method that moved out of ``main_window.py`` into a ``main_window_<part>.py`` module still
calls ``ask_export_path(...)``, ``locked_project_read(...)`` or ``_on_screen(...)`` by its
bare name. Imported from where it is defined, such a name would be bound in the part once,
at import, and a test that replaces ``ask_export_path`` on the module ``nanoscope.gui.main_window``
would no longer reach the call. The part imports the forwarder of the same name from here instead.
The forwarder looks the name up on ``main_window`` each time it is called, so a patch there
bites exactly as it did while the method lived in ``main_window.py``.

This module imports ``main_window`` only inside the forwarders, at call time: ``main_window``
imports the parts and the parts import this module, so an import at the top would be a cycle.
"""
from __future__ import annotations


def ask_export_path(*args, **kwargs):
    from . import main_window

    return main_window.ask_export_path(*args, **kwargs)


def locked_project_read(*args, **kwargs):
    from . import main_window

    return main_window.locked_project_read(*args, **kwargs)


def _on_screen(*args, **kwargs):
    from . import main_window

    return main_window._on_screen(*args, **kwargs)
