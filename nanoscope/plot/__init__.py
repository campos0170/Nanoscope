"""Matplotlib-based publication export for nanoscope.

Kept deliberately separate from the interactive pyqtgraph view (PORT_PLAN D5):
pyqtgraph is the live viewer; matplotlib is the publication-figure export with
accompanying source-data CSV. Importing this package changes nothing about the
caller's matplotlib rcParams — the style is opt-in via
:func:`nanoscope.plot.style.nanoscope_style`.
"""

from __future__ import annotations

from . import figures, style

__all__: list[str] = ["figures", "style"]
