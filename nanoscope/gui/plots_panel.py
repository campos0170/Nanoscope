"""The interactive population-plots window: histograms, scatters, clustering, PCA.

This is the GUI face of :mod:`nanoscope.analysis.populations` (the pure, Qt-free
core, Phase 1) plus the distribution fits in :mod:`nanoscope.analysis.stats`. It
opens from ``Analysis ▸ Distributions…`` and explores the
per-event tables the app already produces -- a detected
:class:`~nanoscope.analysis.events.EventSet` ``.df`` and the manual
:class:`~nanoscope.gui.measure_panel.MeasurePanel` measurements -- without
reaching back into any other panel.

The data flows ONE way in: the owner (MainWindow) snapshots the current tables
and pushes them via :meth:`PlotsWindow.set_data`; the window never queries a
panel or Session for event rows itself. A :attr:`PlotsWindow.refreshRequested`
signal lets the owner re-snapshot on demand (the *Refresh* button), so the plots
are an explicit snapshot the user refreshes, not a live mirror that shifts under
them mid-analysis.

The snapshot carries the detection PROVENANCE alongside each table, and that is
load-bearing rather than decorative. A dwell fit needs the detector's real
resolution ``min_dwell_s``; without it the fit falls back to the shortest
observed dwell, and since the detector can never report a dwell BELOW its own
gate, that substitute floor is always >= the real one -- so tau comes back
biased SHORT by ``min(observed) - min_dwell_s``, and it silently disagrees with
the tau the Fits panel prints for the very same events. The gap is small when
the dwell distribution reaches the gate (measured 0% on a clean synthetic, +1.0%
on a real 22 nm ferritin recording) and large when it does not (measured -21% on
a population whose shortest event was 3x the gate). It is invisible on the plot
either way. :func:`nanoscope.analysis.modules.dwell_fit_inputs` is the
one resolution both windows go through, this window shows the ``t_min`` it used,
and it shades the region below it exactly as
:mod:`nanoscope.plot.figures` does -- the user must see what they cannot see.

Three interactive views over the chosen sources' numeric columns:

* **Histogram** -- one numeric column, adjustable bins, linear/log x, and an
  optional fitted overlay (exponential on a dwell-like column, Gaussian mixture
  on an amplitude/depth-like column) drawn straight from ``populations`` with the
  fitted parameters printed on the plot.
* **Scatter** -- any two numeric columns, per-axis log, coloured either plainly
  or by a clustering.
* **PCA** -- the shared feature set projected to two components, coloured by
  cluster, with the explained variance shown.

OVERLAY: MORE THAN ONE SOURCE AT A TIME
---------------------------------------
The source picker is a CHECKABLE LIST, not a combo: histogram and scatter draw
every checked source, each in its own Okabe-Ito ink with a legend entry naming
it. That is the two-pore and two-segment comparison the window could not express
before. Three rules make an overlay mean something rather than merely look busy:

* **Shared bins + normalization.** Overlaid raw counts are useless the moment the
  recordings differ in length: a 5000-event and a 200-event histogram sit two
  decades apart and the shorter one is a flat line on the axis. So the bin EDGES
  are computed once over the union of every drawn source and reused for all of
  them, and the y quantity is selectable -- ``count``, ``density`` or
  ``probability per bin`` -- defaulting to **density as soon as a second source is
  shown**. Density divides by ``n * bin width``, so two unequal-n sources become
  directly comparable shapes.
* **One fit per source, never a pooled one.** Every source carries its OWN
  detection provenance, therefore its own ``t_min``, its own filter-limited
  exclusion and its own right-truncation state. A per-source fit is resolved
  through :func:`nanoscope.analysis.modules.dwell_fit_inputs` on THAT source's
  ``(frame, provenance)`` pair; one recording's detection floor is never applied
  to another's dwell times, and two sources are never silently pooled into a
  single fit. Each result is reported, labelled by source, in the legend and in
  Details.
* **The colour channel has ONE owner.** Colour means *source* whenever more than
  one source is drawn; it means *cluster* only in the single-source case. The
  cluster-split control is therefore disabled (and says why) while an overlay is
  up, instead of two features quietly fighting over the same ink.

MAY THESE SOURCES BE POOLED FOR THIS COLUMN?
--------------------------------------------
Overlaying is the point of this window, so the window has to answer that, per
metric, out loud. It asks ONE function --
:func:`nanoscope.core.sources.source_comparability` -- which reconciles the two
independent reasons two tables may not be pooled:

* the **current gain**. Uncalibrated by default here, so ``depth_pA`` /
  ``amplitude`` / ``ecd_pA_s`` are pore-specific raw scales across recordings;
* the **baseline_fingerprint**. Every amplitude is a distance from an ESTIMATED
  baseline, and the Preprocess probe correctly proposes a DIFFERENT window per
  pore, so ``depth_pA``, ``frac_depth`` and every other amplitude measured under
  two windows are two different measurements. ``dwell_s`` and the counts are not:
  the Schmitt edges are set by the noise ladder, not the baseline level.

That is the same rule ``Analysis ▸ Voltage dependence`` enforces, called from the
same place -- there is no copy of it in this file, which is the point. Two guards,
ONE verdict: ``frac_depth`` survives a gain disagreement but NOT a baseline one, so
the gain advice stops recommending it exactly when the baselines disagree (it used
to recommend it always, walking the user off one cliff onto another).

The verdict lands in Details AND in the plot title, and the title copy survives a
user-typed title, because the title is the only thing ``Save image`` keeps. The
overlay is still DRAWN: this is an exploration surface, and putting two
distributions side by side to SEE that they differ is exactly what ticking two
sources is for. Warn and draw -- never refuse, never silently pool.

Each drawn detection's OWN caveats (``AnalysisResult.warnings``) are rendered here
too, beside the fit's and the clustering's.

Clustering (GMM / k-means) and PCA share one feature multi-select and operate on
the PRIMARY (first checked) source; both are StandardScaler'd inside
``populations``. Rendering is pyqtgraph so it inherits the app's dark theme (set
globally in ``gui.app``); the static publication figures in
:mod:`nanoscope.plot.figures` are a separate export path, not replaced here.
"""

from __future__ import annotations

import copy
import hashlib
import html as _html
import json
import math
import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as _dc_fields, replace as _dc_replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis import event_shape, events, modules, populations, scoring, stats
from ..analysis.distribution_summary import kde_curve
from ..analysis._compat import trapezoid as _trapezoid
from ..analysis.cancel import OperationCancelled, cancel_scope, check_cancelled
from ..analysis.remeasure import REMEASURED_COLUMNS
from ..core import clock
from ..core.analysis import AnalysisResult
from ..core.session import Session
from ..core.sources import (
    PlotSource,
    SourceVerdict,
    amplitude_comparability,
    amplitude_units_label,
    disambiguate_labels,
    source_colour,
    source_comparability,
)
# AT MODULE SCOPE, AND NOT INSIDE ``save_image``, ON PURPOSE. Importing this module
# is what installs the closepath workaround (``pg_svg.install()`` runs at import), and
# pyqtgraph's OWN right-click "Export… ▸ SVG" dialog on this plot never passes through
# any code of ours -- so an import deferred to our export path would leave that menu
# raising ``ValueError: not enough values to unpack`` exactly as it does today. One
# import anywhere in the process fixes the exporter for the whole process. See the
# "WHERE THE IMPORT GOES" section of nanoscope/plot/pg_svg.py.
from ..plot import pg_svg
from ..plot.style import OKABE_ITO, PALETTE
from .analysis_catalog import ANALYSIS_LABELS
from .axes import ReadableLogAxis
from .categorical import category_brushes, category_colour, category_counts, legend_swatch
from .combos import choose_combo_item
from .column_checklist import ColumnChecklist
from .quantity_selectors import ScientificColumnCombo
from .export_paths import ask_export_path
from .figure_text import (label_args, GATE, HIDDEN_TEXT, AxisState, TextStyle, TextTarget, css_args,
                          figure_text_for, plain_text, rich_text, shown_text,
                          valid_element_key)
from .fit_label import (FIT_FIELDS, FitLabelItem, FitLabelState, FitReadout, automatic_template,
                        format_fit_label)
from .font_picker import (DEFAULT_ROW_LABEL, FontPicker, figure_family, figure_qfont,
                          html_font_family)
from .key_sequences import redo_sequences, undo_sequences
from .plot_appearance import (PlotAppearance, PlotAppearanceWidget, apply_legend_family,
                              preset_store)
from .plot_interaction import (image_value, install_plot_interaction, keep_plot_history,
                               reset_plot_history, view_button, window_readout)
from .plot_lasso import PlotLasso, RowIdentityIndex, toggled_rows, unique_rows
from .plot_workflow import (DEFAULT_REVEAL, ExportEntry, FormatSpec, HistoryState,
                            InspectorGroup, PageAdapter, PendingState)
from .plot_export import (
    apply_png_resolution, apply_svg_physical_size, choose_live_plot_export,
)
from .quantity_reset import announce, reset_sentence, stands, typed_words, words_cleared
from .worker_thread import (
    DEFAULT_JOIN_MS,
    JOB_NONE,
    retire_worker_thread,
    start_worker_thread,
)

__all__ = [
    "PlotsWindow",
    "PLOT_CONFIG_FORMAT",
    "PLOT_CONFIG_VERSION",
    "PLOT_CONFIG_MIN_READ_VERSION",
    "PlotConfigError",
    "PlotConfigVersionError",
    "plot_config_to_json",
    "plot_config_from_json",
    "read_plot_config",
    "write_plot_config",
]


class _NoteWorker(QtCore.QObject):
    """Runs the panel's deferred fit work OFF the GUI thread. One batch, then done.

    THE JOB THIS EXISTS FOR. Three things beside a drawn fit are expensive and none
    of them may sit between a curation click and the repaint: the bootstrap CI of
    the drawn fit itself, the cross-family BIC table, and the "another component
    count fits better" ladder. Deferring them to a 400 ms timer -- the previous
    attempt -- did not remove the freeze, it moved it. Measured through a real
    MainWindow on the owner's -700 mV apo run (n = 984), one curation click then
    sitting still, timing both the settle callback and the worst gap between ticks of
    a 10 ms probe timer (i.e. how long a paint-time callback was kept waiting):

        exponential        368 ms blocked   worst gap  383 ms
        gaussian-mixture   450                         458
        gamma (peaked)     694                         707
        exp + gamma        991                        1007
        inverse-gaussian  1266                        1283
        (none)               --                         35   <- the control

    A timer is still the GUI thread. A worker is not: the same measurement with this
    class in place is 0.1 ms of settle and a 33-38 ms worst gap in every family.

    THE RULES IT IS HELD TO, all of which are this codebase's, not new ones:

    * ``domain=JOB_NONE`` at the run gate. An advisory note is not an analysis and
      must never refuse -- or be refused by -- a detection, a batch pass or a
      re-measure. It writes into no ``ResultStore``; it writes into a redraw cache.
    * Cancellation is the ambient token (:mod:`nanoscope.analysis.cancel`), installed
      here and polled inside ``fit_dwell_peaked``'s bootstrap loop, so a superseded
      job stops inside its own replicate loop instead of running to completion.
    * ``cancel()`` is a PLAIN METHOD setting a ``threading.Event``, never a ``@Slot``:
      a slot on an object living in this thread would queue behind ``run()`` and
      arrive after the work it was meant to stop. Same reasoning as
      ``AnalysisWorker.cancel``.
    * Never terminated, never waited on without a bound -- the panel retires it
      through :func:`nanoscope.gui.worker_thread.retire_worker_thread`.

    Everything it runs is a plain callable built ON THE GUI THREAD out of numpy
    arrays and numbers. It touches no widget: the one place that used to read a
    widget from inside a deferred computation (``_component_range`` off the fit-k
    spin) now has its range captured at the call site, because "read a QWidget from
    a worker" is undefined behaviour that happens to work until it does not.
    """

    #: Emitted once, with ``[(key, value), ...]`` for the jobs that COMPLETED.
    #: A cancelled job contributes nothing -- a superseded interval must not be
    #: cached, or the next redraw would paint it beside data it is not about.
    done = QtCore.Signal(object)

    def __init__(self, jobs: list[tuple]) -> None:
        super().__init__()
        self._jobs = list(jobs)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Plain method, called from the GUI thread. See the class docstring."""
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        out: list[tuple] = []
        try:
            with cancel_scope(self._cancel.is_set):
                for key, compute in self._jobs:
                    if self._cancel.is_set():
                        break
                    try:
                        out.append((key, compute()))
                    except OperationCancelled:
                        break
                    except Exception:
                        # An advisory must never break the plot -- and its failure
                        # must be REMEMBERED, or every redraw re-arms the timer for
                        # work that cannot succeed and the panel never settles.
                        out.append((key, _NOTE_FAILED))
        finally:
            self.done.emit(out)


#: What a note that raised is remembered as. A sentinel rather than ``""`` because
#: the failure value depends on what the key is for (a rivals key caches ``[]``, a
#: text key caches ``""``), and that decision belongs to the panel, on the GUI
#: thread, not to the worker.
_NOTE_FAILED = object()

# THE WINDOW IS ASSEMBLED FROM PARTS. The constants and pure helpers the window and its
# parts share live in ``plots_constants``, and each group of ``PlotsWindow``'s methods
# lives in a ``plots_<part>_mixin`` module (plain-Python mixins, listed before
# ``QMainWindow`` in the class bases below). None of those modules imports this one.
# Everything that moved is imported back here, so ``nanoscope.gui.plots_panel.X`` still
# resolves for every name: the saved-file types, the tests, ``figure_adapter_*`` and
# ``figure_recipes`` ask for them here, and a name a test patches on this module is read
# from this module.
from .plots_constants import (  # noqa: F401  (moved verbatim; still importable from here)
    _DWELL_COLUMNS,
    _ORIGIN_COLUMNS,
    _DEPTH_COLUMNS,
    HISTOGRAM,
    SCATTER,
    PCA,
    KDE,
    HEATMAP,
    _PLOT_TYPES,
    _SHOWS_PLOT,
    _HEATMAP_COLORMAPS,
    _FIT_NONE,
    _FIT_EXP,
    _FIT_GMM,
    _FIT_GAMMA,
    _FIT_INVGAUSS,
    _FIT_EXP_GAMMA,
    _MULTI_COMPONENT_FITS,
    _COLOUR_NONE,
    _COLOUR_CLUSTER,
    _NORM_COUNT,
    _NORM_DENSITY,
    _NORM_PROB,
    _NORMS,
    _NORM_PHRASE,
    _DURATION_COLUMNS,
    _TIME_COLUMNS,
    _TIME_UNITS,
    _SCHEMA_UNITS,
    _DEFAULT_TIME_UNIT,
    _LEGEND_DEFAULT_OFFSET,
    _CONTROLS_MIN_PX,
    _CONTROLS_SLACK_PX,
    _COMBO_CHARS,
    _PLOT_PROMPT,
    _SCATTER_FIT_PARAMS,
    _SFIT_NONE,
    _SFIT_LINEAR,
    _SFIT_POWER,
    _METHOD_GMM,
    _METHOD_KMEANS,
    _METHOD_IDS,
    _K_AUTO,
    _K_FIXED,
    _PREFER_DWELL,
    _PREFER_AMP,
    _PREFER_FEATURES,
    _IDENTIFIER_FEATURES,
    _DATA_HEX,
    _FIT_HEX,
    _WARN_HEX,
    _EXCLUDED_HEX,
    _DROPPED_HEX,
    _MSG_HEX,
    _MSG_LIGHT_HEX,
    _HIST_FILL_ALPHA,
    _POINT_ALPHA,
    _BG_DARK,
    _FG_DARK,
    _BG_LIGHT,
    _FG_LIGHT,
    FrozenMap,
    _NO_ENTRIES,
    _deep_frozen,
    _thawed,
    FIT_LABEL_KEY,
    _TEXT_FIELD_OF,
    LIMIT_LABEL_KEY,
    _LIMIT_LABEL_AUTO,
    _text_kept_in_styles,
    _LEGEND_ROW_KINDS,
    _legend_style_key,
    _legend_entry_id,
    _legend_styles,
    _style_text,
    _unit_factor,
    _display_factor,
    _rescaled,
    _plot_column_quantity,
    _plot_density_quantity,
    _plot_quantity_bases,
    migrate_plot_quantity,
    _HISTORY_DEPTH,
    _ORIGINS_KEPT,
    _drop_oldest,
    _HISTORY_COALESCE_MS,
    _PARAM_SYMBOLS,
    _param_symbol,
    _KS_REJECTED_TAG,
    _FitSummary,
    _LegendText,
    _FitLegendText,
    _LimitLabel,
    _narrow_combo,
    _FONT_DEFAULT_LABEL,
    _swatch_text_color,
    _cluster_color,
    _numeric_columns,
    _detection_key,
    _normalize_sources,
    _prefer,
    _column_quantity,
    _choose_combo_item,
)
from .plots_fit_mixin import _PlotsFitMixin
from .plots_annotation_mixin import _PlotsAnnotationMixin
from .plots_sources_mixin import _PlotsSourcesMixin
from .plots_style_mixin import _PlotsStyleMixin
from .plots_notes_mixin import _PlotsNotesMixin
from .plots_views_mixin import _PlotsViewsMixin

#: Normalization names written by builds BEFORE the rename above, mapped to the current
#: spelling. THIS TABLE IS NOT OPTIONAL AND THE RENAME MUST NOT SHIP WITHOUT IT.
#:
#: The mode's NAME is the persisted value: ``export_config_json`` writes
#: ``"normalization": "density (area = 1)"`` into every v1 sidecar (measured on a file this
#: build wrote), and ``_store_templates`` writes the same payload into QSettings.
#: ``_set_config`` installs it with ``QComboBox.setCurrentText``, which on a non-editable
#: combo is a SILENT no-op for text the combo does not carry. Driven, with a normalization
#: the combo did not offer: ``import_config_json`` returned **True**, the combo stayed on
#: "count", ``config().histogram.normalization`` read "count", ``details_text()`` said
#: nothing, and ``_template_combo`` dropped to "- manual -" because
#: :meth:`PlotsWindow._sync_template_selection` compares configs and the two spellings
#: never match. A user would watch a density figure become a count figure, and their own
#: template's name vanish as they picked it, with no message anywhere. (That silence is
#: itself closed now -- a name this build cannot offer is RECORDED by the one writer and
#: stated in Details, :class:`_NormRefusal` -- but a legacy spelling must not reach it:
#: refusing yesterday's word for a mode that still exists would be a wrong answer said
#: loudly, which is why this table is not optional.)
#:
#: Applied IN THE VALUE (:meth:`HistogramConfig.__post_init__`), which is the one point
#: every door passes through -- the file reader, the template store, an undo entry and a
#: caller's own ``dataclasses.replace`` all end up constructing a ``HistogramConfig``. It
#: was applied at the READ boundary alone to begin with, and the two in-process doors
#: (``apply_config``, ``apply_template``) were measured silently drawing counts under a
#: request for a density; that method's comment carries the numbers. Doing it once means
#: ``config()`` reports ONE spelling everywhere, which is the equality undo, the template
#: comparison and ``PlotConfig.__eq__`` all rest on.
#:
#: READ THROUGH :func:`_canonical_norm` AND NOWHERE ELSE.
#: :meth:`PlotsWindow.set_normalization` needs the same answer for a bare string on its
#: way to a widget -- there is no ``HistogramConfig`` in that path to have canonicalised
#: it -- and for a while it looked the name up itself. Two readings of one table is not a
#: second migration point until they disagree, and they did: one stripped and one did
#: not, so ``" density (area = 1) "`` migrated at the setter and not in the value. Both
#: now ask the function, and ``test_the_migration_table_is_read_in_exactly_one_place``
#: fails if a third reader appears.
#:
#: THE RENAME DID BUMP ``PLOT_CONFIG_VERSION`` (to 2), for the reason spelled out at that
#: constant: this table fixes what THIS build reads, and can do nothing about what an
#: older build does with a file that says ``density``. ``PLOT_CONFIG_MIN_READ_VERSION``
#: stays 1, so every file ever written still opens here -- which is the promise that
#: constant makes.
_LEGACY_NORMS: dict[str, str] = {"density (area = 1)": _NORM_DENSITY}


def _canonical_norm(mode):
    """``mode`` with a pre-S6-03 spelling rewritten into today's. THE ONE READING of
    :data:`_LEGACY_NORMS`, because the two readers had already drifted.

    THE DRIFT THIS REMOVES, MEASURED. :meth:`PlotsWindow.set_normalization` looked the
    name up as ``_LEGACY_NORMS.get(str(mode).strip(), mode)`` and
    :meth:`HistogramConfig.__post_init__` looked it up as
    ``_LEGACY_NORMS.get(self.normalization)`` -- one stripped, one did not -- so
    ``" density (area = 1) "`` migrated through the setter and NOT through the value:
    ``HistogramConfig(normalization=" density (area = 1) ").normalization`` came back
    unchanged while ``set_normalization(" density (area = 1) ")`` landed on ``density``.
    The padded spelling is reachable only from a hand-edited sidecar or QSettings entry
    -- this app never writes padding -- so it was a consistency defect rather than a live
    one, but "one table, two readings, and they disagree" is the exact shape of the
    defect the whole of S6-03 exists to remove. It is now read once, here, so NO LEGACY
    SPELLING can migrate through one door and not the other: measured on this build,
    ``" density (area = 1) "`` and ``"density (area = 1)"`` both come back ``density``
    from the value and from the setter.

    WHAT THIS DOES *NOT* MAKE AGREE, said plainly because the earlier draft of this
    paragraph claimed it did. The two doors still answer differently for a CURRENT
    spelling that needs tidying, because that is a different mechanism and it never
    lived here: :meth:`PlotsWindow.set_normalization` resolves through
    ``combos.choose_combo_item``, which strips, case-folds and takes a unique
    case-insensitive prefix, while the install door :meth:`PlotsWindow._set_config`
    matches with ``QComboBox.findText``, which is exact and case-sensitive. Measured on
    the owner's 982-event file, same string, both doors: ``" density "``, ``"DENSITY"``,
    ``"Density"`` and ``"dens"`` all draw density through the setter (first bars
    ``[1.6585, 6.6341, 4.9756]``) and are REFUSED by ``apply_config``, which goes on
    drawing count ``[1.0, 4.0, 3.0]`` and says so in Details. That asymmetry is
    deliberate and is stated where it is decided
    (:meth:`PlotsWindow.set_normalization`: a bare string is a caller TYPING, so leniency
    and a raise; a whole ``PlotConfig`` from a file is not typing, so an exact match and
    a spoken refusal) -- what this function fixes is one table with two readings, not
    two doors with two matching rules.

    STRIPPED FOR THE LOOKUP, NEVER IN THE ANSWER. A name that is not in the table comes
    back EXACTLY as it was given, padding and all: this function migrates, it does not
    tidy. That is what keeps both callers' refusals quotable -- ``set_normalization``
    raises naming what it was handed, and a foreign name reaches the window's own record
    (:class:`_NormRefusal`) spelled the way the user's file spelled it, which is the
    string they have to go and search for.
    """
    return _LEGACY_NORMS.get(str(mode).strip(), mode)


class _NormRefusal:
    """The y quantity an incoming configuration named that this build cannot offer.

    THE SILENCE THIS REMOVES. :meth:`PlotsWindow._set_config` installs the mode with
    ``QComboBox.setCurrentText``, which on a non-editable combo is a SILENT no-op for
    text the combo has not got. Measured on the owner's 982-event file, recalling a
    stored template whose normalization reads ``density (1/s)`` -- a name no build here
    offers: the bins DID come across (47), the y axis read ``count``, Details read
    ``y = count``, ``import_config_json`` returned True, and the number of modal dialogs
    and of new visible labels was 0 and 0. The recall half-landed and nothing anywhere
    said so.

    WHY A RECORD RATHER THAN A MESSAGE AT THE DOOR. Details is rewritten in full by every
    draw, and THE DOOR IS ITSELF A DRAW: :meth:`PlotsWindow.apply_config` is
    ``_set_config(cfg, draw=True)``, so a sentence printed at install time is wiped by
    the same call that printed it -- measured, a marker written into Details from inside
    the install is gone by the time ``apply_config`` returns, replaced by the new
    figure's own header. A modal for a name in somebody's file is the wrong weight
    entirely. This is the shape
    :class:`nanoscope.gui.event_gallery._RefusalLedger` already holds for the events a
    curation gesture declined: the RECORD stands, and every rebuild of the surface
    re-derives its sentence from it, so the disclosure is a statement about the window
    rather than an artefact of one gesture.

    WHAT RE-DECIDES IT, in one place so it cannot be half-applied:

    * AN INSTALL re-decides it, always. :meth:`PlotsWindow._set_config` is the one
      writer; every door -- a template, an imported sidecar, an undo entry, a script's
      own ``apply_config`` -- ends up there, and it either honours the name (record
      cleared) or cannot (record set). There is no door that can install a y quantity
      without passing this.
    * A MOVE OF THE COMBO THE WRITER DID NOT MAKE clears it: a hand pick, or
      :meth:`PlotsWindow.set_normalization`. The user has re-decided the y quantity, so
      what an install could not honour is no longer the reason for what is on screen.
    * THE AUTOMATIC DEFAULT DOES NOT (:meth:`PlotsWindow._sync_norm_default` swings the
      combo to density when a second source is drawn). Nobody re-decided anything there,
      and the refusal is still the true account of why the requested quantity is absent.

    AND IT IS SPOKEN ON EVERY PAGE, IN EVERY STATE, not only the histogram's and not only
    when the window has something to draw, because the record belongs to the CONFIGURATION
    and a configuration names a y quantity whatever its plot type and whatever the window
    can currently show. :meth:`PlotsWindow._norm_refusal_notes` is the one reading of THIS
    record -- see there for what a scatter config refused in silence cost, and for why the
    sentence has two forms -- and :meth:`PlotsWindow._incoming_notes` is the one surface
    that puts it on a page, asked by the four draw paths AND by :meth:`PlotsWindow._message`,
    the funnel every early return goes through. Without that last one the PCA page said
    nothing on every import it ever receives, because a configuration cannot carry a PCA
    result and the page returns early until Run PCA has been pressed.

    ONLY THE NAME THAT WAS ASKED FOR IS STORED. What is DRAWN is read back off the combo
    each time the sentence is built (:meth:`note`), so the automatic default moving the
    combo underneath cannot leave the disclosure quoting a mode that is no longer on
    screen -- the same reason ``_RefusalLedger`` recounts from the frame instead of
    storing its counts.

    AND IT STANDS FOR AS LONG AS THE STATE IT DESCRIBES IS ON SCREEN, WHICHEVER WRITER
    PUT IT THERE (S7-FIX1 / S7-SK1). "An install re-decides it, always" was the rule
    above, and it had a hole: an install that merely puts THIS WINDOW'S OWN STATE back
    is not a re-decision of anything. Measured on the owner's 982-event table with a
    sidecar naming ``density (1/s)`` and 10**9 bins: after the import the record read
    ``density (1/s)``, the combo ``count``, the bins 500; one ``set_bins(31)`` and one
    Undo later the window was bit-for-bit back in the imported state (``config()`` equal
    to the post-import value) and the record was ``""`` -- because
    ``undo_config -> _install_history -> _set_config`` re-decided it from a history
    entry whose normalization is the legal name the window DREW. A Time-unit change
    reached the same clearing through ``_rebase -> _set_config`` without touching Undo
    at all (the S6-03 density note is what tells the user to change it), and the re-save
    over the colleague's file then wrote ``count`` and 500 with 0 dialogs -- the round-3
    silence, one Ctrl+Z or one combo away.

    THE RULE NOW: the record is tagged with the ORIGIN of the configuration it was
    decided for (see :meth:`_ConfigHistory.open_origin`), and it is spoken while the
    configuration on screen descends from that origin. An import and a template recall
    and a script's ``apply_config`` each open a NEW origin, so they hide what an earlier
    install recorded (and an import re-decides it); an undo, a redo and a rebase move the
    window along its own lineage and record nothing, so a state the file produced -- or
    one reached from it by hand -- keeps its sentence, and an undo that steps back PAST
    the import (or a template recall) hides it, because the pre-import state is not one
    the file produced. A redo that lands back on the file's state shows it again --
    "whichever writer re-installed it". The hand move of the combo clears the record
    outright, as before: the user has re-decided the y quantity, and nothing that
    re-installs the OLD state re-decides it back. ``current_origin`` is the one resolver
    of "what is on screen now" -- the history's -- so this record cannot disagree with
    :class:`_ConfigLosses` about which install is showing.

    ONE RECORD PER ORIGIN, NOT ONE RECORD (A1-N4). The rule above says a sentence
    stands while the state on screen descends from the install that produced it, and
    a single slot cannot keep that promise for the EARLIER of two imports: measured on
    the owner's 982-event table, import A (``density (1/s)``, 10**9 bins), one bins
    change, import B (``density (1/ms)``, 10**8) -- then Undo twice. The window was
    bit-for-bit on A's imported state (``config()`` equal to the post-import value,
    the history's origin A's) and both of A's sentences were gone, because B's import
    had re-taken the one slot under B's origin; Export JSON… over A then wrote
    ``count`` and 500 with 0 dialogs -- S7-FIX1's silence, one import further along.
    So the record is a small dict keyed by origin: an install records under the origin
    it opened and leaves every other lineage's record alone, and what is spoken is the
    entry for the origin on screen. :meth:`clear` (the hand move of the combo)
    forgets the ON-SCREEN lineage's record only -- the user re-decided the y quantity
    for the state in front of them, not for a state an undo can still reach --
    and :meth:`forget` (``reset_state``) empties the lot, because the history that
    could reach any of them is being emptied too. Bounded by :data:`_ORIGINS_KEPT`.
    """

    __slots__ = ("_records", "_current_origin")

    def __init__(self, current_origin: Callable[[], int] | None = None) -> None:
        #: origin -> the refused name, in the order the records were taken; the last
        #: entry is the newest, which is what a record with no resolver speaks. NAMED
        #: `_records`, NOT `_asked`: :attr:`asked` is the ONE reader of what is refused
        #: on screen, and when this dict first went in under the string's old name the
        #: sentence below interpolated the dict itself -- Details read
        #: "“{2: ' density '}” is not a y quantity this build offers" (measured), where
        #: the user has to be able to search their own file for what is quoted.
        self._records: dict[int, str] = {}
        #: Answers "which origin is on screen"; None means "the newest record", for a
        #: record that lives outside a window.
        self._current_origin = current_origin

    def _shown(self) -> int | None:
        """The origin whose record is spoken: the one on screen, or -- for a record
        outside a window -- the newest taken (None when none has been)."""
        if self._current_origin is not None:
            return int(self._current_origin())
        return next(reversed(self._records), None)

    def clear(self) -> None:
        """Nothing is refused for the state on screen: the y quantity has been
        re-decided THERE. Records of other lineages stand, so an undo back onto an
        earlier import's state still finds that import's sentence."""
        if self._current_origin is None:
            self._records.clear()
            return
        self._records.pop(self._shown(), None)

    def forget(self) -> None:
        """Every lineage's record, gone -- for ``reset_state``, where the history that
        could reach any of them is emptied in the same breath."""
        self._records.clear()

    def record(self, asked, origin: int = 0) -> None:
        """``asked`` is a y quantity this build does not carry. Kept verbatim -- the
        user has to be able to search their own file for it. ``origin`` is the lineage
        the install opened (:meth:`_ConfigHistory.open_origin`); the record is spoken
        while the window's state descends from it."""
        key = int(origin)
        self._records.pop(key, None)          # re-taken: newest again
        self._records[key] = str(asked)
        _drop_oldest(self._records)

    @property
    def asked(self) -> str:
        """The refused name, or ``""`` when there is nothing to disclose -- including
        when what was refused belongs to an install that is no longer on screen."""
        shown = self._shown()
        return self._records.get(shown, "") if shown is not None else ""

    @property
    def origin(self) -> int:
        """The lineage of the refusal spoken NOW (0 = none is)."""
        return int(self._shown() or 0) if self.asked else 0

    def note(self, shown: str, *, on_axis: bool = True) -> str:
        """The sentence for the Details notes list, or ``""`` when nothing is refused.

        NAMES WHAT IS ON OFFER, because that is the difference between a refusal and a
        shrug -- :meth:`PlotsWindow.set_normalization` raises with the same list for the
        same reason, and it is the one line that tells a user whether they are looking at
        a typo, a hand-edit, or a file from a build that is newer than theirs.

        ``on_axis`` IS "IS THE PAGE IN FRONT OF THE USER THE HISTOGRAM", and it exists
        because the record is plot-type agnostic while this sentence's middle clause is
        not. A configuration names a y quantity whatever its plot type, so a scatter/KDE/
        PCA config can be refused exactly as a histogram one is (measured: imported with
        ``plot_type`` "Scatter", the record reads ``density (1/s)`` on all four pages) --
        but "y is drawn as count" is FALSE on a scatter, whose y is a column. Saying it
        there would trade one silence for one wrong sentence, so the clause names the
        histogram's setting instead and says the page is not using it.
        """
        asked = self.asked          # the one reader: the name refused ON SCREEN, verbatim
        if not asked:
            return ""
        landed = (f"y is drawn as “{shown}”" if on_axis else
                  f"the histogram's y quantity stays “{shown}”, which this page does "
                  f"not use")
        return (f"“{asked}” is not a y quantity this build offers, so that part of "
                f"the configuration did not land — {landed}. "
                f"Offered here: {', '.join(_NORMS)}.")


#: How many lost settings the sentence NAMES before it stops naming and counts the rest.
#:
#: Eight, measured rather than picked. The worst sidecar anyone has actually produced --
#: the seven-loss one in :class:`_ConfigLosses`'s docstring -- names ALL seven at this cap,
#: in a 410-character sentence, and the Details pane still needs no scrolling with it AND
#: the y-quantity sentence in it: measured on a 1200x780 window, the pane is 803x180 px and
#: its vertical scrollbar range is 0..0, against a control of the same text plus 40 filler
#: lines, which gives 0..37. Pushed to twenty losses the cap bites ("and 12 more"), the
#: sentence is 421 characters and the range is still 0..0. (Both counts include the file
#: NAME, which the sentence leads with -- "colleague.json" in the measured runs.)
#:
#: THE COUNT IS THE NUMBER A USER ACTS ON -- "this file was not fully understood, do not
#: save over it" -- and it is stated whole whatever the cap does, so the cap can shorten
#: the evidence and never the claim.
_LOSSES_NAMED: int = 8

#: How much of one lost VALUE is quoted, before an ellipsis. Long enough to search a file
#: for -- the reason :class:`_NormRefusal` keeps its refused name verbatim -- and bounded
#: so that one very long value (a pasted source key, a path) cannot crowd out the other
#: five names or the count. Every value in the measured sidecar is well under it, so this
#: is a ceiling on the worst case rather than something the ordinary case meets.
_LOSS_VALUE_CHARS: int = 60


class _ConfigLosses:
    """What an imported sidecar named that this window did not take, and did not say.

    THE SILENCE THIS REMOVES, driven on the owner's 982-event table with a v2 sidecar
    that names a foreign y quantity plus SIX other things (a ``fit.kind`` of "stretched
    exponential", a ``histogram.column`` of "conductance_nS", a ``scatter.colour_by`` of
    "pore_id", a ``selection.source_keys`` naming a run this session never loaded, a
    ``style.halo_width_px`` and a whole top-level block this build has never heard of):
    ``import_config_json`` returned True, 0 dialogs appeared, and a re-save wrote
    ``none``, ``dwell_s``, ``None``, one source key and no halo over the sender's file.
    Exactly ONE of those seven -- the y quantity -- was disclosed anywhere, by
    :class:`_NormRefusal`, and only on the pages that draw. A user who opens a
    colleague's file, changes one thing and saves it back had quietly discarded parts of
    their colleague's file.

    THE EXPORT IS NOT THE BUG AND IS NOT CHANGED. :meth:`PlotsWindow.export_config_json`
    writes :meth:`PlotsWindow.config`, the settings actually on screen -- "the same
    gesture ``save_image`` is, on two halves of one figure" -- and a sidecar that carried
    forward a fit this build cannot fit or a column these events do not have would be a
    file describing a picture that does not exist. What was wrong was the SILENCE, so
    this discloses and the contract stands, exactly as S6-03 decided for the y quantity.

    WHAT COUNTS AS A LOSS, in two parts, and why nothing this build understood completely
    is counted. :func:`_config_round_trip_losses` pivots on the payload as this build
    DECODES it:

    * A key in the file with no field on the pivot -- the format has nowhere to put it.
    * A field whose value on the pivot is not the value :meth:`PlotsWindow.config`
      reports after the install -- the window would not, or could not, take it.

    Pivoting there is what keeps the one MIGRATION this format has out of the count: a v1
    file's ``density (area = 1)`` is canonicalised by ``HistogramConfig.__post_init__``
    before the pivot exists, so both sides of the value comparison already read
    ``density`` and the rename is not reported as a loss. Measured, on all four pages: a
    v1 legacy sidecar gives 0 losses, and so does this window's own config exported and
    imported straight back. The envelope keys (``version``, ``format``, ``app_version``)
    are likewise identical on both sides of the pivot and cannot leak in -- a v1 file
    reports 0 even though its ``version`` is 1 and the re-save writes 2.

    THAT IS A CLAIM ABOUT THE CASES DRIVEN, NOT A PROOF. What was driven, and reported 0
    every time: the window's own config round-tripped on each of the four pages, the same
    file with the v1 rename in it on each of the four pages, a two-source window in which
    :meth:`PlotsWindow._sync_norm_default` has swung the combo on its own (ticking the
    second source moves it count -> density with ``norm_user_set`` still False) both ways
    round -- that config exported and re-imported, and a one-source file saying ``count``
    imported onto it -- and the same file imported twice in a row. What
    reports honestly rather than 0: a window with NO data loaded reports 7 (it can honour
    no column and no source, which is exactly true), and a file naming a plot type this
    build has not got reports 1. A case that fires on a file this build understood
    completely would be worth less than no disclosure at all, so that is the direction to
    keep testing in.

    THE IMPORT IS THE HONEST MOMENT, AND THE ONLY DOOR THAT HAS A FILE. A saved TEMPLATE
    goes through :meth:`PlotsWindow.apply_template`, whose whole contract is that a
    template carries PART of a configuration (:meth:`PlotConfig.with_template` owns that
    seam), so diffing what a template holds against what the window then reads would
    report the seam itself as loss after loss -- measured: recalling the shipped
    "Publication figure (light)" over a drawn window reports 0 losses through this
    mechanism and 7 through a template-shaped one, and all seven ARE the seam: the two
    scatter columns, the two KDE columns, the histogram column, the ticked sources and the
    cluster features -- everything a template deliberately does not carry. An undo entry
    and a script's ``apply_config`` have no file to be unfaithful to. So this is recorded
    at the one door that reads a file, and it is spoken for as long as the configuration
    on screen DESCENDS FROM THAT IMPORT -- not one install longer, and not one shorter.

    "CLEARED BY THE ONE WRITER ON EVERY OTHER INSTALL" WAS THE PREVIOUS RULE, AND IT WAS
    A HOLE (S7-FIX1 / S7-SK1). Its argument -- "a template, an undo, another import
    cannot leave a sentence standing about a file that is no longer on screen" -- does
    not cover the undo that lands back ON the file's own state, nor the Time-unit combo,
    whose handler re-installs through ``_rebase -> _set_config`` without changing
    anything about the file. Measured on the owner's 982-event table: after an import
    naming ``fit.kind`` "stretched exponential" and ``histogram.column``
    "conductance_nS", Details carried the losses sentence AND the S6-03 note telling the
    user to "Set Time unit = s"; the user did that, and the sentence was gone; a re-save
    then wrote ``dwell_s`` and ``none`` over the colleague's file with 0 dialogs. The
    same erasure by one Undo after one bins change, with ``config()`` equal to the
    post-import value. A survey of 17 gestures found exactly those two erasers (plus
    the documented template recall); everything else -- bins, column, log axes, plot
    type, a second source, accepted-only, Plot, Refresh, a rename, a title -- preserved
    it, because none of them reach the writer.

    SO THE RECORD IS TAGGED WITH THE ORIGIN IT WAS TAKEN FOR and read against the origin
    on screen (:meth:`_ConfigHistory.open_origin`, :attr:`_ConfigHistory.origin`): the
    import opens an origin and takes the record; a template recall or a script's
    ``apply_config`` opens another, which HIDES it; an undo, a redo and a rebase walk the
    window along its own lineage and touch nothing, so the sentence stands on the file's
    state and on every state reached from it by hand, disappears when Undo steps back
    past the import, and returns when Redo steps forward onto it again -- whichever
    writer re-installed it. Another import re-takes it; ``reset_state`` clears it with
    the rest of the window. ``current_origin`` is the SAME resolver :class:`_NormRefusal`
    reads, so the two disclosures cannot disagree about which install is on screen.

    RE-DERIVED BY EVERY DRAW, not printed once, for :class:`_NormRefusal`'s reason:
    Details is rewritten in full by every redraw, so a sentence printed at the door is
    wiped by the same call that printed it. :meth:`PlotsWindow._incoming_notes` is the
    one reading.

    ONE RECORD PER ORIGIN (A1-N4), for the reason given at length on
    :class:`_NormRefusal`: a single slot is re-taken by the SECOND import, so an undo
    that walks back onto the FIRST file's own state found the history's origin equal
    to that file's and the record's origin equal to the other's, and both sentences
    hidden -- measured, with the re-save over the first file then silent. Keyed by
    origin, each import's account survives the next import for as long as an undo can
    still reach the state it describes; :meth:`clear` forgets the on-screen lineage's
    record only, :meth:`forget` (``reset_state``) the lot.
    """

    __slots__ = ("_records", "_current_origin")

    def __init__(self, current_origin: Callable[[], int] | None = None) -> None:
        #: origin -> (file name, the losses), in the order the records were taken.
        self._records: dict[int, tuple[str, tuple[str, ...]]] = {}
        #: Answers "which origin is on screen"; None means "the newest record", for a
        #: record that lives outside a window.
        self._current_origin = current_origin

    def _shown(self) -> int | None:
        """The origin whose record is spoken: the one on screen, or -- for a record
        outside a window -- the newest taken (None when none has been)."""
        if self._current_origin is not None:
            return int(self._current_origin())
        return next(reversed(self._records), None)

    def clear(self) -> None:
        """No file's losses are outstanding FOR THE STATE ON SCREEN. Records of other
        lineages stand, for an undo that lands back on them."""
        if self._current_origin is None:
            self._records.clear()
            return
        self._records.pop(self._shown(), None)

    def forget(self) -> None:
        """Every lineage's record, gone -- for ``reset_state``, where the history that
        could reach any of them is emptied in the same breath."""
        self._records.clear()

    def take(self, incoming, installed, origin: int = 0) -> None:
        """Record what ``incoming`` asked for and ``installed`` does not carry.

        ``incoming`` is None (no file is being installed -- clear) or the triple
        :meth:`PlotsWindow.import_config_json` stashes: the file's NAME, its raw payload
        and the :class:`PlotConfig` that payload decodes to. ``installed`` is what
        :meth:`PlotsWindow.config` reports once the write is done. ``origin`` is the
        lineage the import opened; the record is spoken while the window's state
        descends from it.
        """
        if incoming is None:
            self.clear()
            return
        source, payload, asked = incoming
        key = int(origin)
        self._records.pop(key, None)        # re-taken: newest again
        self._records[key] = (str(source),
                              tuple(_config_round_trip_losses(payload, asked, installed)))
        _drop_oldest(self._records)

    def _spoken(self) -> tuple[str, tuple[str, ...]]:
        """``(file, losses)`` for the origin on screen, or ``("", ())``."""
        shown = self._shown()
        if shown is None:
            return "", ()
        source, names = self._records.get(shown, ("", ()))
        return (source, names) if names else ("", ())

    @property
    def source(self) -> str:
        """The file the losses are about, or ``""`` when there are none on screen."""
        return self._spoken()[0]

    @property
    def names(self) -> tuple[str, ...]:
        """Every loss, as ``key “value”``, whatever the sentence's cap names -- or ``()``
        when the import they belong to is no longer the state on screen."""
        return self._spoken()[1]

    @property
    def origin(self) -> int:
        """The lineage of the losses spoken NOW (0 = none are)."""
        return int(self._shown() or 0) if self.names else 0

    def note(self) -> str:
        """The sentence for the Details notes list, or ``""`` when nothing was lost.

        THE COUNT IS STATED WHOLE and the names are capped at :data:`_LOSSES_NAMED`, so a
        file this build understands very little of still says how little -- the cap
        shortens the evidence, never the claim.

        NO PROMISE ABOUT THE FUTURE. It says what the import DID and what the export IS,
        both of which are settled facts; it deliberately does not say "a re-save will drop
        these", because a user who then loads the missing source and ticks it has made one
        of them survivable, and a disclosure that a gesture can falsify is worse than none.
        """
        source, names = self._spoken()
        if not names:
            return ""
        shown = list(names[:_LOSSES_NAMED])
        rest = len(names) - len(shown)
        listed = ", ".join(shown) + (f" and {rest} more" if rest else "")
        n = len(names)
        return (f"{source} named {n} setting{'' if n == 1 else 's'} this window "
                f"did not take: {listed}. Saving a configuration from here writes the "
                f"settings on screen, not the file that was opened.")


@dataclass(frozen=True)
class StyleSettings:
    """The user-tunable look of the plot, routed through every render site.

    FROZEN, AND A VALUE ALL THE WAY DOWN. A style is read at every render site and
    handed out whole by :meth:`PlotsWindow.style`, so it has to be safe to KEEP: a
    snapshot must describe the moment it was taken even after the user drags the
    legend and picks a new colour. That needs two things, and the second is the one
    that is easy to miss -- ``frozen=True`` for the fields, and an immutable
    collection type (:class:`FrozenMap`) for the two mapping fields, because
    :func:`dataclasses.replace` copies a ``dict`` field BY REFERENCE and would
    otherwise leave every "copy" writing into one shared mapping.

    Every change therefore goes through ``self._style = replace(self._style, ...)``
    (or one of the ``with_*`` helpers below for the mapping fields), which is also
    exactly the shape an undo stack and a saved plot template need.

    Defaults reproduce the app's original hard-coded appearance: Okabe-Ito data
    (blue) and fit (vermillion) colours, a 7-px marker, a 1-px bar outline (the
    fit curve draws one px thicker), a 10-pt font and the dark theme. A blank
    ``title`` means "use the automatic title"; ``export_w``/``export_h`` of 0
    mean "export at the widget's native size".
    """
    title: str = ""                 # "" => the automatic per-plot title
    x_label: str = ""               # "" => the automatic axis label (unit scaling still applies)
    y_label: str = ""               # "" => the automatic axis label
    show_legend: bool = True        # the legend is an index, and an index is optional
    font_family: str = ""           # "" => the figure default, Arial (figure_fonts); the Font
                                    # row's first entry names it (see _FONT_DEFAULT_LABEL).
                                    # NOT a missing value -- it round-trips through JSON,
                                    # undo and templates as itself.
    font_size_pt: float = 10.
    data_color: str = _DATA_HEX
    fit_color: str = _FIT_HEX
    point_size: float = 7.
    line_width: float = 1.
    background: str = "dark"         # "dark" | "light"
    frame: bool | None = None        # None preserves the original native frame
    export_w: int = 0               # 0 => native width
    export_h: int = 0               # 0 => native height
    #: Owner note #15: "the legend is not editable". A source's legend text, per
    #: source KEY (never per label — the label is exactly what is being replaced).
    #: Absent means "use the source's own label", so an empty dict is today's look.
    #: Keyed rather than positional so a refresh, which re-identifies sources by key,
    #: cannot move one source's name onto another's curve.
    label_overrides: FrozenMap = _NO_ENTRIES
    #: WHERE THE USER DRAGGED THE LEGEND, as a fraction of the plot area: ``(fx, fy)``
    #: is the legend box's top-left corner over the ViewBox's width and height. ``None``
    #: means "never moved" and is what puts a fresh plot back at
    #: :data:`_LEGEND_DEFAULT_OFFSET`.
    #:
    #: A FRACTION, NOT PIXELS, because the plot area is resized constantly -- by the
    #: splitter, by the window, and by ``Save image``'s export size -- and a legend
    #: pinned at x = 287 px is off the canvas as soon as the panel narrows. The
    #: fraction is exactly the quantity ``GraphicsWidgetAnchor.autoAnchor(...,
    #: relative=True)`` already stores internally, which is why re-applying it is
    #: byte-for-byte idempotent (measured: five successive re-applies of (287.0187,
    #: 192.0) all landed on 287.0187, 192.0).
    legend_pos: tuple[float, float] | None = None
    #: Legend text size in pt. ``0`` means "follow the body font, three points below
    #: it" — the historical hard-coded rule, kept as the default so the untouched
    #: window still shrinks its index when the title font grows.
    legend_font_size_pt: float = 0.
    #: Whether an automatic ``(n = N)`` is appended to a legend entry. On by default
    #: (that is today's look), but it has to be defeatable: an entry the user has
    #: RETYPED must not be silently extended with a count they did not ask for.
    legend_counts: bool = True
    #: WHICH LEGEND ENTRIES ARE DRAWN, per entry, keyed ``"<kind>:<key>"`` -- exactly the
    #: id :meth:`PlotsWindow._legend_add` is handed at each of its six call sites
    #: (``series:<source key>``, ``fit:<source key>``, ``cluster:<id>``). Absent means
    #: INCLUDED, so an empty dict is today's look.
    #:
    #: Owner note Q2: "when the fit is performed and the legend is selected, the app
    #: includes icons for both the histogram and the lines. This is incorrect." A fitted
    #: histogram legends the bars AND the curve, and in a paper figure that is two rows
    #: for one population. Which of the two to keep is an editorial choice, not a
    #: default: on a single-source fit the fit row carries tau and the bar row carries
    #: nothing, and on a five-source overlay it is the other way round.
    legend_include: FrozenMap = _NO_ENTRIES
    #: Gridlines (owner note Q7). ``showGrid`` was hard-coded once at construction with
    #: alpha 0.15 and never read again, so there was no way to turn it off for a
    #: publication figure. The alpha is kept at the historical value; only the on/off
    #: is a control, because a half-visible grid is a worse answer than either.
    show_grid: bool = True
    #: INDEPENDENT TEXT SIZES (owner note Q5: "there is only one option to increase the
    #: font size of all text in the plot -- this is incorrect"). ``font_size_pt`` stays
    #: the BODY size and remains the one thing ``set_font_size`` moves; each of these is
    #: an override in pt, and ``0`` means "follow the body size", which is precisely the
    #: single-knob behaviour every existing style already has. So a StyleSettings built
    #: before this round renders byte-for-byte as it did.
    #:
    #: The legend keeps its own older field (``legend_font_size_pt``, whose 0 means
    #: "body - 3") rather than being folded in here: its automatic rule is not "follow
    #: the body", and collapsing the two would silently enlarge every existing legend.
    title_font_size_pt: float = 0.
    axis_label_font_size_pt: float = 0.
    tick_font_size_pt: float = 0.
    #: The bootstrap's confidence BAND around the fitted curve. Off: it is a second
    #: object on the figure and, unlike the parameters' intervals, it does not exist for
    #: every fit (see :func:`nanoscope.analysis.stats.ci_band` for the four ways a fit can
    #: have none) -- a control that silently does nothing on half the fits is worse
    #: defaulted on.
    show_ci_band: bool = False
    #: E2 -- WHAT MADE THESE EVENTS, written on the figure: the detector, the ML model
    #: where there is one, and the preprocessing chain that ran before it. Off by default:
    #: it is an annotation, and one appearing on every plot unasked is clutter. It is
    #: drawn as a plot item so ``Save image…`` keeps it -- which is the whole point, since
    #: the PNG is what leaves the app and the provenance is what a reader of it cannot
    #: otherwise recover.
    show_provenance: bool = False
    #: The residual strip under the histogram, and the reduced χ² that goes with it. Off
    #: for the same reason: a second axis appearing on every fitted histogram unasked is
    #: clutter, not presentation.
    show_residuals: bool = False
    #: The Event-count heatmap's colour map (one of :data:`_HEATMAP_COLORMAPS`).
    #: APPEARANCE, not analysis: it recolours counts already binned, so a change restyles
    #: the drawn image and its colour bar and never re-bins -- the rule every other field
    #: here keeps. Kept in the style rather than on the heatmap page for that reason, and
    #: so it rides undo, templates and JSON with the rest of the look.
    heatmap_colormap: str = "viridis"
    #: PER-ELEMENT TEXT STYLES, from the figure itself (double-click or right-click a
    #: title, an axis, a legend entry or the fit label): element key -> the
    #: ``figure_text.TextStyle`` fields as a frozen map. Never the TEXT of the title and
    #: the two axis labels, which stays in ``title``/``x_label``/``y_label`` -- one store
    #: per label, so the Appearance boxes and the figure cannot disagree -- nor a legend
    #: row's, which is its source's name (``label_overrides``). An annotation's words (the
    #: detection-limit label) have no other home and are kept here. A legend row's style
    #: is ``legend.entry.<source key>`` for a source's row and
    #: ``legend.entry.<kind>:<key>`` for the others (``fit:``, ``cluster:``, ``kde:``).
    #: Empty is today's look: every element follows the figure-wide fields above.
    text_styles: FrozenMap = _NO_ENTRIES
    #: THE FIT LABEL (``fit_label.FitLabelState`` as a frozen map). A fit's numbers are
    #: drawn there and never in the title, and its Numbers row (``fields``) chooses which
    #: (the Residual strip box adds its χ²/dof after the equation: ``_shown_readout``).
    #: Empty means "never set": the label is shown with its default fields
    #: (:meth:`fit_label_state`).
    fit_label: FrozenMap = _NO_ENTRIES
    #: Shared style identity is part of the same undoable native configuration.
    style_binding: FrozenMap = _NO_ENTRIES
    series_styles: FrozenMap = _NO_ENTRIES

    def __post_init__(self) -> None:
        if self.frame is not None and type(self.frame) is not bool:
            raise ValueError("Style frame must be true, false or inherited")
        from ..core.figure_series import validate_series_styles
        object.__setattr__(self, "series_styles", _deep_frozen(validate_series_styles(self.series_styles)))
        if self.style_binding:
            from ..core.figure_style import normalize_style_binding
            from ..core.figure_spec import thaw_json
            object.__setattr__(self, "style_binding", _deep_frozen(thaw_json(
                normalize_style_binding(self.style_binding))))
        for name, low, high in (("font_size_pt", 6, 48), ("point_size", 1, 30),
                               ("line_width", .25, 10), ("legend_font_size_pt", 0, 48),
                               ("title_font_size_pt", 0, 72),
                               ("axis_label_font_size_pt", 0, 72),
                               ("tick_font_size_pt", 0, 72)):
            value = getattr(self, name)
            if self.style_binding:
                low, high = (0 if low == 0 else .000001), float("inf")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid style {name}: expected {low}–{high}")
            object.__setattr__(self, name, float(value))
        # The two maps arrive from JSON, templates and replace() alike; one validation
        # here keeps a malformed entry out of every door, and the deep freeze keeps the
        # style hashable (FrozenMap hashes its items).
        styles = {}
        for key, value in dict(self.text_styles or {}).items():
            style = TextStyle.from_mapping(value if isinstance(value, Mapping) else {})
            if not _text_kept_in_styles(str(key)):
                style = _dc_replace(style, text=None)
            if valid_element_key(key) and not style.is_default():
                styles[str(key)] = _deep_frozen(style.to_dict())
        object.__setattr__(self, "text_styles", FrozenMap(styles))
        label = dict(self.fit_label or {})
        object.__setattr__(self, "fit_label", _deep_frozen(
            FitLabelState.from_mapping(_thawed(label)).to_dict()) if label else _NO_ENTRIES)

    # -- per-element text (figure_text) ----------------------------------------------------------
    def text_style(self, key: str) -> TextStyle:
        """Element ``key``'s whole style, its text included, wherever each part lives."""
        style = TextStyle.from_mapping(_thawed(self.text_styles.get(key, {})))
        field = _TEXT_FIELD_OF.get(key)
        if field is not None:
            return _dc_replace(style, text=_style_text(getattr(self, field)))
        if key == FIT_LABEL_KEY:
            label = self.fit_label_state()
            return _dc_replace(style, text=label.template or None, size_pt=label.size_pt,
                               color=label.color)
        return style

    def effective_text_style(self, key: str) -> TextStyle:
        """A draw-only resolved style; native text history stores explicit values."""
        explicit = self.text_style(key)
        if not self.style_binding:
            return explicit
        from ..core.figure_style import resolve_bound_text
        values, _origins = resolve_bound_text(self.style_binding, key, explicit.to_dict())
        return _dc_replace(explicit, **values)

    def with_text_style(self, key: str, style: TextStyle) -> "StyleSettings":
        """This style, with ``key``'s WHOLE style replaced (text included). The title's
        and axis labels' text go to their own fields, the fit label's text, size and
        colour to ``fit_label``, an annotation's text stays with its style; a legend
        entry's text is the window's to route."""
        if not valid_element_key(key):
            return self
        changes: dict = {}
        field = _TEXT_FIELD_OF.get(key)
        if field is not None:
            changes[field] = ("" if style.text is None else
                              " " if style.text == HIDDEN_TEXT else style.text)
        elif key == FIT_LABEL_KEY:
            changes["fit_label"] = _deep_frozen(_dc_replace(
                self.fit_label_state(), template=style.text or "", size_pt=style.size_pt,
                color=style.color).to_dict())
            style = _dc_replace(style, size_pt=None, color=None)
        if not _text_kept_in_styles(key):
            style = _dc_replace(style, text=None)
        styles = (self.text_styles.remove(key) if style.is_default()
                  else self.text_styles.set(key, style.to_dict()))
        return _dc_replace(self, text_styles=styles, **changes)

    def fit_label_state(self) -> FitLabelState:
        """The fit label as drawn: the saved one, else the default (equation, R², n).

        The switches older builds kept beside it (Fit values on plot, R² on figure, KS
        statistic on figure) are folded into its fields when a saved configuration is
        read (:func:`_fold_legacy_style`); nothing here reads them."""
        if self.fit_label:
            return FitLabelState.from_mapping(_thawed(self.fit_label))
        return FitLabelState()

    def with_fit_label(self, state: FitLabelState) -> "StyleSettings":
        return _dc_replace(self, fit_label=_deep_frozen(state.to_dict()))

    # -- mapping-field helpers -------------------------------------------------------------------
    # The two ``FrozenMap`` fields cannot be written in place, and spelling
    # ``replace(s, label_overrides=s.label_overrides.set(k, v))`` at each call site would
    # bury a one-line intent in plumbing. These say what the call site means and keep the
    # frozen-copy mechanics in one place. Reads need no helper: a FrozenMap already
    # answers ``.get(key, default)`` exactly as the dict did.

    def with_label_override(self, key: str, text: str) -> "StyleSettings":
        """This style, but ``key``'s legend text is ``text``."""
        return _dc_replace(self, label_overrides=self.label_overrides.set(key, text))

    def without_label_override(self, key: str) -> "StyleSettings":
        """This style, but ``key`` falls back to the source's own label."""
        return _dc_replace(self, label_overrides=self.label_overrides.remove(key))

    def with_legend_include(self, entry_id: str, included: bool) -> "StyleSettings":
        """This style, but legend entry ``entry_id`` is drawn / not drawn."""
        return _dc_replace(
            self, legend_include=self.legend_include.set(str(entry_id), bool(included)))


# ================================================================================================
# THE PLOT CONFIG -- every user-tunable setting in this window, as ONE frozen value
# ================================================================================================
#
# WHY A CONFIG OBJECT AT ALL (owner, item A1): "refactor plot state into a single config
# object (data selection, plot type, styling, fit params) that's fully replaced -- not
# patched -- on each render, to prevent stale settings (like titles) from persisting."
#
# The state was spread across ~40 widgets and ~10 attributes, and every rule about what
# a change INVALIDATES was written at the call site that made the change. That is why the
# window had eight measured stale-state bugs at once: the axis-range boxes were hardened
# against a Time-unit switch and the Fit-range boxes -- two rows above them, read in the
# same display unit -- were not; a plain Refresh destroyed the pinned axis range and kept
# the typed title, while a plot-type change did the opposite. Each fix was correct and
# none of them generalised, because there was no single place where "the analysis changed"
# could be said once.
#
# There is now. :class:`AnalysisIdentity` is WHAT THE PLOT IS OF (plot type, the two axis
# columns, the fitted column, the time unit); :meth:`PlotConfig.rebased` is the ONE rule
# for what a move in that identity clears or retunes; and
# :meth:`PlotsWindow._set_config` is the ONE writer that pushes a config onto the widgets.
# A new stale-state bug of this family now requires a change to `rebased`, not the absence
# of one at a call site.
#
# WIDGETS ARE A VIEW OF THIS, not a second source of truth: :meth:`PlotsWindow.config`
# projects them into a value, ``_set_config`` projects a value back onto them, and the two
# round-trip (``w._set_config(c); w.config() == c``). Reads inside the draw path still go
# to the widgets -- migrating 27 of those is the next step, not this one -- but every
# CLEARING RULE has already moved here, which is the half the bugs lived in.


@dataclass(frozen=True)
class AnalysisIdentity:
    """WHAT THE PLOT IS OF: the quantities, never their presentation.

    The distinction is the whole rule. A colour, a font, a bin count and a log axis are
    presentation -- they change the picture of the same numbers, so nothing typed about
    those numbers goes stale. A plot type, an axis column and the fitted column change
    WHICH NUMBERS, so a pinned range, a fit window and a typed caption stop describing
    anything. The time unit is the odd one: it does not change the quantity, only its
    scale, so the typed windows are CARRIED (rescaled) rather than cleared.

    ``fit_column`` is carried separately from ``x_column`` because the Fit-range boxes
    live on the histogram page and bound the histogram's column whatever plot type is on
    screen -- see :meth:`PlotsWindow._fit_range`, their only reader.

    ``fit_model`` is the APPLIED fit of the plot on screen ("" with none): the fit label's
    wording describes it, so a move of the model clears the wording
    (:meth:`PlotConfig.rebased`). A mixture's component count is part of it unless the
    count is automatic, which the draw itself writes back.

    ``features`` is the PCA's feature set (sorted; ``()`` on every other plot type). PCA has
    no axis column, but its two axes are combinations of exactly these columns, so another
    set is other scores: the words and the pins typed for the old ones go with it, as they
    do with a column.
    """
    plot_type: str = HISTOGRAM
    x_column: str = ""
    y_column: str = ""
    fit_column: str = ""
    time_unit: str = _DEFAULT_TIME_UNIT
    fit_model: str = ""
    features: tuple[str, ...] = ()
    density_y: bool = False

    @property
    def axis_factors(self) -> tuple[float, float]:
        """(x, y) factors converting the DISPLAYED axis unit back to data units."""
        fx = _display_factor(self.x_column, self.time_unit)
        return (fx, 1. / fx if self.density_y else _display_factor(self.y_column, self.time_unit))

    @property
    def fit_factor(self) -> float:
        """The factor the Fit-range boxes' text is written in."""
        return _display_factor(self.fit_column, self.time_unit)

    def wording_stale_from(self, was: "AnalysisIdentity") -> bool:
        """Whether the fit label's wording, typed for ``was``, no longer describes this
        analysis: the plot type, a column or the PCA's feature set moved, or the applied fit
        went from one model to another or to none. A first fit (none -> a model) keeps
        wording typed ahead of it."""
        return ((self.plot_type, self.x_column, self.y_column, self.features)
                != (was.plot_type, was.x_column, was.y_column, was.features)
                or bool(was.fit_model and self.fit_model != was.fit_model))


@dataclass(frozen=True)
class SelectionConfig:
    """WHICH ROWS: the checked sources, and whether the accepted filter is on."""
    source_keys: tuple[str, ...] = ()
    accepted_only: bool = True


@dataclass(frozen=True)
class HistogramConfig:
    """The Histogram page. ``column`` is the analysis; the rest is presentation.

    ``normalization`` is CANONICALISED on construction -- see :meth:`__post_init__`.
    """
    column: str = ""
    bins: int = 30
    normalization: str = _NORM_COUNT
    colour_by: str = _COLOUR_NONE
    log_x: bool = False
    log_y: bool = False
    include_filter_limited: bool = False
    #: A 1-D Gaussian KDE (Scott bandwidth) over each drawn series, scaled to the bars'
    #: y quantity. AN ANALYSIS SETTING, not presentation: it estimates a new curve from
    #: the sample, so a hand tick goes behind the Plot / Apply gate (see
    #: :meth:`PlotsWindow._on_kde_overlay_toggled`) and, like every field here, is
    #: carried by templates, undo, the JSON sidecar and the project snapshot.
    kde_overlay: bool = False

    def __post_init__(self) -> None:
        """Rewrite a pre-S6-03 ``normalization`` spelling into today's. The ONE place the
        rename's migration runs, and the only reason this module has a ``__post_init__``.

        WHERE S6-03 PUT IT, AND WHY THAT WAS ONE DOOR SHORT. The migration went in at the
        READ boundary (:func:`plot_config_from_json`) on the argument that both FILE doors
        -- an imported sidecar and a stored template -- pass through it. They do. The
        IN-PROCESS doors do not, and they are the ones a script or a notebook holds a
        :class:`PlotConfig` for. Both were driven on the build that shipped that fix:

        * ``w.apply_config(replace(cfg, histogram=replace(cfg.histogram,
          normalization="density (area = 1)")))`` left the combo on ``count``, made
          ``config().histogram.normalization`` read ``count``, labelled the y axis
          ``count`` and drew raw integers -- on the owner's 982-event file, first five bars
          1, 4, 3, 17, 18 where the pre-rename build drew 1.66, 6.63, 4.98, 28.19, 29.85
          for the same request. ``_set_config`` installs
          the mode with ``QComboBox.setCurrentText``, a silent no-op for text the combo
          does not carry. No exception, nothing in Details, nothing in the banner.
        * ``w.apply_template(the same value)`` lost it too, and for the OPPOSITE reason:
          :meth:`PlotsWindow._installable` deliberately resolves a name the combo cannot
          offer back to the one on screen, and by that test yesterday's spelling is simply
          "not on offer". A fix bolted onto ``_installable`` would have closed that door
          and left ``apply_config`` -- the public one -- open.

        SO IT LIVES IN THE VALUE. Three places that must agree is how this defect was
        born: the reader remembered, :meth:`PlotsWindow.set_normalization` remembered, and
        the two install doors did not. There is no fourth place to forget here, because
        every config-shaped door in this module ends up CONSTRUCTING one of these -- the
        file reader (:func:`_decode_dc` builds the dataclass field by field), the template
        store (same reader), an undo entry (taken from :meth:`PlotsWindow.config`, always
        canonical because it reads the combo), ``dataclasses.replace`` in a caller's own
        script, and a hand-built ``HistogramConfig(normalization=...)``. After this
        constructor there is no such thing as a live ``PlotConfig`` carrying the old
        spelling, which is the property the template comparison
        (:meth:`PlotsWindow._sync_template_selection`), the undo diff and
        ``PlotConfig.__eq__`` were already resting on and only the file doors had.

        TRANSLATED, NOT REFUSED -- :data:`_LEGACY_NORMS`'s reason. The old name still
        names a mode that exists, and this app wrote it into the user's own sidecars and
        templates last week. A name that is genuinely foreign -- a newer build's, or a
        hand-edit -- is left EXACTLY as given: a value object is not the place to judge
        what a widget can offer, so those go on being refused by name in
        ``set_normalization`` and RECORDED, at the one writer, by
        :class:`_NormRefusal`.

        THE LOOKUP ITSELF IS :func:`_canonical_norm`, and it is a function precisely
        because this constructor and ``set_normalization`` had already drifted apart over
        whitespace -- see that function for the measurement. Reading the table twice is
        how this class of defect starts; both readers now ask the same question.

        IT COSTS NOTHING MEASURABLE. Constructing one of these went 0.435 -> 0.48 us
        (200,000 constructions, before and after); :meth:`PlotsWindow.config`, which
        builds eight sections on every sync, measures 19-22 us per call on BOTH sides of
        this change -- one dict lookup is inside that spread, and inside the 71 us
        whole-hook budget :class:`_ConfigHistory`'s docstring works to.
        """
        want = _canonical_norm(self.normalization)
        if want != self.normalization:
            object.__setattr__(self, "normalization", want)   # frozen: the only way in


@dataclass(frozen=True)
class ScatterConfig:
    x: str = ""
    y: str = ""
    log_x: bool = False
    log_y: bool = False
    colour_by: str = _COLOUR_NONE


@dataclass(frozen=True)
class KdeConfig:
    #: LOG X DEFAULTS ON, AND THIS IS THE ONLY PLACE THAT SAYS SO. It used to be said
    #: twice -- here, and as a bare ``setChecked(True)`` in the KDE page's builder -- so
    #: a fresh window and a window that had been through ``reset_state`` (which installs
    #: ``PlotConfig()``) took their default from two different literals and would have
    #: disagreed the first time either moved. The builder now reads this value.
    #:
    #: KEPT ON rather than unified to the other pages' False, deliberately. A KDE is
    #: read as a SHAPE -- the contour set is the whole output -- and a dwell axis drawn
    #: linearly puts 90% of a log-normal population into the leftmost 10% of the picture,
    #: where the contours merge into one blob and the plot says nothing. The histogram
    #: and scatter pages default to linear because they are also used for depth against
    #: depth, where linear is right; the KDE page exists for the dwell-vs-depth density
    #: (owner note #19, the export figure it was lifted from draws log x too), so its
    #: useful default differs. It is a DEFAULT, not a property of the data: the toggle is
    #: live, and the axis label carries the decade ticks that show which one is in force.
    x: str = ""
    y: str = ""
    log_x: bool = True
    log_y: bool = False
    levels: int = 8


@dataclass(frozen=True)
class HeatmapConfig:
    """The Event-count heatmap page: two columns binned into a 2-D COUNT image.

    Its own pair of columns, as the KDE page has, rather than the scatter's: a page
    remembers the axes chosen for it, and moving between a scatter and a heatmap of
    different columns must not drag one page's choice onto the other.

    ``log_x`` DEFAULTS ON for :class:`KdeConfig`'s reason, and because Event waveforms'
    heatmap -- the same kernel, ``event_shape.density_2d`` -- defaults its dwell axis to
    log: a log-distributed dwell drawn linearly lands in the first column of bins.
    ``bins`` is per axis (``bins`` x ``bins``), Event waveforms' default and range.
    """
    x: str = ""
    y: str = ""
    log_x: bool = True
    log_y: bool = False
    bins: int = 40


@dataclass(frozen=True)
class ClusterConfig:
    """Clustering and PCA share the feature set, so it lives here rather than per page."""
    features: tuple[str, ...] = ()
    method: str = _METHOD_GMM
    k_mode: str = _K_AUTO
    k: int = 2


@dataclass(frozen=True)
class FitConfig:
    """The fit, SELECTED and APPLIED, plus the truncation window.

    THE PROPOSAL/APPLIED SPLIT IS CONFIG, BOTH HALVES (owner note #12). ``kind``/``k``/
    ``k_auto``/``scatter_kind`` are what the dropdowns propose; ``applied*`` is what the
    picture actually shows. Restoring a saved config has to put both back, or the window
    reopens badging a pending change nobody made.

    ``range_lo``/``range_hi`` ARE THE BOX TEXT, VERBATIM, and ``range_unit`` is the Time
    unit they were typed under. Storing the unit WITH the bounds is what makes them
    unmisreadable: the only reader (:meth:`PlotsWindow._fit_range`) converts with
    ``range_unit``, not with whatever the combo happens to say now, so a bound cannot be
    read 1000x wrong even by a caller who moved the combo without rebasing. The text is
    kept verbatim rather than parsed because :meth:`PlotsWindow._fit_range_input_notes`
    quotes it back to the user ("Fit range min "nan" is not a usable bound").
    """
    kind: str = _FIT_NONE
    k: int = 1
    k_auto: bool = False
    scatter_kind: str = _SFIT_NONE
    applied: str = _FIT_NONE
    applied_k: int = 1
    applied_k_auto: bool = False
    applied_scatter: str = _SFIT_NONE
    range_lo: str = ""
    range_hi: str = ""
    range_unit: str = _DEFAULT_TIME_UNIT


@dataclass(frozen=True)
class AxesConfig:
    """The Time unit, the four typed axis-range boxes, and the pin they produced.

    TWO REPRESENTATIONS OF ONE THING, DELIBERATELY. The four strings are the box text in
    the unit the axis DISPLAYS; ``manual_x``/``manual_y`` are the pin the Apply button
    made out of them, in DATA units. They are not redundant: a number can sit in a box
    unapplied, and a pin survives a unit change that rewrites the text over it. Both are
    config, and :meth:`PlotConfig.rebased` moves them together so they cannot disagree
    about which quantity they describe.
    """
    time_unit: str = _DEFAULT_TIME_UNIT
    x_min: str = ""
    x_max: str = ""
    y_min: str = ""
    y_max: str = ""
    manual_x: tuple[float, float] | None = None
    manual_y: tuple[float, float] | None = None
    quantity_display: FrozenMap = _NO_ENTRIES

    def __post_init__(self):
        object.__setattr__(self, "quantity_display", _deep_frozen(self.quantity_display))

    def ranges(self) -> tuple[str, ...]:
        """The axes holding a typed limit or a pin, as the status line names them
        (``quantity_reset``): ``("X range", "Y range")``, or the part of it that is set."""
        return tuple(f"{name} range" for name, lo, hi, pin in (
            ("X", self.x_min, self.x_max, self.manual_x),
            ("Y", self.y_min, self.y_max, self.manual_y)) if lo or hi or pin)


@dataclass(frozen=True)
class PlotConfig:
    """EVERY user-tunable setting in the Plots window, as one frozen value.

    Frozen all the way down, for the reason :class:`StyleSettings` is: a config is handed
    out by :meth:`PlotsWindow.config` and kept -- by an undo stack, by a saved template,
    by a test comparing before and after -- so it has to describe the moment it was taken
    even after the user moves every control in the window.

    WHAT IS NOT HERE, and must never be added: anything DERIVED. The loaded sources and
    their frames, the cluster/PCA results, the FitResults, the note cache and its worker,
    every pyqtgraph item handle, and the run gate's ``_plot_armed``/``_stale``. Those are
    outputs of a config applied to data; persisting them would restore a window claiming
    to show a fit of rows it no longer has. ``selection.source_keys`` is the one boundary
    case and it is config, not derived: it names rows without carrying them.
    """
    plot_type: str = HISTOGRAM
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    kde: KdeConfig = field(default_factory=KdeConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    axes: AxesConfig = field(default_factory=AxesConfig)
    style: StyleSettings = field(default_factory=StyleSettings)
    #: Whether the user picked the histogram normalization themselves. Config, not
    #: derived: it is the memory that stops :meth:`PlotsWindow._sync_norm_default` from
    #: overruling a choice the user made, and a reopened window must not inherit it.
    norm_user_set: bool = False

    @property
    def identity(self) -> AnalysisIdentity:
        """WHAT THIS PLOT IS OF -- the projection every clearing rule is written against.

        ``""`` means "no column on that axis": the histogram's y is a count and PCA's two
        axes are scores, so neither can carry a typed limit in a data unit.
        """
        if self.plot_type == HISTOGRAM:
            x, y = self.histogram.column, ""
        elif self.plot_type == SCATTER:
            x, y = self.scatter.x, self.scatter.y
        elif self.plot_type == KDE:
            x, y = self.kde.x, self.kde.y
        elif self.plot_type == HEATMAP:
            x, y = self.heatmap.x, self.heatmap.y
        else:
            x, y = "", ""
        fit = self.fit
        if self.plot_type == HISTOGRAM and fit.applied != _FIT_NONE:
            model = fit.applied
            if fit.applied in _MULTI_COMPONENT_FITS:
                model += ":auto" if fit.applied_k_auto else f":{fit.applied_k}"
        elif self.plot_type == SCATTER and fit.applied_scatter != _SFIT_NONE:
            model = fit.applied_scatter
        else:
            model = ""
        # PCA has no axis column; the feature set is what its axes are made of. Sorted, so
        # the order a template or a file lists them in is not a move.
        features = tuple(sorted(set(self.cluster.features))) if self.plot_type == PCA else ()
        return AnalysisIdentity(plot_type=self.plot_type, x_column=x, y_column=y,
                                fit_column=self.histogram.column,
                                time_unit=self.axes.time_unit, fit_model=model,
                                features=features, density_y=(self.plot_type == HISTOGRAM and
                                    self.histogram.normalization == _NORM_DENSITY))

    @property
    def log_state(self) -> tuple[bool, bool]:
        """WHETHER EACH AXIS IS LOGARITHMIC — one concept, not three widget pairs.

        The three per-page pairs are STORAGE (a log choice made for a dwell histogram is
        not a log choice about a PCA score plot, so each page remembers its own), but
        "is this axis logarithmic" is one question with one answer, and it was being
        asked by switching on the plot type in four separate places: the draw path's
        ``_current_log_state``, the range-pin converter ``_to_log``, the axis labeller
        and the KDE draw. Four switches over one enumeration is four chances to add a
        plot type and update three of them.

        PCA is ``(False, False)`` and is not a widget pair at all: its axes are principal
        component SCORES, which are centred and therefore signed, so a log axis is not a
        presentation choice there — it is undefined.
        """
        if self.plot_type == HISTOGRAM:
            return bool(self.histogram.log_x), bool(self.histogram.log_y)
        if self.plot_type == SCATTER:
            return bool(self.scatter.log_x), bool(self.scatter.log_y)
        if self.plot_type == KDE:
            return bool(self.kde.log_x), bool(self.kde.log_y)
        if self.plot_type == HEATMAP:
            return bool(self.heatmap.log_x), bool(self.heatmap.log_y)
        return False, False

    def with_log_state(self, *, log_x: bool | None = None,
                       log_y: bool | None = None) -> "PlotConfig":
        """This config with the ACTIVE page's log choice set. ``None`` leaves an axis alone.

        The writer half of :attr:`log_state`, so a caller that means "log the x axis of
        whatever is on screen" says it once instead of choosing between three setters by
        plot type. On PCA it is a no-op for the reason ``log_state`` is ``(False, False)``
        there — there is no toggle to move, and inventing one would be a control that
        cannot be honoured.
        """
        if log_x is None and log_y is None:
            return self
        now_x, now_y = self.log_state
        x = now_x if log_x is None else bool(log_x)
        y = now_y if log_y is None else bool(log_y)
        if self.plot_type == HISTOGRAM:
            return _dc_replace(self, histogram=_dc_replace(self.histogram, log_x=x, log_y=y))
        if self.plot_type == SCATTER:
            return _dc_replace(self, scatter=_dc_replace(self.scatter, log_x=x, log_y=y))
        if self.plot_type == KDE:
            return _dc_replace(self, kde=_dc_replace(self.kde, log_x=x, log_y=y))
        if self.plot_type == HEATMAP:
            return _dc_replace(self, heatmap=_dc_replace(self.heatmap, log_x=x, log_y=y))
        return self

    def rebased(self, was: AnalysisIdentity, *, y_moved: bool = False) -> "PlotConfig":
        """THE ONE RULE: this config, corrected for having moved off identity ``was``.

        Every stale-state bug this refactor exists to kill is a clause here, and each is
        a statement about MEANING rather than about a call site:

        1. **The unit moved, the quantity did not** -- so every typed window is CARRIED,
           not cleared: its text is rewritten into the new unit and the physical window it
           describes is unchanged. Both the axis-range boxes (which already did this) and
           the Fit-range boxes (which did not: measured, a fit range of [0.1, 2.0] ms
           fitting 298 events at tau = 2.94e-4 s REFUSED after switching the combo to µs,
           with the boxes unchanged and nothing on screen saying the numbers had changed
           meaning). Rescaling beats clearing because the bounds are a MODELLING decision
           -- "fit the peak, not the tail" -- and a unit switch is a reading convenience;
           throwing the decision away to change the labels would be the larger surprise,
           and it is what the axis boxes have done correctly all along.
        2. **The quantity on an axis moved** -- so that axis's typed limits AND its pin go.
           Rescaling is not available here: 0.1 ms and 0.1 pA are not the same window in
           different clothes. The pin has to go WITH the text (measured: the text was
           rescaled under a pin that stayed, leaving the axis pinned to another quantity's
           numbers with nothing on screen saying so).
        3. **The fitted column moved** -- so the truncation window goes (measured: bounds
           typed as [0.1, 2.0] for dwell silently restricted a ``frac_depth`` fit over a
           column living in [0.15, 0.45]).
        4. **The analysis moved** -- so the typed caption and axis labels go, owner note
           Q6: "the title for a dwell time histogram is not the same as the title for a
           frac_io vs. time KDE plot". Note this is keyed on plot type and axis COLUMNS
           only, never on the source set: he was asked and said a caption survives ticking
           another pore. That is also why a plain Refresh no longer destroys the pinned
           range -- re-pulling the same columns is not an identity change at all, and the
           pin and the title now live or die together instead of by two ad-hoc rules.
        5. **The analysis or the applied fit moved** -- so the fit label's WORDING goes
           back to automatic with it (``style.fit_label``'s template): "Single
           exponential: {equation}" is wrong over a linear fit or a gamma. Its numbers,
           place, size and colour are the look and stay, as do the text styles. A first
           fit (none -> a model) keeps wording typed ahead of it.

        A plot-type change counts as a move on BOTH axes even when a column name survives
        it (Histogram dwell_s -> Scatter dwell_s vs depth_pA): the y axis stopped being a
        count, and a range typed against one pair of axes is not a range against another.

        A histogram's y is a count, which no column changes -- unless the Normalize is DENSITY,
        which is per the unit of its column (``density (1/s)`` over dwell_s, plain ``density``
        over frac_depth): there a new column moves the y axis as well, so its range goes too.

        ``y_moved`` says that the histogram's y QUANTITY moved -- its Normalize, count, density
        or probability per bin -- which the identity does not carry (a normalization name this
        build cannot offer must not leave a phantom anchor: see :meth:`PlotsWindow._installable`).
        The caller says it, for a person's pick of the combo only, and the y axis then moves as
        for a column: its typed limits and pin go (2), with the words (4) and the fit
        label's wording (5).
        """
        now = self.identity
        if now == was and not y_moved:
            return self
        axes, fit, style = self.axes, self.fit, self.style

        # (1) THE UNIT MOVED: carry the typed windows into it.
        if now.time_unit != was.time_unit:
            ofx, ofy = was.axis_factors
            nfx, nfy = now.axis_factors
            axes = _dc_replace(
                axes,
                x_min=_rescaled(axes.x_min, ofx, nfx), x_max=_rescaled(axes.x_max, ofx, nfx),
                y_min=_rescaled(axes.y_min, ofy, nfy), y_max=_rescaled(axes.y_max, ofy, nfy))
            off = _display_factor(was.fit_column, fit.range_unit)
            nff = _display_factor(was.fit_column, now.time_unit)
            fit = _dc_replace(fit, range_lo=_rescaled(fit.range_lo, off, nff),
                              range_hi=_rescaled(fit.range_hi, off, nff),
                              range_unit=now.time_unit)

        # (2) THE QUANTITY ON AN AXIS MOVED: text and pin go together. A moved PCA feature
        # set moves both axes: PC1 and PC2 are other combinations of other columns.
        features_moved = now.features != was.features
        if now.plot_type != was.plot_type or now.x_column != was.x_column or features_moved:
            axes = _dc_replace(axes, x_min="", x_max="", manual_x=None)
        # A histogram's y has no column of its own (a count), except as a DENSITY: that is per
        # the unit of the column it is a density of (1/s over dwell_s, no unit over frac_depth),
        # so a new column is a new y quantity there and a range pinned for the old one frames
        # nothing (measured: 90.4-378 over data topping out at 6.88).
        density_of_a_new_column = (self.plot_type == HISTOGRAM
                                   and self.histogram.normalization == _NORM_DENSITY
                                   and now.x_column != was.x_column)
        if (now.plot_type != was.plot_type or now.y_column != was.y_column or y_moved
                or features_moved or density_of_a_new_column):
            axes = _dc_replace(axes, y_min="", y_max="", manual_y=None)

        # (3) THE FITTED COLUMN MOVED: the truncation window is not transferable.
        if now.fit_column != was.fit_column:
            fit = _dc_replace(fit, range_lo="", range_hi="", range_unit=now.time_unit)

        # (4) THE ANALYSIS MOVED: the caption goes with it (owner note Q6).
        analysis_moved = y_moved or features_moved or (
            (now.plot_type, now.x_column, now.y_column)
            != (was.plot_type, was.x_column, was.y_column))
        if analysis_moved:
            style = _dc_replace(style, title="", x_label="", y_label="")

        # (5) THE FIT LABEL'S WORDS describe one fit of one analysis: they go back to
        # automatic when either moves (probe: "Single exponential: {equation}" over a
        # linear fit). A first fit (none -> a model) keeps words typed ahead of it. The
        # label's numbers, place, size and colour are the look, and stay. A label left with no
        # words, numbers, place, size or colour of its own is the label nobody touched (an empty
        # ``fit_label``), so Reset figure pressed next has nothing to do.
        if now.wording_stale_from(was) or y_moved:
            label = style.fit_label_state()
            if label.template:
                cleared = _dc_replace(label, template="")
                style = (_dc_replace(style, fit_label=_NO_ENTRIES) if cleared == FitLabelState()
                         else style.with_fit_label(cleared))

        return _dc_replace(self, axes=axes, fit=fit, style=style)

    def with_pruned_legend(self, source_keys: Iterable[str],
                           offered_ids: Iterable[str]) -> "PlotConfig":
        """This config with legend decisions about things that no longer exist dropped.

        ``label_overrides`` and ``legend_include`` grew monotonically and were never
        pruned, which is two bugs in one: a retyped name and an unticked row for a source
        that has been closed sat in the style forever (and would be written into every
        saved template), and -- the visible half -- an exclusion made for ``cluster:3``
        was stranded across a re-clustering at k = 2, silently hiding a row of a DIFFERENT
        clustering that happened to reuse the id.

        THE TWO KINDS ARE PRUNED AGAINST DIFFERENT THINGS, because they are stable against
        different things. ``series:``/``fit:``/``label_overrides`` are keyed by SOURCE key,
        which survives unticking, a Refresh and a reopen -- so they are pruned only when
        the source itself is gone. ``cluster:`` ids are ordinals of one clustering run and
        mean nothing across runs, so they are pruned against what the last draw actually
        OFFERED.

        ...but only when that draw offered a clustering AT ALL. A histogram with no
        cluster split, or a walk over to the Scatter page, offers no ``cluster:`` row and
        is therefore not evidence about clustering -- pruning on it would quietly discard
        a decision the moment the user looked at another page. What supersedes a cluster
        exclusion is a DIFFERENT clustering, and that is exactly the draw this sees.
        """
        keys = {str(k) for k in source_keys}
        offered = {str(i) for i in offered_ids}
        judged_clusters = any(i.startswith("cluster:") for i in offered)

        def alive(entry_id: str) -> bool:
            kind, _, key = str(entry_id).partition(":")
            if kind in ("series", "fit"):
                return key in keys
            return entry_id in offered or not judged_clusters

        keep_label = {k: v for k, v in self.style.label_overrides.items() if k in keys}
        keep_include = {entry_id: included
                        for entry_id, included in self.style.legend_include.items()
                        if alive(entry_id)}
        # A legend row's Text style is keyed like its inclusion (a source's row by the
        # source key alone), so it is pruned by the same rule.
        keep_styles = {key: style for key, style in self.style.text_styles.items()
                       if not key.startswith("legend.entry.")
                       or alive(_legend_entry_id(key, keys))}
        if (keep_label == dict(self.style.label_overrides)
                and keep_include == dict(self.style.legend_include)
                and keep_styles == dict(self.style.text_styles)):
            return self
        return _dc_replace(self, style=_dc_replace(
            self.style, label_overrides=FrozenMap(keep_label),
            legend_include=FrozenMap(keep_include), text_styles=FrozenMap(keep_styles)))

    # -- A3: the TEMPLATE projection -------------------------------------------------------------
    #: What a saved template deliberately forgets, as names, so the tooltip beside the
    #: combo, :meth:`as_template` and ``tests/gui/test_plots_templates_json.py`` read ONE
    #: list instead of three copies that can disagree about where the seam is. Not a
    #: dataclass field (no annotation), so ``test_no_derived_state_ever_enters_the_config``
    #: still sees exactly the dataclass fields it walks.
    TEMPLATE_DROPS = ("selection.source_keys", "fit.applied", "fit.applied_k",
                      "fit.applied_k_auto", "fit.applied_scatter",
                      "style.label_overrides", "style.legend_include",
                      "style.text_styles[legend.entry.*]")

    def as_template(self) -> "PlotConfig":
        """This config as a REUSABLE TEMPLATE -- everything except what names THIS
        session's rows, or claims a computation was performed over them.

        THE SPLIT, AND WHY IT FALLS WHERE IT DOES. A template has to work on a recording
        it was not made on, so the question asked of each field is not "is it
        presentation?" but "does it still MEAN the same thing over other rows?". Exactly
        three groups fail that test:

        * ``selection.source_keys`` -- a source key is ``<doc_id>:<channel>`` of the
          documents THIS session has open. Applied elsewhere it either matches nothing
          (and silently unticks every source the user actually has) or matches by
          coincidence. Recalling a template must never move the tick boxes, so this is
          the one field :meth:`with_template` takes from the live window instead.
        * ``fit.applied`` / ``applied_k`` / ``applied_k_auto`` / ``applied_scatter`` --
          the fit the PICTURE IS SHOWING, as against the one the dropdowns propose. A
          template is a recipe, not a picture: restoring ``applied = "gamma"`` would badge
          the window as already showing a gamma fit of rows nothing has fitted, which is
          owner note #12's exact failure (read a k off a plot still drawing another
          model) re-created by the restore path. The PROPOSAL -- ``kind``/``k``/
          ``k_auto``/``scatter_kind`` -- IS carried, so a recalled template arrives with
          the right model selected and the Apply/Plot gate armed, and the user commits.
        * ``style.label_overrides`` and ``style.legend_include`` -- both keyed by
          something purely local: a source key, or a ``cluster:<i>`` ordinal that means
          nothing across two clustering runs (see :meth:`with_pruned_legend`). A legend
          entry retyped for "apo -500 mV" is not a legend entry for anything else. The
          legend rows' Text styles (``text_styles["legend.entry.…"]``) are keyed the same
          way and go with them; every other element style, and its words, is carried.

        EVERYTHING ELSE IS CARRIED, and three of those look like data selection and are
        not:

        * THE COLUMN NAMES (``histogram.column``, the scatter/KDE axes,
          ``cluster.features``). They are SCHEMA columns -- every event table this app
          produces carries ``dwell_s`` -- and they are the substance of a recipe: "a
          log-x dwell histogram under an exponential fit" is not a template without
          them. A blank one means "keep what the window has" and a column the recording
          does not offer is resolved back to the window's own, both at install time
          (:meth:`PlotsWindow._installable`), so neither kind can leave the identity
          anchor pointing at a column that is not on screen. (Blank columns do not make
          a template presentational-only: what does is a blank PLOT TYPE, see
          :meth:`with_template`.)
        * ``selection.accepted_only`` and ``histogram.include_filter_limited``. Neither
          names a row; both are POLICIES about rows ("accepted events only", "include the
          filter-limited ones") that mean the same thing on any recording. Dropping
          ``accepted_only`` would be the worse of the two available bugs: a recalled
          template that silently re-admits events the user rejected by hand.
        * the axis pins and the fit-truncation window. Both are numbers in a STATED unit
          (``axes.time_unit``, ``fit.range_unit`` travel with them), both are decisions
          the user made about a figure -- "fit the peak, not the tail", "every panel of
          figure 3 runs 0-2 ms" -- and reproducing a decision is what a named recipe is
          for. They are also VISIBLE in their own boxes the moment the template lands, so
          an unwanted one is one keystroke from being cleared; that visibility is the
          difference between these and the source keys, which would go wrong invisibly.
        """
        return _dc_replace(
            self,
            selection=_dc_replace(self.selection, source_keys=()),
            fit=_dc_replace(self.fit, applied=_FIT_NONE, applied_k=1,
                            applied_k_auto=False, applied_scatter=_SFIT_NONE),
            style=_dc_replace(self.style, label_overrides=_NO_ENTRIES,
                              legend_include=_NO_ENTRIES, series_styles=_NO_ENTRIES,
                              text_styles=_legend_styles(self.style.text_styles, False)))

    @property
    def carries_analysis(self) -> bool:
        """Whether this config, used as a TEMPLATE, names an analysis. A blank plot type says it
        does not: the template is a LOOK, and :meth:`with_template` takes only the style
        settings it names. Not a field, so it is neither saved nor compared; the shipped
        "Publication figure (light)" is the one config in the app that has it blank."""
        return bool(self.plot_type)

    def with_template(self, template: "PlotConfig") -> "PlotConfig":
        """``template`` recalled onto this config: the template's everything, over THIS
        window's data selection and legend memory.

        The inverse of :meth:`as_template`, written as one function rather than as a
        ``replace`` at the call site because the two halves have to agree about where the
        seam is: a field :meth:`as_template` drops has to be taken from ``self`` here, or
        recalling a template would install that field's DEFAULT -- unticking every source
        -- rather than leaving it alone.

        A BLANK COLUMN MEANS "KEEP WHAT YOU HAVE", not "no column". It keeps the identity
        anchor honest, because a config naming a column the pickers cannot select is
        exactly the stale-anchor shape :meth:`PlotsWindow._set_config` warns about.

        A BLANK PLOT TYPE MEANS THE TEMPLATE IS A LOOK, and that is what makes a purely
        presentational template possible. Blank columns were never enough: the plot type,
        the bins, the normalisation, the log axes, the fit, the Time unit and the typed
        ranges have no blank, so a template that left only the columns blank installed the
        dataclass defaults for all of them (the shipped "Publication figure (light)" turned
        a fitted log-log scatter into a 30-bin count histogram with no fit). A template
        with no plot type names no analysis at all: it is recalled as the style settings it
        names (:meth:`_with_look`), and every other field is the window's own. A saved
        template always has a plot type (the combo is never blank), so it is recalled whole,
        as it always was.
        """
        tpl = template.as_template()    # idempotent, and it sanitises a hand-edited file
        if not tpl.carries_analysis:
            return self._with_look(tpl.style)
        hist = tpl.histogram if tpl.histogram.column else _dc_replace(
            tpl.histogram, column=self.histogram.column)
        scatter = _dc_replace(tpl.scatter, x=tpl.scatter.x or self.scatter.x,
                              y=tpl.scatter.y or self.scatter.y)
        kde = _dc_replace(tpl.kde, x=tpl.kde.x or self.kde.x,
                          y=tpl.kde.y or self.kde.y)
        heatmap = _dc_replace(tpl.heatmap, x=tpl.heatmap.x or self.heatmap.x,
                              y=tpl.heatmap.y or self.heatmap.y)
        cluster = tpl.cluster if tpl.cluster.features else _dc_replace(
            tpl.cluster, features=self.cluster.features)
        return _dc_replace(
            tpl, histogram=hist, scatter=scatter, kde=kde, heatmap=heatmap,
            cluster=cluster,
            selection=_dc_replace(tpl.selection,
                                  source_keys=self.selection.source_keys),
            style=_dc_replace(tpl.style,
                              label_overrides=self.style.label_overrides,
                              legend_include=self.style.legend_include,
                              series_styles=self.style.series_styles,
                              text_styles=FrozenMap({
                                  **_legend_styles(tpl.style.text_styles, False),
                                  **_legend_styles(self.style.text_styles, True)})))

    def _with_look(self, look: StyleSettings) -> "PlotConfig":
        """This config with the style settings ``look`` NAMES, and nothing else changed.

        A setting is named when ``look`` moves it off its default: a look is a sparse
        value, and what it leaves at a default it leaves to the window. (So a look cannot
        say "back to the default" for a setting; the shipped one has no need to.) Only
        the single-valued settings are taken. The maps -- the element text styles, the fit
        label, the legend memory and the shared-style binding -- are the figure's own and
        stay, and the shipped look names no words either, as an appearance preset names none
        (``PlotAppearance.style_only``).

        A PAGE BOUND TO A SHARED STYLE resolves its look from the binding, so a setting
        written under it would be put straight back by the next projection. It is written
        the way every look gesture is (:meth:`PlotsWindow._presentation_changed`): as an
        override the binding then resolves, so the figure stays bound to its style.
        """
        blank = StyleSettings()
        named = {f.name: getattr(look, f.name) for f in _dc_fields(blank)
                 if not isinstance(getattr(blank, f.name), FrozenMap)
                 and getattr(look, f.name) != getattr(blank, f.name)}
        if all(getattr(self.style, name) == value for name, value in named.items()):
            return self
        return _dc_replace(self, style=_project_style_settings(
            _dc_replace(self.style, **named), edit=True))


# ================================================================================================
# THE CONFIG ON DISK -- a versioned, standalone .json sidecar (owner item B3)
# ================================================================================================
#
# WHY A SIDECAR AND NOT THE PROJECT FILE. A ``.nanoscope`` project is a session -- these
# documents, these calibrations, these curated event tables. A plot configuration is a
# FIGURE RECIPE, and the thing the owner wants to do with one is send it to somebody, or
# apply it to next week's recording, neither of which involves the 180 MB of signal the
# project is a reference to. So it is its own small file, readable in a text editor, and
# it carries the whole config -- source keys included, which is the one difference from a
# TEMPLATE (:meth:`PlotConfig.as_template`): a template is deliberately portable, an
# exported config is deliberately exact, and the restore is silent about source keys this
# session does not have (:meth:`PlotsWindow._apply_checked_keys`).

PLOT_CONFIG_FORMAT: str = "nanoscope-plot-config"

# WHY THERE IS A VERSION AT ALL, AND WHY AN ADDITIVE KEY MUST STILL BUMP IT.
#
# This is ``nanoscope.core.project``'s discipline, applied to a much smaller file, and it
# is restated here rather than merely referenced because the failure it prevents is
# SILENT and the temptation to skip the bump is strongest exactly when the change looks
# harmless.
#
# Additive-ness governs whether THIS build can read OLDER files. It says nothing about
# whether an older build can safely read THIS one -- and it cannot. A build that predates
# a key accepts the file, ignores the key it does not know, and then DROPS IT on its next
# export, without a word, because it never knew the key was there. The user's typed axis
# labels, their legend exclusions, their fit-truncation window: gone from the file they
# just re-saved, with nothing on screen having said so. That is precisely the loss the
# config object exists to end, reintroduced one level up and made worse by living in a
# file that gives no sign of it.
#
# The bump converts that silent strip into :class:`PlotConfigVersionError`'s clean
# refusal, and it costs nothing in the other direction: ``MIN_READ_VERSION`` is 1, so
# every file this app has ever written still opens. The reader is additive by
# construction -- :func:`_decode_dc` walks the DATACLASS's fields, not the payload's
# keys, so a key a future build adds is ignored here and a key an older file omits reads
# as that field's default rather than as an error.
#
# v1 is the first shape: format/version/app_version, ``plot_type``, ``norm_user_set`` and
# the eight nested sections. (Named PLOT_CONFIG_* rather than the bare VERSION/
# MIN_READ_VERSION ``project`` uses, because this module is not about one file format --
# a bare ``VERSION`` in a 7,600-line GUI module is a name asking to be misread.)
#
# v2 IS THE SAME SHAPE WITH ONE WORD RESPELLED: S6-03 renamed the histogram's
# ``normalization`` from "density (area = 1)" to "density", and that string is a persisted
# VALUE, not a label. The fix shipped without a bump, on the argument that the FORMAT had
# not changed. It had not -- and the rule this block opens with is not about the format,
# it is about what an older build does with a file this one writes, which is the half
# additive-ness says nothing about. Measured, with a real sidecar this build wrote and the
# frozen pre-rename build reading it (the file re-stamped "version": 1, which is what that
# build was handed before this bump): `import_config_json` returned **True**, the combo
# stayed on "count", the y axis said "count", Details said "y = count", and the window drew
# the count histogram -- 982.0 events summed as integers -- where the file's author had
# saved the density that sums to 1628.66 (1/s). That is the silent strip this constant
# exists to convert into a refusal, and a renamed enum value causes it just as surely as a
# dropped key: the old build recognises the word "normalization" and not the word
# "density", so it ignores the value and carries on.
#
# WHAT THE BUMP BUYS AND WHAT IT COSTS. An older build now says "This plot configuration
# is version 2; this build of NanoScope reads versions 1 to 1. Update NanoScope to open
# it." -- by name, with the number, before drawing anything. The cost is that it says that
# about EVERY v2 sidecar, including one whose histogram is a plain count and would have
# read back perfectly. Declaring the version per payload ("2 only when this file says
# density") was considered and rejected: it would make the version mean something
# different from what it means in `project`, and it would be a second rule that has to
# stay in agreement with `_LEGACY_NORMS` -- the exact shape of the defect this round is
# here to close. A version is a statement about the WRITER's vocabulary, and this writer's
# vocabulary changed.
#
# Nothing is lost in the direction that matters: MIN_READ stays 1, and a v1 sidecar
# written by the pre-rename build still opens here and draws the numbers it drew there
# (proved on a real exported file, not a hand-built payload -- see
# `test_a_real_v1_sidecar_from_the_pre_rename_build_still_opens`). The same goes for a
# QSettings template written before the rename; and a stored template a build cannot read
# is refused with a message naming the version (:meth:`PlotsWindow._template_config`),
# never swallowed.
#
# v3 ADDS, AND BUMPS FOR THE SAME REASON: ``histogram.kde_overlay``, a ``heatmap`` section,
# ``style.heatmap_colormap`` and the plot type "Event-count heatmap". A v2 build reading a
# v3 file would ignore all four and re-save a plain histogram (or fall back off a plot
# type it has not got) over the sender's KDE overlay and heatmap, with nothing said -- the
# silent strip this constant converts into a refusal. Every v1 and v2 file still opens:
# the four keys read as their defaults (overlay off, heatmap defaults, viridis).
PLOT_CONFIG_VERSION: int = 4
#: The oldest sidecar this build can still read. v1 is the first, so this is 1 and stays
#: 1 until a change is genuinely un-migratable -- raising it is a promise to the user that
#: their older files are unopenable, which no additive key ever justifies.
PLOT_CONFIG_MIN_READ_VERSION: int = 1


class PlotConfigError(ValueError):
    """A plot-configuration file could not be read (not one, corrupt, or unreadable version)."""


class PlotConfigVersionError(PlotConfigError):
    """A plot-configuration file declares a version this build cannot read."""


#: The nested sections of a :class:`PlotConfig`, by the key they are written under. The
#: dict is the SCHEMA for both directions, so a section cannot be written without being
#: read back (which is the shape of every "we forgot to persist half of it" bug).
_CONFIG_SECTIONS: dict[str, type] = {
    "selection": SelectionConfig,
    "histogram": HistogramConfig,
    "scatter": ScatterConfig,
    "kde": KdeConfig,
    "heatmap": HeatmapConfig,
    "cluster": ClusterConfig,
    "fit": FitConfig,
    "axes": AxesConfig,
    "style": StyleSettings,
}


def _encode_dc(obj) -> dict:
    """One flat frozen dataclass -> a JSON-safe dict, driven by its own FIELDS.

    Field-driven rather than hand-listed, because a hand-listed encoder is how half a
    config comes to be persisted: add ``show_provenance`` to :class:`StyleSettings` and a
    listed encoder keeps writing files that silently lack it, while this one carries it
    the moment it exists. Only three shapes need translating -- a :class:`FrozenMap`
    becomes a plain dict, a tuple becomes a list (JSON has no tuple), and everything else
    is already a str/int/bool/None.
    """
    out: dict = {}
    for f in _dc_fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, FrozenMap):
            # Thawed all the way down: the style's text and fit-label maps nest.
            out[f.name] = _thawed(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def _decoded(default, raw):
    """``raw`` from JSON, coerced to the shape ``default`` (the field's own default) has.

    THE DEFAULT IS THE TYPE MAP. Reading the coercion off the live dataclass rather than
    off a second table is what keeps the two from drifting; a field added tomorrow decodes
    correctly with no edit here.

    ``str`` LAST AND VERBATIM, which matters for exactly four fields: ``fit.range_lo`` /
    ``range_hi`` and the four ``axes`` boxes are the USER'S TYPED TEXT, not numbers.
    "nan", "inf" and "abc" are legal values there -- ``_fit_range`` declines them and
    ``_fit_range_input_notes`` quotes them back to the user -- so a round trip that parsed
    them would destroy the very text the message is about.
    """
    if isinstance(default, FrozenMap):
        return _deep_frozen(raw or {})
    if isinstance(default, tuple):
        return tuple(raw)
    if default is None:
        # Optional pairs (`legend_pos`, `manual_x`, `manual_y`). Other optional types
        # are dispatched explicitly by field in _decode_dc before reaching this path.
        return None if raw is None else tuple(float(x) for x in raw)
    if isinstance(default, bool):
        return bool(raw)        # BEFORE int: bool is a subclass of it
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        if isinstance(raw, bool):
            raise ValueError("A style size must be a finite number")
        return float(raw)
    if isinstance(default, str):
        return str(raw)
    return raw


def _decode_dc(cls, payload: dict):
    """A JSON dict -> one flat frozen dataclass, ADDITIVELY.

    Walks ``cls``'s fields, not the payload's keys: a key an older file omits reads as
    that field's default (which is what makes MIN_READ_VERSION honest), and a key a newer
    file carries that this build has no field for is ignored rather than fatal.

    A section that is not an object at all is a different thing from a section that is
    absent, and is refused: ``"axes": ["not", "a", "section"]`` would otherwise read as
    "no axes keys" and restore the DEFAULTS in silence, which is a malformed file
    quietly changing the user's axes rather than being reported.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__.lower()} section is "
                        f"{type(payload).__name__}, not an object")
    blank = cls()
    kw = {}
    for field in _dc_fields(blank):
        if field.name not in payload:
            continue
        raw = payload[field.name]
        if cls is StyleSettings and field.name == "frame":
            if raw is not None and type(raw) is not bool:
                raise ValueError("Style frame must be true, false or inherited")
            kw[field.name] = raw
        else:
            kw[field.name] = _decoded(getattr(blank, field.name), raw)
    return _dc_replace(blank, **kw) if kw else blank


def _project_style_settings(style: StyleSettings, *, edit=False) -> StyleSettings:
    """The shared binding owns its native renderer projection, including body size."""
    if not style.style_binding:
        return style
    from .figure_style_binding import project_native, reconcile_native_edit
    raw = _encode_dc(style)
    projected = project_native(raw, "style_settings")
    if edit:
        projected = reconcile_native_edit(projected, raw, "style_settings")
    return _decode_dc(StyleSettings, projected)


#: The Style keys older builds wrote that this one has no field for (the duplicate fit-number
#: switches and Title weight). A saved configuration that carries one still opens and looks
#: the same: :func:`_fold_legacy_style` moves it to its new home, and nothing writes the key
#: again. The project format is not bumped for it -- an older build opening a newer file
#: reads these keys as their defaults, and the folded look is in keys it already knows.
_LEGACY_STYLE_KEYS = ("fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic")
_LEGACY_STYLE_PATHS = frozenset(f"style.{name}" for name in _LEGACY_STYLE_KEYS)


def _fold_legacy_style(style: StyleSettings, saved: Mapping) -> StyleSettings:
    """``style`` with the switches an older build kept in ``saved`` (its Style mapping,
    JSON or a project's tagged tree) moved to where they live now.

    * THE FIT LABEL'S NUMBERS. The label showed its fields, less R² when R² on figure was
      off and less the KS statistic when KS statistic on figure was off. Fit values on plot
      added BIC and KS only to a label nobody had saved: a saved one (a drag, any Fit label
      edit) showed its own fields whatever that box said, so it is not asked again here.
      The user's own wording is taken the same way: a ``{r2}`` or ``{ks}`` in it printed a
      dash while its switch was off, so it becomes that dash (no field is left to say it).
    * TITLE WEIGHT is the title's own text style: bold and italic, each wherever the title's
      style has not set that one itself.

    Read the way the old decoder read them (``bool(value)``), so a file means what it did.
    """
    numbers = bool(saved.get("fit_values_on_plot", False))
    no_r2 = not bool(saved.get("show_r2", True))
    no_ks = not bool(saved.get("show_gof", True))
    if numbers or no_r2 or no_ks:
        kept = bool(saved.get("fit_label"))
        label = style.fit_label_state() if kept else FitLabelState()
        fields = set(label.fields)
        template = label.template
        if numbers and not kept:
            fields |= {"bic", "ks"}
        if no_r2:
            fields.discard("r2")
            template = template.replace("{r2}", "—")
        if no_ks:
            fields.discard("ks")
            template = template.replace("{ks}", "—")
        fields = tuple(name for name in FIT_FIELDS if name in fields)
        if (fields, template) != (label.fields, label.template):
            style = style.with_fit_label(_dc_replace(label, fields=fields, template=template))
    for key, attr in (("title_bold", "bold"), ("title_italic", "italic")):
        if bool(saved.get(key, False)):
            title = TextStyle.from_mapping(_thawed(style.text_styles.get("title", {})))
            if getattr(title, attr) is None:
                style = _dc_replace(style, text_styles=style.text_styles.set(
                    "title", _dc_replace(title, **{attr: True}).to_dict()))
    return style


def plot_config_to_json(cfg: PlotConfig) -> dict:
    """A :class:`PlotConfig` as the JSON-safe dict a sidecar file holds."""
    from .. import __version__ as _app_version   # local: the package imports this module

    payload: dict = {
        "format": PLOT_CONFIG_FORMAT,
        "version": PLOT_CONFIG_VERSION,
        "app_version": _app_version,
        "plot_type": cfg.plot_type,
        "norm_user_set": bool(cfg.norm_user_set),
    }
    # A LOOK (:attr:`PlotConfig.carries_analysis`) names no analysis, so it is stored without
    # one: the style, and any analysis section it was given away from the defaults. The reader
    # takes a missing section as the defaults, so nothing stored has to be read differently.
    sparse = not cfg.carries_analysis
    blank = PlotConfig() if sparse else cfg
    if sparse and not cfg.norm_user_set:
        del payload["norm_user_set"]
    for name in _CONFIG_SECTIONS:
        if sparse and name != "style" and getattr(cfg, name) == getattr(blank, name):
            continue
        payload[name] = _encode_dc(getattr(cfg, name))
    return payload


def plot_config_from_json(payload) -> PlotConfig:
    """The inverse of :func:`plot_config_to_json`. Raises :class:`PlotConfigError` only.

    ONE EXCEPTION TYPE, for :mod:`nanoscope.core.project`'s reason: the caller catches one
    class and shows the user one message, so a payload malformed in a way nobody thought
    of ("axes" written as a list, "bins" as a dict) has to arrive as this and not as
    whatever the decoding expression happened to raise.
    """
    if not isinstance(payload, Mapping):
        raise PlotConfigError(
            f"This is not a plot configuration: the file holds "
            f"{type(payload).__name__}, not an object.")
    fmt = payload.get("format")
    if fmt != PLOT_CONFIG_FORMAT:
        raise PlotConfigError(
            f"This is not a NanoScope plot configuration (its “format” is "
            f"{fmt!r}, not {PLOT_CONFIG_FORMAT!r}).")
    raw_version = payload.get("version")
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        raise PlotConfigError(
            f"This plot configuration does not say which version it is "
            f"(“version” is {raw_version!r}).") from None
    # NAMED IN THE MESSAGE, BOTH WAYS. "Cannot read this file" sends the user hunting; the
    # version they have and the versions this build reads is a sentence they can act on.
    if version > PLOT_CONFIG_VERSION or version < PLOT_CONFIG_MIN_READ_VERSION:
        raise PlotConfigVersionError(
            f"This plot configuration is version {version}; this build of NanoScope "
            f"reads versions {PLOT_CONFIG_MIN_READ_VERSION} to {PLOT_CONFIG_VERSION}. "
            + ("Update NanoScope to open it."
               if version > PLOT_CONFIG_VERSION
               else "It was written by a build too old to be read."))
    try:
        cfg = PlotConfig(
            plot_type=str(payload.get("plot_type", PlotConfig().plot_type)),
            norm_user_set=bool(payload.get("norm_user_set", False)),
            **{name: _decode_dc(cls, payload.get(name) or {})
               for name, cls in _CONFIG_SECTIONS.items()})
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        detail = f"missing key {exc}" if isinstance(exc, KeyError) else str(exc)
        raise PlotConfigError(
            f"This plot configuration is malformed ({detail}).") from exc
    # THE SECOND MIGRATION RUNS HERE, because it has to: the switches it folds have no field
    # left for a ``__post_init__`` to canonicalise, and the decoding above has dropped them.
    saved_style = payload.get("style")
    if isinstance(saved_style, Mapping) and any(key in saved_style for key in _LEGACY_STYLE_KEYS):
        cfg = _dc_replace(cfg, style=_fold_legacy_style(cfg.style, saved_style))
    # THE ONE MIGRATION THIS FORMAT HAD UNTIL THEN (S6-03's rename) HAS ALREADY RUN BY HERE, and
    # deliberately not in this function. `_decode_dc` above builds each section by calling
    # its dataclass, so a v1 file's "density (area = 1)" is canonicalised by
    # `HistogramConfig.__post_init__` before this line is reached -- which is what lets a
    # config that never touched a file (the `apply_config` a notebook makes with
    # `dataclasses.replace`) get the same treatment as one that did. A copy of the
    # translation here would be a second place to keep in agreement, and the defect this
    # module has already paid for once was exactly that: a migration that lived at some of
    # the doors. `test_a_v1_sidecar_naming_the_old_density_still_selects_density` pins the
    # file half regardless of which layer does the work.
    return cfg


def _flat_config_json(payload: Mapping, prefix: str = "") -> dict[str, object]:
    """A sidecar payload flattened to dotted keys, so two of them can be compared.

    Walked rather than listed, for :func:`_encode_dc`'s reason: a field added to a config
    dataclass tomorrow is compared here with no edit. A mapping with keys is walked;
    everything else -- INCLUDING AN EMPTY MAPPING, which has no keys to walk and would
    otherwise vanish from the comparison entirely -- is a leaf.
    """
    out: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.update(_flat_config_json(value, name + "."))
        else:
            out[name] = value
    return out


def _loss_value_text(want, got) -> str:
    """What to quote for one lost setting: the value the FILE asked for.

    A LIST QUOTES ONLY THE ENTRIES THAT ARE GONE, because the useful half of
    ``["apo -700 mV", "a run this session never loaded"]`` losing its second entry is the
    second entry -- printing both would make the user diff two lists in their head. When
    nothing is missing but the lists still differ (a reorder), the whole asked-for value
    is quoted, because then it is the shape that changed and no single entry explains it.
    """
    if isinstance(want, list) and isinstance(got, list):
        gone = [str(v) for v in want if v not in got]
        if gone:
            text = ", ".join(gone)
            return text[:_LOSS_VALUE_CHARS - 1] + "…" if len(text) > _LOSS_VALUE_CHARS else text
    text = str(want)
    return text[:_LOSS_VALUE_CHARS - 1] + "…" if len(text) > _LOSS_VALUE_CHARS else text


def _config_round_trip_losses(payload: Mapping, asked: PlotConfig,
                              installed: PlotConfig) -> list[str]:
    """Everything ``payload`` named that ``installed`` does not carry, as ``key “value”``.

    ``asked`` is what ``payload`` DECODES to and is the pivot both halves are measured
    against; ``installed`` is what the window reports after writing it. See
    :class:`_ConfigLosses` for what this is for, what the two halves are, and the
    measurements that show the pivot keeps the format's one migration and its three
    envelope keys out of the count.

    SORTED BY KEY, not by file order: a dict's order is whatever the sender's editor left
    behind, and a sentence that names the same seven losses in a different order on two
    machines is a sentence nobody can quote at a colleague.
    """
    raw = _flat_config_json(payload)
    pivot = _flat_config_json(plot_config_to_json(asked))
    now = _flat_config_json(plot_config_to_json(installed))
    losses: list[str] = []
    for key, value in raw.items():
        # THE FORMAT HAS NOWHERE TO PUT IT. `_decode_dc` walks this build's FIELDS and
        # ignores a key it has none for -- additively and on purpose, so a newer build's
        # file still opens -- which is exactly why the drop has to be said out loud here.
        # An older build's switch that `_fold_legacy_style` has moved was taken, not lost; and
        # so was the empty label or element-style map it moved into (that key is now many).
        moved_into = value == {} and any(path.startswith(key + ".") for path in pivot)
        if key not in pivot and key not in _LEGACY_STYLE_PATHS and not moved_into:
            losses.append(f"{key} “{_loss_value_text(value, None)}”")
    for key, want in pivot.items():
        # THE WINDOW WOULD NOT, OR COULD NOT, TAKE IT. A column these events have not
        # got, a fit or a y quantity this build does not carry, a source this session
        # never loaded: `_set_config` and `_installable` each decline some of these for
        # documented reasons, and the reasons are all "do the sensible thing", which is
        # only defensible while the doing is disclosed.
        if key in now and now[key] != want:
            losses.append(f"{key} “{_loss_value_text(want, now[key])}”")
    return sorted(losses)


def _load_plot_config(path: str | Path) -> tuple[Mapping, PlotConfig]:
    """Read a ``.json`` sidecar as BOTH its raw payload and the config it decodes to.

    The payload is what :func:`_config_round_trip_losses` needs and the config cannot
    give: a key this build has no field for is dropped by :func:`_decode_dc`, so by the
    time there is a :class:`PlotConfig` the evidence that the key was ever in the file is
    gone. ONE read, not two -- re-opening the file to look at it again would be looking
    at a different file if anything touched it in between.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlotConfigError(f"{p.name} could not be read ({exc}).") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PlotConfigError(f"{p.name} is not valid JSON ({exc}).") from exc
    try:
        return payload, plot_config_from_json(payload)
    except PlotConfigError as exc:
        # Re-raised with the FILE NAME on the front, keeping the class (so a version
        # refusal still reads as one) -- the user chose a file, so the message has to say
        # which file it is about.
        raise type(exc)(f"{p.name}: {exc}") from exc


def read_plot_config(path: str | Path) -> PlotConfig:
    """Read a ``.json`` sidecar. Raises :class:`PlotConfigError` for every failure."""
    return _load_plot_config(path)[1]


def write_plot_config(path: str | Path, cfg: PlotConfig) -> Path:
    """Write ``cfg`` to ``path`` as a versioned JSON sidecar. STAGED, like every export.

    Staged for the reason USER_GUIDE 12 gives for the others: a failure half-way through
    must leave the previous file whole rather than truncated. Two spaces of indent and a
    trailing newline because this file is meant to be opened in an editor and diffed.
    """
    from ..core.atomic_write import staged_write

    out = Path(path)
    body = (json.dumps(plot_config_to_json(cfg), indent=2, ensure_ascii=False)
            + "\n").encode("utf-8")
    with staged_write(out) as fh:
        fh.write(body)
    return out


class _ConfigHistory:
    """A bounded undo/redo stack of :class:`PlotConfig` VALUES, local to one window.

    WHY NOT :class:`nanoscope.core.history.EventHistory`, WHICH ALREADY EXISTS. The recon
    proposed a local stack and that is right, for three reasons -- and the third is the
    one that would have bitten:

    1. **Different object, different key.** ``EventHistory`` is keyed
       ``(doc_id, channel, module_id)`` and its own docstring gives the argument for that:
       with four voltages open, one flat stack means Ctrl+Z silently repairs a table on a
       recording that is not on screen. A plot config has no such key -- it spans every
       checked source at once, and half its content (a font, a title) belongs to no
       recording at all. There is nothing to file it under.
    2. **Different cost, so a different mechanism is affordable.** ``EventHistory`` had to
       invert to a pristine-copy-and-DIFF design because eleven call sites mutate a shared
       DataFrame in place and the twelfth would forget to announce itself; the price it
       pays is a 38.7 MiB deep copy and a 128 MiB session ceiling. Here the config is
       ALREADY a frozen value -- measured on a loaded window, ``config()`` costs 20 us,
       ``==`` costs 0.4 us, and one whole pass of the hook (diff, buttons, template
       name) costs 71 us, which is a tenth of a frame on the keystroke path it runs on
       -- so the same forget-proof inversion is available for nothing. This class therefore
       borrows the inversion (see :meth:`note`: a caller says "something moved", never
       "here is the step") and none of the machinery that paid for it.
    3. **The two must not share a stack, or a menu, or a shortcut.** Ctrl+Z in the Plots
       window has to undo the colour the user just picked. If plot configuration were
       pushed onto the session's ``EventHistory``, Ctrl+Z would sometimes un-reject an
       event on a recording behind this window instead -- two different objects, two
       different user intents, interleaved in one order by the accident of which was
       touched last. That is worse than either alone.

    What IS borrowed, deliberately: the diff-not-announce inversion, an explicit bound
    with a counter for what it threw away, and the rule that the stack dies with the data
    it describes (:meth:`PlotsWindow.reset_state`).

    THE SHAPE. ``base`` is the configuration the widgets are believed to be showing;
    ``_past`` is what came before it, oldest first; ``_future`` is what an undo stepped
    off. Nothing here touches Qt, so the whole rule is testable without a window.

    EVERY ENTRY ALSO CARRIES ITS ORIGIN: which whole-configuration install the state
    descends from. An ordinary edit inherits the base's origin; an install from
    OUTSIDE the window's own lineage -- an imported sidecar, a template recall, a
    script's ``apply_config`` -- opens a new one (:meth:`open_origin`), and undo/redo
    carry the origin along with the configuration. That is the one fact the two import
    disclosures (:class:`_NormRefusal`, :class:`_ConfigLosses`) hang on: a sentence
    about a file stands while the state on screen descends from that file's install,
    whichever writer put the state there, and not otherwise. Kept HERE rather than on
    the records because the stack is the only object that knows, after an undo, where
    the window's state came from; a record deciding that from config VALUES cannot
    tell "undone back onto the import" from "undone past it".
    """

    __slots__ = ("_base", "_past", "_future", "_depth", "dropped",
                 "_origin", "_past_origins", "_future_origins", "_pending_origin",
                 "_opened")

    def __init__(self, base: PlotConfig, depth: int = _HISTORY_DEPTH) -> None:
        self._base = base
        self._past: list[PlotConfig] = []
        self._future: list[PlotConfig] = []
        self._depth = int(depth)
        #: Steps thrown away at the depth bound, ever. Counted rather than merely dropped
        #: because a silent bound is indistinguishable from a broken Undo button.
        self.dropped = 0
        #: The origin of ``base``, of each ``_past``/``_future`` entry (parallel lists),
        #: the origin an install has opened but no step has been noted under yet, and
        #: the count of origins ever opened (the next id).
        self._origin: int = 0
        self._past_origins: list[int] = []
        self._future_origins: list[int] = []
        self._pending_origin: int | None = None
        self._opened: int = 0

    @property
    def base(self) -> PlotConfig:
        """The configuration this stack believes the window is showing."""
        return self._base

    @property
    def origin(self) -> int:
        """Which install the state ON SCREEN descends from.

        The pending origin while one is open -- an install has written the widgets and
        the draw that reports on them runs BEFORE the step is noted, so a reader that
        waited for the note would hide the import's own disclosure on the very draw the
        import makes (measured: the draw inside ``_set_config`` precedes
        ``_history_settle``) -- and the base's origin otherwise.
        """
        return self._pending_origin if self._pending_origin is not None else self._origin

    def open_origin(self) -> int:
        """A whole configuration from outside this window's lineage is being installed:
        start a new origin and return its id. The next :meth:`note` files the resulting
        state under it (even a note that moves nothing, so an install that happens to
        equal the current state still changes what the state descends from)."""
        self._opened += 1
        self._pending_origin = self._opened
        return self._opened

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    def depth(self) -> tuple[int, int]:
        """``(steps back, steps forward)`` -- for the buttons and for tests."""
        return len(self._past), len(self._future)

    def note(self, now: PlotConfig) -> bool:
        """Record that the window has moved from :attr:`base` to ``now``. True if it had.

        The DIFF is the whole mechanism: a caller that says "something may have moved" too
        often costs one comparison, and a caller that forgets to say it at all still gets
        the change recorded by the next caller that does -- coarser, never wrong. Nothing
        is pushed when the window came back to where it was, which is why walking a combo
        away and back does not fill the stack with pairs.
        """
        pending, self._pending_origin = self._pending_origin, None
        if now == self._base:
            if pending is not None:
                self._origin = pending
            return False
        self._past.append(self._base)
        self._past_origins.append(self._origin)
        self._base = now
        # The new state inherits its origin from the state it was reached from, unless
        # an install opened a new one -- see `open_origin`.
        if pending is not None:
            self._origin = pending
        # A NEW STEP DESTROYS THE REDO BRANCH. Keeping it would offer the user a "forward"
        # into a configuration that is no longer reachable from where they are.
        self._future.clear()
        self._future_origins.clear()
        while len(self._past) > self._depth:
            self._past.pop(0)
            self._past_origins.pop(0)
            self.dropped += 1
        return True

    def undo(self) -> PlotConfig | None:
        """Step back one configuration, or None when there is nothing behind."""
        if not self._past:
            return None
        self._future.append(self._base)
        self._future_origins.append(self._origin)
        self._base = self._past.pop()
        self._origin = self._past_origins.pop()
        return self._base

    def redo(self) -> PlotConfig | None:
        """Step forward one configuration, or None when there is nothing ahead."""
        if not self._future:
            return None
        self._past.append(self._base)
        self._past_origins.append(self._origin)
        self._base = self._future.pop()
        self._origin = self._future_origins.pop()
        return self._base

    def rebase(self, cfg: PlotConfig) -> None:
        """Move the base WITHOUT recording a step.

        For the one caller that has just installed a config itself: after an undo the
        window shows what the undo produced, and noting that as a fresh change would push
        the state we just left back on and make Undo a toggle between two entries.
        """
        self._base = cfg

    def reset(self, base: PlotConfig) -> None:
        """Forget everything and start again from ``base``.

        For :meth:`PlotsWindow.reset_state`, on ``EventHistory.clear``'s argument: a stack
        that outlived the data it describes could be applied to nothing, so Undo would
        name a step and then appear to do nothing.
        """
        self._base = base
        self._past.clear()
        self._future.clear()
        self._past_origins.clear()
        self._future_origins.clear()
        # ...AND THE ORIGIN: a fresh one, so nothing recorded against the old lineage
        # can match the blank window (`reset_state` clears those records too).
        self._opened += 1
        self._origin = self._opened
        self._pending_origin = None
        # AND THE LEDGER, because what it counted were losses of THIS data's history.
        # ``EventHistory.clear`` does exactly this to ``_depth_dropped`` (pinned by
        # tests/gui/test_s5_persistence_undo_depth_disclosure.py), and the argument is
        # the same one door down: the only caller here is
        # :meth:`PlotsWindow.reset_state`, which runs on close and blanks the window, so
        # a surviving count would greet the next session's first figure with a warning
        # about the last one's.
        self.dropped = 0


class _PlotsTextHost:
    """The Plots window's ``figure_text.FigureTextHost``: in-figure text and axis edits.

    A WEAK reference to the window, as every per-instance wrap of the controller holds:
    the controller lives on the plot, and the plot must not keep a closed window alive.
    Every write is the window's (:meth:`PlotsWindow._apply_figure_text`,
    :meth:`PlotsWindow._apply_figure_axis`), so it lands in StyleSettings / AxesConfig and
    is ONE step of this window's own Undo, exactly as the same edit made in the panel.
    """

    def __init__(self, window: "PlotsWindow") -> None:
        self._window = weakref.ref(window)

    def _live(self):
        window = self._window()
        if window is None:
            raise RuntimeError("the Plots window is gone")
        return window

    def text_targets(self) -> dict:
        window = self._window()
        return {} if window is None else window._figure_text_targets()

    def apply_text(self, key: str, style: TextStyle) -> None:
        self._live()._apply_figure_text(key, style)

    def apply_text_run(self, key: str, style: TextStyle) -> None:
        """One write of a RUN (a size spin's arrow clicks): the same edit as
        :meth:`apply_text`, but the Undo step is held open by the window's text grain
        (:meth:`PlotsWindow._history_touch_text`: closed by focus-out or a 700 ms pause)
        instead of being closed at once, so five clicks are one step."""
        self._live()._apply_figure_text(key, style, run=True)

    def commit_text_run(self) -> None:
        """Finish a Format spin edit when focus leaves the control."""
        self._live()._history_commit()

    def apply_format_styles(self, styles: Mapping[str, TextStyle], *,
                            auto_range_of: str = "") -> None:
        """Reset or paste several element styles as one configuration-history step."""
        window = self._live()
        if auto_range_of not in ("", "bottom", "left"):
            raise ValueError(f"Unknown axis: {auto_range_of}")
        with window._format_transaction():
            for key, style in styles.items():
                window._apply_figure_text(key, style)
            if auto_range_of:
                window._reset_axis_range(axis="x" if auto_range_of == "bottom" else "y")

    def axis_state(self, side: str) -> AxisState:
        return self._live()._figure_axis_state(side)

    def apply_axis(self, side: str, values: AxisState) -> None:
        self._live()._apply_figure_axis(side, values)

    def menu_entries(self, key: str) -> list:
        return []

    def background_entries(self) -> list:
        window = self._window()
        return [] if window is None else window._figure_background_entries()


class _DecadeLabelAxis(ReadableLogAxis):
    """The dwell/depth plots' log axis: :class:`~nanoscope.gui.axes.ReadableLogAxis`
    with this window's notation.

    Everything structural -- no two labels touching, decades never lost to a minor
    label, no pyqtgraph SI prefix inventing a ``(x0.001)`` multiplier, and the
    window-owned :attr:`~nanoscope.gui.axes.ReadableLogAxis.display_factor` that shows
    seconds as milliseconds without touching the plotted log10 values -- lives in the
    base class, shared with the PSD window. Only the NOTATION is decided here.

    THE NOTATION RULE: plain decimals while every labelled decade lands in a readable
    range, exponents otherwise, never a mix. ``AxisItem.logTickStrings`` formats each
    decade independently with ``%0.1g``, which flips to exponent notation at 10 and at
    1e-5; measured on the owner's -700 mV dwell histogram the labels read
    ``0.01, 0.1, 1, 10^1, 10^2`` -- two notations on one axis, so ``1`` and ``10^1``
    sat two ticks apart looking like different KINDS of number (report High-7). The
    choice is made once for the whole axis, from every decade at once.

    WHAT CHANGED AFTER THE FIRST FIX. This class used to return ``""`` for every
    non-decade tick. That removed the smear, and it introduced two defects of its own,
    both measured on the -600 mV merged dwells through the real window:

    * each of the five decade labels was painted TWICE at identical coordinates (10 text
      specs at 5 unique positions), because pyqtgraph emits the decades on more than one
      tick level and blanking the minors does nothing about that;
    * an axis zoomed to 15-25 ms -- inside one decade, astride the median dwell of all
      three recordings -- drew ZERO labels.

    Emitting every label and dropping only the ones that would collide fixes both, and
    is what the PSD window was already doing.
    """

    def choose_notation(self, exponents):
        return bool(exponents) and all(-4.0 <= e <= 4.0 for e in exponents)


#: The second half of the large-SVG warning: what to do instead. ``pg_svg`` writes the
#: measured first half (how many points, how many MB, how many seconds) and ends by
#: offering PNG, because when it was written PNG was the only alternative this panel
#: had. It now has a better one, and it is a VECTOR one -- measured on this window, a
#: 10 000-point scatter is 5.28 MB / 3.06 s as SVG and 0.151 MB / 0.10 s as PDF, the
#: same picture both times. Kept here rather than pushed into ``pg_svg`` because it is
#: a statement about what THIS panel offers, and ``pg_svg`` is a workaround that should
#: be deletable the day pyqtgraph fixes its exporter.
_SVG_SIZE_ALTERNATIVE = (
    "Saving under a .pdf name instead gives the same vector art in a fraction of the "
    "size and the time (measured here: 5.3 MB and 3.1 s as SVG against 0.15 MB and "
    "0.1 s as PDF, for one 10 000-point scatter), and every vector editor opens it. "
    "Cancel and choose a .pdf or .png name if you would rather have one of those."
)


def confirm_large_svg(parent: QtWidgets.QWidget | None, message: str) -> bool:
    """Ask before writing an SVG big enough to be a problem. True to go ahead.

    MODULE-LEVEL, not a method, for the reason :func:`export_paths.confirm_overwrite`
    is: a test has to be able to substitute it without a live modal.

    Two buttons and no third "save it as PNG for me" option, deliberately. Rewriting
    the extension the user just chose would write a file that the overwrite
    confirmation never named -- which is the exact defect ``gui/export_paths.py``
    exists to prevent -- so the alternative is offered as ADVICE and the user goes
    back through the dialog to act on it. The default is Cancel: the user asked for a
    picture, not for a nine-second wait they did not know about.
    """
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle("Save plot image")
    box.setText("This plot is large for SVG.")
    box.setInformativeText(f"{message}\n\n{_SVG_SIZE_ALTERNATIVE}")
    go = box.addButton("Export SVG anyway", QtWidgets.QMessageBox.AcceptRole)
    cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    # Named, never indexed -- ``buttons()`` returns LAYOUT order, which puts these two
    # in a platform-dependent order (the same trap ``confirm_overwrite`` documents).
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec()
    return box.clickedButton() is go


class PlotsWindow(
    _PlotsFitMixin,
    _PlotsAnnotationMixin,
    _PlotsSourcesMixin,
    _PlotsStyleMixin,
    _PlotsNotesMixin,
    _PlotsViewsMixin,
    QtWidgets.QMainWindow,
):
    """A standalone window for population-level exploration of event tables.

    Fed per-event tables via :meth:`set_data`, it offers histogram / scatter /
    PCA views plus shared GMM/k-means clustering, over ONE OR MORE checked
    sources. Nothing here mutates the source data; it reads snapshots the owner
    pushes in.

    Emits:
        refreshRequested(): the user clicked *Refresh*; the owner should
            re-snapshot the current tables and call :meth:`set_data` again.
        eventRowsSelected(source_key, row_ids): Scatter/PCA points were clicked;
            detector sources emit their stable ``event_id`` values; legacy/manual
            frames without that column emit their dataframe index values.
    """

    refreshRequested = Signal()
    #: A click on one or more points in a Scatter or PCA plot. ``source_key`` is the
    #: stable key of the :class:`PlotSource` supplied to :meth:`set_data`; ``row_ids``
    #: contains ``event_id`` for a detector source (dataframe index values only for a
    #: legacy/manual frame without that column). Filtering and PCA keep those identities
    #: rather than manufacturing positional row numbers. Dataframe labels alone are not
    #: sufficient: EventSet explicitly permits duplicate and NaN indices.
    #: Ctrl/Cmd/Shift-click extends (or toggles) the selection for one source.
    eventRowsSelected = Signal(str, object)
    #: A project-owned recipe or computed snapshot changed.  MainWindow is the sole
    #: project writer; this signal only tells it that the next save/recovery capture
    #: must include a fresh analytical-workspace artifact.
    workspaceChanged = Signal()
    #: The window is about to discard its live rendering cache on close.  Unlike
    #: :attr:`workspaceChanged`, this does *not* mean the durable workspace changed:
    #: MainWindow uses it to take one synchronous, clean checkpoint before
    #: :meth:`reset_state` blanks the reusable window.
    workspaceCheckpointRequested = Signal()
    #: What Update plot would commit changed: the count or the reasons behind it (the
    #: stale selection, a proposed fit, the first plot). Emitted at the end of
    #: :meth:`_sync_run_state`, which every site that moves the gate already calls. Read by
    #: the Analysis Explorer's toolbar through :meth:`inspector_adapter`.
    pendingChanged = Signal()
    #: Undo or Redo became available or unavailable, or the stack moved (a step taken, an
    #: undo, a redo, a reset). Emitted at the end of :meth:`_sync_config_buttons`, so the
    #: toolbar's buttons and the Format tab re-read after ``undo_config`` / ``redo_config``
    #: too, which emitted nothing before.
    historyChanged = Signal()
    # Explicit managed Update/Apply publication receipts; never emitted by close/eviction.
    figureResultsPublished = Signal(object)

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle(f"NanoScope — {ANALYSIS_LABELS['distributions']}")
        self.resize(1120, 720)

        self._sources: dict[str, PlotSource] = {}    # key -> source, in offer order
        #: ``(doc_id, channel, module_id)`` of the detection the Analysis panel is
        #: SHOWING, set by :meth:`set_active_detection`. See :meth:`_active_source_key`.
        self._active_detection: tuple | None = None
        #: The Results panel's judge of a stored table, set by
        #: :meth:`set_staleness_provider`. See :meth:`_table_staleness`.
        self._staleness_provider = None
        self._cluster_result: populations.ClusterResult | None = None
        self._pca_result: populations.PCAResult | None = None
        #: Non-None only while :meth:`_render_workspace_snapshot` is painting values
        #: that were already computed and restored from a project.  The ordinary draw
        #: path is deliberately reused for bars/points/labels, while the numerical
        #: seams below read these frozen results instead of invoking an optimiser.
        self._workspace_snapshot_fits: tuple[dict[str, object], dict[str, str]] | None = None
        self._workspace_snapshot_rendering = False
        self._manual_x: tuple[float, float] | None = None   # user axis pins, data units (None = auto)
        self._manual_y: tuple[float, float] | None = None
        self._updating = False   # suppress redraws during bulk widget updates
        #: (sample digest, censoring, k) -> a computed note. See :meth:`_cached_note`.
        self._note_cache: dict[tuple, str] = {}
        #: (sample digest, log x) -> a histogram KDE overlay's curve. See :meth:`_kde_for`.
        self._kde_cache: dict[tuple, object] = {}
        #: Set by a snapshot render that declined to estimate the KDE overlay.
        self._kde_pending_after_snapshot = False
        #: ...and by one that declined to fit an applied scatter regression.
        self._scatter_fit_pending_after_snapshot = False
        #: Whether the histogram's KDE overlay is ON THE PICTURE -- the box is only the
        #: proposal. `_applied_fit`'s split, for its reason: a live redraw (a bin count, a
        #: log toggle) must not estimate a curve the user ticked but has not committed.
        #: Committed by :meth:`_armed_redraw`, which every run gesture ends in.
        self._applied_kde_overlay = False
        #: The Event-count heatmap's image and its colour bar, while one is drawn.
        self._heatmap_image = None
        self._heatmap_bar = None
        #: key -> zero-arg callable, for notes DEFERRED off the redraw path.
        #: See :meth:`_deferred_note`.
        self._pending_notes: dict[tuple, object] = {}
        #: Keys currently being computed by :class:`_NoteWorker`. A key in here is
        #: NOT re-deferred and does NOT re-arm the timer: without that, every redraw
        #: while a job is in flight would abort and restart it, and on a long
        #: bootstrap the panel would never settle at all.
        self._inflight_notes: set[tuple] = set()
        self._note_worker: _NoteWorker | None = None
        self._note_thread: QtCore.QThread | None = None
        self._note_timer = QtCore.QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.timeout.connect(self._settle_notes)
        #: Coalesces the Results panel's re-judgements (a Baseline-form knob held down) into
        #: one look at the STALE mark. See :meth:`schedule_stale_redraw`.
        self._stale_redraw_timer = QtCore.QTimer(self)
        self._stale_redraw_timer.setSingleShot(True)
        self._stale_redraw_timer.setInterval(300)
        self._stale_redraw_timer.timeout.connect(self._redraw_stale_marks)
        #: What :meth:`_stale_signature` read at the last draw: the marks on screen.
        self._stale_drawn: tuple = ()
        self._style = StyleSettings()   # the tunable plot design (title/colour/font/size/export)
        # Captions belong to the rendered result, which may differ from pending
        # controls. Presentation edits reuse them without reading/re-fitting data.
        self._title_presentation: tuple[str | None, str | None, str | None] = (None, None, None)
        self._axis_auto_labels: dict[str, str] = {}

        # -- A2: UNDO/REDO over the configuration -------------------------------------
        #: The stack itself. Re-based onto the real opening config at the foot of this
        #: __init__, once `_build_ui` has given `config()` something to read.
        self._history = _ConfigHistory(PlotConfig())
        #: True while something OTHER THAN A USER GESTURE is moving the config, so the
        #: hook records nothing: `_install_history` putting a stack entry back, and
        #: `set_data` letting new data choose columns and tick boxes. The name and the
        #: idea are `EventHistory.set_suspended`'s -- "a reconstruction never becomes an
        #: offer to undo the project you just opened". Without it, the `_armed_redraw` at
        #: the end of `_set_config` would note an undo's own result as a fresh change and
        #: Undo would become a two-entry toggle.
        self._history_suspended = False
        self._format_text_run_key: str | None = None
        #: Holds a run of KEYSTROKES open. See `_history_touch_text` for the grain rule.
        self._history_timer = QtCore.QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.timeout.connect(self._history_commit)
        #: Rows taken out of a checkable list, kept ALIVE until the event that removed
        #: them has unwound. See :meth:`_retire_row` -- this is the crash fix, not a
        #: cache: Qt marshals a signal's `QListWidgetItem*` argument once per receiver,
        #: and a row destroyed by the first receiver is a dangling pointer to the second.
        self._retired_rows: list = []
        self._retire_timer = QtCore.QTimer(self)
        self._retire_timer.setSingleShot(True)
        self._retire_timer.timeout.connect(self._drop_retired_rows)
        #: True while the template combo is being rebuilt or set by code, so its own
        #: `currentTextChanged` cannot be read as the user picking a template (and so
        #: `_sync_template_selection` cannot knock the name it has just installed back
        #: to "- manual -"). The same guard `DetectionPanel._protocol_updating` is.
        self._template_updating = False

        # -- THE RUN GATE (round-4 Q1: "the data should not be plotted by default --
        # the user needs to make the selection of the data and type of plot, and then
        # run it") --------------------------------------------------------------------
        #
        # Two flags, and the distinction between them is the whole design:
        #
        # ``_plot_armed`` -- has a plot been COMMITTED at all? While it is False the
        #   canvas carries the "choose your data and plot type, then press Plot"
        #   prompt and no cosmetic control draws anything, because there is nothing to
        #   restyle yet.
        # ``_stale`` -- the committed plot no longer matches the controls. Set by a
        #   change of DATA SELECTION, PLOT TYPE or AXIS COLUMN and by nothing else:
        #   fonts, colours, grid and legend are cosmetic and stay LIVE, because making
        #   a user press Plot to see a colour is the opposite complaint.
        #
        # WHAT ARMS IT. Every PROGRAMMATIC entry point -- :meth:`set_data`,
        # :meth:`set_fit`, :meth:`set_plot_type`, the whole ``set_*`` surface and
        # MainWindow's refresh path -- arms and draws immediately, exactly as it always
        # has. Naming a plot in code IS the run gesture, and ~96 existing test sites
        # plus ``MainWindow._refresh_plots`` depend on it. The gate is about the MOUSE:
        # ticking a source or walking the plot-type combo marks the plot stale and
        # waits, which is where the cost the owner is complaining about actually is.
        #
        # ``auto_plot`` is the one remaining half. With it True (the default) the first
        # ``set_data`` draws, so nothing about the programmatic contract moves; set it
        # False before pushing data and the window opens on the prompt instead. See the
        # note in :meth:`set_data`.
        self.auto_plot = True
        self._plot_armed = False
        self._stale = False
        self._stale_reason = ""
        #: EVERY reason the plot on screen is out of date, in the order they arose and
        #: without repeats ("the plot type", "an axis column", ...): ``_stale_reason`` holds
        #: only the last one, and Update plot's count is the number of independent reasons.
        #: Not part of ``PlotConfig``, the history or the saved workspace.
        self._stale_reasons: list[str] = []
        #: True once the Analysis Explorer's inspector holds this window's controls (set
        #: by the adapter's ``on_adopt``): the verb is then Update plot, not the Plot button.
        self._hosted = False
        #: The source keys that were checked at the last :meth:`_on_sources_changed`.
        #: ``QListWidget.itemChanged`` fires for a TICK and for a RENAME with no way to
        #: tell them apart from the item alone, and the two must be gated differently:
        #: a tick is a data change (stale), a rename is legend text (live). This is the
        #: comparison that separates them.
        self._checked_snapshot: tuple[str, ...] = ()
        #: ``entry id -> label`` for every legend entry the LAST draw wanted, included
        #: or not. What the Legend-entries checklist is rebuilt from (owner note Q2).
        self._legend_offered: dict[str, str] = {}

        # THE APPLIED FIT, as distinct from the SELECTED one (owner note #12: "the app
        # applies the fit as soon as the user selects it; better to have the user
        # confirm"). Picking a model from the dropdown now only ARMS the Apply button;
        # every draw reads these fields, so nothing is fitted -- and none of the
        # advisory machinery behind a fit is paid for -- until the user commits.
        self._applied_fit = _FIT_NONE
        self._applied_fit_k = 1
        self._applied_fit_k_auto = False
        self._applied_scatter_fit = _SFIT_NONE
        #: The DISPLAY unit factors the four axis-range boxes' text is currently
        #: written in. Derived from :attr:`_identity` and kept in step by
        #: :meth:`_set_config`; the identity is the authority.
        self._quantity_display = _NO_ENTRIES
        self._range_factors: tuple[float, float] = (1.0, 1.0)
        #: THE ANCHOR EVERY CLEARING RULE MEASURES FROM: the analysis identity the typed
        #: text and pins currently in the widgets were tuned FOR. Qt has no "about to
        #: change" signal, so a handler that fires after its widget moved has no other
        #: way to know what the window used to be showing. Moved only by
        #: :meth:`_set_config` and :meth:`_rebase`.
        self._identity = AnalysisIdentity()
        #: The sentence the last reset left on the status bar ("" for none): see ``_rebase``.
        self._reset_said = ""
        #: The words and ranges that sentence says went back to automatic, until one of them is
        #: typed or pinned again (:meth:`_lapse_reset_sentence`).
        self._reset_took: tuple[str, ...] = ()
        #: The Time unit the Fit-range boxes' text is written in -- stored WITH the
        #: bounds so :meth:`_fit_range` converts them by what they were typed under
        #: rather than by whatever the combo says now. See :class:`FitConfig`.
        self._fit_range_unit = _DEFAULT_TIME_UNIT
        #: True only for the duration of a :meth:`_set_config` write. The plot-type and
        #: Time-unit handlers rebase BEFORE they honour ``_updating`` (they must: the
        #: callers that move those combos in bulk still need the pins retuned), so
        #: without this the writer would re-enter the rule for every combo it writes,
        #: rebasing against a half-installed window. The writer sets the identity itself
        #: when it is finished, which is what makes the suppressed rebases redundant
        #: rather than merely skipped.
        self._installing = False
        #: False until :meth:`_build_ui` has finished. The config reader and writer touch
        #: every group, and the groups are built in sequence -- so a signal emitted while
        #: the Style group does not exist yet must not reach them.
        self._ui_ready = False

        # Live plot items (kept for tests + incremental clears). The singular
        # attributes are the PRIMARY (first-drawn) item and stay for the
        # single-source case; the plural lists carry the whole overlay.
        self._hist_item: pg.PlotDataItem | None = None
        self._hist_items: list[pg.PlotDataItem] = []
        self._fit_item: pg.PlotDataItem | None = None
        self._fit_items: list[pg.PlotDataItem] = []
        #: The confidence band's three items (upper, lower, fill). A list rather than one
        #: handle because `_clear_plot` is the only thing that removes them and a partial
        #: clear is how a band from the previous fit survives onto the next picture.
        self._band_items: list = []
        #: E2's caption. Parented to the ViewBox like the fit label, so `clear()`
        #: cannot reach it and `_clear_plot` detaches it by hand.
        self._prov_item = None
        self._limit_region: pg.LinearRegionItem | None = None   # below-detection-limit shading
        self._limit_line: pg.InfiniteLine | None = None
        self._last_fit = None    # the FitResult behind the PRIMARY overlay (None if none)
        self._fits: dict[str, object] = {}   # source key -> FitResult for this draw
        #: source key -> the COLUMN that fit was of. A FitResult does not name the
        #: column it came from, and :meth:`_persist_results` has to file it under
        #: ``fit.dwell`` or ``fit.depth`` -- the module ids the rest of the app reads
        #: (``main_window._stored_fit``) -- so the pairing is recorded as it is made.
        self._fit_columns: dict[str, str] = {}
        self._scatter_item: pg.ScatterPlotItem | None = None
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._scatter_fit_item: pg.PlotDataItem | None = None
        self._scatter_fit_items: list[pg.PlotDataItem] = []
        self._message_item: pg.TextItem | None = None
        #: Clickable point clouds in the current draw, grouped by stable source key.
        #: Presentation handles only: each pyqtgraph spot carries its exact row identity
        #: in ``data`` and that is what :attr:`eventRowsSelected` emits.
        self._event_point_items: dict[str, list[pg.ScatterPlotItem]] = {}
        self._selected_event_rows: dict[str, list[object]] = {}
        self._lasso_generation = 0
        #: The KDE or heatmap draw's own frame ``(x0, x1, y0, y1)`` in axis coordinates,
        #: or None for every other draw. Kept because Reset view must repeat it: the
        #: isocurves give ``autoRange()`` no bounds to fit (see `_draw_kde`), and on the
        #: heatmap it would pad the image's edges. View state, never saved.
        self._kde_view_frame: tuple[float, float, float, float] | None = None
        #: The residual strip's drawn ±span (`_render_cached_residual`), which Reset view puts
        #: back: the strip takes wheel and drag in y, and its x follows the plot by link.
        self._resid_view_span = 3.0

        #: THE Y QUANTITY THIS WINDOW WAS ASKED FOR AND COULD NOT OFFER. Written only by
        #: the one writer (:meth:`_set_config`) and cleared by any move of the combo the
        #: writer did not make; the whole rule, and the measurements behind it, are in
        #: :class:`_NormRefusal`. Created BEFORE ``_build_ui`` because the combo's
        #: ``currentTextChanged`` is connected in there, and the very first config write
        #: reaches this object. It reads the history's ORIGIN to decide whether the
        #: install it was recorded for is still the one on screen.
        self._norm_refusal = _NormRefusal(lambda: self._history.origin)

        #: EVERYTHING ELSE AN IMPORTED SIDECAR NAMED AND THIS WINDOW DID NOT TAKE.
        #: Written by the one writer out of `_incoming_config_file` below, read against
        #: the same origin as `_norm_refusal`; the rule and the measurements are in
        #: :class:`_ConfigLosses`. Created before ``_build_ui`` for `_norm_refusal`'s
        #: reason -- the first config write reaches this object.
        self._config_losses = _ConfigLosses(lambda: self._history.origin)

        #: THE FILE CURRENTLY BEING INSTALLED, or None. Set only by
        #: :meth:`import_config_json`, for the length of its one ``apply_config`` call,
        #: as ``(file name, raw payload, the config it decodes to)``.
        #:
        #: A STASH RATHER THAN AN ARGUMENT, deliberately. The losses have to be recorded
        #: BEFORE the draw that reports them -- a record written after ``apply_config``
        #: returns would miss the very draw the import causes, which is the one the user
        #: is looking at when they decide whether to save -- and :meth:`_set_config`'s
        #: signature is a contract several tests pin ("install exactly this value"). This
        #: keeps the one writer the one writer: it re-takes the record when this is set,
        #: and the ORIGIN it opens for every other outside install is what hides it.
        self._incoming_config_file: tuple[str, Mapping, PlotConfig] | None = None

        self._build_ui()
        self._ui_ready = True
        # THE HISTORY HOOK IS WIRED LAST, over the finished control tree, so no signal
        # emitted while a group was still being built can reach it. See `_wire_history`.
        self._wire_history(self._controls_host)
        self._history.reset(self.config())
        self._sync_config_buttons()
        # The window's opening identity, so the first real change measures from a truth
        # rather than from the empty default.
        self._identity = self.config().identity
        self._redraw()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        controls = self._build_controls()
        #: Kept because `_wire_history` walks it AFTER construction: the history hook is
        #: attached to whatever controls exist rather than to a hand-kept list of them.
        self._controls_host = controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        # THE CONTROLS COLUMN WAS CAPPED NARROWER THAN ITS OWN CONTENTS. Measured on
        # this window at its default 1120x720: the controls widget reports a
        # minimumSizeHint of 444 px, the QScrollArea was pinned to a maximum of 420, and
        # the viewport therefore offered 400 px (20 of the 420 went to the vertical
        # scrollbar) -- so a HORIZONTAL scrollbar with a 44 px range was on screen at
        # rest, permanently, on a column the user could not widen because the same
        # maximum froze the splitter handle. That is the whole of the owner's "the
        # Population Plots panel needs more room to expand".
        #
        # Both halves are removed rather than re-tuned:
        #   * no maximum at all -- the splitter is the width control, and it can now
        #     actually be dragged (measured: asking the splitter for 700 px used to
        #     return 420, and returns 700);
        #   * a minimum the CONTENT can actually honour, ASKED FOR rather than guessed.
        #     The 444 px came almost entirely from one row -- the histogram page's
        #     "Fit" line, which put the five-model combo ("inverse-gaussian (first
        #     passage)") and an "Apply fit" button side by side -- so that row was
        #     unstacked and the long combos told to size to a fixed character count
        #     instead of to their widest item, which brought the content minimum from
        #     444 px to 331. (Measured on this build: 309 with no group hidden, where
        #     the Style group is the widest, and 303 on the Data section the window
        #     opens on.) The number below is then derived from THAT, plus the vertical
        #     scrollbar and the frame -- 329 on this build -- so the two can never
        #     disagree again the way they did. (A literal would have rotted the first
        #     time a control was added to a group.)
        scroll.setMinimumWidth(max(_CONTROLS_MIN_PX, self._controls_min_width(scroll, controls)))
        # The column also has to STOP growing when the window does; without this the
        # splitter hands it half of every extra pixel and the plot never gets wider.
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self._section_area = scroll
        # NO CODE IN THIS WINDOW SCROLLS THIS COLUMN, so no handle is kept on it. There
        # was one, for exactly one caller: a refused Ctrl+Z at the undo floor used to call
        # `ensureWidgetVisible` on the depth-ceiling sentence at the foot of this column.
        # That is gone -- the whole account is at `undo_config` -- and the attribute went
        # with it rather than being left behind as a handle only the tests read. Anything
        # that needs this scroll area can find it the way the tests and
        # `docs/sweep6/repro_S6-04_undo_ceiling.py` do: it is the QScrollArea whose
        # `widget()` is an ancestor of `_undo_btn`, which is a property of the layout
        # rather than of any one fix.
        #
        # WHAT MOVES THIS COLUMN IS THE USER, and the gestures that do not are driven in
        # `test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`: crossing the
        # ceiling, a refused Ctrl+Z, a refused click on the dead Undo button, a
        # right-click, Space/Return/Enter aimed at the dead button (which will not take
        # focus: they reach whatever did), an undo that succeeds, a template recall, a
        # plot-type change and thirty ordinary edits all leave the scrollbar on the single
        # value it was parked on. Qt's focus-follow -- a scroll area following focus off a
        # widget that has just been disabled -- WAS the standing exception to that and is
        # not one any more: `_retire_history_button` decides where focus goes when Undo or
        # Redo is retired. The click that used to move it is driven in
        # `test_a_click_that_empties_the_undo_stack_does_not_move_the_column`.
        #
        # "A RESIZE" USED TO BE ON THAT LIST AND SHOULD NOT HAVE BEEN. What moves then is
        # not a scroll but QScrollBar's own clamp: a TALLER window shortens the range, a
        # value parked past the new maximum is clamped to it, and shrinking back does not
        # give it back. Measured at the undo floor, resized away and back -- 775x450
        # parked at the fold 1928 -> 1788, at the bottom 2238 -> 1788 by way of 1512x900,
        # 1120x720 at the bottom 1968 -> 1788 -- while any parking below the tallest
        # window's maximum (1788 here, with the ceiling line standing) does not move at
        # all, which is why the fold at 1120x720 reads as still and the fold at 775x450
        # does not. The pre-sweep tree does the same thing with its own numbers (775x450
        # fold 1928 -> 1737, 1120x720 fold 1658 -> 1658), so this is Qt's clamp and not
        # this sweep's doing; it is pinned, at a parking that can catch it, in
        # `test_a_resize_that_shortens_the_range_clamps_the_column`.
        #
        # The claim here is about this window's own calls, and
        # `test_no_code_in_this_window_moves_the_controls_column` is what makes it one.
        splitter.addWidget(scroll)

        right = QtWidgets.QSplitter(Qt.Vertical)
        # Decade-only labels on the log axes. pyqtgraph adds an extra MINOR tick level
        # whenever a log axis spans fewer than ~3 decades (AxisItem.logTickValues appends
        # `(None, minor)` when it has under three decade-spaced levels), and then labels
        # every one of them -- which on the owner's dwell scatter rendered the x axis as
        # "0.40.50.60.70.8.91" and "3.10^1 4.10^1 5.10...", an unreadable smear. The ticks
        # themselves are useful; only their labels collide, so this blanks the labels and
        # keeps the marks.
        self._plot_widget = pg.PlotWidget(
            axisItems={"bottom": _DecadeLabelAxis(orientation="bottom"),
                       "left": _DecadeLabelAxis(orientation="left")})
        self._plot_widget.setMinimumSize(420, 320)
        self._plot_widget.setAccessibleName("Population plot canvas")
        self._plot_widget.setAccessibleDescription(
            "Interactive population plot. Scatter and PCA points can be clicked to "
            "select their source events in the event table and trace; hold Control, "
            "Command, or Shift to select several points.")
        self._plot_widget.setToolTip(
            "Scatter and PCA: click a point to inspect its event. "
            "Ctrl/Cmd/Shift-click adds or removes points from the selection. "
            "Enable Lasso select in the toolbar to select an outlined region.")
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._lasso = PlotLasso(self._plot_widget, self)
        self._lasso.completed.connect(self._on_lasso_completed)
        self._lasso.enabledChanged.connect(self._on_lasso_mode_changed)
        self._lasso.cancelled.connect(lambda message: self.statusBar().showMessage(message, 7000))
        # The grid is NOT set here any more. It used to be hard-coded once, at
        # construction, and never read again -- so the Style group could change every
        # other thing about the canvas and not this (owner note Q7). It is applied on
        # every draw now, from :attr:`StyleSettings.show_grid`, by :meth:`_apply_grid`.
        self._apply_grid()
        right.addWidget(self._plot_widget)

        # THE RESIDUAL STRIP: a second axis UNDER the main one, sharing its x.
        #
        # WHY A SECOND WIDGET AND NOT A SECOND ROW OF ONE. `self._plot` is a PlotItem
        # inside a PlotWidget and `Save image…` exports THAT item; re-homing both axes
        # into a GraphicsLayoutWidget so one export could catch them would change the
        # object every draw path, the export and eleven tests already hold. The cost of
        # not doing it is stated where the user meets it (the checkbox tooltip): the
        # strip is on screen, not in the PNG -- so the number it produces, the reduced
        # χ², is routed into Details and into the fit label (joined to its equation), which the
        # PNG does keep. The picture is the exploratory half; the statistic is the reportable
        # half, and the reportable half exports.
        #
        # `setXLink` rather than a copied range: pan and zoom on the histogram must move
        # the strip with it, or a bar and its residual stop lining up the moment anyone
        # touches the plot -- which is the only way this picture is read.
        self._resid_widget = pg.PlotWidget(
            axisItems={"bottom": _DecadeLabelAxis(orientation="bottom"),
                       "left": _DecadeLabelAxis(orientation="left")})
        self._resid: pg.PlotItem = self._resid_widget.getPlotItem()
        self._resid.setXLink(self._plot)
        self._resid.showGrid(x=False, y=True, alpha=0.15)
        # A STRIP, not a second plot: it reads as an annotation of the axis above it, and
        # a residual panel that takes half the window buries the data it is about.
        self._resid_widget.setMinimumHeight(90)
        self._resid_widget.setMaximumHeight(160)
        self._resid_widget.setVisible(False)
        right.addWidget(self._resid_widget)

        # THE SHORT ANSWER BEFORE THE AUDIT TRAIL. The full Details prose is valuable
        # (and persisted with analytical results), but making it the only result surface
        # forced a newcomer to scan a monospaced report to find the one outcome and the
        # one warning that govern the plot. Keep that report intact in ``_details`` and
        # derive a compact, always-visible reading above it.
        summary_box = QtWidgets.QFrame()
        summary_box.setObjectName("populationPlotResultSummary")
        summary_box.setFrameShape(QtWidgets.QFrame.StyledPanel)
        summary_box.setAccessibleName("Plot result summary")
        summary_lay = QtWidgets.QVBoxLayout(summary_box)
        summary_lay.setContentsMargins(8, 6, 8, 6)
        summary_lay.setSpacing(2)
        summary_heading = QtWidgets.QLabel("RESULT")
        summary_heading.setStyleSheet("font-size: 10px; font-weight: 600;")
        self._result_summary = QtWidgets.QLabel("No result yet.")
        self._result_summary.setObjectName("populationPlotResult")
        self._result_summary.setWordWrap(True)
        self._result_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._result_summary.setAccessibleName("Result")
        self._result_warning = QtWidgets.QLabel("")
        self._result_warning.setObjectName("populationPlotWarning")
        self._result_warning.setWordWrap(True)
        self._result_warning.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._result_warning.setStyleSheet(f"color:{_WARN_HEX};")
        self._result_warning.setAccessibleName("Result warning")
        self._result_warning.setVisible(False)
        summary_lay.addWidget(summary_heading)
        summary_lay.addWidget(self._result_summary)
        summary_lay.addWidget(self._result_warning)
        right.addWidget(summary_box)

        details_box = QtWidgets.QGroupBox("Details — methods and provenance")
        details_box.setCheckable(True)
        details_box.setChecked(False)
        details_box.setAccessibleName("Detailed methods and provenance")
        dlay = QtWidgets.QVBoxLayout(details_box)
        dlay.setContentsMargins(6, 6, 6, 6)
        self._details = QtWidgets.QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setMaximumHeight(180)
        self._details.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        self._details.setAccessibleName("Full plot details")
        dlay.addWidget(self._details)
        self._details_box = details_box
        self._details.setVisible(False)
        details_box.setMaximumHeight(30)
        details_box.toggled.connect(self._set_details_expanded)
        right.addWidget(details_box)
        # Indices name the widgets in order: plot, residual strip, compact result,
        # Details. The plot is the only one that grows.
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setStretchFactor(2, 0)
        right.setStretchFactor(3, 0)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # THE 26 PX DEAD GUTTER. The splitter allotted the column 446 px while the
        # scroll area painted 420 of them (its own maximum), so 26 px of the window sat
        # empty between the controls and the plot at rest. With the maximum gone the
        # widget fills whatever it is given, and the opening split is stated here
        # explicitly instead of being whatever the size hints happened to add up to.
        want = scroll.minimumWidth() + _CONTROLS_SLACK_PX
        splitter.setSizes([want, max(420, self.width() - want)])
        splitter.setChildrenCollapsible(False)   # a collapsed-to-zero column cannot be found again
        self._splitter = splitter

        self.setCentralWidget(splitter)
        self._build_workflow_toolbar()
        self._install_plot_interaction()

    def _install_plot_interaction(self) -> None:
        """View only (plot_interaction): Reset view, zoom history, app menu, pointer readout.

        NONE OF IT IS CONFIG. Reset view, Zoom back and Zoom forward move the view and
        nothing else: they never reach `_history_touch`, never edit the Axis range boxes and
        never mark the plot stale, so this window's Cmd+Z (the whole PlotConfig) does not
        see them. The residual strip is x-linked to the plot and shares its history. The
        readout sits on the status line beside the Mode label, reads the drawn axes (their
        log mode and the Time unit's tick scale), and on Scatter/PCA names the nearest
        point's source and event_id -- the identity a click on it would select. Over the
        Event-count heatmap it adds the count of the cell under the pointer.
        """
        # Defined ONCE: the right-click menu and the Explorer's Export ▾ (``inspector_adapter``)
        # both list these, so they cannot disagree.
        exports = self._exports = (
            ("Export current figure…", lambda: self.save_image()),
            ("Export plot data…", lambda: self.export_data()),
            ("Export configuration JSON…", lambda: self.export_config_json()))
        self._readout = window_readout(self)

        def heatmap_count(vx: float, vy: float) -> str:
            # The drawn counts themselves (the image is placed in log10 view coordinates
            # on a log axis, so the view position maps onto the original cell grid).
            value = image_value(getattr(self, "_heatmap_image", None), vx, vy)
            return "" if value is None else f"Event count {value:.0f}"

        self._plot_nav = install_plot_interaction(
            self._plot, readout=self._readout, reset=self._reset_plot_view, exports=exports,
            points=self._readout_points, extra=heatmap_count, text=_PlotsTextHost(self))
        # The fit label: one item for the window's lifetime (a ViewBox child, which
        # plot.clear() does not remove), empty -- and so hidden -- until a fit is drawn.
        self._fit_label_item = FitLabelItem(self._plot.getViewBox())
        self._fit_label_item.apply_state(self._style.fit_label_state(), [])
        self._fit_label_item.moved.connect(self._on_fit_label_moved)
        self._appearance_widget.set_fit_label_available(True)
        figure_text_for(self._plot).refresh()
        install_plot_interaction(self._resid, readout=self._readout, exports=exports,
                                 navigation=self._plot_nav)
        button = view_button(lambda: self._plot_nav, (self._plot_nav,))
        self._view_button = button
        self._workflow_toolbar.insertWidget(self._workflow_spacer_action, button)

    # -- in-figure text and axes (figure_text) ---------------------------------------------------
    def _figure_text_targets(self) -> dict:
        """What a double-click or right-click on the figure may edit, keyed as
        ``figure_text`` names elements. The title and axis labels show their automatic
        text when nothing is typed; the fit label opens on its TEMPLATE, so retyped words
        keep live numbers.

        The title is a target whatever it shows -- empty (no automatic title) or hidden --
        so the plot's menu can offer to add one (it opens this target's editor)."""
        style = self._style
        plot = self._plot
        out = {"title": TextTarget(plot.titleLabel, plain_text(self._title_presentation[0] or ""),
                                   style.text_style("title"), "title",
                                   frozenset({"edit", "style", "hide", "align"}))}
        for side in ("bottom", "left"):
            axis = plot.getAxis(side)
            out[f"axis.{side}.label"] = TextTarget(
                axis, plain_text(self._axis_auto_labels.get(side, "")),
                style.text_style(f"axis.{side}.label"), "label")
            out[f"axis.{side}.ticks"] = TextTarget(
                axis, "", style.text_style(f"axis.{side}.ticks"), "tick labels",
                frozenset({"style"}))
        legend = getattr(plot, "legend", None)
        rows = {}
        if legend is not None:
            for sample, label in list(legend.items):
                rows[id(getattr(sample, "item", None))] = label
        for _entry_id, (item, _label, kind, key) in getattr(self, "_presentation_legend", {}).items():
            src = self._sources.get(key)
            row = rows.get(id(item))
            if row is None:
                continue
            element = _legend_style_key(kind, key)
            row_style = style.text_style(element)
            if kind in ("series", "fit") and src is not None:
                # A source's name, never a typed caption: Rename writes the ONE name
                # resolver (``label_overrides``), so the legend, the fit's own row ("Apo
                # fit (…)"), the fit label's line prefix, the Sources list and Details all
                # follow. A one-source fit row ("fit (…)") names no source, so it has
                # nothing to rename: a double-click there opens Text style….
                named = kind == "series" or getattr(item, "_presentation_fit_named", True)
                override = style.label_overrides.get(key)
                out[element] = TextTarget(
                    row, src.label, _dc_replace(row_style, text=override or None),
                    "legend entry",
                    frozenset({"edit", "style", "hide"} if named else {"style", "hide"}))
            elif kind in _LEGEND_ROW_KINDS:
                # A cluster's or the KDE overlay's row: the window's words, styled or hidden.
                out[element] = TextTarget(row, plain_text(row.text),
                                          _dc_replace(row_style, text=None), "legend entry",
                                          frozenset({"style", "hide"}))
        line = getattr(self, "_limit_line", None)
        label = getattr(line, "label", None) if line is not None else None
        if isinstance(label, _LimitLabel):
            out[LIMIT_LABEL_KEY] = TextTarget(
                label, _LIMIT_LABEL_AUTO, style.text_style(LIMIT_LABEL_KEY),
                "detection-limit label", frozenset({"edit", "style", "hide"}))
        item = getattr(self, "_fit_label_item", None)
        if item is not None:
            out[FIT_LABEL_KEY] = TextTarget(
                item, automatic_template(style.fit_label_state()),
                style.text_style(FIT_LABEL_KEY), "fit label", frozenset({"edit", "style", "hide"}))
        return out

    def _figure_background_entries(self) -> list:
        """A right-click on the plot itself: "Show detection-limit label" while the words
        are hidden, because a hidden label leaves nothing to right-click (the dashed line
        and the shading stay drawn)."""
        if LIMIT_LABEL_KEY not in self._figure_text_targets():
            return []
        style = self._style.text_style(LIMIT_LABEL_KEY)
        if style.text != HIDDEN_TEXT:
            return []
        return [("Show detection-limit label",
                 lambda: self._apply_figure_text(LIMIT_LABEL_KEY, _dc_replace(style, text=None)))]

    def _apply_figure_text(self, key: str, style: TextStyle, *, run: bool = False) -> None:
        """One in-figure text edit: written into the style, the picture restyled (never
        recomputed), and ONE step of this window's Undo.

        ``run=True`` is one write of a RUN of edits (a size spin's arrow clicks in the
        Format tab or the mini toolbar): the step is held open by the text grain
        (:meth:`_history_touch_text`, closed by focus-out or a 700 ms pause) instead of
        being closed at once, so the run is one step. A single edit is still one step.

        A legend row's text is never stored as typed: Hide entry writes ``legend_include``
        for that row, and a rename (from a source's row or its fit's) writes the source's
        one name, ``label_overrides``. Its font, size, weight and colour are the row's own
        (``text_styles["legend.entry.…"]``)."""
        if self._history_live():
            # A different element or a discrete edit ends the preceding spin/typing run.
            if not run or self._format_text_run_key != key:
                self._history_commit()
            if run:
                self._format_text_run_key = key
        if key.startswith("legend.entry."):
            entry_id = _legend_entry_id(key, self._sources)
            kind, _, src_key = entry_id.partition(":")
            if style.text == HIDDEN_TEXT:
                self._style = self._style.with_legend_include(entry_id, False)
            elif kind in ("series", "fit") and src_key in self._sources:
                if style.text is None:
                    self._style = self._style.without_label_override(src_key)
                else:
                    self._style = self._style.with_label_override(src_key, style.text)
            self._style = self._style.with_text_style(key, _dc_replace(style, text=None))
        else:
            self._style = self._style.with_text_style(key, style)
        updating = self._updating
        self._updating = True
        try:
            self._apply_style_widgets(self._style)
            self._sync_source_row_labels()
        finally:
            self._updating = updating
        if key.startswith("legend.entry."):
            self._restyle_committed_legend()
            self._sync_legend_entry_list()
        if run:
            # The restyle ends in `_history_settle`, which closes a step for anything that
            # is not typed text; held, so the run stays ONE step the text grain closes.
            suspended = self._history_suspended
            self._history_suspended = True
            try:
                self._presentation_changed()
            finally:
                self._history_suspended = suspended
            self._history_touch_text()
        else:
            self._presentation_changed()
            self._history_touch()

    def _on_fit_label_moved(self, offset) -> None:
        """A drag of the fit label: its place is saved (``fit_label.offset``), as one
        step. The view never moved -- the label took the drag."""
        if self._updating:
            return
        state = self._style.fit_label_state()
        self._style = self._style.with_fit_label(_dc_replace(
            state, placement="custom", offset=(float(offset[0]), float(offset[1]))))
        self._presentation_changed()
        self._history_touch()

    def _log_pair(self):
        """The current page's (Log x, Log y) boxes, or None where the view has none."""
        for pair in ((self._hist_logx, self._hist_logy), (self._scatter_logx, self._scatter_logy),
                     (self._kde_logx, self._kde_logy), (self._heatmap_logx, self._heatmap_logy)):
            # Visible TO THE STACK that holds the plot-type pages, not to the window: the
            # boxes are not individually hidden, and in the Explorer they sit in an
            # inspector group that may be collapsed (or hidden by a Section, standalone),
            # which says nothing about which plot type is showing.
            if pair[0].isVisibleTo(self._pages):
                return pair
        return None

    def _figure_axis_state(self, side: str) -> AxisState:
        """Axis properties for ``side``: range in the unit the axis DISPLAYS (the Axis
        range boxes' unit), the pin if there is one, and this page's Log box."""
        index = 0 if side == "bottom" else 1
        log = self._current_log_state()[index]
        factor = self._axis_unit_factors()[index]
        lo, hi = self._plot.getViewBox().viewRange()[index]

        def shown(v: float) -> float:
            try:
                return (10.0 ** v if log else v) * factor
            except OverflowError:
                return math.inf

        manual = self._manual_x if index == 0 else self._manual_y
        pair = self._log_pair()
        return AxisState(
            label=self._style.text_style(f"axis.{side}.label").text,
            label_auto=plain_text(self._axis_auto_labels.get(side, "")),
            range=None if manual is None else (manual[0] * factor, manual[1] * factor),
            shown_range=(shown(lo), shown(hi)),
            log=bool(pair[index].isChecked()) if pair is not None else log,
            log_available=pair is not None,
            log_reason="This view has no log scale for this axis.",
            tick_pt=self._style.text_style(f"axis.{side}.ticks").size_pt,
            tick_major_available=False)       # Distributions keeps no major tick spacing

    def _apply_figure_axis(self, side: str, values: AxisState) -> None:
        """Axis properties, committed: the label and tick size go to the style, the range
        to the Axis range boxes and their pin (exactly what Apply / Auto do), as one step.
        Log goes through the page's own Log box, so it is gated, redrawn and undone
        exactly as a tick in the panel is."""
        if self._history_live():
            self._history_commit()
        before = self._figure_axis_state(side)
        # The popover shows a range to 6 significant figures: OK on the same text is no
        # change, so it neither rounds the pin nor records a step.
        if (values.range is not None and before.range is not None
                and all(f"{a:.6g}" == f"{b:.6g}" for a, b in zip(values.range, before.range))):
            values = _dc_replace(values, range=before.range)
        x = side == "bottom"
        lo_key, hi_key = ("xmin", "xmax") if x else ("ymin", "ymax")
        self._style = self._style.with_text_style(f"axis.{side}.label", _dc_replace(
            self._style.text_style(f"axis.{side}.label"), text=values.label))
        self._style = self._style.with_text_style(f"axis.{side}.ticks", _dc_replace(
            self._style.text_style(f"axis.{side}.ticks"), size_pt=values.tick_pt))
        self._updating = True
        try:
            self._apply_style_widgets(self._style)
            if values.range is not None and values.range != before.range:
                self._range_edits[lo_key].setText(f"{values.range[0]:.12g}")
                self._range_edits[hi_key].setText(f"{values.range[1]:.12g}")
        finally:
            self._updating = False
        # The range's Apply and the Log box each commit on their own; held, so a range
        # and Log typed in one OK are one step and one Undo puts both back.
        suspended = self._history_suspended
        self._history_suspended = True
        try:
            if values.range != before.range:
                if values.range is None:
                    self._reset_axis_range(axis="x" if x else "y")
                else:
                    self._apply_axis_range()
            self._presentation_changed()
            pair = self._log_pair()
            if pair is not None and values.log != before.log:
                pair[0 if x else 1].setChecked(bool(values.log))
        finally:
            self._history_suspended = suspended
        self._history_touch()

    def _reset_plot_view(self) -> None:
        """Reset view: the framing the last draw ended with, the user's axis pin included.

        The same two steps as `_finalize_view` (or the KDE's grid frame, or the heatmap's
        image frame, in place of the auto-fit), read off the DRAWN axes rather than the controls: a Log x ticked but not
        yet applied must not change what "reset" means for the plot on screen. The residual
        strip follows in x by its link and gets its drawn ±span back in y.
        """
        log_x = bool(self._plot.getAxis("bottom").logMode)
        log_y = bool(self._plot.getAxis("left").logMode)
        frame = self._kde_view_frame
        if frame is not None:
            self._plot.vb.setRange(xRange=frame[:2], yRange=frame[2:], padding=0)
        else:
            self._plot.autoRange()
        self._apply_stored_ranges(log_x, log_y)
        if not self._resid_widget.isHidden() and self._resid.items:
            span = self._resid_view_span
            self._resid.setYRange(-span, span, padding=0)

    def _readout_points(self):
        """Scatter/PCA point clouds for the readout's nearest point, with how to name one."""
        out = []
        for key, items in self._event_point_items.items():
            source = self._sources.get(key)
            label = source.label if source is not None else key
            ident = ("event_id" if source is not None and source.kind == "detection"
                     else "row")
            for item in items:
                out.append((item, lambda value, label=label, ident=ident:
                            f"{label} {ident} {value}"))
        return out

    def _build_workflow_toolbar(self) -> None:
        """Persistent section navigation and the two verbs used on every plot.

        The controls remain ordinary widgets (and retain their established public
        attributes), while this compact strip makes the long column navigable and keeps
        Plot/Apply and Export reachable when Appearance or Configuration is in view.
        """
        bar = QtWidgets.QToolBar("Population plot workflow", self)
        bar.setObjectName("populationPlotWorkflow")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        bar.setAccessibleName("Population plot workflow")

        section_label = QtWidgets.QLabel("Section:")
        section_label.setAccessibleName("Control section")
        bar.addWidget(section_label)
        self._section_combo = QtWidgets.QComboBox()
        self._section_combo.addItems(["Data", "Analysis", "Appearance", "Export"])
        self._section_combo.setToolTip(
            "Jump to a section of the controls without searching the full column.")
        self._section_combo.setAccessibleName("Population plot control section")
        self._section_combo.setAccessibleDescription(
            "Choose Data, Analysis, Appearance, or Export to move that controls section "
            "into view.")
        self._section_combo.activated.connect(self._show_control_section)
        bar.addWidget(self._section_combo)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        # Kept so the view-only Reset view button can sit left of it, beside the sections.
        self._workflow_spacer_action = bar.addWidget(spacer)

        self._run_action = bar.addAction("Plot / Apply")
        self._run_action.setObjectName("populationPlotRunAction")
        self._run_action.setToolTip(
            "Draw the selected plot and apply any pending fit choice (Ctrl+Return).")
        self._run_action.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        self._run_action.triggered.connect(self._run_plot)
        run_button = bar.widgetForAction(self._run_action)
        if run_button is not None:
            run_button.setAccessibleName("Plot or apply pending analysis")

        self._lasso_action = bar.addAction("Lasso select")
        self._lasso_action.setObjectName("populationPlotLassoSelect")
        self._lasso_action.setCheckable(True)
        self._lasso_action.setEnabled(False)
        self._lasso_action.setToolTip(
            "Switch between pan/zoom and region selection on Scatter/PCA. "
            "Drag a closed outline; Ctrl/Cmd/Shift toggles its events. "
            "Esc or right-click cancels and returns to pan/zoom. "
            "If several sources are enclosed, choose one source to inspect. "
            "Selection never accepts/rejects events or reruns analysis.")
        self._lasso_action.toggled.connect(self._lasso.set_enabled)
        self._selection_mode_label = QtWidgets.QLabel("Mode: Pan/zoom")
        self._selection_mode_label.setObjectName("populationPlotInteractionMode")
        self.statusBar().addPermanentWidget(self._selection_mode_label)

        export_button = QtWidgets.QToolButton()
        export_button.setObjectName("populationPlotExportMenu")
        export_button.setText("Export ▾")
        export_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_button.setToolTip(
            "Export the current image, its numeric data, or a reproducible configuration.")
        export_button.setAccessibleName("Export population plot")
        export_menu = QtWidgets.QMenu(export_button)
        export_menu.addAction("Export current figure…", lambda: self.save_image())
        export_menu.addAction("Export plot data…", lambda: self.export_data())
        export_menu.addAction("Export configuration JSON…", lambda: self.export_config_json())
        export_button.setMenu(export_menu)
        self._export_menu = export_menu
        self._export_menu_btn = export_button
        # Reset figure: the Appearance section's own button, in reach from every Section.
        reset_button = self._appearance_widget.take_reset_button()
        reset_button.setFlat(True)          # as flat as the toolbar's other text
        # The shared tip says "its look"; these rows of this window are not in the shared look.
        reset_button.setToolTip(
            reset_button.toolTip() + " Here it also keeps the Axis range boxes, Font size (body), "
            "the legend's position, entries and (n = …) switch, source names, Confidence band, "
            "Residual strip, Provenance caption and Heatmap colormap.")
        bar.addWidget(reset_button)
        bar.addWidget(export_button)

        self._workflow_toolbar = bar
        self.addToolBar(Qt.TopToolBarArea, bar)
        # BEGINNER VIEW: source choice plus the selected plot's essential controls.
        # The advanced analysis, appearance, and export/reuse groups still exist (and
        # remain fully addressable by the programmatic API), but only one purposeful
        # section is shown at a time instead of one 2,600 px undifferentiated column.
        self._show_control_section("Data")

    def _show_control_section(self, section: int | str) -> None:
        """Show one purposeful controls section and suppress unrelated advanced rows.

        Hosted in the Analysis Explorer the controls live in its inspector and a Section
        name is only a request to look at a part of it: the host's reveal (set at adoption
        as ``_inspector_reveal``) selects the tab and group, and no widget is shown or
        hidden here."""
        if isinstance(section, str):
            name = section
        else:
            name = self._section_combo.itemText(int(section))
        reveal = getattr(self, "_inspector_reveal", None)
        if reveal is not None and self._hosted:
            if name not in DEFAULT_REVEAL:
                return      # an unknown name asked for nothing before, and asks for nothing now
            if self._section_combo.currentText() != name:
                self._section_combo.setCurrentText(name)
            try:
                reveal(name)
                return
            except RuntimeError:
                pass        # the inspector is gone (a closed Explorer): the old behaviour
        visible = {
            # The first-run path: pick rows, choose the figure, press the persistent
            # Plot / Apply verb. No styling or clustering wall precedes that action.
            "Data": {self._source_group, self._plot_group},
            # The plot recipe remains visible beside its dependent feature/cluster
            # controls; source management is one quick Section pick away.
            "Analysis": {self._plot_group, self._axis_range_group,
                         self._features_group, self._cluster_group},
            "Appearance": {self._style_group},
            "Export": {self._config_group, self._save_btn, self._save_data_btn},
        }.get(name)
        if visible is None:
            return
        all_sections = (self._source_group, self._plot_group, self._axis_range_group,
                        self._features_group, self._cluster_group, self._style_group,
                        self._config_group, self._save_btn, self._save_data_btn)
        for widget in all_sections:
            widget.setVisible(widget in visible)
        if self._section_combo.currentText() != name:
            self._section_combo.setCurrentText(name)
        # Each section begins at the top. Direct scrollbar arithmetic is deliberate:
        # it happens only after an explicit navigator gesture and avoids focus-follow.
        self._section_area.verticalScrollBar().setValue(0)

    # -- the Analysis Explorer's inspector (Phase 1) -----------------------------------------------
    def inspector_adapter(self) -> PageAdapter:
        """How the Analysis Explorer hosts this window: its controls MOVE into the
        inspector's groups, once, for as long as the window is hosted.

        Built once and cached; nothing but ``AnalysisExplorer.add_page`` (through
        ``InspectorHost.adopt``) and the hosting test helpers call it. It computes nothing:
        ``update`` is :meth:`_run_plot`, ``pending`` reads the gate's own flags, history
        is :meth:`undo_config` / :meth:`redo_config` (the whole ``PlotConfig``, analysis
        and look), and every widget keeps its attribute name, its signals and its history
        hook (``_wire_history`` connected them on the widgets, so a moved widget keeps
        them). It builds no widget: the groups hold the page's own group boxes and
        controls, and the Format tab's extras are the page's own widgets, named, not
        moved until the Format tab shows them.

        Fit stays INSIDE Plot in Phase 1: the fit rows live in each plot type's own page
        layout. "Fit display" and "Recipes" are the only regroupings."""
        if getattr(self, "_inspector_adapter", None) is not None:
            return self._inspector_adapter
        appearance = self._appearance_widget
        # The labels of the Style group's rows travel with the widgets the Format tab moves
        # (PlotAppearanceWidget's convention): taken from the form, never from its layout items.
        form = self._style_group.layout()
        for control in (self._font_size, self._legend_entries, self._heatmap_cmap):
            label = form.labelForField(control) if hasattr(form, "labelForField") else None
            if label is not None and label.text():
                control._nanoscope_row_label = label
        groups = [
            InspectorGroup("sources", "Sources", [self._source_group],
                           tooltip="Which recordings are plotted, and their legend names."),
            InspectorGroup("plot", "Plot", [self._plot_group],
                           tooltip="Plot type, columns, bins, normalization, log axes and "
                                   "the fit rows."),
            InspectorGroup("fit_display", "Fit display",
                           [self._ci_band_check, self._residual_check], expanded=False),
            InspectorGroup("groups", "Groups", [self._features_group, self._cluster_group],
                           expanded=False),
            # The template rows only: Undo/Redo are the toolbar's and "Export JSON…" is in
            # Export ▾; "Import JSON…" has no place in an export menu, so it stays here.
            InspectorGroup("recipes", "Recipes",
                           [self._template_combo, self._template_save_btn,
                            self._template_rename_btn, self._template_delete_btn,
                            self._template_split_note, self._import_cfg_btn,
                            self._history_note], expanded=False)]

        def pending() -> PendingState:
            if not self._plot_armed:
                return PendingState(1, ("not plotted yet",))
            reasons = list(self._stale_reasons)
            if self._stale and not reasons:
                reasons.append(self._stale_reason or "the selection")
            if self._pending_fit_change():
                reasons.append("fit model")
            return PendingState(len(reasons), tuple(reasons))

        def history() -> HistoryState:
            # The tooltips already carry the 200-step ceiling clause (``_sync_config_buttons``).
            return HistoryState(
                self._undo_btn.isEnabled(), self._redo_btn.isEnabled(),
                self._undo_btn.toolTip(), self._redo_btn.toolTip(), self.history_mark())

        def controllers() -> list[object]:
            found = figure_text_for(self._plot)
            return [found] if found is not None else []

        def axis_log_target(_plot: str, side: str):
            pair = self._log_pair()
            return ("plot", pair[0 if side == "bottom" else 1]) if pair else None

        self._heatmap_cmap._nanoscope_format_visible = lambda: self._plot_type() == HEATMAP

        # Only population-specific controls are extras. Passing the entire Style group
        # (an ancestor of `appearance`) would select Format's legacy fallback and hide the
        # element inspector. The shared appearance widget supplies the common controls.
        spec = FormatSpec(
            appearance, controllers,
            extras={"figure": [self._font_size],
                    "legend": [self._legend_hint, self._legend_reset_btn,
                               self._legend_counts, self._legend_entries],
                    "caption": [self._prov_check],
                    "colour_map": [self._heatmap_cmap]},
            # Distributions reports seven concrete values at their defaults (grid on,
            # dark background, colours, line 1, marker 7, legend on): "set here" is
            # measured against these, not against ``PlotAppearance()``.
            defaults=lambda: PlotAppearance.from_mapping(
                PlotsWindow._appearance_of(StyleSettings())),
            views_changed=self._plot_type_combo.currentTextChanged,
            series_provider=self.figure_series,
            axis_log_target=axis_log_target)
        self._inspector_adapter = PageAdapter(
            key="distributions", page=self, groups=groups, format=spec,
            update=self._run_plot,
            update_tooltip="Draw the plot with the data, plot type and fit chosen in the "
                           "inspector, and apply any pending fit choice (⌘Return).",
            pending=pending, pending_changed=self.pendingChanged,
            undo=self.undo_config, redo=self.redo_config, history=history,
            history_changed=self.historyChanged,
            apply_look=self._apply_inspector_look,
            reset_figure=self._reset_inspector_figure,
            exports=lambda: [ExportEntry(text, run, kind=kind) for (text, run), kind
                             in zip(self._exports, ("figure", "data", "recipe"))],
            navigation=lambda: self._plot_nav, navigations=lambda: (self._plot_nav,),
            status_bar=self.statusBar, canvas_actions=(self._lasso_action,),
            chrome=[self._workflow_toolbar],
            on_adopt=self._on_adopt, on_retire=self._on_retire)
        return self._inspector_adapter

    def _on_adopt(self) -> None:
        """Hosting steps: what the Explorer's toolbar and inspector replace stays
        constructed and hidden (the compatibility facade behind the attribute names the
        workspace capture, the tests and the old Section API read)."""
        self._hosted = True
        # The shell owns Ctrl+Return from here on: Qt fires neither of two shortcuts on
        # one key in a window, and one on a hidden toolbar fires nothing.
        self._run_action.setShortcut(QtGui.QKeySequence())
        # The controls column is empty now: the central splitter holds the plot alone.
        self._section_area.hide()
        self._plot_widget.setMinimumWidth(280)
        self._plot_widget.setToolTip(
            "Scatter and PCA: click a point to inspect its event. "
            "Ctrl/Cmd/Shift-click adds or removes points from the selection. "
            "Enable Lasso select on the status line to select an outlined region.")
        for hidden in (self._plot_btn, self._fit_apply_btn, self._scatter_fit_apply_btn,
                       self._undo_btn, self._redo_btn, self._export_cfg_btn,
                       self._save_btn, self._save_data_btn, self._axis_range_group):
            hidden.hide()
        # The group boxes now sit inside the inspector's own group frames: no second frame
        # and no second set of side margins (the Plot group is 295 px wide with them, 5 over
        # the column's 290 px budget; 277 without). Their titles stay as sub-headings.
        for box in (self._source_group, self._plot_group, self._features_group,
                    self._cluster_group):
            box.setFlat(True)
            box.layout().setContentsMargins(0, 2, 0, 0)
        # The host made every moved root visible; the ceiling sentence is hidden until a
        # step has been dropped (``_sync_config_buttons`` owns that).
        self._history_note.setVisible(bool(self._history_note.text()))
        self._sync_run_state()          # the amber sentence names the verb that is on screen

    def _on_retire(self) -> None:
        self._inspector_reveal = None
        self._hosted = False

    @contextmanager
    def _format_transaction(self):
        """One inspector gesture, separated from any preceding open text edit."""
        suspended = self._history_suspended
        if not suspended:
            self._history_commit()
        self._history_suspended = True
        try:
            yield
        finally:
            self._history_suspended = suspended
            if not suspended:
                self._history_touch()

    def _apply_inspector_look(self, look: PlotAppearance, keep_hand: bool) -> bool:
        with self._format_transaction():
            return self._appearance_widget.apply_look(look, keep_hand=keep_hand)

    def _reset_inspector_figure(self) -> None:
        with self._format_transaction():
            self._appearance_widget.reset_figure()

    def _set_details_expanded(self, expanded: bool) -> None:
        """Collapse the audit trail to its title row while keeping the result visible."""
        self._details.setVisible(bool(expanded))
        self._details_box.setMaximumHeight(230 if expanded else 30)

    @staticmethod
    def _controls_min_width(scroll: QtWidgets.QScrollArea, controls: QtWidgets.QWidget) -> int:
        """The narrowest the controls column can be WITHOUT a horizontal scrollbar.

        The exact quantity the old code got backwards: it capped the column at 420 px
        while its contents reported a 444 px minimum, so 24 px of every row was always
        off the left edge and a horizontal scrollbar (range 0-44, measured) was pinned
        to the panel at rest. Stated as arithmetic rather than as a literal so the two
        halves cannot drift apart again when a control is added to a group.
        """
        return (controls.minimumSizeHint().width()
                + scroll.verticalScrollBar().sizeHint().width()
                + 2 * scroll.frameWidth())

    def _build_controls(self) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(host)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Keep handles for the compact section navigator. The groups themselves and
        # every child attribute remain unchanged for config/test/API compatibility.
        self._source_group = self._build_source_group()
        self._plot_group = self._build_plot_group()
        self._axis_range_group = self._build_axis_range_group()
        self._features_group = self._build_features_group()
        self._cluster_group = self._build_cluster_group()
        self._style_group = self._build_style_group()
        self._config_group = self._build_config_group()

        root.addWidget(self._source_group)
        root.addWidget(self._plot_group)
        root.addWidget(self._axis_range_group)
        root.addWidget(self._features_group)
        root.addWidget(self._cluster_group)
        root.addWidget(self._style_group)
        root.addWidget(self._config_group)

        self._save_btn = QtWidgets.QPushButton("Export current figure…")
        self._save_btn.setAccessibleName("Export current figure")
        self._save_btn.setToolTip(
            "Export the current plot as a picture. The extension chooses the format: "
            "PNG for a slide or a notebook, SVG or PDF for vector art a journal can "
            "typeset. PDF is the smaller and faster of the two on a busy scatter — "
            "SVG stores every point as its own element.")
        self._save_btn.clicked.connect(lambda: self.save_image())
        root.addWidget(self._save_btn)

        # THE NUMBERS, BESIDE THE PICTURE. ``save_figure`` has exported the source data
        # of every publication figure since it was written; the interactive panel --
        # where the user actually decides what the figure is -- exported none of it, so
        # a histogram's bin edges lived only inside a running Qt process.
        self._save_data_btn = QtWidgets.QPushButton("Export plot data…")
        self._save_data_btn.setAccessibleName("Export numeric plot data")
        self._save_data_btn.setToolTip(
            "Write the numbers behind the plot as a CSV: the bin edges and heights of "
            "a histogram (with the fitted curve), the plotted points of a scatter or "
            "PCA (with its loadings), or the density grid of a KDE. Everything is in "
            "stored units — seconds, picoamps — never the unit the axis is showing.")
        self._save_data_btn.clicked.connect(lambda: self.export_data())
        root.addWidget(self._save_data_btn)

        root.addStretch(1)
        return host

    def _build_source_group(self) -> QtWidgets.QGroupBox:
        """The source picker: a CHECKABLE LIST, because comparing two pores (or two
        segments of one recording) is the whole point and a combo can only ever
        show one of them. Each row is inked with the source's own overlay colour,
        so the list doubles as the legend key."""
        box = QtWidgets.QGroupBox("Data sources")
        lay = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(
            "Tick two or more to overlay them. Colour then means SOURCE, and the "
            "histogram normalizes so unequal-n recordings stay comparable.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{_MSG_HEX};")
        lay.addWidget(hint)

        self._source_list = QtWidgets.QListWidget()
        self._source_list.setMaximumHeight(150)
        self._source_list.setToolTip(
            "Every open recording (segments included) that has a stored detection, "
            "plus the manual-measurement table.\n"
            "Double-click a name to retype the legend entry for that source.")
        self._source_list.setAccessibleName("Plot data sources")
        self._source_list.itemChanged.connect(self._on_source_item_changed)
        self._source_list.currentItemChanged.connect(self._on_source_current_changed)
        lay.addWidget(self._source_list)

        # THE EDIT AFFORDANCE (owner note Q3). Renaming a row here has worked end to
        # end since the last round -- what was missing is that nothing on screen said
        # so, and a feature nobody can see is a feature nobody has. The rows cannot
        # carry a pencil in their own text (the rename is detected by comparing that
        # text with the source's real label, so a decoration would read as a rename),
        # so the affordance goes here, in the group, where it is legible before the
        # first click rather than after it.
        edit_hint = QtWidgets.QLabel(
            "✎ Double-click a name to retype its legend entry — clear it to go back "
            "to the file name.")
        edit_hint.setWordWrap(True)
        edit_hint.setStyleSheet(f"color:{_MSG_HEX};")
        lay.addWidget(edit_hint)

        # STACKED, not side by side. The controls column's minimum width is the widest
        # row in it and is a HARD budget (<= 360 px, pinned by
        # test_the_controls_column_is_never_capped_below_its_own_contents). Abreast,
        # these two buttons made a 339 px row, which measured this group at 363 px and
        # took the whole column to 379; stacked, the group measures 212 and the column
        # 303, where the Plot group below it is the widest thing showing.
        source_defaults = QtWidgets.QVBoxLayout()
        self._project_source_label = QtWidgets.QPushButton("Save name as project default")
        self._project_source_color = QtWidgets.QPushButton("Project source color…")
        self._project_source_label.setToolTip(
            "Use the highlighted source's edited legend name for newly generated figures in this project. "
            "This does not rename the file or change event labels.")
        self._project_source_color.setToolTip(
            "Choose the highlighted source's color for newly generated source-colored comparisons. "
            "The current figure keeps its committed ink until explicit Plot. Single-series, fit and cluster "
            "colors remain figure-specific.")
        self._project_source_label.clicked.connect(self._save_project_source_label)
        self._project_source_color.clicked.connect(self._pick_project_source_color)
        source_defaults.addWidget(self._project_source_label)
        source_defaults.addWidget(self._project_source_color)
        lay.addLayout(source_defaults)

        self._accepted_check = QtWidgets.QCheckBox("Use accepted events only")
        self._accepted_check.setChecked(True)
        self._accepted_check.toggled.connect(self._on_accepted_changed)
        lay.addWidget(self._accepted_check)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-pull the current measurements and events.")
        self._refresh_btn.clicked.connect(self.refreshRequested.emit)
        lay.addWidget(self._refresh_btn)
        return box

    def _build_plot_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Plot")
        lay = QtWidgets.QVBoxLayout(box)

        self._plot_type_combo = QtWidgets.QComboBox()
        self._plot_type_combo.addItems(list(_PLOT_TYPES))
        self._plot_type_combo.setAccessibleName("Population plot type")
        self._plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        lay.addWidget(self._plot_type_combo)

        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Time unit"))
        self._time_unit = QtWidgets.QComboBox()
        self._time_unit.addItem("ms")
        self._time_unit.setEnabled(False)
        self._time_unit.setToolTip(
            "Time values display in milliseconds. Stored coordinates, scientific fits "
            "and original CSV columns retain their declared units.")
        self._time_unit.currentTextChanged.connect(self._on_time_unit_changed)
        unit_row.addWidget(self._time_unit, 1)
        lay.addLayout(unit_row)

        self._pages = QtWidgets.QStackedWidget()
        self._pages.addWidget(self._build_histogram_page())
        self._pages.addWidget(self._build_scatter_page())
        self._pages.addWidget(self._build_kde_page())
        self._pages.addWidget(self._build_heatmap_page())   # _PLOT_TYPES order
        self._pages.addWidget(self._build_pca_page())
        lay.addWidget(self._pages)

        # THE RUN GATE'S ONE BUTTON. Placed at the FOOT of this group because that is
        # where the sentence ends: tick your sources above, pick a plot type and its
        # columns here, then press this. See the block in ``__init__`` for what is
        # gated and what stays live.
        self._plot_btn = QtWidgets.QPushButton("Plot")
        self._plot_btn.setAccessibleName("Plot or apply pending analysis")
        self._plot_btn.clicked.connect(self._run_plot)
        lay.addWidget(self._plot_btn)

        # ...and the gate has to be VISIBLE, or it is just a control that stopped
        # working -- the same reasoning that put the ● on "Apply fit". A stale plot is
        # a plot whose picture and whose controls disagree, and the picture is the
        # convincing half, so the disagreement is named in words and in the warning ink
        # the rest of the app uses for a verdict.
        self._plot_state = QtWidgets.QLabel("")
        self._plot_state.setWordWrap(True)
        self._plot_state.setVisible(False)
        lay.addWidget(self._plot_state)
        return box

    def _build_histogram_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)

        self._hist_col = ScientificColumnCombo()
        self._hist_col.currentTextChanged.connect(
            lambda t, w=self._hist_col: self._on_axis_column_changed(t, w))
        form.addRow("Column", self._hist_col)

        self._bins = QtWidgets.QSpinBox()
        self._bins.setRange(2, 500)
        self._bins.setValue(30)
        self._bins.setToolTip('Live preview: changing bins immediately redraws the current histogram. Pending source, column and fit selections still require Update plot.')
        self._bins.valueChanged.connect(self._changed)
        form.addRow("Bins (live preview)", self._bins)

        self._norm_combo = QtWidgets.QComboBox()
        self._norm_combo.addItems(list(_NORMS))
        # THE TOOLTIP CARRIES THE ARITHMETIC THE ITEM'S NAME USED TO CLAIM. "density
        # (area = 1)" was true only on a linear axis ticked in the column's stored unit --
        # which is not this app's default (Time unit = ms) and not what its own shipped
        # log-x dwell template selects. The promise is restated here WITH its condition
        # attached, and the figure states its own unit on the y axis (S6-03).
        self._norm_combo.setToolTip(
            "What the y axis counts.\n"
            "count: raw events per bin — only comparable at equal n.\n"
            "density: counts / (n × bin width) — the shape, independent of how long each "
            "recording ran. THE one to use when overlaying. The heights are per unit of "
            "the column IN ITS STORED UNIT (per second for a dwell, per pA for a depth), "
            "which is what the y axis says; they integrate to 1 over that unit — not over "
            "a ms/µs tick scale, and not over a log axis.\n"
            "probability per bin: counts / n — bin heights sum to 1.")
        # AHEAD OF `_changed`, WHICH REDRAWS. Slots run in connection order, and the
        # draw is what rebuilds Details out of `_norm_refusal` -- connected the other way
        # round, a hand pick of the y quantity would redraw ONCE MORE carrying the
        # refusal sentence for a mode the user has just replaced by hand. See
        # `_on_norm_moved` for what this clears and why the writer's own writes do not.
        self._norm_combo.currentTextChanged.connect(self._on_norm_moved)
        self._norm_combo.currentTextChanged.connect(self._changed)
        form.addRow("Normalize", self._norm_combo)
        # Set by _sync_norm_default when the source count crosses 1, unless the user
        # has picked a normalization themselves (then their choice is left alone).
        self._norm_user_set = False
        self._norm_combo.activated.connect(lambda *_: setattr(self, "_norm_user_set", True))

        self._hist_colour_by = QtWidgets.QComboBox()
        self._hist_colour_by.addItems([_COLOUR_NONE, _COLOUR_CLUSTER])
        self._hist_colour_by.setToolTip(
            "Split the histogram by cluster label after Run clustering — the other half of "
            "the clustering workflow (scatter/PCA already colour by cluster).\n"
            "Single source only: with an overlay up, colour already means SOURCE.")
        self._hist_colour_by.currentTextChanged.connect(self._changed)
        form.addRow("Colour by", self._hist_colour_by)

        self._hist_logx = QtWidgets.QCheckBox("Log x")
        self._hist_logx.toggled.connect(self._changed)
        form.addRow(self._hist_logx)
        self._hist_logy = QtWidgets.QCheckBox("Log y (count)")
        self._hist_logy.setToolTip("Log-scale the count axis — makes an exponential tail read as a line.")
        self._hist_logy.toggled.connect(self._changed)
        form.addRow(self._hist_logy)

        # THE KDE OVERLAY IS AN ANALYSIS, so unlike the two log toggles above it does
        # not redraw on a click: it estimates a new curve from the sample, and a curve
        # that appears the moment the box is ticked is the "a proposal took effect on
        # its own" surprise the Plot / Apply gate exists to prevent (owner note #12).
        self._hist_kde = QtWidgets.QCheckBox("KDE overlay")
        self._hist_kde.setAccessibleName("Histogram KDE overlay")
        self._hist_kde.setToolTip(
            "Draw one Gaussian KDE (Scott bandwidth, as the violins use) per drawn series, "
            "dotted, scaled to the y quantity the bars show — count: n × bin width × f; "
            "density: f per the column's stored unit; probability per bin: bin width × f.\n"
            "On a log x axis the KDE is estimated on log10 of the column, matching the "
            "log-spaced bins.\nAn analysis setting: update the plot to draw it. "
            "Details names each series' bandwidth.")
        self._hist_kde.toggled.connect(self._on_kde_overlay_toggled)
        form.addRow(self._hist_kde)

        self._fit_combo = QtWidgets.QComboBox()
        self._fit_combo.addItems([_FIT_NONE, _FIT_EXP, _FIT_GAMMA, _FIT_INVGAUSS,
                                  _FIT_EXP_GAMMA, _FIT_GMM])
        self._fit_combo.setToolTip(
            "exponential: the classic escape model — monotonic, cannot peak.\n"
            "gamma / inverse-gaussian: densities WITH a mode, for a dwell histogram\n"
            "that peaks away from zero. Dwell models condition on the detector's\n"
            "min_dwell_s and explicit fit range. Gaussian mixtures condition on an\n"
            "explicit fit range; an inferred detector depth floor is warned about\n"
            "but is not silently treated as a hard truncation.")
        self._fit_combo.currentTextChanged.connect(self._on_fit_changed)
        # THE CONFIRMATION STEP (owner note #12). Choosing a model from the dropdown
        # used to fit it, draw it and start its advisory batch on the spot -- so
        # walking down the list to READ the options ran five maximisations, one of
        # which (exponential + gamma) is a nine-second twelve-start optimisation. The
        # selection is now a proposal; this button is what makes it the drawn fit.
        # UNSTACKED (layout note): combo and button used to share one row, and between
        # them they set the whole controls column's minimum width to 444 px -- 24 px
        # more than the column's own maximum, which is what pinned a horizontal
        # scrollbar to the panel permanently. The button is a full-width row of its own
        # now, which also reads better: it is a COMMIT, and a commit that looks like a
        # spinner's neighbour is easy to miss (that was the whole failure mode note #12
        # was written about).
        _narrow_combo(self._fit_combo)
        form.addRow("Fit", self._fit_combo)
        self._fit_apply_btn = QtWidgets.QPushButton("Apply fit")
        # ONE GATE, TWO DOORS (round-4 Q1). This no longer means something different
        # from the window's Plot button: both run :meth:`_run_plot`, which commits the
        # pending fit AND the pending data/type/column selection and then draws once.
        # Two buttons that each half-recompute is the "two adjacent buttons that both
        # mean recompute" problem; two buttons that are the same commit, offered where
        # the user's hand already is, is not.
        self._fit_apply_btn.clicked.connect(self._run_plot)
        form.addRow("", self._fit_apply_btn)

        # Component count, with the SAME Auto (BIC) affordance the clustering group has
        # had all along. Its absence here was the asymmetry in report Medium-10: the
        # panel would tell you another k fits markedly better and then make you go and
        # set it by hand, having searched a narrower range than the spin offers.
        fit_k_row = QtWidgets.QHBoxLayout()
        self._fit_k = QtWidgets.QSpinBox()
        self._fit_k.setRange(1, 4)
        self._fit_k.setValue(1)
        self._fit_k.setToolTip("Number of mixture components for the fit.")
        self._fit_k.setEnabled(False)
        self._fit_k.valueChanged.connect(self._on_fit_k_changed)
        fit_k_row.addWidget(self._fit_k)
        self._fit_k_auto = QtWidgets.QCheckBox("Auto (BIC)")
        self._fit_k_auto.setEnabled(False)
        self._fit_k_auto.setToolTip(
            "Choose the component count by BIC, over the whole range this spin offers "
            "(1–4), refitting the SAME sample under the SAME censoring so the values "
            "compare.\nOff: your k stands, and Details still reports when another one "
            "fits markedly better.")
        self._fit_k_auto.toggled.connect(self._on_fit_k_auto_toggled)
        fit_k_row.addWidget(self._fit_k_auto)
        fit_k_row.addStretch(1)
        form.addRow("Fit components", fit_k_row)

        # FIT RANGE. Distinct from the Axis range group below, which only moves the
        # camera: this changes WHICH DATA the fit sees, so it is a modelling control and
        # belongs beside the model.
        fit_range = QtWidgets.QHBoxLayout()
        self._fit_lo = QtWidgets.QLineEdit(); self._fit_lo.setPlaceholderText("min")
        self._fit_hi = QtWidgets.QLineEdit(); self._fit_hi.setPlaceholderText("max")
        for e in (self._fit_lo, self._fit_hi):
            e.setToolTip(
                "Fit only this part of the distribution, in the AXIS unit shown.\n\n"
                "This is a TRUNCATION, not a crop: the likelihood is renormalised to the\n"
                "range, so restricting to a peak does not bias the estimates the way\n"
                "silently dropping data would. BIC is only comparable between fits over\n"
                "the SAME range.")
            e.editingFinished.connect(self._changed)
            fit_range.addWidget(e)
        form.addRow("Fit range", fit_range)
        self._fit_lo.textChanged.connect(self._validate_fit_range_fields)
        self._fit_hi.textChanged.connect(self._validate_fit_range_fields)

        # Filter-limited events are EXCLUDED by default, matching the Fits panel
        # and analysis.modules. Their dwell is within ~2x the filter rise time, so
        # the low-pass never let the current reach the true blockade level: depth
        # attenuated, dwell biased long. figures.scatter_dwell_depth already draws
        # them as open markers over a shaded band; the fits are as honest.
        self._use_filter_limited = QtWidgets.QCheckBox("include filter-limited events")
        self._use_filter_limited.setChecked(False)
        self._use_filter_limited.setEnabled(False)
        self._use_filter_limited.setToolTip(
            "Off (default): events whose dwell is within ~2x the filter rise time are "
            "excluded from the fit (depth attenuated, dwell biased long) and the count "
            "is reported in Details.\nOn: they are fit alongside the resolved events.")
        self._use_filter_limited.toggled.connect(self._changed)
        form.addRow(self._use_filter_limited)
        return page

    def _build_scatter_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)

        self._scatter_x = ScientificColumnCombo()
        self._scatter_x.currentTextChanged.connect(
            lambda t, w=self._scatter_x: self._on_axis_column_changed(t, w))
        form.addRow("X", self._scatter_x)

        self._scatter_y = ScientificColumnCombo()
        self._scatter_y.currentTextChanged.connect(
            lambda t, w=self._scatter_y: self._on_axis_column_changed(t, w))
        form.addRow("Y", self._scatter_y)

        self._scatter_logx = QtWidgets.QCheckBox("Log x")
        self._scatter_logx.toggled.connect(self._changed)
        form.addRow(self._scatter_logx)
        self._scatter_logy = QtWidgets.QCheckBox("Log y")
        self._scatter_logy.toggled.connect(self._changed)
        form.addRow(self._scatter_logy)

        self._colour_by = QtWidgets.QComboBox()
        self._colour_by.addItems([_COLOUR_NONE, _COLOUR_CLUSTER])
        self._colour_by.currentTextChanged.connect(self._changed)
        form.addRow("Colour by", self._colour_by)

        self._scatter_fit = QtWidgets.QComboBox()
        self._scatter_fit.addItems([_SFIT_NONE, _SFIT_LINEAR, _SFIT_POWER])
        self._scatter_fit.setToolTip(
            "Quick regression: straight line uses raw response; power law fits in log-log space. "
            "Use Update plot (or Apply fit) to apply a selection. Curve fit… offers raw-response "
            "models, including a straight line, with constraints and diagnostics.")
        self._scatter_fit.currentTextChanged.connect(self._on_scatter_fit_changed)
        # Same confirmation step as the histogram's (owner note #12): the selection is
        # a proposal, this button draws it. Cheaper per fit than the histogram's, but a
        # regression line that appears the instant the combo is opened is the same
        # surprise, and one gate that behaves the same way on both pages is one rule.
        _narrow_combo(self._scatter_fit)
        form.addRow("Quick fit", self._scatter_fit)
        self._scatter_fit_apply_btn = QtWidgets.QPushButton("Apply fit")
        self._scatter_fit_apply_btn.clicked.connect(self._run_plot)   # the same one gate
        form.addRow("", self._scatter_fit_apply_btn)
        self._scatter_fit_status = QtWidgets.QLabel("Quick fit applied: none.")
        self._scatter_fit_status.setWordWrap(True);form.addRow(self._scatter_fit_status)
        self._nonlinear_fit_btn = QtWidgets.QPushButton("Curve fit…")
        self._nonlinear_fit_btn.setToolTip("Straight line, Gaussian peaks, exponential decay, logistic / Hill and custom X–Y models in raw-response space; parameter constraints, uncertainty and residuals.")
        self._nonlinear_fit_btn.clicked.connect(self.open_nonlinear_fit)
        form.addRow("", self._nonlinear_fit_btn)
        self._nonlinear_fit_status = QtWidgets.QLabel("No curve fit applied.")
        self._nonlinear_fit_status.setWordWrap(True);form.addRow(self._nonlinear_fit_status)
        self._sync_fit_apply_buttons()
        return page

    def _build_kde_page(self) -> QtWidgets.QWidget:
        """Owner note #19: the 2-D KDE used to live only in the separate export figure
        and the density window; here it is a plot type beside the others, with the same
        source overlay, styling and Save image the rest of the window has."""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)
        self._kde_quantity_note = QtWidgets.QLabel(
            "KDE probability density — smoothed contours, not event-count bins. "
            "Log axes express density per decade; Details records the full units.")
        self._kde_quantity_note.setWordWrap(True)
        form.addRow(self._kde_quantity_note)

        self._kde_x = ScientificColumnCombo()
        self._kde_x.currentTextChanged.connect(
            lambda t, w=self._kde_x: self._on_axis_column_changed(t, w))
        form.addRow("X", self._kde_x)

        self._kde_y = ScientificColumnCombo()
        self._kde_y.currentTextChanged.connect(
            lambda t, w=self._kde_y: self._on_axis_column_changed(t, w))
        form.addRow("Y", self._kde_y)

        self._kde_logx = QtWidgets.QCheckBox("Log x")
        # THE DEFAULT LIVES IN THE CONFIG, not in this line: see KdeConfig.log_x for the
        # value and for why this page's default differs from the other two. A literal
        # here is how a fresh window and a `reset_state`'d one come to disagree.
        self._kde_logx.setChecked(KdeConfig().log_x)
        self._kde_logx.setToolTip(
            "On by default on this page only — a dwell axis drawn linearly crushes a "
            "log-distributed population into the left edge, where the contours merge "
            "into one blob. It is a default, not a claim about your data; turn it off "
            "for a density over two linear quantities.")
        self._kde_logx.toggled.connect(self._changed)
        form.addRow(self._kde_logx)
        self._kde_logy = QtWidgets.QCheckBox("Log y")
        self._kde_logy.toggled.connect(self._changed)
        form.addRow(self._kde_logy)

        self._kde_levels = QtWidgets.QSpinBox()
        self._kde_levels.setRange(3, 20)
        self._kde_levels.setValue(8)
        self._kde_levels.setToolTip(
            "Contour lines per source, evenly spaced between 5% and 95% of that "
            "source's own peak density — each population keeps its shape even when "
            "another has far more events.")
        self._kde_levels.valueChanged.connect(self._changed)
        form.addRow("Contours", self._kde_levels)
        return page

    def _build_heatmap_page(self) -> QtWidgets.QWidget:
        """The Event-count heatmap: two columns binned into a 2-D count image.

        THE SAME KERNEL AS EVENT WAVEFORMS' HEATMAP (``event_shape.density_2d``), so the two
        windows cannot disagree about a bin. ONE SOURCE AT A TIME, the first ticked one: a
        count image has one colour channel and it is spent on the count, so two sources
        cannot share it the way two histograms share an axis, and summing them would be
        a pooled count nobody asked for. Said here, in the title and in Details.
        """
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)
        self._heatmap_note = QtWidgets.QLabel(
            "Event counts in two-dimensional bins — counts, not a KDE density. Shows the "
            "first ticked source only. Lasso and point-click selection need points and "
            "are not available on this plot.")
        self._heatmap_note.setWordWrap(True)
        form.addRow(self._heatmap_note)

        self._heatmap_x = ScientificColumnCombo()
        self._heatmap_x.currentTextChanged.connect(
            lambda t, w=self._heatmap_x: self._on_axis_column_changed(t, w))
        form.addRow("X", self._heatmap_x)

        self._heatmap_y = ScientificColumnCombo()
        self._heatmap_y.currentTextChanged.connect(
            lambda t, w=self._heatmap_y: self._on_axis_column_changed(t, w))
        form.addRow("Y", self._heatmap_y)

        self._heatmap_logx = QtWidgets.QCheckBox("Log x")
        self._heatmap_logx.setChecked(HeatmapConfig().log_x)   # the default lives there
        self._heatmap_logx.setToolTip(
            "Bin x in log10 space (geometric edges); non-positive values are dropped and "
            "counted in Details. On by default, as in Event waveforms.")
        self._heatmap_logx.toggled.connect(self._changed)
        form.addRow(self._heatmap_logx)
        self._heatmap_logy = QtWidgets.QCheckBox("Log y")
        self._heatmap_logy.setChecked(HeatmapConfig().log_y)
        self._heatmap_logy.toggled.connect(self._changed)
        form.addRow(self._heatmap_logy)

        self._heatmap_bins = QtWidgets.QSpinBox()
        self._heatmap_bins.setRange(4, 400)
        self._heatmap_bins.setValue(HeatmapConfig().bins)
        self._heatmap_bins.setToolTip(
            "Bins per axis (bins × bins cells), spanning the source's own min to max on "
            "each axis — Event waveforms' rule.")
        self._heatmap_bins.valueChanged.connect(self._changed)
        form.addRow("Bins", self._heatmap_bins)
        return page

    def _build_pca_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        hint = QtWidgets.QLabel(
            "Projects the feature set below to 2 components. Colour follows the "
            "clustering if one has been run.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{_MSG_HEX};")
        lay.addWidget(hint)
        self._pca_btn = QtWidgets.QPushButton("Run PCA")
        self._pca_btn.clicked.connect(self.run_pca)
        lay.addWidget(self._pca_btn)
        return page

    def _build_features_group(self) -> QtWidgets.QGroupBox:
        # "&&": a QGroupBox title takes "&" as a mnemonic, which drew "clustering  PCA"
        # with an underlined P.
        box = QtWidgets.QGroupBox("Features (clustering && PCA)")
        lay = QtWidgets.QVBoxLayout(box)
        # A CHECKLIST, not a multi-selection list: a click on one row used to be the only
        # way to change a selection that a stray click could also wipe. One click, Space,
        # Select all, Clear or Default is one gesture -- one `changed`, one Undo step
        # (`_wire_history` cannot see a custom signal, so the slot touches the history).
        self._features = ColumnChecklist(noun="features", minimum=2)
        self._features.set_visible_rows(7)
        self._features.layout().setContentsMargins(0, 0, 0, 0)
        self._features.view.setAccessibleName("Clustering and PCA features")
        self._features.set_separator_before(_IDENTIFIER_FEATURES[0], "Identifiers")
        self._features.changed.connect(self._on_features_checked)
        lay.addWidget(self._features)
        note = QtWidgets.QLabel("Pick ≥2 columns; they are z-scored before fitting.")
        note.setStyleSheet(f"color:{_MSG_HEX};")
        note.setWordWrap(True)
        lay.addWidget(note)
        return box

    def _build_cluster_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Clustering")
        form = QtWidgets.QFormLayout(box)

        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems([_METHOD_GMM, _METHOD_KMEANS])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        form.addRow("Method", self._method_combo)

        krow = QtWidgets.QHBoxLayout()
        self._k_mode = QtWidgets.QComboBox()
        self._k_mode.addItems([_K_AUTO, _K_FIXED])
        self._k_mode.currentTextChanged.connect(self._on_kmode_changed)
        self._k_spin = QtWidgets.QSpinBox()
        self._k_spin.setRange(1, 20)
        self._k_spin.setValue(2)
        self._k_spin.setEnabled(False)
        krow.addWidget(self._k_mode, 1)
        krow.addWidget(self._k_spin)
        form.addRow("k", krow)

        self._cluster_btn = QtWidgets.QPushButton("Run clustering")
        self._cluster_btn.clicked.connect(self.run_clustering)
        form.addRow(self._cluster_btn)
        return box

    def _build_axis_range_group(self) -> QtWidgets.QGroupBox:
        """Explicit axis-range control. Blank fields = auto (fit the data); typed limits pin that
        axis and persist across redraws until Auto is pressed or the axis comes to show another
        quantity (a new plot type, or a new column on that axis: ``PlotConfig.rebased``, clause
        2); Reset figure leaves them. Values are read in the unit the axis DISPLAYS (a time axis
        follows the Time-unit combo — type the number you read off the ticks), converted to data
        units and to log10 internally as needed. (The plot is also mouse zoom/pan-able.)"""
        box = QtWidgets.QGroupBox("Axis range")
        box.setToolTip(
            "Type the numbers as the axis shows them: a time axis follows the Time "
            "unit combo (ms by default), everything else is in data units.")
        form = QtWidgets.QFormLayout(box)
        self._range_edits: dict[str, QtWidgets.QLineEdit] = {}
        # THE ROW LABELS ARE KEPT, because they are where the unit goes. "X min" beside
        # a box that is read in whatever the Time-unit combo says is an invitation to
        # type a number in the wrong unit -- owner note #4, second half: the tooltip
        # said which unit, the label did not, and a tooltip is not on screen while you
        # are typing. :meth:`_sync_range_unit_labels` rewrites these to "X min (ms)" /
        # "Y min (pA)" from the column actually on that axis.
        self._range_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label in (("xmin", "X min"), ("xmax", "X max"),
                           ("ymin", "Y min"), ("ymax", "Y max")):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("auto")
            edit.setValidator(QtGui.QDoubleValidator())
            edit.returnPressed.connect(self._apply_axis_range)
            self._range_edits[key] = edit
            self._range_labels[key] = QtWidgets.QLabel(label)
            form.addRow(self._range_labels[key], edit)
        row = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_axis_range)
        # AUTO IS PER AXIS NOW, and "Auto" alone is the both-axes case rather than the
        # only case. Measured need: pinning y to a fixed count ceiling so two overlaid
        # histograms are comparable, while x is re-fit for each pore, was a five-gesture
        # job (Auto, retype ymin, retype ymax, Apply) because the one button threw both
        # axes away. The three buttons are the same operation at three scopes, so they
        # share one implementation and differ by one argument.
        # ONE-LETTER SCOPES, for the same width budget the Style checkboxes obey: four
        # full-width buttons on one row set the whole controls column's minimum. The
        # tooltips carry the sentence.
        auto_x = QtWidgets.QPushButton("Auto x")
        auto_x.setToolTip("Clear the X limits only; the Y pin stands.")
        auto_x.clicked.connect(lambda *_: self._reset_axis_range(axis="x"))
        auto_y = QtWidgets.QPushButton("Auto y")
        auto_y.setToolTip("Clear the Y limits only; the X pin stands.")
        auto_y.clicked.connect(lambda *_: self._reset_axis_range(axis="y"))
        auto_btn = QtWidgets.QPushButton("Auto")
        auto_btn.setToolTip("Clear the limits and auto-fit BOTH axes to the data.")
        auto_btn.clicked.connect(lambda *_: self._reset_axis_range(axis="both"))
        # TWO ROWS OF TWO, not one row of four. The controls column's minimum width is
        # the widest row in it and is a HARD budget (<= 360 px, pinned by
        # test_the_controls_column_is_never_capped_below_its_own_contents -- a column
        # wider than its cap is what pinned a horizontal scrollbar to this panel at
        # rest). Four buttons abreast measured this group at 362 px, which took the whole
        # column to 378; stacked, the same four controls measure 190 and the column 331.
        # The commit pair goes on top, the per-axis pair
        # under it, which also reads as scope: whole plot, then one axis.
        row.addWidget(apply_btn)
        row.addWidget(auto_btn)
        form.addRow(row)
        per_axis = QtWidgets.QHBoxLayout()
        per_axis.addWidget(auto_x)
        per_axis.addWidget(auto_y)
        form.addRow(per_axis)
        # NO SILENT NO-OPS (the axis half of the rule the Fit-range boxes already follow;
        # see _fit_range_input_notes). Three typed instructions were declined in silence
        # -- reversed limits, a non-positive bound on a log axis, and a half-filled pair
        # -- with the numbers still sitting in the boxes and the axis behaving as if they
        # were empty. This label is where the window says which, ON the group that was
        # ignored rather than in Details: Apply does not redraw, so a Details line would
        # not appear until the next draw, by which time the user has moved on.
        self._range_note = QtWidgets.QLabel("")
        self._range_note.setWordWrap(True)
        self._range_note.setStyleSheet(f"color:{_WARN_HEX};")
        self._range_note.setVisible(False)
        form.addRow(self._range_note)
        return box

    def _build_style_group(self) -> QtWidgets.QGroupBox:
        """Plot-design controls: title, font, colours, marker/line sizes, theme and export size.

        Every control writes straight onto :attr:`_style` and re-renders live, so a colour or
        title change is visible immediately. Values are set BEFORE the change signals are wired
        so building the group never triggers a redraw mid-construction (the same pattern the other
        groups use)."""
        box = QtWidgets.QGroupBox("Style")
        form = QtWidgets.QFormLayout(box)
        # NO HISTORY OF ITS OWN (design §4.2). Every appearance edit already lands on
        # this window's `_ConfigHistory` as part of the whole `PlotConfig`
        # (`_appearance_from_user` -> `_history_settle`), and one window must not have
        # two stacks: a second would answer the same Ctrl+Z this window binds itself,
        # and Qt fires neither of two shortcuts that share a key in one window.
        self._appearance_widget = PlotAppearanceWidget(compact=True, history=False)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        form.addRow(self._appearance_widget)

        self._title_edit = QtWidgets.QLineEdit()
        self._title_edit.setPlaceholderText("auto")
        self._title_edit.setToolTip("Override the plot title; blank uses the automatic title.")
        self._title_edit.textChanged.connect(self._on_style_title)
        form.addRow("Title", self._title_edit)

        # Axis-label overrides, beside the title they parallel (owner request: the
        # title was editable and the axes were not). TEXT ONLY — the tick scaling
        # (Time unit, log decades) still runs, so a custom label sits over correctly
        # scaled numbers rather than being allowed to lie about them.
        self._x_label_edit = QtWidgets.QLineEdit()
        self._x_label_edit.setPlaceholderText("auto")
        self._x_label_edit.setToolTip(
            "Override the x-axis label; blank uses the automatic one. Text only — "
            "the tick numbers keep their unit scaling.")
        self._x_label_edit.textChanged.connect(self._on_style_x_label)
        form.addRow("X label", self._x_label_edit)

        self._y_label_edit = QtWidgets.QLineEdit()
        self._y_label_edit.setPlaceholderText("auto")
        # THE UNIT GOES WITH THE TEXT, and the tooltip says so (S6-03). The automatic
        # label is the only place an exported density figure states what its heights are
        # in ("density (1/s)"); an override REPLACES it, exactly as the x override does,
        # so a user who types "density" here gets the pre-S6-03 picture back. That is
        # theirs to choose -- Details still names the unit -- but it must not be a
        # surprise, because the figure is the half that leaves the app.
        self._y_label_edit.setToolTip(
            "Override the y-axis label; blank uses the automatic one. Text only — "
            "the tick numbers keep their unit scaling.\n"
            "Your text REPLACES the whole label, including the unit the automatic one "
            "states (e.g. “density (1/s)”), so put the unit in yours if the figure "
            "needs it.")
        self._y_label_edit.textChanged.connect(self._on_style_y_label)
        form.addRow("Y label", self._y_label_edit)

        self._legend_check = QtWidgets.QCheckBox("Show legend")
        self._legend_check.setChecked(self._style.show_legend)
        self._legend_check.setToolTip(
            "Untick to draw every plot without a legend — the full source names and "
            "fit summaries stay in Details.")
        self._legend_check.toggled.connect(self._on_style_show_legend)
        form.addRow("", self._legend_check)

        # LEGEND POSITION IS A DRAG, NOT A SETTING (round-4 Q4). The four-corner combo
        # that stood here is exactly what the owner rejected: "the location should not
        # be limited to four fixed positions in the panel — it should be interactive,
        # allowing the user to decide exactly where to place the legend." pyqtgraph's
        # LegendItem always was draggable; the combo's per-draw ``setOffset`` is what
        # broke it. What is left is a hint that says the legend can be dragged, and a
        # button to undo a drag that went somewhere useless.
        legend_move = self._legend_hint = QtWidgets.QLabel("Drag the legend on the plot to place it.")
        legend_move.setWordWrap(True)
        legend_move.setStyleSheet(f"color:{_MSG_HEX};")
        form.addRow("", legend_move)
        self._legend_reset_btn = QtWidgets.QPushButton("Reset legend position")
        self._legend_reset_btn.clicked.connect(self._reset_legend_position)
        form.addRow("", self._legend_reset_btn)
        self._sync_legend_reset_button()

        self._legend_font_size = QtWidgets.QDoubleSpinBox()
        self._legend_font_size.setDecimals(3)
        self._legend_font_size.setSingleStep(.25)
        self._legend_font_size.setRange(0, 48)
        self._legend_font_size.setSpecialValueText("auto (body − 3)")
        self._legend_font_size.setValue(self._style.legend_font_size_pt)
        self._legend_font_size.setToolTip(
            "Legend text size in pt. 0 keeps the automatic rule — three points below "
            "the body font — which is what this window has always used.")
        self._legend_font_size.valueChanged.connect(self._on_style_legend_font_size)
        form.addRow("Legend font size", self._legend_font_size)

        self._legend_counts = QtWidgets.QCheckBox("Legend shows (n = …)")
        self._legend_counts.setChecked(self._style.legend_counts)
        self._legend_counts.setToolTip(
            "Append each series' event count to its legend entry. Untick it when you "
            "have RETYPED a legend entry (double-click a source on the left) and do "
            "not want the count welded onto your own text.")
        self._legend_counts.toggled.connect(self._on_style_legend_counts)
        form.addRow("", self._legend_counts)

        # WHICH ROWS THE LEGEND CARRIES (owner note Q2). A fitted histogram legends the
        # bars AND the curve, and the owner is right that one population wanting two
        # rows is usually wrong in a figure — but WHICH of the two to drop is editorial,
        # so neither is dropped by default and both are unticking distance away.
        #
        # A LIST REBUILT FROM THE DRAW, rather than a fixed pair of category
        # checkboxes: the entries are per SOURCE as well as per kind (five overlaid
        # pores produce five bar rows and up to five fit rows), so a control that
        # scales has to be as long as the legend is. It lists exactly what the last
        # draw offered, ticked or not, so nothing can be unticked and then lost.
        self._legend_entries = QtWidgets.QListWidget()
        self._legend_entries.setMaximumHeight(96)
        self._legend_entries.setToolTip(
            "Every row this plot offers the legend. Untick one to leave it out — "
            "typically the bars or the fit curve, not both, when a fit is applied.")
        self._legend_entries.itemChanged.connect(self._on_legend_entry_toggled)
        form.addRow("Legend entries", self._legend_entries)

        # THE FONT ROW IS THE SHARED ``FontPicker``: Arial (default), Helvetica, Times New Roman,
        # Courier New and Other font…, where this row used to list every installed family.
        # Its first row stores the blank this style has always meant by "no family chosen"
        # (and draws in Arial), a family the list lacks gets one extra row carrying its exact
        # name, and showing a style on it (``_select_font_family``) never commits anything:
        # only a pick reaches ``_on_style_font_family``, through ``changed``.
        self._font_combo = FontPicker()
        self._select_font_family(self._style.font_family)
        self._font_combo.changed.connect(self._on_style_font_family)
        form.addRow("Font", self._font_combo)

        self._font_size = QtWidgets.QDoubleSpinBox()
        self._font_size.setDecimals(3)
        self._font_size.setSingleStep(.25)
        self._font_size.setRange(6, 48)
        self._font_size.setValue(self._style.font_size_pt)
        self._font_size.setToolTip(
            "The BODY size, in pt. Every text below follows it unless it has been "
            "given a size of its own.")
        self._font_size.valueChanged.connect(self._on_style_font_size)
        form.addRow("Font size (body)", self._font_size)

        # INDEPENDENT SIZES (owner note Q5: "there is only one option to increase the
        # font size of all text in the plot — this is incorrect. The title and axis
        # labels need independent styling controls"). One number used to drive the
        # title, both axis labels, the tick labels, the on-plot fit box and — minus
        # three — the legend, so a title big enough for a talk dragged the tick labels
        # up with it until they collided. Each spin's 0 means "follow the body size",
        # which is the single-knob behaviour, so an untouched window looks exactly as
        # it did.
        self._title_size = self._pt_override_spin(
            self._style.title_font_size_pt,
            "Title size in pt. 0 follows the body size.")
        form.addRow("Title size", self._title_size)
        self._title_size.valueChanged.connect(self._on_style_title_size)

        self._axis_label_size = self._pt_override_spin(
            self._style.axis_label_font_size_pt,
            "Axis-label size in pt (the x and y captions). 0 follows the body size.")
        form.addRow("Axis label size", self._axis_label_size)
        self._axis_label_size.valueChanged.connect(self._on_style_axis_label_size)

        self._tick_size = self._pt_override_spin(
            self._style.tick_font_size_pt,
            "Tick-number size in pt. 0 follows the body size.")
        form.addRow("Tick label size", self._tick_size)
        self._tick_size.valueChanged.connect(self._on_style_tick_size)

        # GRIDLINES (owner note Q7: "the user needs control over whether gridlines are
        # shown or not"). It was `showGrid(x=True, y=True, alpha=0.15)` once, at
        # construction, and never read again — the one part of the canvas the Style
        # group could not touch.
        self._grid_check = QtWidgets.QCheckBox("Gridlines")
        self._grid_check.setChecked(self._style.show_grid)
        self._grid_check.setToolTip(
            "Faint gridlines behind the data (the app's usual 15% alpha). Off is the "
            "usual choice for a publication figure.")
        self._grid_check.toggled.connect(self._on_style_grid)
        form.addRow("", self._grid_check)

        self._data_swatch = self._make_swatch(
            self._style.data_color, self._pick_data_color,
            "Colour of the histogram bars and the plain scatter/PCA points.")
        form.addRow("Data colour", self._data_swatch)

        self._fit_swatch = self._make_swatch(
            self._style.fit_color, self._pick_fit_color,
            "Colour of the fitted overlay curve and its fit-label line.")
        form.addRow("Fit colour", self._fit_swatch)

        # The heatmap's colour map, beside the other colours because it is one: it
        # recolours the counts already drawn (see StyleSettings.heatmap_colormap).
        self._heatmap_cmap = QtWidgets.QComboBox()
        self._heatmap_cmap.addItems(list(_HEATMAP_COLORMAPS))
        self._heatmap_cmap.setCurrentText(self._style.heatmap_colormap)
        self._heatmap_cmap.setAccessibleName("Event-count heatmap colormap")
        self._heatmap_cmap.setToolTip(
            "Sequential colour map of the Event-count heatmap and its colour bar. "
            "Appearance only: the counts are not re-binned and the zoom is kept.")
        self._heatmap_cmap.currentTextChanged.connect(self._on_style_heatmap_colormap)
        form.addRow("Heatmap colormap", self._heatmap_cmap)

        self._ci_band_check = QtWidgets.QCheckBox("Confidence band")
        self._ci_band_check.setToolTip(
            "A 95% POINTWISE band from the bootstrap replicates of the whole parameter "
            "vector — not the curve redrawn at the ends of each parameter's interval, "
            "which is a corner of a box the data rule out.\nSaved projects retain the "
            "computed band and bootstrap replicates. A band is unavailable when the "
            "chosen family has no bootstrap or fewer than 20 usable replicates.")
        self._ci_band_check.setChecked(self._style.show_ci_band)
        self._ci_band_check.toggled.connect(self._on_style_ci_band)
        form.addRow("", self._ci_band_check)

        self._residual_check = QtWidgets.QCheckBox("Residual strip")
        self._residual_check.setToolTip(
            "A second axis under the plot showing (observed − expected) / √expected per "
            "bin, sharing the x axis, plus the reduced χ² that is the sum of their "
            "squares.\nA run of same-sign bars is the finding — KS gives one number for "
            "the whole curve and cannot say WHERE it misses.\n"
            "Export current figure… offers an explicit option to include the visible "
            "residual panel. The χ² is also written after the equation in the fit label.")
        self._residual_check.setChecked(self._style.show_residuals)
        self._residual_check.toggled.connect(self._on_style_residuals)
        form.addRow("", self._residual_check)

        self._prov_check = QtWidgets.QCheckBox("Provenance caption")
        self._prov_check.setToolTip(
            "Write what produced these events onto the figure: the detector, the ML "
            "model file and preset where there is one, and the preprocessing chain "
            "(detrend, notch, filters, normalization) that ran before it.\n"
            "Drawn as part of the plot, so Save image… keeps it — a PNG of a dwell "
            "histogram is otherwise indistinguishable between three models.")
        self._prov_check.setChecked(self._style.show_provenance)
        self._prov_check.toggled.connect(self._on_style_provenance)
        form.addRow("", self._prov_check)

        self._point_size = QtWidgets.QDoubleSpinBox()
        self._point_size.setDecimals(3)
        self._point_size.setSingleStep(.25)
        self._point_size.setRange(1, 30)
        self._point_size.setValue(self._style.point_size)
        self._point_size.setToolTip("Marker size for scatter / PCA points.")
        self._point_size.valueChanged.connect(self._on_style_point_size)
        form.addRow("Point size", self._point_size)

        self._line_width = QtWidgets.QDoubleSpinBox()
        self._line_width.setDecimals(3)
        self._line_width.setSingleStep(.25)
        self._line_width.setRange(.25, 10)
        self._line_width.setValue(self._style.line_width)
        self._line_width.setToolTip("Width of the histogram outline and the fit curve.")
        self._line_width.valueChanged.connect(self._on_style_line_width)
        form.addRow("Line width", self._line_width)

        self._bg_combo = QtWidgets.QComboBox()
        self._bg_combo.addItems(["Dark", "Light"])
        self._bg_combo.setToolTip("Light gives a white canvas with dark ink for publication.")
        self._bg_combo.currentTextChanged.connect(self._on_style_background)
        form.addRow("Background", self._bg_combo)

        self._export_w = QtWidgets.QSpinBox()
        self._export_w.setRange(0, 20000)
        self._export_w.setValue(self._style.export_w)
        self._export_w.setToolTip(
            "Saved width in px for the PNG, and in points for the PDF page "
            "(0 = native size). SVG has no fixed size and ignores both boxes.")
        self._export_w.valueChanged.connect(self._on_style_export_size)
        self._export_h = QtWidgets.QSpinBox()
        self._export_h.setRange(0, 20000)
        self._export_h.setValue(self._style.export_h)
        self._export_h.setToolTip(
            "Saved height in px for the PNG, and in points for the PDF page "
            "(0 = native / follow aspect). SVG ignores both boxes.")
        self._export_h.valueChanged.connect(self._on_style_export_size)
        # Old project/programmatic dimensions still round-trip. Interactive export
        # now presents format, physical units and resolution together in one dialog.
        self._export_w.setParent(box)
        self._export_h.setParent(box)
        self._export_w.hide()
        self._export_h.hide()
        # Keep legacy controls as the compatibility layer for saved configurations
        # and programmatic clients, but offer each basic field only once on screen.
        # Population-specific fit ornaments and legend entries stay below the common
        # controls; they do not exist on every analytical view.
        for widget in (self._title_edit, self._x_label_edit, self._y_label_edit,
                       self._legend_check, self._legend_font_size, self._font_combo,
                       self._title_size, self._axis_label_size, self._tick_size,
                       self._grid_check, self._data_swatch, self._fit_swatch,
                       self._point_size, self._line_width, self._bg_combo):
            form.setRowVisible(widget, False)
        return box

    # -- the config: one reader, one writer, one rebase rule --------------------------------------
    def config(self) -> PlotConfig:
        """EVERY setting in this window as one frozen value -- the reader.

        Safe to KEEP: nothing in the returned value aliases a widget or a mutable field,
        so a config taken before a change still describes the moment it was taken. That
        is what makes it usable as an undo entry, as a saved template and as the "before"
        half of a test.

        Widgets are the live editing surface and this is their projection; the inverse is
        :meth:`_set_config`, and the two round-trip
        (``w._set_config(c); assert w.config() == c``).
        """
        if not self._ui_ready:
            return migrate_plot_quantity(PlotConfig())
        cfg = PlotConfig(
            plot_type=self._plot_type_combo.currentText(),
            selection=SelectionConfig(
                source_keys=tuple(self.selected_source_keys()),
                accepted_only=bool(self._accepted_check.isChecked())),
            histogram=HistogramConfig(
                column=self._hist_col.currentText(),
                bins=int(self._bins.value()),
                normalization=self._norm_combo.currentText(),
                colour_by=self._hist_colour_by.currentText(),
                log_x=bool(self._hist_logx.isChecked()),
                log_y=bool(self._hist_logy.isChecked()),
                include_filter_limited=bool(self._use_filter_limited.isChecked()),
                kde_overlay=bool(self._hist_kde.isChecked())),
            scatter=ScatterConfig(
                x=self._scatter_x.currentText(), y=self._scatter_y.currentText(),
                log_x=bool(self._scatter_logx.isChecked()),
                log_y=bool(self._scatter_logy.isChecked()),
                colour_by=self._colour_by.currentText()),
            kde=KdeConfig(
                x=self._kde_x.currentText(), y=self._kde_y.currentText(),
                log_x=bool(self._kde_logx.isChecked()),
                log_y=bool(self._kde_logy.isChecked()),
                levels=int(self._kde_levels.value())),
            heatmap=HeatmapConfig(
                x=self._heatmap_x.currentText(), y=self._heatmap_y.currentText(),
                log_x=bool(self._heatmap_logx.isChecked()),
                log_y=bool(self._heatmap_logy.isChecked()),
                bins=int(self._heatmap_bins.value())),
            cluster=ClusterConfig(
                features=tuple(self.feature_selection()),
                method=self._method_combo.currentText(),
                k_mode=self._k_mode.currentText(),
                k=int(self._k_spin.value())),
            fit=FitConfig(
                kind=self._fit_combo.currentText(),
                k=int(self._fit_k.value()),
                # The same "enabled AND checked" reading `_commit_fit_selection` uses: a
                # ticked box on a family that has no components is not a request for auto-k.
                k_auto=bool(self._fit_k_auto.isEnabled() and self._fit_k_auto.isChecked()),
                scatter_kind=self._scatter_fit.currentText(),
                applied=self._applied_fit, applied_k=int(self._applied_fit_k),
                applied_k_auto=bool(self._applied_fit_k_auto),
                applied_scatter=self._applied_scatter_fit,
                range_lo=self._fit_lo.text(), range_hi=self._fit_hi.text(),
                range_unit=self._fit_range_unit),
            axes=AxesConfig(
                time_unit=self._time_unit.currentText(),
                x_min=self._range_edits["xmin"].text(),
                x_max=self._range_edits["xmax"].text(),
                y_min=self._range_edits["ymin"].text(),
                y_max=self._range_edits["ymax"].text(),
                manual_x=self._manual_x, manual_y=self._manual_y),
            style=self._style,
            norm_user_set=bool(self._norm_user_set))
        from ..core.figure_quantity_basis import quantity_witness
        return _dc_replace(cfg, axes=_dc_replace(cfg.axes, quantity_display=_deep_frozen(
            quantity_witness(_plot_quantity_bases(cfg), original=self._quantity_display))))

    def _set_config(self, cfg: PlotConfig, *, draw: bool = True,
                    own: bool = False) -> None:
        """THE ONE WRITER: replace the whole window state with ``cfg``.

        ``own`` SAYS WHERE ``cfg`` CAME FROM, and that is the one fact the two import
        disclosures hang on. ``own=True`` means "this is the window's own state, put
        back or corrected" -- an undo/redo entry (:meth:`_install_history`) or a rebase
        (:meth:`_rebase`) -- and such an install opens no new origin and touches neither
        :class:`_NormRefusal` nor :class:`_ConfigLosses`: a sentence about the file
        whose state is being re-installed is still exactly true. ``own=False`` (the
        default, and what every outside door gets: an import, a template recall, a
        script's ``apply_config``) opens a new origin on the history, which is what hides
        a disclosure belonging to an earlier install; an import additionally re-decides
        both records under the origin it opened. The measurements that forced this
        split are on the two record classes.

        FULLY REPLACED, NOT PATCHED (owner item A1). Every control is written, including
        the ones the caller did not think about -- which is the point: a partial write is
        how a title, a pin or a fit window from the previous analysis survived into the
        next one. Callers build the config they want with :func:`dataclasses.replace` and
        hand the whole thing over.

        ONE REDRAW AT MOST. The ~40 widget writes below each emit a change signal, so the
        whole pass runs under ``_updating`` -- which every handler honours -- and the draw
        happens once at the end, or not at all when the caller owns it (``draw=False``,
        used by :meth:`_rebase`, where the handler that triggered the rebase is about to
        decide between drawing and marking the plot stale).

        ``_identity`` is updated LAST and is the anchor the next rebase measures from: it
        records which analysis the typed text and pins in this config were tuned for.
        """
        if not self._ui_ready:
            return
        cfg = migrate_plot_quantity(_dc_replace(cfg, style=_project_style_settings(cfg.style)))
        importing = self._incoming_config_file is not None
        # WHICH INSTALL THE RESULTING STATE DESCENDS FROM. Opened before the widgets are
        # written so the draw at the foot of this method -- the one the user is looking
        # at when they decide whether to save -- already reads the new origin.
        origin = None if own else self._history.open_origin()
        was, self._updating = self._updating, True
        self._installing = True
        try:
            # PLOT TYPE FIRST. It selects which page's columns the identity reads, so
            # every clearing decision downstream depends on it already being current.
            if cfg.plot_type in _PLOT_TYPES:
                self._plot_type_combo.setCurrentText(cfg.plot_type)
                self._pages.setCurrentIndex(_PLOT_TYPES.index(cfg.plot_type))
            self._time_unit.setCurrentText(cfg.axes.time_unit)
            self._quantity_display = cfg.axes.quantity_display

            self._accepted_check.setChecked(cfg.selection.accepted_only)
            self._apply_checked_keys(cfg.selection.source_keys)

            h = cfg.histogram
            # An empty column name means "no column on offer" and CANNOT be written: Qt
            # ignores `setCurrentText` for a value the combo does not carry, so writing it
            # would silently leave the old one selected. The pickers are refilled from the
            # data by `_repopulate_columns`; this only chooses among what they hold.
            if h.column:
                self._hist_col.setCurrentText(h.column)
            self._bins.setValue(int(h.bins))
            # THE Y QUANTITY IS A NAME THIS WRITER CAN BE HANDED AND NOT BE ABLE TO
            # WRITE, and the branch below is the difference between saying so and not.
            # `setCurrentText` on a non-editable combo is a SILENT no-op for text the
            # combo has not got, so the bare call this replaces drew whatever was already
            # on screen under a request for something else: measured on the owner's
            # 982-event file with a template naming `density (1/s)`, the bins came across
            # (47), the y axis read `count`, Details read `y = count`,
            # `import_config_json` returned True, and 0 dialogs and 0 new labels appeared.
            #
            # RECORDED HERE BECAUSE THIS IS THE ONE WRITER. Every door -- a template, an
            # imported sidecar, an undo entry, a script's own `apply_config` -- ends up in
            # this method, so a refusal recorded here cannot be reached around; a fix
            # bolted onto `_installable` would have missed `apply_config`, which is the
            # public one. `_NormRefusal` holds what happens to the record afterwards.
            #
            # THE COMBO IS LEFT ALONE RATHER THAN FORCED. There is nothing honest to
            # write: the window can only show a mode it carries, and inventing a
            # substitute *here* is what deleted the tell (see `_installable`). The name
            # stays refused, `config()` goes on reporting what is actually drawn, and the
            # window says so in Details on every draw.
            #
            # ONLY THE NORMALIZATION, NOT THE COLUMNS. A column this window cannot select
            # is a fact about the RECORDING ("these events have no `ecd_pA_s`") and is
            # resolved for the caller, deliberately and in advance, by `_installable`. The
            # y quantity is a fact about the BUILD: this combo's three items are compiled
            # in, so a name that misses them can only have come from a newer build or a
            # hand-edit, and no recording can explain it away.
            #
            # RE-DECIDED ONLY BY AN INSTALL FROM OUTSIDE THE WINDOW'S OWN LINEAGE. A
            # history entry or a rebase (`own`) always names the mode the window DREW --
            # `config()` reads the combo -- so "re-decide it from the config" there is
            # what erased a standing refusal on the very undo that put the refused
            # install's state back (S7-FIX1) and on a Time-unit change (S7-SK1). An
            # install that CAN be honoured records nothing under the origin it opened,
            # which is all "re-deciding" is now (A1-N4): the record is per origin, so
            # an earlier import's refusal is neither cleared nor overwritten here -- it
            # stays under ITS origin, hidden while this one is on screen and spoken
            # again when an undo walks back onto the state that import produced.
            if self._norm_combo.findText(h.normalization) >= 0:
                self._norm_combo.setCurrentText(h.normalization)
            elif origin is not None:
                self._norm_refusal.record(h.normalization, origin)
            self._hist_colour_by.setCurrentText(h.colour_by)
            self._hist_logx.setChecked(bool(h.log_x))
            self._hist_logy.setChecked(bool(h.log_y))
            self._use_filter_limited.setChecked(bool(h.include_filter_limited))
            self._hist_kde.setChecked(bool(h.kde_overlay))

            s = cfg.scatter
            if s.x:
                self._scatter_x.setCurrentText(s.x)
            if s.y:
                self._scatter_y.setCurrentText(s.y)
            self._scatter_logx.setChecked(bool(s.log_x))
            self._scatter_logy.setChecked(bool(s.log_y))
            self._colour_by.setCurrentText(s.colour_by)

            k = cfg.kde
            if k.x:
                self._kde_x.setCurrentText(k.x)
            if k.y:
                self._kde_y.setCurrentText(k.y)
            self._kde_logx.setChecked(bool(k.log_x))
            self._kde_logy.setChecked(bool(k.log_y))
            self._kde_levels.setValue(int(k.levels))

            hm = cfg.heatmap
            if hm.x:
                self._heatmap_x.setCurrentText(hm.x)
            if hm.y:
                self._heatmap_y.setCurrentText(hm.y)
            self._heatmap_logx.setChecked(bool(hm.log_x))
            self._heatmap_logy.setChecked(bool(hm.log_y))
            self._heatmap_bins.setValue(int(hm.bins))

            c = cfg.cluster
            self._method_combo.setCurrentText(c.method)
            self._k_mode.setCurrentText(c.k_mode)
            self._k_spin.setValue(int(c.k))
            self._apply_feature_selection(c.features)

            f = cfg.fit
            self._fit_combo.setCurrentText(f.kind)
            self._fit_k.setValue(int(f.k))
            self._fit_k_auto.setChecked(bool(f.k_auto))
            self._scatter_fit.setCurrentText(f.scatter_kind)
            self._applied_fit = f.applied
            self._applied_fit_k = int(f.applied_k)
            self._applied_fit_k_auto = bool(f.applied_k_auto)
            self._applied_scatter_fit = f.applied_scatter
            self._fit_lo.setText(f.range_lo)
            self._fit_hi.setText(f.range_hi)
            self._fit_range_unit = f.range_unit

            a = cfg.axes
            self._range_edits["xmin"].setText(a.x_min)
            self._range_edits["xmax"].setText(a.x_max)
            self._range_edits["ymin"].setText(a.y_min)
            self._range_edits["ymax"].setText(a.y_max)
            self._manual_x, self._manual_y = a.manual_x, a.manual_y

            # The style widgets each write onto `_style` from their own handler, so the
            # config's style is installed AFTER them and is the value that stands --
            # exactly the ordering `reset_state` has always used.
            style = cfg.style
            if style.heatmap_colormap not in _HEATMAP_COLORMAPS:
                # A COLOUR MAP THIS BUILD HAS NOT GOT (a newer build's file, a hand-edit
                # naming `jet`) is replaced by the default HERE, in the one writer, rather
                # than carried: the image could only ever be drawn in viridis, so a style
                # that went on naming `jet` would re-export a map nobody saw, and
                # `_ConfigLosses` -- which compares the file with `config()` -- would
                # have no difference to name. Written, it is disclosed like any other
                # setting the window did not take.
                style = _dc_replace(style,
                                    heatmap_colormap=StyleSettings().heatmap_colormap)
            self._apply_style_widgets(style)
            self._style = style
            # THE SOURCE ROWS ARE PART OF THE CONFIG TOO, and this call is the only
            # reason they behave like it. `label_overrides` is edited THROUGH the
            # Sources list -- the row is the rename affordance (`ItemIsEditable`, see
            # the row build in `set_data` and the hint in `_build_sources_group`) -- so
            # a row's TEXT is a projection of the style exactly as the Title box is, and
            # the one writer has to write it back. It did not, and the window then said
            # two different things about one source: MEASURED, an undo of a rename put
            # `label_overrides` back to {} and put both the legend checklist and the
            # canvas legend back to the filename while the row kept the typed name, and
            # a re-Plot did not clear it -- only a Refresh did, because `set_data`
            # REBUILDS the rows from `_legend_label`. A change-then-undo diff over 23
            # controls found this was the only widget in the window left unsynced.
            #
            # AFTER `self._style`, not before: `_legend_label` reads `self._style`, so
            # syncing one line earlier would write the OUTGOING names onto the rows.
            self._sync_source_row_labels()
            self._norm_user_set = bool(cfg.norm_user_set)

            self._identity = cfg.identity
            if cfg.plot_type == PCA:
                # `set_checked` ignores a feature the list does not offer (a template or an
                # old undo entry from another table), so the anchor takes the set from the
                # list: the stale-anchor hazard `_installable` closes for the columns.
                self._identity = _dc_replace(
                    self._identity, features=tuple(sorted(set(self.feature_selection()))))
            # Kept in step for any reader of the old memo; the identity is the authority.
            self._range_factors = self._identity.axis_factors
        finally:
            self._updating = was
            self._installing = False

        # WHAT A FILE ASKED FOR AND THIS WRITE DID NOT TAKE -- the other six kinds of
        # loss beside the y quantity `_NormRefusal` already holds. Here for the same
        # reason that one is here: this is the ONE writer, so every door passes through
        # it. Taken under the origin this import opened; every other install leaves the
        # record alone and lets the origin decide whether it is spoken (a template or a
        # script's `apply_config` opened a new one, so it is not; an undo, a redo or a
        # rebase opened none, so it is -- see :class:`_ConfigLosses` for why "clear it
        # on every other install" was the wrong rule). Read here rather than after the
        # draw so the value is the one this write produced: the automatic density
        # default can move the combo during the draw, and a swing nobody asked for is
        # not something the file lost.
        if importing and origin is not None:
            self._config_losses.take(self._incoming_config_file, self.config(), origin)

        self._sync_range_unit_labels()
        # The writer moved the four range boxes AND possibly the log toggles, so the
        # amber line under them is re-derived here rather than only on an Apply click:
        # an undo that restores a reversed pair must restore its explanation with it.
        self._sync_range_notes()
        self._sync_fit_apply_buttons()   # ...which ends by syncing the run state too
        self._sync_legend_reset_button()
        self._sync_legend_entry_list()
        if draw:
            self._armed_redraw()

    def apply_config(self, cfg: PlotConfig) -> None:
        """Public: install ``cfg`` and draw once. The entry point undo/redo and saved
        plot templates use -- naming a whole configuration IS the run gesture, exactly as
        ``set_plot_type`` and ``set_fit`` are."""
        self._set_config(cfg, draw=True)

    def capture_history_capsule(self):
        """Commit typing and preserve the full configuration/origin history as JSON values."""
        from nanoscope.core.figure_history_state import HistoryCapsule

        self._history_commit()
        history = self._history
        if history._pending_origin is not None or history.base != self.config():
            raise RuntimeError("Cannot checkpoint an uncommitted plot configuration")
        return HistoryCapsule(
            "nanoscope.plot-config-history/1", plot_config_to_json(history.base),
            tuple(plot_config_to_json(cfg) for cfg in history._past),
            tuple(plot_config_to_json(cfg) for cfg in history._future), history.dropped,
            extra={"depth": history._depth, "origin": history._origin,
                   "past_origins": history._past_origins,
                   "future_origins": history._future_origins, "opened": history._opened,
                   "norm_refusal_records": [[key, value] for key, value
                                            in self._norm_refusal._records.items()],
                   "config_loss_records": [[key, [value[0], list(value[1])]] for key, value
                                           in self._config_losses._records.items()]})

    def restore_history_capsule(self, capsule) -> None:
        """Install session history only after the same frozen configuration was restored."""
        from nanoscope.core.figure_history_state import HistoryCapsule
        from nanoscope.core.figure_spec import FigureValidationError, thaw_json

        if not isinstance(capsule, HistoryCapsule):
            capsule = HistoryCapsule.from_dict(capsule)
        if capsule.codec != "nanoscope.plot-config-history/1":
            raise FigureValidationError("Incompatible plot configuration history codec")
        extra = thaw_json(capsule.extra)
        depth = extra.get("depth")
        if type(depth) is not int or not 1 <= depth <= _HISTORY_DEPTH:
            raise FigureValidationError("Invalid plot history depth")
        if len(capsule.past) + len(capsule.future) > depth:
            raise FigureValidationError("Plot history exceeds its depth bound")

        def decode(value):
            raw = thaw_json(value)
            config = plot_config_from_json(raw)
            expected = dict(raw)
            expected["version"] = PLOT_CONFIG_VERSION
            expected["axes"] = {"quantity_display": {}, **dict(expected.get("axes", {}))}
            if plot_config_to_json(config) != expected:
                raise FigureValidationError("Plot history cannot be restored losslessly")
            return migrate_plot_quantity(config)

        base = decode(capsule.base)
        past = [decode(value) for value in capsule.past]
        future = [decode(value) for value in capsule.future]
        if base != self.config():
            raise FigureValidationError("Plot history does not match the restored figure")
        opened, origin = extra.get("opened"), extra.get("origin")
        if type(opened) is not int or opened < 0:
            raise FigureValidationError("Invalid plot history origin counter")

        def valid_origin(value):
            return type(value) is int and 0 <= value <= opened

        past_origins, future_origins = extra.get("past_origins"), extra.get("future_origins")
        if (not valid_origin(origin) or not isinstance(past_origins, list)
                or not isinstance(future_origins, list)
                or len(past_origins) != len(past) or len(future_origins) != len(future)
                or not all(valid_origin(value) for value in past_origins + future_origins)):
            raise FigureValidationError("Plot history origin arrays do not match its states")

        def records(name, losses=False):
            values = extra.get(name)
            if not isinstance(values, list) or len(values) > _ORIGINS_KEPT:
                raise FigureValidationError("Invalid plot import disclosure records")
            out = {}
            for row in values:
                if not isinstance(row, list) or len(row) != 2 or not valid_origin(row[0]):
                    raise FigureValidationError("Invalid plot import disclosure origin")
                key, value = row
                if key in out:
                    raise FigureValidationError("Duplicate plot import disclosure origin")
                if losses:
                    if (not isinstance(value, list) or len(value) != 2
                            or not isinstance(value[0], str) or not isinstance(value[1], list)
                            or not all(isinstance(name, str) for name in value[1])):
                        raise FigureValidationError("Invalid plot import loss disclosure")
                    value = (value[0], tuple(value[1]))
                elif not isinstance(value, str):
                    raise FigureValidationError("Invalid plot normalization disclosure")
                out[key] = value
            return out

        refusals = records("norm_refusal_records")
        losses = records("config_loss_records", losses=True)
        history = self._history
        self._history_timer.stop()
        self._format_text_run_key = None
        history._base, history._past, history._future = base, past, future
        history._depth, history.dropped = depth, capsule.dropped
        history._origin, history._opened = origin, opened
        history._past_origins, history._future_origins = past_origins, future_origins
        history._pending_origin = None
        self._norm_refusal._records = refusals
        self._config_losses._records = losses
        self._sync_config_buttons()
        self._sync_template_selection()

    def adopt_restored_state(self) -> None:
        """A saved project has been restored into this window: its history starts here.

        The rule :meth:`set_data` and :meth:`reset_state` follow ("a reconstruction never
        becomes an offer to undo the project you just opened"). The restore writes the saved
        configuration with ``_set_config(own=True)``, which moves neither the stack nor its
        base, so without this the stack still began at the blank window: the first change
        after opening was undone to the built-in defaults, and the typed title and look the
        project was saved with did not come back.
        """
        self._history_timer.stop()
        self._history.reset(self.config())
        self._sync_config_buttons()
        self._sync_template_selection()

    @staticmethod
    def _rebased(cfg: PlotConfig, was: AnalysisIdentity, y_moved: bool = False) -> PlotConfig:
        """``cfg``, corrected for having moved off analysis identity ``was``.

        A pure function and the ONLY statement of what a change to the analysis
        invalidates; see :meth:`PlotConfig.rebased` for the four clauses and the measured
        bug behind each. Exposed here because this is where its callers live.
        """
        return cfg.rebased(was, y_moved=y_moved)

    def _rebase(self, *, y_moved: bool = False) -> None:
        """Apply :meth:`_rebased` to the live window: the ONE call every identity-moving
        handler makes. ``y_moved`` is the histogram's Normalize, which the identity does not
        carry (:meth:`PlotConfig.rebased`): :meth:`_on_norm_moved` says it for a pick.

        The handler has already let the widget move -- Qt has no "about to change" -- so
        the previous identity comes from ``_identity``, the anchor the last
        :meth:`_set_config` left behind. That anchor is what makes this work for changes
        NOBODY typed: :meth:`_repopulate_columns` can move a column picker on its own when
        a refreshed source no longer carries the old column, and the pin, the fit window
        and the caption are cleared by the same rule that handles a mouse click.

        A BULK WIDGET UPDATE KEEPS ITS CAPTION, and that is the one asymmetry in the
        whole rule. ``run_pca`` moves the plot-type combo itself, under ``_updating``, to
        show the analysis the user just pressed Run on -- a title they typed for THAT
        plot is not stale, it is the caption of the picture about to appear. The window
        still owns the quantities: a pin, a typed limit and a fit window can be wrong in
        a way no caller could plausibly intend, so those are rebased either way. Said
        here once rather than at each such caller, which is how the eight bugs happened.

        No draw: every caller either marks the plot stale or draws for itself.
        """
        if not self._ui_ready or self._installing:
            return
        cfg = self.config()
        acted = y_moved or cfg.identity.wording_stale_from(self._identity)   # the words had a say
        rebased = self._rebased(cfg, self._identity, y_moved)
        if self._updating:
            rebased = _dc_replace(rebased, style=_dc_replace(
                rebased.style, title=cfg.style.title,
                x_label=cfg.style.x_label, y_label=cfg.style.y_label))
        if rebased == cfg:
            self._identity = cfg.identity   # nothing to clear, but the anchor still moved
            self._range_factors = self._identity.axis_factors
            self._sync_range_unit_labels()
            self._announce_reset((), cfg.identity, acted)
            return
        # THE WINDOW'S OWN STATE, CORRECTED -- not an install from outside, so no new
        # origin and no re-decision of the import disclosures (S7-SK1: the Time-unit
        # combo reaches here, and the losses sentence must survive it).
        self._set_config(rebased, draw=False, own=True)
        went = tuple(name for name in cfg.axes.ranges() if name not in rebased.axes.ranges())
        self._announce_reset(words_cleared(cfg.style, rebased.style) + went, rebased.identity,
                             acted, cfg.histogram.normalization if y_moved else "")

    def _announce_reset(self, cleared: tuple[str, ...], ident: AnalysisIdentity,
                        acted: bool, norm: str = "") -> None:
        """Say which of the typed words and ranges went (quantity_reset). The picture follows
        the control only at Plot, so it says what the plot is set to show (``norm``: the
        Normalize the histogram was just set to, when that is what moved). When the rule acted
        and nothing was typed, the sentence an earlier change left on the bar goes instead."""
        said = ""
        if cleared:
            shows = _SHOWS_PLOT[ident.plot_type].format(x=ident.x_column, y=ident.y_column)
            if norm:
                shows += f" as {_NORM_PHRASE.get(norm, norm)}"
            if ident.fit_model:
                shows += f", fitted with {ident.fit_model.partition(':')[0]}"
            said = reset_sentence(cleared, shows, lead="the plot is now set to show")
        if acted:
            self._reset_said = announce(self.statusBar(), said, self._reset_said)
            self._reset_took = tuple(cleared) if said else ()

    def _lapse_reset_sentence(self) -> None:
        """Take the sentence of the last reset off the status bar once what it names is typed or
        pinned again (the Undo of the reset, a title typed since, a range pinned again). A change
        of look, and Reset figure with its Undo, leave it: nothing it names is typed then. Only
        if the bar still shows it, as :func:`~nanoscope.gui.quantity_reset.announce` has it."""
        if not self._reset_took:
            return
        cfg = self.config()
        if stands(self._reset_took, typed_words(cfg.style) + cfg.axes.ranges()):
            return
        self._reset_took = ()
        self._reset_said = announce(self.statusBar(), "", self._reset_said)

    def _apply_checked_keys(self, keys: Sequence[str]) -> None:
        """Tick exactly ``keys`` in the source list. Silent about keys that are not on
        offer: a config may name sources this session has not loaded (a template, an undo
        entry taken before a file was closed), and the right answer is to restore what CAN
        be restored rather than to refuse the whole config."""
        wanted = {str(k) for k in keys}
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted
                               else Qt.Unchecked)

    def _sync_source_row_labels(self) -> None:
        """Write each source row's TEXT from the style -- in place, destroying nothing.

        The other half of :meth:`_apply_checked_keys`. That one restores WHICH rows are
        ticked; this one restores what they are CALLED. Both exist because this list is
        two controls in one: a checklist, and -- since owner note #15, "the legend is not
        editable" -- the legend's edit affordance. A config restore that wrote only the
        check states left a typed name standing over a reverted legend.

        WHY NOTHING IS TAKEN, CLEARED OR RE-CREATED, and why that is a crash rule rather
        than a style preference. This method can run from INSIDE Qt's delivery of this
        very list's ``itemChanged``: a tick reaches :meth:`_on_source_item_changed` ->
        :meth:`_on_sources_changed` -> :meth:`_repopulate_columns` -> :meth:`_rebase` ->
        :meth:`_set_config` -> here, while Qt still holds the ticked row as that
        emission's argument and re-marshals it for every receiver it has not reached yet.
        A row FREED under that is the EXC_BAD_ACCESS the shipped build died on (see
        :meth:`_sync_legend_entry_list` and :meth:`_retire_row`). ``setText`` on an
        EXISTING row frees nothing, so the crashing shape is never built here -- no
        ``clear()``, no ``takeItem``, no new ``QListWidgetItem``, and consequently no
        parking in :meth:`_retire_row` to undo it with.

        ONLY WHAT DIFFERS IS WRITTEN, for :meth:`_write_legend_row`'s reason: every
        ``setText`` goes out as a ``dataChanged`` and re-emits ``itemChanged``, and the
        common case -- a restore that touches no name -- should cost a string comparison
        and emit nothing at all. The re-emission a genuine change does cause is harmless:
        this runs under :meth:`_set_config`'s ``_updating``, which is the first thing
        :meth:`_on_source_item_changed` tests, so the rename detector cannot read the
        window's own write back as a user gesture and turn it into a history step.

        The TOOLTIP is deliberately left alone: :meth:`_source_tooltip` names the
        source's REAL label, which a rename does not change -- that is what tells a user
        who retyped a row which recording it still is.

        A row whose key is no longer a source is SKIPPED, not blanked: ``reset_state``
        clears ``_sources`` before writing the blank config, and a row with no source
        behind it has no real label to fall back to.
        """
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            src = self._sources.get(item.data(Qt.UserRole) or "")
            if src is None:
                continue
            text = self._legend_label(src)
            if item.text() != text:
                item.setText(text)

    def _apply_feature_selection(self, cols: Sequence[str]) -> None:
        """Select exactly ``cols`` in the feature list (those it currently offers)."""
        self._features.set_checked({str(c) for c in cols})

    def _apply_style_widgets(self, style: StyleSettings) -> None:
        """Write a style onto the Style group's controls.

        Split out of :meth:`_set_config` because it is the half `reset_state` needs too,
        and because the Font row is a genuine exception worth stating once: its handler
        writes the value the CURRENT ROW stores, and the default "" ("no family chosen",
        drawn in Arial) is a row of its own rather than a family -- so the row is moved by
        :meth:`_select_font_family`, which says nothing, and the style installed by the
        caller stands.
        """
        from .plot_appearance import set_projected_spin_value
        self._title_edit.setText(style.title)
        self._x_label_edit.setText(style.x_label)
        self._y_label_edit.setText(style.y_label)
        self._legend_check.setChecked(bool(style.show_legend))
        set_projected_spin_value(self._legend_font_size, float(style.legend_font_size_pt))
        self._legend_counts.setChecked(bool(style.legend_counts))
        self._ci_band_check.setChecked(bool(style.show_ci_band))
        self._residual_check.setChecked(bool(style.show_residuals))
        self._prov_check.setChecked(bool(style.show_provenance))
        set_projected_spin_value(self._font_size, float(style.font_size_pt))
        set_projected_spin_value(self._title_size, float(style.title_font_size_pt))
        set_projected_spin_value(self._axis_label_size, float(style.axis_label_font_size_pt))
        set_projected_spin_value(self._tick_size, float(style.tick_font_size_pt))
        self._grid_check.setChecked(bool(style.show_grid))
        set_projected_spin_value(self._point_size, float(style.point_size))
        set_projected_spin_value(self._line_width, float(style.line_width))
        self._bg_combo.setCurrentText("Light" if style.background == "light" else "Dark")
        self._export_w.setValue(int(style.export_w))
        self._export_h.setValue(int(style.export_h))
        self._select_font_family(style.font_family)
        self._paint_swatch(self._data_swatch, style.data_color)
        self._paint_swatch(self._fit_swatch, style.fit_color)
        # `_set_config` has already replaced a name this build has not got by the
        # default, so this finds it; the guard only keeps a stray caller from writing a
        # name the combo would ignore (`_heatmap_colormap` falls back to viridis too).
        if self._heatmap_cmap.findText(style.heatmap_colormap) >= 0:
            self._heatmap_cmap.setCurrentText(style.heatmap_colormap)
        # The shared Appearance widget is part of this group. It is told the style too, so
        # after an install (a project, a template, an import) its next edit builds on what
        # is on the figure -- a label's numbers and a title's bold are its to keep.
        if hasattr(self, "_appearance_widget"):
            self._appearance_widget.set_state(self._appearance_of(style))

    def _select_font_family(self, family: str) -> None:
        """Point the Font row at ``family`` (``""`` = the default). Never commits: the picker's
        ``set_value`` says nothing, whatever the caller did to the row's signals, so a sync
        (an undo, an import, a reset) cannot write a family the user did not choose.

        A family the list does not have -- a config from a colleague's machine naming
        ``Calibri`` -- gets one extra row carrying the exact string, marked as absent here, and
        the style keeps it as written; ``courier new`` shows the Courier New row and the
        style keeps its own spelling."""
        self._font_combo.set_value(family)

    def _prune_legend_memory(self) -> None:
        """Drop legend decisions about sources and cluster rows that no longer exist.

        Called after every draw (the draw is what settles which entries were OFFERED) and
        after a source-set change. See :meth:`PlotConfig.with_pruned_legend` for why the
        two kinds of entry are pruned against two different things.
        """
        if not self._ui_ready:
            return   # `config()` answers with DEFAULTS before the UI exists, and writing
                     # that default style back would be a silent style reset, not a prune.
        pruned = self.config().with_pruned_legend(self._sources.keys(),
                                                  self._legend_offered.keys())
        self._style = pruned.style

    # ============================================================================================
    # THE CONFIGURATION GROUP: undo/redo (A2), named templates (A3), JSON sidecars (B3)
    # ============================================================================================
    #
    # All three are the same feature seen from three distances, and all three exist only
    # because `config()` returns a VALUE: undo is a stack of them, a template is one of
    # them under a name, an export is one of them in a file. Nothing below reaches for a
    # widget -- every one of them goes through `config()` to read and `_set_config` (via
    # `apply_config`) to write, which is what keeps the identity anchor honest.

    def _build_config_group(self) -> QtWidgets.QGroupBox:
        """Undo/Redo, the template combo and the two JSON buttons, in one group.

        Kept together, at the foot of the column, because they are the three things you
        do to a whole configuration rather than to one control -- and because the
        template combo needs the sentence beneath it (what a template does and does not
        carry) to be on screen, not in a tooltip nobody opens.
        """
        box = QtWidgets.QGroupBox("Configuration")
        lay = QtWidgets.QVBoxLayout(box)

        # -- A2: undo / redo -------------------------------------------------------------
        undo_row = QtWidgets.QHBoxLayout()
        self._undo_btn = QtWidgets.QPushButton("Undo")
        self._undo_btn.clicked.connect(self.undo_config)
        # NO LINE OF THIS MODULE ASKS FOR A SCROLL OF THE CONTROLS COLUMN. That is the
        # claim, and it is the whole claim -- the column is Qt's to move, and the two
        # ways it still does are written down further down this block. The ceiling
        # disclosure used to be part scroll: a refused undo called `ensureWidgetVisible`
        # on the sentence at the foot of this column. It produced a new major in three
        # consecutive rounds and it is gone on every trigger -- the measurement, driven
        # end to end, is at `undo_config`, and the property that replaces it is driven in
        # `test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`: nine gestures
        # (the ceiling crossed, a refused Ctrl+Z, a refused click here, a right-click,
        # Space/Return/Enter aimed at the dead button, a template recall, a plot-type
        # change, thirty ordinary edits and an undo that succeeds) each leave the
        # scrollbar on the single value the user parked it on.
        #
        # "AT THE FOCUSED DEAD BUTTON" IS NOT WHAT THAT SIXTH GESTURE IS, and the comment
        # said so for two rounds. Driven on this build at the undo floor, this button will
        # not take focus once it is disabled: `setFocus()` on it leaves
        # `QApplication.focusWidget()` on the QScrollArea. The keys therefore go to
        # whatever DID take focus, which is the state the test drives and says out loud.
        #
        # WHAT THE MOUSE USER AT THE FLOOR GETS IS THIS BUTTON'S OWN TOOLTIP, on the
        # button their pointer is already resting on, for no movement at all: the text
        # `_sync_config_buttons` writes, which at the floor reads "Nothing left to undo --
        # this plot's oldest history was dropped at the 200-step ceiling (N steps lost)."
        # and while there is still history to step back carries the same clause after the
        # count ("(200 to step back). Its oldest history was dropped at the 200-step
        # ceiling (10 steps lost)."). Qt renders a tooltip for a DISABLED widget: measured
        # on this real dead button both by sending the `QHelpEvent` and by a REAL hover at
        # the platform seam, which delivers exactly one ToolTip event to it and shows the
        # text. The standing amber line two rows below says the same thing with no gesture
        # at all, for the user who scrolls to the controls it is about.
        #
        # A DISABLED QPushButton EMITS NO `clicked` (measured: real clicks at the window
        # handle over this button produce 0 signals and leave the stack where it is), so
        # at the floor the refused undo never reaches `undo_config` from the mouse. That
        # is a fact about this button while it is dead, and it says nothing about the
        # ENABLED button, which is where the damage was. Qt moves a scroll area on its
        # own when focus leaves a widget that has just been disabled, so one click on the
        # ENABLED button that empties the stack used to scroll the column out from under
        # a pointer that had not moved, because `QScrollArea.focusNextPrevChild` follows
        # focus to the next widget in the tab chain and the amber line below adds its
        # 51 px to the trip. Measured at 1120x720 on a 78-cell grid (parking x pointer
        # row, over the 60 px band below the fold, a fresh typed run per cell): with the
        # line hidden 36 of 78 cells moved, by 55..80 px, and 24 left a control the user
        # never aimed at under the pointer; with the line standing at the undo floor
        # 78 of 78 moved, by 71..131 px, and 56 escalated -- 30 of them so that the next
        # click at the unmoved point opened a modal "Save plot template" and 26 the
        # template dropdown. THAT IS CLOSED: `_retire_history_button` hands focus to the
        # partner button on the same row before either of these two is retired. The same
        # grid now reads 0 of 78 moved and 0 of 78 escalated -- at 1120x720 with the line
        # standing AND with it hidden, so the line's presence costs nothing, and at the
        # undo floor at 775x450, 900x600, 1120x720 and 1512x900. The same shape was on
        # Redo -- clicking it until the redo stack empties, 1658 -> 1789 at the fold --
        # and the hand-off is symmetric, so it is closed there too.
        #
        # WHAT IS LEFT IS NOT A SCROLL THIS MODULE ASKS FOR -- grep the code past its
        # comments: zero `ensureWidgetVisible` -- and there are exactly two things, both
        # written up where they happen. QScrollBar's own clamp moves the VALUE when a
        # taller window shortens the range past where the user parked it (`_build_ui`,
        # pinned in `test_a_resize_that_shortens_the_range_clamps_the_column`). The
        # ceiling line's appearance moves the ROWS 51 px without moving the value at all
        # (`_history_note`, below, with the census of what lands under the pointer).
        # Neither is a call, and `test_no_code_in_this_window_moves_the_controls_column`
        # is the static half that keeps it that way.
        self._redo_btn = QtWidgets.QPushButton("Redo")
        self._redo_btn.clicked.connect(self.redo_config)
        undo_row.addWidget(self._undo_btn)
        undo_row.addWidget(self._redo_btn)
        lay.addLayout(undo_row)
        # WHERE THE 200-STEP CEILING SPEAKS. `_ConfigHistory` has always counted what it
        # discarded -- "a silent bound is indistinguishable from a broken Undo button"
        # is its own comment -- and nothing read the count, so the bound was precisely
        # what that comment says a bound must never be. Measured on the owner's -700 mV
        # table: 210 control changes leave depth() == (200, 0) and dropped == 10 with
        # the tooltip still saying "(200 to step back)".
        #
        # THIS IS THE AXIS-RANGE GROUP'S DEVICE (`_range_note`), reused deliberately
        # rather than re-invented: an amber line, hidden until it has something to say,
        # placed ON the group that did the thing. A window with two dialects for one
        # idea is the thing `_TPL_MANUAL` already refuses.
        #
        # NOT DETAILS, for a reason particular to this window: `details_text()` is the
        # written record of what the window said about the DATA and `_persist_results`
        # stores it into the session under the clustering/PCA results, where a note
        # about this session's EDITING has no business -- and Details is rewritten by
        # every draw, so the line would blink in and out with the picture.
        #
        # NOT A STATUS BAR: this QMainWindow has none (measured: no QStatusBar child),
        # and adding one to carry a sentence seen once a day is chrome that is empty the
        # rest of the time. NOT A DIALOG: nothing was lost that the user can act on, and
        # a modal for it would be the second-worst answer after silence.
        #
        # AND THE SPACE IS NOT RESERVED, deliberately, after measuring what reserving it
        # would cost. Appearing costs a ONE-TIME 51 px shift of what is below this label
        # -- measured at 1120x720: template combo 2382 -> 2433, Save image 2569 -> 2620,
        # scroll range 1917 -> 1968 -- while the Undo and Redo buttons, the two things
        # the user of this group is actually looking at, move 0 px, and the width does
        # not move at all (controls minimumSizeHint 331, window minimumWidth 775, before
        # and after).
        #
        # Reserving it would have to reserve a WRONG number: this label's height is its
        # own text's `heightForWidth`, and the column is user-resizable through the
        # splitter -- measured, 45 px while the column is 291-315 px wide (every window
        # width from 775 to 1920, since the column stops growing) and 30 px once the
        # splitter is dragged past ~360 px. So a static reservation is either 45 px of
        # permanent blank inside the Configuration group that still shifts by 15 px on a
        # wide column, or a `resizeEvent` hook recomputing a hole. A permanent gap under
        # the Undo row, on every window, for a sentence most sessions never see, is a
        # worse thing to look at than one 51 px settle in the session that earned it.
        #
        # WHAT THAT SETTLE COSTS, DRIVEN AS A POINTER EVENT rather than left as a layout
        # number, because "a shift the window makes for itself" is the shape of defect
        # this whole area has been faulted for. Parked at the BOTTOM of the column at
        # 1120x720 -- the worst case, since everything below this label is on screen --
        # the change that first drops a step moves the scrollbar 0 px (it stays at 1917
        # while the range grows to 1968) and pushes the rows below down 51 px.
        #
        # THE CENSUS OF WHAT LANDS UNDER THE POINTER, taken densely rather than from a
        # handful of points, because a thin sample of this is how the count came out
        # wrong the first time. 93 points -- the centre of every visible child of the
        # column plus a 12 px grid down the viewport -- of which 28 are over a DIFFERENT
        # widget after the shift; the Undo and Redo buttons, the two the user of this
        # group is looking at, are not among them. THREE of the 28 are actionable, not
        # one:
        #   * a point in the dead band inside this group box lands on the template combo,
        #     and a click there opens its dropdown;
        #   * the label under the combo ("A template carries the plot type...") lands on
        #     the same combo, with the same dropdown -- a real label, not a dead band;
        #   * and the one that is not a click eaten by inert text: a point on
        #     "Export plot data..." lands on "Save image...", so a click aimed at one
        #     BUTTON arrives at a different button and opens a modal "Save plot image".
        # The remaining 25 are inert -- a click changes no config, opens no modal and
        # opens no popup. The row order below this label is doing less work than it was
        # once credited with: it is why the two history buttons are safe, not why the
        # column is.
        #
        # So the settle is small, one-time and mostly harmless, NOT harmless, and it is
        # left in place with that written down rather than silently: reserving it means a
        # permanent 45 px hole (30 px on a column dragged past ~360 px) in every session
        # for a sentence most sessions never see, and the user it could catch is one who
        # made the 200th change from somewhere other than this column while resting the
        # pointer on one of those three points. Reported, not closed.
        self._history_note = QtWidgets.QLabel("")
        self._history_note.setWordWrap(True)
        self._history_note.setStyleSheet(f"color:{_WARN_HEX};")
        self._history_note.setVisible(False)
        lay.addWidget(self._history_note)
        # The PLATFORM's own sequences rather than literals, so this window agrees with
        # every other undo the owner uses. Parented to the window, so they are
        # window-scoped and die with it.
        #
        # ...plus Ctrl/Cmd+Shift+Z by name for Redo, each RESOLVED key once, through the
        # shared ``key_sequences`` rule (design §3.7) rather than a copy of it. On a Mac
        # the standard Redo IS Cmd+Shift+Z, and a key registered twice is an "ambiguous
        # shortcut overload" that Qt answers by doing NOTHING -- Redo gone on exactly the
        # platform that had it right. ``key_sequences`` holds the measurement.
        for keys, slot in ((undo_sequences(), self.undo_config),
                           (redo_sequences(), self.redo_config)):
            for key in keys:
                QtGui.QShortcut(key, self).activated.connect(slot)

        # -- A3: named templates ---------------------------------------------------------
        self._template_combo = QtWidgets.QComboBox()
        _narrow_combo(self._template_combo)
        self._template_combo.setToolTip(
            "Your saved plot templates. Picking one applies its plot type, columns, fit "
            "recipe, axes and style at once — and leaves your ticked sources alone.")
        self._template_combo.currentTextChanged.connect(self._on_template_selected)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Template:"))
        trow.addWidget(self._template_combo, 1)
        lay.addLayout(trow)

        brow = QtWidgets.QHBoxLayout()
        self._template_save_btn = QtWidgets.QPushButton("Save as…")
        self._template_save_btn.setToolTip(
            "Save everything below the source list under a name, so this figure's look "
            "and recipe can be put on another recording.")
        self._template_save_btn.clicked.connect(self._save_template)
        self._template_rename_btn = QtWidgets.QPushButton("Rename")
        self._template_rename_btn.setToolTip(
            "Rename the selected template. Its saved settings move unchanged — this is "
            "not a re-save of what is on screen.")
        self._template_rename_btn.clicked.connect(self._rename_template)
        self._template_delete_btn = QtWidgets.QPushButton("Delete")
        self._template_delete_btn.setToolTip("Forget the selected template.")
        self._template_delete_btn.clicked.connect(self._delete_template)
        for b in (self._template_save_btn, self._template_rename_btn,
                  self._template_delete_btn):
            brow.addWidget(b)
        lay.addLayout(brow)

        # THE SPLIT, ON SCREEN. A template that quietly did or did not move the tick
        # boxes would be found out by experiment, once, on somebody's real figure; the
        # rule is short enough to simply state where it is used.
        split = self._template_split_note = QtWidgets.QLabel(
            "A template carries the plot type, columns, fit recipe, axis limits and the "
            "whole style. It never carries which sources are ticked, your retyped or "
            "restyled legend entries, or an applied fit — the fit arrives selected, and you "
            "press Apply.")
        split.setWordWrap(True)
        split.setStyleSheet(f"color:{_MSG_HEX};")
        lay.addWidget(split)

        self._template_updating = True
        try:
            self._load_templates_into_combo()
        finally:
            self._template_updating = False
        # A project just opened may have added templates (gui.plot_preset_sync).
        preset_store().changed.connect(self._reload_templates_keeping_selection)

        # -- B3: the JSON sidecar --------------------------------------------------------
        jrow = QtWidgets.QHBoxLayout()
        self._export_cfg_btn = QtWidgets.QPushButton("Export JSON…")
        self._export_cfg_btn.setToolTip(
            "Write this plot's WHOLE configuration — ticked sources included — to a "
            "small .json file you can keep beside the figure or send to somebody.")
        self._export_cfg_btn.clicked.connect(lambda: self.export_config_json())
        self._import_cfg_btn = QtWidgets.QPushButton("Import JSON…")
        self._import_cfg_btn.setToolTip(
            "Read a plot configuration back. Sources this session has not loaded are "
            "skipped; everything else is restored exactly.")
        self._import_cfg_btn.clicked.connect(lambda: self.import_config_json())
        jrow.addWidget(self._export_cfg_btn)
        jrow.addWidget(self._import_cfg_btn)
        lay.addLayout(jrow)
        return box

    # -- A2: the history hook --------------------------------------------------------------------
    def _wire_history(self, root: QtWidgets.QWidget) -> None:
        """Connect every control under ``root`` to the history hook, GENERICALLY.

        WHY A WALK AND NOT A LIST OF CONNECTIONS. The window has ~40 controls and its
        handlers do not funnel: ``_changed`` / ``_mark_stale`` / ``_armed_redraw`` catch
        most of them, but the fit proposal (``_fit_selection_moved``), the clustering
        method and k-mode, and the axis-range Apply/Auto buttons all change the config
        and reach none of the three. A hand-kept list of hooks would therefore be
        incomplete on the day it was written -- and this file's own history says what
        happens next: ``reset_state``'s hand-kept widget list went stale and silently
        stopped blanking six controls. A walk cannot go stale, and a control added to any
        group next month is covered without anyone remembering this method exists.

        The hook is idempotent (it ends in a DIFF, see :meth:`_history_commit`), so
        connecting a widget twice, or connecting one whose signal changes nothing about
        the config, costs one comparison and records nothing.

        TWO KINDS OF CONTROL, TWO SLOTS. A text box coalesces (a keystroke is not a
        step); everything else commits at once. QLineEdits that BELONG to a spin box or a
        combo are skipped -- they are that widget's internals, they fire on every spin,
        and routing a spin through the text path would give it the text grain.
        """
        for w in root.findChildren(QtWidgets.QWidget):
            if isinstance(w, QtWidgets.QLineEdit):
                if isinstance(w.parent(), (QtWidgets.QAbstractSpinBox,
                                           QtWidgets.QComboBox)):
                    continue
                if w.objectName() == "columnChecklistFilter":
                    continue    # filtering a checklist is not an edit
                w.textChanged.connect(self._history_touch_text)
                # Focus-out AND Return both land here, which is what makes "until
                # focus-out" a real rule rather than a hope: clicking any other control
                # takes focus off the box, so Qt closes the text run for us BEFORE that
                # control's own signal arrives.
                w.editingFinished.connect(self._history_touch)
                continue
            # ``itemChanged`` IS DELIBERATELY NOT IN THIS TUPLE, and it is the one
            # exception the walk is allowed. Its argument is a raw ``QListWidgetItem*``,
            # which PySide converts ONCE PER RECEIVER: put a second receiver behind a
            # handler that rebuilds its own list, and the second conversion reads an
            # item the first one has already destroyed. That is exactly how the shipped
            # build segfaulted (see `_sync_legend_entry_list`). The narrowing is by
            # SIGNAL, not by widget, so it cannot go stale the way a hand-kept widget
            # list does -- and the two handlers that carry `itemChanged` call
            # `_history_touch` themselves, in the same position this connection held.
            for name in ("currentTextChanged", "valueChanged", "toggled",
                         "itemSelectionChanged", "currentFontChanged"):
                signal = getattr(w, name, None)
                if signal is not None and hasattr(signal, "connect"):
                    signal.connect(self._history_touch)

    def _history_live(self) -> bool:
        """Whether a change happening now is the USER'S and should be recordable.

        Three exclusions, each already meaning "this is not a user gesture" elsewhere in
        this window: ``_updating`` (a bulk widget write), ``_installing`` (inside
        ``_set_config`` itself) and ``_history_suspended`` (an undo putting a config
        back, or new data arriving).
        """
        return bool(self._ui_ready and not self._updating
                    and not self._installing and not self._history_suspended)

    def _history_touch(self, *_a) -> None:
        """A control moved: close the step now.

        THE GRAIN RULE, half one. Everything that is not typed text -- a combo, a spin, a
        checkbox, a tick in the source list, a colour, a button -- is one gesture and
        therefore one undo entry, committed on the spot. Called both from the generic
        wiring above and from the handful of handlers that change the config without
        moving a control (:meth:`_apply_axis_range`, :meth:`_reset_axis_range`).
        """
        if not self._history_live():
            return
        self._history_commit()

    def _history_touch_text(self, *_a) -> None:
        """A text box changed: hold the step open.

        THE GRAIN RULE, half two, and the reason there is a rule at all: every keystroke
        in the Title box emits ``textChanged``, so a per-signal stack would need eleven
        Ctrl+Z to walk back out of "figure 3b" and the twelfth would be the colour change
        before it. A run of typing is ONE change, and it is closed by whichever comes
        first:

        * **focus-out or Return** -- ``editingFinished``, wired above. This is the common
          case and it costs nothing, because clicking the next control is what ends the
          typing;
        * **a pause** of :data:`_HISTORY_COALESCE_MS` -- for the user who types a title,
          stops to look at the plot, and then keeps typing. Without it their whole title
          would be one entry however long they paused.

        The buttons and the template name are synced IMMEDIATELY even though the step is
        not committed yet, because "Undo is available" is true from the first keystroke
        (:meth:`_sync_config_buttons` counts an open step) and a template's name stops
        describing the window the moment its title is retyped.
        """
        if not self._history_live():
            return
        # Dirty/recovery tracking cannot wait for the undo coalescing timer: a user
        # may type a title and quit immediately.  The recovery controller coalesces
        # these revision announcements independently, while the plot's own undo stack
        # still treats the run of keystrokes as one gesture below.
        self.workspaceChanged.emit()
        self._history_timer.start(_HISTORY_COALESCE_MS)
        self._sync_config_buttons()
        self._sync_template_selection()

    #: The config fields the user types CHARACTER BY CHARACTER, as
    #: ``(section, field)``. Every one of them is a free-text box whose ``textChanged``
    #: fires per keystroke; nothing else in the config can move a character at a time.
    #: Listed rather than derived from "is it a str", because ``plot_type``,
    #: ``normalization``, ``method`` and ``range_unit`` are strings too and are all
    #: PICKED from a combo -- one gesture, one step.
    _TEXT_FIELDS = (("style", "title"), ("style", "x_label"), ("style", "y_label"),
                    ("fit", "range_lo"), ("fit", "range_hi"),
                    ("axes", "x_min"), ("axes", "x_max"),
                    ("axes", "y_min"), ("axes", "y_max"))

    @classmethod
    def _typed_text_only(cls, was: PlotConfig, now: PlotConfig) -> bool:
        """Whether ``now`` differs from ``was`` in TYPED TEXT and in nothing else.

        THE RULE IS ABOUT THE CHANGE, NOT ABOUT WHICH SIGNAL DELIVERED IT, and that is
        deliberate. The obvious implementation -- "defer while the coalescing timer is
        running" -- depends on the ORDER Qt delivers one signal's slots in, and that
        order is against us: ``_title_edit.textChanged`` reaches ``_on_style_title``
        (connected while the Style group was built) before it reaches the history hook
        (connected afterwards, over the finished tree), so the choke point sees the
        keystroke with no timer armed yet. Measured with the timer test: "fig 3b"
        committed its first keystroke as a step of its own and coalesced only the rest.
        Comparing the two CONFIGS cannot be fooled that way, and it needs no
        per-handler bookkeeping -- which is the thing this whole round exists to remove.
        """
        if was == now:
            return False
        def _blank(cfg: PlotConfig) -> PlotConfig:
            out = cfg
            for section in ("style", "fit", "axes"):
                names = [n for sec, n in cls._TEXT_FIELDS if sec == section]
                out = _dc_replace(out, **{section: _dc_replace(
                    getattr(out, section), **{n: "" for n in names})})
            # The fit label's template box is typed text too; compared as the label it
            # draws, so the first keystroke (which saves the implied label) is text only.
            label = _dc_replace(out.style.fit_label_state(), template="")
            return _dc_replace(out, style=out.style.with_fit_label(label))
        return _blank(was) == _blank(now)

    def _history_settle(self) -> None:
        """Something may have changed: close the step UNLESS the change is typed text.

        The third door, and the one that keeps the two halves of the grain rule from
        fighting. The choke points (:meth:`_changed`, :meth:`_armed_redraw`,
        :meth:`_mark_stale`) are reached by BOTH kinds of change -- a colour picked in a
        modal arrives there, and so does every keystroke in the Title box, because
        ``_on_style_title`` ends in ``_changed``. Committing unconditionally there would
        make the coalescing rule a dead letter.

        Deferring cannot swallow a change, because everything that ENDS a run of typing
        commits outright: ``editingFinished`` fires the moment focus leaves the box --
        which is what clicking any other control does, BEFORE that control's own signal
        arrives -- and the timer armed below fires on a pause. And a change that is text
        PLUS anything else is not deferred at all, so a mixed gesture is never held open
        by its typed half.
        """
        if not self._history_live():
            return
        if self._typed_text_only(self._history.base, self.config()):
            self._history_timer.start(_HISTORY_COALESCE_MS)
            self._sync_config_buttons()
            self._sync_template_selection()
            return
        self._history_touch()

    def _history_commit(self) -> None:
        """Close any open step: push the previous config if the window has actually moved.

        The DIFF, and the only place a step is ever recorded. Callers say "something may
        have changed" and this decides; a caller that says it too often costs one 0.4 us
        comparison, and one that forgets costs a coarser entry rather than a lost state
        (the next caller's push carries both changes). That is
        :class:`nanoscope.core.history.EventHistory`'s inversion, bought here for nothing
        because a config is already a value.
        """
        self._history_timer.stop()
        if not self._ui_ready or self._installing or self._history_suspended:
            return
        self._format_text_run_key = None
        if self._history.note(self.config()):
            self.workspaceChanged.emit()
        self._sync_config_buttons()
        self._sync_template_selection()

    def undo_config(self) -> bool:
        """Step the whole configuration back one change. True if anything moved.

        Public: the Undo button, Ctrl+Z and tests. Any run of typing still open is
        committed FIRST, so Ctrl+Z immediately after typing a title undoes the title
        rather than stepping past it to whatever came before.
        """
        self._history_commit()
        moved = self._install_history(self._history.undo())
        # A REFUSED UNDO MOVES NOTHING -- not the plot, not the stack, and not the view.
        # Rounds 3 and 4 answered it by scrolling the ceiling sentence into view, and that
        # scroll was an ambush on whichever gesture could reach it. Driven on the build
        # that still had it, at 1120x720, as one unbroken session: scroll with a
        # pixel-precise wheel until the Undo button appears (it stops at 1660), click that
        # button 200 times to the floor without the pointer ever moving, press Ctrl+Z once
        # more -- the column goes 1660 -> 1729, the widget under the unmoved pointer goes
        # QPushButton -> the TEMPLATE COMBO, the next click at that same point opens its
        # dropdown (a 277x94 popup), and one further click on the row 31 px below installs
        # a whole configuration (bins 19 -> 30, the combo left reading "Publication figure
        # (light)"). `ensureWidgetVisible` always lands the column on the same value, so
        # how far it travels -- and so what arrives under the pointer -- was decided by
        # where the user happened to stop scrolling. It is gone on every trigger, and
        # `test_no_gesture_at_the_undo_floor_scrolls_the_controls_column` is what holds
        # it out.
        #
        # THE DISCLOSURE NEVER DEPENDED ON IT. Two surfaces stand on their own, both
        # measured at the floor on THIS build, with `dropped == 10`:
        #   * THE UNDO BUTTON'S OWN TOOLTIP, in both of the states it has here. While
        #     there is still history to step back: "Undo the last change to this plot's
        #     configuration (200 to step back). Its oldest history was dropped at the
        #     200-step ceiling (10 steps lost)." At the floor, on the dead button:
        #     "Nothing left to undo -- this plot's oldest history was dropped at the
        #     200-step ceiling (10 steps lost)." Qt renders a tooltip for a DISABLED
        #     widget -- measured twice on the real dead button, once by sending the
        #     `QHelpEvent` through `QApplication.sendEvent` and once by a REAL hover at
        #     the platform seam (`QTest.mouseMove` to the window handle, then wait), which
        #     delivers exactly one ToolTip event to the disabled button and shows the
        #     text. That is what the mouse user at the floor gets, for no movement at all.
        #   * THE STANDING AMBER LINE, `_history_note`, which lives 6 px under the Undo
        #     row it explains -- so it is found by scrolling to the controls it is about,
        #     which is where somebody hunting for an undo goes. It is NOT on screen merely
        #     because the button is, and that is said plainly rather than papered over:
        #     parked with the button flush at the fold, the line's rect clears the
        #     viewport's bottom edge by 51 px at 775x450, 900x600, 1120x720 and 1512x900
        #     alike, `visibleRegion()` empty at all four. Buying those 51 px with a scroll
        #     is exactly what cost three rounds.
        #
        # NOT Details, which is the other surface that is on screen without scrolling
        # (measured: 180 px tall, `visibleRegion()` non-empty at rest): it is the
        # written record of what the window said about the DATA, and it is rewritten in
        # full by every draw -- a note about this session's EDITING would both sit in
        # the wrong file and blink out with the next picture.
        return moved

    def redo_config(self) -> bool:
        """Step the whole configuration forward one change. True if anything moved."""
        self._history_commit()
        return self._install_history(self._history.redo())

    def history_mark(self) -> int:
        """How deep the Undo stack is, as one number that only a step taken, undone or
        redone can move: the steps back, the steps the 200-step ceiling dropped (so it
        keeps growing past the ceiling), and one for a run of typing still open.

        The Analysis Explorer's Style menu compares it before and after a look to know
        whether a page still stands where the look left it. Read-only. This is depth,
        not a unique revision: the Style service also watches `historyChanged`, so an
        Undo followed by a different edit cannot revive an earlier batch."""
        back, _forward = self._history.depth()
        pending = self._ui_ready and self._history.base != self.config()
        return back + int(self._history.dropped) + int(bool(pending))

    def _install_history(self, cfg: PlotConfig | None) -> bool:
        """Put a config from the stack back on the widgets, without recording it.

        ONE WRITER, AND NO CASCADE. Everything goes through :meth:`_set_config`, which
        writes all ~40 controls under ``_updating`` so no handler runs, and moves the
        identity anchor itself. ``_history_applying`` covers the one gap ``_updating``
        does not: the draw at the END of ``_set_config``, which runs with ``_updating``
        already restored and would otherwise be noted as a fresh change.

        THE RUN GATE IS RESPECTED. On an armed window the restored config is drawn --
        undo has to SHOW you what you got back, and ``apply_config``'s contract is that
        naming a whole configuration is the run gesture. On a window that has never been
        run, it is not: drawing would put a picture on screen the user never asked for,
        which is the complaint the gate exists to answer, so the prompt goes back up.
        """
        if cfg is None:
            return False
        self._history_suspended = True
        try:
            armed = self._plot_armed
            # `own`: a stack entry is this window's own state coming back, and the
            # origin it descends from came back with it (`_ConfigHistory.undo`), so
            # the import disclosures are decided by that origin, not re-decided here
            # (S7-FIX1: an undo that lands on the imported state keeps its sentences).
            current = self.config()
            if _dc_replace(cfg, style=current.style) == current:
                # Undoing ink does not commit unrelated pending source/axis/fit settings.
                prior = self._updating
                self._updating = True
                try:
                    self._style = cfg.style
                    self._apply_style_widgets(cfg.style)
                    self._sync_source_row_labels()
                finally:
                    self._updating = prior
                self._presentation_changed()
            else:
                self._set_config(self._installable(cfg), draw=armed, own=True)
            if not armed:
                self._show_plot_prompt()
        finally:
            self._history_suspended = False
        # Re-based onto what is ACTUALLY on screen rather than onto ``cfg``: a column the
        # window can no longer offer was resolved away by `_installable`, and a base that
        # disagreed with the widgets would make the very next touch push a phantom step.
        self._history.rebase(self.config())
        self._sync_config_buttons()
        self._sync_template_selection()
        # An Undo or Redo is a durable edit like the step it takes back: a restored (frozen)
        # page is saved from the checkpoint this announcement refreshes, and the project is
        # now different from the file it was opened from.
        self.workspaceChanged.emit()
        return True

    @staticmethod
    def _retire_history_button(dying: QtWidgets.QAbstractButton,
                               partner: QtWidgets.QAbstractButton,
                               partner_wanted: bool) -> None:
        """Hand focus off a history button that is about to be disabled.

        Costs at most two ``hasFocus()`` calls per ``_sync_config_buttons``, which runs on
        every commit, and does nothing at all unless the button that is going away is the
        one holding focus -- the only case Qt would otherwise resolve by walking the tab
        chain and scrolling the column to follow it.

        The partner is enabled first: ``setFocus`` on a disabled widget is refused
        without complaint, and the caller's own ``setEnabled`` pair runs immediately
        afterwards, so this cannot leave either button in a state the stack disagrees
        with. When the partner is not being offered either -- both buttons dead, which
        is a window with no history in either direction -- focus is dropped rather than
        handed anywhere: nothing is scrolled, and the next Tab starts the chain again.
        """
        if not dying.hasFocus():
            return
        if partner_wanted:
            partner.setEnabled(True)
            partner.setFocus()
        else:
            dying.clearFocus()

    def _sync_config_buttons(self) -> None:
        """Enable/label Undo and Redo from the stack, counting an OPEN text run.

        ``pending`` is the run of typing that has not been committed yet. Undo is offered
        for it (committing is the first thing :meth:`undo_config` does, so there really
        is something to step back to), and Redo is withdrawn during it -- committing will
        destroy the redo branch, and a button that does something other than what it says
        is worse than a disabled one.
        """
        if not hasattr(self, "_undo_btn"):
            return   # fires during construction, before this group exists
        # Every change of the configuration, whichever door it came through, ends here: the
        # sentence of the last reset stops standing once what it names is typed or pinned again.
        self._lapse_reset_sentence()
        pending = self._ui_ready and self._history.base != self.config()
        back, forward = self._history.depth()
        lost = int(self._history.dropped)
        steps = back + bool(pending)
        want_undo = bool(self._history.can_undo or pending)
        want_redo = bool(self._history.can_redo and not pending)
        # WHERE FOCUS GOES WHEN ONE OF THESE TWO IS RETIRED, decided here instead of by
        # the tab chain. Qt's default is `focusNextPrevChild`, and inside a QScrollArea
        # that also SCROLLS -- so a click on the enabled Undo button that empties the
        # stack disabled the button under the pointer, sent focus on down the chain to
        # the template combo, and dragged the column after it. Measured at 1120x720
        # parked with the button at the fold: 1658 -> 1738 (+80) with nothing dropped and
        # 1658 -> 1789 (+131) at the undo floor, where the amber line below adds its 51 px
        # to the distance focus has to travel; "Save as..." or the template combo was left
        # under a pointer that had not moved, so the user's next click reached a modal
        # "Save plot template" or the combo's dropdown. Same shape on Redo, measured
        # 1658 -> 1789 at the fold by clicking it until the redo stack emptied.
        #
        # THE ANSWER IS THE PARTNER BESIDE IT, not the next widget in the chain: the two
        # buttons share one QHBoxLayout, so the partner is on the same row and therefore
        # already on screen whenever the dying button was, and the gesture that retires
        # one is the gesture that enables the other (an undo that empties the stack fills
        # the redo stack). A plain `setFocus` does not route through
        # `QScrollArea.focusNextPrevChild`, so nothing scrolls -- measured 0 px on the
        # four doors that could move it: a mouse click at the fold (0 of 78 cells of the
        # grid that used to move 78 of 78, note standing and note hidden alike), a
        # keyboard Space on the tab-focused button, a click on a button that was already
        # focused, and the same shape on Redo. Where focus ENDS is the partner button:
        # measured on screen, `visibleRegion` non-empty, on the same row as the button
        # that died (both at y=694 at the fold at 1120x720).
        #
        # `_retire_history_button` enables the partner BEFORE handing it focus because
        # `setFocus` on a disabled widget is silently refused -- measured: with the two
        # `setEnabled` calls left in their old order the hand-off fell through to
        # `clearFocus` and focus ended on None. The authoritative pair below still runs,
        # so the enabled state is what `want_undo`/`want_redo` say either way.
        if not want_undo:
            self._retire_history_button(self._undo_btn, self._redo_btn, want_redo)
        if not want_redo:
            self._retire_history_button(self._redo_btn, self._undo_btn, want_undo)
        self._undo_btn.setEnabled(want_undo)
        self._redo_btn.setEnabled(want_redo)
        # THE CEILING SPEAKS HERE BECAUSE THIS IS WHERE IT CAN. `_history_commit` is the
        # ONLY caller of `_ConfigHistory.note` and it calls this method on the next line,
        # so one derivation here covers every path that can drop a step -- no signal is
        # needed, unlike `EventHistory`, whose stack lives in another object under many
        # keys and therefore needed `stepsDepthDropped`.
        #
        # The wording is the EVENT table's, subject swapped: "its N oldest steps have
        # been dropped and cannot be undone" and "(N steps lost)" are lifted from
        # `main_window._on_steps_depth_dropped` and `_sync_undo_actions`. Two windows in
        # one app must not have two dialects for one idea -- which is why the plural was
        # fixed in BOTH of them in the same edit (see the setText below).
        #
        # TWO SURFACES, NOT ONE, because they are reached differently. The tooltip DOES
        # render on this button while it is disabled -- measured on the real dead button
        # two ways, `QApplication.sendEvent` with a `QHelpEvent` and a REAL hover at the
        # platform seam (one ToolTip event delivered, `QToolTip.isVisible()` true, the
        # sentence in `QToolTip.text()`), so the Edit menu's "a disabled QAction cannot
        # explain itself" lesson does not transfer -- but it needs a HOVER, which is what
        # the hand on the mouse is already doing and what the hand on Ctrl+Z is not. The
        # standing line below needs no hover, but it is not on screen for free either:
        # measured, parked with this button flush at the fold the line clears the
        # viewport's bottom edge by 51 px at 775x450, 900x600, 1120x720 and 1512x900, so
        # it is found by scrolling on to the controls it explains. Neither surface is
        # bought with a movement the window makes -- that was tried for three rounds and
        # the account is at :meth:`undo_config`.
        ceiling = (f" Its oldest history was dropped at the {_HISTORY_DEPTH}-step "
                   f"ceiling ({lost} step{'s' if lost != 1 else ''} lost).") if lost else ""
        if steps:
            self._undo_btn.setToolTip(
                f"Undo the last change to this plot's configuration "
                f"({steps} to step back).{ceiling}")
        elif lost:
            # NOT "Nothing to undo yet." -- there WAS history here and the window threw
            # it away. Measured: after 210 changes and 200 undos the old tooltip read
            # exactly the sentence a never-edited window shows. The two states look
            # identical to the user and mean opposite things.
            self._undo_btn.setToolTip(
                f"Nothing left to undo — this plot's oldest history was dropped at the "
                f"{_HISTORY_DEPTH}-step ceiling "
                f"({lost} step{'s' if lost != 1 else ''} lost).")
        else:
            self._undo_btn.setToolTip("Nothing to undo yet.")
        self._redo_btn.setToolTip(
            "Redo is cleared by a new change." if pending else
            f"Redo ({forward} to step forward)." if forward else "Nothing to redo.")
        # SHOWN ONCE AND THEN LEFT ALONE. It appears on the change that first crosses the
        # bound and thereafter only its number moves -- no pop, no flash, no modal -- so
        # a user 300 changes into a figure is not interrupted a hundred times. (QLabel
        # early-outs on an unchanged string, so the hidden case costs nothing on the
        # keystroke path this method already runs on.)
        #
        # PAST TENSE, AND THE TOOLTIP'S TENSE, because this line STANDS. `dropped` is a
        # lifetime counter, so what it licenses is a statement about what happened, not
        # about what the stack currently is: undo five steps and make one fresh change
        # (which destroys the redo branch) and the stack holds 196 -- measured -- while
        # a line reading "is at its 200-step ceiling" goes on asserting a present state
        # that stopped being true. The status-bar wording this borrows from
        # (`main_window._on_steps_depth_dropped`) is present tense because it is a
        # 12-second toast fired AT the drop, where the present is the instant it
        # describes; a permanent label has no such instant. So the verb comes from the
        # other half of the same dialect -- the Edit menu's "oldest history WAS dropped"
        # -- while the clause that names the loss stays word for word the toast's, which
        # is the part `test_the_two_windows_speak_one_dialect` holds the two windows to.
        #
        # AND IT IS SPELLED OUT RATHER THAN PARENTHESISED, in both windows at once. The
        # borrowed clause said "N of its oldest step(s) have been dropped", which is
        # wrong on the very first drop -- "1 of its oldest step(s) HAVE been dropped" --
        # and the first drop is the one moment this line is guaranteed to be read, since
        # it is the moment it appears. The tooltip 31 px away already gets it right ("1
        # step lost"), and so does the Edit menu's own clause in the other window
        # (`main_window._sync_undo_actions`: "step" plus a conditional s), so the app
        # was disagreeing with itself in two places on one screen. "step(s)" is a form
        # notice writers use to avoid choosing; a window with one number in hand can
        # choose. Both branches are written out whole -- rather than assembling a verb
        # and a plural from fragments -- so that each finished sentence appears verbatim
        # in the source of BOTH windows and `test_the_two_windows_speak_one_dialect` can
        # go on reading them back out of `inspect.getsource` and matching them; a
        # sentence stitched together at run time would be pinnable only by rendering it.
        self._history_note.setText(
            f"This plot's undo history reached its {_HISTORY_DEPTH}-step ceiling — "
            + ("its oldest step has been dropped and cannot be undone." if lost == 1 else
               f"its {lost} oldest steps have been dropped and cannot be undone.")
            if lost else "")
        self._history_note.setVisible(bool(lost))
        if not self._history_suspended:
            self.historyChanged.emit()

    # -- A3: named plot templates ----------------------------------------------------------------
    #: Sentinel entry: the settings are whatever the controls say, with no template's
    #: name over them. Spelled exactly as ``DetectionPanel._PROTO_MANUAL`` is, because
    #: the two combos sit in the same app and must not have two dialects for one idea.
    _TPL_MANUAL = "— manual —"
    _TPL_SETTINGS_KEY = "plots/templates"

    #: SHIPPED templates, listed above the user's own and not editable.
    #:
    #: Deliberately only three, and deliberately thin: unlike the DETECTION protocols --
    #: which encode measured, cross-checked preprocessing chains from the owner's own
    #: notebooks -- a plot template is a presentation choice, and inventing numbers for
    #: somebody's figure is not a service. Each of these is either pure appearance or the
    #: analysis shape this window ALREADY defaults to, so none of them asserts anything
    #: about data.
    #:
    #: The first names no plot type at all, which is the point: a template without one is
    #: a LOOK (:meth:`PlotConfig.with_template`), recalled as the nine style settings it
    #: names and nothing more, so it restyles whatever is on screen -- a fitted log-log
    #: scatter, a log-x density histogram -- without touching its analysis or its words. (Blank
    #: columns alone were not enough: the other analysis fields have no blank, and were
    #: installed as defaults.) The second and third are analyses and are recalled whole;
    #: they carry the fit as a PROPOSAL only -- ``applied`` stays ``none`` -- so recalling
    #: one arms Apply and waits, exactly as picking the model by hand does. They also
    #: carry the default look, which a recall puts on the window like any saved template's.
    _BUILTIN_TEMPLATES: dict[str, PlotConfig] = {
        # Appearance only: a white canvas, no grid, text sized for a printed column, a
        # heavier curve, and a 1600x1200 export. The blank plot type says it names no
        # analysis; the other analysis fields are inert defaults that are never read.
        "Publication figure (light)": PlotConfig(
            plot_type="",
            style=StyleSettings(
                background="light", show_grid=False, font_size_pt=11,
                title_font_size_pt=13, axis_label_font_size_pt=12, tick_font_size_pt=11,
                line_width=2, export_w=1600, export_h=1200)),
        # The shape this window already opens on (dwell in ms, exponential), plus the two
        # choices a dwell histogram usually wants and does not get by default: a log x
        # axis, because dwell is log-distributed, and density rather than counts, because
        # counts cannot be overlaid across recordings of different length.
        "Dwell-time histogram (log-x, exponential)": PlotConfig(
            plot_type=HISTOGRAM,
            histogram=HistogramConfig(column="dwell_s", bins=50, log_x=True,
                                      normalization=_NORM_DENSITY),
            fit=FitConfig(kind=_FIT_EXP, k=1),
            axes=AxesConfig(time_unit="ms")),
        # The other standard nanopore view. Log x for the same reason; no fit line, since
        # whether dwell and depth should be regressed at all is the user's call.
        "Dwell vs depth scatter (log-x)": PlotConfig(
            plot_type=SCATTER,
            scatter=ScatterConfig(x="dwell_s", y="depth_pA", log_x=True),
            axes=AxesConfig(time_unit="ms")),
    }

    @staticmethod
    def _load_templates() -> dict[str, dict]:
        """The user's saved templates, as raw JSON payloads. A corrupt store means none.

        Never a crash and never a partial read: a store this build cannot make sense of
        is exactly as useful as an empty one, and refusing to open the Plots window over
        it would be the worse failure. Decoding each payload happens per template, in
        :meth:`_template_config`, so ONE unreadable entry cannot hide the others.
        """
        raw = QtCore.QSettings().value(PlotsWindow._TPL_SETTINGS_KEY, "")
        if not raw:
            return {}
        try:
            got = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return got if isinstance(got, dict) else {}

    @staticmethod
    def _store_templates(templates: dict[str, dict]) -> None:
        QtCore.QSettings().setValue(PlotsWindow._TPL_SETTINGS_KEY,
                                    json.dumps(templates))

    def _template_config(self, name: str) -> PlotConfig | None:
        """The config saved under ``name``, or None (reporting) if it cannot be read.

        Shipped templates win over a same-named stored one, on
        :meth:`DetectionPanel._on_protocol_selected`'s argument: the name is refused at
        save time, so a collision can only come from a store written before the shipped
        set existed.

        A stored template goes through the SAME reader an imported file does, so a
        version it cannot read is refused with the same sentence naming the same version
        rather than swallowed -- a template silently recalling as "nothing happened" is
        the shape of bug that gets diagnosed as "the app is broken".
        """
        builtin = self._BUILTIN_TEMPLATES.get(name)
        if builtin is not None:
            return builtin
        raw = self._load_templates().get(name)
        if raw is None:
            return None
        try:
            return plot_config_from_json(raw)
        except PlotConfigError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Plot template", f"“{name}” could not be recalled.\n\n{exc}")
            return None

    def _load_templates_into_combo(self) -> None:
        """Rebuild the combo: the sentinel, the shipped set in its declared order, then
        the user's own, sorted. Under ``_template_updating`` so the rebuild's own
        ``currentTextChanged`` cannot be read as the user picking something."""
        was, self._template_updating = self._template_updating, True
        try:
            self._template_combo.clear()
            self._template_combo.addItem(self._TPL_MANUAL)
            for name in self._BUILTIN_TEMPLATES:
                self._template_combo.addItem(name)
            for name in sorted(self._load_templates()):
                if name in self._BUILTIN_TEMPLATES:
                    continue    # a store written before the shipped set existed
                self._template_combo.addItem(name)
            self._set_template_buttons_enabled(False)
        finally:
            self._template_updating = was

    def _reload_templates_keeping_selection(self) -> None:
        """Rebuild the combo after the store changed elsewhere, keeping a named selection.

        Still accurate: templates ADDED by a project open did not change these settings, and
        the rebuild alone would drop the name to "— manual —".
        """
        name = self._template_combo.currentText()
        self._load_templates_into_combo()
        if name and name != self._TPL_MANUAL and self._template_combo.findText(name) >= 0:
            self._select_template(name)

    def _set_template_buttons_enabled(self, on: bool) -> None:
        """Rename and Delete act on the SELECTED template, so they move together.

        A SHIPPED template is never editable, on the built-in protocols' argument:
        renaming or deleting one would leave the combo naming settings the app can no
        longer produce.
        """
        editable = bool(on) and self._template_combo.currentText() not in (
            self._BUILTIN_TEMPLATES)
        self._template_rename_btn.setEnabled(editable)
        self._template_delete_btn.setEnabled(editable)

    def _select_template(self, name: str) -> None:
        """Put ``name`` in the combo without it re-applying, and without the sync
        knocking it straight back to the sentinel."""
        was, self._template_updating = self._template_updating, True
        try:
            self._template_combo.setCurrentText(name)
        finally:
            self._template_updating = was
        self._set_template_buttons_enabled(name != self._TPL_MANUAL)

    def _template_would_give(self, tpl: PlotConfig) -> PlotConfig:
        """What would be ON SCREEN if ``tpl`` were recalled right now.

        The only fair thing to compare the live window against when deciding whether a
        template's name still describes it. A template may leave a field blank ("keep what
        you have") and may name a column this recording does not carry, and both are
        resolved at install time -- so comparing the RAW stored value would drop the name
        to "— manual —" the instant the template was recalled, which is the one moment it
        is certainly still accurate.
        """
        # Installation materializes the derived quantity witness. Compare the same
        # canonical value here, otherwise the first cosmetic edit drops a valid name.
        return migrate_plot_quantity(self._installable(self.config().with_template(tpl)))

    def _sync_template_selection(self) -> None:
        """Drop the combo to "— manual —" once the controls stop matching the named
        template. Called from the same hook the history is.

        A COMPARISON, NOT A BLANKET DROP. ``DetectionPanel`` drops on any change at all;
        this one asks whether the change actually made a difference to what the template
        describes, so nudging a control and putting it back leaves the name in place --
        and so does moving something the template does not carry, like ticking another
        source. The combo then means what it says: "these settings are that template".
        """
        if not hasattr(self, "_template_combo") or self._template_updating:
            return
        name = self._template_combo.currentText()
        if not name or name == self._TPL_MANUAL:
            return
        tpl = self._BUILTIN_TEMPLATES.get(name)
        if tpl is None:
            raw = self._load_templates().get(name)
            try:
                tpl = None if raw is None else plot_config_from_json(raw)
            except PlotConfigError:
                tpl = None      # unreadable: it cannot be describing the window
        if tpl is not None and self._template_would_give(tpl).as_template() == (
                self.config().as_template()):
            return
        self._select_template(self._TPL_MANUAL)

    def _on_template_selected(self, name: str) -> None:
        if self._template_updating or not name or name == self._TPL_MANUAL:
            self._set_template_buttons_enabled(bool(name) and name != self._TPL_MANUAL)
            return
        tpl = self._template_config(name)
        if tpl is None:
            return
        self.apply_template(tpl)
        self._select_template(name)

    def apply_template(self, template: PlotConfig) -> None:
        """Recall ``template`` over the current data selection, and draw once.

        Public, and the ONLY way a template should be installed: the seam between what a
        template carries and what the window keeps lives in
        :meth:`PlotConfig.with_template`, and a caller that assembled the config itself
        would be re-deciding that seam by hand.
        """
        self.apply_config(self._installable(self.config().with_template(template)))

    def _template_name_ok(self, name: str, *, title: str) -> bool:
        """Reject the reserved sentinel and the shipped names. Reports, returns False."""
        if name == self._TPL_MANUAL:
            QtWidgets.QMessageBox.information(
                self, title, "That name is reserved for the manual state.")
            return False
        if name in self._BUILTIN_TEMPLATES:
            QtWidgets.QMessageBox.information(
                self, title,
                f"“{name}” is a built-in template and cannot be replaced. Save yours "
                f"under a different name — the built-in one stays available, so you can "
                f"always get back to it.")
            return False
        return True

    def _save_template(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save plot template",
            "Name for this plot's recipe and style\n"
            "(the ticked sources are NOT saved):")
        name = (name or "").strip()
        if not ok or not name:
            return
        if not self._template_name_ok(name, title="Save template"):
            return
        templates = self._load_templates()
        # SAVING ONTO AN EXISTING NAME IS A DESTRUCTIVE EDIT, and the names are typed
        # from memory, so a collision is a typo at least as often as an intent to update.
        # The same question the preprocessing protocols ask, for the same reason.
        if name in templates:
            answer = QtWidgets.QMessageBox.question(
                self, "Overwrite template?",
                f"A template named “{name}” already exists.\n\n"
                f"Overwriting replaces its saved settings with the ones on screen. This "
                f"cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        templates[name] = plot_config_to_json(self.config().as_template())
        self._store_templates(templates)
        self._load_templates_into_combo()
        self._select_template(name)

    def _rename_template(self) -> None:
        """Rename the selected template, MOVING its stored settings unchanged.

        Deliberately not "save under the new name and delete the old": that would capture
        what is on screen, which after any edit is not what was stored under the old name.
        """
        old = self._template_combo.currentText()
        if old == self._TPL_MANUAL or not old:
            return
        templates = self._load_templates()
        if old not in templates:
            return
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename plot template", f"New name for “{old}”:", text=old)
        new = (new or "").strip()
        if not ok or not new or new == old:
            return
        if not self._template_name_ok(new, title="Rename template"):
            return
        if new in templates:
            answer = QtWidgets.QMessageBox.question(
                self, "Overwrite template?",
                f"A template named “{new}” already exists.\n\n"
                f"Renaming “{old}” onto it replaces its saved settings. This cannot be "
                f"undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        templates[new] = templates.pop(old)
        self._store_templates(templates)
        self._load_templates_into_combo()
        self._select_template(new)

    def _confirm_delete_template(self, name: str) -> bool:
        """Ask before forgetting the template ``name``. True to delete.

        A template lives in the preferences, not the project, so no undo reaches it
        (design §3.8) and this question is the only thing between a stray click and the
        saved recipe. A separate method so a test can answer it without a live modal.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        # PLAIN TEXT: the name is the user's own typing, and a confirmation must show the
        # name it is asking about, not whatever Qt's rich-text guess makes of it.
        box.setTextFormat(QtCore.Qt.PlainText)
        box.setWindowTitle("Delete plot template")
        # The name goes in the body: macOS does not show a message box's window title.
        box.setText(f"Delete the plot template “{name}”?")
        box.setInformativeText(
            "Its saved recipe and style are forgotten, and this cannot be undone. The plot "
            "on screen does not change.")
        delete = box.addButton("Delete template", QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        # Named, never indexed (docs/ARCHITECTURE.md): Return and Esc both keep it.
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is delete

    def _delete_template(self) -> None:
        name = self._template_combo.currentText()
        if name == self._TPL_MANUAL or name in self._BUILTIN_TEMPLATES:
            return
        if not self._confirm_delete_template(name):
            return
        templates = self._load_templates()
        templates.pop(name, None)
        self._store_templates(templates)
        self._load_templates_into_combo()
        # The settings themselves stay on screen -- deleting the NAME is not a request to
        # change the plot -- so the combo lands on the sentinel, which is now the truth.
        self._select_template(self._TPL_MANUAL)

    # -- B3: the JSON sidecar --------------------------------------------------------------------
    def export_config_json(self, path: str | Path | None = None) -> Path | None:
        """Write the WHOLE configuration to a versioned ``.json`` sidecar.

        With ``path`` None a file dialog is shown; returns the written path, or None when
        the user cancelled or the write failed (a failed write is reported in a dialog,
        never returned as a path) -- the same contract :meth:`save_image` has, because
        they are the same gesture on two halves of one figure.
        """
        if path is None:
            chosen = ask_export_path(
                self, title="Export plot configuration",
                suggestion="plot-config.json",
                filters="Plot configuration (*.json);;All files (*)")
            if chosen is None:
                return None
            path = chosen
        out = Path(path)
        try:
            write_plot_config(out, self.config())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export plot configuration",
                f"{out.name} could not be written.\n\n{exc}")
            return None
        self._set_details(self.details_text() + f"\n\nSaved plot configuration → {out.name}")
        return out

    def import_config_json(self, path: str | Path | None = None) -> bool:
        """Read a configuration back from a ``.json`` sidecar and apply it. True if applied.

        REFUSES, IT DOES NOT CRASH. Every failure this can meet -- not JSON, not a plot
        configuration, malformed, or a version this build cannot read -- arrives as
        :class:`PlotConfigError` carrying a sentence with the file name (and, for a
        version refusal, the version) in it, and that sentence is what the user sees.

        Applied through :meth:`apply_config`, so a restored configuration is installed by
        the one writer and drawn once. Sources the file names that this session has not
        loaded are still SKIPPED -- see :meth:`_apply_checked_keys` for why refusing the
        whole file over them would be the worse answer -- but no longer in silence: they
        are one of the seven kinds of loss :class:`_ConfigLosses` counts and names in
        Details, along with a column these events have not got, a fit or a y quantity this
        build does not carry, and any key the format has no field for at all.

        AND THIS IS THE ONLY DOOR THAT RECORDS THEM, because it is the only door with a
        FILE behind it: a template carries part of a configuration on purpose, and an undo
        entry and a script's own ``apply_config`` have nothing to be unfaithful to. The
        record is taken under the origin this import opens and spoken while the state on
        screen descends from it (:class:`_ConfigLosses`), so the sentence stands on the
        file's own state -- an undo back onto it included -- and on no other.
        """
        if path is None:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import plot configuration", "",
                "Plot configuration (*.json);;All files (*)")
            if not chosen:
                return False
            path = chosen
        try:
            payload, cfg = _load_plot_config(path)
        except PlotConfigError as exc:
            QtWidgets.QMessageBox.warning(self, "Import plot configuration", str(exc))
            return False
        # THE ONE DOOR WITH A FILE BEHIND IT, so the one door that can say what the file
        # asked for and did not get. The raw payload travels with the config because the
        # decoding has already dropped every key this build has no field for; the stash
        # is consumed by `_set_config` and cleared here whatever happens, so no later
        # install can be mistaken for this one. See :class:`_ConfigLosses`.
        self._incoming_config_file = (Path(path).name, payload, cfg)
        try:
            self.apply_config(self._installable(cfg))
        finally:
            self._incoming_config_file = None
        return True

    # -- shared by all three: making a config INSTALLABLE ----------------------------------------
    def _installable(self, cfg: PlotConfig) -> PlotConfig:
        """``cfg`` with every column name this window cannot actually select replaced by
        the one it is currently showing.

        THE STALE-ANCHOR HAZARD THIS CLOSES. :meth:`_set_config` deliberately declines to
        write a column a picker does not carry (Qt ignores ``setCurrentText`` for an
        absent value, so writing it would silently leave the old one selected) -- but it
        then sets ``_identity`` from the CONFIG, which now names a column that is not on
        screen. The next rebase measures from that phantom and clears the axis pins, the
        fit window and the title of a plot nothing changed about. Harmless for an undo
        entry taken seconds ago; certain for a template or an imported file, which are
        the whole point of this round.

        Resolved HERE rather than inside ``_set_config`` because it needs the widgets
        (only they know what is on offer) and because the writer's contract -- "install
        exactly this value" -- is what the round-trip test pins. The three callers that
        can be handed a config from outside this window do this first; the value they
        then install names only columns the window can honour.

        COLUMNS ONLY, AND THE Y QUANTITY DELIBERATELY NOT -- the fix that cost the app
        its one visible tell, and the reason this docstring says "column" everywhere.
        S6-03 added a sixth clause here that resolved an unofferable ``normalization``
        back to the one on screen. :meth:`_sync_template_selection` decides whether a
        template's NAME still describes the window by comparing
        :meth:`_template_would_give` -- which runs this method -- against
        :meth:`config`. Resolving the name away makes those two agree, so the comparison
        that exists to catch "this template no longer describes what you see" was
        answered with the SUBSTITUTION instead of the REQUEST. Measured on the owner's
        982-event file: recalling a template whose normalization reads ``density (1/s)``
        left the combo reading the template's own name -- the window asserting that these
        settings ARE that template -- where the build before that clause dropped it to
        "— manual —", with the bins visibly across (47) and nothing said anywhere.

        A REQUEST HAS TO SURVIVE LONG ENOUGH TO BE REFUSED. The clause was there to stop
        an unofferable name reaching ``_set_config``, where ``setCurrentText`` swallows it
        in silence; the answer is that ``_set_config`` no longer swallows it (it records
        the refusal -- :class:`_NormRefusal` -- and says so in Details on every draw), so
        there is nothing left here to protect against. The two kinds of name are also not
        the same kind of thing: a missing COLUMN is a fact about the recording, which the
        window can and should resolve against the data in front of it, while the y
        quantity's three items are compiled into this build, so a name that misses them
        cannot be explained by the data and must not be answered as though it could. That
        asymmetry is about where the substitution is DECIDED, not about whether it is
        disclosed: the column this method resolves away is now counted and named in Details
        like every other thing an imported file asked for and did not get -- see
        :class:`_ConfigLosses` -- so "resolve it against the data" no longer means
        "and say nothing".

        This costs nothing at the identity anchor, which is what the resolution above is
        for: :class:`AnalysisIdentity` carries the plot type, the columns and the time
        unit and NOT the normalization, so a foreign y-quantity name travelling on into
        ``_set_config`` cannot leave ``_identity`` pointing at a phantom the way a column
        would.
        """
        now = self.config()
        pairs = ((self._hist_col, cfg.histogram.column, now.histogram.column),
                 (self._scatter_x, cfg.scatter.x, now.scatter.x),
                 (self._scatter_y, cfg.scatter.y, now.scatter.y),
                 (self._kde_x, cfg.kde.x, now.kde.x),
                 (self._kde_y, cfg.kde.y, now.kde.y),
                 (self._heatmap_x, cfg.heatmap.x, now.heatmap.x),
                 (self._heatmap_y, cfg.heatmap.y, now.heatmap.y))
        hist_col, sx, sy, kx, ky, hx, hy = [
            want if want and combo.findText(want) >= 0 else have
            for combo, want, have in pairs]
        # THE Y QUANTITY IS NOT TOUCHED HERE, and that omission is load-bearing rather
        # than an oversight -- see this method's docstring for the measurement and for
        # what now catches a name this build cannot offer (`_set_config`, which records
        # the refusal, and `_NormRefusal`, which keeps it visible). The request travels on
        # intact so that `_sync_template_selection`, whose whole job is to notice that a
        # template no longer describes the window, is comparing the request.
        if (hist_col, sx, sy, kx, ky, hx, hy) == (
                cfg.histogram.column, cfg.scatter.x, cfg.scatter.y, cfg.kde.x, cfg.kde.y,
                cfg.heatmap.x, cfg.heatmap.y):
            return cfg
        return _dc_replace(
            cfg,
            histogram=_dc_replace(cfg.histogram, column=hist_col),
            scatter=_dc_replace(cfg.scatter, x=sx, y=sy),
            kde=_dc_replace(cfg.kde, x=kx, y=ky),
            heatmap=_dc_replace(cfg.heatmap, x=hx, y=hy))

    def drawn_table_is_stale(self) -> bool:
        """Judge the committed columns, including a figure with pending control edits."""
        cfg=getattr(self, '_figure_applied_config', None) or self.config()
        from .figure_adapter_plots import plotted_staleness_columns
        columns=plotted_staleness_columns(cfg)
        keys=tuple(self._figure_sources) or cfg.selection.source_keys
        sources=[self._sources[key] for key in keys if key in self._sources]
        return any(set(columns).intersection(self._superseded_columns(source))
                   for source in sources)

    # -- control-change handlers -----------------------------------------------------------------
    _APPEARANCE_FIELDS = {
        "title": "title", "x_label": "x_label", "y_label": "y_label",
        "font_family": "font_family", "title_size": "title_font_size_pt",
        "axis_size": "axis_label_font_size_pt", "tick_size": "tick_font_size_pt",
        "grid": "show_grid", "background": "background", "frame": "frame", "data_color": "data_color",
        "fit_color": "fit_color", "line_width": "line_width", "marker_size": "point_size",
        "legend_visible": "show_legend", "legend_size": "legend_font_size_pt",
    }

    def appearance_state(self) -> dict:
        """Appearance-only bridge; no sources, axes scales, bins or fit settings."""
        return self._appearance_of(self._style)

    @classmethod
    def _appearance_of(cls, style: StyleSettings) -> dict:
        out = {key: getattr(style, field) for key, field in cls._APPEARANCE_FIELDS.items()
               if getattr(style, field) not in (None, "")
               and not (key.endswith("size") and getattr(style, field) == 0)}
        # The figure's own element styles and the fit label ride the same bridge, so the
        # embedded Appearance section (its Fit label group) and shared presets see them.
        if style.text_styles:
            out["text"] = _thawed(style.text_styles)
        if style.fit_label:
            out["fit_label"] = _thawed(style.fit_label)
        if style.style_binding:
            out["style_binding"] = _thawed(style.style_binding)
        if style.series_styles:
            out["series_styles"] = _thawed(style.series_styles)
        return out

    def set_appearance_state(self, mapping) -> None:
        """Restore only appearance, silently and without committing pending analysis."""
        state = mapping if isinstance(mapping, PlotAppearance) else PlotAppearance.from_mapping(mapping)
        defaults = StyleSettings()
        values = {}
        for key, field in self._APPEARANCE_FIELDS.items():
            value = getattr(state, key)
            value = getattr(defaults, field) if value is None else value
            if isinstance(getattr(defaults, field), int) and not isinstance(value, bool):
                value = int(round(value))
                if field in ("line_width", "point_size"):
                    value = max(1, value)
            current = getattr(self._style, field)
            if (field in ("data_color", "fit_color") and isinstance(value, str)
                    and isinstance(current, str) and value.lower() == current.lower()):
                # The widget names colours in lower case ('#0072b2'); the same colour
                # spelt '#0072B2' is no change, or a Title keystroke stops being typed
                # text only and the first one becomes an Undo step of its own (D13).
                value = current
            values[field] = value
        values["text_styles"] = FrozenMap(
            {key: (style if _text_kept_in_styles(key) else _dc_replace(style, text=None)).to_dict()
             for key, style in (state.text or ())})
        values["fit_label"] = (FrozenMap(state.fit_label.to_dict()) if state.fit_label is not None
                               else _NO_ENTRIES)
        values["style_binding"] = _deep_frozen(state.to_dict().get("style_binding", {}))
        values["series_styles"] = _deep_frozen(state.to_dict().get("series_styles", {}))
        prior = self._updating
        self._updating = True
        try:
            from .figure_style_binding import preserve_legacy_baseline
            native = preserve_legacy_baseline(
                _encode_dc(_dc_replace(self._style, **values)), _encode_dc(self._style), "style_settings")
            self._style = _project_style_settings(_decode_dc(StyleSettings, native))
            self._apply_style_widgets(self._style)
            self._appearance_widget.set_state(state)
        finally:
            self._updating = prior
        if not prior:
            suspended = self._history_suspended
            self._history_suspended = True
            try:
                self._presentation_changed()
            finally:
                self._history_suspended = suspended

    def _appearance_from_user(self, state: PlotAppearance) -> None:
        if self._updating:
            return
        self.set_appearance_state(state)
        self._history_settle()

    def figure_export_target(self):
        return self._plot

    def figure_residual_target(self):
        return self._resid if not self._resid_widget.isHidden() else None

    def figure_applied_config(self) -> dict:
        """Shown scientific settings, with current presentation, not pending drafts."""
        applied = getattr(self, "_figure_applied_config", None)
        return {} if applied is None else plot_config_to_json(
            _dc_replace(applied, style=self._style))

    @staticmethod
    def fold_saved_config(config: PlotConfig, saved) -> PlotConfig:
        """A project's config (``config``, as the workspace decoded it) with the switches an
        older build saved in ``saved`` -- the workspace's tagged tree for it -- moved to
        where they live now (:func:`_fold_legacy_style`). The decoder reads only the fields
        this build has, so the old keys must be taken from the tree itself."""
        fields = saved.get("fields") if isinstance(saved, Mapping) else None
        style = fields.get("style") if isinstance(fields, Mapping) else None
        style = style.get("fields") if isinstance(style, Mapping) else None
        if not isinstance(style, Mapping) or not any(key in style for key in _LEGACY_STYLE_KEYS):
            return config
        return _dc_replace(config, style=_fold_legacy_style(config.style, style))

    def _drawn_config(self) -> PlotConfig:
        """:meth:`config` as the picture just drawn has it: the KDE overlay as APPLIED.

        ``config()`` reports the box, which is a proposal (a hand tick waits for Plot,
        and project open never estimates the curve); the fit needs no such step because
        its applied state is a config field of its own. Without this a Figure Library
        capture taken over a pending tick recorded a KDE the image and its exported data
        do not have.
        """
        cfg = self.config()
        if cfg.histogram.kde_overlay == self._applied_kde_overlay:
            return cfg
        return _dc_replace(cfg, histogram=_dc_replace(
            cfg.histogram, kde_overlay=bool(self._applied_kde_overlay)))

    def figure_source_recipes(self) -> list[dict]:
        """Durable sources of the shown numbers, never pending source checkboxes."""
        return copy.deepcopy(list(getattr(self, "_figure_sources", {}).values()))

    def _cache_figure_source(self, source: PlotSource) -> None:
        """Stamp each source once, only when a genuine draw adds its data items."""
        if source.key not in self._figure_sources:
            from ..core.analysis_workspace import _plot_source_recipe
            self._figure_sources[source.key] = _plot_source_recipe(self, source)

    def figure_series(self):
        from .plots_series import descriptors
        return descriptors(self)

    def _tag_series(self, item, role, key, label, *, identity_kind="source", colour=None):
        from .plots_series import tag_series
        tag_series(item, self._plot_type(), role, key, label,
                   identity_kind=identity_kind, colour=colour)

    def _apply_series_styles(self):
        from .plots_series import apply_series_styles
        apply_series_styles(self)
        if hasattr(self, "_nonlinear_fitter"):
            self._nonlinear_fitter.set_band_visible(self._style.show_ci_band)

    def _restyle_committed_items(self) -> None:
        """Style graphical objects, preserving category colors and selected rings."""
        objects = [*self._plot.items, *self._resid.items,
                   *(entry[0] for entry in getattr(self, "_presentation_legend", {}).values())]
        for item in dict.fromkeys(objects):
            role = getattr(item, "_presentation_role", None)
            if role is None:
                continue
            colour = (self._series_colour(role, index=getattr(item, "_presentation_series_index", 0))
                      if role in ("data", "fit") else None)
            if isinstance(item, pg.ScatterPlotItem):
                item.setSize(self._style.point_size)
                if colour:
                    tint = QtGui.QColor(colour)
                    tint.setAlpha(_POINT_ALPHA)
                    item.setBrush(pg.mkBrush(tint))
            elif isinstance(item, pg.FillBetweenItem):
                if colour:
                    tint = QtGui.QColor(colour)
                    tint.setAlpha(60)
                    item.setBrush(pg.mkBrush(tint))
            elif isinstance(item, (pg.PlotDataItem, pg.IsocurveItem)):
                existing = item.opts.get("pen") if isinstance(item, pg.PlotDataItem) else item.pen
                if existing is not None:
                    pen = pg.mkPen(existing)
                    if colour:
                        pen.setColor(QtGui.QColor(colour))
                    pen.setWidthF(self._style.line_width + getattr(item, "_presentation_width_extra", 0))
                    item.setPen(pen)
                if isinstance(item, pg.PlotDataItem) and colour and item.opts.get("fillBrush") is not None:
                    brush = pg.mkBrush(item.opts["fillBrush"])
                    tint = QtGui.QColor(colour)
                    tint.setAlpha(brush.color().alpha())
                    item.setFillBrush(pg.mkBrush(tint))
                if isinstance(item, pg.PlotDataItem) and item.opts.get("symbol") is not None:
                    item.setSymbolSize(max(4, self._style.point_size - 2))
                    if colour:
                        item.setSymbolBrush(pg.mkBrush(colour))
        self._apply_series_styles()
        self._paint_event_point_selection()

    def _committed_legend_names(self) -> dict:
        sources = getattr(self, "_presentation_sources", ())
        entries = [{"label": self._style.label_overrides.get(key, label) or label,
                    "channel_name": label, "is_segment": segment}
                   for key, label, segment in sources]
        disambiguate_labels(entries)
        return {source[0]: entry["label"] for source, entry in zip(sources, entries)}

    def _restyle_committed_legend(self) -> None:
        entries = getattr(self, "_presentation_legend", {})
        names = self._committed_legend_names()
        self._drop_legend()
        for entry_id, (item, original, kind, key) in entries.items():
            label = str(original)
            if isinstance(original, _LegendText):
                name = names.get(key, original.name) if kind == "series" else original.name
                label = str(self._legend_entry(name, original.count))
            elif kind == "series":
                label = names.get(key, label)
            elif kind == "fit":
                if getattr(item, "_presentation_fit_family", None) is not None:
                    # A one-source fit row is "fit (…)", unnamed; an overlay's is named.
                    named = getattr(item, "_presentation_fit_named", True)
                    label = self._fit_legend_label(item, names.get(key, "") if named else "",
                                                   key)
                elif " fit:" in label and key in names:
                    label = names[key] + " fit:" + label.split(" fit:", 1)[1]
            self._legend_offered[entry_id] = str(label)
            if self._style.show_legend and self._legend_included(entry_id):
                legend = self._ensure_legend()
                legend.addItem(item, self._legend_markup(label, _legend_style_key(kind, key)))
                self._legend_row_added(legend)
        self._sync_legend_entry_list()
        self._appearance_widget.set_legend_available(bool(entries))
        self._refresh_committed_legend_disclosure(names)

    def _refresh_committed_legend_disclosure(self, names) -> None:
        """Rename disclosure from the shown source identities, never pending inputs."""
        groups = {}
        for key, label, _segment in getattr(self, "_presentation_sources", ()):
            shown = self._style.label_overrides.get(key, label) or label
            groups.setdefault(shown, []).append((key, label))
        notes = []
        for label, sources in groups.items():
            if len(sources) < 2:
                continue
            which = ", ".join(original for _key, original in sources)
            apart = ", ".join(f"“{names[key]}”" for key, _original in sources)
            notes.append(
                f"LEGEND: {len(sources)} of the drawn sources are both called “{label}” "
                f"({which}), so the legend and any saved image name them {apart} "
                "— two curves under one name cannot be read. Retype one of them in "
                "the Sources list to name them apart.")
        old = self.details_text()
        lines = [line for line in old.splitlines() if not line.startswith("LEGEND:")]
        updated = "\n".join([*lines, *notes])
        if updated != old:
            self._set_details(updated)

    def _set_presentation_annotation(self, attribute, lines, visible, *, bottom=False):
        item = getattr(self, attribute, None)
        text = "<br>".join(_html.escape(str(line)) for line in lines if str(line).strip())
        if not visible or not text:
            if item is not None:
                item.setParentItem(None)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                setattr(self, attribute, None)
            return
        colour = self._msg_hex() if bottom else self._style.fit_color
        size = max(6, self._style.font_size_pt - 2) if bottom else self._style.font_size_pt
        if item is None:
            item = pg.LabelItem(justify="left")
            item.setParentItem(self._plot.vb)
            item.anchor(itemPos=(0, 1 if bottom else 0), parentPos=(0, 1 if bottom else 0),
                        offset=(8, -6) if bottom else (10, 8))
            item.setZValue(30)
            setattr(self, attribute, item)
        options = {"color": colour, "size": f"{size:g}pt",
                   "family": html_font_family(self._style.font_family)}
        if bottom:
            options.update(label_args(self._style.effective_text_style("annotation.provenance")))
        item.setText(text, **options)

    def _restyle_committed_decorations(self) -> None:
        if hasattr(self, "_nonlinear_fitter"):
            self._nonlinear_fitter.set_band_visible(self._style.show_ci_band)
        names = self._committed_legend_names()
        # The fit's numbers: reformatted from the cached readouts, never refitted, and
        # never written into the title.
        self._draw_fit_label()
        self._draw_limit_label()
        provenance = getattr(self, "_presentation_provenance", [])
        keyed = getattr(self, "_presentation_provenance_by_source", [])
        if keyed:
            multi = len(getattr(self, "_presentation_sources", ())) > 1
            provenance = [(f"{names.get(key, key)}: " if multi else "") + text
                          for key, text in keyed]
        self._set_presentation_annotation("_prov_item", provenance,
                                          self._style.show_provenance, bottom=True)
        if self._style.show_ci_band and not self._band_items:
            for grid, lo, hi, colour, _note, role, *metadata in getattr(self, "_presentation_bands", []):
                self._add_cached_band(grid, lo, hi, colour, role, metadata[0] if metadata else None)
        for item in self._band_items:
            item.setVisible(self._style.show_ci_band)
        residual = getattr(self, "_presentation_residual", None)
        if self._style.show_residuals and residual is not None and not self._resid.items:
            self._render_cached_residual(*residual)
        self._resid_widget.setVisible(bool(self._style.show_residuals and residual is not None))
        # The first opt-in uses precomputed geometry, but must still carry its
        # original statistical qualifications into the full methods record.
        notes = []
        if self._style.show_ci_band:
            notes.extend(entry[4] for entry in getattr(self, "_presentation_bands", []))
        if self._style.show_residuals:
            notes.append(getattr(self, "_presentation_residual_note", ""))
        details = self.details_text()
        additions = [note for note in dict.fromkeys(notes) if note and note not in details]
        if additions:
            self._set_details("\n".join([details, *additions]))

    def _presentation_changed(self) -> None:
        if self._updating:
            return
        self._style = _project_style_settings(self._style, edit=True)
        view = self._plot.getViewBox()
        ranges = [list(pair) for pair in view.viewRange()]
        auto = list(view.autoRangeEnabled())
        view.disableAutoRange()
        try:
            self._apply_presentation_changes()
        finally:
            view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
            for axis, enabled in enumerate(auto):
                if enabled:
                    view.enableAutoRange(axis=axis, enable=enabled)

    def _apply_presentation_changes(self) -> None:
        """Restyle existing text/canvas items without reconstructing the result.

        Do not consult pending axes, source selections, or fits here. Their run
        gate still owns when the numerical picture changes. Cached warnings remain
        in Details regardless of the authored title. This path never
        clears items, sets plot ranges, or touches numerical/deferred-note caches.
        Fit decorations consume cached geometry prepared at the genuine draw.
        """
        if self._updating:
            return
        if hasattr(self, "_appearance_widget"):
            self._updating = True
            try:
                self._appearance_widget.set_state(self.appearance_state())
            finally:
                self._updating = False
        if self._plot_armed and self._message_item is None:
            self._restyle_committed_legend()
            self._restyle_committed_decorations()
        self._restyle_committed_items()
        # The heatmap's colour map and its bar's ink: cached image, new LUT, no re-bin.
        self._style_heatmap_bar()
        self._apply_background()
        self._apply_fonts()
        if self._plot_armed and self._message_item is None:
            self._apply_title(self._title_presentation[0], self._title_presentation[2])
            for side, auto in self._axis_auto_labels.items():
                override = self._style.x_label if side == "bottom" else self._style.y_label
                self._plot.setLabel(side, rich_text(override.strip()) if override else auto,
                                    **self._label_style())
            self._apply_element_styles()
        elif self._message_item is not None:
            self._message_item.setColor(self._msg_hex())

        legend = getattr(self._plot, "legend", None)
        if legend is not None:
            fg = QtGui.QColor(self._fg_hex())
            backing = QtGui.QColor(self._bg_spec())
            backing.setAlpha(200)
            border = QtGui.QColor(fg)
            border.setAlpha(90)
            legend.setLabelTextColor(fg)
            legend.setBrush(pg.mkBrush(backing))
            legend.setPen(pg.mkPen(border))
            pt = float(self._style.legend_font_size_pt) or max(6, self._style.font_size_pt - 3)
            legend.setLabelTextSize(f"{max(1, pt)}pt")
        for item, size, colour in (
                (getattr(self, "_prov_item", None), max(6, self._style.font_size_pt - 2),
                 self._msg_hex()),):
            if item is not None:
                item.setText(item.text, **{"size": f"{size:g}pt", "color": colour,
                    **label_args(self._style.effective_text_style("annotation.provenance"))})
        if hasattr(self, "_resid"):
            self._style_residual_axes()
        # The original history/coalescing hook also records public setter calls
        # and preserves workspaceChanged without creating a new run gesture.
        self._history_settle()

    # -- sources (public: owner + tests) ---------------------------------------------------------
    def source_labels(self) -> list[str]:
        """Every source on offer, in the order the picker lists them."""
        return [s.label for s in self._sources.values()]

    def sources(self) -> list[PlotSource]:
        return list(self._sources.values())

    def _resolve_key(self, name: str) -> str | None:
        """A source key from either a key or a display label (the friendly form)."""
        name = str(name)
        if name in self._sources:
            return name
        return next((k for k, s in self._sources.items() if s.label == name), None)

    def set_source(self, name: str) -> None:
        """Draw EXACTLY this one source (the single-source case). Accepts a key or a
        label; an unknown name leaves the selection alone."""
        self.set_sources([name])

    def set_sources(self, names: Sequence[str]) -> None:
        """Draw exactly these sources, overlaid, in picker order.

        Accepts keys or labels. Unknown names are ignored, and a request that
        resolves to nothing leaves the current selection alone rather than blanking
        the window."""
        keys = [k for k in (self._resolve_key(n) for n in names) if k]
        if not keys:
            return
        self._updating = True
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in keys else Qt.Unchecked)
        self._updating = False
        self._on_sources_changed()

    def selected_source_keys(self) -> list[str]:
        """The checked source keys, in picker order."""
        if not hasattr(self, "_source_list"):
            return []
        out = []
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            if item.checkState() == Qt.Checked:
                key = item.data(Qt.UserRole)
                if key in self._sources:
                    out.append(key)
        return out

    def selected_sources(self) -> list[PlotSource]:
        """The checked :class:`~nanoscope.core.sources.PlotSource` objects, in picker order."""
        return [self._sources[k] for k in self.selected_source_keys()]

    def _primary_key(self) -> str:
        """The FIRST checked source — the one clustering/PCA and the single-source
        accessors operate on."""
        keys = self.selected_source_keys()
        return keys[0] if keys else ""

    def primary_source(self) -> PlotSource | None:
        key = self._primary_key()
        return self._sources.get(key) if key else None

    def current_source_name(self) -> str:
        """The PRIMARY source's label ("" when nothing is checked)."""
        src = self.primary_source()
        return src.label if src is not None else ""

    def set_normalization(self, mode: str) -> None:
        """``count`` / ``density`` / ``probability per bin``.

        Setting it explicitly pins the choice: the automatic switch to density on a
        second source no longer overrides the user.

        THE OLD SPELLING STILL WORKS, AND ONLY THE OLD SPELLING. ``density (area = 1)``
        is what every script, notebook and pre-S6-03 test says, and it names a mode that
        still exists -- so it is translated (:func:`_canonical_norm`, the same reading of
        :data:`_LEGACY_NORMS` the value itself does) rather than refused. Measured
        without the translation: this call raises ``unknown normalization 'density
        (area = 1)'; this window offers ['count', 'density', 'probability per bin']``,
        which is the right answer for a typo and the wrong one for a name this app itself
        wrote into the user's own files last week. Anything else unmatched still raises
        with the offered items named -- the invariant :mod:`nanoscope.gui.combos` exists
        to hold, and the reason this setter has never used ``setCurrentText``.

        THIS IS THE LOUD DOOR, and it is deliberately the only one. A bare string on its
        way to a widget is a caller TYPING a mode, so a name that is not on offer is a
        mistake worth raising on. A whole ``PlotConfig`` arriving from a file or a
        template is not typing -- refusing the entire recall over one field would throw
        away the forty settings that ARE installable -- so that path installs what it can
        and records what it could not (:class:`_NormRefusal`). Two shapes of caller, two
        answers, and neither of them is silence.
        """
        self._norm_user_set = True
        _choose_combo_item(self._norm_combo, _canonical_norm(mode), "normalization")

    def normalization(self) -> str:
        return self._norm_combo.currentText()

    def set_histogram_colour_by(self, mode: str) -> None:
        """``None`` or ``Cluster`` — the histogram's cluster split. Ignored while an
        overlay is up: colour means SOURCE then (see :meth:`_sync_cluster_split_toggle`)."""
        if not self._hist_colour_by.isEnabled():
            return
        _choose_combo_item(self._hist_colour_by, mode, "histogram colour-by")

    def histogram_colour_by(self) -> str:
        return self._hist_colour_by.currentText()

    def set_plot_type(self, name: str) -> None:
        """``Histogram`` / ``Scatter`` / ``PCA``. RAISES on anything else -- see
        :func:`nanoscope.gui.combos.choose_combo_item`."""
        _choose_combo_item(self._plot_type_combo, name, "plot type")

    def set_time_unit(self, unit: str) -> None:
        """Display unit for time columns ('s' / 'ms' / 'µs'). DISPLAY ONLY — see
        :data:`_TIME_UNITS`: it rescales the axis, never the data or any fit.

        RAISES on a unit this window does not offer: ``set_time_unit("milliseconds")``
        used to leave the axis in seconds and report seconds, silently."""
        _choose_combo_item(self._time_unit, unit, "time unit")

    def time_unit(self) -> str:
        return self._time_unit.currentText()

    def set_histogram_column(self, col: str) -> None:
        """The column to histogram. RAISES on a column this table does not offer --
        ``set_histogram_column("frac")`` used to leave ``depth_pA`` selected and report
        picoamps under a request for a ratio."""
        _choose_combo_item(self._hist_col, col, "histogram column")

    def set_bins(self, n: int) -> None:
        self._bins.setValue(int(n))

    def set_log_x(self, on: bool) -> None:
        self._hist_logx.setChecked(bool(on))

    def set_log_y(self, on: bool) -> None:
        self._hist_logy.setChecked(bool(on))

    def set_log_state(self, log_x: bool | None = None, log_y: bool | None = None) -> None:
        """Log the CURRENT plot's axes, whichever page that is. ``None`` leaves one alone.

        The page-agnostic setter beside the three page-specific ones: a caller that means
        "log the x axis of what is on screen" no longer has to choose between
        ``set_log_x``, ``set_scatter_log`` and ``set_kde_log`` by reading the plot type
        first. One config write, so it is one draw and one round trip — see
        :meth:`PlotConfig.with_log_state`, which is where the per-page storage is
        resolved. A no-op on PCA, whose scores are signed.
        """
        self.apply_config(self.config().with_log_state(log_x=log_x, log_y=log_y))

    def set_axis_range(self, *, xmin: float | None = None, xmax: float | None = None,
                       ymin: float | None = None, ymax: float | None = None) -> None:
        """Pin axis limits in DATA units (None leaves that edge auto). Applies to the current plot.

        The BOXES read in the unit the axis displays (a time axis follows the Time-unit
        combo), so this converts data units into that unit before filling them — the
        programmatic contract stays data units while the typed-in contract matches the
        ticks. _apply_axis_range then converts back, so the round trip is exact."""
        fx, fy = self._axis_unit_factors()
        for key, val, f in (("xmin", xmin, fx), ("xmax", xmax, fx),
                            ("ymin", ymin, fy), ("ymax", ymax, fy)):
            self._range_edits[key].setText("" if val is None else repr(float(val) * f))
        self._apply_axis_range()

    def set_fit(self, kind: str) -> None:
        """Choose the histogram fit overlay, by combo label or an unambiguous prefix.

        RAISES on a name this combo does not offer. ``QComboBox.setCurrentText`` on a
        non-editable combo is a no-op for unmatched text, so ``set_fit("gamma")`` --
        one word short of ``"gamma (peaked)"`` -- left the previous model selected and
        the panel went on reporting THAT model's numbers, with nothing raised and
        nothing on screen to say so. It cost a full run of wrong numbers in the
        apo-ferritin report (Low-21) before a read-back check caught it.

        APPLIED IMMEDIATELY, unlike the dropdown. The interactive Apply gate (owner
        note #12) exists so that a user BROWSING the model list does not fit five
        distributions on the way past; a caller that names the model it wants has
        already made that decision, and a setter that left the plot showing the
        previous model until someone clicked a button would be the silent no-op this
        method's own docstring exists to prevent.
        """
        _choose_combo_item(self._fit_combo, kind, "fit")
        self._apply_fit()

    def set_fit_components(self, k: int) -> None:
        """Mixture components for the histogram fit. Applied at once — see :meth:`set_fit`."""
        self._fit_k.setValue(int(k))
        self._apply_fit()

    def set_scatter_columns(self, x: str, y: str) -> None:
        """Both scatter axes. RAISES on either name -- a silently-ignored y column is
        a plot of the wrong quantity under the right label."""
        _choose_combo_item(self._scatter_x, x, "scatter x column")
        _choose_combo_item(self._scatter_y, y, "scatter y column")

    def set_scatter_log(self, x: bool, y: bool) -> None:
        self._scatter_logx.setChecked(bool(x))
        self._scatter_logy.setChecked(bool(y))

    def set_kde_columns(self, x: str, y: str) -> None:
        """Both KDE axes. RAISES on either name — same contract as the scatter's."""
        _choose_combo_item(self._kde_x, x, "KDE x column")
        _choose_combo_item(self._kde_y, y, "KDE y column")

    def set_kde_log(self, x: bool, y: bool) -> None:
        self._kde_logx.setChecked(bool(x))
        self._kde_logy.setChecked(bool(y))

    def set_kde_overlay(self, on: bool) -> None:
        """The histogram's 1-D KDE overlay. APPLIED AT ONCE, for :meth:`set_fit`'s
        reason: a caller that names the setting has made the decision the Plot gate
        exists to wait for. A hand tick goes behind the gate instead."""
        self._hist_kde.setChecked(bool(on))

    def set_heatmap_columns(self, x: str, y: str) -> None:
        """Both heatmap axes. RAISES on either name — same contract as the scatter's."""
        _choose_combo_item(self._heatmap_x, x, "heatmap x column")
        _choose_combo_item(self._heatmap_y, y, "heatmap y column")

    def set_heatmap_log(self, x: bool, y: bool) -> None:
        self._heatmap_logx.setChecked(bool(x))
        self._heatmap_logy.setChecked(bool(y))

    def set_heatmap_bins(self, n: int) -> None:
        self._heatmap_bins.setValue(int(n))

    def set_heatmap_colormap(self, name: str) -> None:
        """The heatmap's colour map (appearance only). RAISES on a name not offered."""
        _choose_combo_item(self._heatmap_cmap, name, "heatmap colormap")

    def set_colour_by(self, mode: str) -> None:
        """Scatter colour channel. RAISES on an unknown mode."""
        _choose_combo_item(self._colour_by, mode, "colour-by")

    def set_scatter_fit(self, kind: str) -> None:
        """Choose the scatter best-fit overlay.

        Accepts the exact combo labels (``"none"``, ``"linear"``,
        ``"power law (log-log)"``) and the friendly short forms ``"power law"`` /
        ``"power"`` for the log-log fit. RAISES on anything else — see
        :meth:`set_fit`: this used to fall through to ``setCurrentText`` with the raw
        string, which silently keeps the previous selection."""
        k = str(kind).strip().lower()
        if "power" in k:
            resolved = _SFIT_POWER
        elif "linear" in k:
            resolved = _SFIT_LINEAR
        else:
            resolved = kind
        _choose_combo_item(self._scatter_fit, resolved, "scatter fit")
        self._apply_scatter_fit()

    def set_feature_selection(self, cols: Sequence[str]) -> None:
        self._features.set_checked({str(c) for c in cols})
        self._on_features_changed()

    def feature_selection(self) -> list[str]:
        return self._features.checked()

    def set_cluster_method(self, method: str) -> None:
        """Clustering algorithm. RAISES on a name this window does not offer."""
        _choose_combo_item(self._method_combo, method, "cluster method")

    def set_cluster_k(self, k: int | None) -> None:
        """Pin the cluster count; ``None`` selects Auto-BIC (GMM only)."""
        if k is None:
            _choose_combo_item(self._k_mode, _K_AUTO, "cluster k mode")
        else:
            _choose_combo_item(self._k_mode, _K_FIXED, "cluster k mode")
            self._k_spin.setValue(int(k))

    # -- style setters (used by the owner + tests) -----------------------------------------------
    def set_title(self, title: str) -> None:
        """Override the plot title; an empty string restores the automatic title."""
        self._title_edit.setText(str(title))

    def set_font_size(self, pt: int) -> None:
        self._font_size.setValue(float(pt))

    def set_font_family(self, family: str) -> None:
        """Font family for the plot's text. ``""`` restores the default, Arial.

        The empty string is not "no change": it is the value the Font row's first entry
        stands for, and before S6-01 this setter could not reach it --
        ``setCurrentFont(QFont(""))`` left the combo showing whatever it had been showing
        while the style went blank, so the panel and the figure disagreed.
        """
        self._style = _dc_replace(self._style, font_family=str(family))
        self._select_font_family(str(family))
        self._presentation_changed()

    def set_data_color(self, hex_color: str) -> None:
        self._style = _dc_replace(self._style, data_color=str(hex_color))
        self._paint_swatch(self._data_swatch, str(hex_color))
        self._presentation_changed()

    def set_fit_color(self, hex_color: str) -> None:
        self._style = _dc_replace(self._style, fit_color=str(hex_color))
        self._paint_swatch(self._fit_swatch, str(hex_color))
        self._presentation_changed()

    def set_point_size(self, n: float) -> None:
        self._point_size.setValue(float(n))

    def set_line_width(self, n: float) -> None:
        self._line_width.setValue(float(n))

    def set_background(self, mode: str) -> None:
        """``"dark"`` or ``"light"`` (light = white canvas + dark ink, for publication)."""
        _choose_combo_item(
            self._bg_combo, "Light" if str(mode).lower() == "light" else "Dark",
            "background")

    def set_export_size(self, w: int, h: int) -> None:
        """PNG export size in px; 0 on either axis means native / follow aspect."""
        self._export_w.setValue(int(w))
        self._export_h.setValue(int(h))

    def style(self) -> StyleSettings:
        """The current style, as a VALUE the caller may keep.

        This hands out the live object rather than a copy, and that is now safe --
        confirmed by :func:`test_style_snapshot_does_not_see_later_changes`. Every
        change rebinds ``self._style`` to a ``replace()`` copy instead of writing
        into the instance, and the two mapping fields are :class:`FrozenMap`, so a
        style already returned from here can no longer be reached by any later edit.
        Before the freeze this same line leaked the window's mutable state: a
        snapshot taken for an undo stack would have tracked the live window.
        """
        return self._style

    # -- results (used by tests) -----------------------------------------------------------------
    def cluster_result(self) -> populations.ClusterResult | None:
        return self._cluster_result

    def pca_result(self) -> populations.PCAResult | None:
        return self._pca_result

    def fit_result(self):
        """The :class:`~nanoscope.analysis.stats.FitResult` behind the PRIMARY overlay.

        None when no fit is selected or the last one failed. Exposed so the exact
        parameters this window computed can be compared with the Fits panel's --
        the two must agree to floating point on the same events. With an overlay up
        use :meth:`fit_results` instead: each source is fit separately, and there is
        no such thing as "the" fit."""
        return self._last_fit

    def fit_results(self) -> dict:
        """``{source key: FitResult}`` for the current draw — ONE FIT PER SOURCE.

        Two sources are never pooled into a single fit, and each one's ``t_min`` /
        filter-limited exclusion / right-truncation state comes from its OWN
        provenance. That is what makes overlaid taus comparable numbers rather than
        one recording's detection floor imposed on another's dwell times."""
        return dict(self._fits)

    def details_text(self) -> str:
        return self._details.toPlainText()

    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def legend_labels(self) -> list[str]:
        """The legend entry texts currently on the plot (empty when there is none)."""
        legend = self._plot.legend
        if legend is None:
            return []
        return [label.text for _sample, label in legend.items]

    # -- rendering -------------------------------------------------------------------------------
    def _render_workspace_snapshot(self) -> None:
        """Paint restored results without running an analysis kernel.

        Project restore installs the saved :class:`PlotConfig`, PCA/clustering
        objects and distribution-fit objects before calling this seam.  The normal
        drawing code remains the authority for bars, points, axes and styling, but
        its three computational exits are fenced while this method is active:

        * histogram overlays evaluate the already-restored ``FitResult``;
        * a scatter regression is left pending because no fitted regression result
          is stored yet, and the gate's ● says so; and
        * KDE is left pending because its sampled density grid is not a saved result.

        PCA and cluster colouring are direct projections of the restored arrays and
        therefore need no special case.  This method intentionally bypasses
        :meth:`_armed_redraw`: opening a project is reconstruction, not a Run gesture,
        undo entry, or dirty notification.
        """
        saved_fits = dict(self._fits)
        saved_columns = dict(self._fit_columns)
        saved_last = self._last_fit
        was_updating = self._updating
        self._workspace_snapshot_fits = (saved_fits, saved_columns)
        self._workspace_snapshot_rendering = True
        self._kde_pending_after_snapshot = False
        self._scatter_fit_pending_after_snapshot = False
        # Opening a project is never a KDE run, whatever page it opens on: the restored
        # picture has no curve, so the applied state -- and with it the Figure Library's
        # `figure_applied_config` -- says none until Plot is pressed.
        self._applied_kde_overlay = False
        self._updating = True
        try:
            self._redraw()
            # A restored, successfully painted snapshot is the picture the controls
            # describe.  Update only the run-gate projection; never history/dirty.
            self._plot_armed = self._message_item is None
            self._stale = False
            self._stale_reason = ""
            self._stale_reasons = []
            if self._plot_armed and self._kde_pending_after_snapshot:
                # ...EXCEPT a KDE overlay, which the draw declined to estimate: the
                # controls ask for a curve the picture does not have, and that is
                # exactly what the gate's ● is for.
                self._stale = True
                self._stale_reason = "the KDE overlay"   # Details says it was not recomputed
                self._stale_reasons = [self._stale_reason]
            elif self._plot_armed and self._scatter_fit_pending_after_snapshot:
                # ...and an applied scatter fit, for the same reason: the saved figure
                # had its line and its fit label, and this picture has neither.
                self._stale = True
                self._stale_reason = "the scatter fit"
                self._stale_reasons = [self._stale_reason]
            self._sync_fit_apply_buttons()
            self._sync_run_state()
        finally:
            # ``_clear_plot`` correctly retires the previous picture's result maps
            # during an ordinary draw.  A workspace draw is different: these are the
            # frozen project values it is *rendering*, so put the complete maps back
            # (including a fit whose source is temporarily unavailable).
            self._fits = saved_fits
            self._fit_columns = saved_columns
            if saved_last is not None:
                self._last_fit = saved_last
            elif saved_fits:
                primary = next((key for key in self.selected_source_keys()
                                if key in saved_fits), None)
                self._last_fit = saved_fits.get(primary)
            self._updating = was_updating
            self._workspace_snapshot_rendering = False
            self._workspace_snapshot_fits = None

    def _overlay_workspace_fit(
        self, kind: str, col: str, v: np.ndarray, edges: np.ndarray, log_x: bool,
        src: PlotSource | None = None, colour: str | None = None, who: str = "",
    ) -> tuple[_FitSummary, list[str]]:
        """Draw one frozen project fit; do not estimate, test, or bootstrap it.

        ``FitResult.predict`` evaluates the saved parametric density and is the same
        closure-free reconstruction used by project loading.  Everything that would
        derive a *new* number from the sample (MLE, component search, KS/rival tests,
        residual statistics) is deliberately absent here.
        """
        snapshot = self._workspace_snapshot_fits
        if snapshot is None:  # defensive: this helper is only a restore seam
            raise RuntimeError("no restored fit snapshot is active")
        fits, columns = snapshot
        src = src if src is not None else self.primary_source()
        key = "" if src is None else src.key
        fit = fits.get(key)
        if fit is None:
            raise RuntimeError("the saved fit for this source is unavailable")
        fitted_column = str(columns.get(key, ""))
        if fitted_column and fitted_column != col:
            raise RuntimeError(
                f"the saved fit belongs to {fitted_column!r}, not the selected {col!r}")

        if log_x:
            grid = np.geomspace(edges[0], edges[-1], 400)
            du = float(np.log10(edges[1]) - np.log10(edges[0]))
            width = grid * np.log(10.0) * du
        else:
            grid = np.linspace(edges[0], edges[-1], 400)
            width = float(edges[1] - edges[0])

        # A Gaussian mixture fitted to an all-negative column is a magnitude model;
        # the ordinary fitter reflects it onto that column.  Preserve the picture
        # using only the already-saved parameters.
        params = dict(getattr(fit, "params", {}) or {})
        all_negative = bool(v.size) and bool(np.all(np.asarray(v) < 0))
        at = np.abs(grid) if all_negative and any(
            str(name).startswith("mu") for name in params) else grid
        density = np.asarray(fit.predict(at), dtype=np.float64)
        expected = (int(fit.n) * density * width
                    * self._fit_scale(edges, width, int(v.size)))
        pen = (pg.mkPen(colour, width=self._style.line_width + 1, style=Qt.DashLine)
               if who else self._fit_pen())
        item = pg.PlotDataItem(x=grid, y=expected, pen=pen)
        self._plot.addItem(item)
        self._fit_items.append(item)
        # The live draw's roles, so a Fit colour or Line width change after a reopen
        # restyles this curve as it would a fresh one.
        item._presentation_role = "category" if who else "fit"
        self._tag_series(item, "fit", src.key if src is not None else "fit",
                         f"{src.label if src is not None else 'Data'} fit", colour=colour if who else None)
        item._presentation_width_extra = 1
        if self._fit_item is None:
            self._fit_item = item
        if self._last_fit is None:
            self._last_fit = fit
        if src is not None:
            self._fits[src.key] = fit
            self._fit_columns[src.key] = col

        # Match the ordinary headline parameter, but read every value from the saved
        # fit.  The fit label and the legend row are built by the SAME helpers as a
        # fresh fit's (``_fit_parts``, ``_fit_legend_label``), so a reopened project
        # draws the figure that was saved: every parameter, n, BIC and KS in the label,
        # and a number-free "fit (…)" row. The KS numbers are the ones ``stats`` stamped
        # on the fit (``meta['gof']``), saved with it -- read, never recomputed.
        head = {
            _FIT_EXP: "tau", _FIT_GMM: "mu", _FIT_GAMMA: "k",
            _FIT_INVGAUSS: "mu", _FIT_EXP_GAMMA: "k",
        }.get(kind, "")
        keys = [name for name in params if str(name).partition("_")[0] == head]
        if not keys and params:
            head = str(next(iter(params))).partition("_")[0]
            keys = [name for name in params if str(name).partition("_")[0] == head]
        param_unit = fit.unit_of(keys[0]) if keys else ""
        scale, unit = self._display_unit(col, param_unit)
        shown = ", ".join(f"{float(params[name]) * scale:.4g}" for name in keys)
        label = _param_symbol(head) if head else "fit"
        if all_negative and head == "mu":
            label = f"μ(|{col}|)"
        unit_text = f" {unit}" if unit else ""
        summary = f"{label} = {shown}{unit_text}   (n = {int(fit.n)})"
        stamped = (getattr(fit, "meta", None) or {}).get("gof")
        gof = stamped if isinstance(stamped, dict) and "ks" in stamped else None
        full = self._full_fit_summary(fit, col, gof)
        legend_who = self._legend_name(src) if (who and src is not None) else who
        item._presentation_fit_family = kind
        item._presentation_fit_named = bool(legend_who)
        self._legend_add(item, self._fit_legend_label(item, legend_who, key),
                         kind="fit", key=key)
        plain_summary = summary
        rejected = bool(gof) and not gof.get("describes_data", True)
        if rejected:
            summary = f"{summary} — {_KS_REJECTED_TAG}"

        notes = [
            str(getattr(fit, "model", "saved fit")),
            (f"BIC = {float(fit.bic):.6g}   AIC = {float(fit.aic):.6g}   "
             f"(restored snapshot; not recomputed)"),
        ]
        ci_line = self._ci_note(fit, col)
        if ci_line:
            notes.append(ci_line)
        if gof and gof.get("note"):
            notes.append(f"{gof['note']} (as saved with the fit)")
        notes.extend(str(note) for note in getattr(fit, "meta", {}).get("warnings", ()))
        band_note = self._draw_ci_band(
            fit, grid, at, int(fit.n) * width * self._fit_scale(edges, width, int(v.size)),
            colour or self._style.fit_color, "category" if who else "fit")
        if band_note:
            notes.append(band_note)
        residual = getattr(fit, "meta", {}).get("saved_residual_geometry")
        if residual and not self._presentation_residual_ready:
            res = dict(residual["result"])
            self._presentation_residual = (res, bool(residual.get("log_x", log_x)))
            self._presentation_residual_note = str(res.get("note", ""))
            self._presentation_residual_ready = True
            if self._style.show_residuals:
                self._render_cached_residual(*self._presentation_residual)
                notes.append(self._presentation_residual_note)
        result = _FitSummary(summary, full, rejected=rejected, plain=plain_summary)
        result.parts = self._fit_parts(fit, col, gof, headline=f"{label} = {shown}{unit_text}",
                                       headline_keys=keys)
        return result, notes

    def _clear_plot(self) -> None:
        self._nonlinear_drawn_datasets = []
        if hasattr(self, "_nonlinear_fitter"):
            self._nonlinear_fitter.clear_graphics()
            if self._plot_type() != SCATTER:
                self._nonlinear_fitter.state = None
        self._lasso_generation += 1
        if hasattr(self, "_lasso"):
            self._lasso.cancel("Lasso cancelled because the plotted result changed.")
            self._lasso.set_enabled(False)
        self._figure_sources = {}
        self._presentation_legend = {}
        self._presentation_sources = ()
        self._presentation_fit_lines = []
        self._presentation_full_variants = []
        self._presentation_scatter_variants = []
        self._presentation_provenance = []
        self._presentation_provenance_by_source = []
        #: ``(source key, FitReadout)`` per fitted source of the committed draw: what the
        #: fit label (and, with In legend instead, the legend's fit rows) is drawn from,
        #: so an appearance change reformats the numbers without refitting.
        self._presentation_readouts = []
        self._presentation_bands = []
        self._presentation_residual = None
        self._presentation_residual_note = ""
        self._presentation_residual_ready = False
        self._figure_applied_config = None
        self._plot.clear()
        reset_plot_history(self._plot)   # new geometry: old zoom ranges mean nothing now
        self._kde_view_frame = None
        self._plot.setTitle(None)
        self._title_presentation = (None, None, None)
        self._axis_auto_labels = {}
        self._drop_legend()
        self._hist_item = self._fit_item = self._scatter_item = None
        self._scatter_fit_item = None
        self._hist_items = []
        self._fit_items = []
        self._scatter_items = []
        self._scatter_fit_items = []
        self._event_point_items = {}
        self._selected_event_rows = {}
        self._message_item = None
        self._limit_region = None
        self._limit_line = None
        # What the LAST draw offered the legend belongs to the last draw. Emptied here,
        # with the plot items, so the Style group's checklist can only ever list rows
        # the picture on screen actually produced.
        self._legend_offered = {}
        # The fit label is a ViewBox child that lives as long as the window (plot.clear()
        # cannot reach it); a draw without a fit leaves it empty, which hides it.
        if getattr(self, "_fit_label_item", None) is not None:
            self._fit_label_item.apply_state(self._style.fit_label_state(), [])
        # Parented to the ViewBox, not added through addItem — plot.clear() cannot
        # reach it, so it is detached explicitly (same reason _drop_legend exists).
        if getattr(self, "_prov_item", None) is not None:
            self._prov_item.setParentItem(None)
            _sc = self._prov_item.scene()
            if _sc is not None:
                _sc.removeItem(self._prov_item)
        self._prov_item = None
        self._remove_heatmap_bar()
        self._band_items = []
        self._clear_residual_strip()
        self._last_fit = None
        self._fits = {}
        self._fit_columns = {}

    # -- legend ----------------------------------------------------------------------------------
    def _draw_fit_label(self) -> None:
        """The committed fits' numbers in the fit label (``fit_label.FitLabelItem``).

        THE ONE PLACE A FIT'S NUMBERS ARE DRAWN ON THE FIGURE. They used to be the plot's
        automatic title (and, with the old Fit values on plot box, a second copy), so a typed
        title removed them from the figure and every export. The label is its own
        element: one line per fitted source, prefixed with the figure's name for it and
        in its colour, top-left unless dragged, and on whenever a fit is applied.

        Rebuilt from the cached readouts, never by refitting, so the restyle path can call
        it on every appearance toggle. The label's Numbers row (``FitLabelState.fields``)
        chooses which numbers it shows: it drops a plain NUMBER, never a refusal, which
        :func:`fit_label.format_fit_label` prints whenever R² is asked for. The one other
        thing that writes into the label is the Residual strip box, whose χ²/dof is joined
        to the equation (:meth:`_shown_readout`), so unticking Equation takes it away too.
        """
        item = getattr(self, "_fit_label_item", None)
        if item is None:
            return
        state = self._style.fit_label_state()
        readouts_in = getattr(self, "_presentation_readouts", [])
        names = self._committed_legend_names()
        if not names and len(readouts_in) > 1:
            # Mid-draw the committed sources are not recorded yet; the live resolver
            # names the same ticked sources then. Only a draw reaches here: an
            # appearance change always has its committed names, and must not collect
            # sources (the restyle path never does).
            names = self._legend_names()
        drawn_rows = self._fit_rows_in_legend() if state.in_legend else set()
        readouts = [self._shown_readout(key, readout, names)
                    for key, readout in readouts_in
                    # In legend instead: a fit whose row is drawn carries its numbers there.
                    if key not in drawn_rows]
        resolved = self._style.effective_text_style(FIT_LABEL_KEY)
        draw_state = _dc_replace(state, in_legend=False,
                                 size_pt=resolved.size_pt or float(self._style.font_size_pt),
                                 color=resolved.color)
        style = _dc_replace(resolved, text=None, size_pt=None, color=None)
        item.apply_state(draw_state, readouts, style, self._style.font_family)

    def _redraw(self) -> None:
        """Draw the current view, then re-offer its legend rows to the Style group.

        The checklist is rebuilt in a ``finally`` because every one of the draw's
        early-outs (no sources, nothing checked, an empty frame) is a legitimate result
        with its own -- usually empty -- set of legend entries, and a checklist still
        listing the previous plot's rows is a control pointing at nothing.
        """
        self._stale_redraw_timer.stop()     # this draw writes the marks a pending look wanted
        try:
            self._redraw_current()
            self._stale_drawn = self._stale_signature()
            if self._message_item is None:
                self._figure_applied_config = self._drawn_config()
                self._apply_series_styles()
        finally:
            # THE DRAW IS WHAT SETTLES WHICH ENTRIES EXIST, so it is where decisions
            # about entries that no longer exist are dropped -- a cluster row unticked
            # at k = 4 must not still be hiding "cluster:3" after a re-run at k = 2.
            # Before the checklist is rebuilt, so the list shows the pruned truth.
            self._prune_legend_memory()
            self._sync_legend_entry_list()

    def _redraw_current(self) -> None:
        # ``_pending_notes`` MEANS "THE CURRENT DRAW WANTS THIS", so the current draw
        # is where it is emptied. Its own docstring (:meth:`_deferred_note`) said this
        # already; the code did not implement it. Keys went in and only ever came out
        # when a batch was answered or aborted, so the set was really "everything any
        # draw has asked for since the last batch" -- and one thing depended on the
        # difference: :meth:`_settle_notes`'s "nothing in flight is still wanted, so
        # cancel it" branch. Because a dispatched key stayed in ``_pending_notes``, the
        # in-flight set was a SUBSET of pending by construction, the intersection was
        # never empty, and that branch could not execute. A guard whose condition is
        # not the one its docstring documents is not a guard.
        #
        # Anything still wanted is re-registered a few lines below, synchronously,
        # inside this same call -- including keys already on the worker, which
        # :meth:`_deferred_note` records without re-arming the timer. A drain loop that
        # waits on ``_pending_notes`` therefore sees exactly what it did before, except
        # for keys the user has drawn away from, which is the point.
        self._pending_notes = {}
        if not self._sources:
            self._message("No data yet — run detection or add measurements, then Refresh.")
            return
        drawn = self._drawn_frames()
        if not drawn:
            self._message("Tick a source on the left — none of the checked ones has rows to plot.")
            return
        kind = self._plot_type()
        if kind == HISTOGRAM:
            self._draw_histogram(drawn)
        elif kind == SCATTER:
            self._draw_scatter(drawn)
        elif kind == KDE:
            if self._workspace_snapshot_rendering:
                from .figure_adapter_plots import restore_sampled_contours
                if restore_sampled_contours(self):
                    self._draw_provenance_caption()
                    return
                # Legacy projects store the recipe, not the sampled density grid.
                # Calling scipy's estimator here would make project-open an implicit
                # analysis run, so leave an explicit pending picture instead.
                self._message(
                    "Saved KDE configuration restored. Press Plot to recompute its "
                    "density contours from the attached source data.")
                return
            self._draw_kde(drawn)
        elif kind == HEATMAP:
            self._draw_heatmap(drawn)
            # ONE source is drawn, so one source is presented: the provenance caption,
            # the legend-name disclosure and the figure's source recipes must not name
            # a ticked source whose events are not in the image.
            drawn = drawn[:1]
        else:
            df = self._current_frame()
            if df is None or df.shape[0] == 0:
                self._message("This source has no rows to plot.")
                return
            self._draw_pca(df)
        # E2, once for every plot type rather than four times. Each branch above ends by
        # having called `_clear_plot`, so the caption is added to a fresh figure and can
        # never survive from the previous one; it declines to draw over a message.
        self._presentation_sources = tuple((src.key, src.label, bool(src.is_segment))
                                           for src, _df in drawn)
        if self._style.fit_label_state().in_legend and getattr(
                self, "_presentation_readouts", None):
            # In legend instead: a fit's row was added before its readout was recorded,
            # so it was built without its numbers; rebuild the rows now that they exist.
            self._restyle_committed_legend()
        self._draw_provenance_caption()

    def _draw_histogram(self, drawn: list[tuple[PlotSource, pd.DataFrame]]) -> None:
        col = self._hist_col.currentText()
        if not col:
            self._message("Pick a numeric column to histogram.")
            return

        log_x = self._hist_logx.isChecked()
        log_y = self._hist_logy.isChecked()
        notes: list[str] = []

        # Cluster split replaces the source axis with the cluster axis; it is offered
        # only for a single source (see _sync_cluster_split_toggle), so the two never
        # compete for the colour channel.
        split = (len(drawn) == 1
                 and self._hist_colour_by.currentText() == _COLOUR_CLUSTER
                 and self._cluster_result is not None)

        # (label, colour, values, source, n_for_normalization)
        groups: list[tuple[str, str, np.ndarray, PlotSource, int]] = []
        if split:
            src, df = drawn[0]
            groups = self._cluster_groups(df, col, log_x, notes, src)
            if not groups:
                self._message(self._nothing_to_draw([df], col, log_x)
                              + " (Nothing to split by cluster, either.)")
                return
        else:
            for src, df in drawn:
                who = f"{src.label}: " if len(drawn) > 1 else ""
                v = self._hist_sample(df, col, log_x, notes, who)
                if v is None:
                    notes.append(f"{src.label}: no plottable values for “{col}”.")
                    continue
                # Each source normalizes to ITSELF: that is what makes two unequal-n
                # recordings comparable shapes rather than one dwarfing the other.
                groups.append((src.label, src.colour, v, src, int(v.size)))
        if not groups:
            self._message(self._nothing_to_draw([d for _s, d in drawn], col, log_x))
            return

        edges = self._shared_edges([g[2] for g in groups], self._bins.value(), log_x)

        self._clear_plot()
        multi = len(groups) > 1
        details = [
            f"Histogram of {col}   bins = {self._bins.value()}   "
            f"y = {self._norm_combo.currentText()}",
            # ...and IN WHAT, immediately under the line that names the quantity. See
            # `_density_unit_notes`: on the default ms axis the drawn area is 1000 and on
            # the shipped log-x template it is 23.3, and this pane is the only surface with
            # room to say so. It says nothing at all when there is nothing to say (Time
            # unit = s, linear x), because a pane that cries wolf is not read.
            *self._density_unit_notes(col, log_x, [df for _s, df in drawn]),
        ]
        if multi:
            details.append(
                "shared bin edges across all drawn series "
                f"[{edges[0]:.4g}, {edges[-1]:.4g}]"
                + ("  ·  colour = cluster" if split else "  ·  colour = source"))

        for i, (label, colour, v, src, norm_n) in enumerate(groups):
            counts, _ = np.histogram(v, bins=edges)
            y = self._normalized(counts, edges, norm_n)
            pen_colour = self._series_colour(index=i, identity=colour if multi else None)
            fill = QtGui.QColor(pen_colour)
            fill.setAlpha(_HIST_FILL_ALPHA if not multi else 70)
            # Log-count axis: a fill-to-zero baseline is log10(0) = -inf, so drop the fill and
            # draw the step outline only (empty bins read as gaps, as expected there).
            item = pg.PlotDataItem(
                x=edges, y=y, stepMode="center",
                fillLevel=None if log_y else 0,
                brush=None if log_y else pg.mkBrush(fill),
                pen=pg.mkPen(pen_colour, width=self._style.line_width))
            self._plot.addItem(item)
            self._hist_items.append(item)
            self._cache_figure_source(src)
            item._presentation_role = "category" if multi else "data"
            self._tag_series(item, "data", label if split else (src.key if src else "data"),
                             label, identity_kind="category" if split else "source",
                             colour=colour if multi or split else None)
            if i == 0:
                self._hist_item = item
            # A cluster group's `label` is "cluster 1", not a source name; only a
            # SOURCE row is renameable, so only that branch goes through the override.
            self._legend_add(
                item, self._legend_entry(label if split else self._legend_name(src), v.size),
                # A split histogram's groups ARE clusters, so they are remembered under
                # the cluster kind and share their inclusion with the scatter/PCA
                # swatches naming the same populations.
                kind="cluster" if split else "series",
                key=label if split else (src.key if src is not None else ""))
            details.append(f"  {label}: n = {v.size}")

        # THE APPLIED OVERLAY, NOT THE BOX: a hand tick is a proposal until Plot (see
        # `_on_kde_overlay_toggled`), so this draw -- also reached by a live bin-count
        # or log change -- only estimates what was committed. On project open the box
        # is the SAVED request and is never estimated; it is said, and left for Plot ●.
        if getattr(self, "_workspace_snapshot_rendering", False) and self._hist_kde.isChecked():
            self._kde_pending_after_snapshot = True
            details.append("KDE overlay: restored as a setting and NOT recomputed on project "
                           "open (estimating it is an analysis run). Press Plot to draw it "
                           "from the attached source data.")
        elif self._applied_kde_overlay:
            details.extend(self._draw_kde_overlay(groups, col, edges, log_x, log_y,
                                                  multi, split))

        # The detection floor is drawn whether or not a fit is on: the histogram's
        # left edge is the detector's, not the physics'. (figures.py, same rule.)
        # With an overlay the floors may differ, so the SHALLOWEST is shaded (the
        # region no source could report) and every one of them is named in Details.
        floors = []
        for src in (dict.fromkeys(g[3] for g in groups)):
            inp = self._dwell_inputs(col, src)
            if inp is not None:
                floors.append((src.label, inp))
        if floors:
            t_min = min(f[1].t_min for f in floors)
            self._shade_below_limit(t_min, min(float(g[2].min()) for g in groups), log_x)
            for label, inp in floors:
                who = f"{label}: " if len(floors) > 1 else ""
                details.append(f"{who}detection limit t_min = {inp.t_min:g} s "
                               f"(from {inp.t_min_source})")
            if len({round(f[1].t_min, 15) for f in floors}) > 1:
                details.append(
                    "NOTE: these sources were detected at DIFFERENT resolutions, so their "
                    "shortest reportable events differ. Each fit below uses its OWN t_min; "
                    "the shading marks the shallowest of them.")

        #: The title's KS headline -- "" until a fit below is thrown out. See
        #: `_ks_headline` for why the verdict rides the title's WARN channel and not
        #: its automatic text.
        ks_warn = ""
        # THE APPLIED FIT, never the SELECTED one (owner note #12). The combo is a
        # proposal until the Apply button copies it here; reading the combo would put
        # the fit back on the selection and undo the gate.
        kind = self._applied_fit
        if kind != _FIT_NONE:
            titles: list[str] = []
            #: The same summaries with NO verdict tag -- what the title's own text is
            #: built from, so the verdict is not said twice on one line once the warn
            #: channel has prepended it.
            headlines: list[str] = []
            #: Which fits the test threw out, by the name the FIGURE calls their
            #: source ("" for the one fit of a one-source plot).
            rejected: list[str] = []
            #: The SAME fits said in full — every parameter, n, BIC and the KS verdict —
            #: for the box drawn ON the figure (owner note #13). The title keeps the
            #: one-liner: a title is a caption and has one line's worth of room, while
            #: the on-plot box is the thing a saved PNG carries INSTEAD of Details, so
            #: it has to hold everything a reader would otherwise have to ask for.
            full: list[str] = []
            # WHAT THE FIT RANGE BOXES SAY BUT DID NOT DO -- once for the pane, not once
            # per source, and OUTSIDE the try below so a refusal keeps it. See
            # :meth:`_fit_range_input_notes`.
            notes.extend(self._fit_range_input_notes())
            if split:
                # ONE fit over the whole source, not one per cluster: the labels come
                # from a multi-feature clustering, not from a mixture model of this one
                # column, so per-cluster fits would claim a decomposition the clustering
                # never made. Said out loud rather than left to be inferred from the plot.
                notes.append(
                    "the fitted curve is over the WHOLE source, not per cluster: the cluster "
                    "labels come from a multi-feature clustering, not from a mixture model "
                    "of this column.")
                src, df = drawn[0]
                all_v = self._hist_sample(df, col, log_x, notes)
                if all_v is not None:
                    try:
                        t, fit_notes = self._overlay_fit(
                            kind, col, all_v, edges, log_x, src, self._style.fit_color, "")
                        self._presentation_readouts.append(
                            (src.key if src is not None else "", self._readout_of(t, None)))
                        titles.append(t)
                        headlines.append(getattr(t, "plain", t))
                        if getattr(t, "rejected", False):
                            rejected.append("")
                        full.append(getattr(t, "full", t))
                        notes.extend(fit_notes)
                    except (ValueError, RuntimeError) as exc:
                        notes.append(f"Fit unavailable: {exc}")
                        # A refusal names its real cause -- see _fit_range_blame.
                        blame = self._fit_range_blame(col, all_v)
                        if blame:
                            notes.append(blame)
            else:
                for label, colour, v, src, _n in groups:
                    try:
                        t, fit_notes = self._overlay_fit(
                            kind, col, v, edges, log_x, src,
                            colour if multi else self._style.fit_color,
                            label if multi else "")
                        self._presentation_readouts.append(
                            (src.key if src is not None else "",
                             self._readout_of(t, colour if multi else None)))
                        t_full = getattr(t, "full", t)
                        titles.append(f"{label}: {t}" if multi else t)
                        headlines.append(getattr(t, "plain", t))
                        if getattr(t, "rejected", False):
                            # By the figure's own name for the source, not `label`
                            # (the raw `src.label`): the headline is on the figure.
                            rejected.append(self._legend_name(src) if multi else "")
                        full.append(f"{label}: {t_full}" if multi else t_full)
                        notes.extend((f"{label}: {n}" if multi else n) for n in fit_notes)
                    except (ValueError, RuntimeError) as exc:
                        # Surface the stats guardrail (≥20 events, positive dwell, …) rather
                        # than swallow it; the histogram itself still stands.
                        notes.append(f"{label}: Fit unavailable: {exc}" if multi
                                     else f"Fit unavailable: {exc}")
                        # ...and, in the same pane that shows the range the user typed,
                        # says whether that range is what emptied the sample.
                        blame = self._fit_range_blame(col, v)
                        if blame:
                            notes.append(f"{label}: {blame}" if multi else blame)
            # Details carries the full list; the TITLE carries none of it -- a fit's
            # numbers are the fit label's, one line per source, so a typed title no
            # longer takes them off the figure and two taus never run off its top.
            details.extend(titles)
            ks_warn = self._ks_headline(rejected)
            # The FULL parameter set ON the figure (owner notes #13/#14): a saved image
            # whose parameters live only in Details is not a figure that can defend
            # itself in a talk.
            self._draw_fit_label()

        # The pooled verdict FIRST in the note list: whether these rows may be put on
        # one axis at all outranks anything said about the curve drawn through them.
        verdict = self._comparability(col)
        cluster_warnings = self._cluster_amplitude_warnings() if split else ()
        # ...and, ahead of even that, what is wrong with the COLUMN itself: a
        # superseded calibration means the numbers on this axis are not the numbers a
        # fresh run would produce, which outranks whether they may be pooled.
        col_notes = self._column_notes([col])
        # ...and AHEAD OF EVEN THAT, what the window was asked for and could not give.
        # A superseded calibration says these numbers are not the numbers a fresh run
        # would produce; this says the picture is not the picture that was requested,
        # which outranks anything said about the quantity in it.
        #
        # THE NOTES LIST IS WHERE IT BELONGS -- "the notes list is for drops and
        # refusals", in `_density_unit_notes`'s own words, and this is a refusal. It is
        # rebuilt from `_norm_refusal` on EVERY draw rather than printed once at the door
        # it happened at, which is the whole reason that record exists: Details is
        # rewritten in full by every redraw, and a template recalled onto an armed window
        # redraws immediately. `note` reads the mode back off the combo, so the sentence
        # names what is actually drawn even after `_sync_norm_default` has moved it.
        #
        # AND ON THE OTHER THREE PAGES, AND IN THEIR EMPTY STATES TOO, which is why it
        # comes through `_incoming_notes` rather than off the record here: a configuration
        # names a y quantity whatever its plot type and whatever the window can currently
        # draw, and this was the only page that spoke the refusal (see `_norm_refusal_notes`
        # for the measurement, and for why it was not moved to the door or into the
        # exported file instead). The same call carries the OTHER six kinds of loss an
        # imported sidecar can suffer -- see `_ConfigLosses`.
        #
        # AND NOT A MODAL: a name in somebody's file is not worth stopping the app for,
        # and Details is on screen without scrolling (measured: 180 px tall,
        # `visibleRegion()` non-empty at rest). The other tell is the template combo,
        # which drops to "— manual —" on its own now that `_installable` lets the
        # request through.
        notes[:0] = self._incoming_notes() + col_notes + list(verdict.warnings)
        notes.extend(cluster_warnings)
        notes.extend(self._detection_notes())

        self._plot.setLogMode(x=log_x, y=log_y)
        self._plot.setLabel("bottom", self._axis_label_for("bottom", col, log_x),
                            **self._label_style())
        # THE UNIT GOES ON THE AXIS, NOT INTO THE NUMBERS (S6-03). `_y_axis_label` states
        # what the heights are in; the heights themselves are untouched, because the one
        # thing that must never move is the sample the fit is taken from -- see
        # `_axis_label_for`'s docstring for why this window scales ticks and nothing else,
        # and `_y_axis_label`'s for the measurements that killed the alternative. The log
        # tag is folded inside `_y_axis_label` now, so it lands in the same bracket as the
        # unit (`density (1/s, log)`) instead of a second one. The user's override still
        # REPLACES THE TEXT ENTIRELY, exactly as it does on x: overriding is a deliberate
        # act, Details still names the unit, and silently appending to somebody's typed
        # label would be its own small lie.
        self._plot.setLabel("left",
                            self._remember_axis_label("left", self._y_axis_label(col, log_y)),
                            **self._label_style())
        # Record the warning separately; _set_details places it outside artwork.
        self._apply_title(None,
                          warn=self._warn_headline(self._stale_headline([col]),
                                                   verdict.headline, ks_warn,
                                                   "CHECK SELECTED AMPLITUDES" if cluster_warnings else ""))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(log_x, log_y)
        if notes:
            details.append("")
            details.extend(notes)
        self._set_details("\n".join(details))

    #: How many KDE curves the window remembers. A curve is a function of the exact
    #: sample and the log choice only (not of bins or normalization, which only rescale
    #: it), so a bin-count spin or a y-quantity switch redraws without re-estimating.
    _KDE_CACHE_SIZE = 32

    def _kde_for(self, v: np.ndarray, log_x: bool):
        """``kde_curve`` of ``v``, cached by the sample's bytes and the log choice."""
        cache = self._kde_cache
        key = (self._sample_digest(v), bool(log_x))
        curve = cache.pop(key, None)
        if curve is None:
            curve = kde_curve(v, log=bool(log_x))
        cache[key] = curve                      # most recent last
        while len(cache) > self._KDE_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        return curve

    #: How long the panel waits for the user to stop before computing the expensive
    #: advisory notes. Long enough that a burst of accept/reject clicks never pays for
    #: them; short enough that a user who pauses to read gets the answer without asking.
    _NOTE_SETTLE_MS = 400

    def _deferred_note(self, key, compute):
        """``(ready, value)`` for a note too expensive to compute on the redraw path.

        THE INVARIANT: **drawing a fit and explaining it are two operations with two
        latency budgets.** Drawing is on the click path and must stay interactive;
        explaining is advisory and may lag. Nothing that runs here may sit between a
        curation click and the repaint.

        This is the third attempt at that cost and the first to state it as a rule, so
        the history is worth keeping:

        1. The rival-family and better-k scans ran unconditionally. One reject click on
           the owner's `_0006` recording cost **3.40 s**, almost all of it two rivals'
           discarded bootstrap CIs.
        2. `stats.point_fits_only()` removed the CIs and the scans were gated on the
           goodness-of-fit verdict being REJECT -- costed as an exceptional path, with
           the gate's own docstring resting on "a model that passes its KS test costs
           nothing here". **On the owner's data no model passes.** The exponential is
           rejected on all three apo-ferritin recordings, so the exceptional path was the
           default path and a reject click still cost **0.4-1.0 s**, against 68-93 ms with
           the fit box set to none.
        3. Caching keyed on the sample's bytes could not help either: during curation the
           accepted-event sample changes on *every* click, so the key is new every time by
           construction.

        4. Deferral to a TIMER separated the click from the cost and did not remove the
           cost, because a timer is the same thread 400 ms later. Measured through a
           real MainWindow on the owner's -700 mV apo run -- one click, then sitting
           still -- the GUI thread blocked for 368 ms (`exponential`), 694 ms (`gamma`),
           991 ms (`exponential + gamma`), 1266 ms (`inverse-gaussian`). On
           `exponential` the user experienced 62 ms of click plus 368 ms of stall, just
           not at once. See :class:`_NoteWorker` for the whole table.

        Deferral is what separates the two budgets; a WORKER is what pays the second
        one. A miss records the work and arms a single-shot timer; each further redraw
        re-arms it, so a burst of clicks computes nothing. When the user pauses,
        :meth:`_settle_notes` hands what is pending to :class:`_NoteWorker` and the GUI
        thread does nothing but install the answer and redraw once -- and that redraw
        hits the cache, so the notes appear with no further delay.

        What is deferred is also no longer only the two advisory scans: the drawn fit's
        own bootstrap CI goes through here too (see :meth:`_overlay_fit`), because it is
        a REPORTED number and therefore cannot hide inside ``stats.point_fits_only()``
        the way a rival's discarded CI can.

        A key already being computed is recorded as wanted but does NOT re-arm the
        timer. Without that, every redraw during a long bootstrap would re-arm, and
        every settle would find the same key already running; with a naive "restart on
        every settle" it would never finish at all.

        ``_pending_notes`` therefore means "the current draw wants this and does not
        have it yet", whether it is waiting for the timer or already on the worker.
        That is the panel's own definition of not-yet-settled, and the drain loops that
        wait on it (and the user, watching the placeholder) both need it to stay true
        until the answer is actually in.
        """
        if key in self._note_cache:
            return True, self._note_cache[key]
        self._pending_notes[key] = compute
        if key not in self._inflight_notes:
            self._note_timer.start(self._NOTE_SETTLE_MS)
        return False, None

    def _settle_notes(self) -> None:
        """Hand what :meth:`_deferred_note` deferred to a WORKER. Computes nothing here.

        The method the timer fires. It must stay cheap enough to be invisible: every
        line of arithmetic that used to live here is now inside :class:`_NoteWorker`,
        and what is left is building the job list and spinning the thread.

        ONE BATCH AT A TIME, AND THE RUNNING ONE IS ONLY KILLED WHEN IT HAS STOPPED
        BEING ABOUT WHAT IS ON SCREEN. Every key carries the bytes of the sample it was
        computed from, so during curation a click makes a NEW key rather than
        invalidating an old one. "Abort whatever is running whenever a redraw arrives"
        would therefore restart the bootstrap on every click and finish never. So: if
        anything in flight is still wanted, let the batch finish; if none of it is, it
        is about events nobody is looking at any more -- cancel it (the kernels poll the
        ambient token and stop inside themselves), detach it, and dispatch the
        new one. Its answer is dropped either way, by the sender guard.

        "Still wanted" is now genuinely what the CURRENT draw asked for -- see the top
        of :meth:`_redraw`. Until that, a dispatched key stayed in ``_pending_notes``
        for ever, ``_inflight_notes`` was a subset of ``pending`` by construction, and
        the abort branch below was unreachable code. The path that reaches it: a batch
        is running for one sample and the user changes the bins, the component count or
        the fit range -- no ``set_data``, so nothing else aborts anything -- and the new
        draw wants a key the running batch is not computing.
        """
        pending = {k: c for k, c in self._pending_notes.items()
                   if k not in self._note_cache}
        if self._inflight_notes:
            if self._inflight_notes & set(pending):
                return
            self._abort_note_job()
        if not pending:
            self._pending_notes = {}
            return
        self._inflight_notes = set(pending)
        worker = _NoteWorker(list(pending.items()))
        worker.done.connect(self._on_notes_ready)
        self._note_worker = worker
        self._note_thread = start_worker_thread(
            worker, domain=JOB_NONE, label="plots-fit-notes", owner=self)
        # OFF THE GUI THREAD IS NOT THE SAME AS FREE, and it is worth being honest
        # about why. ``scipy.optimize.minimize(method="Nelder-Mead")`` -- which is what
        # every one of these jobs is made of -- is implemented in PYTHON, so the worker
        # holds the GIL for most of its run. The GUI thread is no longer BLOCKED (that
        # is the whole win: a 1.6 s freeze becomes a window that keeps painting), but
        # while a batch runs the two threads share the interpreter and a click costs
        # more than it does on an idle window. Priority is the one lever available from
        # here: an advisory note has no deadline at all, so it goes to the back of
        # whatever queue the OS keeps. Best-effort by nature -- CPython's GIL handoff is
        # not priority-aware -- which is why the settle timer, not this, is what keeps a
        # burst of curation clicks from meeting a running batch in the first place.
        try:
            self._note_thread.setPriority(QtCore.QThread.LowestPriority)
        except (RuntimeError, AttributeError):   # pragma: no cover - platform-dependent
            pass

    @QtCore.Slot(object)
    def _on_notes_ready(self, results) -> None:
        """Install a finished batch on the GUI thread, then redraw once.

        The sender guard is load-bearing, not defensive noise: :meth:`_abort_note_job`
        severs a superseded worker's connection, but a ``done`` already queued on this
        thread's event loop would otherwise still be delivered -- and that is exactly
        the "paint a stale interval beside data it is not about" failure.

        ``self._note_worker is not None`` is half of that guard, for the reason C40
        recorded: once a detached worker's C++ object has been collected ``sender()``
        returns None too, and ``None is None`` would let the stale batch straight
        through.
        """
        if self._note_worker is None or self.sender() is not self._note_worker:
            return
        self._inflight_notes = set()
        self._retire_note_job()
        for key, value in results or ():
            # The worker does not decide what a failure LOOKS like; the shape of a
            # cached value depends on the key, and that is the panel's business.
            self._note_cache[key] = ("" if key[0] != "rivals" else []) \
                if value is _NOTE_FAILED else value
            while len(self._note_cache) > self._NOTE_CACHE_MAX:
                self._note_cache.pop(next(iter(self._note_cache)))
        # The batch is answered, so nothing here is outstanding any more. The redraw
        # below re-registers exactly what the CURRENT draw still wants -- which is how
        # a key belonging to a sample the user has since curated away stops being
        # pending instead of keeping the panel permanently unsettled.
        self._pending_notes = {}
        # ...but NOT over a stale plot. A note is an advisory about the picture that
        # was drawn; redrawing here while the user has changed their selection and has
        # not pressed Plot would let a background thread walk through the run gate and
        # draw the new data behind them.
        if not self._stale:
            self._redraw()

    def _retire_note_job(self) -> None:
        """Drop the current worker/thread pair. Cancels nothing; emits nothing."""
        worker, thread = self._note_worker, self._note_thread
        self._note_worker, self._note_thread = None, None
        if worker is not None or thread is not None:
            retire_worker_thread(worker, thread, wait_ms=DEFAULT_JOIN_MS)

    def _abort_note_job(self) -> None:
        """Ask the in-flight batch to stop, sever it, and return.

        Its results go nowhere: the connection is cut by ``retire_worker_thread`` and
        the sender guard in :meth:`_on_notes_ready` refuses anything already queued.
        Its keys are NOT cached, so the next redraw simply defers them again.

        AND ITS KEYS LEAVE ``_pending_notes`` WITH IT. Found in this fix's own verify
        pass: without that line a superseded key stays pending, nothing re-requests it
        (it is about a sample the user has curated away), and the NEXT settle happily
        dispatches the stale job to a fresh worker -- which then computes an answer
        nobody wants, caches it, and evicts a live entry from a 16-slot cache to do so.
        A key that is still wanted is re-added by the redraw that follows every abort;
        one that is not, is not, which is the whole point of aborting."""
        worker = self._note_worker
        if worker is not None:
            worker.cancel()
        for key in self._inflight_notes:
            self._pending_notes.pop(key, None)
        self._inflight_notes = set()
        self._retire_note_job()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        """Stop the advisory worker, KEEP the results, then blank the window.

        Qt aborts the process if a QThread is destroyed while running and a note has
        nothing to come back to, so the worker goes first.

        Then owner note #16, whose two halves are one order: "once the window closes
        the results should be saved and the panels cleared; when the user opens again
        everything needs to be blank." SAVE BEFORE CLEAR, obviously, but also: this
        window was read-only against the session (it only ever *got* results), so a fit
        the user computed here -- chose the family, set k, set the fit range, watched
        the KS verdict -- was thrown away on close while the same fit made in the Fits
        panel survived. Saving it puts the two on the same footing and makes it
        available to the figure export, which reads exactly these keys.
        """
        if getattr(self, "_phase2_figure_managed", False):
            from .figure_adapter_plots import PlotsFigureAdapter
            adapter = PlotsFigureAdapter(self)
            allowed, reason = adapter.can_suspend()
            if not allowed:
                self.statusBar().showMessage(reason, 10000)
                event.ignore()
                return
            self.workspaceCheckpointRequested.emit()
            adapter.suspend()
            super().closeEvent(event)
            return
        # Synchronous by Qt contract.  MainWindow snapshots the live recipe/results
        # into its project-owned v11 workspace before ``reset_state`` deliberately
        # blanks this widget, preserving the long-standing blank-on-reopen behavior
        # without losing the analysis from Save Project.  Closing changes visibility,
        # not project content, so this is a clean checkpoint request rather than a
        # ``workspaceChanged`` dirty notification.
        self.workspaceCheckpointRequested.emit()
        self._note_timer.stop()
        # A coalescing undo step still open belongs to a window that is about to be
        # blanked, and `reset_state` below drops the stack it would have been pushed onto.
        self._history_timer.stop()
        self._stale_redraw_timer.stop()
        self._pending_notes = {}
        self._abort_note_job()
        self._persist_results()
        self.reset_state()
        super().closeEvent(event)

    #: Module ids the rest of the app files a fit under, by the KIND of column it was a
    #: fit OF. The same two ``main_window._stored_fit`` reads back for the figure export
    #: and ``project`` persists -- named here rather than re-derived, because a fit
    #: written under a third id would be a fit nothing ever reads.
    _FIT_MODULE_IDS: tuple[tuple[tuple[str, ...], str], ...] = (
        (_DWELL_COLUMNS, "fit.dwell"),
        (_DEPTH_COLUMNS, "fit.depth"),
    )
    #: Where this window's own derived results go. Distinct ids from the fits: a
    #: clustering is not a distribution fit and must not be read back as one.
    _CLUSTER_MODULE_ID = "plots.cluster"
    _PCA_MODULE_ID = "plots.pca"

    def _report_takes_saved_fit(self, src: PlotSource | None, col: str) -> bool:
        """Whether Generate Report figures would draw the fit the session already holds for
        ``src``'s table in ``col``'s slot: the report's own rule
        (``modules.report_fit_problem``, as ``MainWindow._stored_fit`` asks it)."""
        module_id = next((mid for cols, mid in self._FIT_MODULE_IDS if col in cols), "")
        if (not module_id or src is None or src.doc_id is None or src.channel is None
                or not src.module_id):
            return False
        try:
            stored = self._session.results.get(src.doc_id, int(src.channel), module_id)
            table = self._session.results.get(src.doc_id, int(src.channel), src.module_id)
            return (stored is not None and table is not None
                    and hasattr(stored.payload, "predict")
                    and not modules.report_fit_problem(stored, table))
        except Exception:       # pragma: no cover - a caveat must never break a plot
            return False

    def _persist_results(self) -> tuple:
        """Push what this window computed back into the session, before it is cleared.

        One :class:`~nanoscope.core.analysis.AnalysisResult` per fitted SOURCE, keyed
        by that source's own ``(doc_id, channel)`` and stamped with that source's own
        calibration fingerprint -- never the primary's, because with an overlay up
        there are several fits of several recordings and filing them all under one
        recording would attach one pore's tau to another pore's events.

        Never raises into ``closeEvent``: a window that refuses to close because a
        result could not be stored is worse than one that closes without storing it,
        and the user is told nothing either way at this point in the interaction.
        """
        published = []
        try:
            for key, fit in dict(self._fits).items():
                src = self._sources.get(key)
                if src is None or src.doc_id is None or src.channel is None:
                    continue          # the manual table has no document to hang it on
                col = self._fit_columns.get(key, "")
                module_id = next((mid for cols, mid in self._FIT_MODULE_IDS
                                  if col in cols), "")
                if not module_id:
                    # A fit of `snr` or `ecd_pA_s` is a real fit, but there is no
                    # agreed key for it and inventing one would put a fit of the wrong
                    # quantity where the export expects a dwell or a depth.
                    continue
                if (fit.meta.get(modules.FIT_RANGE_KEY) is not None
                        or self._fit_range(col) != (None, None)):
                    # A fit over a typed Fit range is of a narrowed sample, so the report
                    # would draw it over the whole column as "your saved fit". The owner's
                    # call (S10-LIN3): not filed. The fit records its range
                    # (`modules.FIT_RANGE_KEY`), restored from a project too; the boxes
                    # are asked as well, for a range typed and not yet applied.
                    continue
                self._session.results.put(self._as_result(
                    module_id, src, payload=fit,
                    params={"column": col, "model": self._applied_fit,
                            "n_components": int(self._applied_fit_k),
                            "use_filter_limited": bool(
                                self._use_filter_limited.isChecked())},
                    scalars={**{k: float(v) for k, v in fit.params.items()},
                             "aic": float(fit.aic), "bic": float(fit.bic),
                             "n": float(fit.n)},
                    warnings=tuple(str(w) for w in fit.meta.get("warnings", ()))))
                published.append((str(src.doc_id), int(src.channel), module_id))

            # The clustering and the PCA belong to the PRIMARY source (that is the one
            # they were computed on; see `run_clustering`), and they travel with the
            # Details text, which is the only written record of what the window said
            # about them.
            primary = self.primary_source()
            if primary is not None and primary.doc_id is not None \
                    and primary.channel is not None:
                details = self.details_text()
                for res, mid in ((self._cluster_result, self._CLUSTER_MODULE_ID),
                                 (self._pca_result, self._PCA_MODULE_ID)):
                    if res is None:
                        continue
                    self._session.results.put(self._as_result(
                        mid, primary, payload=res,
                        params={"details": details}, scalars={}, warnings=()))
                    published.append((str(primary.doc_id), int(primary.channel), mid))
        except Exception:      # pragma: no cover - a save must never block a close
            pass
        return tuple(published)

    def _as_result(self, module_id: str, src: PlotSource, *, payload,
                   params: dict, scalars: dict, warnings: tuple[str, ...]):
        """One stored result for ``src``, stamped with ITS calibration fingerprint.

        The stamp is what ``main_window._stored_fit`` checks before it will reuse a
        saved fit on an exported figure, so a missing one has to mean "uncalibrated"
        (``(1.0, 0.0)``, the app's own default) rather than be left off."""
        fp = src.calibration_fingerprint or (1.0, 0.0)
        return AnalysisResult(
            module_id=module_id,
            module_version="1.0",
            doc_id=src.doc_id,
            channel=int(src.channel),
            params=dict(params),
            calibration_fingerprint=tuple(float(v) for v in fp),
            payload=payload,
            overlays=(),
            scalars=dict(scalars),
            warnings=tuple(warnings),
        )

    def reset_state(self) -> None:
        """Blank the window: no sources, no fits, no pins, default everything.

        The other half of owner note #16 -- "when the user opens again everything needs
        to be blank". The window is created ONCE and re-shown, so without this a
        reopened Plots came back carrying the last session's typed title, its pinned
        axis range (in whatever unit was selected then), its light background and its
        renamed legend entries, over data the user may have replaced entirely.

        IT NOW ACTUALLY BLANKS EVERYTHING, which is the point of doing it through the
        config. The old version wrote out a hand-kept list of widgets, and the list was
        incomplete in exactly the way a hand-kept list always becomes: a reopened window
        came back with the previous session's bin count, log axes, KDE contour count,
        scatter/KDE log flags, colour-by, cluster method and k -- and with its
        normalization combo still on "density" while the flag that remembers the user
        chose it had been reset, so the automatic default could never take that combo
        back. ``PlotConfig()`` IS "blank": one value, written by the one writer, and a
        control added later is blank on reopen without anyone remembering to add it here.

        DERIVED STATE IS CLEARED SEPARATELY AND FIRST, because it is not config and has
        no representation in a config: the frames, the fits, the cluster/PCA results, the
        note cache, the plot items and the run gate all describe rows this window is
        about to stop having.
        """
        was, self._updating = self._updating, True
        try:
            self._figure_pending_sources = None
            self._sources = {}
            self._fits = {}
            self._fit_columns = {}
            self._last_fit = None
            self._cluster_result = None
            self._pca_result = None
            self._note_cache = {}
            self._kde_cache = {}      # curves of rows this window is about to stop having
            self._applied_kde_overlay = False   # the blank box's applied twin
            self._empty_list_widget(self._source_list)   # see `_retire_row`
            # With no sources there are no columns to offer, so every column picker and
            # the feature list empty themselves -- which is also what stops the config
            # write below from re-selecting a column out of a stale list.
            self._repopulate_columns()
            self._empty_list_widget(self._legend_entries)
            self._set_details("")
            self._clear_plot()
            # ...and the RUN GATE, which is as much "everything needs to be blank" as
            # the style is: a reopened window that still counted as run would restyle
            # live over a picture the user has not asked for yet.
            self._plot_armed = False
            self._stale = False
            self._stale_reason = ""
            self._stale_reasons = []
            self._checked_snapshot = ()
            self._legend_offered = {}
        finally:
            self._updating = was

        # THE CONFIG, ALL OF IT, IN ONE WRITE. `draw=False` because the redraw at the
        # foot of this method is the one that puts "No data yet" on screen, and because
        # `_set_config`'s own draw would ARM the gate we just cleared.
        self._set_config(PlotConfig(), draw=False)
        # THE IMPORT DISCLOSURES GO TOO. They are scoped to an origin the history is
        # about to forget (`_ConfigHistory.reset` takes a fresh one), so they could never
        # be spoken again; cleared outright rather than left orphaned.
        self._norm_refusal.forget()
        self._config_losses.forget()
        # The dragged legend position went with the rest of the style. The LIVE legend
        # item keeps its anchor until something draws again, and the next draw is the one
        # that puts it back in the corner (`legend_pos` is None now, so
        # :meth:`_apply_legend_position` takes the default branch).
        self._sync_filter_limited_toggle()
        self._sync_cluster_split_toggle()
        self._sync_run_state()
        # THE UNDO STACK GOES WITH THE DATA IT DESCRIBED -- `EventHistory.clear`'s
        # argument, one object down: every entry in it names sources this window no
        # longer has, so Undo would offer a step and then visibly do nothing. The
        # template name goes for the same reason the run gate does -- it described a
        # configuration that has just been replaced by the blank one.
        self._history.reset(self.config())
        self._select_template(self._TPL_MANUAL)
        self._sync_config_buttons()
        self._redraw()

    # -- scatter ---------------------------------------------------------------------------------
    def _draw_scatter(self, drawn: list[tuple[PlotSource, pd.DataFrame]]) -> None:
        xcol, ycol = self._scatter_x.currentText(), self._scatter_y.currentText()
        if not xcol or not ycol:
            self._message("Pick numeric X and Y columns.")
            return
        log_x, log_y = self._scatter_logx.isChecked(), self._scatter_logy.isChecked()
        multi = len(drawn) > 1

        # Extract each source's plottable points first, so an empty overlay reports
        # itself once rather than half-drawing.
        series: list[tuple[PlotSource, np.ndarray, np.ndarray, object, list[object]]] = []
        # EVERY DROP DISCLOSED WITH ITS COUNT, in the same voice the histogram and the
        # clustering pane use. This mask silently deleted rows on both grounds -- a
        # non-finite x or y, and (with a log axis up) a non-positive one -- and printed
        # only the shrunken total: "Scatter amplitude vs dwell_s   n = 1960" over a
        # 2000-row table, beside a clustering pane saying "1960 kept, 40 dropped
        # (non-finite features)" about the identical rows.
        drops: list[str] = []
        from .nonlinear_fit_dialog import CurveDataset
        nonlinear_datasets = []
        for src, df in drawn:
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=np.float64)
            who = f"{src.label}: " if multi else ""
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.size and not finite.all():
                drops.append(
                    f"{who}{int(finite.size - finite.sum())} of {int(finite.size)} "
                    f"row(s) dropped: “{xcol}” or “{ycol}” is not a finite number there.")
            mask = finite.copy()
            if log_x:
                mask &= x > 0
            if log_y:
                mask &= y > 0
            n_nonpos = int(finite.sum() - mask.sum())
            if n_nonpos:
                axes = " and ".join(a for a in (f"log x (“{xcol}”)" if log_x else "",
                                                f"log y (“{ycol}”)" if log_y else "") if a)
                drops.append(f"{who}{n_nonpos} further row(s) dropped as non-positive "
                             f"for {axes}.")
            if not mask.any():
                continue
            series.append((src, x[mask], y[mask], df.index[mask],
                           self._point_identities(src, df, mask)))
            sigma_columns = {str(name): pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)[mask]
                             for name in df.select_dtypes(include="number").columns
                             if name not in (xcol, ycol)}
            nonlinear_datasets.append(CurveDataset(
                src.key, src.label, x[mask].copy(), y[mask].copy(), xcol, ycol,
                _column_quantity(xcol)[1], _column_quantity(ycol)[1], sigma_columns,
                dict(src.provenance)))
        if not series:
            self._message("No points to plot for this pair (check the log-axis toggles).")
            return

        total = sum(s[1].size for s in series)
        # A y quantity the incoming configuration named and this build could not install
        # is disclosed HERE as well as on the histogram: the config that carried it may
        # well have been a scatter, and this is the page the user is looking at when they
        # save it back. `_incoming_notes` is the one surface; `_norm_refusal_notes` holds
        # the measurement and the wording.
        details = [f"Scatter {ycol} vs {xcol}   n = {total}",
                   *self._incoming_notes(), *drops]
        if multi:
            details.append("colour = source")

        # Cluster colouring is a SINGLE-SOURCE mode: with an overlay up the colour
        # channel already carries the source, so cluster colours would overwrite the
        # very distinction the user asked for. Said in Details rather than silently.
        by_cluster = (not multi
                      and self._colour_by.currentText() == _COLOUR_CLUSTER
                      and self._cluster_result is not None)
        if multi and self._colour_by.currentText() == _COLOUR_CLUSTER:
            details.append(
                "Colour by Cluster is ignored while several sources are overlaid — colour "
                "means SOURCE here. Untick all but one source to colour by cluster.")

        self._clear_plot()
        self._nonlinear_drawn_datasets = nonlinear_datasets
        for i, (src, x, y, index, identities) in enumerate(series):
            cluster_labels = None
            if by_cluster:
                cluster_labels = self._cluster_result.labels.reindex(
                    index, fill_value=populations.DROPPED_LABEL).to_numpy(dtype=int)
                brushes: list[QtGui.QBrush] | QtGui.QBrush = category_brushes(
                    cluster_labels)
                details.append(self._cluster_summary())
            elif multi:
                c = QtGui.QColor(src.colour)
                c.setAlpha(_POINT_ALPHA)
                brushes = pg.mkBrush(c)
            else:
                brushes = self._point_brush()
                if self._colour_by.currentText() == _COLOUR_CLUSTER:
                    details.append("Run clustering to colour these points.")
            # LOG COORDINATES BY HAND. ``PlotItem.setLogMode`` transforms PlotDataItem
            # children; a ScatterPlotItem added straight to the scene is left alone -- the
            # same trap ``_shade_below_limit`` documents for its shading. Untransformed, a
            # dwell of 0.018 s was drawn at position 0.018 on an axis whose ticks read
            # 10**x, so every point piled up against the right edge while the fitted line
            # (a PlotDataItem, and therefore transformed) spanned the axis correctly. The
            # picture said the fit missed the data; the fit was fine and the points were in
            # the wrong place.
            px = np.log10(x) if log_x else x
            py = np.log10(y) if log_y else y
            item = pg.ScatterPlotItem(
                x=px, y=py, brush=brushes, pen=None, size=self._style.point_size)
            self._plot.addItem(item)
            self._scatter_items.append(item)
            self._cache_figure_source(src)
            item._presentation_role = "category" if (multi or by_cluster) else "data"
            if cluster_labels is not None:
                item._presentation_cluster_labels = cluster_labels.copy()
                item._presentation_cluster_source = src.key
            else:
                self._tag_series(item, "data", src.key, src.label, colour=src.colour if multi else None)
            self._make_event_points_clickable(item, src.key, identities)
            if i == 0:
                self._scatter_item = item
            if multi:
                self._legend_add(item, self._legend_entry(self._legend_name(src), x.size),
                                 kind="series", key=src.key)
                details.append(f"  {src.label}: n = {x.size}")
            elif cluster_labels is not None:
                self._legend_clusters(cluster_labels)

        # Optional least-squares best-fit line, PER SOURCE (never one line through
        # two pooled clouds). The summary always lands in Details, and its numbers in
        # the fit label -- never in the title, which stays the user's (or none).
        for src, x, y, _index, _identities in series:
            self._overlay_scatter_fit(x, y, xcol, ycol, details, src if multi else None,
                                      key=src.key)
        # The equation, units, R² with its degrees-of-freedom tag and n, ON the figure,
        # so the exported PNG carries them (owner note #13).
        self._draw_fit_label()

        verdict = self._comparability(xcol, ycol)
        cluster_warnings = self._cluster_amplitude_warnings() if by_cluster else ()
        details.extend(self._column_notes([xcol, ycol]))
        details.extend(verdict.warnings)
        details.extend(self._detection_notes())

        self._plot.setLogMode(x=log_x, y=log_y)
        self._plot.setLabel("bottom", self._axis_label_for("bottom", xcol, log_x),
                            **self._label_style())
        self._plot.setLabel("left", self._axis_label_for("left", ycol, log_y),
                            **self._label_style())
        self._apply_title(None,
                          warn=self._warn_headline(self._stale_headline([xcol, ycol]),
                                                   verdict.headline,
                                                   "CHECK SELECTED AMPLITUDES" if cluster_warnings else ""))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(log_x, log_y)
        self._set_details("\n".join(details))
        if hasattr(self, "_nonlinear_fitter"):
            self._nonlinear_fitter.redraw()
            state = self.nonlinear_fit_state()
            if state:
                from .nonlinear_fit_dialog import result_summary
                self._set_details(self.details_text() + "\n\n" + result_summary(state))
                self._install_nonlinear_diagnostics()

    # -- PCA -------------------------------------------------------------------------------------
    # -- KDE (density) ---------------------------------------------------------------------------
    #: Grid resolution per axis for the 2-D KDE. 128² keeps a three-source overlay
    #: interactive (three kde evaluations of 16k points each per draw) while the
    #: contours stay smooth at plot size.
    _KDE_GRID = 128

    def _draw_kde(self, drawn: list[tuple[PlotSource, pd.DataFrame]]) -> None:
        """2-D KDE contours, one set per checked source (owner note #19).

        The same figure ``Export density figure…`` renders with matplotlib, brought
        into this window as a live plot type: shared axes across sources so shapes are
        comparable, but each source contoured against its OWN peak density — a
        population with a tenth the events keeps its shape instead of flattening into
        the dominant one's floor. Log axes are handled the way the scatter handles
        them: coordinates transformed BY HAND, because ``setLogMode`` only transforms
        PlotDataItem children and an IsocurveItem is not one.
        """
        from scipy.stats import gaussian_kde

        xcol, ycol = self._kde_x.currentText(), self._kde_y.currentText()
        if not xcol or not ycol:
            self._message("Pick numeric X and Y columns.")
            return
        log_x, log_y = self._kde_logx.isChecked(), self._kde_logy.isChecked()
        multi = len(drawn) > 1

        series: list[tuple[PlotSource, np.ndarray, np.ndarray]] = []
        drops: list[str] = []
        for src, df in drawn:
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=np.float64)
            who = f"{src.label}: " if multi else ""
            mask = np.isfinite(x) & np.isfinite(y)
            if log_x:
                mask &= x > 0
            if log_y:
                mask &= y > 0
            dropped = int(mask.size - mask.sum())
            if dropped:
                drops.append(f"{who}{dropped} of {int(mask.size)} row(s) dropped "
                             f"(non-finite, or non-positive on a log axis).")
            x, y = x[mask], y[mask]
            # The same floor the export figure uses: a KDE of four points is a
            # decoration, not a density.
            if x.size < 5:
                drops.append(f"{who}only {x.size} usable event(s) — a KDE needs at "
                             "least 5, source skipped.")
                continue
            series.append((src, np.log10(x) if log_x else x,
                           np.log10(y) if log_y else y))
        if not series:
            self._message("No source has enough points for a KDE "
                          "(5 needed; check the log-axis toggles).")
            return

        # ONE grid over the union range, so every source's contours share axes.
        all_x = np.concatenate([s[1] for s in series])
        all_y = np.concatenate([s[2] for s in series])
        x0, x1 = float(all_x.min()), float(all_x.max())
        y0, y1 = float(all_y.min()), float(all_y.max())
        # 5% margin so the outermost contour is not clipped by the grid edge.
        mx, my = 0.05 * (x1 - x0 or 1.0), 0.05 * (y1 - y0 or 1.0)
        x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
        n = self._KDE_GRID
        gx = np.linspace(x0, x1, n)
        gy = np.linspace(y0, y1, n)
        mesh = np.stack([np.repeat(gx, n), np.tile(gy, n)])   # (2, n*n), x-major

        n_levels = int(self._kde_levels.value())
        total = sum(s[1].size for s in series)
        # KDE is an analysis surface in its own right, not an export-only preview.
        # Carry the SAME metric-specific pooling verdict that the histogram and
        # scatter views show: contours over different baseline windows are useful for
        # exploration, but must never export with an unqualified title or Details pane.
        verdict = self._comparability(xcol, ycol)
        # ...and here, for the reason `_draw_scatter` states at the same line: the
        # refusal belongs to the configuration, not to the histogram page.
        details = [f"KDE {ycol} vs {xcol}   n = {total}",
                   *self._incoming_notes(), *drops]
        details.extend(self._column_notes([xcol, ycol]))
        details.extend(verdict.warnings)
        details.extend(self._detection_notes())
        # THE CONTOURS' OWN UNIT, for the histogram's reason (S6-03): a level IS a density,
        # and a density with no unit is a number nobody can check. Two things this page gets
        # wrong quietly and this line says out loud. (1) The axes are re-scaled for display
        # exactly as the histogram's are -- measured, the x label reads `dwell (ms, log)` --
        # while the levels are per stored unit. (2) With a log toggle on, `gaussian_kde` is
        # fitted to log10 of that column, so the denominator is a DECADE of it, and
        # `plot_source_data` un-logs the exported grid's x back to linear -- so the CSV
        # pairs a linear coordinate with a per-decade density. Measured on the owner's
        # -700 mV set: levels 0.4994 ... 9.4887. Nothing here claims area = 1, so no value
        # and no label changes: this is prose only. The CSV column stays named `density` for
        # the same reason the histogram's does -- its x/y columns are the schema's own
        # unit-bearing names.
        details.append(
            "contour levels are a 2-D probability density per "
            f"({'decade of ' if log_x else ''}{xcol} × "
            f"{'decade of ' if log_y else ''}{ycol}), drawn as fractions of each source's "
            "own peak; the exported grid is in those same units, with x and y un-logged "
            "back to linear.")
        if multi:
            details.append("colour = source; each source is contoured against its "
                           "own peak density")

        self._clear_plot()
        # Index→data transform shared by every isocurve: grid index (i, j) lands at
        # (x0 + i·dx, y0 + j·dy) in (possibly log10) axis coordinates.
        tr = QtGui.QTransform()
        tr.translate(x0, y0)
        tr.scale((x1 - x0) / max(n - 1, 1), (y1 - y0) / max(n - 1, 1))
        for src, xs, ys in series:
            try:
                kde = gaussian_kde(np.vstack([xs, ys]))
                z = kde(mesh).reshape(n, n)          # z[i, j]: i → x, j → y
            except np.linalg.LinAlgError:
                details.append(f"{src.label}: KDE unavailable — the points are "
                               "degenerate (zero variance along an axis).")
                continue
            colour = QtGui.QColor(self._series_colour(identity=src.colour if multi else None))
            self._cache_figure_source(src)
            pen = pg.mkPen(colour, width=self._style.line_width)
            zmax = float(z.max())
            proxy = None
            for lv in np.linspace(0.05 * zmax, 0.95 * zmax, n_levels):
                curve = pg.IsocurveItem(data=z, level=float(lv), pen=pen)
                curve._presentation_role = "category" if multi else "data"
                self._tag_series(curve, "data", src.key, src.label, colour=src.colour if multi else None)
                curve.setTransform(tr)
                self._plot.addItem(curve, ignoreBounds=False)
                proxy = proxy or curve
            details.append(f"{src.label}: n = {xs.size}, "
                           f"bandwidth factor = {kde.factor:.3f} (Scott)")
            if multi and proxy is not None:
                # The legend needs an item with a pen; the first contour serves.
                sample = pg.PlotDataItem(pen=pen)
                sample._presentation_role = "category"
                self._tag_series(sample, "data", src.key, src.label, colour=src.colour)
                self._legend_add(sample, self._legend_name(src),
                                 kind="series", key=src.key)

        self._plot.setLogMode(x=log_x, y=log_y)
        self._plot.setLabel("bottom", self._axis_label_for("bottom", xcol, log_x),
                            **self._label_style())
        self._plot.setLabel("left", self._axis_label_for("left", ycol, log_y),
                            **self._label_style())
        self._apply_title(
            f"KDE {ycol} vs {xcol}",
            warn=self._warn_headline(self._stale_headline([xcol, ycol]),
                                     verdict.headline))
        self._apply_background()
        self._apply_fonts()
        # The isocurves are drawn directly in (log-)axis coordinates, so the view is
        # ranged on the grid bounds by hand — autorange sees no PlotDataItems here.
        self._plot.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self._kde_view_frame = (x0, x1, y0, y1)   # what Reset view repeats
        # ...AND THE USER'S PIN STILL WINS OVER THAT DEFAULT (owner note #4, and the
        # half of note #18 that is real: KDE is a plot type beside the others, so it
        # has to obey the same controls). The three other draws end with
        # `_finalize_view`, which auto-fits and then re-applies the stored limits; this
        # one could not use it, because `autoRange()` sees no PlotDataItem among the
        # isocurves and would leave the view wherever it was. So the grid-bounds
        # setRange above stands in for the auto-fit and the pin is re-applied after it,
        # exactly as `_finalize_view` does — without which typing an axis range and
        # pressing Apply silently did nothing on this one page.
        self._apply_stored_ranges(log_x, log_y)
        if drops or details:
            self._set_details("\n".join(details))

    def _draw_pca(self, df: pd.DataFrame) -> None:
        if self._pca_result is None:
            self._message("Select ≥2 feature columns on the left, then click Run PCA.")
            return
        res = self._pca_result
        scores = res.scores
        pc1, pc2 = scores[:, 0], scores[:, 1]

        # ONE VOICE FOR ONE DROP. `feature_matrix` discards any row with a non-finite
        # entry in a selected column, and this line used to read a bare "n = 1960 kept"
        # while the clustering pane of this same window, over the identical rows,
        # said "n = 1960 kept, 40 dropped (non-finite features)". The count comes from
        # the result itself (populations.pca stamps n_input/n_dropped), so the two
        # surfaces cannot drift apart again.
        kept_line = f"n = {res.n_kept} kept"
        dropped = int(res.meta.get("n_dropped", 0) or 0)
        if dropped:
            kept_line += f", {dropped} dropped (non-finite features)"
        details = [
            f"PCA on {res.columns}",
            kept_line,
            self._explained_variance_text(res),
            # ...and the fourth page that has to speak a refused y quantity, for the
            # reason `_draw_scatter` states: it is the configuration's, not the
            # histogram's -- and, since a configuration cannot carry a PCA RESULT, the
            # page-state a real import lands in is the EMPTY one above, which is why
            # `_message` asks for the same notes. See `_incoming_notes`.
            *self._incoming_notes(),
        ]
        # A SELECTED FEATURE THAT DID NOTHING IS NOT A SILENT NO-OP. A constant column
        # contributes no variance, so it cannot load on any component -- the projection
        # is the one the user would have got without ticking it, and nothing said so.
        flat = [c for c in res.meta.get("constant_columns", ()) if c]
        if flat:
            details.append(
                f"{', '.join(flat)} {'is' if len(flat) == 1 else 'are'} constant over "
                f"these {res.n_kept} rows and contributed nothing to either component: "
                f"this projection is the one the remaining "
                f"{len(res.columns) - len(flat)} feature(s) give on their own.")
        caveat = self._pca_dof_caveat(res)
        if caveat:
            details.append(caveat)
        if self._cluster_result is not None:
            labels = self._cluster_result.labels.reindex(
                res.kept_index, fill_value=populations.DROPPED_LABEL).to_numpy(dtype=int)
            brushes: list[QtGui.QBrush] | QtGui.QBrush = category_brushes(labels)
            details.append(self._cluster_summary())
        else:
            brushes = self._point_brush()

        self._clear_plot()
        self._scatter_item = pg.ScatterPlotItem(
            x=pc1, y=pc2, brush=brushes, pen=None, size=self._style.point_size)
        self._plot.addItem(self._scatter_item)
        self._scatter_items.append(self._scatter_item)
        self._scatter_item._presentation_role = "category" if self._cluster_result is not None else "data"
        primary = self.primary_source()
        if self._cluster_result is not None:
            self._scatter_item._presentation_cluster_labels = labels.copy()
            self._scatter_item._presentation_cluster_source = primary.key if primary else "pca"
        else:
            self._tag_series(self._scatter_item, "data", primary.key if primary else "pca",
                             primary.label if primary else "PCA")
        if primary is not None:
            self._cache_figure_source(primary)
            identities: list[object] = list(res.kept_index)
            if primary.kind == "detection" and "event_id" in df.columns:
                numeric = df.loc[:, list(res.columns)].apply(
                    pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                finite = np.all(np.isfinite(numeric), axis=1)
                candidate = self._point_identities(primary, df, finite)
                if len(candidate) == len(self._scatter_item.points()):
                    identities = candidate
            self._make_event_points_clickable(
                self._scatter_item, primary.key, identities)
        if self._cluster_result is not None:
            self._legend_clusters(labels)
        self._plot.setLogMode(x=False, y=False)
        evr = res.explained_variance_ratio
        self._plot.getAxis("bottom").setScale(1.0)   # clear any time scale from a prior draw
        self._plot.getAxis("left").setScale(1.0)
        # Through the same formatter as the title and Details -- an axis label is a
        # printed number too, and "PC1 (nan%)" is the same refusal-shaped hole.
        self._plot.setLabel("bottom",
                            self._remember_axis_label("bottom", f"PC1 ({self._evr_pct(evr[0])})"),
                            **self._label_style())
        self._plot.setLabel("left",
                            self._remember_axis_label("left",
                                f"PC2 ({self._evr_pct(evr[1])})" if evr.size > 1 else "PC2"),
                            **self._label_style())
        # Even one source can mix optional amplitude estimators across its rows.
        primary = self.primary_source()
        one = [primary] if primary is not None else []
        amplitude_columns = list(res.columns)
        if self._cluster_result is not None:
            amplitude_columns.extend(self._cluster_result.columns)
        amplitude_warnings = (amplitude_comparability([("PCA / clustering features", df)])
                              if "amplitude_pA" in amplitude_columns else ())
        details.extend(amplitude_warnings)
        details.extend(self._column_notes(list(res.columns), one))
        details.extend(self._detection_notes(one))
        # No explained-variance sentence in the title: the PC1/PC2 axis labels already
        # state it, and a title is the user's (the sentence is in Details, above).
        self._apply_title(None,
                          warn=self._warn_headline(
                              self._stale_headline(list(res.columns), one),
                              "CHECK SELECTED AMPLITUDES" if amplitude_warnings else ""))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(False, False)
        self._set_details("\n".join(details))

    # -- export ----------------------------------------------------------------------------------
    #
    # THREE FORMATS FOR THE PICTURE AND ONE FOR THE NUMBERS, and every one of them
    # through ``export_paths.ask_export_path`` (the overwrite seam) and a STAGED write
    # (``figures.save_plot_image`` / ``figures.save_plot_data``), so a refusal costs
    # the user nothing and the file already in the folder is never truncated.
    #
    # WHY SVG AND PDF ARE HERE AT ALL. The filter used to be ``PNG image (*.png);;All
    # files (*)``, and a ``.svg`` name typed through "All files" reached pyqtgraph's
    # SVG exporter, which raised on EVERY plot on this Qt (see ``plot/pg_svg.py``) --
    # so the one vector format the app appeared to offer was the one that could not
    # work. Wave 1 fixed the exporter; this is where it becomes reachable.
    #
    # AND WHY PDF IS NOT LEFT TO ``Analysis ▸ Export ▸ Generate Report figures…``. That command renders
    # six FIXED matplotlib figures (``figures.save_figure``) and cannot draw the live
    # view -- none of the interactive window's choices (the columns, the fit, the
    # clustering colours, the Style group) reach it. "SVG from here, PDF from the
    # publication figures" would therefore have meant "no PDF of the plot you are
    # actually looking at". ``figures.export_plot_pdf`` paints the live scene onto a
    # ``QPdfWriter`` instead, and on the measurements in that function's comment it is
    # the BETTER of the two vector formats for this panel's biggest plot type: a
    # 10 000-point scatter is 0.151 MB / 0.10 s as PDF against 5.28 MB / 3.06 s as SVG.

    #: What *Save image…* offers, in the order the dialog lists them. PNG stays first
    #: because it is the right answer for a slide or a lab notebook; the two vector
    #: formats are what a journal asks for.
    _IMAGE_FILTERS = ("PNG image (*.png);;SVG vector image (*.svg);;"
                      "PDF vector document (*.pdf);;All files (*)")

    def save_image(self, path: str | None = None) -> Path | None:
        """Export the current plot as a PNG, an SVG or a PDF, chosen by the extension.

        With ``path`` None, size/format and destination dialogs are shown; physical
        dimensions apply to PNG, PDF and SVG. Returns the written path, or None on
        cancellation/failure (reported in a dialog). Explicit paths preserve the
        legacy style.export_w/export_h contract: PNG pixels, PDF points, and SVG
        native viewport. Zero means native; one pinned dimension follows aspect.

        Anything that is not ``.svg`` or ``.pdf`` goes to ``ImageExporter``, which is
        what it was before -- so a ``.tif``/``.jpg`` name still works, and a format
        QImage cannot write still fails into ``image_write_failure_text`` with the
        user's previous file untouched.
        """
        options = None
        w, h = self._style.export_w, self._style.export_h
        if path is None:
            selection = choose_live_plot_export(
                self, self._plot, "plot.png", initial_width=w, initial_height=h,
                residual_item=self.figure_residual_target())
            if selection is None:
                return None
            chosen, options = selection
            path = str(chosen)
            w, h = options.width, options.height
        out = Path(path)
        suffix = out.suffix.lower()

        # STAGED, like every other export the guide describes (§12: "Every export
        # replaces its target the same way a project save does… so a failure leaves
        # the previous file whole rather than truncated"). pyqtgraph's ImageExporter
        # ends in ``QImage.save``, which OPENS AND TRUNCATES the file it is given and
        # then reports failure by RETURNING False — writing ``out`` directly zeroed
        # the user's previous export for any format QImage cannot write, and said
        # "The file was not created" over the wreckage. Rendering into a staged
        # sibling makes the refusal cost nothing, which is what lets
        # ``image_write_failure_text`` tell the truth about the file still in the
        # folder. Both halves are shared by all five panels (EXP3-06). SVG and PDF
        # join that same seam rather than growing their own: ``QSvgGenerator`` and
        # ``QPdfWriter`` open and truncate the file they are handed exactly as QImage
        # does, so an unstaged vector write would destroy the previous export in
        # precisely the way the staged raster one no longer can.
        from ..plot.figures import (export_plot_pdf, image_write_failure_text,
                                    save_plot_image)

        frozen = getattr(options, "frozen_snapshot", None)
        if frozen is not None:
            # Do not even construct a live exporter after a modal dialog: a
            # background result may have changed or replaced this canvas.
            render = None
        elif suffix == ".svg":
            # LIMITATION (b) OF THE SVG EXPORTER, SAID OUT LOUD BEFORE THE WAIT, NEVER
            # WORKED AROUND BEHIND THE USER'S BACK. pyqtgraph emits one ``<path>`` per
            # scatter symbol, so the file and the write grow linearly and fast (~528
            # bytes and ~1 ms per point, superlinear past ~20k). ``svg_size_warning``
            # is None below the threshold, so a histogram or a small scatter is never
            # interrupted. Nothing here subsamples the points or swaps the format: a
            # figure that silently lost events would be a worse outcome than a big
            # file, and a PNG written under the .svg name the user typed would be a
            # file the overwrite prompt never named.
            warning = pg_svg.svg_size_warning(self._plot)
            if warning is not None and not confirm_large_svg(self, warning):
                return None
            # LIMITATION (a) is handled INSIDE ``export_svg`` (its ``refresh``
            # argument, left at its default): ``SVGExporter`` paints items directly
            # and is the one render path in pyqtgraph that skips
            # ``GraphicsScene.prepareForPaint``, so on a plot no repaint has
            # reconciled — a window driven by ``set_plot_type``/``_run_plot`` and
            # never shown, which is exactly what a scripted or batch export is — the
            # ViewBox matrix still describes the default [0,1] range. Measured on this
            # panel's KDE: max |coordinate| 3 146 290 px on a 640 px canvas without the
            # pre-paint pass, 643.7 px with it. Not passing ``refresh=False`` from here
            # is therefore load-bearing, not a default nobody thought about.
            def render(staged):
                result = pg_svg.export_svg(self._plot, staged)
                if options is not None and not options.native:
                    apply_svg_physical_size(
                        staged, width_mm=options.physical_width_mm,
                        height_mm=options.physical_height_mm)
                return result
        elif suffix == ".pdf":
            def render(staged):
                return export_plot_pdf(self._plot, staged, width=w, height=h)
        else:
            exporter = ImageExporter(self._plot)
            params = exporter.parameters()
            if w > 0 and h > 0:
                # Both pinned: force EXACTLY that pixel size. ImageExporter locks the
                # aspect ratio -- setting one axis rewrites the other -- so a plain
                # ``params["width"]=…; params["height"]=…`` lets the height assignment
                # recompute (and clobber) the width. Set each with its own change-slot
                # blocked so both values stick verbatim.
                params.param("width").setValue(w, blockSignal=exporter.widthChanged)
                params.param("height").setValue(h, blockSignal=exporter.heightChanged)
            elif w > 0:
                params["width"] = w        # height follows the aspect ratio
            elif h > 0:
                params["height"] = h       # width follows the aspect ratio

            def render(staged):
                result = exporter.export(str(staged))
                if (result is not False and suffix == ".png"
                        and options is not None and not options.native):
                    apply_png_resolution(staged, dpi=options.dpi)
                return result

        try:
            if frozen is not None:
                from .figure_bundle import write_frozen_figure_image
                notes = write_frozen_figure_image(out, frozen, options)
            elif options is not None and getattr(options, "include_residual", False):
                from .plot_export import write_live_plot_image
                notes = write_live_plot_image(
                    out, self._plot, width=w, height=h,
                    residual_item=self.figure_residual_target(), include_residual=True,
                    physical_width_mm=None if options.native else options.physical_width_mm,
                    physical_height_mm=None if options.native else options.physical_height_mm,
                    dpi=None if options.native else options.dpi)
            else:
                notes = save_plot_image(out, render)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            # ...naming the three formats THIS window renders. The default of that
            # helper is PNG alone, which is the truth for the four panels that only
            # have ``ImageExporter``; from here it would send the user away from the
            # two vector formats they came for.
            QtWidgets.QMessageBox.warning(
                self, "Save plot image",
                image_write_failure_text(out, exc,
                                         formats=(".png", ".svg", ".pdf")))
            return None
        # What the replace changed about the FILE ITSELF — a symlink followed to
        # somewhere else, hard-linked names now holding the PREVIOUS contents. An
        # atomic replace leaves no trace of the link it broke, so this is the only
        # moment it can be said; shown here the way File ▸ Save Project shows it,
        # which is the promise USER_GUIDE §11 makes for exports too (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save plot image", "\n\n".join(notes))
        self._set_details(self.details_text() + f"\n\nSaved image → {out.name}")
        return out

    def export_data(self, path: str | None = None) -> Path | None:
        """Write the numbers behind the current plot as a CSV. The B2 half of *export*.

        With ``path`` None a file dialog is shown; returns the written path, or None
        when the user cancelled, when there is nothing plotted, or when the write
        failed (reported in a dialog, never returned as a path).

        THE FRAME IS BUILT FIRST, BEFORE THE DIALOG. An empty plot has no source data,
        and finding that out after the user has picked a filename -- and confirmed an
        overwrite of a file that then does not get written -- would be the worst
        possible order to discover it in.
        """
        data = self._plot_data()
        if not len(data):
            QtWidgets.QMessageBox.information(
                self, "Export plot data",
                "There is nothing plotted yet, so there are no numbers to export.\n\n"
                "Draw a plot first — then this writes the bars, points or density grid "
                "that are on screen, exactly as they are drawn.")
            return None
        if path is None:
            from ..plot.figures import plot_data_paths

            chosen = ask_export_path(
                self, title="Export plot data", suggestion="plot_data.csv",
                filters="CSV table (*.csv);;All files (*)",
                outputs=plot_data_paths)
            if chosen is None:
                return None
            path = str(chosen)
        # ``plot_data_paths`` applies the ``.csv`` suffix the dialog promised, so a
        # name typed through "All files (*)" without one still lands where the
        # overwrite prompt said it would -- the two must not be able to disagree.
        from ..plot.figures import plot_data_paths, save_plot_data

        out = plot_data_paths(path)[0]
        try:
            notes = save_plot_data(out, data)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export plot data",
                f"Could not write the plot data:\n{exc}\n\nNothing was written, so any "
                f"{out.name} already in that folder is untouched.")
            return None
        # The same disclosure the image export makes, for the same reason: an atomic
        # replace leaves no trace of the symlink or hard link it broke, so this is the
        # only moment it can be said (USER_GUIDE §11).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Export plot data", "\n\n".join(notes))
        self._set_details(
            self.details_text()
            + f"\n\nSaved plot data → {out.name} ({len(data)} row(s))")
        return out
