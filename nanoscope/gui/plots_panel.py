"""The interactive population-plots window: histograms, scatters, clustering, PCA.

This is the GUI face of :mod:`nanoscope.analysis.populations` (the pure, Qt-free
core, Phase 1) plus the distribution fits in :mod:`nanoscope.analysis.stats`. It
opens as its own top-level window from ``Analysis ▸ Plots…`` and explores the
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
from collections.abc import Mapping
from dataclasses import dataclass, field, fields as _dc_fields, replace as _dc_replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis import events, modules, populations, scoring, stats
from ..analysis._compat import trapezoid as _trapezoid
from ..analysis.cancel import OperationCancelled, cancel_scope, check_cancelled
from ..analysis.remeasure import REMEASURED_COLUMNS
from ..core import clock
from ..core.analysis import AnalysisResult
from ..core.session import Session
from ..core.sources import (
    PlotSource,
    SourceVerdict,
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
from .axes import ReadableLogAxis
from .categorical import category_brushes, category_colour, category_counts, legend_swatch
from .combos import choose_combo_item
from .export_paths import ask_export_path
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

# Columns the detector's own dwell provenance (min_dwell_s / max_dwell_s /
# filter_rise_s) actually describes. A histogram of rise_s or ecd_pA_s is not a
# dwell distribution, so it must not borrow the dwell detection floor.
_DWELL_COLUMNS = ("dwell_s", "dwell", "dwell_time_s")
#: Time columns that are an INSTANT rather than an interval, i.e. the ones that mean
#: nothing without a time origin. ``dwell_s``/``rise_s``/``fall_s`` are durations and
#: are origin-free; ``t_start_s``/``t_end_s`` are "seconds since THIS recording's first
#: loaded sample" (:mod:`nanoscope.core.clock`) and are the only ones this window can
#: pool across recordings that do not share a clock.
#:
#: TAKEN FROM THE MODULE THAT DECLARES IT, not restated here. ``analysis.scoring``
#: already owns this list (``RECORDING_CLOCK_COLUMNS``, "columns in the tables this
#: module writes that are INSTANTS on the recording clock") and ``gui/panels`` reads
#: the same set to badge the event-table header. A private copy is how two surfaces
#: end up disagreeing about which columns are on a clock -- the same reason the
#: absolute-current and baseline-dependent lists live in ``core.sources``.
_ORIGIN_COLUMNS: tuple[str, ...] = tuple(scoring.RECORDING_CLOCK_COLUMNS)
# ... and the columns the detector's depth floor (threshold_out_pA) describes.
_DEPTH_COLUMNS = ("frac_depth", "depth_pA", "max_dev_pA", "mean_dev_pA")
# NOTE: the absolute-current column list and the baseline-dependent column list both
# live in core.sources -- the module that owns the cross-source comparability rule
# (:func:`nanoscope.core.sources.source_comparability` reads them). Restating either
# here is how the two surfaces drifted apart in the first place.

# Plot-type labels (also the QComboBox item texts / QStackedWidget page order).
HISTOGRAM = "Histogram"
SCATTER = "Scatter"
PCA = "PCA"
KDE = "KDE (density)"
_PLOT_TYPES = (HISTOGRAM, SCATTER, KDE, PCA)

# Fit-overlay labels for the histogram.
_FIT_NONE = "none"
_FIT_EXP = "exponential"
_FIT_GMM = "gaussian-mixture"
#: Dwell families with a MODE. An exponential mixture is monotonically decreasing at every
#: component count, so it cannot describe a histogram that peaks away from the origin --
#: see stats.DWELL_PEAKED_MODELS. Truncated identically, so BIC compares directly.
_FIT_GAMMA = "gamma (peaked)"
_FIT_INVGAUSS = "inverse-gaussian (first passage)"
#: A fast escape AND a slow peaked passage in one model. Neither of the families above can
#: describe a BIMODAL dwell distribution whose slow lobe has a mode: an exponential mixture
#: is monotone in linear density, and a single gamma forced to explain a fast population as
#: well collapses to k ~ 1 (= an exponential). Measured on the owner's -600 mV apo file:
#: exp-mixture BIC -2062, gamma -2040, this -2682.
_FIT_EXP_GAMMA = "exponential + gamma (bimodal)"
#: The fits whose component count is a free choice. ``gamma``, ``inverse-gaussian`` and
#: ``exponential + gamma`` have a FIXED parameterisation, so a component spin — and an
#: Auto-BIC over it — means nothing for them.
_MULTI_COMPONENT_FITS = (_FIT_EXP, _FIT_GMM)

# Colour-by labels for the scatter AND (new) the histogram. "Source" is implicit:
# with >1 source checked the colour channel belongs to the source, always.
_COLOUR_NONE = "None"
_COLOUR_CLUSTER = "Cluster"

# Histogram y-quantity. Raw counts cannot be overlaid across recordings of
# different length; density and probability-per-bin can.
#
# WHY THIS ITEM IS NO LONGER CALLED "density (area = 1)" (report S6-03). The heights are
# ``counts / (n * bin_width)`` with the width in the column's STORED unit -- seconds for a
# dwell -- and ``_axis_label_for`` re-scales the x TICKS for display without touching them.
# It must not touch them: the exponential fit takes its detection floor from the detector's
# ``min_dwell_s`` in seconds, out of provenance, so scaling the sample would bias tau
# silently. Measured on the owner's -700 mV apo file (982 events, dwell_s, 30 bins) by
# driving this window offscreen, the curve as DRAWN integrates to
#
#     linear x:  1 (Time unit = s)   1000 (ms, the default)   1e6 (us)
#     log x:     23.3011485 -- IN ALL THREE UNITS
#
# and the log row is why no rescale was taken. On a log axis the abscissa is log10(s):
# ``AxisItem.scale`` stays 1, ``_DecadeLabelAxis.display_factor`` carries the unit, and the
# drawn area is a property of the DATA's range, not of any display factor. Dividing the
# heights by the display factor -- the obvious repair, prototyped and measured -- moves the
# log-x area to 0.0233011485 at ms and 2.33e-05 at us, i.e. FURTHER from 1, and costs a us
# y axis whose ticks render 0.00000 / 0.00005 / 0.00010 / 0.00015 on a publication figure.
# The shipped "Dwell-time histogram (log-x, exponential)" template below selects log x AND
# this mode AND ms, so the app's own preset lands on the unfixable case: the name asserted,
# on screen, something the same screen contradicted.
#
# The values are deliberately left alone; the FIGURE states its unit instead (see
# :meth:`PlotsWindow._y_axis_label`, which labels this axis ``density (1/s)`` /
# ``density (1/pA)``, and :meth:`PlotsWindow._density_unit_notes`, which says in Details
# where the drawn area is not 1 and by what factor -- and, on a log axis, that it is not a
# factor at all). "density" alone promises nothing that can be false, and costs nothing:
# measured, the combo's sizeHint stays 139 px and the window minimum 775 px, because
# "probability per bin" is still the widest item. ("density (area = 1 in stored units)" is
# also true and measures 221 px / 802 px -- it would make this the widest control here.)
_NORM_COUNT = "count"
_NORM_DENSITY = "density"
_NORM_PROB = "probability per bin"
_NORMS = (_NORM_COUNT, _NORM_DENSITY, _NORM_PROB)

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


# Time display units. The event schema stores every time in SECONDS, which is rarely
# the unit anyone reads a dwell in -- ms is.
#
# THIS IS A DISPLAY SCALE, NOT A DATA CHANGE. It is applied through
# ``AxisItem.setScale``, which multiplies the tick LABELS and leaves the plotted values
# untouched. That distinction is load-bearing: the exponential dwell fit takes its
# detection floor from the detector's own ``min_dwell_s`` -- in seconds, out of
# provenance, see this module's header -- so scaling the SAMPLE without scaling that
# floor would bias tau silently, which is precisely the failure the header warns about.
# An axis scale cannot do that, because the fit never sees it.
#
# EVERY TIME COLUMN, DURATIONS *AND* INSTANTS, DERIVED RATHER THAN LISTED. This tuple
# is what makes a column read in the Time-unit combo's unit -- the axis label and its
# scale (:meth:`PlotsWindow._axis_label_for`), the printed parameters
# (:meth:`_display_time`) and the Fit-range boxes, which interpret what the user types
# in that same unit. Spelling the four names out left `t_end_s` behind on all three:
# it is an INSTANT this window already annotates with its clock in Details
# (:data:`_ORIGIN_COLUMNS`), yet its axis read a bare "t_end_s" -- no t₀ and no unit --
# while `t_start_s` beside it read "t_start (ms, from recording start (t₀ = 61.3034 s))".
# The axis label is the half an exported image keeps, so that is the half that mattered.
# Written as durations + :data:`_ORIGIN_COLUMNS` so an instant column added to the
# shared declaration in ``analysis.scoring`` arrives here already carrying its unit and
# its clock, instead of inheriting this hole in silence.
_DURATION_COLUMNS = ("dwell_s", "rise_s", "fall_s")
_TIME_COLUMNS: tuple[str, ...] = _DURATION_COLUMNS + _ORIGIN_COLUMNS
_TIME_UNITS: dict[str, tuple[float, str]] = {
    "s": (1.0, "s"), "ms": (1e3, "ms"), "µs": (1e6, "µs"),
}

#: What each EVENT COLUMN IS IN, taken from the schema that declares it rather than
#: guessed from its name. ``events.SCHEMA`` has carried a unit per column since the
#: file was written; this window was reading none of them (see ``_column_quantity``).
#:
#: Two translations, both because these strings are pasted straight onto a number:
#: the schema's ``"-"`` for dimensionless becomes ``""`` (" -" beside a value reads as
#: a minus sign), and ``"pA*s"`` becomes the ``pA·s`` this window already prints.
#: Everything else is passed through exactly as declared, ``"sample"`` included -- an
#: index IS in samples, and saying so is the difference between a labelled number and
#: a bare one.
_SCHEMA_UNITS: dict[str, str] = {
    col: ("" if unit == "-" else "pA·s" if unit == "pA*s" else unit)
    for col, (_dtype, unit, _desc) in events.SCHEMA.items()
}
_DEFAULT_TIME_UNIT = "ms"

#: Where a legend that has never been DRAGGED sits: 12 px in from the top-right corner,
#: which is where this window has always put it.
#:
#: THE ONLY TIME THIS IS APPLIED IS WHEN ``StyleSettings.legend_pos`` IS NONE, and that
#: restriction is the whole of round-4 Q4. pyqtgraph 0.14's ``LegendItem`` is draggable
#: out of the box -- ``LegendItem.mouseDragEvent`` calls ``autoAnchor`` on every mouse
#: move -- so the user could already put the legend anywhere they liked. A previous
#: round added a four-corner combo and implemented it by calling ``setOffset`` inside
#: :meth:`PlotsWindow._ensure_legend`, i.e. on EVERY draw; ``setOffset`` re-runs
#: ``anchor()``, so the next redraw threw the dragged position away. Measured on the
#: owner's two-source dwell overlay: a drag moved the legend from (487, 12) to
#: (287, 192) and the very next redraw put it back at (486.7, 12).
#:
#: So a corner is a STARTING POINT now, not a setting: once the user has moved the
#: legend, this constant is never consulted again until they press *Reset position*.
_LEGEND_DEFAULT_OFFSET: tuple[int, int] = (-12, 12)

#: The controls column's minimum width, and the width the splitter opens at. Both are
#: MEASURED quantities rather than round numbers: the first is the widest
#: ``minimumSizeHint`` the six control groups produce once the histogram page's Fit row
#: is unstacked, and the second is that plus a little air so nothing is at its limit on
#: a fresh window. ``tests/gui/test_plots_layout_room.py`` asserts the relationship the
#: old code got backwards -- the maximum was 24 px NARROWER than the content minimum --
#: so it cannot come back.
_CONTROLS_MIN_PX = 240
#: Air between the column's minimum and the width it OPENS at, so a fresh window is not
#: sitting on its own limit (which is what made the 420/444 mismatch invisible: the
#: column was already as wide as it could go).
_CONTROLS_SLACK_PX = 24

#: How long a combo has to be able to show, in characters, before it is allowed to stop
#: growing. ``QComboBox`` sizes itself to its WIDEST ITEM by default, so a single
#: 33-character entry ("inverse-gaussian (first passage)") set the width of the whole
#: controls column and, through it, of the horizontal scrollbar that would not go away.
#: The full text is still readable -- the popup is as wide as it needs to be, and the
#: tooltip spells every option out.
_COMBO_CHARS = 14

#: What an un-run window says on its canvas (round-4 Q1). The instruction is the whole
#: point: an empty plot with no words on it reads as a bug, and this window's empty
#: states have always said what to do next ("No data yet — run detection…", "Tick a
#: source on the left…") rather than merely being empty.
_PLOT_PROMPT = ("Choose your data and plot type, then press Plot.")

#: Free parameters in EITHER scatter best fit: (slope, intercept) for the straight
#: line, (coefficient, exponent) for the log-log one. What
#: :func:`nanoscope.analysis.stats.r2_dof_tag` subtracts from n to decide whether the
#: printed R² has any residual degrees of freedom left to be evidence over.
_SCATTER_FIT_PARAMS = 2

# Best-fit-line labels for the scatter (GUI text -> populations.linear_fit mode).
_SFIT_NONE = "none"
_SFIT_LINEAR = "linear"
_SFIT_POWER = "power law (log-log)"

# Clustering method labels (GUI text -> populations method id).
_METHOD_GMM = "GMM"
_METHOD_KMEANS = "k-means"
_METHOD_IDS = {_METHOD_GMM: "gmm", _METHOD_KMEANS: "kmeans"}

# k-selection modes for GMM.
_K_AUTO = "Auto (BIC)"
_K_FIXED = "Fixed k"

# Columns worth reaching for first when seeding the pickers, in priority order.
_PREFER_DWELL = ("dwell_s", "dwell", "rise_s", "fall_s")
_PREFER_AMP = ("amplitude", "depth_pA", "frac_depth", "frac_amplitude", "max_dev_pA")
_PREFER_FEATURES = (
    "dwell_s", "depth_pA", "frac_depth", "amplitude", "frac_amplitude",
    "max_dev_pA", "ecd_pA_s", "snr", "rise_s", "fall_s", "kurtosis", "skewness",
    "rms",
)

# Drawing colours (Okabe-Ito, reused for colour-blind-safe cluster colours).
_DATA_HEX = OKABE_ITO[0]        # blue -- histogram bars / plain scatter
_FIT_HEX = OKABE_ITO[1]         # vermillion -- fitted overlay
#: Orange -- the NOT-COMPARABLE verdict in the plot title. The same ink the
#: voltage-dependence window's banner uses, so one verdict reads as one verdict
#: wherever the user meets it.
_WARN_HEX = OKABE_ITO[4]
_EXCLUDED_HEX = PALETTE["excluded"]   # below-detection-limit shading (same ink as figures.py)
_DROPPED_HEX = "#9A9A9A"        # rows dropped from a clustering (label -1)
_MSG_HEX = "#8b9199"            # muted -- empty/invalid messages (on the dark theme)
_MSG_LIGHT_HEX = "#555b63"      # the same, darkened so it reads on a white background

# Fill alphas: the data colour is opaque as a hex, but the histogram fill and the plain
# scatter/PCA points draw it translucent (matching the old hard-coded mkBrush alphas).
_HIST_FILL_ALPHA = 110
_POINT_ALPHA = 160

# Theme foreground/background. "dark" reproduces the app theme exactly (gui.app), so the default
# style is byte-for-byte today's look; "light" is a white publication canvas with dark ink.
_BG_DARK = "#111316"
_FG_DARK = "#b9bfc7"
_BG_LIGHT = "#ffffff"
_FG_LIGHT = "#1a1a1a"


class FrozenMap(Mapping):
    """A read-only, hashable mapping -- the collection type a frozen style may hold.

    WHY THIS EXISTS RATHER THAN A PLAIN ``dict``. ``StyleSettings`` is frozen so a
    snapshot of it is a VALUE: hand one out, change the window, and the snapshot still
    describes the moment it was taken. ``frozen=True`` alone does not deliver that,
    because it only forbids rebinding the FIELDS -- a ``dict`` field stays shared, and
    ``dataclasses.replace()`` copies the reference, not the mapping. Measured on the
    pre-freeze class: ``replace(s, title="x").label_overrides is s.label_overrides`` was
    True, so writing one override through the copy also wrote it into the "old" style,
    and an undo stack built on that would have had every entry aliasing one live dict.

    Reading is a plain mapping (``m["k"]``, ``"k" in m``, ``m.get(k, default)``,
    ``m == {}``), which is why the ~6 read sites and the tests that compare against a
    literal dict did not have to change. Writing is by descendant: :meth:`set` and
    :meth:`remove` return a NEW map and leave this one alone.
    """
    __slots__ = ("_data",)

    def __init__(self, data: Mapping | Iterable = ()) -> None:
        # COPIED, never adopted. Building on the caller's dict would leave them a handle
        # that edits this "immutable" map from outside -- the same aliasing bug one
        # level down. ``__slots__`` above keeps the copy the only state there is.
        self._data: dict = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        # Compare equal to an ordinary dict with the same items, so a test may keep
        # asserting ``style.label_overrides == {}`` and a caller may keep using dicts.
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        # ``Mapping`` sets ``__hash__ = None``; a frozen dataclass holding this field
        # needs it back, and order-independence is what ``frozenset`` buys here.
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def set(self, key, value) -> "FrozenMap":
        """This map plus ``key -> value``, as a new map (this one is untouched)."""
        return FrozenMap({**self._data, key: value})

    def remove(self, key) -> "FrozenMap":
        """This map without ``key``, as a new map. A missing key is not an error --
        the call sites are "forget this override", which is idempotent by nature."""
        if key not in self._data:
            return self
        rest = dict(self._data)
        del rest[key]
        return FrozenMap(rest)


#: The one empty map every default style shares. Safe to share precisely because it
#: cannot be written to -- which is the whole point of the type.
_NO_ENTRIES: FrozenMap = FrozenMap()


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
    font_family: str = ""           # "" => the application default family; the Font row's
                                    # first entry names it (see _FONT_DEFAULT_LABEL). NOT a
                                    # missing value -- it round-trips through JSON, undo
                                    # and templates as itself.
    font_size_pt: int = 10
    data_color: str = _DATA_HEX
    fit_color: str = _FIT_HEX
    point_size: int = 7
    line_width: int = 1
    background: str = "dark"         # "dark" | "light"
    export_w: int = 0               # 0 => native width
    export_h: int = 0               # 0 => native height
    #: Owner note #14: the fitted parameters as a text box ON the figure, so a saved
    #: image carries its own numbers (Details is not exported). Off by default — a
    #: box appearing on every histogram unasked is clutter, not presentation.
    fit_values_on_plot: bool = False
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
    legend_font_size_pt: int = 0
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
    title_font_size_pt: int = 0
    axis_label_font_size_pt: int = 0
    tick_font_size_pt: int = 0
    #: Title weight/slant. Cheap because the title is the one text item this window
    #: already sets a real ``QFont`` on (the CSS span carries size and colour only).
    title_bold: bool = False
    title_italic: bool = False
    #: THE FIT-STATISTICS SWITCHES. Two of these were already COMPUTED and printed with
    #: no way to turn them off, and two are new pictures that must not appear unasked --
    #: hence the asymmetric defaults, which between them reproduce today's figure exactly.
    #:
    #: A SWITCH THAT CAN HIDE A REFUSAL WOULD BE A BUG, not a preference, so neither of
    #: the first two can. ``show_r2`` and ``show_gof`` govern the FIGURE -- the title, the
    #: legend entry and the on-plot values box, which is what a paper figure has to be
    #: able to be clean of -- and Details keeps every sentence that says the number is not
    #: evidence: an R² with no residual degrees of freedom, an undefined R², and a KS
    #: verdict that REJECTED the curve. Turning the statistic off is a request to stop
    #: decorating the picture, never a request to be told a fit is good when it is not.
    #:
    #: WHAT ``show_gof`` ACTUALLY GOVERNS, so this paragraph cannot promise more than the
    #: code does again (S7-JRN2): the KS statistic and the PASSED verdict appear in the
    #: on-plot values box when ``fit_values_on_plot`` asks for that box, and nowhere else
    #: on the figure -- a passing fit's title and legend carry the parameter and ``n``
    #: only. A REJECTED verdict is written into the title and the legend entry
    #: themselves (:data:`_KS_REJECTED_TAG`) and into the values box, whatever this
    #: switch and ``fit_values_on_plot`` are set to -- and ONE of those three cannot be
    #: switched off or typed over: the title takes the verdict through
    #: :meth:`PlotsWindow._apply_title`'s ``warn`` channel, the one the comparability
    #: verdict already uses, which is PREPENDED to whatever the title would otherwise
    #: be, a typed Style ▸ Title included (A1-N1: with the automatic title carrying the
    #: tag instead, a typed title plus "Show legend" off shipped the owner's rejected
    #: -600 mV exponential with no verdict anywhere on the saved SVG). The legend entry
    #: carries it while the legend is shown; the values box while that box is.
    #: Measured before any of this: at the shipped dwell template's defaults
    #: (``show_gof`` True, ``fit_values_on_plot`` False) the exported title and legend
    #: carried "τ = 20.41 ms (n = 275)" for a curve rejected at p = 1.6e-39 and no
    #: verdict, because the values box -- the only surface the verdict reached -- was off.
    show_r2: bool = True
    show_gof: bool = True
    #: The bootstrap's confidence BAND around the fitted curve. Off: it is a second
    #: object on the figure and, unlike the parameters' intervals, it does not exist for
    #: every fit (see :func:`nanoscope.analysis.stats.ci_band` for the four ways a fit can
    #: have none) -- a control that silently does nothing on half the fits is worse
    #: defaulted on.
    show_ci_band: bool = False
    #: E2 -- WHAT MADE THESE EVENTS, written on the figure: the detector, the ML model
    #: where there is one, and the preprocessing chain that ran before it. Off by default
    #: for the reason ``fit_values_on_plot`` is: it is an annotation, and one appearing on
    #: every plot unasked is clutter. It is drawn as a plot item so ``Save image…`` keeps
    #: it -- which is the whole point, since the PNG is what leaves the app and the
    #: provenance is what a reader of it cannot otherwise recover.
    show_provenance: bool = False
    #: The residual strip under the histogram, and the reduced χ² that goes with it. Off
    #: for the same reason the fit-values box is: a second axis appearing on every fitted
    #: histogram unasked is clutter, not presentation.
    show_residuals: bool = False

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
    """
    plot_type: str = HISTOGRAM
    x_column: str = ""
    y_column: str = ""
    fit_column: str = ""
    time_unit: str = _DEFAULT_TIME_UNIT

    @property
    def axis_factors(self) -> tuple[float, float]:
        """(x, y) factors converting the DISPLAYED axis unit back to data units."""
        return (_display_factor(self.x_column, self.time_unit),
                _display_factor(self.y_column, self.time_unit))

    @property
    def fit_factor(self) -> float:
        """The factor the Fit-range boxes' text is written in."""
        return _display_factor(self.fit_column, self.time_unit)


def _unit_factor(time_unit: str) -> float:
    """Data-units-per-display-unit for the Time-unit combo's choice (s -> 1.0, ms -> 1e3)."""
    return _TIME_UNITS.get(time_unit, (1.0, "s"))[0]


def _display_factor(col: str, time_unit: str) -> float:
    """The factor ``col``'s axis is DISPLAYED in: the time unit for a time column, else 1.

    A non-time column's axis shows data units, so its typed limits need no conversion --
    which is exactly why a number typed for a dwell axis is nonsense over a picoamp one
    and :meth:`PlotConfig.rebased` clears it rather than converting it.
    """
    return _unit_factor(time_unit) if col in _TIME_COLUMNS else 1.0


def _rescaled(text: str, old_factor: float, new_factor: float) -> str:
    """A typed bound, rewritten from ``old_factor``'s display unit into ``new_factor``'s.

    ``repr`` rather than a format spec because these are round-trip values, not a
    readout: "0.1" ms -> "100.0" µs has to come back as exactly 1e-4 s when the box is
    read again. Anything that is not a finite number is left verbatim -- a half-typed
    "0." is not a bound yet, and "nan"/"inf" are reported to the user as typed by
    :meth:`PlotsWindow._fit_range_input_notes`.
    """
    if old_factor == new_factor or not text.strip():
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if not np.isfinite(value):
        return text
    return repr(value * new_factor / old_factor)


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
        else:
            x, y = "", ""
        return AnalysisIdentity(plot_type=self.plot_type, x_column=x, y_column=y,
                                fit_column=self.histogram.column,
                                time_unit=self.axes.time_unit)

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
        return self

    def rebased(self, was: AnalysisIdentity) -> "PlotConfig":
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

        A plot-type change counts as a move on BOTH axes even when a column name survives
        it (Histogram dwell_s -> Scatter dwell_s vs depth_pA): the y axis stopped being a
        count, and a range typed against one pair of axes is not a range against another.
        """
        now = self.identity
        if now == was:
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

        # (2) THE QUANTITY ON AN AXIS MOVED: text and pin go together.
        if now.plot_type != was.plot_type or now.x_column != was.x_column:
            axes = _dc_replace(axes, x_min="", x_max="", manual_x=None)
        if now.plot_type != was.plot_type or now.y_column != was.y_column:
            axes = _dc_replace(axes, y_min="", y_max="", manual_y=None)

        # (3) THE FITTED COLUMN MOVED: the truncation window is not transferable.
        if now.fit_column != was.fit_column:
            fit = _dc_replace(fit, range_lo="", range_hi="", range_unit=now.time_unit)

        # (4) THE ANALYSIS MOVED: the caption goes with it (owner note Q6).
        if (now.plot_type, now.x_column, now.y_column) != (
                was.plot_type, was.x_column, was.y_column):
            style = _dc_replace(style, title="", x_label="", y_label="")

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
        keep_label = {k: v for k, v in self.style.label_overrides.items() if k in keys}
        keep_include = {}
        for entry_id, included in self.style.legend_include.items():
            kind, _, key = str(entry_id).partition(":")
            if kind in ("series", "fit"):
                alive = key in keys
            else:
                alive = entry_id in offered or not judged_clusters
            if alive:
                keep_include[entry_id] = included
        if (keep_label == dict(self.style.label_overrides)
                and keep_include == dict(self.style.legend_include)):
            return self
        return _dc_replace(self, style=_dc_replace(
            self.style, label_overrides=FrozenMap(keep_label),
            legend_include=FrozenMap(keep_include)))

    # -- A3: the TEMPLATE projection -------------------------------------------------------------
    #: What a saved template deliberately forgets, as names, so the tooltip beside the
    #: combo, :meth:`as_template` and ``tests/gui/test_plots_templates_json.py`` read ONE
    #: list instead of three copies that can disagree about where the seam is. Not a
    #: dataclass field (no annotation), so ``test_no_derived_state_ever_enters_the_config``
    #: still sees exactly the ten fields it walks.
    TEMPLATE_DROPS = ("selection.source_keys", "fit.applied", "fit.applied_k",
                      "fit.applied_k_auto", "fit.applied_scatter",
                      "style.label_overrides", "style.legend_include")

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
          entry retyped for "apo -500 mV" is not a legend entry for anything else.

        EVERYTHING ELSE IS CARRIED, and three of those look like data selection and are
        not:

        * THE COLUMN NAMES (``histogram.column``, the scatter/KDE axes,
          ``cluster.features``). They are SCHEMA columns -- every event table this app
          produces carries ``dwell_s`` -- and they are the substance of a recipe: "a
          log-x dwell histogram under an exponential fit" is not a template without
          them. A blank one means "keep what the window has" and a column the recording
          does not offer is resolved back to the window's own, both at install time
          (:meth:`PlotsWindow._installable`), so a presentational-only template can be
          saved by leaving the pickers alone and neither kind can leave the identity
          anchor pointing at a column that is not on screen.
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
                              legend_include=_NO_ENTRIES))

    def with_template(self, template: "PlotConfig") -> "PlotConfig":
        """``template`` recalled onto this config: the template's everything, over THIS
        window's data selection and legend memory.

        The inverse of :meth:`as_template`, written as one function rather than as a
        ``replace`` at the call site because the two halves have to agree about where the
        seam is: a field :meth:`as_template` drops has to be taken from ``self`` here, or
        recalling a template would install that field's DEFAULT -- unticking every source
        -- rather than leaving it alone.

        A BLANK COLUMN MEANS "KEEP WHAT YOU HAVE", not "no column". That is what makes a
        purely presentational template possible: save one from a window whose pickers you
        never touched and it carries a look and nothing else. It also keeps the identity
        anchor honest, because a config naming a column the pickers cannot select is
        exactly the stale-anchor shape :meth:`PlotsWindow._set_config` warns about.
        """
        tpl = template.as_template()    # idempotent, and it sanitises a hand-edited file
        hist = tpl.histogram if tpl.histogram.column else _dc_replace(
            tpl.histogram, column=self.histogram.column)
        scatter = _dc_replace(tpl.scatter, x=tpl.scatter.x or self.scatter.x,
                              y=tpl.scatter.y or self.scatter.y)
        kde = _dc_replace(tpl.kde, x=tpl.kde.x or self.kde.x,
                          y=tpl.kde.y or self.kde.y)
        cluster = tpl.cluster if tpl.cluster.features else _dc_replace(
            tpl.cluster, features=self.cluster.features)
        return _dc_replace(
            tpl, histogram=hist, scatter=scatter, kde=kde, cluster=cluster,
            selection=_dc_replace(tpl.selection,
                                  source_keys=self.selection.source_keys),
            style=_dc_replace(tpl.style,
                              label_overrides=self.style.label_overrides,
                              legend_include=self.style.legend_include))



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
PLOT_CONFIG_VERSION: int = 2
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
    "cluster": ClusterConfig,
    "fit": FitConfig,
    "axes": AxesConfig,
    "style": StyleSettings,
}


def _encode_dc(obj) -> dict:
    """One flat frozen dataclass -> a JSON-safe dict, driven by its own FIELDS.

    Field-driven rather than hand-listed, because a hand-listed encoder is how half a
    config comes to be persisted: add ``title_italic`` to :class:`StyleSettings` and a
    listed encoder keeps writing files that silently lack it, while this one carries it
    the moment it exists. Only three shapes need translating -- a :class:`FrozenMap`
    becomes a plain dict, a tuple becomes a list (JSON has no tuple), and everything else
    is already a str/int/bool/None.
    """
    out: dict = {}
    for f in _dc_fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, FrozenMap):
            out[f.name] = {str(k): value[k] for k in value}
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
        return FrozenMap(raw or {})
    if isinstance(default, tuple):
        return tuple(raw)
    if default is None:
        # The three optional pairs -- `legend_pos`, `manual_x`, `manual_y` -- are the only
        # fields whose default is None, and all three are a 2-tuple of floats or nothing.
        return None if raw is None else tuple(float(x) for x in raw)
    if isinstance(default, bool):
        return bool(raw)        # BEFORE int: bool is a subclass of it
    if isinstance(default, int):
        return int(raw)
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
    kw = {f.name: _decoded(getattr(blank, f.name), payload[f.name])
          for f in _dc_fields(blank) if f.name in payload}
    return _dc_replace(blank, **kw) if kw else blank


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
    for name in _CONFIG_SECTIONS:
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
    # THE ONE MIGRATION THIS FORMAT HAS (S6-03's rename) HAS ALREADY RUN BY HERE, and
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
        if key not in pivot:
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


# ================================================================================================
# UNDO / REDO -- a LOCAL, bounded stack of config values (owner item A2)
# ================================================================================================

#: How many past configurations one Plots window keeps.
#:
#: MEASURED, not guessed. A ``PlotConfig`` is 1,446 bytes fully expanded (every nested
#: dataclass and every string counted) and 1,584 with a typed title and two axis labels,
#: so this depth costs about 320 kB -- three orders of magnitude under
#: :data:`nanoscope.core.history.MAX_BYTES`, which is why this stack needs no byte ceiling
#: where the event-table one needs one first. That IS the difference between the two: an
#: event snapshot is a deep-copied DataFrame (38.7 MiB on a 100k-event table), a config
#: snapshot is forty scalars, and comparing two of them costs 0.4 us.
#:
#: Bounded anyway, because this window is constructed ONCE and re-shown for the whole
#: session (see :meth:`PlotsWindow.reset_state`), so an unbounded stack is a leak that
#: grows with every colour click of every figure made that day. 200 is deliberately far
#: above :data:`nanoscope.core.history.MAX_DEPTH` (30 per event table): walking back
#: thirty curation decisions is unusual, walking back thirty control moves while tuning
#: one figure is an ordinary afternoon.
_HISTORY_DEPTH = 200

#: How many origins' records the two import disclosures (:class:`_NormRefusal`,
#: :class:`_ConfigLosses`) keep before the oldest is dropped. A record is spoken only
#: while its origin is the one on screen, and the history can reach at most the base,
#: ``_HISTORY_DEPTH`` past entries, ``_HISTORY_DEPTH`` future entries and one pending
#: origin -- so this bound can never drop a record an undo or a redo could still land
#: on. It is a memory ceiling on the INERT ones: a template recalled five hundred times
#: opens five hundred origins and pushes no step at all.
_ORIGINS_KEPT = 2 * _HISTORY_DEPTH + 2


def _drop_oldest(records: dict) -> None:
    """Hold a per-origin record dict to :data:`_ORIGINS_KEPT` entries, oldest first
    (dicts keep insertion order; both records re-insert a re-taken origin at the end)."""
    while len(records) > _ORIGINS_KEPT:
        del records[next(iter(records))]

#: How long a run of KEYSTROKES may stay open before it becomes one undo entry. See
#: :meth:`PlotsWindow._history_touch_text` for the whole grain rule; this number only
#: decides how long the user must pause MID-TITLE, without leaving the box, before
#: "figure" and "figure 3b" become two entries instead of one.
_HISTORY_COALESCE_MS = 700


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


#: The symbol a fitted parameter is WRITTEN as, by the name ``stats`` stores it under.
#: ``FitResult.params`` keys are ASCII (``tau_1``, ``mu_2``, ``sigma_1``, ``w_2``)
#: because they travel through JSON into a saved project; the plot title has always
#: rendered the headline one as Greek, and the on-plot values box has to use the SAME
#: alphabet or one figure names a quantity two ways in two corners of itself.
_PARAM_SYMBOLS: dict[str, str] = {
    "tau": "τ", "mu": "μ", "sigma": "σ", "lambda": "λ", "theta": "θ",
    "alpha": "α", "beta": "β", "w": "w", "k": "k",
}


def _param_symbol(key: str) -> str:
    """``"tau_1"`` -> ``"τ_1"``; anything unrecognised is passed through untouched."""
    head, sep, tail = str(key).partition("_")
    return f"{_PARAM_SYMBOLS.get(head, head)}{sep}{tail}"


#: What the one-line fit summary -- the legend entry's text and Details' summary line
#: -- carries after the headline parameter when the KS test has REJECTED the drawn
#: curve, and what the title's warning headline is built from
#: (:meth:`PlotsWindow._ks_headline`). SIXTEEN CHARACTERS, measured against
#: :meth:`PlotsWindow._elide_legend`: a legend entry longer than 34 characters is
#: middle-elided to its first 17 and last 16, so a tag of 16 survives whole at the end
#: of "fit: τ = 20.41 ms…KS REJECTED (1%)" where the 17-character "KS REJECTED at 1%"
#: came out as "…S REJECTED at 1%" on the owner's -600 mV dwell fit. The full statistic
#: and the full sentence stay in the values box and in Details.
_KS_REJECTED_TAG = "KS REJECTED (1%)"


class _FitSummary(str):
    """A fit's one-line summary, carrying the FULL parameter set alongside it.

    Owner note #13 wants every fitted number on the plot, and owner note #14's box is
    where they go -- but the one-line form is still what the title and the legend
    take, because a title has one line's worth of room and a legend entry less. So one
    fit produces two strings.

    A ``str`` SUBCLASS rather than a third element of ``_overlay_fit``'s return tuple:
    the summary is consumed as text at five sites (the title, the legend, Details, the
    overlay list, the pending-fit placeholder) and every one of them keeps working
    unchanged, including callers outside this module that unpack ``(summary, notes)``.
    The richer form is read off ``.full`` by the one caller that needs it.

    ``rejected`` and ``plain`` are the KS verdict's two halves (A1-N1). The string
    itself carries :data:`_KS_REJECTED_TAG` when the test threw the curve out -- that
    is the legend entry's text and Details' summary line. The TITLE does not take the
    tagged string: it takes ``plain`` (the summary with no tag) as its automatic text
    and the verdict through :meth:`PlotsWindow._apply_title`'s ``warn`` channel, which
    is PREPENDED to a typed title rather than displaced by it. ``rejected`` is what the
    histogram reads to build that headline.
    """

    full: str = ""
    rejected: bool = False
    plain: str = ""

    def __new__(cls, text: str, full: str = "", *, rejected: bool = False,
                plain: str = ""):
        obj = super().__new__(cls, text)
        obj.full = full or text
        obj.rejected = bool(rejected)
        obj.plain = plain or text
        return obj


def _narrow_combo(combo: QtWidgets.QComboBox, chars: int = _COMBO_CHARS) -> None:
    """Stop a combo from sizing the whole controls column to its longest item.

    ``QComboBox.minimumSizeHint`` is the width of its WIDEST entry, which is why five
    fit families -- one of them "inverse-gaussian (first passage)" -- were between them
    responsible for a 444 px minimum on a 420 px column and the permanent horizontal
    scrollbar that followed. ``AdjustToMinimumContentsLengthWithIcon`` caps the CLOSED
    width at ``chars`` characters; the popup still opens at full width, so no option
    becomes unreadable.
    """
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)


#: What the Font row's first entry is called, and the value it stands for.
#:
#: ``StyleSettings.font_family`` defaults to ``""`` = "whatever font this computer hands
#: the application", and that is a value NO ``QFont`` can spell: ``QFont()`` reports
#: ``.AppleSystemUIFont`` on macOS -- a PRIVATE family ``QFontComboBox`` filters out of
#: its own list -- so ``setCurrentFont(QFont())`` fell through to the combo's first entry
#: and the Style panel told the user their figure was in "Academy Engraved LET" while it
#: was drawn in the system UI font (sweep-6 S6-01; measured under cocoa AND offscreen).
#: The blank is not a missing value to be papered over -- it is a CHOICE the user can
#: make and unmake -- so it gets a row of its own that says so in words.
_FONT_DEFAULT_LABEL = "Application default"


def _font_families() -> list[str]:
    """Exactly the families ``QFontComboBox`` lists, in the same order.

    MEASURED, not guessed: on this machine ``QFontDatabase.families()`` returns 201 names
    and ``QFontComboBox`` shows 181 of them; the twenty it drops are the ones a leading
    ``.`` or ``isPrivateFamily`` marks as system-internal, and this expression reproduces
    its list element-for-element (set difference empty, order identical). Spelled out here
    because the Font row can no longer BE a ``QFontComboBox`` (see
    :data:`_FONT_DEFAULT_LABEL`), and the one thing a replacement must not do is quietly
    offer a different set of fonts than this app always has.

    (Re-measured under offscreen too, where the two lists agree at 183 rather than 181 --
    constructing a ``QFontComboBox`` there populates two alias families into the database
    as a side effect, so compare the two AFTER building one or the counts differ by those
    two aliases alone. The expression still reproduces whatever the combo would show.)
    """
    return [f for f in QtGui.QFontDatabase.families()
            if not f.startswith(".") and not QtGui.QFontDatabase.isPrivateFamily(f)]


def _swatch_text_color(hex_color: str) -> str:
    """Black or white ink for a colour swatch button, whichever is legible on it."""
    c = QtGui.QColor(hex_color)
    luma = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if luma > 140 else "#ffffff"


#: Colour for a cluster id; the dropped-row sentinel (-1) renders grey.
#:
#: THE ALIAS IS THE POINT. This used to be a private four-line function here, and
#: the subpopulation report window needed the identical map for its embedding
#: scatter -- a second copy of which would have been this project's signature
#: failure in colour. There is now one map, in ``gui.categorical``, and a test
#: asserts this name *is* that function rather than a copy of it.
_cluster_color = category_colour


def _numeric_columns(df: pd.DataFrame | None) -> list[str]:
    """Numeric, non-boolean columns of ``df`` that carry at least one finite value.

    Booleans (``accepted``, ``filter_limited``) are excluded -- they are flags,
    not quantities to histogram or cluster on. All-NaN numeric columns are
    dropped too, so the pickers never offer a column that can only draw an empty
    plot.
    """
    if df is None or df.shape[1] == 0:
        return []
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(v).any():
            out.append(str(c))
    return out


def _detection_key(result) -> tuple | None:
    """``(doc_id, channel, module_id)`` of a stored ``AnalysisResult``, or None.

    Duck-typed, because the only caller (:meth:`PlotsWindow._active_source_key`) is
    reaching across a window boundary at something it must not assume the shape of.
    """
    if result is None:
        return None
    doc_id = getattr(result, "doc_id", None)
    channel = getattr(result, "channel", None)
    module_id = getattr(result, "module_id", None)
    if doc_id is None or channel is None or not module_id:
        return None
    return (str(doc_id), int(channel), str(module_id))


def _normalize_sources(sources, provenance: dict | None = None) -> dict[str, PlotSource]:
    """Coerce either accepted ``set_data`` shape into ``{key: PlotSource}``.

    The owner pushes real :class:`~nanoscope.core.sources.PlotSource` objects (each
    already carrying its own provenance, colour and stable key). The legacy
    ``{label: DataFrame}`` mapping is still accepted -- it is how a bare frame gets
    explored in a test or a notebook-style session -- and is wrapped, keyed by its
    label, with a palette colour assigned in order.
    """
    if not sources:
        return {}
    if isinstance(sources, dict):
        prov = {str(k): dict(v) for k, v in (provenance or {}).items() if v}
        out: dict[str, PlotSource] = {}
        for i, (label, df) in enumerate(sources.items()):
            if df is None:
                continue
            key = str(label)
            out[key] = PlotSource(
                key=key, doc_key=key, label=key, df=df, provenance=prov.get(key, {}),
                colour=source_colour(i), display_name=key,
                kind="manual" if not prov.get(key) else "detection",
            )
        return out
    return {s.key: s for s in sources if s is not None and s.df is not None}


def _prefer(cols: Sequence[str], prefs: Sequence[str], fallback: int = 0) -> str | None:
    """First of ``prefs`` present in ``cols``; else the ``fallback``-th column."""
    for p in prefs:
        if p in cols:
            return p
    if cols:
        return cols[min(fallback, len(cols) - 1)]
    return None


def _column_quantity(col: str) -> tuple[str, str]:
    """``(quantity, unit)`` for a fitted column: what it measures, and in what.

    Fed to :func:`nanoscope.analysis.stats.fit_depth_gaussians`, which uses the first to
    NAME the model and the second to caption every ``mu``/``sigma`` it returns. Both used
    to be the constant "dI/I0", so a mixture fit to ``depth_pA`` came back labelled with
    the dimensionless ratio and printed picoamps beside it (report High-4).

    ``frac_depth`` keeps the dimensionless reading because that is what the column IS --
    and on the owner's apo-ferritin data it is 1.81-2.57, i.e. an enhancement, not a
    blockade fraction, which is precisely the number that must never be confused with a
    picoamp.

    THE HOLE THIS CLOSED. Everything not matched above used to answer ``""`` -- "this
    number has no unit" -- which for the TIME columns is false, and false by a factor of
    1000 on the surface where it shows. A ``gaussian-mixture`` fit to ``dwell_s`` printed

        mu_1 [0.01987, 0.02027]

    on the same window, the same axis and the same events where the ``exponential``
    printed ``tau_1 [19.85, 20.27] ms``: same quantity, same axis, a thousand-fold apart,
    and nothing on the page to say which was which. The estimate and CI lines convert a
    parameter whose unit is ``"s"`` into the axis's own display unit
    (:meth:`PlotsWindow._display_time`) and leave anything else alone -- so a time column
    that answers ``""`` here opts itself out of the one mechanism that would have made
    the two agree.

    Answered by SUFFIX, not by the enumerated time columns
    (:data:`_TIME_COLUMNS`), because the next ``*_s`` column added to the schema would
    otherwise inherit exactly this hole in silence. ``_pA_s`` is tested first: ``ecd_pA_s``
    also ends in ``_s`` and is a charge, not a time.

    The remaining ``""`` are true dimensionless readings -- ``snr`` and ``n_levels``.

    THE SCHEMA IS ASKED FIRST, and that is the whole point of this second pass. The
    suffix rules below are a GUESS from a column's spelling; ``events.SCHEMA`` is the
    project's own declaration of what each column is in, and it already carried the
    answer for every column in the frozen twenty. Guessing where a declaration exists
    is how ``start_idx``/``end_idx`` -- ``"sample"`` in the schema, since 0.1 -- went
    on answering "dimensionless" here: nothing in their spelling says otherwise, and
    a mixture fit to one printed ``mu = 1.196e+06`` with no unit beside it. Reading
    the schema also means a column ADDED to it arrives here already labelled, which
    the suffix table could only manage for names that happen to end the right way.

    The suffix rules stay, underneath, because the frames this window plots are not
    only schema frames: a merged table, a hand-measured table or a derived segment can
    carry a column the schema has never heard of, and ``*_pA`` / ``*_s`` is a better
    answer for those than silence. ``_pA_s`` is tested first: ``ecd_pA_s`` also ends
    in ``_s`` and is a charge, not a time.

    Two spellings are translated on the way out of the schema: ``"-"`` (its way of
    writing "dimensionless") becomes ``""``, because this string is CONCATENATED onto
    a number and " -" would read as a minus sign; and ``"pA*s"`` becomes the ``pA·s``
    this window already prints.
    """
    name = str(col)
    if name == "frac_depth":
        return stats.DEFAULT_DEPTH_QUANTITY, ""
    declared = _SCHEMA_UNITS.get(name)
    if declared is not None:
        return name, declared
    if name.endswith("_pA_s"):          # ecd_pA_s -- checked first, it also ends in _s
        return name, "pA·s"
    if name.endswith("_pA"):
        return name, "pA"
    if name.endswith("_s") or name in _TIME_COLUMNS:
        return name, "s"
    return name, ""


#: THE INVARIANT LIVES IN :mod:`nanoscope.gui.combos`, not here.
#:
#: ``QComboBox.setCurrentText`` on a non-editable combo is a no-op for unmatched text --
#: no exception, no signal, no change -- so every setter built on it fails silently and
#: the window reports the PREVIOUS selection's numbers. The first fix put this function in
#: this file and wired it into 2 of this window's 7 combo setters and 0 of the sibling
#: windows; the same defect then bit `VDepWindow.set_metric` during verification. It is a
#: shared module now, and `test_plots_panel.py` asserts no setter here bypasses it.
_choose_combo_item = choose_combo_item


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


class PlotsWindow(QtWidgets.QMainWindow):
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

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Population plots")
        self.resize(1120, 720)

        self._sources: dict[str, PlotSource] = {}    # key -> source, in offer order
        #: ``(doc_id, channel, module_id)`` of the detection the Analysis panel is
        #: SHOWING, set by :meth:`set_active_detection`. See :meth:`_active_source_key`.
        self._active_detection: tuple | None = None
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
        self._style = StyleSettings()   # the tunable plot design (title/colour/font/size/export)

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
        self._range_factors: tuple[float, float] = (1.0, 1.0)
        #: THE ANCHOR EVERY CLEARING RULE MEASURES FROM: the analysis identity the typed
        #: text and pins currently in the widgets were tuned FOR. Qt has no "about to
        #: change" signal, so a handler that fires after its widget moved has no other
        #: way to know what the window used to be showing. Moved only by
        #: :meth:`_set_config` and :meth:`_rebase`.
        self._identity = AnalysisIdentity()
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
        #: E2's caption. Parented to the ViewBox like the fit-values box, so `clear()`
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
        #     444 px to 331. The number below is then derived from THAT, plus the
        #     vertical scrollbar and the frame, so the two can never disagree again the
        #     way they did. (A literal would have rotted the first time a control was
        #     added to a group.)
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
            "Ctrl/Cmd/Shift-click adds or removes points from the selection.")
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
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
        # χ², is routed into Details and into the on-plot values box, which the PNG does
        # keep. The picture is the exploratory half; the statistic is the reportable half,
        # and the reportable half exports.
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
        bar.addWidget(spacer)

        self._run_action = bar.addAction("Plot / Apply")
        self._run_action.setObjectName("populationPlotRunAction")
        self._run_action.setToolTip(
            "Draw the selected plot and apply any pending fit choice (Ctrl+Return).")
        self._run_action.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        self._run_action.triggered.connect(self._run_plot)
        run_button = bar.widgetForAction(self._run_action)
        if run_button is not None:
            run_button.setAccessibleName("Plot or apply pending analysis")

        export_button = QtWidgets.QToolButton()
        export_button.setObjectName("populationPlotExportMenu")
        export_button.setText("Export ▾")
        export_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_button.setToolTip(
            "Export the current image, its numeric data, or a reproducible configuration.")
        export_button.setAccessibleName("Export population plot")
        export_menu = QtWidgets.QMenu(export_button)
        export_menu.addAction("Save image…", lambda: self.save_image())
        export_menu.addAction("Export plot data…", lambda: self.export_data())
        export_menu.addAction("Export configuration JSON…", lambda: self.export_config_json())
        export_button.setMenu(export_menu)
        self._export_menu = export_menu
        self._export_menu_btn = export_button
        bar.addWidget(export_button)

        self._workflow_toolbar = bar
        self.addToolBar(Qt.TopToolBarArea, bar)
        # BEGINNER VIEW: source choice plus the selected plot's essential controls.
        # The advanced analysis, appearance, and export/reuse groups still exist (and
        # remain fully addressable by the programmatic API), but only one purposeful
        # section is shown at a time instead of one 2,600 px undifferentiated column.
        self._show_control_section("Data")

    def _show_control_section(self, section: int | str) -> None:
        """Show one purposeful controls section and suppress unrelated advanced rows."""
        if isinstance(section, str):
            name = section
        else:
            name = self._section_combo.itemText(int(section))
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

        self._save_btn = QtWidgets.QPushButton("Save image…")
        self._save_btn.setAccessibleName("Save plot image")
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
        self._time_unit.addItems(list(_TIME_UNITS))
        self._time_unit.setCurrentText(_DEFAULT_TIME_UNIT)
        # THE TOOLTIP LISTS THE REAL LIST. Spelled out by hand it said "dwell_s, rise_s,
        # fall_s, t_start_s" -- the same four names `_TIME_COLUMNS` used to hold, and it
        # went stale for the same reason: `t_end_s` is a time column this combo governs
        # and the sentence denied it. Two surfaces, one fact.
        self._time_unit.setToolTip(
            f"Unit for time columns ({', '.join(_TIME_COLUMNS)}).\n"
            "This rescales the AXIS only — the data, and every fit computed on it, "
            "stay in seconds, so the dwell fit keeps the detector's own min_dwell_s.")
        self._time_unit.currentTextChanged.connect(self._on_time_unit_changed)
        unit_row.addWidget(self._time_unit, 1)
        lay.addLayout(unit_row)

        self._pages = QtWidgets.QStackedWidget()
        self._pages.addWidget(self._build_histogram_page())
        self._pages.addWidget(self._build_scatter_page())
        self._pages.addWidget(self._build_kde_page())
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

        self._hist_col = QtWidgets.QComboBox()
        self._hist_col.currentTextChanged.connect(
            lambda t, w=self._hist_col: self._on_axis_column_changed(t, w))
        form.addRow("Column", self._hist_col)

        self._bins = QtWidgets.QSpinBox()
        self._bins.setRange(2, 500)
        self._bins.setValue(30)
        self._bins.valueChanged.connect(self._changed)
        form.addRow("Bins", self._bins)

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

        self._fit_combo = QtWidgets.QComboBox()
        self._fit_combo.addItems([_FIT_NONE, _FIT_EXP, _FIT_GAMMA, _FIT_INVGAUSS,
                                  _FIT_EXP_GAMMA, _FIT_GMM])
        self._fit_combo.setToolTip(
            "exponential: the classic escape model — monotonic, cannot peak.\n"
            "gamma / inverse-gaussian: densities WITH a mode, for a dwell histogram\n"
            "that peaks away from zero. All three are truncated at the detector's\n"
            "min_dwell_s, so their BIC values compare directly.")
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

        self._scatter_x = QtWidgets.QComboBox()
        self._scatter_x.currentTextChanged.connect(
            lambda t, w=self._scatter_x: self._on_axis_column_changed(t, w))
        form.addRow("X", self._scatter_x)

        self._scatter_y = QtWidgets.QComboBox()
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
            "Overlay a least-squares best-fit line; power law fits in log-log space.")
        self._scatter_fit.currentTextChanged.connect(self._on_scatter_fit_changed)
        # Same confirmation step as the histogram's (owner note #12): the selection is
        # a proposal, this button draws it. Cheaper per fit than the histogram's, but a
        # regression line that appears the instant the combo is opened is the same
        # surprise, and one gate that behaves the same way on both pages is one rule.
        _narrow_combo(self._scatter_fit)
        form.addRow("Fit", self._scatter_fit)
        self._scatter_fit_apply_btn = QtWidgets.QPushButton("Apply fit")
        self._scatter_fit_apply_btn.clicked.connect(self._run_plot)   # the same one gate
        form.addRow("", self._scatter_fit_apply_btn)
        self._sync_fit_apply_buttons()
        return page

    def _build_kde_page(self) -> QtWidgets.QWidget:
        """Owner note #19: the 2-D KDE used to live only in the separate export figure
        and the density window; here it is a plot type beside the others, with the same
        source overlay, styling and Save image the rest of the window has."""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)

        self._kde_x = QtWidgets.QComboBox()
        self._kde_x.currentTextChanged.connect(
            lambda t, w=self._kde_x: self._on_axis_column_changed(t, w))
        form.addRow("X", self._kde_x)

        self._kde_y = QtWidgets.QComboBox()
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
        box = QtWidgets.QGroupBox("Features (clustering & PCA)")
        lay = QtWidgets.QVBoxLayout(box)
        self._features = QtWidgets.QListWidget()
        self._features.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self._features.setMaximumHeight(150)
        self._features.itemSelectionChanged.connect(self._on_features_changed)
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
        axis and persist across redraws until Auto is pressed. Values are read in the unit the
        axis DISPLAYS (a time axis follows the Time-unit combo — type the number you read off the
        ticks), converted to data units and to log10 internally as needed. (The plot is also
        mouse zoom/pan-able.)"""
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
        legend_move = QtWidgets.QLabel("Drag the legend on the plot to place it.")
        legend_move.setWordWrap(True)
        legend_move.setStyleSheet(f"color:{_MSG_HEX};")
        form.addRow("", legend_move)
        self._legend_reset_btn = QtWidgets.QPushButton("Reset legend position")
        self._legend_reset_btn.clicked.connect(self._reset_legend_position)
        form.addRow("", self._legend_reset_btn)
        self._sync_legend_reset_button()

        self._legend_font_size = QtWidgets.QSpinBox()
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

        # THE FONT ROW IS A PLAIN COMBO, NOT A QFontComboBox, AND THAT IS THE WHOLE OF THE
        # S6-01 FIX. ``QFontComboBox`` can only ever show a family that EXISTS, and this
        # style's default -- "" = "whatever font this computer hands the application" -- is
        # not one: ``setCurrentFont(QFont())`` asks it to show ``.AppleSystemUIFont``, which
        # it filters out as private, so it silently fell back to its first entry and the
        # panel told the user their figure was in "Academy Engraved LET".
        #
        # THREE WAYS OUT WERE MEASURED AND TWO FAILED:
        #   * show the RESOLVED family instead -- ``QFontInfo(QFont()).family()`` is ALSO
        #     '.AppleSystemUIFont' here, under cocoa and offscreen both, so the combo falls
        #     back to exactly the same wrong entry. It fixes nothing.
        #   * put a sentinel row INTO the QFontComboBox -- ``setCurrentFont()`` calls Qt's
        #     private ``_q_updateModel()``, which rebuilds the list from the font database
        #     and destroys the row (measured: 182 rows -> 181), after which the next
        #     ``setCurrentIndex(0)`` emitted ``currentFontChanged("Academy Engraved LET")``
        #     -- it COMMITTED the lie. Its editable line edit is worse still:
        #     ``setEditText(...)`` emits ``currentFontChanged`` with the fake string on the
        #     spot, no focus-out needed.
        #   * seeding a real family into the default was rejected on sight: it changes how
        #     every existing figure is drawn, it needs a per-platform guess, and every
        #     config already on disk still carries "" and would still lie.
        #
        # So the list is built here -- element-for-element what QFontComboBox shows (see
        # ``_font_families``) with one honest row on top -- and EVERY ROW CARRIES ITS STYLE
        # VALUE IN ``Qt.UserRole``. That is the invariant this buys: readers take
        # ``itemData``, never ``currentText()``, so what the user sees IS what the style
        # holds, and "" is finally a value the widget can spell.
        #
        # Not editable, unlike QFontComboBox: a typed family name is a string that resolves
        # to something else without saying so, which is the defect being fixed wearing a
        # different hat. Type-to-search on the closed combo still reaches a family in a
        # keystroke.
        self._font_combo = QtWidgets.QComboBox()
        # A FRESH QListView because the popup's own private delegate ignores
        # ``Qt.SizeHintRole`` -- measured, row heights ran 13-29 px with it, a jittery list
        # where QFontComboBox's was a uniform 22. With a plain view the hint is honoured and
        # every row is the same 22 px the old widget drew.
        self._font_combo.setView(QtWidgets.QListView())
        self._font_row_h = QtGui.QFontMetrics(self._font_combo.font()).height() + 6
        self._font_rows: dict[str, str] = {}    # casefolded family -> the family as listed
        self._font_off_list = False             # is row 1 a "named by the config, absent
                                                # here" row? see `_select_font_family`
        self._add_font_row(_FONT_DEFAULT_LABEL, "", QtGui.QFont())
        for family in _font_families():
            self._add_font_row(family, family, QtGui.QFont(family))
            self._font_rows[family.casefold()] = family
        # THE TOOLTIP IS COMPOSED WHEN IT IS SHOWN, NOT BAKED IN HERE. This line used to
        # interpolate ``QFontInfo(QFont()).family()`` once, at construction, and so put
        # ".AppleSystemUIFont" in front of the reader: a leading-dot internal name that
        # means nothing to a person, that ``_font_families`` above deliberately filters OUT
        # of this very list as system-internal, and that a later ``QApplication.setFont``
        # would leave stale. What a reader needs is not the platform's private spelling but
        # whether the default is a family they could ALSO pin by name, so the sentence is
        # built by ``_font_row_tooltip`` and rebuilt in the instant before Qt shows it (see
        # :meth:`eventFilter`). It is set here too so that anything reading ``toolTip()``
        # without a hover -- a test, an accessibility client -- gets the same words.
        self._font_combo.setToolTip(self._font_row_tooltip())
        self._font_combo.installEventFilter(self)
        # Selected BEFORE the handler is connected, so populating the list cannot count as
        # the user choosing anything -- and explicitly, rather than by trusting that row 0
        # happens to be current, because that is only true while the default is "".
        # (No ``blockSignals`` wrapper: :meth:`_select_font_family` now blocks the combo
        # for the whole of its own body, which is the only way that rule can be a
        # guarantee rather than a convention three call sites have to remember.)
        self._select_font_family(self._style.font_family)
        # ``currentIndexChanged``, not ``activated``: every other control in this group
        # (``_bg_combo``'s ``currentTextChanged``, the spins' ``valueChanged``) fires for
        # code as well as for a person and leans on the sync discipline -- ``_set_config``'s
        # ``_updating``, and for this one control the block ``_select_font_family`` takes
        # for itself -- to stay quiet. One control with a rule of its own is how a setter
        # ends up leaving the widget and the style disagreeing.
        self._font_combo.currentIndexChanged.connect(self._on_style_font_family)
        form.addRow("Font", self._font_combo)

        self._font_size = QtWidgets.QSpinBox()
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

        trow = QtWidgets.QHBoxLayout()
        self._title_bold = QtWidgets.QCheckBox("Bold")
        self._title_bold.setChecked(self._style.title_bold)
        self._title_bold.toggled.connect(self._on_style_title_bold)
        self._title_italic = QtWidgets.QCheckBox("Italic")
        self._title_italic.setChecked(self._style.title_italic)
        self._title_italic.toggled.connect(self._on_style_title_italic)
        trow.addWidget(self._title_bold)
        trow.addWidget(self._title_italic)
        trow.addStretch(1)
        form.addRow("Title weight", trow)

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
            "Colour of the fitted overlay curve and its title.")
        form.addRow("Fit colour", self._fit_swatch)

        self._fit_values_check = QtWidgets.QCheckBox("Fit values on plot")
        self._fit_values_check.setToolTip(
            "Write the fitted parameters (τ, μ, KS verdict…) as a text box on the "
            "figure itself, so a saved image carries its own numbers. The same text "
            "always appears in Details below.")
        self._fit_values_check.setChecked(self._style.fit_values_on_plot)
        self._fit_values_check.toggled.connect(self._on_style_fit_values)
        form.addRow("", self._fit_values_check)

        # THE FIT STATISTICS, AS FOUR SWITCHES. The first two were computed and printed
        # unconditionally -- correct numbers with no way to keep them off a figure that
        # already carries its parameters -- and the second two are new. Each tooltip says
        # what stays behind when it is off, because the one thing none of them may do is
        # hide a refusal (see StyleSettings.show_r2).
        self._r2_check = QtWidgets.QCheckBox("R² on figure")
        self._r2_check.setToolTip(
            "The scatter fit's R², in the title, the legend entry and the values box.\n"
            "Off: the equation stands alone on the figure. Details still says when R² is "
            "undefined or has no residual degrees of freedom — a fit that cannot be "
            "judged must not be able to look unjudged.")
        self._r2_check.setChecked(self._style.show_r2)
        self._r2_check.toggled.connect(self._on_style_show_r2)
        form.addRow("", self._r2_check)

        self._gof_check = QtWidgets.QCheckBox("KS verdict on figure")
        self._gof_check.setToolTip(
            "The KS statistic and its verdict in the values box (Fit values on plot), "
            "beside the fitted parameters.\n"
            "Off: only for a fit that PASSED. A curve the test REJECTS says so whatever "
            "this box is set to — in the plot title, AHEAD of any title you type, the "
            "way the comparability verdict is; in its legend entry and the values box "
            "while those are shown; all of which Save image… keeps — and in Details.")
        self._gof_check.setChecked(self._style.show_gof)
        self._gof_check.toggled.connect(self._on_style_show_gof)
        form.addRow("", self._gof_check)

        self._ci_band_check = QtWidgets.QCheckBox("Confidence band")
        self._ci_band_check.setToolTip(
            "A 95% POINTWISE band from the bootstrap replicates of the whole parameter "
            "vector — not the curve redrawn at the ends of each parameter's interval, "
            "which is a corner of a box the data rule out.\nUnavailable, and silently "
            "absent, for a fit restored from a saved project (the replicates are not "
            "stored), for the exponential+gamma family (which offers no bootstrap), and "
            "wherever the bootstrap produced fewer than 20 usable replicates.")
        self._ci_band_check.setChecked(self._style.show_ci_band)
        self._ci_band_check.toggled.connect(self._on_style_ci_band)
        form.addRow("", self._ci_band_check)

        self._residual_check = QtWidgets.QCheckBox("Residual strip")
        self._residual_check.setToolTip(
            "A second axis under the plot showing (observed − expected) / √expected per "
            "bin, sharing the x axis, plus the reduced χ² that is the sum of their "
            "squares.\nA run of same-sign bars is the finding — KS gives one number for "
            "the whole curve and cannot say WHERE it misses.\nOn screen only: "
            "Save image… exports the main axes, so the χ² is also written into Details "
            "and into the on-plot values box.")
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

        self._point_size = QtWidgets.QSpinBox()
        self._point_size.setRange(1, 30)
        self._point_size.setValue(self._style.point_size)
        self._point_size.setToolTip("Marker size for scatter / PCA points.")
        self._point_size.valueChanged.connect(self._on_style_point_size)
        form.addRow("Point size", self._point_size)

        self._line_width = QtWidgets.QSpinBox()
        self._line_width.setRange(1, 10)
        self._line_width.setValue(self._style.line_width)
        self._line_width.setToolTip("Width of the histogram outline and the fit curve.")
        self._line_width.valueChanged.connect(self._on_style_line_width)
        form.addRow("Line width", self._line_width)

        self._bg_combo = QtWidgets.QComboBox()
        self._bg_combo.addItems(["Dark", "Light"])
        self._bg_combo.setToolTip("Light gives a white canvas with dark ink for publication.")
        self._bg_combo.currentTextChanged.connect(self._on_style_background)
        form.addRow("Background", self._bg_combo)

        erow = QtWidgets.QHBoxLayout()
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
        erow.addWidget(self._export_w, 1)
        erow.addWidget(QtWidgets.QLabel("×"))
        erow.addWidget(self._export_h, 1)
        form.addRow("Export size", erow)
        return box

    @staticmethod
    def _pt_override_spin(value: int, tip: str) -> QtWidgets.QSpinBox:
        """A point-size spin whose 0 reads "follow the body font", spelled out on itself.

        The same shape (and the same 0-means-automatic rule) as the legend's own size
        spin, so the four text controls in this group behave alike rather than each
        having its own idea of what "unset" looks like."""
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 72)
        spin.setSpecialValueText("auto (body)")
        spin.setValue(int(value))
        spin.setToolTip(tip)
        return spin

    def _add_font_row(self, text: str, family: str, preview: QtGui.QFont) -> None:
        """Append one row to the Font combo: what it SAYS, what it MEANS, how it LOOKS.

        The three roles are deliberately three different things, and keeping them apart is
        what makes the Font row honest. ``text`` is prose and may not be a family name at
        all (row 0, and the "not on this computer" row); ``family`` is the exact string
        that goes into ``StyleSettings.font_family`` and is the ONLY thing any reader of
        this combo is allowed to take; ``preview`` draws the row in the font it is about,
        which is what ``QFontComboBox`` gave for free and is worth keeping -- row 0 previews
        in ``QFont()``, so the "application default" row is literally drawn in the
        application default.
        """
        combo = self._font_combo
        combo.addItem(text, family)
        row = combo.count() - 1
        preview.setPointSize(combo.font().pointSize())
        combo.setItemData(row, preview, Qt.FontRole)
        combo.setItemData(row, QtCore.QSize(0, self._font_row_h), Qt.SizeHintRole)

    def _font_row_tooltip(self) -> str:
        """The Font row's tooltip, composed against the application font AS IT IS NOW.

        WHY THIS IS A METHOD AND NOT A STRING LITERAL. The first version of S6-01 wrote
        the resolved family into the tooltip at construction time --
        ``QFontInfo(QFont()).family()`` -- and on macOS that is ".AppleSystemUIFont": a
        PRIVATE family name, the same kind ``_font_families`` strips out of the list
        because it is an implementation detail of the platform rather than something a
        person can choose, ask a colleague for, or type into a figure's style. Naming it
        told the reader nothing and looked like a leak, and because it was interpolated
        once it also went stale the moment anything called ``QApplication.setFont``.

        WHAT A READER GETS INSTEAD. The default row's whole job is to say "this figure is
        not pinned to a family", and the useful follow-up question is whether the font it
        is currently getting is one they COULD pin. So the family is named only when this
        very list offers it -- then the name is actionable, and the case is the one where
        an application font was set to a real face. When the resolved family is not in the
        list (every stock macOS run: the system UI font, private, unlistable) it is
        described rather than named, because printing a name the user cannot find in the
        dropdown below is worse than printing no name at all.

        The resolved family is read through ``QFontInfo`` rather than ``QFont().family()``
        because that is what the plot is actually DRAWN in -- measured: under cocoa the two
        agree at ".AppleSystemUIFont", under offscreen ``QFont().family()`` claims "Sans
        Serif" while ``QFontInfo`` resolves the same private face.
        """
        resolved = QtGui.QFontInfo(QtGui.QFont()).family()
        listed = self._font_rows.get(str(resolved).casefold())
        if listed is not None:
            here = (f"On this computer that is {listed} today, which is also a row in this "
                    "list — pick it by name if the figure must keep that face wherever it "
                    "is opened.")
        else:
            here = ("On this computer that is the system's own interface font, which is "
                    "not a row in this list and not a name another machine would resolve "
                    "the same way.")
        return ("Font family for the title, axis captions and tick numbers.\n\n"
                f"“{_FONT_DEFAULT_LABEL}” is not a family of its own: it is whatever font "
                "this computer hands the application, which is what an untouched plot has "
                f"always been drawn in. {here} Name a family when the figure has to look "
                "the same on someone else's machine.")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        """One thing this window can only learn from the event stream itself.

        THE FONT ROW'S TOOLTIP is rebuilt in the instant before Qt paints it. A tooltip
        is the one piece of text in this window whose exact moment of display is known,
        so it is the one that can afford to be recomputed instead of cached: the
        application font can change under a running window (a theme, a platform setting,
        an owner calling ``QApplication.setFont``), and a sentence about "whatever font
        this computer hands the application" that was composed at construction would then
        be describing a font nothing is drawn in any more. Qt reads ``toolTip()`` in its
        own handler for this event, which runs after every filter, so writing the fresh
        text here is enough -- the event is passed on untouched.

        THE PRESS ON THE DISABLED UNDO BUTTON IS DELIBERATELY NOT HANDLED HERE, which is
        worth saying in the place someone would reach for it. Qt does deliver that press
        to this filter before dropping it for being disabled, so answering the mouse
        user's refused undo from here is easy, and it was tried: it scrolled the ceiling
        sentence into view, which took the Undo button out from under a pointer that had
        not moved and put the template combo there, whose next click opened a dropdown of
        whole configurations. That scroll is now gone on EVERY trigger, not only this one
        -- see :meth:`undo_config` for the measurement and
        ``test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`` for the
        property it leaves behind. Nothing about the undo floor is answered from the event
        stream: the disclosure is the button's own tooltip and the standing amber line,
        both of which are simply there.
        """
        if obj is self._font_combo and event.type() == QtCore.QEvent.ToolTip:
            self._font_combo.setToolTip(self._font_row_tooltip())
        return super().eventFilter(obj, event)

    def _make_swatch(self, hex_color: str, on_click, tip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton()
        btn.setToolTip(tip)
        btn.setFixedHeight(24)
        btn.clicked.connect(on_click)
        self._paint_swatch(btn, hex_color)
        return btn

    @staticmethod
    def _paint_swatch(btn: QtWidgets.QPushButton, hex_color: str) -> None:
        """Show the chosen colour ON the button, with legible ink and its hex as the label."""
        btn.setText(hex_color)
        btn.setStyleSheet(
            f"background-color: {hex_color}; color: {_swatch_text_color(hex_color)}; "
            "border: 1px solid #555; padding: 2px;")

    # -- style-change handlers -------------------------------------------------------------------
    def _on_style_title(self, text: str) -> None:
        self._style = _dc_replace(self._style, title=text.strip())
        self._changed()

    def _on_style_font_family(self, index: int) -> None:
        """The Font row moved. What the row CARRIES is committed, not what it says.

        ``itemData``, never ``currentText()``: row 0 reads "Application default" and carries
        ``""``, and an absent family's row reads "Calibri — not on this computer" and
        carries "Calibri". Reading the words would write prose into a font name.

        ``index < 0`` is "no row is current", which happens only if something clears the
        combo, and it is NOT the same thing as the default row. Writing "" for it would be
        this window choosing the application font on the user's behalf -- the half of S6-01
        that was never broken and has to stay that way.

        A PICK ALSO RETIRES THE "not on this computer" ROW -- see below.
        """
        if index < 0:
            return
        family = self._font_combo.itemData(index)
        # THE ABSENT-FAMILY ROW IS TAKEN DOWN BY THE CHOICE THAT REPLACES IT. Row 1 exists
        # for one purpose: to say "the config that arrived named Calibri and this machine
        # has not got it" (:meth:`_select_font_family`). The moment the user picks any
        # other row, that sentence is about a decision they have just reversed -- but it
        # was cleared only on the NEXT `_select_font_family`, which is an undo, an import
        # or a reset, NOT the pick itself. So a colleague's config plus one ordinary font
        # change left a selectable "Calibri — not on this computer" row in the dropdown for
        # the rest of the session, advertising a font this computer cannot draw as though
        # it were still in play.
        #
        # Three details make the removal safe, all of them measured on both platforms:
        #   * ``family`` is read FIRST, because removing row 1 shifts every index above it
        #     down by one and the `index` this handler was handed goes stale.
        #   * the combo's signals are blocked across the removal. QComboBox keeps the
        #     current ITEM across a removal below it (measured on both platforms: index 2
        #     -> 1, same item) and emits ``currentIndexChanged`` for that renumbering --
        #     every pick above row 1, which is all of them but row 0. Unblocked, that
        #     would re-enter this handler and take a SECOND undo step for one gesture, and
        #     one gesture is one step is the rule the S6-01 tests pin.
        #   * index 1 itself is left alone: that is the user deliberately re-selecting the
        #     absent family, which is a legitimate choice the style still carries.
        if self._font_off_list and index != 1:
            was_blocked = self._font_combo.blockSignals(True)
            self._font_combo.removeItem(1)
            self._font_off_list = False
            self._font_combo.blockSignals(was_blocked)
        self._style = _dc_replace(
            self._style, font_family="" if family is None else str(family))
        self._changed()

    def _on_style_font_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, font_size_pt=int(value))
        self._changed()

    def _on_style_point_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, point_size=int(value))
        self._changed()

    def _on_style_line_width(self, value: int) -> None:
        self._style = _dc_replace(self._style, line_width=int(value))
        self._changed()

    def _on_style_background(self, text: str) -> None:
        self._style = _dc_replace(
            self._style, background="light" if text == "Light" else "dark")
        self._changed()

    def _on_style_show_r2(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_r2=bool(on))
        self._changed()

    def _on_style_show_gof(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_gof=bool(on))
        self._changed()

    def _on_style_ci_band(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_ci_band=bool(on))
        self._changed()

    def _on_style_residuals(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_residuals=bool(on))
        self._changed()

    def _on_style_provenance(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_provenance=bool(on))
        self._changed()

    def _on_style_fit_values(self, on: bool) -> None:
        self._style = _dc_replace(self._style, fit_values_on_plot=bool(on))
        self._changed()

    def _on_style_x_label(self, text: str) -> None:
        self._style = _dc_replace(self._style, x_label=text.strip())
        self._changed()

    def _on_style_y_label(self, text: str) -> None:
        self._style = _dc_replace(self._style, y_label=text.strip())
        self._changed()

    def _on_style_show_legend(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_legend=bool(on))
        self._changed()

    def _on_style_title_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, title_font_size_pt=int(value))
        self._changed()

    def _on_style_axis_label_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, axis_label_font_size_pt=int(value))
        self._changed()

    def _on_style_tick_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, tick_font_size_pt=int(value))
        self._changed()

    def _on_style_title_bold(self, on: bool) -> None:
        self._style = _dc_replace(self._style, title_bold=bool(on))
        self._changed()

    def _on_style_title_italic(self, on: bool) -> None:
        self._style = _dc_replace(self._style, title_italic=bool(on))
        self._changed()

    def _on_style_grid(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_grid=bool(on))
        self._changed()

    def _on_legend_entry_toggled(self, item) -> None:
        """One row of the Legend-entries checklist moved: remember it and restyle.

        Cosmetic by construction — nothing here re-reads a row of data — so it redraws
        live rather than going behind the run gate."""
        if self._updating:
            return
        try:
            entry_id = item.data(Qt.UserRole)
            checked = item.checkState() == Qt.Checked
        except RuntimeError:
            # Belt and braces. Shiboken raises this for a row whose C++ half is already
            # gone; the ORDERING fix (nothing this window frees is freed during its own
            # signal) is the real protection, because a genuinely dangling pointer never
            # reaches Python at all -- it crashes in the converter.
            return
        if not entry_id:
            return
        self._style = self._style.with_legend_include(str(entry_id), checked)
        self._changed()
        # THE HISTORY HOOK, BY HAND. `_wire_history`'s walk deliberately skips
        # `itemChanged` (see the note there), so this handler calls the hook itself, in
        # the position -- after the change is applied -- the generic connection held.
        self._history_touch()

    def _sync_legend_entry_list(self) -> None:
        """Re-offer the Legend-entries checklist from what the last draw offered.

        RECONCILED IN PLACE, NEVER CLEARED AND REFILLED -- and that is a CRASH FIX, not
        an optimisation. This method runs inside :meth:`_redraw`'s ``finally``, which a
        legend tick reaches through :meth:`_on_legend_entry_toggled` -> :meth:`_changed`:
        i.e. from INSIDE Qt's delivery of ``itemChanged(QListWidgetItem*)``, while Qt is
        still holding the ticked row as that emission's argument. ``clear()`` DELETED
        that C++ row on the spot, and every receiver the emission had not reached yet
        then had the same argument marshalled from a freed pointer -- PySide reads the
        dead object's vtable to build a wrapper, which is the EXC_BAD_ACCESS the shipped
        build died on. ``blockSignals`` (kept, for the re-entrant emission this method's
        own writes would otherwise cause) stops the SIGNAL and does nothing whatever
        about the POINTER.

        Reconciling makes the crashing path free-nothing: a tick changes whether an
        entry is INCLUDED, never which entries are OFFERED (``_legend_add`` records the
        offer either way), so every row is reused. When the offered set genuinely does
        change, the surplus rows are ``takeItem``-ed and parked by :meth:`_retire_row`
        so that their deletion happens after the event that caused it rather than inside
        it. Writing only what differs also ends the full model reset this list used to do
        on every restyle. That is the same OPERATION the session's
        ``qt.accessibility.table: ... Resizing table model`` warning complains about (an
        item view repopulated under something holding its old shape), though that line
        names a five-row view and this list had one -- it was a witness to the churn, not
        to this crash.
        """
        if not hasattr(self, "_legend_entries"):
            return   # fires during construction, before the Style group exists
        widget = self._legend_entries
        blocked = widget.blockSignals(True)
        try:
            offered = list(self._legend_offered.items())
            while widget.count() > len(offered):
                self._retire_row(widget.takeItem(widget.count() - 1))
            for row, (entry_id, label) in enumerate(offered):
                item = widget.item(row)
                if item is None:
                    item = QtWidgets.QListWidgetItem()
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    # EXPLICIT, even though `checkState()` already answers Unchecked: a
                    # row with no CheckStateRole *data* is drawn without a checkbox at
                    # all, and `_write_legend_row` below writes only what differs.
                    item.setCheckState(Qt.Unchecked)
                    widget.addItem(item)
                self._write_legend_row(item, entry_id, label)
        finally:
            widget.blockSignals(blocked)
        widget.setEnabled(bool(self._legend_offered) and self._style.show_legend)

    def _write_legend_row(self, item, entry_id: str, label: str) -> None:
        """Put one offered entry onto an EXISTING row, touching only what differs.

        Every setter here goes out as a ``dataChanged``, so writing back a value the row
        already carries would re-create exactly the model churn reconciling removed.
        """
        text = self._elide_legend(label)
        tip = f"{label}\n\nUntick to leave this row out of the legend."
        state = Qt.Checked if self._legend_included(entry_id) else Qt.Unchecked
        if item.text() != text:
            item.setText(text)
        if item.data(Qt.UserRole) != entry_id:
            item.setData(Qt.UserRole, entry_id)
        if item.checkState() != state:
            item.setCheckState(state)
        if item.toolTip() != tip:
            item.setToolTip(tip)

    def _retire_row(self, item) -> None:
        """Park a row taken out of a list widget until the current event has unwound.

        ``takeItem`` hands the C++ row's ownership to Python, so holding its wrapper
        here keeps the object ALIVE -- which is the entire point. A row can be removed
        from inside the delivery of a signal that carries a row of the same list, and Qt
        marshals that argument again for every receiver it has not reached yet; a row
        freed underneath that is a dangling pointer, and PySide dereferences it. One
        zero-timer later the event is over and letting go is safe.
        """
        if item is None:
            return
        self._retired_rows.append(item)
        if not self._retire_timer.isActive():
            self._retire_timer.start(0)

    def _drop_retired_rows(self) -> None:
        """Let go of every parked row: the event that removed them is long over."""
        self._retired_rows.clear()

    def _empty_list_widget(self, widget) -> None:
        """``clear()`` for a list whose rows may be in flight -- see :meth:`_retire_row`.

        Same end state as ``QListWidget.clear()``; the rows are handed to the parking
        lot instead of being destroyed inside whatever event asked for the rebuild.
        """
        while widget.count():
            self._retire_row(widget.takeItem(widget.count() - 1))

    def _on_style_legend_font_size(self, value: int) -> None:
        self._style = _dc_replace(self._style, legend_font_size_pt=int(value))
        self._changed()

    def _on_style_legend_counts(self, on: bool) -> None:
        self._style = _dc_replace(self._style, legend_counts=bool(on))
        self._changed()

    def _on_style_export_size(self, *_args) -> None:
        # Export size only affects save_image; no redraw needed.
        self._style = _dc_replace(self._style,
                                  export_w=int(self._export_w.value()),
                                  export_h=int(self._export_h.value()))

    def _pick_data_color(self) -> None:
        self._pick_color("data_color", self._data_swatch)

    def _pick_fit_color(self) -> None:
        self._pick_color("fit_color", self._fit_swatch)

    def _pick_color(self, attr: str, swatch: QtWidgets.QPushButton) -> None:
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._style, attr)), self, "Pick colour")
        if col.isValid():
            hex_color = col.name()
            # The field is chosen by NAME by the two one-line callers below, so the
            # frozen-copy spelling has to be by name too -- `replace` takes kwargs.
            self._style = _dc_replace(self._style, **{attr: hex_color})
            self._paint_swatch(swatch, hex_color)
            self._changed()

    # -- style helpers (colours / fonts / theme applied at each render site) ---------------------
    def _qfont(self, pt: int | None = None) -> QtGui.QFont:
        """The chosen family at ``pt``, defaulting to the BODY size.

        ``pt`` is the whole of owner note Q5 at this level: one family, four sizes.
        Every caller passes the size that belongs to the text it is about to set, and
        each of those sizes falls back to the body size when its own override is 0, so
        an untouched style still produces one number in four places."""
        qf = QtGui.QFont()
        if self._style.font_family:
            qf.setFamily(self._style.font_family)
        qf.setPointSize(int(self._style.font_size_pt if pt is None else pt))
        return qf

    def _title_pt(self) -> int:
        """Title size in pt: its own override, or the body size."""
        return int(self._style.title_font_size_pt or self._style.font_size_pt)

    def _axis_label_pt(self) -> int:
        """Axis-caption size in pt: its own override, or the body size."""
        return int(self._style.axis_label_font_size_pt or self._style.font_size_pt)

    def _tick_pt(self) -> int:
        """Tick-number size in pt: its own override, or the body size."""
        return int(self._style.tick_font_size_pt or self._style.font_size_pt)

    def _fg_hex(self) -> str:
        return _FG_LIGHT if self._style.background == "light" else _FG_DARK

    def _bg_spec(self) -> str:
        return _BG_LIGHT if self._style.background == "light" else _BG_DARK

    def _msg_hex(self) -> str:
        return _MSG_LIGHT_HEX if self._style.background == "light" else _MSG_HEX

    def _hist_brush(self) -> QtGui.QBrush:
        c = QtGui.QColor(self._style.data_color)
        c.setAlpha(_HIST_FILL_ALPHA)
        return pg.mkBrush(c)

    def _point_brush(self) -> QtGui.QBrush:
        c = QtGui.QColor(self._style.data_color)
        c.setAlpha(_POINT_ALPHA)
        return pg.mkBrush(c)

    def _data_pen(self) -> QtGui.QPen:
        return pg.mkPen(self._style.data_color, width=self._style.line_width)

    def _fit_pen(self) -> QtGui.QPen:
        # The fit curve reads one px heavier than the bar outline (default 2 px, as before).
        return pg.mkPen(self._style.fit_color, width=self._style.line_width + 1)

    def _scatter_fit_pen(self) -> QtGui.QPen:
        # The scatter best-fit line uses the fit colour at (at least) the chosen line width,
        # so it reads clearly over the marker cloud.
        return pg.mkPen(self._style.fit_color, width=max(self._style.line_width, 1))

    def _label_style(self) -> dict:
        """CSS-ish kwargs for an axis label: theme-aware colour + the chosen font.

        Sized by :meth:`_axis_label_pt` rather than by the body size — the axis
        captions are the second of the four independent text sizes owner note Q5 asks
        for, and they are the ones a poster usually wants larger than the ticks."""
        style = {"color": self._fg_hex(), "font-size": f"{self._axis_label_pt()}pt"}
        if self._style.font_family:
            style["font-family"] = self._style.font_family
        return style

    def _apply_grid(self) -> None:
        """Show or hide the gridlines, from the style, on every draw (owner note Q7).

        The alpha stays at the 0.15 the window has always used: whether there is a grid
        is an editorial choice, how faint it is is a house style, and offering both
        would let a user produce a grid that is technically on and invisible.
        """
        show = bool(self._style.show_grid)
        self._plot.showGrid(x=show, y=show, alpha=0.15)

    def _apply_background(self) -> None:
        """Paint the canvas + axis lines/ticks for the current theme so light mode stays legible.

        The grid rides along: it is part of the canvas, it has to be re-read on every
        draw for the same reason the ink does, and every render site in this file
        already calls this one (owner note Q7)."""
        self._apply_grid()
        self._plot_widget.setBackground(self._bg_spec())
        pen = pg.mkPen(self._fg_hex())
        for name in ("bottom", "left"):
            ax = self._plot.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)

    def _apply_fonts(self) -> None:
        """Tick-label font. Its OWN size (owner note Q5): the ticks are the text a
        larger title used to drag up with it until the numbers collided."""
        qf = self._qfont(self._tick_pt())
        for name in ("bottom", "left"):
            self._plot.getAxis(name).setStyle(tickFont=qf)

    def _axis_label_for(self, side: str, col: str, log: bool) -> str:
        """Label for ``side`` ('bottom'/'left'), rescaling the axis for time columns.

        ONE PLACE DECIDES THE UNIT NAME AND THE NUMBER THE TICKS SHOW. That is the whole
        contract of this method, and it is why the tick multiplier is set here as a side
        effect rather than anywhere the label is not also being written. Both are reset
        on every draw — a stale x1000 left on a pA axis would be a silent
        factor-of-1000 lie.

        The two modes reach the same place by different mechanics:

        * **linear** — ``AxisItem.setScale(factor)`` multiplies the tick labels, which is
          what it is for.
        * **log** — ``setScale`` is unusable, because pyqtgraph's ``tickValues``
          multiplies the log10 RANGE by it, i.e. scales an exponent. So the axis stays at
          ``setScale(1.0)`` and :attr:`_DecadeLabelAxis.display_factor` multiplies the
          decade VALUES instead, which is the same arithmetic in the one place it is
          meaningful.

        The log branch used to give up here and label the axis ``dwell (s, log)`` while
        leaving the ticks alone. pyqtgraph then supplied its own automatic SI prefix,
        multiplied the tick values by 1000 anyway and appended ``(x0.001)`` — so with
        Time unit = ms an 18 ms event was drawn at ``x = 10^1`` under a label saying
        seconds (report High-7). The prefix is now off (see :class:`_DecadeLabelAxis`)
        and the factor is this method's to set, so the label and the ticks cannot
        disagree.
        """
        axis = self._plot.getAxis(side)
        factor, name = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))
        is_time = col in _TIME_COLUMNS
        setter = getattr(axis, "set_display_factor", None)
        if setter is not None:
            setter(factor if (is_time and log) else 1.0)
        # A DURATION HAS NO ORIGIN; AN INSTANT DOES. `dwell_s`/`rise_s`/`fall_s` mean
        # the same thing on every recording, but `t_start_s` is "seconds since THIS
        # recording's first loaded sample" -- so the axis has to name its clock, and
        # say when it is showing more than one of them at once. The axis, not only
        # Details, because the axis label is what an exported image keeps.
        clock_tag = self._axis_clock_tag(col)
        # The user's override, if any, REPLACES THE TEXT ONLY: the tick scaling above has
        # already run, so a custom label sits over correctly scaled numbers. Overriding
        # both text and scale would let a label say "ms" over ticks showing seconds.
        override = (self._style.x_label if side == "bottom" else self._style.y_label).strip()
        if is_time and not log:
            axis.setScale(factor)
            return override or f"{col.removesuffix('_s')} ({name}{clock_tag})"
        axis.setScale(1.0)
        if override:
            return override
        if is_time and log:
            return f"{col.removesuffix('_s')} ({name}{clock_tag}, log)"
        return col + (" (log)" if log else "")

    def _axis_clock_tag(self, col: str) -> str:
        """The clock clause a time axis needs, or ``""`` for a duration.

        Short by design: :meth:`_column_notes` carries the t₀ values and the arithmetic
        in Details, and this is the half that has to survive inside an axis label.
        """
        if col not in _ORIGIN_COLUMNS:
            return ""
        try:
            n_sources = len([src for src, _df in self._drawn_frames()])
        except Exception:       # pragma: no cover - a label must never break a draw
            return ""
        if n_sources > 1:
            return ", from EACH recording's own start — origins differ"
        # One source: state ITS origin rather than assume zero. On the owner's own
        # files t₀ is 0 s, 10.2526 s and 61.3034 s, and it is the two non-zero ones
        # that make a number read off the trace disagree with this axis.
        srcs = [src for src, _df in self._drawn_frames()]
        off = self._clock_offset(srcs[0]) if srcs else None
        if off is None:
            return ", from recording start"
        return f", from recording start (t₀ = {clock.format_t0(off)} s)"

    def _plot_title(self, auto: str | None) -> str | None:
        """The title to show: the user override if any, else the automatic title."""
        return self._style.title or auto

    def _apply_title(self, auto: str | None, accent: str | None = None,
                     warn: str | None = None) -> None:
        """Set the title from the override-or-auto text, in the style font size + a sensible colour.

        An automatic accent (the fit vermillion / PCA foreground) is honoured only when the user has
        not typed their own title; an explicit override renders in the neutral foreground colour.

        ``warn`` is the figure's verdict -- the comparability verdict, a superseded
        calibration, a fit the KS test rejected (:meth:`_ks_headline`) -- and it is
        NOT part of the override-or-auto choice: it is PREPENDED to whatever the
        title would otherwise be, the user's own text included. The title is the
        only thing ``Save image`` keeps — Details is not exported — so a verdict
        that a typed title could displace would let exactly the figure this guard
        exists to prevent leave the app. The user may retitle the plot; they may not
        retitle away the fact that the two depths were measured under different
        baselines, or that the curve drawn through them does not describe them."""
        text = self._plot_title(auto)
        if warn:
            text = f"{warn}   ·   {text}" if text else warn
        if not text:
            # setTitle(None) only HIDES pyqtgraph's title label -- the previous
            # draw's text stays on the item. Harmless in the exported PNG (a hidden,
            # zero-height row is not rendered), but it means "what does the title
            # say" stops being a function of the current draw: switch from an
            # incomparable frac_depth overlay to dwell_s and the item still reads
            # NOT COMPARABLE. Cleared explicitly so the verdict cannot outlive the
            # plot that earned it.
            self._plot.setTitle(None)
            self._plot.titleLabel.setText("")
            return
        color = (_WARN_HEX if warn else
                 accent if (accent and not self._style.title) else self._fg_hex())
        self._plot.setTitle(text, color=color, size=f"{self._title_pt()}pt")
        # The CSS span carries size and colour only, so family, bold and italic go on
        # through the item's own QFont. Set whenever ANY of the three is asked for --
        # the old guard was `if font_family`, which silently made bold conditional on
        # the user having also picked a family.
        if self._style.font_family or self._style.title_bold or self._style.title_italic:
            qf = self._qfont(self._title_pt())
            qf.setBold(bool(self._style.title_bold))
            qf.setItalic(bool(self._style.title_italic))
            self._plot.titleLabel.item.setFont(qf)

    # -- axis range ------------------------------------------------------------------------------
    @staticmethod
    def _to_log(rng: tuple[float, float] | None, is_log: bool) -> tuple[float, float] | None:
        """A user range is typed in DATA units; pyqtgraph's setRange wants log10 on a log axis."""
        if rng is None:
            return None
        lo, hi = rng
        if is_log:
            if lo <= 0 or hi <= 0:
                return None   # a non-positive bound is meaningless on a log axis; treat as auto
            lo, hi = float(np.log10(lo)), float(np.log10(hi))
        return (lo, hi) if hi > lo else None

    def _read_range_pair(self, lo_key: str, hi_key: str,
                         is_log: bool = False) -> tuple[float, float] | None:
        """The typed pair, or ``None`` when it is not a usable window.

        REFUSED HERE RATHER THAN AT PIN TIME. ``_to_log`` already dropped a reversed pair
        and a non-positive bound on a log axis, but only on the way to the ViewBox -- so
        ``_manual_y`` was left holding ``(50.0, 1.0)``, a pin that is CONFIG (it is saved,
        restored and handed to an undo stack) and that nothing can ever apply. One rule,
        one place: a pair this declines is not stored, and
        :meth:`_axis_range_input_notes` says why in the same words.
        """
        lo_t = self._range_edits[lo_key].text().strip()
        hi_t = self._range_edits[hi_key].text().strip()
        if not lo_t or not hi_t:
            return None
        try:
            lo, hi = float(lo_t), float(hi_t)
        except ValueError:
            return None
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
        if is_log and lo <= 0:
            return None
        return lo, hi

    def _range_axis_columns(self) -> tuple[str, str]:
        """The ``(x, y)`` columns the CURRENT plot type puts on its two axes.

        One resolution, three readers: the unit factors the range boxes are read in,
        the unit names printed on their row labels, and the rescale that runs when
        either moves. Written once because the failure it prevents is precisely two
        of them disagreeing about which column the y axis is showing.
        ``""`` means "no column on that axis" — the histogram's y is a count, and
        PCA's axes are scores.
        """
        kind = self._plot_type()
        if kind == HISTOGRAM:
            return self._hist_col.currentText(), ""
        if kind == SCATTER:
            return self._scatter_x.currentText(), self._scatter_y.currentText()
        if kind == KDE:
            return self._kde_x.currentText(), self._kde_y.currentText()
        return "", ""

    def _axis_unit_factors(self) -> tuple[float, float]:
        """(x, y) factors converting the DISPLAYED axis unit back to data units.

        The ticks on a time axis show the Time-unit combo's choice (ms by default),
        so the range boxes must be read in that same unit — the number the user just
        read off the axis (owner: "Axis range defaults to microseconds. The user
        needs control"). A non-time column's factor is 1.0: its axis shows data units.
        """
        factor = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))[0]
        xcol, ycol = self._range_axis_columns()
        return (factor if xcol in _TIME_COLUMNS else 1.0,
                factor if ycol in _TIME_COLUMNS else 1.0)

    def _axis_unit_names(self) -> tuple[str, str]:
        """The unit each range box is read in, as text for its row label.

        From the SCHEMA that declares the column (:data:`_SCHEMA_UNITS`) for everything
        but a time column, which follows the Time-unit combo like its axis does.
        ``""`` for a dimensionless column, an axis with no column (a count, a PCA
        score), or a column the schema does not know."""
        unit = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))[1]
        out = []
        for col in self._range_axis_columns():
            if not col:
                out.append("")
            elif col in _TIME_COLUMNS:
                out.append(unit)
            else:
                out.append(_SCHEMA_UNITS.get(col, ""))
        return out[0], out[1]

    def _sync_range_unit_labels(self) -> None:
        """Write the unit the boxes are READ IN onto their own row labels.

        Owner note #4, second half: the Axis-range tooltip already said a time axis
        follows the Time-unit combo, but a tooltip is not on screen while you are
        typing into the box it describes — so "X min" over a box wanting milliseconds
        is where a µs/ms mix-up starts. The label says which."""
        if not hasattr(self, "_range_labels"):
            return   # may fire during construction, before the group exists
        x_unit, y_unit = self._axis_unit_names()
        for key, base, unit in (("xmin", "X min", x_unit), ("xmax", "X max", x_unit),
                                ("ymin", "Y min", y_unit), ("ymax", "Y max", y_unit)):
            self._range_labels[key].setText(f"{base} ({unit})" if unit else base)

    def _retune_range_boxes(self) -> None:
        """Rewrite the typed limits into the unit the axis NOW shows, then relabel.

        THE 1000x HAZARD THIS CLOSES. The boxes are read in the DISPLAY unit and stored
        in data units, so their text is only meaningful against the unit that was
        selected when it was typed. Switching Time unit from ms to µs, or the histogram
        column from ``dwell_s`` to ``depth_pA``, left "0.5" sitting in the box -- and the
        next Apply read it as 0.5 µs (a 1000x smaller window than the user pinned) or as
        0.5 s of picoamps. Nothing on screen said the number had changed meaning.

        NOW ONE CLAUSE OF ONE RULE, not a rule of its own. The retune lived here and
        covered the axis-range boxes only; the Fit-range boxes two rows above them are
        read in the SAME display unit and were left out, which is precisely the bug this
        round is fixing. Both are handled by :meth:`PlotConfig.rebased` now, and this
        name is kept as the door onto it because it is what the axis-range group's own
        code and comments have always called the operation.
        """
        self._rebase()

    def _on_time_unit_changed(self, _unit: str) -> None:
        """The axis unit moved: carry every typed limit with it, then redraw.

        EVERY typed limit -- the four axis-range boxes AND the two Fit-range boxes. The
        second half is the fix: the unit is not part of what the plot is OF, so a window
        the user typed keeps the physical span it described and only its text moves.
        A rescale, not a stale plot: the redraw below is safe because nothing about
        WHICH rows or columns are drawn has changed.
        """
        self._rebase()
        self._changed()

    def _on_axis_column_changed(self, _col: str, widget=None) -> None:
        """A column picker moved: retune the range boxes, blank the analysis-specific
        text, and put the plot behind the run gate.

        ``widget`` is the combo that moved, passed in rather than read off
        ``sender()`` because the gate has to tell a person's pick from
        ``set_histogram_column``'s -- see :meth:`_from_the_user`.
        """
        if self._updating:
            return   # `_repopulate_columns` rebases once, after the whole refill
        # THE ONE RULE. An axis column IS the analysis, so this one call unpins that
        # axis's range, drops a fit window typed against the old column, and blanks the
        # caption (owner note Q6, second half: "Dwell time (ms)" over a frac_depth axis
        # is a mislabelled figure, and the label is the half a saved PNG keeps).
        self._rebase()
        if widget is not None and self._from_the_user(widget):
            self._mark_stale("an axis column")
        else:
            self._armed_redraw()

    def _apply_axis_range(self) -> None:
        """Store the typed limits and pin them on the current plot, and SAY what was
        declined. Typed in the unit the axis DISPLAYS (see _axis_unit_factors); stored in
        data units."""
        fx, fy = self._axis_unit_factors()
        log_x, log_y = self._current_log_state()
        x = self._read_range_pair("xmin", "xmax", log_x)
        y = self._read_range_pair("ymin", "ymax", log_y)
        self._manual_x = None if x is None else (x[0] / fx, x[1] / fx)
        self._manual_y = None if y is None else (y[0] / fy, y[1] / fy)
        self._apply_stored_ranges(*self._current_log_state())
        self._sync_range_notes()
        # The PIN is config and this is the only place it moves, so the undo hook is
        # called by hand: `_wire_history`'s walk hooks CONTROLS, and pressing Apply
        # changes `_manual_x`/`_manual_y` without any control's own value having moved.
        self._history_touch()

    def _reset_axis_range(self, *, axis: str = "both") -> None:
        """Unpin ``"x"``, ``"y"`` or ``"both"``: clear those boxes and auto-fit that axis.

        ONE IMPLEMENTATION, THREE SCOPES. The all-or-nothing version threw the other
        axis's pin away as a side effect of wanting this one back, which is a silent
        edit to a setting the user did not touch.
        """
        do_x = axis in ("x", "both")
        do_y = axis in ("y", "both")
        if do_x:
            self._manual_x = None
            self._range_edits["xmin"].clear()
            self._range_edits["xmax"].clear()
        if do_y:
            self._manual_y = None
            self._range_edits["ymin"].clear()
            self._range_edits["ymax"].clear()
        vb = self._plot.getViewBox()
        vb.enableAutoRange(x=do_x, y=do_y)
        self._plot.autoRange()
        # An axis that is still pinned has just been auto-ranged along with the other
        # one by `autoRange()`, which fits BOTH; putting the surviving pin back is what
        # makes "Auto X" mean "auto x, and leave y alone" rather than "auto x, and quietly
        # unpin y as well".
        self._apply_stored_ranges(*self._current_log_state())
        self._sync_range_notes()
        self._history_touch()   # unpinning is a change; see `_apply_axis_range`

    def _axis_range_input_notes(self) -> list[str]:
        """What the Axis range boxes CONTAIN but did not do. The axis half of the rule
        :meth:`_fit_range_input_notes` states for the Fit range boxes.

        THREE SILENT NO-OPS, each a typed instruction the window declined:

        1. **Reversed limits.** ``_to_log`` ends ``return (lo, hi) if hi > lo else None``,
           so "5" over "1" pinned nothing at all — the same empty-interval refusal the
           Fit range boxes make, minus the sentence.
        2. **A non-positive bound on a log axis.** ``_to_log`` returns None for ``lo <= 0``
           because ``log10`` of it is not a number, and 0 is exactly what a person types
           for "start at the origin". The refusal is right; the silence is not, because
           the fix (type a small positive number, or untick Log x) is not guessable from
           an axis that simply did not move.
        3. **A half-filled pair.** ``_read_range_pair`` requires BOTH edges and returns
           None otherwise, so clearing one box unpins the whole axis. Stated rather than
           changed: a one-edged pin is a different feature (``manual_x`` is a pair, and
           the view has to be given both bounds), and inventing the missing edge from the
           data would be a limit the user never typed.

        Worded per axis and with the unit the box is read in, because "min ≥ max" over a
        pair of boxes labelled in µs is only actionable if you know which pair.
        """
        out: list[str] = []
        log_x, log_y = self._current_log_state()
        units = self._axis_unit_names()
        for name, lo_key, hi_key, is_log, unit in (
                ("X", "xmin", "xmax", log_x, units[0]),
                ("Y", "ymin", "ymax", log_y, units[1])):
            lo_t = self._range_edits[lo_key].text().strip()
            hi_t = self._range_edits[hi_key].text().strip()
            if not lo_t and not hi_t:
                continue
            tag = f" ({unit})" if unit else ""
            if not lo_t or not hi_t:
                filled, blank = ((f"{name} min", f"{name} max") if lo_t
                                 else (f"{name} max", f"{name} min"))
                out.append(
                    f"{blank}{tag} is blank, so {filled} was IGNORED and the {name} axis "
                    f"is auto-fitting the data. An axis is pinned by BOTH of its edges — "
                    f"type the other one, or clear this one to stop the amber.")
                continue
            try:
                lo, hi = float(lo_t), float(hi_t)
            except ValueError:
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} is not a pair of numbers and was "
                    f"IGNORED; the {name} axis is auto-fitting the data.")
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} contains a non-finite bound and "
                    f"was IGNORED; the {name} axis is auto-fitting the data.")
            elif hi <= lo:
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} is empty (min ≥ max), so BOTH "
                    f"bounds were IGNORED and the {name} axis is auto-fitting the data.")
            elif is_log and lo <= 0:
                out.append(
                    f"{name} min {lo_t}{tag} is not positive and the {name} axis is "
                    f"logarithmic, so the whole {name} range was IGNORED — log10 of it is "
                    f"not a number. Type a small positive bound, or untick Log "
                    f"{name.lower()}.")
        return out

    def _sync_range_notes(self) -> None:
        """Show or hide the Axis-range group's amber line. Idempotent; safe before the
        group exists (``set_axis_range`` can be called from a script at any time)."""
        if not hasattr(self, "_range_note"):
            return
        notes = self._axis_range_input_notes()
        self._range_note.setText("\n".join(notes))
        self._range_note.setVisible(bool(notes))

    def _current_log_state(self) -> tuple[bool, bool]:
        """Is each axis of the CURRENT plot logarithmic? One question, one answer.

        Delegates to :attr:`PlotConfig.log_state` rather than switching on the plot type
        a second time. The switch is written once, in the config, and this is the door
        onto it that the draw path and the range pin already call by this name.
        """
        return self.config().log_state

    def _apply_stored_ranges(self, log_x: bool, log_y: bool) -> None:
        vb = self._plot.getViewBox()
        xr = self._to_log(self._manual_x, log_x)
        yr = self._to_log(self._manual_y, log_y)
        if xr is not None:
            vb.setXRange(xr[0], xr[1], padding=0)
        if yr is not None:
            vb.setYRange(yr[0], yr[1], padding=0)

    def _finalize_view(self, log_x: bool, log_y: bool) -> None:
        """End every draw by auto-fitting, then re-pinning any manual limits the user set."""
        self._plot.autoRange()
        self._apply_stored_ranges(log_x, log_y)

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
            return PlotConfig()
        return PlotConfig(
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
                include_filter_limited=bool(self._use_filter_limited.isChecked())),
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
            self._apply_style_widgets(cfg.style)
            self._style = cfg.style
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

    @staticmethod
    def _rebased(cfg: PlotConfig, was: AnalysisIdentity) -> PlotConfig:
        """``cfg``, corrected for having moved off analysis identity ``was``.

        A pure function and the ONLY statement of what a change to the analysis
        invalidates; see :meth:`PlotConfig.rebased` for the four clauses and the measured
        bug behind each. Exposed here because this is where its callers live.
        """
        return cfg.rebased(was)

    def _rebase(self) -> None:
        """Apply :meth:`_rebased` to the live window: the ONE call every identity-moving
        handler makes.

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
        rebased = self._rebased(cfg, self._identity)
        if self._updating:
            rebased = _dc_replace(rebased, style=_dc_replace(
                rebased.style, title=cfg.style.title,
                x_label=cfg.style.x_label, y_label=cfg.style.y_label))
        if rebased == cfg:
            self._identity = cfg.identity   # nothing to clear, but the anchor still moved
            self._range_factors = self._identity.axis_factors
            self._sync_range_unit_labels()
            return
        # THE WINDOW'S OWN STATE, CORRECTED -- not an install from outside, so no new
        # origin and no re-decision of the import disclosures (S7-SK1: the Time-unit
        # combo reaches here, and the losses sentence must survive it).
        self._set_config(rebased, draw=False, own=True)

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
        want = {str(c) for c in cols}
        for i in range(self._features.count()):
            item = self._features.item(i)
            item.setSelected(item.text() in want)

    def _apply_style_widgets(self, style: StyleSettings) -> None:
        """Write a style onto the Style group's controls.

        Split out of :meth:`_set_config` because it is the half `reset_state` needs too,
        and because the font combo is a genuine exception worth stating once: its handler
        writes the style value the CURRENT ROW carries, and the default "" ("the
        application's own font") is a row of its own rather than a family -- so the row is
        moved by :meth:`_select_font_family`, which blocks the combo for its own body, and
        the style installed by the caller stands.
        """
        self._title_edit.setText(style.title)
        self._x_label_edit.setText(style.x_label)
        self._y_label_edit.setText(style.y_label)
        self._legend_check.setChecked(bool(style.show_legend))
        self._legend_font_size.setValue(int(style.legend_font_size_pt))
        self._legend_counts.setChecked(bool(style.legend_counts))
        self._fit_values_check.setChecked(bool(style.fit_values_on_plot))
        self._r2_check.setChecked(bool(style.show_r2))
        self._gof_check.setChecked(bool(style.show_gof))
        self._ci_band_check.setChecked(bool(style.show_ci_band))
        self._residual_check.setChecked(bool(style.show_residuals))
        self._prov_check.setChecked(bool(style.show_provenance))
        self._font_size.setValue(int(style.font_size_pt))
        self._title_size.setValue(int(style.title_font_size_pt))
        self._axis_label_size.setValue(int(style.axis_label_font_size_pt))
        self._tick_size.setValue(int(style.tick_font_size_pt))
        self._title_bold.setChecked(bool(style.title_bold))
        self._title_italic.setChecked(bool(style.title_italic))
        self._grid_check.setChecked(bool(style.show_grid))
        self._point_size.setValue(int(style.point_size))
        self._line_width.setValue(int(style.line_width))
        self._bg_combo.setCurrentText("Light" if style.background == "light" else "Dark")
        self._export_w.setValue(int(style.export_w))
        self._export_h.setValue(int(style.export_h))
        self._select_font_family(style.font_family)
        self._paint_swatch(self._data_swatch, style.data_color)
        self._paint_swatch(self._fit_swatch, style.fit_color)

    def _select_font_family(self, family: str) -> None:
        """Point the Font row at ``family`` (``""`` = the application default).

        THIS METHOD BLOCKS THE COMBO'S SIGNALS ITSELF, FOR ITS WHOLE BODY. It inserts and
        removes rows, each of which moves ``currentIndex``, and an unblocked move goes
        straight to :meth:`_on_style_font_family` -- which is exactly how a SYNC would end
        up committing a family the user never chose. That is the one failure S6-01's fix
        must not re-introduce, and it is the reason this is a guarantee rather than a
        precondition: the first version asserted that the caller had blocked the signals,
        which is not a guard at all under ``python -O`` (the assert is compiled out, and a
        fourth call site that forgot would then commit silently) and which had three call
        sites repeating the same two lines to satisfy it. Blocking here costs the same and
        cannot be switched off. The previous blocked state is restored rather than cleared,
        so a caller that blocks for reasons of its own still gets what it set up.

        A FAMILY THE LIST DOES NOT HAVE GETS A ROW OF ITS OWN rather than being dropped. A
        config exported on a colleague's machine (or on Windows) can name "Calibri";
        selecting nothing would leave the previous row standing -- the same lie in a new
        place -- and selecting the nearest match would silently rewrite the user's style.
        So the exact string is shown, marked as absent, and left in the style untouched:
        ``config()`` still exports "Calibri" and the panel still says what is missing.
        Exactly one such row can exist, at index 1, and it is cleared on every call --
        and, since the row is about a decision rather than about a font, also by the
        user picking any other row (:meth:`_on_style_font_family` retires it there).

        The lookup is CASE-INSENSITIVE because ``QFont`` matching is: a hand-edited sidecar
        saying "courier new" really is drawn in Courier New, so the row naming it is the
        honest one to show, and the style keeps its own spelling untouched.
        """
        combo = self._font_combo
        was_blocked = combo.blockSignals(True)
        try:
            if self._font_off_list:
                combo.removeItem(1)
                self._font_off_list = False
            if not family:
                combo.setCurrentIndex(0)
                return
            listed = self._font_rows.get(str(family).casefold())
            if listed is not None:
                combo.setCurrentIndex(combo.findData(listed))
                return
            combo.insertItem(1, f"{family} — not on this computer", str(family))
            combo.setItemData(1, QtGui.QFont(str(family)), Qt.FontRole)
            combo.setItemData(1, QtCore.QSize(0, self._font_row_h), Qt.SizeHintRole)
            self._font_off_list = True
            combo.setCurrentIndex(1)
        finally:
            # ``finally``, not a line at the end: the body returns from three places, and
            # a combo left silent because an early return skipped the restore would be a
            # Font row that stops reporting the user's picks altogether.
            combo.blockSignals(was_blocked)

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
        # ...plus Ctrl/Cmd+Shift+Z by name, because what ``StandardKey.Redo`` resolves to
        # is a THEME decision and the themes disagree: measured, Ctrl+Y under the generic
        # theme the offscreen platform falls back to, Cmd+Shift+Z under the Mac one. The
        # list is deduped on the resolved sequence, so on a platform where the standard
        # key IS Shift+Z nothing is registered twice -- Qt answers a double registration
        # with an "ambiguous shortcut overload" and then does NOTHING, which would take
        # Redo away on exactly the platform that had it right.
        registered: list[QtGui.QKeySequence] = []
        for seq, slot in ((QtGui.QKeySequence.Undo, self.undo_config),
                          (QtGui.QKeySequence.Redo, self.redo_config),
                          (QtGui.QKeySequence("Ctrl+Shift+Z"), self.redo_config)):
            key = QtGui.QKeySequence(seq)
            if key.isEmpty() or any(key == seen for seen in registered):
                continue
            registered.append(key)
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
        split = QtWidgets.QLabel(
            "A template carries the plot type, columns, fit recipe, axis limits and the "
            "whole style. It never carries which sources are ticked, your retyped legend "
            "entries, or an applied fit — the fit arrives selected, and you press Apply.")
        split.setWordWrap(True)
        split.setStyleSheet(f"color:{_MSG_HEX};")
        lay.addWidget(split)

        self._template_updating = True
        try:
            self._load_templates_into_combo()
        finally:
            self._template_updating = False

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
            return out
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
    #: The first names no column at all, which is the point: a blank column means "keep
    #: what you have" (:meth:`PlotConfig.with_template`), so it restyles whatever is on
    #: screen without touching the analysis. The second and third carry the fit as a
    #: PROPOSAL only -- ``applied`` stays ``none`` -- so recalling one arms Apply and
    #: waits, exactly as picking the model by hand does.
    _BUILTIN_TEMPLATES: dict[str, PlotConfig] = {
        # Appearance only: a white canvas, no grid, text sized for a printed column, a
        # heavier curve, and a 1600x1200 export. Every analysis field is left blank.
        "Publication figure (light)": PlotConfig(
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
        return self._installable(self.config().with_template(tpl))

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

    def _delete_template(self) -> None:
        name = self._template_combo.currentText()
        if name == self._TPL_MANUAL or name in self._BUILTIN_TEMPLATES:
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
                 (self._kde_y, cfg.kde.y, now.kde.y))
        hist_col, sx, sy, kx, ky = [
            want if want and combo.findText(want) >= 0 else have
            for combo, want, have in pairs]
        # THE Y QUANTITY IS NOT TOUCHED HERE, and that omission is load-bearing rather
        # than an oversight -- see this method's docstring for the measurement and for
        # what now catches a name this build cannot offer (`_set_config`, which records
        # the refusal, and `_NormRefusal`, which keeps it visible). The request travels on
        # intact so that `_sync_template_selection`, whose whole job is to notice that a
        # template no longer describes the window, is comparing the request.
        if (hist_col, sx, sy, kx, ky) == (cfg.histogram.column, cfg.scatter.x,
                                          cfg.scatter.y, cfg.kde.x, cfg.kde.y):
            return cfg
        return _dc_replace(
            cfg,
            histogram=_dc_replace(cfg.histogram, column=hist_col),
            scatter=_dc_replace(cfg.scatter, x=sx, y=sy),
            kde=_dc_replace(cfg.kde, x=kx, y=ky))

    # -- data in ---------------------------------------------------------------------------------
    def set_active_detection(self, key: tuple | None) -> None:
        """Tell the window WHICH stored detection the Analysis panel is showing, as
        ``(doc_id, channel, module_id)`` (or None when there is none).

        The owner should call this before (or with) :meth:`set_data`; it costs nothing
        when it is not called, because :meth:`_active_source_key` falls back. See that
        method for what this exists to stop.
        """
        self._active_detection = tuple(key) if key else None

    def _active_source_key(self) -> str:
        """The key of the source that IS the detection on screen in the Analysis panel,
        or ``""`` when it cannot be resolved.

        THE ANSWER TO "the window drew a different detector's table and said nothing".
        A channel can hold several detections at once (``ResultStore`` is keyed
        (doc_id, channel, module_id)), so "the current recording" no longer identifies
        a table. Measured on the owner's -1000 mV recording, 2026-09-02: Schmitt 110
        events and CUSUM 250 both stored, the panel showing CUSUM, and Plots opening
        with Schmitt's 110 rows ticked, drawn and captioned "(n = 110)".

        Three sources of the answer, in order: what the owner told us
        (:meth:`set_active_detection`); a public ``active_detection_key()`` on the
        owner window if it grows one; and -- as a bridge until MainWindow does -- its
        Analysis panel's own public ``current_detection()``. The last one is the only
        reach outside this window and it is deliberately total: every step is a
        ``getattr`` behind a ``try``, so a window with no such parent (every test that
        builds one bare, and the notebook-style callers) simply gets "" and the
        ``PlotSource.current`` flag decides, exactly as before.
        """
        key = self._active_detection
        if key is None:
            owner = self.parent()
            for probe in (lambda w: w.active_detection_key(),
                          lambda w: _detection_key(w._analysis_panel.current_detection())):
                try:
                    key = probe(owner)
                except Exception:
                    key = None
                if key:
                    break
        if not key or len(tuple(key)) < 3:
            return ""
        doc_id, channel, module_id = tuple(key)[:3]
        for k, src in self._sources.items():
            if (src.doc_id, src.channel, src.module_id) == (doc_id, int(channel), module_id):
                return k
        return ""

    def set_data(
        self,
        sources,
        provenance: dict[str, dict] | None = None,
    ) -> None:
        """Install the per-event tables to explore, with their detection provenance.

        Args:
            sources: EITHER a sequence of :class:`~nanoscope.core.sources.PlotSource`
                (what :class:`~nanoscope.gui.main_window.MainWindow` pushes -- one
                per open recording/segment with a stored detection, plus the manual
                table), OR the legacy ``{label: DataFrame}`` mapping, which is
                wrapped into single-colour sources keyed by their label. Pass FULL
                event frames (not pre-filtered): the *Use accepted events only*
                check does that filtering, per source, and is enabled only when a
                checked source carries an ``accepted`` column.
            provenance: legacy ``{label: EventSet.provenance}``, applied only to the
                mapping form. A :class:`PlotSource` already carries its own -- and
                that pairing is load-bearing: the dwell fit uses the detector's real
                ``min_dwell_s`` instead of the shortest observed dwell (the
                difference is a ~25% bias in tau at a 50 us floor against a 200 us
                tau, and it is invisible on the plot), and with an overlay up EACH
                source must be fit with ITS OWN floor. A source without provenance
                still fits, but the window says out loud what it fell back to.

        Sources still present after a refresh keep their checked state, so
        re-pulling data doesn't yank the user off their view. Any previously
        computed clustering/PCA is dropped -- the row set may have changed
        underneath it.
        """
        # NEW ROWS SUPERSEDE ANY ADVISORY WORK IN FLIGHT, IMMEDIATELY.
        #
        # This is the entry point a curation click reaches, and every deferred note is
        # keyed on the bytes of the sample it was computed from -- so a batch that was
        # running when this was called is, by construction, about events that are no
        # longer what the user is looking at. Waiting for the settle timer to notice
        # (400 ms) is not free: the jobs are `scipy.optimize.minimize(Nelder-Mead)`,
        # which is PYTHON, so a stale batch does not merely waste a core, it holds the
        # GIL against the next click. Measured on the owner's -700 mV run with
        # `exponential + gamma (bimodal)` on screen -- the family whose own point fit
        # (~250 ms) makes its click period brush the 400 ms settle -- a batch left to
        # run alongside the burst took the median click from 258 ms to 1014 ms. Stopping
        # it here takes it back.
        #
        # Nothing is lost: the redraw at the end of this method asks for whatever the
        # new table needs, and the timer starts again from the user's last action.
        self._abort_note_job()

        prev_checked = self.selected_source_keys()
        prev_primary = self._primary_key()

        self._sources = _normalize_sources(sources, provenance)
        self._cluster_result = None
        self._pca_result = None

        want = [k for k in prev_checked if k in self._sources]
        if not want:
            want = [k for k in (prev_primary,) if k and k in self._sources]
        if not want:
            # First push (or every previous source closed): follow the app's active
            # DETECTION so opening Plots shows the table the user was already looking
            # at -- not merely the active recording, which since one-source-per-detector
            # discovery can offer several tables. `_active_source_key` says what that
            # distinction cost. The `PlotSource.current` flag is the fallback: it is
            # right whenever the channel carries one detection, and it is a GUESS
            # (the channel's first detector) when it carries more and nobody named the
            # active one.
            current = self._active_source_key() or next(
                (k for k, s in self._sources.items() if s.current), None)
            want = [current or next(iter(self._sources), "")]
            want = [k for k in want if k]

        self._updating = True
        # NOT `clear()`: this can be reached from inside a source row's own
        # `itemChanged` (a tick, or a detection result landing mid-click), where
        # destroying the rows would leave Qt marshalling a freed one. See `_retire_row`.
        self._empty_list_widget(self._source_list)
        for src in self._sources.values():
            item = QtWidgets.QListWidgetItem(self._legend_label(src))
            item.setData(Qt.UserRole, src.key)
            # EDITABLE, because this row IS the legend key (the hint above says so) and
            # owner note #15 is that the legend was not editable. Double-clicking a row
            # renames the series everywhere it is named; the source's real label is kept
            # on the PlotSource and restored by clearing the edit back to empty.
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked if src.key in want else Qt.Unchecked)
            item.setForeground(QtGui.QBrush(QtGui.QColor(src.colour)))
            item.setToolTip(self._source_tooltip(src))
            self._source_list.addItem(item)
        if self._source_list.count():
            self._source_list.setCurrentRow(
                max(0, next((i for i in range(self._source_list.count())
                             if self._source_list.item(i).data(Qt.UserRole) in want), 0)))
        self._updating = False
        # THE ONE PLACE ``auto_plot`` IS READ. A push of data is a programmatic commit
        # by default -- MainWindow's Refresh, the test corpus and every caller that
        # expects a plot after ``set_data`` keep exactly the behaviour they had. Set
        # ``auto_plot = False`` before the first push and the window comes up on the
        # prompt instead, which is the "opens showing an empty plot" half of Q1; the
        # gate on the user's own gestures is on either way.
        # ARRIVING DATA IS NOT A CONFIGURATION CHANGE, so it is not an undo step --
        # `EventHistory.set_suspended`'s rule ("a reconstruction never becomes an offer to
        # undo the project you just opened") one object down. Which columns the pickers
        # end up on and which rows are ticked are a CONSEQUENCE of the push, not something
        # the user did, and MainWindow re-pushes on every detection; recording each one
        # would put "the window before your data existed" at the bottom of every stack and
        # make Undo untick everything. The stack SURVIVES -- the styling steps on it are
        # still applicable, and `_installable` resolves away a column that has gone --
        # only the base moves.
        was, self._history_suspended = self._history_suspended, True
        try:
            self._on_sources_changed(draw=self.auto_plot or self._plot_armed)
            if not self._plot_armed:
                self._show_plot_prompt()
        finally:
            self._history_suspended = was
        self._history.rebase(self.config())
        self._sync_config_buttons()
        self._sync_template_selection()

    @staticmethod
    def _source_tooltip(src: PlotSource) -> str:
        bits = [f"{src.n_rows:,} rows"]
        if src.is_segment:
            bits.append("derived segment")
        if src.module_id:
            # NAME, then id. The row may be one of two tables from the same recording
            # (one per detector), so which detector produced these rows is the first
            # thing a hovering user needs; the id stays for the provenance-minded.
            bits.append(f"{src.detector_label} ({src.module_id})")
        md = src.provenance.get("min_dwell_s") if src.provenance else None
        if md:
            bits.append(f"min_dwell_s = {float(md):g} s")
        elif src.kind == "manual":
            bits.append("no detector provenance (hand measurements)")
        # ...and the edit hint on the row itself, not only in the group's label: the
        # tooltip is what a user who is already hovering the row will read (note Q3).
        return (f"{src.display_name or src.label}\n" + "  ·  ".join(bits)
                + "\n✎ Double-click to retype this source's legend entry.")

    # -- source / column plumbing ----------------------------------------------------------------
    def _on_source_item_changed(self, item) -> None:
        """A tick changed: the drawn set changed, so everything tied to rows is stale.

        ...OR the row was RENAMED, which is how the legend is edited (owner note #15).
        One signal carries both, so the rename is resolved by reading the row's text
        AGAINST THE STYLE rather than against the source's label: whatever the row now
        says is what the override becomes, and a row that says the source's own name --
        or nothing -- removes the override rather than legending a blank entry.

        WHY THERE IS NO ``text_now != src.label`` GUARD ANY MORE (S7-FIX2 / S7-SK2).
        That guard returned before either branch below could run whenever the row was
        retyped to the source's ORIGINAL name -- the gesture a person uses to undo a
        rename by hand -- so the override stayed. MEASURED with real double-click and
        per-character key events on the owner's -700 mV source: rename to "Apo" (row,
        override, canvas all "Apo"); type "-700 mV" back -> row "-700 mV", override
        STILL "Apo", canvas STILL "Apo  (n = 982)"; a re-Plot left it; the next Refresh
        (``set_data`` rebuilds the rows from ``_legend_label``) put "Apo" back on the
        row, and the typed correction was gone. Worse, giving "Apo" to the -600 mV
        source next drew ['Apo  (n = 982)', 'Apo  (n = 236)'] on the canvas and in the
        exported SVG -- two voltages under one name, with the Sources list reading
        ['-700 mV', 'Apo']. Clearing the row to EMPTY removed the override cleanly, so
        the two ways of saying "put it back" disagreed. The branch below already did the
        right thing for exactly this text; only the guard stopped it reaching there. A
        TICK on a row whose text equals its label now costs one idempotent
        ``without_label_override`` (a no-op on a missing key), which is why the guard was
        never load-bearing."""
        if self._updating:
            return
        try:
            key, text_now = item.data(Qt.UserRole), item.text()
        except RuntimeError:
            return   # see `_on_legend_entry_toggled`; belt and braces
        src = self._sources.get(key or "")
        if src is not None:
            text = text_now.strip()
            if text and text != src.label:
                self._style = self._style.with_label_override(src.key, text)
            else:
                self._style = self._style.without_label_override(src.key)
            # ...AND THEN THE ROW IS WRITTEN BACK FROM THE STYLE, on this path too.
            # `_set_config` states the rule (see the comment over its
            # `_sync_source_row_labels` call): a row's TEXT is a projection of the style
            # exactly as the Title box is. That held on a config RESTORE and not here,
            # because the two branches above store `text_now.strip()` while the row keeps
            # whatever the item editor committed. MEASURED: typing "  Apo, -700 mV  "
            # into a row left the Sources list reading '  Apo, -700 mV  ' while the
            # legend checklist and the drawn legend read 'Apo, -700 mV' -- one source
            # under two names -- and a re-Plot did NOT reconcile them, because nothing on
            # the draw path rewrites a row; only an undo/redo did, because that goes
            # through `_set_config`. The stripping itself is right (a legend entry with a
            # trailing space is not what anyone typed on purpose); showing the user
            # something other than what was stored is what was wrong.
            #
            # `_legend_label` RATHER THAN `text`, because it is the one reader that
            # answers "what is this source called now" from the style, and it answers for
            # both branches above in one line: the override just stored, or `src.label`
            # when the override was just removed. That second case is why the blank-name
            # write that used to sit under `if not text` is gone rather than merely
            # joined -- it was this same write, spelled out for one branch of two. The
            # third case it now also covers is the one that has no branch at all: a name
            # that differs from the label ONLY by padding ("  Apo  " over "Apo") stores
            # no override and used to leave the padding standing on the row.
            #
            # UNDER `_updating`, AND IN PLACE. `_updating` because `setText` re-emits
            # `itemChanged` synchronously and this handler is one of that signal's own
            # receivers: unguarded, the write re-enters here, reads the window's own text
            # back as a fresh user gesture and pushes a SECOND history step for one
            # rename. The flag is the first thing this method tests, so the re-entry
            # returns immediately -- the same guard the blank-name write already used.
            # In place because this can run from INSIDE Qt's delivery of that same
            # `itemChanged`, which still holds `item` and re-marshals it to every
            # receiver it has not reached yet: `setText` on an existing row frees
            # nothing, while a `clear()`, a `takeItem` or a fresh `QListWidgetItem` here
            # would rebuild the EXC_BAD_ACCESS the shipped build died on (the rule is
            # stated at length in `_sync_source_row_labels`).
            #
            # ONLY WHAT DIFFERS IS WRITTEN, for `_write_legend_row`'s reason: the
            # ordinary rename needs no stripping, and that common case should cost one
            # string comparison and emit nothing at all.
            shown = self._legend_label(src)
            if item.text() != shown:
                was, self._updating = self._updating, True
                try:
                    item.setText(shown)
                finally:
                    self._updating = was
        # WHICH OF THE TWO IT WAS decides which side of the run gate this falls on. A
        # TICK changes the rows that are drawn, so it makes the plot stale; a RENAME is
        # legend text, which is as cosmetic as a colour and redraws live. One Qt signal
        # carries both, so the checked set is compared with the one the last draw used.
        if tuple(self.selected_source_keys()) != self._checked_snapshot:
            self._on_sources_changed(draw=False)
            self._mark_stale("the data selection")
        else:
            self._changed()
        # THE HISTORY HOOK, BY HAND -- see `_on_legend_entry_toggled`.
        self._history_touch()

    def _on_source_current_changed(self, *_a) -> None:
        """Highlighting a row without ticking it does not change the plot; it only
        moves the keyboard focus. Redrawing here would fight the user's tick."""
        return

    def _on_sources_changed(self, *, draw: bool = True) -> None:
        """Re-sync everything that depends on WHICH sources are checked.

        ``draw=False`` does the syncing and stops short of the picture: the run gate's
        caller has already marked the plot stale and the user has not pressed Plot yet.
        The syncing itself is never skipped -- the column pickers must offer the new
        source's columns before the user can choose one to plot.
        """
        # A new row set invalidates labels/scores tied to the old rows.
        self._cluster_result = None
        self._pca_result = None
        # NO BLANKET UNPIN HERE ANY MORE. This used to clear the pinned axis range on
        # every source change, INCLUDING a plain Refresh -- while leaving the typed title
        # standing, which is the same bug in mirror image and was the evidence the two
        # were being cleared by different ad-hoc rules. Re-pulling the same columns is
        # not a change of analysis at all (the owner, asked directly: "no need to clear
        # the data... the user selects the data from the file manager as is done now"),
        # so nothing is cleared for it. When the refill below genuinely DOES move a
        # column -- a refreshed source that no longer carries it -- `_repopulate_columns`
        # ends in `_rebase`, and the pin, the fit window and the caption go together.
        self._repopulate_columns()
        frames = [s.df for s in self.selected_sources()]
        has_accepted = any(df is not None and "accepted" in df.columns for df in frames)
        self._accepted_check.setEnabled(bool(has_accepted))
        self._sync_norm_default()
        self._sync_cluster_split_toggle()
        self._sync_filter_limited_toggle()
        self._checked_snapshot = tuple(self.selected_source_keys())
        # A source that is GONE takes its legend decisions with it; one merely unticked
        # keeps them (see `PlotConfig.with_pruned_legend`).
        self._prune_legend_memory()
        self._sync_legend_entry_list()
        if draw:
            self._armed_redraw()

    def _sync_norm_default(self) -> None:
        """Default to DENSITY the moment a second source is drawn (and back to count
        when it is alone), unless the user has chosen a normalization themselves.

        Overlaying a 5000-event and a 200-event recording as raw counts hides the
        second one under the axis; density is the only default that makes
        the comparison the user just asked for actually readable."""
        if self._norm_user_set or not hasattr(self, "_norm_combo"):
            return
        want = _NORM_DENSITY if len(self.selected_sources()) > 1 else _NORM_COUNT
        if self._norm_combo.currentText() != want:
            was, self._updating = self._updating, True
            self._norm_combo.setCurrentText(want)
            self._updating = was

    def _on_norm_moved(self, _text: str) -> None:
        """The y quantity moved by SOMETHING OTHER THAN THE ONE WRITER, so whatever an
        install could not honour is no longer the reason for what is on screen.

        ``_updating`` is what separates the two, and it is the same flag every other
        handler in this window reads: :meth:`_set_config` and :meth:`_sync_norm_default`
        both write this combo under it. The writer therefore does not clear its own
        record through here -- it decides the record itself, in the same statement that
        writes the combo -- and the automatic density default does not clear it either,
        which is deliberate: nobody re-decided anything when the window swung to density
        because a second source was ticked, and the refusal is still the true account of
        why the requested quantity is not on screen. See :class:`_NormRefusal`.
        """
        if not self._updating:
            self._norm_refusal.clear()

    def _sync_cluster_split_toggle(self) -> None:
        """One owner for the colour channel: SOURCE while an overlay is up, CLUSTER
        only when a single source is drawn. Rather than let the two fight silently,
        the cluster-split control is disabled (and explains itself) during an
        overlay, and any selection it held is reset."""
        if not hasattr(self, "_hist_colour_by"):
            return   # may fire during construction, before the histogram page exists
        multi = len(self.selected_sources()) > 1
        self._hist_colour_by.setEnabled(not multi)
        self._hist_colour_by.setToolTip(
            "Colour already means SOURCE while more than one source is drawn — untick "
            "all but one to split by cluster instead."
            if multi else
            "Split the histogram by cluster label after Run clustering — the other half "
            "of the clustering workflow.")
        if multi and self._hist_colour_by.currentText() != _COLOUR_NONE:
            was, self._updating = self._updating, True
            self._hist_colour_by.setCurrentText(_COLOUR_NONE)
            self._updating = was

    def _on_accepted_changed(self, _checked: bool) -> None:
        if self._updating:
            return
        # Filtering to accepted-only changes which rows exist; drop stale fits.
        self._cluster_result = None
        self._pca_result = None
        # ...and it is a DATA SELECTION change, so a CLICK waits for Plot like a tick
        # does. Code moving the check is the same confirmation `set_fit` is (see
        # :meth:`_from_the_user`): this box has no `set_*` of its own, so owner and test
        # code drive the widget itself and must still take effect at once.
        if self._from_the_user(self._accepted_check):
            self._mark_stale("the accepted-events filter")
        else:
            self._armed_redraw()

    def _repopulate_columns(self) -> None:
        """Refill every column picker from the checked sources' numeric columns.

        With ONE source that is simply its columns. With an overlay it is the
        INTERSECTION: a column only one source carries cannot be overlaid, and
        offering it would draw a single curve while the legend promised two."""
        selected = self.selected_sources()
        if not selected:
            cols: list[str] = []
        else:
            cols = _numeric_columns(selected[0].df)
            for src in selected[1:]:
                other = set(_numeric_columns(src.df))
                cols = [c for c in cols if c in other]

        self._updating = True
        try:
            x_prev, y_prev = self._scatter_x.currentText(), self._scatter_y.currentText()
            kx_prev, ky_prev = self._kde_x.currentText(), self._kde_y.currentText()
            h_prev = self._hist_col.currentText()
            feat_prev = set(self.feature_selection())

            for combo in (self._hist_col, self._scatter_x, self._scatter_y,
                          self._kde_x, self._kde_y):
                combo.clear()
                combo.addItems(cols)

            self._features.clear()
            for c in cols:
                self._features.addItem(QtWidgets.QListWidgetItem(c))

            if not cols:
                return

            self._hist_col.setCurrentText(h_prev if h_prev in cols
                                          else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._scatter_x.setCurrentText(x_prev if x_prev in cols
                                           else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._scatter_y.setCurrentText(y_prev if y_prev in cols
                                           else (_prefer(cols, _PREFER_AMP, fallback=1) or cols[-1]))
            # KDE defaults follow the signature translocation figure: dwell on X,
            # fractional blockade on Y when the table has it (it survives a gain
            # revision; see _DENSITY_Y's rationale in main_window), amplitude otherwise.
            self._kde_x.setCurrentText(kx_prev if kx_prev in cols
                                       else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._kde_y.setCurrentText(
                ky_prev if ky_prev in cols
                else ("frac_depth" if "frac_depth" in cols
                      else (_prefer(cols, _PREFER_AMP, fallback=1) or cols[-1])))

            keep = feat_prev & set(cols)
            if not keep:
                keep = {c for c in _PREFER_FEATURES if c in cols} or set(cols[:4])
            for i in range(self._features.count()):
                item = self._features.item(i)
                item.setSelected(item.text() in keep)
        finally:
            self._updating = False
        # The pickers may have moved on their own -- a refreshed source that no longer
        # carries the old column -- and a move nobody typed invalidates exactly what a
        # typed one does. One rebase after the whole refill, not one per combo.
        self._rebase()

    # -- control-change handlers -----------------------------------------------------------------
    def _changed(self, *_args) -> None:
        """A COSMETIC control moved: restyle the drawn plot at once.

        The live half of the run gate. Fonts, colours, the grid, the legend and the
        axis-range pins all land here and all redraw immediately, because none of them
        re-reads a row and making the user press Plot to see a colour is the opposite
        of the complaint that produced the gate. Anything that changes WHICH ROWS or
        WHICH COLUMNS are drawn goes through :meth:`_mark_stale` instead.

        With nothing committed yet there is nothing to restyle, so the prompt stays up.
        """
        if self._updating:
            return
        if not self._plot_armed:
            self._show_plot_prompt()
        else:
            self._redraw()
        # THE UNDO HOOK. `_wire_history`'s walk covers everything the user MOVES; this
        # covers what changes without a control moving -- a colour picked in a modal, a
        # legend dragged on the canvas, a source row renamed into `label_overrides`.
        # AFTER the draw, not before: a draw can write a config value back (Auto-BIC's
        # chosen k lands in the spin and in `_applied_fit_k`), and a step taken first
        # would name a state that was never on screen.
        self._history_settle()

    # -- the RUN gate (round-4 Q1) ---------------------------------------------------------------
    def _run_plot(self) -> None:
        """Commit every pending selection and draw. THE one recompute in this window.

        Wired to the Plot button and to BOTH "Apply fit" buttons, which is the answer
        to "should this and Apply fit be one gate": they are one gate with three
        doors. Two buttons that each commit a different half of the same picture is
        how a user ends up reading a gamma's k off a plot still showing an
        exponential -- the exact failure owner note #12 was written about, re-created
        one level up.
        """
        self._commit_fit_selection()
        self._commit_scatter_fit_selection()
        self._armed_redraw()

    def _armed_redraw(self) -> None:
        """Draw, and record that what is on screen now matches the controls.

        Every programmatic entry point ends here rather than at :meth:`_redraw`, so
        ``set_data``/``set_fit``/``set_plot_type`` keep drawing on the spot AND clear
        any staleness the same click would have cleared by hand.
        """
        self._plot_armed = True
        self._stale = False
        self._stale_reason = ""
        # The applied fit has just become the selected one, so both Apply buttons and
        # the Plot button lose their ● together — they are one gate and they must not
        # be able to disagree about whether anything is pending.
        self._sync_fit_apply_buttons()
        self._sync_run_state()
        self._redraw()
        # The APPLIED fit is config and `_commit_fit_selection` moves it with no control
        # having changed, so the run gesture needs the hook of its own. See `_changed`.
        self._history_settle()

    def _mark_stale(self, reason: str) -> None:
        """The drawn plot no longer matches the controls; say so and wait for Plot.

        ``reason`` is shown to the user verbatim, because "press Plot again" without
        saying WHAT changed is indistinguishable from the window having glitched.
        """
        if not self._plot_armed:
            self._show_plot_prompt()
        else:
            self._stale = True
            self._stale_reason = reason
            self._sync_run_state()
        self._history_settle()  # see `_changed`; free, because the hook ends in a diff

    def _show_plot_prompt(self) -> None:
        """The empty canvas a never-run window carries, with the instruction on it."""
        self._stale = False
        self._stale_reason = ""
        self._sync_run_state()
        self._message(_PLOT_PROMPT)
        # `_message` bypasses `_redraw`, so the legend checklist has to be emptied here
        # or it would go on listing the rows of a plot that is no longer on screen.
        self._sync_legend_entry_list()

    def _sync_run_state(self) -> None:
        """Put the gate's state on the Plot button and on the banner beneath it.

        ``pending`` deliberately includes an un-applied FIT as well as a stale
        selection: they are one gate, so the Plot button has to badge everything
        pressing it would commit, not only the half it was named after."""
        if not hasattr(self, "_plot_btn"):
            return   # fires during construction, before the Plot group exists
        pending = self._stale or self._pending_fit_change()
        self._plot_btn.setText("Plot ●" if pending else "Plot")
        self._plot_btn.setToolTip(
            f"Redraw — {self._stale_reason} changed since the plot on screen was drawn."
            if self._stale else
            "Draw the checked sources with the plot type and columns chosen above."
            if not self._plot_armed else
            "Redraw with the current selection.")
        if hasattr(self, "_run_action"):
            self._run_action.setText("Plot / Apply ●" if pending else "Plot / Apply")
            self._run_action.setToolTip(self._plot_btn.toolTip())
        self._plot_state.setVisible(bool(self._stale))
        if self._stale:
            self._plot_state.setText(
                f"⚠ {self._stale_reason} changed — the plot below is the PREVIOUS "
                f"selection. Press Plot to redraw.")
            self._plot_state.setStyleSheet(f"color:{_WARN_HEX};")

    def _on_plot_type_changed(self, name: str) -> None:
        if name in _PLOT_TYPES:
            self._pages.setCurrentIndex(_PLOT_TYPES.index(name))
        # THE ONE RULE: a different plot has different axes, so the pins, the typed
        # limits, the fit window and the caption are settled by `rebased` rather than by
        # a list of clears here. Run BEFORE the `_updating` gate below because
        # `run_pca`/`reset_state` move this combo themselves and their new analysis is
        # every bit as incompatible with the old pins as a click's would be.
        self._rebase()
        if self._updating:
            # A BULK WIDGET UPDATE, not a choice: `run_pca` moves this combo to PCA
            # itself and `reset_state` moves it back to Histogram, and in both cases the
            # caller owns the draw that follows -- drawing here as well would cost a
            # whole extra render per Run PCA. (The caption such a caller owns is kept by
            # `_rebase` above, which is where that exception is stated.)
            return
        if self._from_the_user(self._plot_type_combo):
            self._mark_stale("the plot type")
        else:
            self._armed_redraw()   # code named the plot type; naming it IS the run

    # -- the fit APPLY gate (owner note #12) -----------------------------------------------------
    @staticmethod
    def _from_the_user(widget) -> bool:
        """Whether the PERSON just moved ``widget``, as opposed to code moving it.

        Qt's value signals (``currentTextChanged``, ``toggled``, ``valueChanged``) fire
        identically for a click and for ``setCurrentText``/``setChecked``, and the Apply
        gate has to tell the two apart. Owner note #12 is about the GESTURE: picking a
        model out of the dropdown must not commit it, while the programmatic contract --
        :meth:`set_fit`, :meth:`set_fit_components`, :meth:`set_scatter_fit`, and the
        owner/test code that drives these widgets directly -- IS itself the confirmation
        and must still take effect at once. Gating both would mean "apply the fit"
        having no caller at all outside a mouse.

        A real gesture always leaves a fingerprint on the widget: a combo the user is
        picking from has its popup on screen, and a control they are typing into or
        clicking has the keyboard focus (Qt hands it over on mouse PRESS, before the
        value moves). Code touches neither, so code reads as "already confirmed".
        """
        view = widget.view() if isinstance(widget, QtWidgets.QComboBox) else None
        if view is not None and view.isVisible():
            return True
        return bool(widget.hasFocus())

    def _fit_selection_moved(self, widget) -> None:
        """A histogram-fit control moved: arm Apply, or commit when code moved it."""
        if self._updating:
            return
        if self._from_the_user(widget):
            self._sync_fit_apply_buttons()
        else:
            self._apply_fit()

    def _commit_fit_selection(self) -> None:
        """Copy the histogram-fit PROPOSAL into the applied state, without drawing.

        Everything a draw reads about the fit lives in ``_applied_*``; the combo and
        the spin are only a proposal until this runs. Split out from
        :meth:`_apply_fit` so :meth:`_run_plot` can commit the fit and the data
        selection in one pass and draw ONCE, rather than twice for one click."""
        self._applied_fit = self._fit_combo.currentText()
        self._applied_fit_k = int(self._fit_k.value())
        self._applied_fit_k_auto = bool(self._fit_k_auto.isEnabled()
                                        and self._fit_k_auto.isChecked())

    def _commit_scatter_fit_selection(self) -> None:
        self._applied_scatter_fit = self._scatter_fit.currentText()

    def _apply_fit(self) -> None:
        """Commit the selected histogram fit and draw it.

        The PROGRAMMATIC commit: ``set_fit``/``set_fit_components`` reach it through
        :meth:`_fit_selection_moved`. The buttons do not -- they run
        :meth:`_run_plot`, which commits this and the rest of the selection together."""
        self._commit_fit_selection()
        self._armed_redraw()

    def _apply_scatter_fit(self) -> None:
        """Commit the selected scatter best-fit line and draw it (programmatic path)."""
        self._commit_scatter_fit_selection()
        self._armed_redraw()

    def _on_scatter_fit_changed(self, _name: str) -> None:
        if self._updating:
            return
        if self._from_the_user(self._scatter_fit):
            self._sync_fit_apply_buttons()
        else:
            self._apply_scatter_fit()

    def _sync_fit_apply_buttons(self) -> None:
        """Show, on the buttons themselves, that a chosen fit is NOT the drawn one.

        A gate the user cannot see is a control that stopped working: the whole failure
        mode of note #12's fix is a scientist picking `gamma`, reading the still-visible
        exponential's tau off the plot and never noticing. So a pending selection both
        ENABLES the button and marks it, and the tooltip names the model that is
        actually drawn."""
        if not hasattr(self, "_scatter_fit_apply_btn"):
            return   # fires during construction, before the scatter page exists
        pending = (self._fit_combo.currentText() != self._applied_fit
                   or int(self._fit_k.value()) != int(self._applied_fit_k)
                   or bool(self._fit_k_auto.isEnabled() and self._fit_k_auto.isChecked())
                   != bool(self._applied_fit_k_auto))
        self._fit_apply_btn.setEnabled(pending)
        self._fit_apply_btn.setText("Apply fit ●" if pending else "Apply fit")
        self._fit_apply_btn.setToolTip(
            f"Draw “{self._fit_combo.currentText()}” — the plot still shows "
            f"“{self._applied_fit}”." if pending else
            f"“{self._applied_fit}” is the fit on the plot.")

        s_pending = self._scatter_fit.currentText() != self._applied_scatter_fit
        self._scatter_fit_apply_btn.setEnabled(s_pending)
        self._scatter_fit_apply_btn.setText("Apply fit ●" if s_pending else "Apply fit")
        self._scatter_fit_apply_btn.setToolTip(
            f"Draw “{self._scatter_fit.currentText()}” — the plot still shows "
            f"“{self._applied_scatter_fit}”." if s_pending else
            f"“{self._applied_scatter_fit}” is the line on the plot.")
        # The Plot button badges the SAME pending state, because it is the same gate.
        self._sync_run_state()

    def _pending_fit_change(self) -> bool:
        """Whether either fit combo proposes something the plot is not showing."""
        if not hasattr(self, "_scatter_fit"):
            return False
        return (self._fit_combo.currentText() != self._applied_fit
                or int(self._fit_k.value()) != int(self._applied_fit_k)
                or bool(self._fit_k_auto.isEnabled() and self._fit_k_auto.isChecked())
                != bool(self._applied_fit_k_auto)
                or self._scatter_fit.currentText() != self._applied_scatter_fit)

    def _on_fit_changed(self, name: str) -> None:
        on = name != _FIT_NONE
        self._fit_k_auto.setEnabled(on and name in _MULTI_COMPONENT_FITS)
        self._fit_k.setEnabled(on and not (self._fit_k_auto.isEnabled()
                                           and self._fit_k_auto.isChecked()))
        self._sync_filter_limited_toggle()
        # ...and the fit itself waits for Apply. The enable/disable above is about the
        # PROPOSAL — the user has to be able to set k before committing to it.
        self._fit_selection_moved(self._fit_combo)

    def _on_fit_k_changed(self, _value: int) -> None:
        self._fit_selection_moved(self._fit_k)

    def _on_fit_k_auto_toggled(self, on: bool) -> None:
        """Auto-BIC owns the spin while it is on, exactly as the clustering group's does —
        a number the user can still type but the fit ignores is a lie about who is in
        charge. The chosen k is written BACK into the spin after each draw, so the value
        on screen is always the one that was fitted."""
        self._fit_k.setEnabled(not (on and self._fit_k_auto.isEnabled()))
        self._fit_selection_moved(self._fit_k_auto)

    def _sync_filter_limited_toggle(self) -> None:
        """The include/exclude toggle only means something when a fit is on and some
        checked source actually carries the ``filter_limited`` flag."""
        if not hasattr(self, "_use_filter_limited"):
            return   # may fire during construction, before the histogram page exists
        has_flag = any("filter_limited" in s.df.columns for s in self.selected_sources())
        self._use_filter_limited.setEnabled(
            bool(has_flag and self._fit_combo.currentText() != _FIT_NONE))

    def _on_features_changed(self) -> None:
        if self._updating:
            return
        # PCA axes are defined by the feature set; a change makes any run stale.
        self._pca_result = None
        if self._plot_type() == PCA:
            # Through `_changed`, not `_redraw`, so an un-run window stays on its prompt
            # instead of being drawn into by a control it has not been told to plot yet.
            # The PCA view is gated by its own Run button either way.
            self._changed()

    def _on_method_changed(self, name: str) -> None:
        is_gmm = name == _METHOD_GMM
        self._k_mode.setEnabled(is_gmm)
        if not is_gmm:
            self._updating = True
            self._k_mode.setCurrentText(_K_FIXED)   # k-means needs an explicit k
            self._updating = False
        self._k_spin.setEnabled(self._k_mode.currentText() == _K_FIXED or not is_gmm)

    def _on_kmode_changed(self, mode: str) -> None:
        self._k_spin.setEnabled(mode == _K_FIXED)

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
        want = set(cols)
        self._updating = True
        try:
            for i in range(self._features.count()):
                item = self._features.item(i)
                item.setSelected(item.text() in want)
        finally:
            self._updating = False
        self._on_features_changed()

    def feature_selection(self) -> list[str]:
        return [i.text() for i in self._features.selectedItems()]

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
        self._font_size.setValue(int(pt))

    def set_font_family(self, family: str) -> None:
        """Font family for the plot's text. ``""`` restores the application default.

        The empty string is not "no change": it is the value the Font row's first entry
        stands for, and before S6-01 this setter could not reach it --
        ``setCurrentFont(QFont(""))`` left the combo showing whatever it had been showing
        while the style went blank, so the panel and the figure disagreed.
        """
        self._style = _dc_replace(self._style, font_family=str(family))
        self._select_font_family(str(family))
        self._changed()

    def set_data_color(self, hex_color: str) -> None:
        self._style = _dc_replace(self._style, data_color=str(hex_color))
        self._paint_swatch(self._data_swatch, str(hex_color))
        self._changed()

    def set_fit_color(self, hex_color: str) -> None:
        self._style = _dc_replace(self._style, fit_color=str(hex_color))
        self._paint_swatch(self._fit_swatch, str(hex_color))
        self._changed()

    def set_point_size(self, n: int) -> None:
        self._point_size.setValue(int(n))

    def set_line_width(self, n: int) -> None:
        self._line_width.setValue(int(n))

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

    # -- current frame ---------------------------------------------------------------------------
    def _plot_type(self) -> str:
        return self._plot_type_combo.currentText()

    def _frame_of(self, src: PlotSource) -> pd.DataFrame | None:
        """One source's frame, filtered to accepted rows when that is asked for.

        The original index is preserved through the filter so a clustering's
        labels (indexed by the source ``df.index``) still align onto whatever
        subset is on screen.
        """
        df = src.df
        if df is None:
            return None
        if (self._accepted_check.isEnabled() and self._accepted_check.isChecked()
                and "accepted" in df.columns):
            df = df[df["accepted"].astype(bool)]
        return df

    def _current_frame(self) -> pd.DataFrame | None:
        """The PRIMARY source's frame (clustering / PCA operate on one source)."""
        src = self.primary_source()
        return None if src is None else self._frame_of(src)

    def _drawn_frames(self) -> list[tuple[PlotSource, pd.DataFrame]]:
        """``(source, filtered frame)`` for every checked, non-empty source."""
        out = []
        for src in self.selected_sources():
            df = self._frame_of(src)
            if df is not None and df.shape[0]:
                out.append((src, df))
        return out

    def _source_provenance(self) -> dict:
        """The PRIMARY source's detection provenance ({} when there is none)."""
        src = self.primary_source()
        return dict(src.provenance) if src is not None else {}

    def _accepted_only(self) -> bool:
        return bool(self._accepted_check.isEnabled() and self._accepted_check.isChecked())

    def _dwell_inputs(self, col: str, src: PlotSource | None = None):
        """The shared dwell-fit resolution for ONE source + column, or None.

        This is :func:`nanoscope.analysis.modules.dwell_fit_inputs` -- the very
        function ``modules.fit_dwell`` (and therefore the Fits panel) calls -- so
        the sample, the detection floor ``t_min`` and any modelled right
        truncation are identical in both windows by construction, not by
        coincidence. Returns None for a column the detector's dwell provenance
        does not describe (``rise_s``, ``ecd_pA_s``, ...): borrowing the dwell
        floor for those would be worse than not having one.

        ``src`` defaults to the PRIMARY source. It is a parameter precisely so an
        overlay resolves EACH source against ITS OWN provenance: two recordings
        detected at different ``min_dwell_s`` have different detection floors, and
        applying one's floor to the other's dwell times is the exact bias the
        provenance plumbing exists to remove.
        """
        if col not in _DWELL_COLUMNS:
            return None
        src = src if src is not None else self.primary_source()
        if src is None:
            return None
        df = src.df
        if df is None or col not in df.columns:
            return None
        frame = df if col == "dwell_s" else df.assign(dwell_s=df[col])
        try:
            return modules.dwell_fit_inputs(
                (frame, dict(src.provenance)),
                accepted_only=self._accepted_only(),
                use_filter_limited=self._use_filter_limited.isChecked(),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def _depth_inputs(self, col: str, src: PlotSource | None = None):
        """The shared depth-fit resolution for ONE source (sample + that detector's
        own depth floor), or None. See :meth:`_dwell_inputs` on why the source is a
        parameter rather than an implicit 'the' source."""
        if col not in _DEPTH_COLUMNS:
            return None
        src = src if src is not None else self.primary_source()
        if src is None:
            return None
        df = src.df
        if df is None or col not in df.columns:
            return None
        try:
            return modules.depth_fit_inputs(
                (df, dict(src.provenance)), column=col,
                accepted_only=self._accepted_only(),
                use_filter_limited=self._use_filter_limited.isChecked(),
            )
        except (ValueError, KeyError, TypeError):
            return None

    # -- rendering -------------------------------------------------------------------------------
    def _render_workspace_snapshot(self) -> None:
        """Paint restored results without running an analysis kernel.

        Project restore installs the saved :class:`PlotConfig`, PCA/clustering
        objects and distribution-fit objects before calling this seam.  The normal
        drawing code remains the authority for bars, points, axes and styling, but
        its three computational exits are fenced while this method is active:

        * histogram overlays evaluate the already-restored ``FitResult``;
        * a scatter regression is left pending because no fitted regression result
          is stored yet; and
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
        self._updating = True
        try:
            self._redraw()
            # A restored, successfully painted snapshot is the picture the controls
            # describe.  Update only the run-gate projection; never history/dirty.
            self._plot_armed = self._message_item is None
            self._stale = False
            self._stale_reason = ""
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
        if self._fit_item is None:
            self._fit_item = item
        if self._last_fit is None:
            self._last_fit = fit
        if src is not None:
            self._fits[src.key] = fit
            self._fit_columns[src.key] = col

        # Match the ordinary headline parameter, but read every value from the saved
        # fit.  The full on-plot box remains the common formatter and therefore keeps
        # units, CIs and BIC consistent with a freshly rendered fit.
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
        full = self._full_fit_summary(fit, col, None)
        legend_who = self._legend_name(src) if (who and src is not None) else who
        self._legend_add(item,
                         f"{legend_who} fit: {summary}".strip() if legend_who
                         else f"fit: {summary}",
                         kind="fit", key=key)

        notes = [
            str(getattr(fit, "model", "saved fit")),
            (f"BIC = {float(fit.bic):.6g}   AIC = {float(fit.aic):.6g}   "
             f"(restored snapshot; not recomputed)"),
        ]
        ci_line = self._ci_note(fit, col)
        if ci_line:
            notes.append(ci_line)
        notes.extend(str(note) for note in getattr(fit, "meta", {}).get("warnings", ()))
        return _FitSummary(summary, full, plain=summary), notes

    def _clear_plot(self) -> None:
        self._plot.clear()
        self._plot.setTitle(None)
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
        # Parented to the ViewBox, not added through addItem — plot.clear() cannot
        # reach it, so it is detached explicitly (same reason _drop_legend exists).
        if getattr(self, "_fit_values_item", None) is not None:
            self._fit_values_item.setParentItem(None)
            sc = self._fit_values_item.scene()
            if sc is not None:
                sc.removeItem(self._fit_values_item)
        self._fit_values_item = None
        # Same ViewBox-child rule as the fit-values box above.
        if getattr(self, "_prov_item", None) is not None:
            self._prov_item.setParentItem(None)
            _sc = self._prov_item.scene()
            if _sc is not None:
                _sc.removeItem(self._prov_item)
        self._prov_item = None
        self._band_items = []
        self._clear_residual_strip()
        self._last_fit = None
        self._fits = {}
        self._fit_columns = {}

    # -- legend ----------------------------------------------------------------------------------
    def _draw_fit_values(self, lines: list[str]) -> None:
        """The fitted parameters as a text box ON the figure (owner note #14).

        A ``pg.LabelItem`` anchored to the ViewBox's top-LEFT corner (the legend owns
        the top-right), so it stays put under pan/zoom/log and lands in the saved
        image — which Details, where these numbers otherwise live, does not. One line
        per fitted source, in the fit colour, at the style's font size."""
        if not self._style.fit_values_on_plot or not lines:
            return
        text = "<br>".join(_html.escape(str(l)) for l in lines if str(l).strip())
        if not text:
            return
        item = pg.LabelItem(text, color=self._style.fit_color,
                            size=f"{self._style.font_size_pt}pt", justify="left")
        item.setParentItem(self._plot.vb)
        item.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 8))
        item.setZValue(30)
        self._fit_values_item = item

    # -- E2: what made these events, ON the figure -----------------------------------------------
    #: How a detector id is written for a human. `provenance['detector']` is a module
    #: string ("schmitt_hysteresis", "cusum_sequential", "ml_keras"); an unknown one is
    #: printed VERBATIM rather than dropped, because a detector this table does not know
    #: is exactly the case where the reader most needs to be told which one ran.
    _DETECTOR_NAMES: "dict[str, str]" = {
        "schmitt_hysteresis": "Schmitt threshold",
        "cusum_sequential": "CUSUM",
        "ml_keras": "ML (Keras)",
        "reconciled": "Reconciled detectors",
    }

    @staticmethod
    def _preprocessing_chain(prov: Mapping) -> str:
        """The preprocessing stages that RAN, in the order the signal met them.

        THE STAGES THAT RAN, not the ones that were offered: every ``*_enabled`` key is a
        boolean the module stamped after deciding, so a chain printed from them describes
        this recording rather than the panel's defaults. An empty string means the raw
        signal went to the detector, which is itself worth being able to read off a figure.

        Each stage carries the ONE number that changes what it does -- a notch without its
        frequency and a high-pass without its corner are not reproducible descriptions --
        and nothing else, because this line has to fit under a plot.
        """
        parts: list[str] = []
        if prov.get("detrend_enabled"):
            win = prov.get("detrend_window_s")
            parts.append(f"detrend {prov.get('detrend_method', 'block_median')}"
                         + (f" {float(win):g} s" if win else ""))
        if prov.get("notch_enabled"):
            freqs = prov.get("notch_freqs")
            if freqs:
                parts.append(f"notch {', '.join(f'{float(f):g}' for f in freqs)} Hz")
            else:
                n_h = int(prov.get("notch_harmonics") or 1)
                parts.append(f"notch {float(prov.get('notch_freq_hz') or 0):g} Hz"
                             + (f" ×{n_h}" if n_h > 1 else ""))
        if prov.get("median_enabled"):
            parts.append(f"median {float(prov.get('median_window_s') or 0) * 1e3:g} ms")
        if prov.get("hp_enabled"):
            parts.append(f"high-pass {float(prov.get('hp_cutoff_hz') or 0):g} Hz")
        if prov.get("rmean_enabled"):
            parts.append(f"rolling mean {float(prov.get('rmean_window_s') or 0) * 1e3:g} ms")
        if prov.get("savgol_enabled"):
            parts.append(f"Savitzky–Golay {float(prov.get('savgol_window_s') or 0) * 1e3:g} ms")
        if prov.get("normalize_enabled"):
            parts.append(f"normalize {prov.get('normalize_mode', 'zscore')}")
        return " → ".join(parts)

    def _provenance_line(self, src: PlotSource) -> str:
        """One source's caption: detector, model, preprocessing. ``""`` when unknown.

        WHY THIS AND NOT ``_note_cache``. That cache holds deferred STATISTICS -- keyed by
        a sample digest and a censoring window, invalidated when the numbers move -- and
        provenance moves for none of those reasons: it is a property of the RUN that made
        the rows, already sitting on :attr:`PlotSource.provenance`, and it is stale only
        when the source is. Routing it through the notes machinery would give it an
        expiry rule it does not have and a worker it does not need.

        THE MODEL IS NAMED BY ITS FILE, not by its preset alone. The owner runs three
        differently-trained detectors over the same chemistry (HG, Zn, Cu), and they are
        distinguished by which ``.keras`` file was loaded -- the preset names the decode
        thresholds, not the weights. The basename is used because the full path is a home
        directory and does not belong in a published figure.
        """
        prov = dict(getattr(src, "provenance", None) or {})
        if not prov:
            return ("manual measurement table (no detector provenance)"
                    if getattr(src, "kind", "") == "manual" else "")
        det = str(prov.get("detector") or "")
        parts = [self._DETECTOR_NAMES.get(det, det) if det else "detector not recorded"]
        model = prov.get("model_path")
        if model:
            parts.append(f"model {Path(str(model)).name}")
        preset = prov.get("ml_preset")
        if preset:
            parts.append(f"preset {preset}")
        decode = prov.get("ml_decode_preset")
        if decode and str(decode) != "legacy":
            parts.append(f"decode {decode}")
        chain = self._preprocessing_chain(prov)
        parts.append(chain if chain else "no preprocessing")
        return " · ".join(p for p in parts if p)

    def _draw_provenance_caption(self) -> None:
        """Write each drawn source's provenance under the plot, INSIDE the figure.

        Bottom-left, because the other two corners are taken: the fit-values box owns the
        top-left and the legend the top-right. Parented to the ViewBox exactly as those
        are, which is what puts it in the exported image and keeps it still under pan and
        zoom -- and, like them, means it must be detached by hand in :meth:`_clear_plot`,
        since ``PlotItem.clear()`` cannot reach a child of the ViewBox.

        Silent when the plot is showing a MESSAGE rather than data: "No data yet" with a
        preprocessing chain under it describes nothing.

        Each line is led by the name the FIGURE calls its source (:meth:`_legend_name`,
        the one resolver), not the raw override: this caption is a plot item Save image
        keeps, and with two rows typed to "Apo" it read "Apo: manual measurement table"
        twice under a legend that had just told the two apart (A1-N2).
        """
        if not self._style.show_provenance or self._message_item is not None:
            return
        drawn = self._drawn_frames()
        multi = len(drawn) > 1
        lines: list[str] = []
        for src, _df in drawn:
            line = self._provenance_line(src)
            if not line:
                continue
            lines.append(f"{self._legend_name(src)}: {line}" if multi else line)
        if not lines:
            return
        text = "<br>".join(_html.escape(l) for l in lines)
        item = pg.LabelItem(text, color=self._msg_hex(),
                            size=f"{max(6, self._style.font_size_pt - 2)}pt",
                            justify="left")
        item.setParentItem(self._plot.vb)
        item.anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(8, -6))
        item.setZValue(30)
        self._prov_item = item

    # -- the fit's own uncertainty, and where it misses ------------------------------------------
    def _clear_residual_strip(self) -> None:
        """Empty the strip and take it off screen. Every draw starts here, so a strip can
        only ever show the residuals of the picture currently above it."""
        if not hasattr(self, "_resid"):
            return      # may run during construction, before _build_ui
        self._resid.clear()
        self._resid_widget.setVisible(False)

    def _draw_ci_band(self, fit, grid: np.ndarray, at: np.ndarray, scale, colour) -> str:
        """Shade the bootstrap's 95% pointwise band around the drawn fit curve.

        ``""`` when there is nothing to draw, which is the common case and is NOT an
        error: :func:`stats.ci_band` declines for a fit restored from a project, a fit
        made under ``point_fits_only``, a family with no bootstrap, and a bootstrap with
        too few usable replicates. A note is returned only when a band WAS drawn, because
        the alternative -- explaining the absence on every unfitted redraw -- is the
        clutter the switch exists to avoid.

        The band is put through the SAME ``expected`` arithmetic as the curve
        (``fit.n * density * width * _fit_scale``) rather than being scaled separately:
        the two must agree bin for bin, and one expression they both use is the only way
        to guarantee it under three normalizations and a log x axis.
        """
        band = stats.ci_band(fit, at)
        if band is None:
            return ""
        lo = np.asarray(band["lo"], dtype=np.float64) * scale
        hi = np.asarray(band["hi"], dtype=np.float64) * scale
        pen = pg.mkPen(QtGui.QColor(colour), width=1, style=Qt.DotLine)
        c = QtGui.QColor(colour)
        c.setAlpha(60)      # readable over the bars without hiding them
        top = pg.PlotDataItem(x=grid, y=hi, pen=pen)
        bot = pg.PlotDataItem(x=grid, y=lo, pen=pen)
        fill = pg.FillBetweenItem(top, bot, brush=pg.mkBrush(c))
        for it in (top, bot, fill):
            self._plot.addItem(it)
        # NOT a legend row. The band is an attribute of the fit curve that is already
        # legended, and a second entry for one object is the note-Q2 complaint the legend
        # checklist exists to fix.
        self._band_items.extend([top, bot, fill])
        return str(band["note"])

    def _draw_residual_strip(self, fit, sample, edges: np.ndarray,
                             log_x: bool) -> tuple[str, str]:
        """Standardised residuals per bin under the main axes, and their reduced χ².

        WHAT THIS ANSWERS THAT KS DOES NOT. The KS verdict is one number over the whole
        curve: it says the fit is or is not consistent, never where it fails. A run of
        same-sign bars here is the finding -- a gamma fit that is high through the peak
        and low through the tail is a wrong FAMILY, and looks nothing like the scatter of
        a right one, whatever either one's D happens to be.

        Returns ``(sentence, short)`` -- the full χ² paragraph for Details and the
        one-token form for the on-plot values box, which is the half an exported PNG
        keeps. Both ``""`` when nothing could be computed: a family whose density will not
        integrate over these bins loses its strip, exactly as it loses its KS verdict, and
        never raises into a draw.
        """
        self._clear_residual_strip()
        try:
            res = stats.binned_residuals(fit, sample, edges)
        except (ValueError, RuntimeError, TypeError, ZeroDivisionError,
                FloatingPointError):
            return "", ""
        r = np.asarray(res["residual"], dtype=np.float64)
        x = np.asarray(res["centers"], dtype=np.float64)
        colour = QtGui.QColor(self._style.fit_color)
        # Stems, not a connected line: each residual belongs to ONE bin and joining them
        # draws a curve through numbers that have no order beyond their bins.
        for xi, ri in zip(x, r):
            self._resid.addItem(pg.PlotDataItem(x=[xi, xi], y=[0.0, float(ri)],
                                                pen=pg.mkPen(colour, width=2)))
        self._resid.addItem(pg.PlotDataItem(
            x=x, y=r, pen=None, symbol="o", symbolSize=max(4, self._style.point_size - 2),
            symbolBrush=pg.mkBrush(colour), symbolPen=None))
        # ±2 is where a standardised residual stops being ordinary counting noise; the
        # guides are what turn "a tall bar" into "a tall bar for this many counts".
        for level, style in ((0.0, Qt.SolidLine), (2.0, Qt.DashLine), (-2.0, Qt.DashLine)):
            self._resid.addItem(pg.InfiniteLine(
                pos=level, angle=0,
                pen=pg.mkPen(self._msg_hex(), width=1, style=style)))
        self._resid.setLogMode(x=bool(log_x), y=False)
        self._resid.setLabel("left", "(obs−exp)/√exp", **self._label_style())
        self._resid.getAxis("bottom").setStyle(showValues=False)   # the plot above labels it
        self._resid_widget.setBackground(self._bg_spec())
        # Symmetric about zero and never tighter than ±3, so the guides are always in
        # frame and two strips of the same data are comparable by eye.
        span = max(3.0, float(np.max(np.abs(r))) * 1.15) if r.size else 3.0
        self._resid.setYRange(-span, span, padding=0)
        self._resid_widget.setVisible(True)
        # "approx." is not decoration: the dof came from an unbinned MLE, and a bare
        # "χ²/dof = 1.3" on an exported figure is read as the textbook quantity.
        short = ("" if res["reduced_chi2"] is None
                 else f"χ²/dof = {float(res['reduced_chi2']):.2f} (approx. dof)")
        return str(res["note"]), short

    def _drop_legend(self) -> None:
        """Empty the legend and hide it.

        ``PlotItem.clear()`` empties the ViewBox but leaves ``self.legend`` in place
        holding its stale rows, and ``addLegend`` then returns that same item -- so
        without this the entries of every previous draw accumulate forever. The item
        is emptied rather than destroyed: ``LegendItem.setParentItem(None)`` re-runs
        its anchoring and raises "Cannot anchor; parent is not set", so detaching one
        is a trap. Hidden while empty, because an empty box floating over a
        single-series plot is noise.
        """
        legend = getattr(self._plot, "legend", None)
        if legend is None:
            return
        legend.clear()
        legend.setVisible(False)

    def _ensure_legend(self):
        """The legend, created if needed and re-inked for the CURRENT theme.

        Restyled on every draw because its text colour and translucent backing have
        to follow the dark/light switch the Style group offers -- a legend that stays
        light-grey on the white publication canvas is not a legend, it is a smudge.
        """
        fg = QtGui.QColor(self._fg_hex())
        backing = QtGui.QColor(self._bg_spec())
        backing.setAlpha(200)     # readable over bars/points without hiding them
        border = QtGui.QColor(fg)
        border.setAlpha(90)
        existing = getattr(self._plot, "legend", None)
        legend = self._plot.addLegend(offset=_LEGEND_DEFAULT_OFFSET)
        if existing is None:
            # First legend of this window's life. ``PlotItem.addLegend`` returns the
            # EXISTING item untouched from here on, so this is the one chance to hook
            # the drag -- and the hook is how a dragged position gets remembered.
            self._install_legend_drag(legend)
        self._apply_legend_position(legend)
        legend.setLabelTextColor(fg)
        legend.setBrush(pg.mkBrush(backing))
        legend.setPen(pg.mkPen(border))
        # Three points BELOW the body font, floored at 6, with tight padding. A legend
        # is an INDEX, not content: at font-1 with the default 10 pt body it rendered
        # large enough to cover a meaningful part of the plot area, and what it covers
        # is the data it names. That rule is the DEFAULT (legend_font_size_pt = 0), not
        # a law: a figure going into a talk needs a legend that reads from the back of
        # the room, and one going into a dense multi-panel needs it smaller still.
        pt = (int(self._style.legend_font_size_pt) or
              max(6, self._style.font_size_pt - 3))
        legend.setLabelTextSize(f"{max(1, pt)}pt")
        try:
            legend.layout.setContentsMargins(2, 2, 2, 2)
            legend.layout.setHorizontalSpacing(4)
            legend.layout.setVerticalSpacing(0)
        except Exception:      # pragma: no cover - pyqtgraph layout internals
            pass
        legend.setVisible(True)
        return legend

    # -- legend PLACEMENT: the user's, dragged, and kept (round-4 Q4) -----------------------------
    def _install_legend_drag(self, legend) -> None:
        """Record where the user drops the legend, every time they drop it.

        pyqtgraph's ``LegendItem`` has been draggable since long before this window
        existed -- ``LegendItem.mouseDragEvent`` calls ``autoAnchor`` on each mouse
        move -- so the FEATURE was never missing; what was missing was memory of it,
        and what actively destroyed it was a ``setOffset`` on every draw (see
        :data:`_LEGEND_DEFAULT_OFFSET`). This wraps the real handler rather than
        replacing it: pyqtgraph still does the dragging, and the wrapper's only job is
        to read the result back afterwards.

        Wrapped on the INSTANCE, which works because pyqtgraph dispatches these itself
        in Python (``GraphicsScene`` calls ``item.mouseDragEvent(ev)``), so the
        instance attribute shadows the class method the way a normal attribute lookup
        does.
        """
        base = legend.mouseDragEvent

        def _dragged(ev, _base=base):
            _base(ev)
            self._remember_legend_position()

        legend.mouseDragEvent = _dragged

    def _remember_legend_position(self) -> None:
        """Store the legend's current spot as a fraction of the plot area.

        Read off the live item rather than out of the drag event, so it is the position
        pyqtgraph actually settled on -- ``autoAnchor`` picks the nearest corner and
        may land a pixel away from the raw pointer delta.
        """
        legend = getattr(self._plot, "legend", None)
        parent = legend.parentItem() if legend is not None else None
        if parent is None:
            return
        br = parent.boundingRect()
        if br.width() <= 0 or br.height() <= 0:
            return
        pos = legend.pos()
        self._style = _dc_replace(
            self._style, legend_pos=(pos.x() / br.width(), pos.y() / br.height()))
        self._sync_legend_reset_button()

    def _apply_legend_position(self, legend) -> None:
        """Put the legend where the user left it -- or, if they never moved it, where
        this window has always started it.

        THE RE-APPLY IS SAFE TO RUN ON EVERY DRAW, which is the fact the four-corner
        combo got wrong. ``autoAnchor(pos, relative=True)`` is idempotent: measured on
        the owner's two-source dwell overlay, five successive re-applies of a dragged
        (287.0187, 192.0) all landed on exactly (287.0187, 192.0). ``setOffset`` is
        not -- it re-anchors to a CORNER, which is why it threw the drag away (measured:
        drag to (287, 192), next redraw put it back at (486.7, 12)).
        """
        parent = legend.parentItem()
        if parent is None:      # pragma: no cover - addLegend always parents to the vb
            return
        pos = self._style.legend_pos
        if pos is None:
            # Never dragged: the historical top-right corner. Applied every draw is
            # harmless here precisely because there is no user position to destroy.
            legend.setOffset(_LEGEND_DEFAULT_OFFSET)
            return
        br = parent.boundingRect()
        if br.width() <= 0 or br.height() <= 0:   # pragma: no cover - un-laid-out view
            return
        legend.autoAnchor(pg.Point(pos[0] * br.width(), pos[1] * br.height()),
                          relative=True)

    def _reset_legend_position(self) -> None:
        """Send the legend back to the default corner (the Style group's button).

        The four-position combo this replaces is what the owner rejected -- "the
        location should not be limited to four fixed positions ... it should be
        interactive". A RESET is a different thing from a position setting: it does not
        run on every draw, so it cannot fight the drag, and it answers the one real
        need a free drag creates, which is getting back from a legend dropped somewhere
        useless.
        """
        self._style = _dc_replace(self._style, legend_pos=None)
        self._sync_legend_reset_button()
        self._changed()

    def _sync_legend_reset_button(self) -> None:
        """Enable Reset only when there is something to reset -- the button doubles as
        the only on-screen statement that the legend has been moved at all."""
        if not hasattr(self, "_legend_reset_btn"):
            return   # fires during construction, before the Style group exists
        moved = self._style.legend_pos is not None
        self._legend_reset_btn.setEnabled(moved)
        self._legend_reset_btn.setToolTip(
            "Put the legend back in the top-right corner." if moved else
            "The legend is where it started — drag it on the plot to move it.")

    #: Longest legend entry drawn in full. A legend is sized by its widest label, and a
    #: source label is a RECORDING FILENAME -- the owner's run to a hundred characters
    #: ("500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv"), which made the
    #: legend span most of the plot width no matter how small the font was set. Shrinking
    #: the text alone could never fix that; the width comes from the string.
    _LEGEND_LABEL_CHARS = 34

    @classmethod
    def _elide_legend(cls, label: str) -> str:
        """Middle-elide a long legend label. The middle goes because the ends carry the
        identity -- the leading concentration/sample and the trailing voltage/`n` are what
        tell two of this owner's recordings apart; the centre is boilerplate they share.
        The full string stays in Details, which is where the unabridged provenance lives."""
        if len(label) <= cls._LEGEND_LABEL_CHARS:
            return label
        keep = cls._LEGEND_LABEL_CHARS - 1
        head, tail = (keep + 1) // 2, keep // 2
        return f"{label[:head]}…{label[-tail:]}"

    def _legend_label(self, src: PlotSource | None) -> str:
        """What the legend calls ``src``: the user's own text if they retyped it.

        Owner note #15, "the legend is not editable". A source's label is a RECORDING
        FILENAME — "500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv" —
        and no figure caption in a paper says that; it says "apo, −500 mV". The
        override is edited in place on the source list (double-click a row) and stored
        per source KEY, so it survives a Refresh that re-pulls the same recordings.
        """
        if src is None:
            return ""
        return self._style.label_overrides.get(src.key, src.label) or src.label

    def _legend_names(self) -> dict[str, str]:
        """What the DRAWN legend calls each ticked source: :meth:`_legend_label`, made
        unique among the sources on the figure.

        THE ONE RESOLVER FOR THE FIGURE'S NAMES, and the reason it is not
        :meth:`_legend_label` itself. Source LABELS are unique by construction
        (:func:`nanoscope.core.sources.disambiguate_labels` runs in discovery), but an
        OVERRIDE is free text, and two rows can be typed to one name -- deliberately, or
        by the stranded-override route S7-SK2 measured: rename -700 mV to "Apo", type
        the original back (the row obeyed, the override did not -- fixed in
        :meth:`_on_source_item_changed`), then give "Apo" to -600 mV, and the canvas and
        the exported SVG both read ['Apo (n = 982)', 'Apo (n = 236)'] with no warning
        anywhere. A figure with two curves under one name cannot be read, and Save
        image exports exactly the legend drawn here.

        SO A COLLISION IS TOLD APART ON THE FIGURE, by the same function the source
        list uses, with the source's own label as the distinguishing name ("Apo —
        -700 mV" / "Apo — -600 mV"): it is the one name guaranteed unique per source, it
        is what the row's tooltip already shows, and it lets a reader of the exported
        figure see which recording is which without the window. The ROWS keep showing
        what was typed (``_legend_label``): they are the rename affordance, and writing
        the disambiguated form back onto them would be read as a new override by the
        next tick and would move the user's text under them. That the figure differs
        from the rows is said in Details (:meth:`_legend_collision_notes`).

        Cheap: a handful of ticked sources, a Counter and a dict, per legend entry.
        """
        sources = self.selected_sources()
        entries = [{"label": self._legend_label(src), "channel_name": src.label,
                    "is_segment": bool(src.is_segment)} for src in sources]
        disambiguate_labels(entries)
        return {src.key: entry["label"] for src, entry in zip(sources, entries)}

    def _legend_name(self, src: PlotSource | None) -> str:
        """The legend entry's name for ``src`` on THIS figure -- see :meth:`_legend_names`.
        Falls back to :meth:`_legend_label` for a source that is not ticked (a fit drawn
        for the primary source on a single-series plot)."""
        if src is None:
            return ""
        return self._legend_names().get(src.key) or self._legend_label(src)

    def _legend_collision_notes(self) -> list[str]:
        """One sentence per legend name shared by two or more ticked sources, or ``[]``.

        The disclosure half of :meth:`_legend_names`: the figure has been drawn with the
        names told apart, and the user has to be told that it differs from what they
        typed and which row to retype -- in Details, the pane every other refusal is
        said in and the one that is on screen without scrolling.
        """
        sources = self.selected_sources()
        if len(sources) < 2:
            return []
        drawn = self._legend_names()
        groups: dict[str, list[PlotSource]] = {}
        for src in sources:
            groups.setdefault(self._legend_label(src), []).append(src)
        out: list[str] = []
        for label, group in groups.items():
            if len(group) < 2:
                continue
            which = ", ".join(src.label for src in group)
            told_apart = ", ".join(f"“{drawn[src.key]}”" for src in group)
            out.append(
                f"LEGEND: {len(group)} of the drawn sources are both called “{label}” "
                f"({which}), so the legend and any saved image name them {told_apart} "
                f"— two curves under one name cannot be read. Retype one of them in "
                f"the Sources list to name them apart.")
        return out

    def _legend_entry(self, label: str, n: int | None = None) -> str:
        """A legend entry, with the automatic ``(n = N)`` only when it is wanted.

        The count is genuinely useful on an overlay and it is today's default, but it
        must be defeatable: silently welding "  (n = 986)" onto a label the user has
        just RETYPED overrides the edit they made in the same breath as honouring it.
        """
        if n is None or not self._style.legend_counts:
            return label
        return f"{label}  (n = {n})"

    @staticmethod
    def _legend_id(kind: str, key: str = "") -> str:
        """The stable id one legend entry is remembered by: ``"<kind>:<key>"``.

        KIND, NOT LABEL, because the label is the volatile part -- it carries the fitted
        tau, the n, and whatever the user retyped, so keying inclusion on it would lose
        the setting the moment a number moved. ``series`` and ``fit`` are keyed by the
        SOURCE key for the same reason ``label_overrides`` is: a refresh re-identifies
        sources by key, and an inclusion decision must not slide onto another curve.
        """
        return f"{kind}:{key}"

    def _legend_included(self, entry_id: str) -> bool:
        """Whether this entry is in the legend. Absent means yes -- an empty
        ``legend_include`` is today's look."""
        return bool(self._style.legend_include.get(entry_id, True))

    def _legend_add(self, item, label: str, *, kind: str = "series", key: str = "") -> None:
        """Offer one entry to the legend, subject to the two inclusion gates.

        THE OFFER IS RECORDED WHETHER OR NOT IT IS DRAWN. ``_legend_offered`` is what
        the Style group's *Legend entries* checklist is rebuilt from, and a checklist
        that only listed the entries currently shown would be a list you could untick
        rows out of and never get back.

        Owner note Q2 -- "when the fit is performed and the legend is selected, the app
        includes icons for both the histogram and the lines. This is incorrect; the
        user needs control over which icons to add to the legend."
        """
        entry_id = self._legend_id(kind, key)
        self._legend_offered[entry_id] = label
        # The one gate for "Show legend" (owner request): every entry flows through
        # here, so unticking the box empties the next draw's legend everywhere at once.
        if not self._style.show_legend or not self._legend_included(entry_id):
            return
        legend = self._ensure_legend()
        legend.addItem(item, self._elide_legend(label))
        legend.setVisible(True)

    def _shade_below_limit(self, t_min: float, lo: float, log_x: bool) -> None:
        """Shade the region the detector could never report, as figures.py does.

        ``figures.dwell_histogram`` shades ``[left, t_min]`` and drops a dashed
        line on the limit because the histogram's left edge is an artefact of the
        detector, not of the physics -- and that is exactly why the fit is a
        censored MLE. The interactive histogram must not be the one view that
        hides it. Coordinates are converted to log10 by hand: ``setLogMode``
        transforms plot DATA items, not items added straight to the scene.
        """
        left = min(t_min, lo) * 0.5
        if not (t_min > left > 0):
            return
        to_view = (lambda x: float(np.log10(x))) if log_x else float
        shade = QtGui.QColor(_EXCLUDED_HEX)
        shade.setAlphaF(0.30)   # the alpha figures.py's axvspan uses
        self._limit_region = pg.LinearRegionItem(
            values=(to_view(left), to_view(t_min)), movable=False,
            brush=pg.mkBrush(shade), pen=pg.mkPen(None))
        self._limit_region.setZValue(-10)
        self._plot.addItem(self._limit_region)
        self._limit_line = pg.InfiniteLine(
            pos=to_view(t_min), angle=90,
            pen=pg.mkPen(_EXCLUDED_HEX, width=1, style=Qt.DashLine),
            label="below detection limit",
            labelOpts={"position": 0.06, "color": _EXCLUDED_HEX, "movable": False})
        self._plot.addItem(self._limit_line)

    def _message(self, text: str) -> None:
        """Blank the plot and show a friendly centred note (empty/invalid state).

        DETAILS CARRIES THE REFUSALS TOO, and the centred note does not. This method is
        the ONE funnel for every early return in every draw path -- "no data yet", "tick
        a source", "pick a numeric column", "pick numeric X and Y", "no points for this
        pair", "select ≥2 feature columns, then click Run PCA" -- and it replaced Details
        wholesale, so a window in any of those states said nothing about a configuration
        it had just been handed and could not honour. Fixing it HERE rather than at the
        fourteen call sites is the point: an early return that skips the disclosure was
        the bug, not any one page, and a fifteenth empty state arrives disclosed.

        MEASURED BOTH WAYS on all four pages, by unticking every source so that each one
        takes the shared "tick a source" early return with a refused config installed:
        before this line the sentence was absent on 4 of 4, after it present on 4 of 4.

        THE CENTRED TEXT ITEM IS LEFT ALONE deliberately. It is drawn INTO the figure and
        goes into an exported image; the instruction it carries is what the user has to do
        next, and a paragraph about somebody's file printed across the canvas is neither
        that nor something anyone wants in a saved PNG. Details is where the window's own
        account of itself already lives and it is on screen without scrolling even with
        both sentences in it -- measured on a 1200x780 window: the pane is 803x180 px, its
        ``visibleRegion()`` is non-empty at rest and its vertical scrollbar range is 0..0,
        where the same probe over the same text plus 40 filler lines reports 0..37.
        """
        self._clear_plot()
        self._apply_background()
        self._plot.setLogMode(x=False, y=False)
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=self._msg_hex())
        self._plot.addItem(item)
        self._plot.setXRange(0, 1, padding=0)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._set_details("\n".join([text, *self._incoming_notes()]))

    def _set_details(self, text: str) -> None:
        self._details.setPlainText(text)
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        self._result_summary.setText(lines[0] if lines else "No result yet.")

        # One actionable warning, not a second copy of the report. Explicit warning
        # markers win; otherwise take the first line that describes a material omission
        # or invalidation. The complete wording remains one click away in Details.
        warning = next((line for line in lines
                        if line.startswith(("!", "⚠")) or "NOT COMPARABLE" in line.upper()), "")
        if not warning:
            markers = (" failed", " unavailable", " dropped", " ignored", " refused",
                       " exact by construction", " could not", " cannot ",
                       " not certified")
            warning = next((line for line in lines
                            if any(marker in f" {line.lower()}" for marker in markers)), "")
        if len(warning) > 280:
            warning = warning[:277].rstrip() + "…"
        self._result_warning.setText(warning)
        self._result_warning.setVisible(bool(warning))

    @staticmethod
    def _same_row_identity(left: object, right: object) -> bool:
        """Equality for pandas index scalars without leaking array-valued comparisons."""
        try:
            equal = left == right
            return bool(equal) if np.ndim(equal) == 0 else bool(np.all(equal))
        except (TypeError, ValueError):
            return left is right

    def _make_event_points_clickable(
        self, item: pg.ScatterPlotItem, source_key: str, row_ids: Sequence[object],
    ) -> None:
        """Attach exact source-row identities to a plotted Scatter/PCA point cloud."""
        identities = list(row_ids)
        if len(identities) != len(item.points()):
            raise RuntimeError("point/event identity count mismatch")
        # ``setData`` would replace x/y; assigning through the spots preserves their
        # coordinates and lets ``SpotItem.data()`` be the sole identity seam.
        for point, row_id in zip(item.points(), identities):
            point.setData(row_id)
        item.setCursor(Qt.PointingHandCursor)
        item.setToolTip(
            "Click to inspect this event; Ctrl/Cmd/Shift-click to select several.")
        item.sigClicked.connect(
            lambda clicked, points, event, key=source_key:
                self._on_event_points_clicked(key, clicked, points, event))
        self._event_point_items.setdefault(source_key, []).append(item)

    @staticmethod
    def _point_identities(src: PlotSource, df: pd.DataFrame,
                          mask: np.ndarray | None = None) -> list[object]:
        """Stable identities for plotted rows, in their exact plotted order.

        Detector ``event_id`` is the table/trace navigation key. A dataframe index is
        only a label and is explicitly allowed to be duplicated or NaN by EventSet, so
        using it could make one point select another row. Bare/manual frames keep their
        index because they have no detector event to navigate to.
        """
        if src.kind == "detection" and "event_id" in df.columns:
            values = df["event_id"].to_numpy(copy=False)
        else:
            values = df.index.to_numpy(copy=False)
        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]
        return list(values)

    def _on_event_points_clicked(self, source_key: str, _item: pg.ScatterPlotItem,
                                 points: Sequence[object], event: object | None) -> None:
        """Maintain an additive visual selection and emit its exact dataframe rows."""
        clicked = [point.data() for point in points]
        if not clicked:
            return
        modifiers = (event.modifiers() if event is not None
                     and hasattr(event, "modifiers") else Qt.NoModifier)
        additive = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier))
        if additive:
            # MainWindow can display one detector table at a time. Leaving source A
            # ringed while emitting only source B would show a selection that was not
            # represented in the table or signal, so additive selection is deliberately
            # scoped to one source.
            self._selected_event_rows = {
                key: value for key, value in self._selected_event_rows.items()
                if key == source_key
            }
            selected = list(self._selected_event_rows.get(source_key, ()))
            for row_id in clicked:
                found = next((i for i, old in enumerate(selected)
                              if self._same_row_identity(old, row_id)), None)
                if found is None:
                    selected.append(row_id)
                else:
                    selected.pop(found)
        else:
            self._selected_event_rows.clear()
            selected = clicked
        self._selected_event_rows[source_key] = selected
        self._paint_event_point_selection()
        self.eventRowsSelected.emit(source_key, tuple(selected))

    def _paint_event_point_selection(self) -> None:
        """Ring selected points without replacing source/cluster colour encodings."""
        ring = pg.mkPen(_WARN_HEX, width=2)
        for source_key, items in self._event_point_items.items():
            selected = self._selected_event_rows.get(source_key, ())
            for item in items:
                for point in item.points():
                    on = any(self._same_row_identity(point.data(), row_id)
                             for row_id in selected)
                    point.setPen(ring if on else pg.mkPen(None))
                    point.setSize(self._style.point_size + 4 if on else self._style.point_size)

    def _redraw(self) -> None:
        """Draw the current view, then re-offer its legend rows to the Style group.

        The checklist is rebuilt in a ``finally`` because every one of the draw's
        early-outs (no sources, nothing checked, an empty frame) is a legitimate result
        with its own -- usually empty -- set of legend entries, and a checklist still
        listing the previous plot's rows is a control pointing at nothing.
        """
        try:
            self._redraw_current()
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
                # KDE currently stores its recipe, not the sampled density grid.
                # Calling scipy's estimator here would make project-open an implicit
                # analysis run, so leave an explicit pending picture instead.
                self._message(
                    "Saved KDE configuration restored. Press Plot to recompute its "
                    "density contours from the attached source data.")
                return
            self._draw_kde(drawn)
        else:
            df = self._current_frame()
            if df is None or df.shape[0] == 0:
                self._message("This source has no rows to plot.")
                return
            self._draw_pca(df)
        # E2, once for every plot type rather than four times. Each branch above ends by
        # having called `_clear_plot`, so the caption is added to a fresh figure and can
        # never survive from the previous one; it declines to draw over a message.
        self._draw_provenance_caption()

    # -- histogram -------------------------------------------------------------------------------
    @staticmethod
    def _shared_edges(samples: Sequence[np.ndarray], nbins: int, log_x: bool) -> np.ndarray:
        """ONE bin-edge set spanning every drawn sample.

        Binning each source over its own min/max would put the bars of two sources
        on different grids: the eye reads two shapes that were never measured
        against the same ruler, and the taller-looking one may simply have narrower
        bins. Shared edges are what makes an overlay a comparison.

        The outermost edges are pinned back onto the exact ``lo``/``hi`` the samples
        actually contain. ``np.logspace`` round-trips through ``log10`` and back, so
        its end edges can land an ulp *inside* the data range (e.g. ``lo =
        5.001781298025952e-05`` came back as ``5.0017812980259555e-05``) -- and
        ``np.histogram`` then silently drops the smallest and/or largest event. That
        is not cosmetic on an overlay: the loss lands on whichever source owns each
        extreme, so under *density* the two series integrate to DIFFERENT
        values (measured: 0.9998 vs 0.9950 for a 5000- and a 200-event source) and
        the small-n source is penalised ~25x harder -- destroying exactly the
        comparability shared edges exist to provide. It also made the drawn bars
        disagree with the ``n = ...`` the legend prints.
        """
        lo = min(float(np.min(s)) for s in samples)
        hi = max(float(np.max(s)) for s in samples)
        if log_x:
            if hi <= lo:
                hi = lo * 10.0 if lo > 0 else 1.0
            edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
        else:
            if hi <= lo:
                hi = lo + 1.0     # a single-valued column still draws one bar
            edges = np.linspace(lo, hi, nbins + 1)
        edges[0], edges[-1] = lo, hi
        return edges

    def _normalized(self, counts: np.ndarray, edges: np.ndarray, n: int) -> np.ndarray:
        """Bin heights in the chosen y quantity.

        * ``count``: raw counts.
        * ``density``: ``counts / (n * width_i)`` — ``sum(y_i * w_i) = 1``,
          so a 5000-event and a 200-event recording become the same kind of number.
        * ``probability per bin``: ``counts / n`` — ``sum(y_i) = 1``.

        Widths are per-bin, so this is correct for the unequal widths of log bins
        as well as for uniform linear ones.

        THE WIDTHS ARE IN THE COLUMN'S STORED UNIT, which is what makes ``sum(y_i *
        w_i) = 1`` true and is the ONE thing this method must never learn about the
        display. ``_axis_label_for`` re-scales the x ticks for reading (ms, µs) and
        must not re-scale the sample -- the exponential fit takes its detection floor
        from the detector's ``min_dwell_s`` in seconds -- so ``y`` is per second for a
        dwell whatever the ticks say, and the drawn area is 1000 on the app's default
        ms axis and 23.3 on a log10 abscissa. This mode was called "density (area = 1)"
        until report S6-03; the values here were correct and the NAME was not, so the
        name went and :meth:`_y_axis_label` now puts the unit on the picture instead.
        """
        mode = self._norm_combo.currentText()
        y = counts.astype(np.float64)
        if mode == _NORM_COUNT or n <= 0:
            return y
        if mode == _NORM_PROB:
            return y / float(n)
        widths = np.diff(edges).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = y / (float(n) * widths)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_scale(self, edges: np.ndarray, grid_width, n: int) -> np.ndarray | float:
        """The factor turning an expected-COUNT fit curve into the drawn y quantity.

        Same three cases as :meth:`_normalized`, evaluated on the fit grid: for a
        density the local bin width divides straight back out (which is why a
        density curve is scale-free), for a probability only ``n`` does.
        """
        mode = self._norm_combo.currentText()
        if mode == _NORM_COUNT or n <= 0:
            return 1.0
        if mode == _NORM_PROB:
            return 1.0 / float(n)
        return 1.0 / (float(n) * np.asarray(grid_width, dtype=np.float64))

    def _y_label(self) -> str:
        """The y QUANTITY, with no unit on it.

        Read by :meth:`_y_axis_label` below AND by :meth:`_plot_data`, and it does not
        carry a unit because of the second reader -- see the comment there.
        """
        mode = self._norm_combo.currentText()
        return {_NORM_COUNT: "count", _NORM_DENSITY: "density", _NORM_PROB: "probability"}[mode]

    def _y_axis_label(self, col: str, log_y: bool) -> str:
        """The y label FOR THE PICTURE: the quantity, and the unit it is actually in.

        THIS IS THE HALF AN EXPORTED IMAGE KEEPS, which is the whole argument for putting
        the unit here and not only in Details. ``save_image`` renders this very PlotItem,
        so the SVG/PDF/PNG carries this string verbatim -- measured, the exported file
        contains ``dwell (ms)`` and, with this method in place, ``density (1/s)``, with no
        change to ``save_image`` at all. Details is screen-only and never leaves the app,
        and a figure that goes into a paper has to be checkable from itself.

        THE UNIT IS THE STORED ONE AND IT DOES NOT FOLLOW THE TIME-UNIT COMBO. The heights
        are ``counts / (n * bin_width)`` with the width in seconds, always, because
        ``_axis_label_for`` scales the TICKS and never the data (it must not: see its
        docstring on ``min_dwell_s``). So ``dwell_s`` reads ``density (1/s)`` whether the
        ticks say s, ms or us -- measured, all three. A label that tracked the ticks would
        be a second, subtler lie: it would say 1/ms over numbers that are per second.

        WHY THE NUMBERS ARE NOT RESCALED TO MATCH THE TICKS INSTEAD, which is the obvious
        alternative and was prototyped and measured before being rejected. (a) It cannot
        work on a log axis: the drawn abscissa is log10(seconds), the drawn area is
        23.3011485 in every time unit, and dividing the heights by the display factor moves
        it to 0.0233 (ms) / 2.33e-05 (us) -- further from 1, not closer -- while the app's
        own shipped log-x dwell template is exactly that case. (b) It makes one time unit
        unusable: the us y ticks render 0.00000 / 0.00005 / 0.00010 / 0.00015, and
        ``ReadableLogAxis`` has ``enableAutoSIPrefix(False)`` precisely so pyqtgraph cannot
        "help" with a ``(x1e-06)`` multiplier (report High-7). (c) It would make a DISPLAY
        control change a drawn number, contradicting ``AnalysisIdentity``'s own rule, and
        would force the y axis-range pin, ``_axis_unit_factors``, ``_apply_stored_ranges``
        and ``PlotConfig.rebased`` to carry the RECIPROCAL of the x factor -- six new places
        that must agree, to close a defect whose whole shape is one place that silently
        disagreed. Measured baseline this design preserves untouched: pin y at 0-200, switch
        ms -> us, and the box, ``_manual_y`` and the view all stay at 200.

        ONE RESOLVER FOR THE UNIT. :func:`_column_quantity` is the module's schema-first
        answer to "what is this column in" (``events.SCHEMA``, then the ``*_pA_s`` /
        ``*_pA`` / ``*_s`` suffix rules for the merged, hand-measured and derived tables
        this window also plots). Asking it rather than a second table here is what stops the
        two drifting -- the argument its own docstring makes. Surveyed over all 15
        histogrammable columns of the owner's table: nothing produced an empty or a doubled
        bracket.

        ``1/(pA·s)`` IS PARENTHESISED AND ``1/s`` IS NOT: ``ecd_pA_s`` is the schema's one
        compound unit, and ``1/pA·s`` reads as ``(1/pA)·s`` to precisely the reader this
        label exists for. A dimensionless column (``frac_depth``, ``snr``, ``n_levels``)
        gets no suffix at all -- ``1/`` there would be arithmetic nonsense dressed as rigour.

        ``log_y`` joins the SAME parenthesis instead of trailing a second one, so this reads
        ``density (1/s, log)`` exactly as the x axis reads ``dwell (ms, log)``. The caller
        used to concatenate ``" (log)"`` and would otherwise produce ``density (1/s) (log)``.

        BOTH PARAMETERS ARE REQUIRED AND POSITIONAL, deliberately. A default is what would
        let a future caller quietly put the unit-free label back on the picture -- which is
        this defect, re-introduced with no diagnostic. Missing one is a TypeError at
        import-test time instead.
        """
        base = self._y_label()
        parts: list[str] = []
        if self._norm_combo.currentText() == _NORM_DENSITY:
            unit = _column_quantity(col)[1]
            if unit:
                parts.append(f"1/({unit})" if any(c in unit for c in "·*/ ")
                             else f"1/{unit}")
        if log_y:
            parts.append("log")
        return f"{base} ({', '.join(parts)})" if parts else base

    def _norm_refusal_notes(self) -> list[str]:
        """The refused-y-quantity sentence for the page being drawn, or ``[]``.

        THE ONE READING OF :class:`_NormRefusal`, and the reason it is a method rather
        than two lines inside the histogram. Its callers reach it through
        :meth:`_incoming_notes`, which is the one SURFACE -- four draw paths and
        :meth:`_message` -- so that this record and :class:`_ConfigLosses` cannot be
        half-applied to a page. The refusal was
        rendered into the histogram's notes list only, so a configuration that is not a
        histogram was refused in SILENCE -- measured end to end on the owner's 982-event
        table with a v2 sidecar naming ``plot_type`` "Scatter" and ``density (1/s)``:
        ``import_config_json`` returned True, the window landed on Scatter, the record
        was set, Details said nothing and no dialog appeared. The one tell left was the
        template combo dropping to "— manual —" with no reason attached, and a re-export
        then wrote ``count`` over the sender's name on a path where nothing had ever said
        it was refused. The user has to be told BEFORE they save it back, and the page
        they are on when they save it back is whichever one their colleague's file chose.

        NOT MOVED TO THE DOOR INSTEAD, measured rather than argued: the door is a draw
        (:meth:`apply_config` is ``_set_config(cfg, draw=True)``), so a sentence printed
        there is wiped by the same call -- see :class:`_NormRefusal` for the marker
        measurement. And the sidecar is NOT rewritten to carry the refused name forward:
        ``export_config_json`` writes :meth:`config`, the settings actually on screen,
        and a sidecar saved beside an image ("the same gesture on two halves of one
        figure", in that method's own words) that named a y quantity the figure is not
        drawn in would be a second lie in place of the first. Disclosure, not a
        substitution, is what the whole of S6-03 is.

        WHERE IN EACH PAGE'S DETAILS: directly under that page's header lines and ahead
        of its NOTES -- the drops, the column caveats, the pooled verdict -- for the
        reason :meth:`_draw_histogram` gives at its own call: what the window was asked
        for and could not give outranks anything said about the quantity it drew
        instead.
        """
        note = self._norm_refusal.note(self._norm_combo.currentText(),
                                       on_axis=self._plot_type() == HISTOGRAM)
        return [note] if note else []

    def _incoming_notes(self) -> list[str]:
        """Everything a configuration handed this window and this window did not take.

        THE ONE SURFACE, and the reason it exists beside :meth:`_norm_refusal_notes`
        rather than as five pairs of calls: there are now TWO records saying what an
        incoming configuration lost (:class:`_NormRefusal` for the y quantity,
        :class:`_ConfigLosses` for the other six kinds), they belong in the same place in
        the same order on every page, and a page that asked for one and forgot the other
        would be a half-disclosure that reads like a whole one. Every draw path and the
        empty state ask THIS.

        AND THE EMPTY STATE ASKS IT TOO -- the fix for the hole round 4 left. Every page's
        early return goes through :meth:`_message`, which replaces Details wholesale, so
        a window that cannot draw said why it could not draw and nothing about what it was
        asked for. That is not a corner: a configuration cannot carry a PCA RESULT, so
        EVERY import onto the PCA page lands in exactly that state until Run PCA is
        pressed. Measured before the fix, on the owner's 982-event table with a v2 sidecar
        whose ``plot_type`` is "PCA": import returned True, the record read
        ``density (1/s)``, Details read the eleven-word instruction and nothing else, and
        a re-save then wrote ``count`` over the sender's file with 0 dialogs -- the
        round-3 defect verbatim, in the one page-state round 4's placement stepped over.
        The ORDER is the same everywhere: what the window was asked for and could not give
        comes under the header (or under the instruction) and ahead of that page's notes.

        AND THE THIRD ONE, from the window's OWN configuration rather than a file: two
        ticked sources whose ``label_overrides`` resolve to one legend name. The legend
        does not draw that literally (:meth:`_legend_names` tells them apart), and a
        figure that differs from what the user typed has to say so where the other two
        refusals are said -- see :meth:`_legend_collision_notes` for the figure this
        closes.
        """
        return self._norm_refusal_notes() + (
            [note] if (note := self._config_losses.note()) else []
        ) + self._legend_collision_notes()

    @staticmethod
    def _units_column_reading(frames: Sequence[pd.DataFrame]) -> tuple[str, ...] | None:
        """What the drawn tables' own ``units`` column says about the channel their rows
        were measured on: the distinct non-blank labels, sorted -- ``()`` when the column
        is there and blank everywhere, ``None`` when any drawn frame has no such column
        (a sentence about "this table's units column" has to be true of every table on
        the axis). Cheap: one ``unique`` per frame over a string column."""
        if not frames or any("units" not in df.columns for df in frames):
            return None
        seen: set[str] = set()
        for df in frames:
            labels = df["units"].dropna().astype(str).str.strip().unique()
            seen.update(str(v) for v in labels if v)
        return tuple(sorted(seen))

    def _density_unit_notes(self, col: str, log_x: bool,
                            frames: Sequence[pd.DataFrame] | None = None) -> list[str]:
        """What the bar heights are IN, and -- when it is not 1 -- what the DRAWN area is.

        The axis label states the unit; this states the CONSEQUENCE, which is the sentence a
        reader needs before they trust an area they measured off the screen. Measured on the
        owner's 982 events: the same bars integrate to 1 over seconds, 1000 over the ms
        ticks, 1e6 over us, and 23.3011485 over a log10 axis.

        THE LOG CASE IS A DIFFERENT STATEMENT, NOT A BIGGER FACTOR, and that is why it gets
        its own sentence rather than a number. On a log axis the area under the curve is not
        the integral of the density by any constant -- the abscissa is a logarithm -- so no
        rescaling could ever make ``area = 1`` readable there, and printing "multiply by
        23.3" would be a number true only for this dataset's range and would invite exactly
        the wrong correction.

        Placed directly under the ``y = ...`` header rather than in the notes list at the
        bottom, because it is a statement ABOUT that header and that is where the reader is
        looking when they ask what y is. The notes list is for drops and refusals.

        "UNIT UNKNOWN" IS NOT "DIMENSIONLESS" (S7-FIX3). :func:`_column_quantity` answers
        ``""`` for two different things -- a column the schema DECLARES dimensionless
        (``frac_depth``, ``snr``, ``n_levels``, spelled ``"-"`` there) and a column
        nobody has declared at all -- and this note used to collapse both into "a
        dimensionless density". Measured on the manual-measurement table (Analysis ▸
        Measure ▸ Add to list), whose ``units`` column read ``pA``: 14 of 14 of its
        ``signal``/``signal*s``/``sample`` columns (amplitude, level, baseline, mean,
        median, std, var, min, max, ptp, rms, cal_offset, area_ecd, n_samples) were told
        "bin width in the units of amplitude — a dimensionless density", i.e. the first
        half says the unit is unknown and the second asserts there is none, over heights
        that are per picoamp; the same quantity off the events table (``depth_pA``) read
        "a density in 1/pA". The dimensionless clause is now kept ONLY for a declared
        ``"-"``; an undeclared unit says so. NOT resolved from
        ``measure.MEASUREMENT_SCHEMA`` on purpose: its ``"signal"`` is a placeholder for
        whatever the channel was calibrated to, not a unit, and it declares ``var`` as
        ``signal`` where the variance of a picoamp signal is pA², so a second lookup
        would trade a vague falsehood for a precise one.

        AND "THIS TABLE DOES NOT DECLARE WHAT THAT UNIT IS" WAS THE NEXT LIE (A1-N3 /
        X-SURF-7), on the very table the finding was about: the manual-measurement
        table carries a ``units`` column -- ``measure.MEASUREMENT_SCHEMA``: "the
        channel's units label when this row was measured" -- and it read ``pA`` on
        every row while amplitude, level, baseline and area_ecd were each told the
        table declared nothing. So the note now asks THE TABLE ITSELF
        (:meth:`_units_column_reading`, ``frames``): where every drawn frame carries a
        ``units`` column that agrees on one label, the sentence says the axis cannot
        name the unit and what that column reads -- a fact about the CHANNEL the rows
        were measured on, deliberately not "a density in 1/pA", which is the schema
        trap above (``var`` is pA², ``n_samples`` is a count, ``area_ecd`` is pA·s);
        where there is no such column, or it is blank, or it disagrees with itself,
        the sentence says exactly that instead of "does not declare".
        """
        if self._norm_combo.currentText() != _NORM_DENSITY:
            return []
        unit = _column_quantity(col)[1]
        per = ((f"1/({unit})" if any(c in unit for c in "·*/ ") else f"1/{unit}")
               if unit else "")
        # The schema's own "-" (translated to "" by `_SCHEMA_UNITS`), and the one column
        # `_column_quantity` special-cases to a dimensionless reading.
        declared_dimensionless = col == "frac_depth" or _SCHEMA_UNITS.get(col) == ""
        if per:
            tail = f" — a density in {per}."
        elif declared_dimensionless:
            tail = " — a dimensionless density."
        else:
            if frames is None:
                frames = [df for _src, df in self._drawn_frames()]
            reads = self._units_column_reading(frames)
            if reads is None:
                said = "this table has no units column and does not declare what that unit is"
            elif not reads:
                said = ("this table's units column is blank, so it does not declare what "
                        "that unit is")
            elif len(reads) == 1:
                said = (f"the axis cannot name it, but the channel these rows were "
                        f"measured on reads “{reads[0]}” in this table's units column")
            else:
                said = (f"this table's units column reads more than one label here "
                        f"({', '.join(reads)}), so no single unit can be named")
            tail = f" — a density per unit of {col}; {said}."
        out = [f"  y = counts / (n × bin width), bin width in "
               f"{unit or 'the units of ' + col}" + tail]
        factor, name = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))
        if log_x:
            out.append(
                f"  x is log10 here, so the AREA UNDER THE DRAWN CURVE is not the integral "
                f"of the density at all. ∫ y dx = 1 holds over {col} in "
                f"{unit or 'its own unit'}, not over the log axis on screen.")
        elif col in _TIME_COLUMNS and factor != 1.0:
            out.append(
                f"  the x ticks read {name}; the heights are per s. Integrating the curve "
                f"over the ticks as drawn gives {factor:g}, not 1. Set Time unit = s to "
                f"read area = 1 straight off the picture.")
        return out

    def _hist_sample(self, df: pd.DataFrame, col: str, log_x: bool,
                     notes: list[str], who: str = "") -> np.ndarray | None:
        """One source's finite (and, on a log axis, positive) values for ``col``.

        BOTH DROPS ARE DISCLOSED, in the same voice, with their counts. The log-x
        exclusion below has said so since it was written; the non-finite filter above
        it did not, so a histogram of the owner's real ``rise_s`` column printed
        "apo_700: n = 1016" for a 1164-row table and nothing anywhere said the other
        148 rows had gone -- while the CLUSTERING pane of this same window, over the
        identical rows, said "1960 kept, 40 dropped (non-finite features)". Four
        surfaces of one window gave four different accounts of one drop. This method's
        own sibling :meth:`_finite` states the rule ("Drop non-finite entries, SAYING
        how many ... dropping silently would change n behind the caller's back"); the
        row TOTAL travels with the count because the shrunken n is the only number the
        pane shows and the user has nothing in the window to compare it against.
        """
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size < v.size:
            notes.append(
                f"{who}{v.size - finite.size} of {v.size} row(s) dropped: “{col}” is "
                f"not a finite number there.")
        v = finite
        if v.size == 0:
            return None
        if log_x:
            pos = v[v > 0]
            if pos.size == 0:
                return None
            if pos.size < v.size:
                notes.append(f"{who}{v.size - pos.size} non-positive value(s) dropped for log x.")
            v = pos
        return v

    @staticmethod
    def _nothing_to_draw(frames: Sequence[pd.DataFrame], col: str, log_x: bool) -> str:
        """WHY the histogram is empty, said truthfully. A refusal names its real cause.

        :meth:`_hist_sample` returns ``None`` for three different reasons -- the column
        is absent, every value is non-finite, or the log axis has no positive value to
        put on it -- and the caller printed the SECOND one for all three. On the
        owner's own -700 mV detection, ticking "log x" over ``baseline_pA`` (1164 of
        1164 values finite, all about -49 pA) replaced the plot with

            Column “baseline_pA” has no finite values.

        which is false about a column whose every value is a finite number, and sends
        the reader looking for a corrupt table instead of unticking a checkbox. The
        window already words the PARTIAL case correctly on the same code path ("300
        non-positive value(s) dropped for log x."); this is the all-or-nothing branch
        of the same sentence.
        """
        vals = [pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=np.float64)
                for d in frames if col in d.columns]
        if not vals:
            return (f"Column “{col}” is not in these events.")
        v = np.concatenate(vals) if len(vals) > 1 else vals[0]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return (f"Column “{col}” has no finite values"
                    + (f" (all {v.size} row(s) are NaN, infinite or non-numeric)."
                       if v.size else "."))
        if log_x and finite[finite > 0].size == 0:
            return (f"All {finite.size} value(s) of “{col}” are non-positive "
                    f"({float(finite.min()):g} … {float(finite.max()):g}), and a log x "
                    f"axis cannot show them. Untick “log x” to plot this column.")
        return f"Column “{col}” has nothing plottable here."

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
            pen_colour = colour if multi else self._style.data_color
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

        title = ""
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
            # ONE fit summary fits in a title; several do not — two taus joined into one
            # line runs off the plot and gets clipped. With an overlay the legend already
            # names every fit, so the title stays empty (or the user's own) and Details
            # carries the full list.
            details.extend(titles)
            title = headlines[0] if len(headlines) == 1 else ""
            ks_warn = self._ks_headline(rejected)
            # ...and, when asked for, the FULL parameter set ON the figure (owner notes
            # #13/#14): a saved image whose parameters live only in Details is not a
            # figure that can defend itself in a talk, and the thin one-line summary
            # this used to write carried the leading parameter and nothing else -- no
            # sigma, no mixture weight, no BIC, and not the KS verdict the Style
            # checkbox's own tooltip promises.
            self._draw_fit_values(full)

        # The pooled verdict FIRST in the note list: whether these rows may be put on
        # one axis at all outranks anything said about the curve drawn through them.
        verdict = self._comparability(col)
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
                            self._style.y_label.strip()
                            or self._y_axis_label(col, log_y),
                            **self._label_style())
        # THE KS VERDICT RIDES THE WARN CHANNEL, third after the two the title already
        # carries there, and for their reason: `warn` is prepended over a typed title.
        self._apply_title(title or None, accent=self._style.fit_color,
                          warn=self._warn_headline(self._stale_headline([col]),
                                                   verdict.headline, ks_warn))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(log_x, log_y)
        if notes:
            details.append("")
            details.extend(notes)
        self._set_details("\n".join(details))

    @staticmethod
    def _ks_headline(rejected: Sequence[str]) -> str:
        """The title's own mark for a drawn curve the KS test threw out, or ``""``.

        ON THE TITLE'S WARN CHANNEL, NOT IN ITS AUTOMATIC TEXT (A1-N1). S7-JRN2's
        first repair appended :data:`_KS_REJECTED_TAG` to the one-line fit summary,
        which the automatic title and the legend entry both take -- and both are
        ordinary style controls away from the figure: :meth:`_plot_title` returns the
        typed Style ▸ Title INSTEAD of the automatic one, and "Show legend" off (or the
        fit's entry unticked in the legend checklist) removes the legend entry. Measured
        on the owner's -600 mV events (dwell_s, density, log-x, exponential; KS D =
        0.3472 against 0.1059): defaults -> title "τ = 5.178 ms (n = 236) — KS REJECTED
        (1%)"; ``set_title("Figure 3: …")`` -> title "Figure 3: …"; Show legend off ->
        legend [], values box None (its default), curve still drawn; Save image -> the
        SVG's text nodes were the typed title, the two axis labels and "below detection
        limit" -- no REJECT anywhere, while Details still said DOES NOT DESCRIBE. The
        app already states the rule one method away: :meth:`_apply_title`'s ``warn`` is
        "PREPENDED to whatever the title would otherwise be, the user's own text
        included", because the title is the only thing Save image keeps. So the
        rejection is a third headline on that channel, beside the superseded-calibration
        mark and the comparability verdict, and the automatic title carries the PLAIN
        summary (:attr:`_FitSummary.plain`) so the verdict is not said twice on one
        line. The legend entry and Details keep the tag on the summary itself, per fit
        -- on an overlay that is the surface that says WHICH curve was thrown out, and
        the headline names those sources by the figure's own name for them
        (:meth:`_legend_name`) for the same reason.

        ``rejected`` is one entry per rejected fit: the figure's name for its source on
        an overlay, ``""`` for the single fit of a one-source (or cluster-split) plot.
        """
        if not rejected:
            return ""
        named = [name for name in rejected if name]
        return f"⚠ {_KS_REJECTED_TAG}" + (f": {', '.join(named)}" if named else "")

    def _cluster_groups(self, df: pd.DataFrame, col: str, log_x: bool,
                        notes: list[str], src: PlotSource):
        """One ``(label, colour, values, source, norm_n)`` per cluster of the source.

        The missing half of the clustering workflow: ``run_clustering`` could find
        two populations and the histogram had no way to show their distributions
        apart. Labels are reindexed onto the frame actually on screen, so the
        accepted-only filter and a clustering computed before it still agree; rows
        the clustering dropped (non-finite features) land in the grey ``-1`` group
        rather than vanishing without a word.

        Every cluster normalizes against the SAME total (all clusters together),
        unlike two independent sources: these are PARTS OF ONE distribution, so
        their densities must add up to the whole rather than each be rescaled to
        area 1 — which would erase exactly the "one population is rare" finding the
        clustering just made.
        """
        res = self._cluster_result
        if res is None or col not in df.columns:
            return []
        labels = res.labels.reindex(df.index, fill_value=populations.DROPPED_LABEL)
        codes = labels.to_numpy(dtype=int)
        total = self._hist_sample(df, col, log_x, [])
        norm_n = int(total.size) if total is not None else int(len(df))
        out = []
        for cid in sorted({int(c) for c in codes}, key=lambda c: (c < 0, c)):
            sub = df[codes == cid]
            v = self._hist_sample(sub, col, log_x, notes, f"cluster {cid}: ")
            if v is None:
                continue
            name = "dropped (non-finite features)" if cid < 0 else f"cluster {cid}"
            out.append((name, _cluster_color(cid).name(), v, src, norm_n))
        return out

    def _comparability(self, *columns: str) -> SourceVerdict:
        """MAY these sources be pooled for these columns? The ONE verdict, per metric.

        :func:`nanoscope.core.sources.source_comparability` — the same baseline /
        merge rule ``Analysis ▸ Voltage dependence`` enforces, plus the gain rule
        this window already had, reconciled into one answer.

        Two things this replaced. The window had NO baseline check at all: overlay
        ``frac_depth`` from two recordings the probe had proposed different windows
        for and Details listed the two sources, their n, and nothing else — although
        dI/I0 is ``depth_pA / |baseline_pA|`` and moves with the window (measured on
        the owner's ladder: 11.3 % on the −700 mV recording). And the gain guard,
        firing on ``depth_pA``, actively redirected the user onto ``frac_depth`` —
        onto the unguarded column. One verdict, built by one function that knows
        about both guards, is what stops the advice contradicting the warning.

        Judged over the DRAWN sources (ticked and non-empty), not merely the ticked
        ones, so the verdict describes what is on screen.
        """
        # A single source cannot disagree with anything, and the rule says so
        # itself — no early return here, so there is one code path and one verdict.
        # (A source's OWN detection warnings show regardless: _detection_notes.)
        return source_comparability([src for src, _df in self._drawn_frames()], columns)

    def _detection_notes(self, sources=None) -> list[str]:
        """Every DRAWN detection's own caveats — ``AnalysisResult.warnings``, whole.

        The pane already surfaces a fit's ``meta["warnings"]`` and a clustering's
        ``ClusterResult.warnings``; the detection's were the one channel with no
        outlet here, so a table measured at the unprobed default baseline plotted
        with no caveat while the Summary panel said so about the same numbers.

        Prefixed with the source label because an overlay draws several, and an
        un-attributed caveat on a two-source plot is a caveat about nothing in
        particular. Rendered unfiltered, exactly as
        :meth:`nanoscope.gui.panels.StatsPanel.set_warnings` does: this must not
        become a place that decides which of a detector's advisories a plot inherits.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        srcs = list(sources)
        multi = len(srcs) > 1
        out: list[str] = []
        for src in srcs:
            for w in getattr(src, "warnings", ()) or ():
                text = str(w).strip()
                if text:
                    out.append(f"{src.label}: {text}" if multi else text)
        return out

    # -- what the WINDOW must say about the COLUMNS it drew ----------------------------------------
    #
    # `_detection_notes` above carries a DETECTION's own caveats, whole, and must stay
    # that way. These two carry the facts that belong to a (source, column) pair
    # instead -- whether the column is still measured the way a fresh run would measure
    # it, and which clock a column of instants is on -- and they are collected by ONE
    # method that every draw path calls, because "the plot says less about these
    # numbers than the table they came from" is the defect, not the surface it was
    # found on.

    def _superseded_columns(self, src: PlotSource | None) -> tuple[str, ...]:
        """Columns of ``src`` measured under a calibration that has since been replaced.

        The SAME test the event table's badge is driven by
        (``AnalysisPanel._check_staleness``): the detection's stored
        ``calibration_fingerprint`` against the channel's live one. Same columns too --
        :data:`nanoscope.analysis.remeasure.REMEASURED_COLUMNS`, ``baseline_pA``
        included, because it is the column the other six are computed FROM.

        The app's doctrine for a superseded measurement is "never silently re-measure":
        the numbers stay and every surface that shows them says so (USER_GUIDE §9a --
        such a project stays "marked, badged, and otherwise fully readable, plottable
        and exportable"). *Plottable* is where the mark stopped: two panels away from a
        ⚠ header and amber cells, this window drew the same ``depth_pA`` and printed its
        fitted mean in the title, the legend and Details with nothing to say the
        calibration behind it had been superseded.

        Empty for a source with no document behind it (the manual table), for one whose
        detection is gone, and whenever the stamps match.

        NOT the whole badge: the event table also badges these columns when the
        BASELINE estimator has moved, and that half of the test lives in
        ``gui.analysis_panel._baseline_staleness`` where this window cannot reach it.
        Lifting both halves onto :class:`~nanoscope.core.sources.PlotSource` (where
        ``calibration_fingerprint`` already rides) is the real home for this rule.
        """
        if src is None or src.doc_id is None or src.channel is None or not src.module_id:
            return ()
        try:
            res = self._session.results.get(src.doc_id, int(src.channel), src.module_id)
            live = self._session.calibration(src.doc_id, int(src.channel)).fingerprint()
        except Exception:       # pragma: no cover - a caveat must never break a plot
            return ()
        stamped = getattr(res, "calibration_fingerprint", None) if res is not None else None
        if stamped is None or live is None:
            return ()
        try:
            same = (tuple(float(x) for x in stamped) == tuple(float(x) for x in live))
        except (TypeError, ValueError):     # pragma: no cover - malformed stamp
            return ()
        return () if same else tuple(REMEASURED_COLUMNS)

    def _clock_offset(self, src: PlotSource | None) -> float | None:
        """``t0_s`` of the recording behind ``src``, or None when there is no clock."""
        if src is None or src.doc_id is None:
            return None
        try:
            doc = self._session.document(src.doc_id)
        except Exception:       # pragma: no cover - a closed document
            return None
        return clock.clock_offset_s(getattr(doc, "meta", None))

    def _column_notes(self, columns: Sequence[str], sources=None) -> list[str]:
        """Everything this window owes the reader about the COLUMNS it just drew.

        One collector, called by the histogram, the scatter and the PCA alike, so a
        caveat cannot be true in one view of the same numbers and absent from another.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        srcs = [s for s in sources if s is not None]
        cols = [c for c in dict.fromkeys(str(c) for c in columns if c)]
        out: list[str] = []
        multi = len(srcs) > 1

        # 1. SUPERSEDED CALIBRATION -- per source, naming the drawn columns it touches.
        for src in srcs:
            hit = [c for c in cols if c in self._superseded_columns(src)]
            if not hit:
                continue
            who = f"{src.label}: " if multi else ""
            out.append(
                f"{who}SUPERSEDED CALIBRATION: {', '.join(hit)} on this plot "
                f"{'was' if len(hit) == 1 else 'were'} measured under a gain/offset that "
                f"has since been changed, and {'it has' if len(hit) == 1 else 'they have'} "
                f"NOT been recomputed — the event table badges the same column(s) ⚠. The "
                f"numbers drawn, and every fitted parameter printed from them, are in the "
                f"OLD calibration. Re-run detection on this recording before reporting "
                f"them.")

        # 2. WHICH CLOCK a column of instants is on. `t_start_s` is seconds since THIS
        #    recording's first loaded sample, so pooling it across recordings puts two
        #    different origins on one axis -- and a segment's rows are rebased onto the
        #    segment's own start, which is how ONE physical event lands at two different
        #    x positions here. Every other surface that shows the column says which
        #    clock it is on (the event table header, the measurement table, the PSD
        #    segment spin box); this is the only one that POOLS it, and it said nothing.
        #    core/clock.py's rule: "One clock, or every control that shows or takes a
        #    time says which clock it is in."
        origin_cols = [c for c in cols if c in _ORIGIN_COLUMNS]
        if origin_cols:
            named = ", ".join(origin_cols)
            if multi:
                where = []
                for src in srcs:
                    off = self._clock_offset(src)
                    where.append(f"{src.label} (t₀ = {clock.format_t0(off)} s"
                                 + (", a segment, rebased onto its own start"
                                    if src.is_segment else "") + ")")
                out.append(
                    f"CLOCK: {named} is measured from EACH recording's own first loaded "
                    f"sample, not from a shared origin, and these {len(srcs)} sources are "
                    "pooled on one axis: " + "; ".join(where) + ". Two points at the same "
                    "x are not simultaneous; add each recording's t₀ to put them on the "
                    "instrument clock.")
            elif srcs:
                off = self._clock_offset(srcs[0])
                if off is None:
                    # No open recording behind this source (a pushed frame, the manual
                    # table, a closed document). `clock.field_note` answers None with
                    # "this recording declares no sample rate", which would be a
                    # different -- and here false -- reason for the same silence.
                    out.append(
                        f"CLOCK: {named} is measured from the recording's OWN first "
                        f"loaded sample. No open recording stands behind these rows in "
                        f"this window, so their t₀ on the instrument clock cannot be "
                        f"stated here.")
                else:
                    # `clock.field_note` is a whole sentence and the app's ONE wording
                    # for this fact -- restating it here is how the surfaces drifted
                    # apart in the first place -- so it is joined, not paraphrased.
                    out.append(f"CLOCK: {named} — "
                               + clock.field_note(clock.RECORDING, off))
        return out

    def _stale_headline(self, columns: Sequence[str], sources=None) -> str:
        """The title's own mark for a drawn column under a superseded calibration.

        In the title because the title is the only thing ``Save image`` keeps -- the
        same reason the comparability verdict goes there (see :meth:`_apply_title`).
        An exported figure of a superseded depth must not look like an exported figure
        of a current one.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        cols = {str(c) for c in columns if c}
        for src in sources:
            if cols & set(self._superseded_columns(src)):
                return "⚠ SUPERSEDED CALIBRATION"
        return ""

    @staticmethod
    def _warn_headline(*parts: str) -> str | None:
        """Join the title's warning marks, worst-first, or None when there are none."""
        kept = [p for p in parts if p]
        return "   ·   ".join(kept) if kept else None

    @staticmethod
    def _finite(values: np.ndarray, notes: list[str]) -> np.ndarray:
        """Drop non-finite entries, SAYING how many. ``stats`` refuses to fit a
        column with a NaN in it (dropping silently would change n behind the
        caller's back) -- so the window drops them out loud instead of showing
        "Fit unavailable" for one bad row."""
        good = np.isfinite(values)
        if good.all():
            return values
        notes.append(f"{int((~good).sum())} non-finite value(s) excluded from the fit.")
        return values[good]

    @staticmethod
    def _t_max_source(inp, hi) -> str | None:
        """Which bound the fitter's right truncation ACTUALLY is, or None.

        The mirror of the ``t_min_source`` half of :meth:`_as_fitted`, for the bound
        that had no such half. ``_restrict`` intersects what the detector could report
        with what the user typed, so the surviving ceiling is one or the other -- and
        the fitter's MODELLED-truncation note used to assert the detector's story
        whichever it was. ``modules.dwell_fit_inputs`` puts nothing but
        ``provenance['max_dwell_s']`` into ``inp.t_max`` (see ``_right_truncation``:
        a curated-away tail is warned about, never modelled), so "equal to inp.t_max"
        IS "the detector's", with no third possibility to confuse it with.
        """
        if hi is None:
            return None
        prov_hi = None if inp is None else getattr(inp, "t_max", None)
        if prov_hi is not None and float(hi) == float(prov_hi):
            return stats.T_MAX_FROM_DETECTOR
        return stats.T_MAX_FROM_FIT_RANGE

    @staticmethod
    def _as_fitted(inp, lo, hi):
        """``inp`` restated over the window the FITTER was actually given.

        :func:`nanoscope.analysis.modules.stamp_fit_provenance` writes the
        resolution's ``t_min`` / ``t_min_source`` onto the FitResult
        unconditionally -- correct for the Fits panel, whose resolution IS the fitted
        window, and wrong here the moment the Fit range boxes narrow it. Rather than
        unpick the stamp afterwards (a second copy of the provenance rules, in the
        window that reads them), the resolution handed to it is the one that
        describes this fit.

        A COPY, never a mutation: the caller still reads ``inp.notes`` and
        ``inp.n_input`` after this, and the resolution may be shared with another
        draw. A depth resolution carries no ``t_min`` and comes back untouched, as
        does a window that changed nothing.
        """
        if not isinstance(inp, modules.DwellFitInputs):
            return inp
        new_lo = float(inp.t_min) if lo is None else float(lo)
        new_hi = inp.t_max if hi is None else float(hi)
        if new_lo == float(inp.t_min) and new_hi == inp.t_max:
            return inp
        # The floor is only still the detector's if the detector's is what survived
        # the intersection; otherwise it is the user's typed bound, and saying
        # "from min_dwell_s" about it would be a false provenance.
        source = inp.t_min_source if new_lo == float(inp.t_min) else "explicit"
        return _dc_replace(inp, t_min=new_lo, t_min_source=source, t_max=new_hi)

    def _overlay_fit(
        self, kind: str, col: str, v: np.ndarray, edges: np.ndarray, log_x: bool,
        src: PlotSource | None = None, colour: str | None = None, who: str = "",
    ) -> tuple[_FitSummary, list[str]]:
        """Fit ONE source and draw its curve; return (summary, notes).

        TWO SUMMARIES IN ONE RETURN VALUE (see :class:`_FitSummary`), because they have
        two jobs. The summary itself is the one-liner the plot TITLE and the legend
        entry take -- the headline parameter and n, sized to sit on one line beside a
        comparability verdict. Its ``.full`` is what the on-plot values box takes when
        the Style checkbox asks for it (owner note #13, "once the Fit has been applied,
        include the numerical values on the plot"): EVERY fitted parameter with its own
        unit, n, BIC and the KS verdict. The box is what a saved PNG carries INSTEAD of
        Details, so it has to answer the questions Details would -- the one-liner did
        not, and the checkbox's tooltip was already promising the KS verdict it never
        printed.

        The curve is evaluated on a fine grid and scaled to the drawn y quantity by
        the LOCAL linear bin width at each point -- for log bins that is
        ``x·ln10·Δlog10``, the Jacobian without which a density laid over a log
        histogram is wrong by a factor that spans decades (see
        :mod:`nanoscope.plot.figures`) -- and then by the normalization factor, so a
        density curve overlays a density histogram.

        The SAMPLE and the censoring model come from
        :mod:`nanoscope.analysis.modules` **for this source**, whenever the column is
        one that source's detector provenance describes. That is what keeps this fit
        equal to the Fits panel's on the same events, and what stops an overlay from
        applying one recording's ``t_min`` to another recording's dwell times: each
        source resolves its own. Every substitution that had to be made -- a guessed
        ``t_min``, excluded filter-limited events, an unmodelled right truncation, a
        floor-biased depth component -- comes back in ``notes``.

        ONE LABEL PER NOTE, AND THE CALLER OWNS IT. The returned notes are
        UNATTRIBUTED -- no source label, no separator -- because the pane that
        collects them is the one surface that knows whether more than one series is
        drawn, and it already writes ``f"{label}: {note}"`` for every other note it
        shows (``_hist_sample``'s exclusions, the per-source detection limits,
        ``_detection_notes``). When this method ALSO prefixed the notes it built
        itself, both halves fired and every line of the comparison pane read
        "poreA.abf: poreA.abfBIC = -678.661" -- the label twice, the second copy
        welded on because the copy made here carried no separator. Attribution
        happens once, at the caller's seam, which is also the only way the notes
        this method merely FORWARDS (``_finite``'s exclusions, the fitter's own
        ``meta['warnings']``) get labelled at all.

        ``who`` is therefore not a note prefix: it is the SERIES tag, used for the
        legend entry and to pick the overlay pen, and empty means "single series".
        """
        if self._workspace_snapshot_fits is not None:
            return self._overlay_workspace_fit(
                kind, col, v, edges, log_x, src=src, colour=colour, who=who)
        notes: list[str] = []
        k = int(self._applied_fit_k)   # the APPLIED component count, not the spin's
        src = src if src is not None else self.primary_source()
        # THE FIT RANGE, as a TRUNCATION. Restricting the fit to a peak is exactly the
        # situation the detector's own min_dwell_s already creates -- a sample you can only
        # see part of -- and the fix is the same: tell the likelihood, so it renormalises to
        # the observable window instead of pretending the missing mass was never there.
        # Subsetting the array and saying nothing would bias every estimate, silently.
        fit_lo, fit_hi = self._fit_range(col)
        # The exact array each branch handed the fitter, plus the censoring it was told
        # about: the goodness-of-fit test and any rival family MUST see the same sample
        # and the same truncation, or their verdicts are about different data.
        sample_for_gof: np.ndarray | None = None
        lo_for_gof = hi_for_gof = None
        # WHETHER THE WINDOW ABOVE IS A TRUNCATION OR ONLY A CROP. Every dwell family
        # is HANDED (lo, hi) and renormalises its likelihood to it; the Gaussian
        # mixture is not -- it is fitted to the values that survive the window, with
        # no conditioning. The FIT RANGE note says which, because "TRUNCATED to this
        # window, not merely cropped" printed over a crop is the same sentence read
        # backwards.
        trunc_modelled = True
        #: The window the fit was ACTUALLY conditioned on (or cropped to), which is the
        #: INTERSECTION `_restrict` computes -- never the raw typed pair. See the FIT
        #: RANGE note below.
        trunc_lo = trunc_hi = None
        # THE INTERVAL IS NOT THE ESTIMATE, and they do not share a latency budget.
        # A branch whose fitter can hand its bootstrap back fills these in; see the
        # resolution below `stamp_fit_provenance`.
        ci_jobs: list = []
        ci_key: tuple | None = None
        #: What the CI line says for the families whose whole fit is re-run on the
        #: worker (see :meth:`_point_fit_now`): "" once the interval has landed,
        #: otherwise the pending or failed sentence. It shares the single `ci_state`
        #: slot below with the `deferred_ci` families, because one pane says one thing
        #: about one fit's interval however that interval was obtained.
        fit_ci_state = ""
        #: ``kk -> cache key`` for the family this draw is fitting, set by the branch
        #: that knows the sample and the censoring. Auto (BIC) re-fits at a DIFFERENT
        #: k and must not collide with the key of the fit it replaces, so the key is a
        #: function of k rather than a constant.
        fit_key_of = None
        #: How the drawn x grid maps onto the variable the fit is OF. ``None`` means
        #: "they are the same variable", which is every family but a Gaussian mixture
        #: over an all-negative column -- there the fit is of |x| and the curve belongs
        #: at the reflection of the grid. See that branch.
        grid_of = None
        #: The name of the variable the goodness-of-fit verdict is about, when that is
        #: NOT the plotted column. Printed with the verdict, because "the fitted curve
        #: is consistent with these data" over a fit of |x| is a sentence about
        #: something the axis is not showing.
        gof_frame = ""

        def _point_fit_now(make, key, *, quiet: bool = False):
            """The point fit on the click; the SAME fit, bootstrap and all, on a worker.

            THE INVARIANT (stats.py's own, ~line 185): a fit has two parts with two
            latency budgets -- the maximisation, which belongs on the interaction path,
            and the resampling, which does not. ``gamma``/``inverse-gaussian`` split
            themselves through :func:`stats.deferred_ci`; ``exponential`` and
            ``gaussian-mixture`` cannot, because ``fit_dwell_exponential`` and
            ``fit_depth_gaussians`` do not read ``_CI_DEFER`` -- so their whole
            resampling loop ran here, on the click. Measured through this window:
            setting Fit components = 2 blocked the GUI thread for 1.9-2.6 s on the
            owner's own -700 mV apo run (n = 1164) and 40-45 s at n = 100000, against a
            150 ms budget, with a 50 ms timer starved throughout -- and the pane's own
            "set Fit components = 2 to see it" note is what tells the user to click it.
            Worse, it recurs: with k = 2 on screen every subsequent curation click paid
            it again.

            So the split is made HERE instead, at the one seam both families pass
            through. ``point_fits_only()`` switches off exactly the resampling loops
            (``_bootstrap_dwell`` returns None; ``fit_depth_gaussians`` iterates zero
            times) and changes nothing before them, so the params, loglik, AIC, BIC and
            the drawn curve are bit-identical to the synchronous call. The full call is
            then re-run on :class:`_NoteWorker` and REPLACES the point fit wholesale --
            not merged in -- because a bootstrap is not only an interval here: the
            mixture's UNIDENTIFIED COMPONENT(S) / UNRESOLVED WEIGHT(S) warnings and the
            "bootstrap failed to produce enough replicates" note are statements ABOUT
            the CIs, and installing the numbers without them would print a k=3 mixture
            with error bars and no word that two of its components are unresolved.
            Re-running the whole fit is what makes the delivered result byte-identical
            to what this window prints today: the fitters are deterministic and seed
            their own rng, so the same call gives the same answer, and the point fit it
            recomputes on the way is milliseconds.

            While it is out, ``meta['ci_pending']`` is stamped -- the flag stats.py
            defines as "an interval is being computed and will replace the empty ci" --
            so no surface reading the FitResult (a saved project, ``fit_result()``)
            mistakes "not yet" for "this family offers none". It is discharged either
            by the full fit landing (which carries no such flag) or by
            :func:`stats.ci_failed` below.
            """
            nonlocal fit_ci_state
            ready, payload = self._deferred_note(key, self._full_fit_job(make, quiet))
            if ready and isinstance(payload, stats.FitResult):
                # A COPY: the cache outlives this draw and `stamp_fit_provenance`
                # below MUTATES what it is handed (it prepends to meta['warnings']
                # and is not idempotent), so the cached fit would grow a duplicate
                # warning on every redraw over the same sample. Same reasoning, same
                # line, as the exponential + gamma branch.
                fit_ci_state = ""
                return copy.deepcopy(payload)
            if quiet:
                with stats.quiet_runtime_warnings(), stats.point_fits_only():
                    out = make()
            else:
                with stats.point_fits_only():
                    out = make()
            if ready:
                # `_NoteWorker` filed an exception; the panel caches that as "". The
                # point estimates stand on the likelihood -- there is simply no
                # interval beside them, and the promise has to be settled on the fit
                # as well as in the note.
                stats.ci_failed(out, "the deferred bootstrap raised")
                fit_ci_state = self._CI_FAILED_TEXT
            else:
                out.meta["ci_pending"] = True
                fit_ci_state = self._CI_PENDING_TEXT
            return out

        def _restrict(sample, lo, hi):
            """Apply the user's window to (sample, t_min, t_max). Returns the tightened
            triple: the truncation is the INTERSECTION of what the detector could report
            and what the user asked to fit."""
            sample = np.asarray(sample, dtype=np.float64)
            if fit_lo is not None:
                sample = sample[sample >= fit_lo]
                lo = fit_lo if lo is None else max(float(lo), fit_lo)
            if fit_hi is not None:
                sample = sample[sample <= fit_hi]
                hi = fit_hi if hi is None else min(float(hi), fit_hi)
            return sample, lo, hi

        def _no_provenance_note() -> str:
            """The no-detection-limit sentence, told the truth about the fit range.

            One pane, one truth: with a Fit range typed, `_restrict` above hands the
            user's window to the fitter as a real truncation and the FIT RANGE line
            below says "the likelihood is TRUNCATED to this window" -- so the old
            unconditional "the fit is untruncated" printed a flat contradiction two
            lines away, exactly when the user had exercised the truncation feature.
            The caveat also narrows honestly: a detector floor BELOW a typed lower
            bound is absorbed by that bound's own truncation correction; only a
            floor inside the fitted window can still bias the estimates."""
            if fit_lo is None and fit_hi is None:
                return (
                    f"no detection limit is known for “{col}”; the fit is untruncated, so a "
                    f"floor the detector really has would bias these estimates.")
            return (
                f"no detection limit is known for “{col}”; the likelihood is truncated "
                f"only to the typed fit range, not to a measured detector limit, so a "
                f"detector floor inside the fitted window would still bias these "
                f"estimates.")

        # WHAT A DWELL-FAMILY FITTER MUST BE TOLD WHEN THE COLUMN IS NOT A TIME.
        # The four dwell families below accept any positive column, and their default
        # labelling is the dwell contract -- tau/theta/mu/lambda stamped "s" -- so a
        # fit to depth_pA captioned "τ = 81.54 s" for a number that is PICOAMPS, two
        # lines under this pane correctly saying no detection limit is known for that
        # column. Same mechanism as the depth mixture's `quantity=`/`unit=` (High-4):
        # the truth is stamped ON the FitResult, so the estimate line, `_ci_note` and
        # any later reader of `unit_of` all get it from one place.
        #
        # THE DEFAULT IS THE *DWELL* CONTRACT, NOT THE SECONDS CONTRACT. The gate used
        # to be `unit == "s"`, which handed every seconds-valued column to the fitters
        # as if it were a dwell -- and `rise_s`, `fall_s` and `t_start_s` are seconds
        # without being dwells. Their fits then spoke in a vocabulary that describes a
        # different column: "dwell_s has only 5 events; refusing to fit fewer than 20"
        # three lines under a plot titled "Histogram of rise_s"; "provenance has no
        # 'min_dwell_s'" on a recording whose provenance declares one; the shortest
        # observed FALL time called "the shortest observed dwell"; and a user-typed Fit
        # range max blamed on "detector max_dwell_s" while the provenance's real
        # max_dwell_s was 100x larger. The window already knows which columns the dwell
        # provenance describes -- :data:`_DWELL_COLUMNS`, the same list `_dwell_inputs`
        # refuses on -- so the two cannot drift apart. Only those keep the historical
        # dwell labels, bit-for-bit; every other column is fitted under its own name.
        _q, _u = _column_quantity(col)
        dwell_kw = ({} if (_u == "s" and col in _DWELL_COLUMNS)
                    else {"quantity": _q, "unit": _u})
        if kind == _FIT_EXP_GAMMA:
            inp = self._dwell_inputs(col, src)
            sample = self._finite(inp.dwell, notes) if inp is not None else v[v > 0]
            lo = inp.t_min if inp is not None else None
            hi = inp.t_max if inp is not None else None
            sample, lo, hi = _restrict(sample, lo, hi)
            if inp is None:
                notes.append(_no_provenance_note())
            from ..analysis.stats import fit_dwell_exp_gamma
            # THE POINT FIT ITSELF IS OFF THE CLICK PATH FOR THIS FAMILY, AND ONLY THIS
            # FAMILY.
            #
            # Everywhere else in this method the maximisation is the cheap half and the
            # bootstrap is the expensive one, so `deferred_ci` splits them there. This
            # family has no bootstrap at all -- and its maximisation is twelve
            # Nelder-Mead starts, which is the whole cost. Measured on the owner's own
            # merged dwell tables, point fits only, the five families a curation click
            # can have on screen:
            #
            #     -700 mV (n = 986)        -600 mV (n = 240)
            #     exponential        11 ms                11 ms
            #     gamma (peaked)      4                    6
            #     inverse-gaussian    9                   12
            #     gaussian mixture    <1                  <1
            #     exponential+gamma 207 ms             9479 ms   <- this branch
            #
            # Budget is 150 ms per click. Every other family is an order of magnitude
            # inside it; this one is over on the owner's LARGE recording and 63x over on
            # his small one, where the likelihood has a shallow ridge and Nelder-Mead
            # grinds to its 20000-iteration cap. There is no version of this that fits in
            # the budget on the GUI thread: the twelve starts ARE the number, and cutting
            # or reordering them would move the BIC this window prints.
            #
            # So it goes where the other work that cannot be interrupted goes -- onto
            # `_NoteWorker`, through the panel's own deferral -- and the plot SAYS it is
            # waiting rather than freezing for nine seconds and then showing a curve.
            # The fit is not changed by moving it: same sample, same censoring, same
            # deterministic simplex, bit-identical params/BIC/AIC/KS.
            #
            # A FAILURE IS RETURNED, NOT RAISED, by the job: `_NoteWorker` files any
            # exception as `_NOTE_FAILED`, which this panel caches as `""` for a non-
            # rivals key -- and `""` cannot say WHY. The fitter's own refusals are the
            # useful ones (fewer than 20 positive dwells; events shorter than the stated
            # t_min), and they are what the synchronous call used to print.
            fit_key = ("expgammafit", self._sample_digest(sample),
                       None if lo is None else float(lo),
                       None if hi is None else float(hi),
                       # The labels are part of what was computed: the same bytes
                       # fitted as a different column must not replay a cached fit
                       # wearing the other column's units.
                       dwell_kw.get("quantity"))
            ready, payload = self._deferred_note(
                fit_key, self._exp_gamma_job(sample, lo, hi, **dwell_kw))
            if not ready:
                notes.append(self._FIT_PENDING_TEXT)
                return self._FIT_PENDING_SUMMARY, notes
            if isinstance(payload, BaseException):
                raise RuntimeError(str(payload))
            if not isinstance(payload, stats.FitResult):    # cached `_NOTE_FAILED`
                raise RuntimeError(
                    "the exponential + gamma maximisation did not complete on these "
                    "events. Choose another model, or change the fit range.")
            # A COPY, because the cache outlives the draw and this method MUTATES what it
            # is handed: `modules.stamp_fit_provenance` PREPENDS the resolution's notes to
            # `meta['warnings']` and is not idempotent, so a second redraw over the same
            # sample -- a bin change, a time-unit toggle, a resize -- would print every
            # provenance warning twice, then three times. Found in this fix's own verify
            # pass; the synchronous call could not have this defect because it built a
            # fresh FitResult every draw.
            fit = copy.deepcopy(payload)
            _refit = lambda kk, _s=sample, _lo=lo, _hi=hi, _kw=dict(dwell_kw): (   # noqa: E731
                fit_dwell_exp_gamma(_s, t_min=_lo, t_max=_hi, **_kw))
            names = ("k", "k")
            sample_for_gof, lo_for_gof, hi_for_gof = sample, lo, hi
        elif kind in (_FIT_GAMMA, _FIT_INVGAUSS):
            model = "gamma" if kind == _FIT_GAMMA else "inverse-gaussian"
            inp = self._dwell_inputs(col, src)
            sample = self._finite(inp.dwell, notes) if inp is not None else v[v > 0]
            lo = inp.t_min if inp is not None else None
            hi = inp.t_max if inp is not None else None
            sample, lo, hi = _restrict(sample, lo, hi)
            if inp is None:
                notes.append(_no_provenance_note())
            from ..analysis.stats import fit_dwell_peaked
            # THE POINT FIT ON THE CLICK; THE INTERVAL ON A WORKER.
            #
            # These are the two families report finding #10 gave CIs to, and giving
            # them CIs is what re-created finding #3: a REPORTED interval cannot hide
            # inside `stats.point_fits_only()` the way a rival's discarded one can, so
            # 200 Nelder-Mead replicates landed on the curation click path. Measured
            # through a real MainWindow on the owner's -700 mV apo run (n = 984),
            # median of 12 real checkbox clicks: 507 ms with `gamma (peaked)` on
            # screen, 1101 ms with `inverse-gaussian (first passage)`, against 44 ms
            # with the fit box set to none. Budget 150 ms.
            #
            # `deferred_ci` splits the call where the cost actually splits: the
            # maximisation (3.9 ms / 14.7 ms on that sample -- params, loglik, AIC,
            # BIC and the drawn curve) happens here, and the bootstrap (975 ms /
            # 1634 ms) comes back as a callable that :meth:`_deferred_note` runs on
            # `_NoteWorker`. Same seed, same simplex, same sample, same order, so the
            # interval it produces is bit-identical to the synchronous one -- this
            # moves WHEN it is computed, never WHAT.
            with stats.deferred_ci() as ci_jobs:
                fit = fit_dwell_peaked(sample, model=model, t_min=lo, t_max=hi,
                                       **dwell_kw)
            ci_key = ("ci", model, self._sample_digest(sample),
                      fit.meta.get("t_min"), fit.meta.get("t_max"))
            _refit = lambda kk, _s=sample, _m=model, _lo=lo, _hi=hi, \
                    _kw=dict(dwell_kw): (   # noqa: E731
                fit_dwell_peaked(_s, model=_m, t_min=_lo, t_max=_hi, **_kw))
            names = (("k", "k") if model == "gamma" else ("mu", "μ"))
            sample_for_gof, lo_for_gof, hi_for_gof = sample, lo, hi
        elif kind == _FIT_EXP:
            inp = self._dwell_inputs(col, src)
            # ...and, for THIS family only, the display pair as well: the exponential
            # fitter words its degeneracy note around the tau values themselves, and
            # the scale has to travel WITH the numbers (see fit_depth_gaussians) or
            # the note prints raw storage units under an axis reading another --
            # "tau_1 = 0.01981 s" two lines below "τ = 19.81 ms".
            _exp_kw = dict(dwell_kw, display=self._display_unit(col, _u))
            if inp is not None:
                _sample, _lo, _hi = _restrict(
                    self._finite(inp.dwell, notes), inp.t_min, inp.t_max)
                # ...and WHO the surviving ceiling belongs to, for the same reason the
                # floor's `t_min_source` travels: the fitter words the MODELLED note
                # and cannot otherwise tell a detector's max_dwell_s from a bound the
                # user typed into the Fit range box.
                _exp_kw = dict(_exp_kw, t_max_source=self._t_max_source(inp, _hi))
                _refit = lambda kk, _s=_sample, _a=_lo, _b=_hi, _kw=dict(_exp_kw): (
                    populations.fit_exponential(_s, t_min=_a, t_max=_b, n_components=kk,
                                                **_kw))
                # The 1000-replicate bootstrap is not on the click path. See
                # `_point_fit_now`.
                fit_key_of = lambda kk, _d=self._sample_digest(_sample), _a=_lo, \
                        _b=_hi, _kw=_exp_kw: (
                    ("fullfit", _FIT_EXP, _d, None if _a is None else float(_a),
                     None if _b is None else float(_b), int(kk),
                     _kw.get("quantity"), _kw.get("t_max_source")))
                fit = _point_fit_now(lambda _kk=k: _refit(_kk), fit_key_of(k))
                sample_for_gof, lo_for_gof, hi_for_gof = _sample, _lo, _hi
            else:
                data = v[v > 0]
                if data.size < v.size:
                    notes.append(
                        f"{v.size - data.size} non-positive value(s) excluded from the fit.")
                # THE NOTE DESCRIBES THE FLOOR THE LIKELIHOOD ACTUALLY USED. With a
                # typed Fit range min, `_restrict` below hands that bound to the
                # fitter as an explicit t_min (meta['t_min_source'] == 'explicit') and
                # the shortest-observed-value substitution never runs -- so claiming
                # it did, and warning about its bias direction, was false exactly when
                # the user had typed a floor. Only the fallback branch may claim the
                # fallback.
                if fit_lo is not None:
                    _f, _un = self._display_unit(col, _u)
                    notes.append(
                        f"no detection limit is known for “{col}”: the typed Fit range "
                        f"min ({fit_lo * _f:g}{' ' + _un if _un else ''}) is the floor "
                        f"t_min the likelihood is conditioned on — nothing was "
                        f"substituted from the data. If the detector's real floor sits "
                        f"above that bound, events between them are missing from the "
                        f"sample and the estimates are still biased.")
                else:
                    # THE SUBSTITUTED NUMBER CARRIES ITS UNIT, like the typed one three
                    # lines above and like every other number this pane prints. It was
                    # the one bare value left in the note family: "the shortest observed
                    # value (5.56997e-05)" under an axis reading ms is the same
                    # 1000x-in-waiting the estimate line and the FIT RANGE line already
                    # go through `_display_unit` to avoid -- and on a picoamp column it
                    # was a bare picoamp count with nothing to say so.
                    _f, _un = self._display_unit(col, _u)
                    notes.append(
                        f"no detection limit is known for “{col}”: the fit falls back to the "
                        f"shortest observed value ({float(data.min()) * _f:g}"
                        f"{' ' + _un if _un else ''}) as t_min. The real floor "
                        f"can only be LOWER than that, so this substitute over-subtracts and biases "
                        f"tau SHORT by the difference. Run detection (or histogram dwell_s) to get "
                        f"the detector's real min_dwell_s.")
                # The note above already says it. Through `stats.quiet_runtime_warnings`
                # rather than `warnings.catch_warnings` directly because this runs on
                # the GUI thread WHILE the advisory worker may be inside its own
                # suppression: the filter list is process-global and the two save/restore
                # pairs interleave. This is the GUI side of that single-writer rule.
                with stats.quiet_runtime_warnings():
                    data, _lo2, _hi2 = _restrict(data, None, None)
                    # No provenance at all on this branch, so any ceiling there is came
                    # from the Fit range box -- there is no detector limit here to
                    # credit it to.
                    _exp_kw = dict(_exp_kw, t_max_source=self._t_max_source(None, _hi2))
                _refit = lambda kk, _s=data, _a=_lo2, _b=_hi2, _kw=dict(_exp_kw): (
                    populations.fit_exponential(_s, t_min=_a, t_max=_b, n_components=kk,
                                                **_kw))
                # ...and the bootstrap goes to the worker here too, with the same
                # RuntimeWarning suppression the synchronous call had. See
                # `_point_fit_now`.
                fit_key_of = lambda kk, _d=self._sample_digest(data), _a=_lo2, \
                        _b=_hi2, _kw=_exp_kw: (
                    ("fullfit", _FIT_EXP, _d, None if _a is None else float(_a),
                     None if _b is None else float(_b), int(kk),
                     _kw.get("quantity"), _kw.get("t_max_source")))
                fit = _point_fit_now(lambda _kk=k: _refit(_kk), fit_key_of(k),
                                     quiet=True)
                if dwell_kw:
                    # `populations.fit_exponential` guessed t_min and stamped its
                    # NO_TMIN_WARNING -- a sentence written for dwells, which labels
                    # this column's shortest value " s" and calls it a dwell. The
                    # panel's own note directly above already states the same fact in
                    # the column's own terms, so the pane must not also print the
                    # seconds-labelled copy: one fact, one line, the honest one.
                    _guessed = modules.NO_TMIN_WARNING.split("{", 1)[0]
                    fit.meta["warnings"] = [
                        w for w in fit.meta.get("warnings", [])
                        if not str(w).startswith(_guessed)]
                sample_for_gof, lo_for_gof, hi_for_gof = data, _lo2, _hi2
            names = ("tau", "τ")
        else:  # gaussian mixture -- stats requires non-negative magnitudes
            inp = self._depth_inputs(col, src)
            source = self._finite(inp.values, notes) if inp is not None else v
            source, _crop_lo, _crop_hi = _restrict(source, None, None)
            # THE FIT IS OF |x|, AND EVERY SURFACE THAT PRINTS IT SAYS SO.
            #
            # `fit_depth_gaussians` is a depth model and requires non-negative
            # magnitudes, so a signed column is handed `np.abs(source)`. Every number
            # that comes back -- mu, sigma, the intervals, the BIC and the KS verdict --
            # is therefore about MAGNITUDES, and the pane printed them as if they were
            # about the plotted column. On the owner's real -700 mV detection,
            # `baseline_pA` (115 of 115 values in -71.7..-47.3 pA) produced the TITLE
            #     μ = 56.59 pA   (n = 115)
            # -- positive, for a quantity that is about -57 pA, on the one line
            # `Save image` keeps -- with a "fit" curve of 1.18e-81 against bars of 16,
            # i.e. a flat line on the axis floor, and the only disclosure ("115 negative
            # value(s) fit by magnitude.") buried in Details, where it says nothing
            # about the sign of the reported mu or about the curve. `baseline_pA`,
            # `level_pA` and every raw-current column `measure.MEASUREMENT_SCHEMA`
            # declares (mean, median, level, baseline, min, max) are negative
            # throughout on a negatively-biased recording, so this is two combo clicks
            # away in normal use.
            #
            # Two facts, said where they are read: the fitted quantity is named |col|
            # in the title, the model line, the CI head and the goodness-of-fit note
            # (below), and -- when the column is negative THROUGHOUT -- the curve is
            # drawn where those events actually are. Reflection is exact there and not
            # a cosmetic: if x < 0 and |x| ~ f, then x ~ f(-x), same density, same KS
            # statistic. A MIXED-sign column is not reflectable (folding merges two
            # lobes into one sample), so nothing is mirrored and the note says the
            # curve covers only the positive half of the axis.
            _neg = int(np.sum(source < 0))
            _all_neg = bool(source.size) and _neg == int(source.size)
            if _all_neg:
                notes.append(
                    f"All {_neg} value(s) of “{col}” are negative. The mixture is fitted "
                    f"to |{col}|, so μ and σ below are MAGNITUDES; the curve drawn is "
                    f"that fit reflected onto the negative axis, where these events lie "
                    f"(reflection is exact — it changes no estimate and no KS statistic).")
            elif _neg:
                notes.append(
                    f"{_neg} of {int(source.size)} value(s) of “{col}” are negative. The "
                    f"mixture is fitted to |{col}|, so μ, σ, the intervals and the "
                    f"goodness-of-fit verdict below describe MAGNITUDES, not the signed "
                    f"values on the x axis — and the drawn curve covers only the "
                    f"positive half of that axis.")
            _floor = None if inp is None else inp.depth_floor
            # WHAT WAS FITTED TRAVELS WITH THE FIT. Without it the FitResult comes back
            # labelled "Gaussian mixture on dI/I0" whatever column it was handed, so
            # fitting depth_pA printed picoamp means under a dimensionless caption
            # (report High-4) -- on the one dataset where dI/I0 > 1 is the fact that
            # must not be confused with anything else.
            _quantity, _unit = _column_quantity(col)
            # ...INCLUDING IN THE FITTER'S OWN SENTENCES. `quantity` is what
            # `fit_depth_gaussians` names the model after ("Gaussian mixture on dI/I0")
            # and what its identifiability notes quote; handing it the magnitude bars
            # is what keeps the model line, the notes, the title and the CI head from
            # disagreeing about which quantity the numbers belong to.
            if _neg:
                _quantity = f"|{_quantity or col}|"
            # ...AND SO DOES THE UNIT IT WILL BE READ IN. `_display_unit` is the SAME
            # rule the estimate line and `_ci_note` obey (`unit_of(...) == "s"` ->
            # show it in the axis's unit), handed to the fitter so the sentences IT
            # writes cannot disagree with the lines this method writes. Before it, one
            # pane printed "μ = 0.1923, 1.923, 2.314 ms" and, two lines below,
            # "mu_3 = 0.002314 s has a 95% CI of [0.002127, 0.00256]" -- the same
            # parameter 1000x apart, and the interval with no unit at all.
            #
            # The DATA is untouched: `unit` (and therefore params, ci and param_units)
            # stays seconds; only the words move.
            _display = self._display_unit(col, _unit)
            _refit = lambda kk, _s=np.abs(source), _f=_floor, _q=_quantity, _u=_unit, \
                    _d=_display: (
                populations.fit_gaussian_mixture(_s, k=kk, depth_floor=_f,
                                                 quantity=_q, unit=_u, display=_d))
            # The 400-replicate EM bootstrap is not on the click path either. See
            # `_point_fit_now`; here it also carries the identifiability warnings,
            # which are statements about the intervals it produces.
            # ...AND THE DISPLAY UNIT IS PART OF THE KEY. It is not part of the data —
            # params, ci and param_units all stay in `_unit` — but it IS part of what the
            # fitter WRITES: the identifiability notes quote their own numbers, in words,
            # in the display unit they were handed. Leaving it out of the key meant a unit
            # toggle re-used the fit computed under the PREVIOUS unit and kept its
            # sentences: the axis said ms while "UNIDENTIFIED COMPONENT(S): mu_2 =
            # 0.02436 s ..." sat two lines below it, which is exactly the one-pane-one-unit
            # disagreement this pane was fixed for. Keyed on it, each unit gets the
            # sentences it was written for, and switching back is still a cache hit.
            fit_key_of = lambda kk, _d=self._sample_digest(np.abs(source)), \
                    _f=_floor, _q=_quantity, _u=_unit, _dsp=_display: (
                ("fullfit", _FIT_GMM, _d, None if _f is None else float(_f),
                 int(kk), _q, _u, _dsp))
            fit = _point_fit_now(lambda _kk=k: _refit(_kk), fit_key_of(k))
            # The TITLE is the only thing `Save image` keeps (see the module header),
            # so the magnitude bars go in it, not only in Details.
            names = ("mu", f"μ(|{col}|)" if _neg else "μ")
            # ...and the curve is evaluated where these events are. `np.abs` is the
            # identity on the positive grid of an unsigned column, so this is one
            # expression for both cases.
            if _all_neg:
                grid_of = np.abs
            gof_frame = f"|{col}|" if _neg else ""
            sample_for_gof, lo_for_gof, hi_for_gof = np.abs(source), None, None
            # The mixture is fitted to the cropped values with no censoring model, so
            # the window is recorded for the note and the note says "cropped".
            trunc_lo, trunc_hi, trunc_modelled = _crop_lo, _crop_hi, False

        # THE TRUNCATION THE LIKELIHOOD ACTUALLY GOT. For every censored family that is
        # exactly the pair handed to the fitter -- the same pair the goodness-of-fit test
        # and the rival ladder are given, which is what keeps all three describing one
        # fit. Recorded here, once, rather than re-derived by the note that prints it.
        if trunc_modelled:
            trunc_lo, trunc_hi = lo_for_gof, hi_for_gof

        # AUTO (BIC): the chosen k replaces the fit, and is written back into the spin
        # so the number on screen is the number that was fitted.
        auto_note = ""
        if (self._applied_fit_k_auto and kind in _MULTI_COMPONENT_FITS):
            k_auto, auto_note = self._auto_component_count(_refit, k, fit)
            if k_auto != k:
                # Re-fit at the chosen k so the drawn fit has the CIs (and therefore
                # the identifiability warnings) every other fit this window draws has
                # -- but through the SAME seam, or Auto (BIC) would put the very
                # bootstrap `_point_fit_now` just took off the click path straight back
                # onto it (measured: 4.6 s at n = 1000, 9.6 s at n = 5000).
                fit = _point_fit_now(lambda _kk=k_auto: _refit(_kk),
                                     fit_key_of(k_auto)) if fit_key_of is not None \
                    else _refit(k_auto)
                k = k_auto
                self._updating = True
                try:
                    self._fit_k.setValue(int(k_auto))
                finally:
                    self._updating = False
                # The APPLIED count moves with it, or the Apply button would sit lit
                # for a "change" the window made to itself and the next Apply would
                # re-fit at a k nobody chose.
                self._applied_fit_k = int(k_auto)
                self._sync_fit_apply_buttons()

        # A FIT WITHOUT A FINITE BIC IS NOT A FIT, AND MUST NOT BE DRAWN.
        # `inverse-gaussian (first passage)` returned BIC = +inf on the owner's -700 mV
        # recording and was still offered, still drawn, and still printed its mu and
        # lambda (report Low-19). One check here covers every family, including any
        # added later: the caller turns this into "Fit unavailable: ..." in Details and
        # leaves the histogram standing.
        if not np.isfinite(float(fit.bic)) or not np.isfinite(float(fit.loglik)):
            raise RuntimeError(
                f"“{kind}” produced a non-finite likelihood on these events "
                f"(BIC = {fit.bic}); it cannot describe this sample and has not been "
                f"drawn. Choose another model.")

        # The same provenance stamp the controller applies, so a FitResult read out
        # of this window is indistinguishable from the Fits panel's for the same
        # events -- including where t_min came from and what was excluded.
        #
        # STAMPED WITH THE WINDOW THAT WAS FITTED, not the detector's own. The Fits
        # panel has no Fit range box, so for it the resolution's window IS the fitted
        # window and the two are the same object. Here they are not: `_restrict` may
        # have narrowed the pair before the fitter saw it, and
        # `stamp_fit_provenance` assigns `t_min`/`t_min_source` unconditionally (every
        # other key in it goes through `setdefault`). A fit conditioned on a typed
        # floor of 0.001 s therefore came back carrying meta['t_min'] = 2e-05 "from
        # min_dwell_s" -- one FitResult with two floors, and `plot.figures._fit_caveats`
        # printing the wrong one as "$t_{min}$ = 200.0 $\\mu$s" onto an EXPORTED figure,
        # where it stops being reviewable. So the resolution is replaced by the one
        # that describes this fit before it is stamped.
        if inp is not None:
            modules.stamp_fit_provenance(fit, self._as_fitted(inp, trunc_lo, trunc_hi))

        # THE DEFERRED INTERVAL, RESOLVED. Cached per (model, sample bytes,
        # censoring), so the redraw that follows the worker finding the answer picks
        # it up for free -- and so a fit re-made from the same events on a bin change
        # or a unit toggle never re-bootstraps. While it is pending the panel SAYS so:
        # an empty CI with nothing beside it reads as "this family does not offer
        # one", which for gamma and inverse-gaussian was true until finding #10 and
        # is the exact misreading it was fixed to stop.
        #
        # `ci_state` is what the CI line BELOW the parameters says, and there are three
        # of them -- pending, failed, and the interval itself. They share one slot on
        # purpose: the placeholder promised "they appear when you stop clicking", and a
        # promise has to resolve where it was made or it is not a placeholder, it is a
        # message that vanishes. That is exactly what shipped -- the deferred interval
        # was applied to the FitResult and then never printed anywhere, in EITHER state,
        # so the pending line was replaced by nothing at all.
        # ONE SLOT, whichever mechanism the family's interval came through: the
        # `deferred_ci` families fill it below, the `_point_fit_now` families filled it
        # in the branch. Two lines about one fit's interval is the two-surfaces-one-fact
        # failure this pane exists to avoid.
        ci_state = fit_ci_state
        if ci_key is not None and ci_jobs and fit.meta.get("ci_pending"):
            ready, payload = self._deferred_note(ci_key, ci_jobs[0])
            if not ready:
                ci_state = self._CI_PENDING_TEXT
            elif isinstance(payload, dict):
                stats.apply_ci(fit, payload)
            else:   # the bootstrap raised; _deferred_note remembers the failure
                # AND THE PROMISE IS DISCHARGED ON THE FIT, not only in the note.
                # `meta['ci_pending']` means "an interval is coming"; only `apply_ci`
                # ever cleared it, so a bootstrap that raised left it True for the life
                # of the FitResult -- into `fit_to_json`, into a saved project, into any
                # surface that reads the flag rather than this window's text.
                stats.ci_failed(fit, "the deferred bootstrap raised")
                ci_state = self._CI_FAILED_TEXT

        # Anything the fit flagged (a guessed t_min, exclusions, floor bias, a
        # modelled right truncation) belongs in Details, not just in .meta.
        notes.extend(str(n) for n in fit.meta.get("warnings", []))

        if log_x:
            grid = np.geomspace(edges[0], edges[-1], 400)
            du = float(np.log10(edges[1]) - np.log10(edges[0]))
            width = grid * np.log(10.0) * du
        else:
            grid = np.linspace(edges[0], edges[-1], 400)
            width = float(edges[1] - edges[0])
        # WHERE THE FIT LIVES vs WHERE THE PLOT LOOKS. For every family but a mixture
        # over an all-negative column these are the same axis and `grid_of` is None.
        at = grid if grid_of is None else grid_of(grid)
        density = np.asarray(fit.predict(at), dtype=np.float64)
        expected = (fit.n * density * width
                    * self._fit_scale(edges, width, int(v.size)))

        # AND THE PLOT DOES NOT ASSERT A CURVE IT DID NOT DRAW. A fit whose mass sits
        # outside the plotted range renders as a flat line on the axis floor -- the
        # mirrored Gaussian above was 1.18e-81 against bars of 16 -- while the caption,
        # the interval and the KS verdict all read as a description of the picture. One
        # check, over every family: how much of the fitted distribution is actually
        # inside the window the user is looking at. It is a probability, so it needs no
        # units and no bar heights to compare against.
        #
        # Integrated over the PLOTTED axis, not over the fit's own variable: with a
        # reflection in play those are different variables, and the question this line
        # answers is about the picture. `grid` is ascending in every branch above.
        # Through `analysis._compat.trapezoid`, the project's own numpy-1.26/2.0 shim --
        # a private copy of that getattr is how two modules end up disagreeing about
        # which numpy they are on.
        _seen = float(_trapezoid(density, grid)) if density.size > 1 else 0.0
        if not np.isfinite(_seen) or abs(_seen) < 0.01:
            notes.append(
                f"THE FITTED CURVE IS NOT VISIBLE ON THIS PLOT: {abs(_seen) * 100:.2g}% "
                f"of the fitted distribution's mass falls inside the plotted x range, so "
                f"the overlay is a flat line on the axis floor. The parameters above "
                f"describe a distribution sitting somewhere else; nothing here checks "
                f"them against the bars.")

        pen = (pg.mkPen(colour, width=self._style.line_width + 1, style=Qt.DashLine)
               if who else self._fit_pen())
        item = pg.PlotDataItem(x=grid, y=expected, pen=pen)
        self._plot.addItem(item)
        self._fit_items.append(item)

        # -- the two optional pictures of the fit's own quality ------------------------
        # THE SAME y FACTOR THE CURVE USED, extracted rather than recomputed: band and
        # curve have to agree bin for bin under three normalizations and a log x axis,
        # and one expression they both multiply by is the only way to guarantee it.
        y_scale = float(fit.n) * width * self._fit_scale(edges, width, int(v.size))
        resid_short = ""
        if self._style.show_ci_band:
            band_note = self._draw_ci_band(fit, grid, at, y_scale, colour)
            if band_note:
                notes.append(band_note)
        # ONE STRIP, FOR THE FIRST FITTED SERIES, AND IT SAYS SO. There is one axis under
        # the plot and an overlay has one fit per source; stacking four sources' residuals
        # into it would draw four unlabelled scatters over one another. Drawing nothing at
        # all on an overlay would be the silent no-op this round exists to remove, so the
        # first fit gets the strip and the note names whose it is. `_clear_plot` hid the
        # strip at the top of this draw, so `isHidden()` is exactly "no series has taken
        # it yet" and needs no flag of its own.
        #
        # The reflected-column case (`grid_of`) is skipped for a harder reason: `edges`
        # are then bins of the PLOTTED variable while the fit lives in another, so a
        # residual per bin is not defined without re-binning the sample the fitter saw.
        if (self._style.show_residuals and grid_of is None
                and sample_for_gof is not None and self._resid_widget.isHidden()):
            resid_note, resid_short = self._draw_residual_strip(
                fit, sample_for_gof, edges, log_x)
            if resid_note:
                if who:
                    resid_note = (
                        f"RESIDUAL STRIP: {self._legend_name(src)} only — there is one "
                        f"axis under the plot and this overlay has one fit per source. "
                        + resid_note)
                notes.append(resid_note)
        if self._fit_item is None:
            self._fit_item = item
        if self._last_fit is None:
            self._last_fit = fit
        if src is not None:
            self._fits[src.key] = fit
            self._fit_columns[src.key] = col

        # THE PRINTED NUMBER CARRIES ITS UNIT, PER PARAMETER, FROM THE FIT ITSELF.
        # "τ = 0.02331" beside an axis reading ms is a 1000x error waiting to be
        # copied into a paper (report Low-28), and "μ = 85.46, 99.69, 105.8" with no
        # unit at all was picoamps (High-4). `FitResult.unit_of` knows which of a
        # fit's parameters are times and which are dimensionless shapes; seconds are
        # shown in the unit the axis is already showing, and nothing else is touched.
        keys = [key for key in fit.params if key.startswith(names[0])]
        param_unit = fit.unit_of(keys[0]) if keys else ""
        # THROUGH THE SHARED RULE, not a copy of it: :meth:`_display_unit` is what the
        # interval line below and the fitter's own notes are also given, so the three
        # writers of this pane cannot drift apart again.
        scale, uname = self._display_unit(col, param_unit)
        vals = [fit.params[key] * scale for key in keys]
        unit_txt = f" {uname}" if uname else ""
        shown = ", ".join(f"{val:.4g}" for val in vals)
        summary = f"{names[1]} = {shown}{unit_txt}   (n = {fit.n})"
        # IS THIS MODEL EVEN RIGHT? Decided HERE, before the one-line summary is handed
        # to the title and the legend, because those two are the whole of what Save
        # image exports (S7-JRN2). The verdict used to be drawn only inside the on-plot
        # values box, which `StyleSettings.fit_values_on_plot` keeps off by default, so
        # at the app's own defaults on its own shipped dwell template the -600 mV
        # recording left the app as "τ = 20.41 ms (n = 275)" in the title and
        # "fit: τ = 20.41 ms (n = 275)" in the legend -- for a curve the app's own KS
        # test rejects at D = 0.3966 against a 1% critical value of 0.0981
        # (p = 1.6e-39; the dwell quartiles are 16.61 / 18.14 / 19.48 ms, the drawn
        # exponential predicts 5.89 / 14.17 / 28.32 ms) -- while `show_gof`'s docstring
        # and the "KS verdict on figure" tooltip promised the opposite in writing. A
        # rejected fit now says so on the same line, whatever `show_gof` and
        # `fit_values_on_plot` are set to; the curve is still drawn (whether to suppress
        # it is a decision the owner has not made), but it is labelled. The panel used
        # to state one authoritative number with a tight CI and no goodness-of-fit
        # statistic beside it while the drawn curve missed the histogram (report
        # High-3); the verdict costs a millisecond and is attempted on every fit.
        #
        # THE TAGGED LINE IS THE LEGEND ENTRY'S AND DETAILS'; THE TITLE TAKES `plain`
        # AND THE VERDICT THROUGH ITS WARN CHANNEL (A1-N1, `_ks_headline`): a typed
        # title displaces the automatic text, and this tag with it, while `warn` is
        # prepended over the user's own text.
        gof = self._goodness_note(fit, sample_for_gof)
        plain_summary = summary
        ks_rejected = bool(gof) and not gof.get("describes_data")
        if ks_rejected:
            summary = f"{summary} — {_KS_REJECTED_TAG}"
        legend_who = self._legend_name(src) if (who and src is not None) else who
        self._legend_add(item, f"{legend_who} fit: {summary}".strip() if legend_who
                         else f"fit: {summary}",
                         kind="fit", key=src.key if src is not None else "")
        notes.append(fit.model)
        notes.append(f"BIC = {fit.bic:.6g}   AIC = {fit.aic:.6g}   components = {k}")
        # THE INTERVAL, WHERE ITS PLACEHOLDER PROMISED IT. One slot, three states:
        # the numbers, "still computing", or "the bootstrap failed". Every fitter in
        # `stats` returns a `ci` and this window printed NONE of them -- so a user read
        # "k = 25.45" with no uncertainty beside it whether the interval was unavailable,
        # pending, or sitting finished on the FitResult in front of them.
        ci_line = ci_state or self._ci_note(fit, col)
        if ci_line:
            notes.append(ci_line)
        if auto_note:
            notes.append(auto_note)
        # THE VERDICT IN FULL, in Details (decided above, where the one-line summary
        # took its tag).
        if gof:
            # ...AND IT SAYS WHAT IT IS A VERDICT ABOUT. `goodness_of_fit` is handed
            # `sample_for_gof`, which for a signed column is |x|, and its sentence reads
            # "The fitted curve is consistent with these data" -- a claim the reader
            # attaches to the picture in front of them. Naming the frame here, where the
            # sentence is printed, is the same rule the units follow.
            notes.append(gof["note"] + (
                f" (This verdict is about {gof_frame}, the magnitudes that were fitted, "
                f"not about the signed values on the x axis.)" if gof_frame else ""))
        rivals = self._rival_family_note(kind, col, fit, gof, sample_for_gof, lo_for_gof,
                                         hi_for_gof, k)
        if rivals:
            notes.append(rivals)
        if fit_lo is not None or fit_hi is not None:
            # Through the shared rule, so the range prints in the SAME unit as the
            # parameters above it: the axis's unit for a time column, the column's own
            # otherwise. The old special case answered "" for every non-time column,
            # which left a picoamp window printing bare numbers on the one pane whose
            # job is to say what its numbers are in.
            f, unit = self._display_unit(col, _column_quantity(col)[1])
            # THE WINDOW THE LIKELIHOOD ACTUALLY GOT, NOT THE ONE THAT WAS TYPED.
            # `_restrict` intersects the typed pair with what the detector could report
            # ("the truncation is the INTERSECTION", its own docstring), and the
            # intersection is what every branch hands the fitter. Printing the raw typed
            # pair made this line contradict, in the same pane, the censoring string and
            # the right-truncation note describing the SAME fit: typed [1e-06, 5] s over
            # a fit conditioned on [2e-05, 0.0317424] s -- the lower bound 20x too small
            # and the upper 158x too large, on the one line whose whole job is to say
            # what the reported tau is conditioned on. One pane, one truth.
            #
            # TWELVE SIGNIFICANT FIGURES, not six. A bound the user TYPED is short and
            # prints short at any precision ("0.5", "20", "1e-06"); a bound that came
            # out of an INTERSECTION is a measured quantity -- max_dwell_s of the owner's
            # -700 mV run is 0.03174239924320212 s -- and six figures rounded it by
            # 7.6e-10 s, which is a bound that no longer equals the one the fit reports
            # in `meta['t_max']`. This line exists to be copied into a methods section
            # beside that number; it must round-trip to it.
            eff_lo = fit_lo if trunc_lo is None else trunc_lo
            eff_hi = fit_hi if trunc_hi is None else trunc_hi
            shown_lo = "-inf" if eff_lo is None else f"{eff_lo * f:.12g}"
            shown_hi = "+inf" if eff_hi is None else f"{eff_hi * f:.12g}"
            # ...and when the two differ, SAY the typed pair and why it moved, rather
            # than silently swapping one number for another under the same words.
            def _same(a, b) -> bool:
                if a is None or b is None:
                    return a is None and b is None
                return abs(float(a) - float(b)) <= 1e-15 * max(1.0, abs(float(a)))
            narrowed = ""
            if not (_same(eff_lo, fit_lo) and _same(eff_hi, fit_hi)):
                typed_lo = "-inf" if fit_lo is None else f"{fit_lo * f:.12g}"
                typed_hi = "+inf" if fit_hi is None else f"{fit_hi * f:.12g}"
                narrowed = (
                    f" Your typed range was [{typed_lo}, {typed_hi}] {unit}; it was "
                    f"intersected with what this source's detector could report, and the "
                    f"intersection above is what was fitted.")
            how = ("the likelihood is TRUNCATED to this window, not merely cropped. "
                   "BIC/AIC compare only against fits over the SAME range."
                   if trunc_modelled else
                   "the sample is CROPPED to this window — this family does not "
                   "renormalise its likelihood to it, so the estimates are those of a "
                   "distribution fitted to the surviving values, not of one conditioned "
                   "on the window. BIC/AIC compare only against fits over the SAME range.")
            notes.append(
                f"FIT RANGE [{shown_lo}, {shown_hi}] {unit} — n = {fit.n} of the "
                f"available events; {how}{narrowed}")
        # A COMPONENT-COUNT LADDER OVER A FAMILY THAT HAS NO COMPONENT COUNT IS THE SAME
        # FIT, FOUR TIMES, COMPARED WITH ITSELF.
        #
        # ``gamma``, ``inverse-gaussian`` and ``exponential + gamma`` have a fixed
        # parameterisation: their ``_refit`` above takes ``kk`` and ignores it, so every
        # rung of the ladder re-runs the identical deterministic maximisation and comes
        # back with the identical BIC. The comparison is `rival.bic < best - 10`, which
        # can never fire, so the ladder was guaranteed to print nothing and guaranteed to
        # pay for it. The window already knows which families have a count --
        # :data:`_MULTI_COMPONENT_FITS` is what disables the Auto (BIC) box for exactly
        # these three -- and asking the same question here keeps the two from drifting
        # apart.
        #
        # WHAT IT COST, measured on the owner's own samples. One ``exponential + gamma``
        # maximisation is 9.5 s on the -600 mV merged table (n = 240) and 207 ms on
        # -700 mV (n = 986); the ladder ran FOUR more of them per draw. That is ~38 s of
        # the -600 mV advisory batch -- which the follow-up pass measured at 48.7 s --
        # spent recomputing a number the panel already had, and it is the reason an abort
        # had 48.7 s of work to interrupt instead of 9.5 s. Removing work beats moving it.
        better = ""
        if kind in _MULTI_COMPONENT_FITS:
            better = self._better_component_count(
                _refit, k, fit,
                key=("betterk", self._sample_digest(sample_for_gof), lo_for_gof, hi_for_gof,
                     int(k), kind, col))
        if better:
            notes.append(better)
        return (_FitSummary(summary, self._full_fit_summary(fit, col, gof, resid_short),
                            rejected=ks_rejected, plain=plain_summary),
                notes)

    def _full_fit_summary(self, fit, col: str, gof: dict | None,
                          resid_short: str = "") -> str:
        """Every number this fit has, on one line, for the box drawn ON the figure.

        Owner note #13: "once the Fit has been applied, include the numerical values on
        the plot; make these plots presentation ready". The one-line title summary
        carries only the HEADLINE parameter -- tau, or the mixture's mu's -- so a
        two-component Gaussian mixture put its means on the figure and left both sigmas
        and both weights in a Details pane that the exported PNG does not contain. A
        reader of that PNG cannot tell a tight mixture from a broad one.

        So: every entry of ``fit.params``, each scaled and named through the SAME
        :meth:`_display_unit` rule the title uses (a time parameter in the axis's unit,
        a dimensionless shape bare), then n, then BIC, then the KS verdict -- which the
        Style checkbox's own tooltip has been promising ("τ, μ, KS verdict…") while
        printing none of it.
        """
        parts = []
        for key, value in fit.params.items():
            scale, uname = self._display_unit(col, fit.unit_of(key))
            parts.append(f"{_param_symbol(key)} = {float(value) * scale:.4g}"
                         + (f" {uname}" if uname else ""))
        parts.append(f"n = {fit.n}")
        if np.isfinite(float(fit.bic)):
            parts.append(f"BIC = {float(fit.bic):.6g}")
        # THE SWITCH CANNOT HIDE A REJECTION (StyleSettings.show_gof). Off means "stop
        # decorating a figure whose fit is fine"; a curve the test threw out says so on
        # the figure whatever the box is set to, because that sentence is the difference
        # between a measurement and a decoration.
        if gof and (self._style.show_gof or not gof.get("describes_data")):
            # The VERDICT, not the whole sentence: the box sits on the plot and the
            # sentence is a paragraph. Details still prints it in full.
            parts.append(f"KS = {float(gof['ks']):.3f} — "
                         + ("consistent with these data" if gof.get("describes_data")
                            else "REJECTED at the 1% level"))
        # The strip's number, on the figure, because the strip itself is not exported.
        if resid_short:
            parts.append(resid_short)
        return "   ".join(parts)

    #: What stands in for an advisory note while it is still being computed. It says the
    #: number is not ready rather than that there is nothing to say -- an empty line would
    #: read as "no better model exists", which is the opposite of the truth on data where
    #: the chosen fit has just been rejected.
    #:
    #: ONE PER NOTE, not one shared string. A shared placeholder printed "MODEL COMPARISON"
    #: for the component-count hint too, so a fit that PASSED its verdict -- whose rival
    #: scan is correctly gated off and has nothing to say -- still announced a pending
    #: model comparison that was never going to arrive.
    #: What the plot TITLE says while the exponential + gamma maximisation is on the
    #: worker. It is the summary line for that draw, so it goes where the "τ = …" would
    #: have gone -- a title that simply vanished for nine seconds would read as "this
    #: family has nothing to say about these events", which is the opposite of the truth.
    _FIT_PENDING_SUMMARY = "exponential + gamma: fitting — the curve appears when you stop clicking"

    #: ...and the same state, said in full, in Details. It has to distinguish "waiting"
    #: from "unavailable": `Fit unavailable: …` is the sentence this window prints when a
    #: family REFUSES a sample, and the two must never be confused. It also says what is
    #: NOT waiting, because the histogram under it is complete and a user who thinks the
    #: whole plot is provisional will sit and wait for bars that are already final.
    _FIT_PENDING_TEXT = (
        "FIT: “exponential + gamma (bimodal)” is a twelve-start optimisation — the one "
        "family here whose point estimate costs more than a click can afford — so it is "
        "computed off the click path and its curve, parameters and BIC appear when you "
        "stop clicking. Nothing else on this plot is waiting: the histogram, its counts "
        "and the detection limit are final. (Set Fit = none to curate without it.)")

    _RIVAL_PENDING_TEXT = ("MODEL COMPARISON: still computing — it appears when you stop "
                           "clicking. (Set Fit = none to curate without it.)")
    _BETTER_K_PENDING_TEXT = ("COMPONENT COUNT: still checking whether another number of "
                              "components fits better — it appears when you stop clicking.")

    #: The drawn curve, honest about which state it is in. A fit whose interval has
    #: not landed yet must not be silent about it: for ``gamma`` and
    #: ``inverse-gaussian`` an empty ``ci`` used to MEAN "this family reports no
    #: uncertainty" (report Medium-18), so a blank where the interval goes is the one
    #: reading this line has to prevent. The estimates themselves are final the moment
    #: they are drawn -- they come out of the maximisation, before any resampling --
    #: and the sentence says so, or a user would reasonably wait before copying a tau.
    _CI_PENDING_TEXT = (
        "95% CONFIDENCE INTERVALS: still computing — they appear when you stop clicking. "
        "The estimates, BIC/AIC and the drawn curve above are final and will not change.")
    _CI_FAILED_TEXT = (
        "95% CONFIDENCE INTERVALS: the bootstrap failed on this sample; the point "
        "estimates stand on the likelihood, but there is no interval beside them.")

    @staticmethod
    def _full_fit_job(make, quiet: bool = False):
        """``make()`` -- the WHOLE fit, bootstrap included -- as a job for ``_NoteWorker``.

        ``make`` is built on the GUI thread out of a numpy array and numbers (it is the
        same closure the rival ladder is handed) and touches no widget, which is the
        rule every job on that worker follows.

        ``quiet`` mirrors the call site: the no-detection-limit branch of the
        exponential wraps its synchronous call in :func:`stats.quiet_runtime_warnings`
        because ``populations.fit_exponential`` raises a ``RuntimeWarning`` for the
        guessed ``t_min`` that the pane has already said in its own words. Moving the
        call must not start printing it.

        RETURNS its failure rather than raising it, exactly as :meth:`_exp_gamma_job`
        does -- except for ``OperationCancelled``, which must reach the worker's own
        ``break`` so a superseded job leaves nothing cached behind it.
        """
        def _run():
            try:
                if quiet:
                    with stats.quiet_runtime_warnings():
                        return make()
                return make()
            except OperationCancelled:
                raise
            except Exception as exc:            # noqa: BLE001 - reported, not swallowed
                return exc
        return _run

    @staticmethod
    def _exp_gamma_job(sample, lo, hi, quantity=None, unit=None):
        """The exponential + gamma maximisation, as a plain callable for ``_NoteWorker``.

        Built on the GUI thread out of a numpy array and two numbers -- it touches no
        widget and reads nothing that a curation click can change underneath it, which is
        the rule every job on that worker follows.

        RETURNS its failure. An exception escaping here is filed by the worker as
        ``_NOTE_FAILED`` and cached as ``""``, and ``""`` cannot tell the user that this
        sample has 14 events and the family needs 20, or that an event is shorter than the
        stated detection limit. Those messages are the fitter's, they are worth reading,
        and the synchronous call this replaced printed them.

        ``OperationCancelled`` is re-raised rather than returned, and that distinction is
        the whole cancellation contract: a cancelled job must leave NOTHING behind, so it
        has to reach the worker's own ``except OperationCancelled: break`` instead of
        being cached as this sample's answer for ever.
        """
        def _run():
            # No `quiet_runtime_warnings` here: the synchronous call this replaces had
            # none, and this move is meant to change WHEN the fit runs, not what it says.
            try:
                return stats.fit_dwell_exp_gamma(sample, t_min=lo, t_max=hi,
                                                 quantity=quantity, unit=unit)
            except OperationCancelled:
                raise
            except Exception as exc:            # noqa: BLE001 - reported, not swallowed
                return exc
        return _run

    def _ci_note(self, fit, col: str) -> str:
        """The fit's 95% intervals, in the unit the parameter is printed in, or ``""``.

        THE OTHER HALF OF :data:`_CI_PENDING_TEXT`. That placeholder says the intervals
        "appear when you stop clicking"; they did not, because this window has never
        printed a ``ci`` at all -- not for the deferred peaked families, and not for the
        exponential or the depth mixture either, both of which have bootstrapped one on
        every fit since long before the deferral existed. `FitResult.summary()` prints
        them and the Inspector's Fits box shows that; the Plots window's Details pane
        showed the estimate alone, which reads as a measurement with no uncertainty.

        UNITS FOLLOW THE PARAMETER, NOT THE FIT -- the same rule the estimate line
        obeys, and for the same reason (report Low-28 / High-4): ``gamma`` returns a
        dimensionless shape ``k`` beside a ``theta`` in seconds, so one caption over
        both is guaranteed to mislabel one of them. A time is shown in whatever unit
        the axis is already showing; everything else in its own.

        Silent, deliberately, when there is no interval to show: ``exponential + gamma``
        reports none by design (every replicate is a 12-start optimisation) and SAYS so
        in ``meta['warnings']``, which Details already prints. A line here would be a
        second, emptier voice on the same subject.
        """
        ci = getattr(fit, "ci", None)
        if not ci:
            return ""
        parts = []
        for key, bounds in ci.items():
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
            except (TypeError, ValueError, IndexError):     # pragma: no cover - defensive
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            unit = fit.unit_of(key) if hasattr(fit, "unit_of") else ""
            # Same rule as the estimate line and as the notes the fitter writes -- one
            # implementation, three callers. See :meth:`_display_unit`.
            scale, uname = self._display_unit(col, unit)
            lo, hi = lo * scale, hi * scale
            parts.append(f"{key} [{lo:.4g}, {hi:.4g}]" + (f" {uname}" if uname else ""))
        if not parts:
            return ""
        # ``n_boot_ok`` is what every fitter in `stats` records: the number of
        # replicates that CONVERGED, not the number drawn. Saying "resamples" would
        # quietly overstate a bootstrap that lost half of them, which is exactly the
        # kind of number this pane exists to keep honest -- and a small `n` here is
        # the reader's only clue that a wide interval is the optimiser's fault.
        n_ok = fit.meta.get("n_boot_ok") if hasattr(fit, "meta") else None
        head = "95% CI"
        if n_ok:
            head += f" (percentile bootstrap, {int(n_ok)} usable replicates)"
        return f"{head}: " + "   ".join(parts)

    #: How much better a rival component count must be, in BIC, before the panel says so.
    #: 10 is the conventional "very strong" threshold on the BIC scale -- below it the two
    #: models are not meaningfully distinguishable and nagging would be noise.
    _BIC_HINT = 10.0

    #: How many (sample, censoring, k) results the two expensive notes remember.
    #: Small on purpose: this is a redraw cache, not a store. Eviction is FIFO on the
    #: dict's own insertion order.
    _NOTE_CACHE_MAX = 16

    @staticmethod
    def _sample_digest(sample) -> str:
        """A stable key for the exact numbers a note was computed from.

        The bytes, not the identity: `_redraw` rebuilds the sample array from the frame
        on every call, so `id()` changes while the values do not -- which is precisely
        why the notes were being recomputed on a redraw that changed nothing about the
        data.
        """
        if sample is None:
            return "none"
        arr = np.ascontiguousarray(np.asarray(sample, dtype=np.float64).ravel())
        return f"{arr.size}:{hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()}"

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

    def _persist_results(self) -> None:
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
        except Exception:      # pragma: no cover - a save must never block a close
            pass

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
            self._sources = {}
            self._fits = {}
            self._fit_columns = {}
            self._last_fit = None
            self._cluster_result = None
            self._pca_result = None
            self._note_cache = {}
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

    def _cached_note(self, key, make):
        """``make()``, remembered per key.

        THE COST THIS REMOVES, MEASURED. The goodness-of-fit verdict is ~1 ms and the
        cross-family scan behind it is gated on that verdict being REJECT -- which was
        costed as an exceptional path. On the owner's own data it is not exceptional:
        the exponential is rejected on all three recordings (report §3), so the gate is
        open every time and the scan runs on every redraw. Measured, median of 3, on the
        -600 mV merged dwells (n = 239) through a real window: 2.5 ms with no fit,
        331 ms with the exponential, 656 ms with exponential + gamma -- and every bin
        change, unit change or log-x toggle paid it again for numbers that could not
        have moved.

        The key is the SAMPLE's bytes plus the censoring and the component count, so a
        redraw that changes the data, the fit range or k recomputes, and a redraw that
        changes the axis does not.
        """
        cache = self._note_cache
        if key in cache:
            return cache[key]
        value = make()
        cache[key] = value
        while len(cache) > self._NOTE_CACHE_MAX:
            cache.pop(next(iter(cache)))
        return value

    def _display_time(self, col: str) -> tuple[float, str]:
        """``(factor, unit name)`` for showing a TIME column in the axis's own unit.

        Durations and instants alike (:data:`_TIME_COLUMNS` is both): a ``t_end_s``
        axis reading ms must print its numbers in ms too.

        The one mapping between "seconds, as everything is stored" and "the number the
        user is reading off the plot" -- shared by the axis labels, the fit-range boxes
        and the printed parameters, so a tau in the Details pane is in the same unit as
        the tick it sits under."""
        if col in _TIME_COLUMNS:
            return _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))
        return 1.0, "s"

    def _display_unit(self, col: str, param_unit: str) -> tuple[float, str]:
        """``(factor, name)`` for reading a parameter STORED in ``param_unit`` off this
        window: the one rule, in one place, for every line of the pane.

        It is the rule the estimate line and :meth:`_ci_note` used to spell out
        SEPARATELY -- a parameter in seconds is shown in whatever unit the axis is
        showing, anything else is shown in its own. Both now call this, and so does the
        THIRD writer of this pane's sentences (``stats``, which words the
        identifiability and depth-floor notes, and which was printing the storage unit
        beside their converted ones because nobody had ever handed it the rule).

        A COLUMN's unit is not a PARAMETER's unit, which is why the parameter's own is
        the argument: ``gamma`` on a dwell column returns a dimensionless shape ``k``
        beside a ``theta`` in seconds, and one caption over both mislabels one of them
        (report Low-28 / High-4)."""
        if param_unit == "s":
            return self._display_time(col)
        return 1.0, param_unit

    def _goodness_note(self, fit, sample) -> dict | None:
        """The KS verdict on ``fit`` against ``sample``, or None when it cannot be had.

        Never raises into the draw: a fit that cannot be tested still gets drawn, it just
        does not get a verdict. But it is ATTEMPTED on every fit, because the failure this
        answers -- a confident tau under a curve that misses the histogram -- looks exactly
        like a good fit from the readout alone. Costs ~1 ms; measured on a 1000-event
        redraw, the whole verdict is inside the noise of the draw itself.

        PREFERS THE VERDICT THE FIT ALREADY CARRIES. ``stats`` stamps ``meta['gof']`` on
        every fit it returns (``_stamp_gof``), which is what puts the same verdict in the
        Inspector's Fits box -- the surface High-3 was left open on. Reading it here
        rather than recomputing means the two windows cannot quote different KS numbers
        for one fit, and it is also free. The recomputation stays for a FitResult that
        came from somewhere else (a restored project, a hand-built fit).
        """
        stamped = fit.meta.get("gof") if hasattr(fit, "meta") else None
        if isinstance(stamped, dict) and "ks" in stamped:
            return stamped
        if sample is None or np.asarray(sample).size < 2:
            return None
        try:
            return stats.goodness_of_fit(fit, sample)
        except (ValueError, RuntimeError, TypeError):    # pragma: no cover - never fatal
            return None

    def _rival_family_note(self, kind: str, col: str, fit, gof, sample, lo, hi,
                           k: int) -> str:
        """The cross-family BIC table, WHEN the chosen model has already been rejected.

        THE GATE IS THE POINT. Refitting every dwell family costs ~215 ms on the owner's
        -700 mV recording (measured: 8 ms gamma, 6 ms inverse-Gaussian, 200 ms
        exponential+gamma), and `_better_component_count`'s own docstring records what
        happens when an unconditional rival scan meets a curation click: 3.40 s per
        accept/reject. So the scan runs only once the goodness-of-fit test -- already
        computed, and passed in rather than repeated -- has said the drawn curve does not
        describe the data. At that point the user needs the answer and 215 ms is the
        cheapest thing in the interaction.

        A model that passes its KS test costs nothing here, which is what keeps the
        common case at the speed C-1 bought: measured 168 ms per redraw on a
        well-specified 1000-event exponential fit, against 357 ms on the -700 mV dwell
        times where the scan does run.
        """
        if col not in _DWELL_COLUMNS or sample is None or lo is None:
            return ""
        if gof is None or gof.get("describes_data", True):
            return ""
        # ...and, once past the gate, remembered per (sample, censoring, k): on this
        # dataset the gate is open on every redraw, so the scan was being paid for on
        # every bin change and unit toggle. See :meth:`_cached_note`.
        key = ("rivals", self._sample_digest(sample), float(lo),
               None if hi is None else float(hi), int(k), kind)
        ready, rows = self._deferred_note(
            key,
            lambda: stats.compare_dwell_models(
                sample, t_min=float(lo), t_max=None if hi is None else float(hi),
                n_components=int(k)))
        if not ready:
            return self._RIVAL_PENDING_TEXT
        if not rows:
            return ""
        return self._compare_table(rows, kind, fit, k)

    def _compare_table(self, rows, kind: str, fit, k: int) -> str:
        """Render the cross-family BIC table from ``compare_dwell_models`` rows.

        Split out from :meth:`_rival_family_note` so the ROWS -- the expensive part, and
        the part that depends only on the sample -- can be cached while the rendering,
        which depends on which model is currently shown, is not.
        """
        if not rows or rows[0]["bic"] is None:
            return ""
        best = rows[0]
        mine = float(fit.bic)
        lines = [
            "MODEL COMPARISON (same events, same censoring; lower BIC is better):"]
        for r in rows:
            mark = ""
            if r["label"] == kind:
                mark = "  <- shown"
            elif r is best:
                mark = "  <- best"
            # The exponential is the one family whose component count is a separate
            # control, so its row says which k was compared -- otherwise a k=2 fit and a
            # k=1 fit both read "exponential" and the table looks self-contradictory.
            shown_label = (f"{r['label']} (k={k})"
                           if r["label"] == _FIT_EXP and int(k) > 1 else r["label"])
            if r["bic"] is None:
                lines.append(f"    {shown_label:<34} refused: {r['error']}")
            else:
                lines.append(f"    {shown_label:<34} {r['bic']:>12.6g}{mark}")
        if best["label"] != kind and mine - float(best["bic"]) > self._BIC_HINT:
            lines.append(
                f"    “{best['label']}” fits these data better by "
                f"{mine - float(best['bic']):.6g} BIC. The model shown is not the best one "
                f"this window can fit; choose it in the Fit box before reporting parameters.")
        return "\n".join(lines)

    def _fit_range(self, col: str) -> tuple[float | None, float | None]:
        """The user's fit range in DATA units, or (None, None).

        Typed in whatever unit the axis is showing -- ms for a dwell column when Time unit
        is ms -- because that is the number the user is reading off the plot. Converting
        back here is the same reason the time unit is an axis scale and not a data change:
        one representation in the data, one in the eye, and exactly one place that maps
        between them.
        """
        def _read(edit):
            txt = edit.text().strip()
            if not txt:
                return None
            try:
                v = float(txt)
            except ValueError:
                return None
            # "nan" and "inf" ARE floats, and they are not bounds. `sample >= nan` is
            # False everywhere, so a typed "nan" emptied the fit sample and the pane
            # then blamed the column ("dwell_s is empty; nothing to fit") with nothing
            # to connect the refusal to the text the user had typed. Treated exactly
            # like "abc" -- ignored -- and said out loud by
            # :meth:`_fit_range_input_notes`.
            if not np.isfinite(v):
                return None
            # THE UNIT THE BOXES WERE TYPED UNDER, not whatever the combo says now.
            # The two agree -- a Time-unit change rebases these boxes exactly as it does
            # the axis-range ones -- and reading the recorded unit is what makes that
            # agreement unnecessary to KNOW here. Before the rebase covered these boxes,
            # a combo moved from ms to µs turned a typed [0.1, 2.0] into a window 1000x
            # too narrow and the fit refused with the boxes unchanged.
            return v / _display_factor(col, self._fit_range_unit)
        lo, hi = _read(self._fit_lo), _read(self._fit_hi)
        if lo is not None and hi is not None and lo >= hi:
            return None, None
        return lo, hi

    def _fit_range_input_notes(self) -> list[str]:
        """What the Fit range boxes CONTAIN but did not do. No silent no-ops.

        :meth:`_fit_range` drops a bound it cannot use -- unparseable text, ``nan``,
        ``inf`` -- and drops BOTH when they are inverted, and it did all of that in
        silence: the boxes went on showing the numbers while the fit behaved as if they
        were empty. Every one of those is a typed instruction the window declined to
        follow, so every one of them is said here.
        """
        out: list[str] = []
        for edit, side in ((self._fit_lo, "min"), (self._fit_hi, "max")):
            txt = edit.text().strip()
            if not txt:
                continue
            try:
                ok = np.isfinite(float(txt))
            except ValueError:
                ok = False
            if not ok:
                out.append(
                    f"Fit range {side} “{txt}” is not a usable bound and was IGNORED — "
                    f"the fit below is not restricted on that side. Type a finite "
                    f"number, or clear the box.")
        lo_txt, hi_txt = self._fit_lo.text().strip(), self._fit_hi.text().strip()
        if lo_txt and hi_txt and not out:
            try:
                if float(lo_txt) >= float(hi_txt):
                    out.append(
                        f"Fit range [{lo_txt}, {hi_txt}] is empty (min ≥ max), so BOTH "
                        f"bounds were IGNORED and the fit below covers the whole column.")
            except ValueError:      # pragma: no cover - handled above
                pass
        return out

    def _fit_range_blame(self, col: str, v) -> str:
        """Why a refusal may be the FIT RANGE's fault, said beside the refusal. ``""``
        when no range is typed.

        A fit refusal names the column -- "dwell_s has only 5 events; refusing to fit
        fewer than 20", "dwell_s is empty; nothing to fit" -- and that sentence is true
        of the sample the fitter was handed and false of the column, which the same pane
        is reporting as "A: n = 500" four lines above. The window already fixed the
        sibling of this in sweep 2 (a refusal must name the column the user actually
        fitted); the Fit range is the other half of the same sentence, and it is the
        half the reader who did NOT type it -- a reopened project, a colleague -- has no
        way to guess. The success path has said this all along (the FIT RANGE line);
        only the refusal path was silent.
        """
        lo, hi = self._fit_range(col)
        if lo is None and hi is None:
            return ""
        x = np.asarray(v, dtype=np.float64)
        x = x[np.isfinite(x)]
        keep = np.ones(x.shape, dtype=bool)
        if lo is not None:
            keep &= x >= lo
        if hi is not None:
            keep &= x <= hi
        f, unit = self._display_unit(col, _column_quantity(col)[1])
        shown_lo = "-inf" if lo is None else f"{lo * f:.12g}"
        shown_hi = "+inf" if hi is None else f"{hi * f:.12g}"
        unit_txt = f" {unit}" if unit else ""
        return (f"THE FIT RANGE IS WHY: [{shown_lo}, {shown_hi}]{unit_txt} is typed in "
                f"the Fit range boxes, and only {int(keep.sum())} of the {int(x.size)} "
                f"value(s) of “{col}” on this plot lie inside it — that subset is what "
                f"the fitter was given, and what the refusal above describes. Clear the "
                f"Fit range boxes to fit the whole column.")

    def _component_range(self) -> range:
        """Every component count the user could pick, from the spin itself.

        THE RECOMMENDER MUST SEARCH THE SET THE USER CAN CHOOSE FROM. It used to scan a
        hardcoded ``(1, 2, 3)`` while the spin accepted 1-4, so on the owner's -500 mV
        ``depth_pA`` it advised "3 components fit markedly better" and never mentioned
        that 4 was better still — BIC 7170.43 against 7198.43 (report Medium-10). Reading
        the range off the widget makes the two impossible to drift apart again.

        THE FOURTH RIVAL COSTS 80 ms AND IS WORTH IT. Measured on a 1000-event dwell
        sample through a real window: 85 ms per redraw scanning k = 1..3, 166 ms scanning
        k = 1..4, 17 ms with no scan at all. That is a real cost on the curation click
        `_better_component_count` was optimised for — and it buys the answer to the
        question the panel was already asking out loud. An adaptive search (only try 4 if
        3 won) would have been cheaper and would have re-opened exactly this defect: BIC
        was NON-monotonic in k on that same recording, so the best k is not reachable by
        walking uphill. The whole range, every time.
        """
        return range(int(self._fit_k.minimum()), int(self._fit_k.maximum()) + 1)

    def _auto_component_count(self, refit, k_used: int, fit) -> tuple[int, str]:
        """Pick k by BIC over :meth:`_component_range`; return ``(k, note)``.

        The clustering group's Auto (BIC) with the same contract: same sample, same
        censoring, lowest BIC wins, and every candidate's score is reported so the choice
        is auditable rather than magic. Ties keep the smaller k — the more parsimonious
        model — which is also what BIC's own penalty is expressing.

        Only the CHOICE comes back, never the rival FitResult: the rivals are fit inside
        `point_fits_only`, so they carry no confidence intervals, and two of the three
        identifiability tests are statements ABOUT those intervals. Handing one of them
        straight to the drawing code would silently strip the error bars and the "this
        component is not resolved" warnings off exactly the fits Auto mode selects. The
        caller re-fits the winner properly.
        """
        scores: list[tuple[int, float]] = [(int(k_used), float(fit.bic))]
        best_k, best_bic = int(k_used), float(fit.bic)
        for kk in self._component_range():
            if kk == k_used:
                continue
            try:
                # Single-writer filter swap (see stats.quiet_runtime_warnings): Auto
                # (BIC) runs on the GUI thread and an advisory batch may be inside its
                # own suppression on the worker at the same moment.
                with stats.quiet_runtime_warnings(), stats.point_fits_only():
                    rival = refit(kk)
                bic = float(rival.bic)
            except OperationCancelled:      # never "this rival failed"; see _compute_better_k
                raise
            except Exception:
                continue
            if not np.isfinite(bic):
                continue
            scores.append((int(kk), bic))
            if bic < best_bic - 1e-12 or (bic < best_bic + 1e-12 and kk < best_k):
                best_k, best_bic = int(kk), bic
        scores.sort()
        table = "  ".join(f"k={kk}: {b:.6g}" for kk, b in scores)
        return best_k, f"AUTO (BIC) chose k = {best_k}.  {table}"

    def _better_component_count(self, refit, k_used: int, fit, *, key=None) -> str:
        """Say when a DIFFERENT number of components describes the data much better.

        A single exponential is the default, and on a bimodal dwell distribution it returns
        a tau that describes neither mode -- the curve visibly misses the data while the
        readout states one authoritative number with no goodness-of-fit beside it. On the
        owner's -700 mV apo-ferritin run k=1 gave tau = 16.8 ms through a distribution with
        a 0.4 ms spike and an 18 ms mode; k=2 recovered both (0.40 ms at 9%, 18.3 ms at 91%)
        and improved BIC by 78.

        The app already COMPUTES aic/bic for every fit and never showed them. This does not
        change the fit -- the user's choice stands -- it just stops the panel from keeping
        the evidence against that choice to itself.

        THE RIVALS ARE POINT FITS. This method reads ``rival.bic`` and nothing else --
        no params, no CI, no meta -- and ``bic`` comes out of the likelihood maximisation,
        before any resampling, so it is bit-identical either way. Left to its default each
        rival also ran a 1000-replicate bootstrap for confidence intervals that are
        discarded on the next line: measured on the owner's ``_0006_Process_-600 mV.abf``
        (317 events, exponential fit), ONE reject click cost **3.40 s**, of which 3.38 s
        was two rivals' CIs. Asking for point fits takes the same click to **0.02 s** with
        every drawn and printed number unchanged.

        This matters because of C-1. While the standalone windows were wired to four
        signals the cost was paid on the user's own Refresh click; wiring them to the ten
        signals that actually change what they draw moved it onto every curation click,
        which is the click a curator makes hundreds of times in a row.
        """
        # THE SEARCH RANGE IS READ HERE, ON THE GUI THREAD, and passed down as plain
        # numbers. `_component_range()` reads the fit-k QSpinBox, and the ladder below
        # now runs on `_NoteWorker` -- reading a QWidget from a worker thread is
        # undefined behaviour that happens to work until it does not. Capturing it also
        # pins the range to the click the note is ABOUT: a user who changes the spin
        # while the scan is in flight gets a note for the range they had, not a
        # half-torn mixture of two.
        ks = tuple(self._component_range())
        if key is None:
            return self._compute_better_k(refit, k_used, fit, ks)
        ready, note = self._deferred_note(
            key, lambda: self._compute_better_k(refit, k_used, fit, ks))
        return note if ready else self._BETTER_K_PENDING_TEXT

    def _compute_better_k(self, refit, k_used: int, fit, ks: Sequence[int]) -> str:
        """The BIC ladder itself. Split from :meth:`_better_component_count` so the
        deferral in :meth:`_deferred_note` has a plain callable to run off the redraw
        path -- and, since that callable now runs on a WORKER THREAD, one that touches
        no widget: ``ks`` is :meth:`_component_range` read at the call site. The
        arithmetic below is unchanged.

        AND IT STOPS WHEN IT IS TOLD TO. This is one of the two long-running things the
        advisory worker runs, and it polled nothing: the only cancel poll in ``stats``
        was inside the bootstrap loops, which ``point_fits_only()`` switches off -- so
        the ladder always ran to completion. The panel aborts the worker on every
        curation click and joins it for 100 ms; a ladder that cannot stop turns that
        join into a fixed 100 ms tax AND leaves a GIL-holding worker running into the
        next click. Measured on the owner's -700 mV run at a 600 ms cadence, this is
        the difference between 177 ms and 90 ms per click on `exponential`, and between
        894 ms and the family's own point fit on `exponential + gamma`.

        The poll goes at the top of each rival, not only around the ladder: on
        `exponential + gamma` ONE rival is a 12-start optimisation, so a per-ladder
        poll would still overshoot the join by 4x. The fitters underneath poll too
        (see ``stats.fit_dwell_exp_gamma``), which is what brings the worst case down
        to one multi-start.
        """
        try:
            best_k, best_bic = k_used, float(fit.bic)
            # refit carries the SAME sample and the SAME censoring (t_min/t_max or the
            # depth floor). Comparing BIC across different data or different truncation
            # would be meaningless, and silently so.
            #
            # Suppressed for the SAME reason the primary fit suppresses it: when no
            # min_dwell_s is known the fit warns about the substituted floor, and the
            # note explaining that is already in `notes`. Re-raising it once per rival
            # would print the same sentence three times for one plot.
            #
            # ONE suppression for the whole ladder rather than one per rival:
            # identical suppression (the filter is the same on every pass), and
            # `warnings.filters` is PROCESS-global while this now runs on a worker, so
            # the fewer times it is swapped in and out beside a live GUI thread the
            # better. `stats.quiet_runtime_warnings()` is that swap made single-writer:
            # `warnings.catch_warnings()` has no lock of its own, and two threads
            # entering and leaving it concurrently restore each other's captured filter
            # lists. See the note in _NoteWorker.
            with stats.quiet_runtime_warnings(), stats.point_fits_only():
                for kk in ks:
                    if kk == k_used:
                        continue
                    check_cancelled()
                    rival = refit(kk)
                    if float(rival.bic) < best_bic - self._BIC_HINT:
                        best_k, best_bic = kk, float(rival.bic)
            if best_k == k_used:
                note = ""
            else:
                word = "component" if best_k == 1 else "components"
                note = (f"NOTE: {best_k} {word} fit markedly better "
                        f"(BIC {best_bic:.6g} vs {float(fit.bic):.6g}); "
                        f"set Fit components = {best_k} to see it.")
        # A CANCELLED LADDER IS NOT A LADDER THAT FOUND NOTHING. OperationCancelled is
        # a RuntimeError, so the blanket clause below would turn "stop, nobody is
        # looking at these events any more" into `note = ""` -- which `_NoteWorker`
        # would hand back as a real answer and the panel would CACHE, printing "no
        # better component count" for a sample the ladder never finished scanning.
        except OperationCancelled:
            raise
        except Exception:      # pragma: no cover - a hint must never break the plot
            return ""
        # Caching is the caller's: `_settle_notes` stores what it computed, keyed per
        # (sample, censoring, k), so the ladder is a function of the numbers and the
        # censoring alone and a redraw that changes neither never refits it.
        return note

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
        # two pooled clouds). The summary always lands in Details; it also becomes
        # the title unless the user pinned their own.
        titles = []
        for src, x, y, _index, _identities in series:
            t = self._overlay_scatter_fit(x, y, xcol, ycol, details,
                                          src if multi else None)
            if t:
                titles.append(f"{src.label}: {t}" if multi else t)
        title = "   |   ".join(titles)
        # ...and the same numbers ON the figure when the Style box asks for them.
        # The checkbox lived beside the FIT colour and worked only on the histogram,
        # so the page that owns the scatter-fit combo -- the one where a slope, an
        # intercept and an R² are exactly what a reader needs off the exported PNG --
        # was the one page where ticking it did nothing (owner note #13). The scatter
        # summary is already the full statement (equation, units, R² with its
        # degrees-of-freedom tag, n), so there is no second, richer form to build.
        self._draw_fit_values(titles)

        verdict = self._comparability(xcol, ycol)
        details.extend(self._column_notes([xcol, ycol]))
        details.extend(verdict.warnings)
        details.extend(self._detection_notes())

        self._plot.setLogMode(x=log_x, y=log_y)
        self._plot.setLabel("bottom", self._axis_label_for("bottom", xcol, log_x),
                            **self._label_style())
        self._plot.setLabel("left", self._axis_label_for("left", ycol, log_y),
                            **self._label_style())
        self._apply_title(title or None, accent=self._style.fit_color,
                          warn=self._warn_headline(self._stale_headline([xcol, ycol]),
                                                   verdict.headline))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(log_x, log_y)
        self._set_details("\n".join(details))

    def _overlay_scatter_fit(
        self, x: np.ndarray, y: np.ndarray, xcol: str, ycol: str, details: list[str],
        src: PlotSource | None = None,
    ) -> str | None:
        """Fit and draw the scatter best-fit line; return its title summary (or None).

        The line is sampled in DATA units across the plotted x-range and handed to
        the SAME viewbox as the points, so pyqtgraph's ``setLogMode`` renders it
        correctly on log axes. A ``populations`` guardrail (too few points, constant
        x, no positive values for a log-log fit) is caught and shown in Details --
        the scatter itself still stands. The summary is appended to ``details`` in
        every non-``none`` case (whether the fit succeeded or explained why it
        couldn't).

        ``src`` is passed only for an overlay, and it means the line belongs to THAT
        source alone: with several clouds on screen a single pooled regression would
        describe a population that does not exist."""
        if self._workspace_snapshot_rendering:
            if self._applied_scatter_fit != _SFIT_NONE:
                details.append(
                    "Scatter-fit configuration restored; press Plot to recompute the "
                    "regression. No fit was run while opening the project.")
            return None
        # The APPLIED selection, never the combo's (owner note #12) — see _apply_fit.
        kind = self._applied_scatter_fit
        if kind == _SFIT_NONE:
            return None
        who = f"{src.label}: " if src is not None else ""
        log_log = kind == _SFIT_POWER
        try:
            fit = populations.linear_fit(
                x, y, log_log=log_log, xlabel=xcol, ylabel=ycol)
            if log_log:
                # THE LINE SPANS THE POINTS THE FIT USED. `linear_fit(log_log=True)`
                # drops every pair without a logarithm (x <= 0, y <= 0, non-finite)
                # by design and fits the rest -- but this method used to hand
                # `line_points` min/max over ALL plotted x, so one zero rise_s made
                # it refuse with "Fit unavailable: log-log line_points needs xmin,
                # xmax > 0" -- a false claim (the fit succeeded) quoting an internal
                # helper. Same mask as the fitter's, so the drawn range and `fit.n`
                # describe the same points; the exclusion is SAID below, not silent.
                good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                dropped = int(x.size) - int(np.count_nonzero(good))
                if dropped:
                    details.append(
                        f"{who}power-law fit: {dropped} of {x.size} point(s) have "
                        f"x ≤ 0, y ≤ 0, or a non-finite value — no logarithm exists "
                        f"for them, so they are excluded from the fit and the line is "
                        f"drawn over the remaining {fit.n} points' x-range.")
                lx, ly = populations.line_points(
                    fit, float(np.min(x[good])), float(np.max(x[good])))
            else:
                lx, ly = populations.line_points(
                    fit, float(np.min(x)), float(np.max(x)))
        except ValueError as exc:
            details.append(f"{who}Fit unavailable: {exc}")
            return None

        pen = (pg.mkPen(src.colour, width=max(self._style.line_width, 1) + 1,
                        style=Qt.DashLine)
               if src is not None else self._scatter_fit_pen())
        item = pg.PlotDataItem(x=lx, y=ly, pen=pen)
        self._plot.addItem(item)
        self._scatter_fit_items.append(item)
        if self._scatter_fit_item is None:
            self._scatter_fit_item = item

        # THE PRINTED EQUATION IS IN THE AXES' OWN DISPLAY UNITS, through the same
        # :meth:`_display_unit` rule the exponential's tau and the mixture's mu
        # already go through. The fit runs on the STORED values — seconds for the
        # time columns — while the axis under it shows ms/µs via a tick multiplier,
        # so "y = 2000·x + 5" over "dwell (ms)" handed the reader the per-SECOND
        # slope with nothing on screen to say so: plugging axis values into it is
        # the 1000x error waiting to be copied into a paper. Slope and intercept
        # rescale as slope·fy/fx and intercept·fy; the power law's coefficient goes
        # as fy/fx^b (its x-unit dependence carries the exponent) while b itself is
        # dimensionless and untouched. The units are also SAID, because an equation
        # that is only silently consistent with its axes is one axis-relabel away
        # from being wrong again.
        fx, x_unit = self._display_unit(xcol, _column_quantity(xcol)[1])
        fy, y_unit = self._display_unit(ycol, _column_quantity(ycol)[1])
        units = ", ".join(p for p in (f"x in {x_unit}" if x_unit else "",
                                      f"y in {y_unit}" if y_unit else "") if p)
        unit_txt = f"   ({units})" if units else ""
        # NEVER let a non-finite R² reach a bare format spec (the PSD panel's _fmt_q
        # rule). `linear_fit` documents r2 = nan for exactly one cause -- the response
        # has no variance (SStot == 0) -- and "R² = nan" in a title reads as a
        # computation failure, or worse gets pasted into a figure caption. Say the
        # cause instead: a horizontal cloud is described perfectly and explained not
        # at all by a line, so there is no goodness-of-fit to report.
        # ...AND THE MIRROR CASE, through the SAME rule the voltage-dependence panel
        # uses (:func:`nanoscope.analysis.stats.r2_dof_tag`): a 2-parameter line
        # through 2 points has zero residual degrees of freedom, so R² = 1.000 is
        # arithmetic and not a measurement -- printed bare in this window's title and
        # legend while its own histogram refuses the identical two events as
        # "decoration, not a measurement". Both lines this method fits (straight and
        # log-log) have exactly two free parameters.
        _dof_tag = stats.r2_dof_tag(fit.n, _SCATTER_FIT_PARAMS)
        # THE SWITCH TURNS OFF A STATISTIC, NEVER A REFUSAL (StyleSettings.show_r2). Both
        # branches above this one are refusals -- "y does not vary" and the dof tag are
        # the sentences that say this R² is arithmetic rather than evidence -- and they
        # print whatever the box is set to. Only the plain, meaningful number is optional,
        # which is the one a clean paper figure has a reason to omit.
        if not np.isfinite(fit.r2):
            r2_txt = "R² undefined (y does not vary)"
        elif _dof_tag:
            r2_txt = f"R² = {fit.r2:.3f} ({_dof_tag})"
        elif self._style.show_r2:
            r2_txt = f"R² = {fit.r2:.3f}"
        else:
            r2_txt = ""
        # Joined rather than interpolated, so switching R² off leaves one gap and not
        # three spaces in the middle of an exported caption.
        if fit.log_log:
            coeff = fit.meta["coefficient"] * fy / fx ** fit.meta["exponent"]
            head = f"y = {coeff:.4g}·x^{fit.meta['exponent']:.4g}{unit_txt}"
        else:
            head = f"y = {fit.slope * fy / fx:.4g}·x + {fit.intercept * fy:.4g}{unit_txt}"
        summary = "   ".join(p for p in (head, r2_txt, f"(n = {fit.n})") if p)
        if src is not None:
            # By the FIGURE's name for the source (`_legend_name`, the one resolver),
            # as the series entry above it is: with two rows typed to "Apo" the
            # series read "Apo — -700 mV" / "Apo — -600 mV" and these two fit entries
            # read "Apo fit: …" twice in the same legend (A1-N2).
            self._legend_add(item, f"{self._legend_name(src)} fit: {summary}",
                             kind="fit", key=src.key)
        details.append(f"{who}{summary}")
        if not np.isfinite(fit.r2):
            details.append(
                f"{who}R² is undefined here because the fitted y values have no "
                f"variance: a flat response is matched exactly by its own mean, so a "
                f"line explains nothing and no goodness-of-fit exists to report.")
        else:
            dof_note = stats.r2_dof_caveat(
                fit.n, _SCATTER_FIT_PARAMS,
                model="log-log line" if fit.log_log else "straight line")
            if dof_note:
                details.append(f"{who}{dof_note}")
        return summary

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
            colour = QtGui.QColor(src.colour if multi else self._style.data_color)
            pen = pg.mkPen(colour, width=self._style.line_width)
            zmax = float(z.max())
            proxy = None
            for lv in np.linspace(0.05 * zmax, 0.95 * zmax, n_levels):
                curve = pg.IsocurveItem(data=z, level=float(lv), pen=pen)
                curve.setTransform(tr)
                self._plot.addItem(curve, ignoreBounds=False)
                proxy = proxy or curve
            details.append(f"{src.label}: n = {xs.size}, "
                           f"bandwidth factor = {kde.factor:.3f} (Scott)")
            if multi and proxy is not None:
                # The legend needs an item with a pen; the first contour serves.
                sample = pg.PlotDataItem(pen=pen)
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
        primary = self.primary_source()
        if primary is not None:
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
                            self._style.x_label.strip() or f"PC1 ({self._evr_pct(evr[0])})",
                            **self._label_style())
        self._plot.setLabel("left",
                            self._style.y_label.strip()
                            or (f"PC2 ({self._evr_pct(evr[1])})" if evr.size > 1 else "PC2"),
                            **self._label_style())
        # PCA runs on ONE source, so there is nothing pooled to judge -- but that
        # source's own detection caveats belong here as much as anywhere else.
        primary = self.primary_source()
        one = [primary] if primary is not None else []
        details.extend(self._column_notes(list(res.columns), one))
        details.extend(self._detection_notes(one))
        self._apply_title(self._explained_variance_text(res),
                          warn=self._warn_headline(
                              self._stale_headline(list(res.columns), one)))
        self._apply_background()
        self._apply_fonts()
        self._finalize_view(False, False)
        self._set_details("\n".join(details))

    def _legend_clusters(self, labels: np.ndarray) -> None:
        """One legend swatch per cluster present, in cluster order (dropped last).

        A cluster-coloured cloud is an overlay of populations in one colour channel,
        so it needs naming exactly as a multi-source overlay does. The swatches are
        zero-point scatter items: they exist to be legend samples, never to draw."""
        for cid, n in category_counts(labels):
            # legend_swatch's body, not a copy of it: this used to hand-write the
            # identical ScatterPlotItem call, so a pen or alpha added to the shared
            # helper would have changed the report window's swatches and silently
            # left this panel's alone -- two pictures drifting apart one colour at
            # a time, which is the failure `categorical` exists to prevent.
            swatch = legend_swatch(cid, size=self._style.point_size)
            name = "dropped (non-finite features)" if cid < 0 else f"cluster {cid}"
            self._legend_add(swatch, self._legend_entry(name, n),
                             kind="cluster", key=str(name))

    @staticmethod
    def _evr_pct(ratio) -> str:
        """One explained-variance percentage, or the refusal that replaces it.

        NEVER a bare format spec on a value that can be non-finite -- the same rule
        the scatter's R² already follows two hundred lines above ("R² undefined (y
        does not vary)") and the PSD panel's ``_fmt_q``. ``explained_variance_ratio_``
        is ``explained_variance_ / total_var``; with every selected feature constant
        that denominator is zero and sklearn hands back ``nan``, which this window
        printed verbatim into its title, both axis labels, Details and an exported
        PNG. :func:`populations.pca` now refuses that input outright, so this is the
        second line of the same defence rather than the only one -- and the one that
        holds if a future solver produces a non-finite ratio for a reason nobody has
        thought of yet.
        """
        r = float(ratio)
        return f"{r * 100:.1f}%" if np.isfinite(r) else "undefined"

    @staticmethod
    def _pca_dof_tag(res: populations.PCAResult) -> str:
        """"exact by construction", when the cumulative percentage cannot be anything else.

        THE SAME DOF HONESTY THE SCATTER'S R² ALREADY GETS
        (:func:`nanoscope.analysis.stats.r2_dof_tag`). Centred data of ``n`` rows has
        rank at most ``n - 1``, and a projection onto ``p`` features has rank at most
        ``p``; when the number of components shown reaches that rank the components
        span the whole space and the cumulative share is 100.0% for ANY data
        whatsoever. Six pure-noise tables at n = 2 and n = 3 all reported "cumulative
        100.0%" with nothing beside it, in the window whose scatter, on the same two
        rows, prints "R² = 1.000 (exact by construction)".
        """
        rank = int(res.meta.get("max_rank", 0) or 0)
        shown = min(int(res.explained_variance_ratio.size), 2)
        return "exact by construction" if rank and shown >= rank else ""

    @classmethod
    def _explained_variance_text(cls, res: populations.PCAResult) -> str:
        evr = res.explained_variance_ratio
        parts = [f"PC{i + 1} {cls._evr_pct(r)}" for i, r in enumerate(evr[:2])]
        cum = float(np.sum(evr[:2]))
        tag = cls._pca_dof_tag(res)
        return (f"Explained variance: {', '.join(parts)}  "
                f"(cumulative {cls._evr_pct(cum)}{'; ' + tag if tag else ''})")

    @staticmethod
    def _pca_dof_caveat(res: populations.PCAResult) -> str:
        """The full sentence behind :meth:`_pca_dof_tag`, for Details. ``""`` when moot."""
        rank = int(res.meta.get("max_rank", 0) or 0)
        shown = min(int(res.explained_variance_ratio.size), 2)
        if not rank or shown < rank:
            return ""
        n, p = int(res.n_kept), len(res.columns)
        why = (f"{n} rows centred on their own mean span at most {n - 1} dimension"
               f"{'' if n - 1 == 1 else 's'}" if (n - 1) <= p
               else f"{p} features span at most {p} dimensions")
        return (f"CUMULATIVE 100% HERE IS EXACT BY CONSTRUCTION, not a finding: {why}, "
                f"and {shown} component{'' if shown == 1 else 's'} already cover"
                f"{'s' if shown == 1 else ''} that space — so ANY {n} rows whatsoever, "
                f"noise included, would report the same cumulative share. Read the "
                f"per-component split as arithmetic, not as evidence that these "
                f"features are collinear.")

    # -- clustering / PCA runs -------------------------------------------------------------------
    def run_clustering(self) -> populations.ClusterResult | None:
        """Cluster the current rows over the selected features and recolour the view."""
        df = self._current_frame()
        if df is None or df.shape[0] == 0:
            self._message("No rows to cluster.")
            return None
        features = self.feature_selection()
        if len(features) < 2:
            self._set_details("Clustering needs at least 2 feature columns selected.")
            return None
        method = _METHOD_IDS[self._method_combo.currentText()]
        if method == "gmm" and self._k_mode.currentText() == _K_AUTO:
            k: int | None = None
        else:
            k = self._k_spin.value()
        try:
            res = populations.cluster(df, features, method=method, k=k)
        except (ValueError, RuntimeError) as exc:
            self._set_details(f"Clustering failed: {exc}")
            return None

        self._cluster_result = res
        # Make the grouping visible on EVERY view that can carry it: the scatter (and
        # PCA) switch to cluster colouring, and so does the histogram — a clustering
        # that could be seen in one view and not in another was the missing half of
        # this workflow. The histogram split stays off while several sources are
        # overlaid, where colour already means source.
        self._updating = True
        self._colour_by.setCurrentText(_COLOUR_CLUSTER)
        if self._hist_colour_by.isEnabled():
            self._hist_colour_by.setCurrentText(_COLOUR_CLUSTER)
        self._updating = False
        self._armed_redraw()   # "Run clustering" is a run gesture; it clears the gate too
        self._set_details(self._cluster_summary())
        # The result itself is durable v11 state even when the controls already had
        # these values (for example, Run clustering pressed a second time).
        self.workspaceChanged.emit()
        return res

    def run_pca(self) -> populations.PCAResult | None:
        """Project the selected features to 2 components and show the scatter."""
        df = self._current_frame()
        if df is None or df.shape[0] == 0:
            self._message("No rows for PCA.")
            return None
        features = self.feature_selection()
        if len(features) < 2:
            self._set_details("PCA needs at least 2 feature columns selected.")
            return None
        try:
            # Project v11 records the seed with the PCA snapshot.  Use an explicit
            # integer here so a deliberate rerun has a reproducible recipe instead of
            # persisting ``None`` (which delegates randomness to library defaults).
            res = populations.pca(df, features, n_components=2, random_state=0)
        except (ValueError, RuntimeError) as exc:
            self._set_details(f"PCA failed: {exc}")
            return None

        self._pca_result = res
        self._updating = True
        self._plot_type_combo.setCurrentText(PCA)
        self._pages.setCurrentIndex(_PLOT_TYPES.index(PCA))
        self._updating = False
        self._armed_redraw()   # "Run PCA" is a run gesture; it clears the gate too
        self.workspaceChanged.emit()
        return res

    def _cluster_summary(self) -> str:
        res = self._cluster_result
        if res is None:
            return ""
        auto = res.meta.get("requested_k") is None and res.method == "gmm"
        head = (f"Clustering: {res.method}, k = {res.chosen_k}"
                f"{' (auto by BIC)' if auto else ''}, n = {res.n_kept} kept")
        dropped = int((res.labels == populations.DROPPED_LABEL).sum())
        if dropped:
            head += f", {dropped} dropped (non-finite features)"
        sizes = "sizes: " + ", ".join(f"{cid}:{n}" for cid, n in sorted(res.sizes.items()))
        means = "means (original units):\n" + res.means.to_string(float_format=lambda x: f"{x:.4g}")
        parts = [head, sizes, means]
        # The caveats FIRST among the trailing lines, and never silently: this readout used
        # to print "sizes: 0:189, 1:1" with no comment that a cluster of one event is not a
        # population. See populations.ClusterResult.warnings.
        for note in getattr(res, "warnings", ()):
            parts.append("! " + str(note))
        return "\n".join(parts)

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
    # AND WHY PDF IS NOT LEFT TO ``Analysis ▸ Export figure…``. That command renders
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

        With ``path`` None a file dialog is shown; returns the written path, or
        None when the user cancelled OR the write failed (a failed write is
        reported in a dialog, never returned as a path). The Style group's export
        width/height pin the output size (0 = native): a width alone lets the
        height follow the aspect ratio; both set forces exactly that pixel size.
        They apply to the PNG's pixels and to the PDF's page (1 pt = 1 canvas px);
        SVG has no size of its own to pin, and is written at the canvas's.

        Anything that is not ``.svg`` or ``.pdf`` goes to ``ImageExporter``, which is
        what it was before -- so a ``.tif``/``.jpg`` name still works, and a format
        QImage cannot write still fails into ``image_write_failure_text`` with the
        user's previous file untouched.
        """
        if path is None:
            chosen = ask_export_path(
                self, title="Save plot image", suggestion="plot.png",
                filters=self._IMAGE_FILTERS)   # writes exactly this file
            if chosen is None:
                return None
            path = str(chosen)
        out = Path(path)
        suffix = out.suffix.lower()
        w, h = self._style.export_w, self._style.export_h

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

        if suffix == ".svg":
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
                return pg_svg.export_svg(self._plot, staged)
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
                return exporter.export(str(staged))

        try:
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

    # -- export: the numbers behind the picture --------------------------------------------------
    def _export_series_labels(self) -> tuple[list[str], list[str]]:
        """``(series names, fit names)`` for the current picture, in DRAW order.

        Taken from ``_legend_offered``, which the draw fills as it adds each item and
        which holds the FULL label -- not the elided text that goes onto the legend
        (``_LEGEND_LABEL_CHARS`` = 34 turns a recording filename into
        "500 pM Apo ferri…-500 mV.csv", which is not a name a data file should carry),
        and not nothing when the user has unticked *Show legend*: ``_legend_add``
        records the offer before either gate. So the exported ``series`` column is what
        the figure calls that series, whether or not the figure is currently showing it.

        The fit entries are separated by their entry KIND rather than by position,
        because a fit that refused (too few events, an empty range) leaves a gap in
        the fit sequence but not in the series one.

        A ``cluster`` ENTRY IS A SERIES ON ONE PAGE AND A SWATCH ON TWO, and the
        difference is not cosmetic -- getting it wrong writes a mislabelled file.
        ``_draw_histogram``'s cluster split really does draw one histogram per cluster
        and offers each bar set under ``kind="cluster"``, so there the cluster names
        ARE the series names. ``_draw_scatter`` and ``_draw_pca`` draw ONE
        ScatterPlotItem with a brush per point and offer ``_legend_clusters``' swatches
        -- items that exist only to sit in the legend and never to draw. Treating those
        as series labelled 200 points of a two-cluster scatter "cluster 0 (n = 100)"
        (measured). On those two pages the series is the SOURCE and the cluster is a
        per-point column; see :meth:`_cluster_of_drawn_points`.

        FALLBACK: the KDE draw offers legend entries only when several sources are
        overlaid, and a single-source scatter offers none at all, so nothing is
        recorded. The series is then the drawn source, which is what the checked-source
        list already says.
        """
        series: list[str] = []
        fits: list[str] = []
        swatches_only = self._plot_type() in (SCATTER, PCA)
        for entry_id, label in self._legend_offered.items():
            if entry_id.startswith("fit:"):
                fits.append(str(label))
            elif entry_id.startswith("cluster:") and swatches_only:
                continue                       # a legend swatch, not a drawn series
            else:
                series.append(str(label))
        if not series:
            # The figure's name for each source, not the raw override -- the one
            # resolver every writer that names a source on or beside the figure reads.
            series = [self._legend_name(src) for src, _df in self._drawn_frames()]
        return series, fits

    def _cluster_of_drawn_points(self) -> np.ndarray | None:
        """The cluster id behind every drawn marker, in draw order — or None.

        READ OFF THE BRUSHES THE DRAW SET, not recomputed from the frame. The draw
        derives them with ``self._cluster_result.labels.reindex(index)`` over an index
        it builds and discards (the finite/positive mask), so the only ways to get them
        back are to re-derive that mask -- a second copy of the draw's rule, free to
        drift from it silently -- or to read the picture. The colour IS what the figure
        shows, and ``categorical.category_colour`` is the same function that put it
        there, so the mapping cannot disagree with the legend beside it.

        An unrecognised colour becomes NaN rather than a guess: a fabricated cluster id
        in a CSV is worse than an admitted gap.

        THE GATE IS THE OFFERED SWATCHES, NOT ``self._cluster_result``. A clustering
        stays on the window after the user switches *Colour by* back to None or ticks a
        second source, and both of those repaint the markers from the SOURCE palette --
        which is Okabe-Ito, the same palette ``category_colour`` draws from, so a
        colour lookup would match and stamp every row with a cluster the picture is not
        showing. ``_legend_clusters`` is the only producer of ``cluster:`` entries on
        these two pages and it runs only when the markers really are cluster-coloured,
        so its offers are the exact condition.
        """
        if self._cluster_result is None or not self._scatter_items:
            return None
        if not any(k.startswith("cluster:") for k in self._legend_offered):
            return None
        labels = np.asarray(self._cluster_result.labels, dtype=int)
        by_rgb: dict[int, int] = {}
        for cid in set(labels.tolist()) | {populations.DROPPED_LABEL}:
            # ``rgb()`` drops the alpha, which ``category_brushes`` is free to vary.
            by_rgb[QtGui.QColor(category_colour(int(cid))).rgb()] = int(cid)
        out: list[float] = []
        for item in self._scatter_items:
            brushes = item.data["brush"]
            for brush in brushes:
                cid = by_rgb.get(QtGui.QColor(brush.color()).rgb())
                out.append(np.nan if cid is None else float(cid))
        return np.asarray(out, dtype=float)

    def _plot_data(self) -> pd.DataFrame:
        """The numbers behind the plot on screen, as the frame :meth:`export_data` writes.

        Split out from the writer so a test -- and a future *Copy data* -- can ask for
        the table without a file dialog anywhere near it.

        WHAT EACH PLOT TYPE CONTRIBUTES, and why that is enough to redraw it:

        * **Histogram** -- one ``bins`` row per bar per series, carrying
          ``bin_left``/``bin_right`` and the bar's DRAWN height, so a reader re-plots
          the exact bars rather than re-binning and hoping. The height is in whatever
          the *y* combo says (count / density / probability), and the y column is NAMED
          that, because a density and a count are different numbers from the same bars.
          A fitted curve adds ``fit`` rows sampled where it was drawn -- already scaled
          to the same y quantity, so the curve and the bars are directly comparable.
        * **Scatter** -- one ``points`` row per plotted marker, which is the figure
          exactly: the rows the draw dropped (non-finite, or non-positive under a log
          axis) are absent here for the same reason they are absent from the picture,
          and Details says how many they were. A best-fit line adds its ``fit`` samples.
        * **KDE** -- the ``grid`` the contours were cut from (128x128 densities per
          source, in data coordinates) plus one ``level`` row per contour. Contour
          VERTICES would reproduce only the levels the user happened to choose; the
          grid re-contours at any level, in any tool, and is what the density actually
          is.
        * **PCA** -- one ``points`` row per score, plus a ``loading`` row per feature
          (its weight on PC1 and PC2). Without the loadings a PCA scatter is a cloud
          nobody can interpret: the axes are combinations of columns, and the CSV would
          not say which.

        Empty (no rows) when nothing is drawn -- a message, a refusal, an empty frame.
        """
        from ..plot.figures import plot_source_data

        identity = self.config().identity
        kind = identity.plot_type
        log_x, log_y = self._current_log_state()
        if kind == HISTOGRAM:
            # The y axis of a histogram is not a column of the events table; it is what
            # the normalization combo made of the counts, and ``_y_label`` is the same
            # function that names the QUANTITY on the plot.
            #
            # `_y_label`, NOT `_y_axis_label`, AND THAT IS DELIBERATE (S6-03). The picture
            # needs its unit spelled out because its x TICKS are re-scaled for display; a
            # CSV has no display scale, and its x column is the schema's own unit-bearing
            # name -- `dwell_s`, `depth_pA` -- so a column called `density` beside `dwell_s`
            # is per-second by construction. Writing `density (1/s)` here would rename a
            # user-facing column for no gain, break
            # `test_the_height_column_is_named_for_the_quantity_it_holds`, and disturb
            # `plot_source_data`'s `density_name in (x_name, y_name)` guard, whose own
            # comment names this exact histogram case. Keeping BOTH the values and this name
            # fixed is also what lets the whole export path stay untouched by S6-03:
            # measured, the exported bars still integrate to 1.0000000000000053 in stored
            # units, in every time unit, before and after.
            x_name, y_name = identity.x_column or "x", self._y_label()
        elif kind == PCA:
            x_name, y_name = "PC1", "PC2"
        else:
            x_name, y_name = identity.x_column or "x", identity.y_column or "y"
        if x_name == y_name:
            # A scatter of a column against itself is legal and draws a diagonal line;
            # two identical column names would silently collapse into one in the frame.
            y_name = f"{y_name}_y"
        series, fits = self._export_series_labels()
        df = plot_source_data(self._plot, x_name=x_name, y_name=y_name,
                              log_x=log_x, log_y=log_y,
                              series_labels=series, fit_labels=fits)
        if kind in (SCATTER, PCA):
            # THE COLOUR CHANNEL IS DATA HERE. On these two pages a clustering is drawn
            # as one item with a brush per point, so without this column the CSV would
            # carry the cloud and lose the very partition the picture is about.
            df = self._with_cluster_column(df)
        if kind == PCA and len(df) and self._pca_result is not None:
            df = pd.concat([df, self._pca_loading_rows(x_name, y_name)],
                           ignore_index=True)
        return df

    def _with_cluster_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``cluster`` column to the ``points`` rows of *df*, when there is one.

        Nullable ``Int64``, so an uncoloured row is ``<NA>`` in the CSV and never a 0
        that would read as "cluster 0". Added only when the count matches the drawn
        points exactly -- a partial alignment would put the right ids against the wrong
        markers, which is the one outcome worse than no column at all.
        """
        if not len(df):
            return df
        ids = self._cluster_of_drawn_points()
        if ids is None:
            return df
        mask = (df["part"] == "points").to_numpy()
        if int(mask.sum()) != ids.size:
            return df
        column = np.full(len(df), np.nan)
        column[mask] = ids
        return df.assign(cluster=pd.Series(column, index=df.index).astype("Int64"))

    def _pca_loading_rows(self, x_name: str, y_name: str) -> pd.DataFrame:
        """One row per feature: its loading on PC1 and on PC2.

        ``PCAResult.components`` is ``(n_components, n_features)``, so a component is a
        ROW of it and a feature is a column -- transposed here rather than in the
        reader's head. A one-component result (two features, one of them constant) has
        no PC2 to report, and gets NaN there rather than a fabricated zero.
        """
        res = self._pca_result
        comps = np.asarray(res.components, dtype=float)
        pc1 = comps[0] if comps.shape[0] > 0 else np.full(len(res.columns), np.nan)
        pc2 = comps[1] if comps.shape[0] > 1 else np.full(len(res.columns), np.nan)
        return pd.DataFrame({
            "series": [str(c) for c in res.columns],
            "part": "loading",
            x_name: pc1,
            y_name: pc2,
        })

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
