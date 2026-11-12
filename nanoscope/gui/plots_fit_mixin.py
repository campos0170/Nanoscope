"""The fit half of the Plots window: overlays, summaries, fit notes, fit range, component count.

``_PlotsFitMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`, listed
before ``QtWidgets.QMainWindow`` in its bases. It holds the histogram and scatter fit
overlays (``_overlay_fit``, ``_overlay_scatter_fit``), the one-line and full fit summaries
and the jobs that compute the bootstrap CI and the cross-family table, the goodness and
rival-family notes, the fit-range boxes, the component-count ladder and the Nonlinear fit
diagnostics hook.

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._style``, ``self._fit_combo`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace as _dc_replace
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from ..analysis import modules, populations, stats
from ..analysis._compat import trapezoid as _trapezoid
from ..analysis.cancel import OperationCancelled, check_cancelled
from ..core.sources import PlotSource
from ..plot.style import PALETTE
from .fit_label import FitReadout
from .plots_constants import (
    _DWELL_COLUMNS,
    _FIT_EXP,
    _FIT_GMM,
    _FIT_GAMMA,
    _FIT_INVGAUSS,
    _FIT_EXP_GAMMA,
    _MULTI_COMPONENT_FITS,
    _TIME_COLUMNS,
    _TIME_UNITS,
    _SCATTER_FIT_PARAMS,
    _SFIT_NONE,
    _SFIT_POWER,
    _display_factor,
    _param_symbol,
    _KS_REJECTED_TAG,
    _FitSummary,
    _column_quantity,
)


class _PlotsFitMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    @staticmethod
    def _warn_headline(*parts: str) -> str | None:
        """Join diagnostic headlines for Details, worst-first, or None."""
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

        The string summary includes the diagnostic tag for Details; ``.plain`` is
        the title/legend summary. ``.full`` supplies every fitted parameter with its
        unit, n, BIC and the numerical KS statistic (the old on-plot values box's text).
        Warnings and verdicts remain in Details and saved fit metadata.

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
        range_errors = self._fit_range_input_notes()
        if range_errors:
            raise ValueError(" ".join(range_errors))
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
            ci_jobs = list(getattr(payload, "_gui_deferred_ci_jobs", ()))
            if hasattr(fit, "_gui_deferred_ci_jobs"):
                del fit._gui_deferred_ci_jobs
            ci_key = ("ci-expgamma", *fit_key[1:], 200)
            _refit = lambda kk, _s=sample, _lo=lo, _hi=hi, _kw=dict(dwell_kw): (   # noqa: E731
                fit_dwell_exp_gamma(_s, t_min=_lo, t_max=_hi, n_boot=200, **_kw))
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
            # Condition the Gaussian magnitude likelihood on the selected range.
            # An asymmetric crop across both signs has no single magnitude-selection
            # law; require a magnitude column instead of inventing one.
            _glo, _ghi = _crop_lo, _crop_hi
            if _neg and not _all_neg and (_crop_lo is not None or _crop_hi is not None):
                raise ValueError("A ranged Gaussian fit of mixed-sign values needs an explicit magnitude column; the signed crop is not a single truncation of |x|.")
            if _all_neg:
                _glo = None if _crop_hi is None else -float(_crop_hi)
                _ghi = None if _crop_lo is None else -float(_crop_lo)
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
                                                 quantity=_q, unit=_u, display=_d,
                                                 lower=_glo, upper=_ghi))
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
                 int(kk), _q, _u, _dsp, _glo, _ghi))
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
            sample_for_gof, lo_for_gof, hi_for_gof = np.abs(source), _glo, _ghi
            trunc_lo, trunc_hi, trunc_modelled = _crop_lo, _crop_hi, True

        # THE TRUNCATION THE LIKELIHOOD ACTUALLY GOT. For every censored family that is
        # exactly the pair handed to the fitter -- the same pair the goodness-of-fit test
        # and the rival ladder are given, which is what keeps all three describing one
        # fit. Recorded here, once, rather than re-derived by the note that prints it.
        if trunc_modelled and kind != _FIT_GMM:
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
        # ...and with the typed Fit range, which the fit carries from here on: into a
        # project's workspace and back, where the boxes are only the boxes. A ranged fit is
        # not filed (`_persist_results`) and not drawn by the report (S10-LIN3). `fit` is
        # this draw's own object (a cached one is a deep copy), so the stamp is too.
        if fit_lo is not None or fit_hi is not None:
            fit.meta[modules.FIT_RANGE_KEY] = [fit_lo, fit_hi]

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
        item._presentation_role = "category" if who else "fit"
        self._tag_series(item, "fit", src.key if src is not None else "fit",
                         f"{src.label if src is not None else 'Data'} fit", colour=colour if who else None)
        item._presentation_width_extra = 1

        # -- the two optional pictures of the fit's own quality ------------------------
        # THE SAME y FACTOR THE CURVE USED, extracted rather than recomputed: band and
        # curve have to agree bin for bin under three normalizations and a log x axis,
        # and one expression they both multiply by is the only way to guarantee it.
        y_scale = float(fit.n) * width * self._fit_scale(edges, width, int(v.size))
        resid_short = ""
        band_note = self._draw_ci_band(fit, grid, at, y_scale, colour,
                                       role=item._presentation_role)
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
        if (grid_of is None and sample_for_gof is not None
                and not self._presentation_residual_ready):
            resid_note, resid_short = self._draw_residual_strip(
                fit, sample_for_gof, edges, log_x)
            if who and self._presentation_residual_note:
                self._presentation_residual_note = (
                    f"RESIDUAL STRIP: {self._legend_name(src)} only — there is one "
                    "axis under the plot and this overlay has one fit per source. "
                    + self._presentation_residual_note)
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
        # Keep the same goodness-of-fit calculation and full verdict in Details.
        # The title and legend use the untagged parameter summary.
        gof = self._goodness_note(fit, sample_for_gof)
        plain_summary = summary
        ks_rejected = bool(gof) and not gof.get("describes_data")
        if ks_rejected:
            summary = f"{summary} — {_KS_REJECTED_TAG}"
        legend_who = self._legend_name(src) if (who and src is not None) else who
        item._presentation_fit_family = kind
        item._presentation_fit_named = bool(legend_who)
        self._legend_add(item, self._fit_legend_label(
                             item, legend_who, src.key if src is not None else ""),
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
            # Said where the range is: closing the window files a fit for the report,
            # and this one is not (see `_persist_results`). What the report draws
            # instead is a whole-column fit already saved for the table, when it would
            # take that one, or a fresh fit.
            kept = ""
            if any(col in cols for cols, _mid in self._FIT_MODULE_IDS):
                kept = (" This fit is not kept for Generate Report figures, which "
                        + ("draw the whole-column fit already saved for this table."
                           if self._report_takes_saved_fit(src, col) else
                           "fit the whole column afresh."))
            notes.append(
                f"FIT RANGE [{shown_lo}, {shown_hi}] {unit} — n = {fit.n} of the "
                f"available events; {how}{narrowed}{kept}")
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
        variants = {show_residuals: self._full_fit_summary(
            fit, col, gof, resid_short if show_residuals else "") for show_residuals in (False, True)}
        self._presentation_full_variants.append((who, variants))
        result = _FitSummary(summary, variants[self._style.show_residuals],
                             rejected=ks_rejected, plain=plain_summary)
        result.parts = self._fit_parts(fit, col, gof, resid_short,
                                       headline=f"{names[1]} = {shown}{unit_txt}",
                                       headline_keys=keys)
        return result, notes

    def _fit_parts(self, fit, col: str, gof: dict | None, resid_short: str = "", *,
                   headline: str = "", headline_keys=()) -> dict:
        """The fit label's tokens for a histogram fit: the headline the title used to
        carry (its symbol names the frame -- ``μ(|mean|)`` on a reflected column), then
        every other parameter in the axis's unit (the :meth:`_display_unit` rule of
        :meth:`_full_fit_summary`), n, BIC, the numerical KS statistic and the residual
        strip's number. The KS VERDICT is not here: it stays in Details, not in artwork."""
        params = [headline] if headline else []
        for key, value in fit.params.items():
            if headline and key in headline_keys:
                continue
            scale, uname = self._display_unit(col, fit.unit_of(key))
            params.append(f"{_param_symbol(key)} = {float(value) * scale:.4g}"
                          + (f" {uname}" if uname else ""))
        parts = {"equation": ", ".join(params), "n": f"n = {fit.n}"}
        if np.isfinite(float(fit.bic)):
            parts["bic"] = f"BIC = {float(fit.bic):.6g}"
        if gof:
            parts["ks"] = f"KS = {float(gof['ks']):.3f}"
        if resid_short:
            parts["resid"] = resid_short
        return parts

    @staticmethod
    def _readout_of(summary, colour: str | None) -> FitReadout:
        """A histogram fit's :class:`FitReadout`. A summary without parts (the
        exponential + gamma placeholder while it is on the worker) says so in the label,
        where the numbers will appear."""
        parts = getattr(summary, "parts", None) or {"equation": str(summary)}
        return FitReadout(parts, (), None, colour)

    def _full_fit_summary(self, fit, col: str, gof: dict | None,
                          resid_short: str = "") -> str:
        """Every number this fit has, on one line: the text of the box that used to be drawn ON
        the figure (the fit label takes the same numbers, token by token, from
        :meth:`_fit_parts`).

        Owner note #13: "once the Fit has been applied, include the numerical values on
        the plot; make these plots presentation ready". The one-line title summary
        carries only the HEADLINE parameter -- tau, or the mixture's mu's -- so a
        two-component Gaussian mixture put its means on the figure and left both sigmas
        and both weights in a Details pane that the exported PNG does not contain. A
        reader of that PNG cannot tell a tight mixture from a broad one.

        So: every entry of ``fit.params``, each scaled and named through the SAME
        :meth:`_display_unit` rule the title uses (a time parameter in the axis's unit,
        a dimensionless shape bare), then n, BIC and the numerical KS statistic. The
        verdict itself stays in Details, not on the figure.
        """
        parts = []
        for key, value in fit.params.items():
            scale, uname = self._display_unit(col, fit.unit_of(key))
            parts.append(f"{_param_symbol(key)} = {float(value) * scale:.4g}"
                         + (f" {uname}" if uname else ""))
        parts.append(f"n = {fit.n}")
        if np.isfinite(float(fit.bic)):
            parts.append(f"BIC = {float(fit.bic):.6g}")
        # The numerical statistic only; its interpretation and rejection warning are
        # retained in Details and fit metadata.
        if gof:
            parts.append(f"KS = {float(gof['ks']):.3f}")
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
                with stats.deferred_ci() as jobs:
                    fit = stats.fit_dwell_exp_gamma(sample, t_min=lo, t_max=hi,
                        quantity=quantity, unit=unit, n_boot=200)
                # Transient callables stay outside serialized metadata. The point fit
                # is published first; its expensive bootstrap is another cancellable job.
                fit._gui_deferred_ci_jobs = tuple(jobs)
                return fit
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
        """Validate requested bounds before fitting; invalid limits never broaden a fit."""
        out = []
        for edit, side in ((self._fit_lo, "min"), (self._fit_hi, "max")):
            txt = edit.text().strip()
            if not txt:
                continue
            try:
                valid = np.isfinite(float(txt))
            except ValueError:
                valid = False
            if not valid:
                out.append(f"Fit range {side} must be a finite number; correct or clear “{txt}”.")
        lo, hi = self._fit_lo.text().strip(), self._fit_hi.text().strip()
        if lo and hi and not out and float(lo) >= float(hi):
            out.append("Fit range minimum must be smaller than maximum.")
        return out

    def _validate_fit_range_fields(self, *_):
        errors = self._fit_range_input_notes()
        for edit in (self._fit_lo, self._fit_hi):
            edit.setStyleSheet(f"border: 1px solid {PALETTE['fit']};" if errors else "")
            edit.setAccessibleDescription(" ".join(errors))
        if errors and hasattr(self, "_fit_apply_btn"):
            self._fit_apply_btn.setEnabled(False)
        elif hasattr(self, "_scatter_fit_apply_btn"):
            self._sync_fit_apply_buttons()

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
        if self._fit_range_input_notes():
            return "No new fit was run because the requested fit range is invalid."
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

    def _nonlinear_controller(self):
        from .nonlinear_fit_dialog import NonlinearFitController
        if not hasattr(self, "_nonlinear_fitter"):
            self._nonlinear_fitter = NonlinearFitController(
                self, lambda: getattr(self, "_nonlinear_drawn_datasets", []), self._plot,
                self._nonlinear_applied, "scatter")
        return self._nonlinear_fitter

    def _nonlinear_applied(self):
        from .nonlinear_fit_dialog import result_summary
        state = self.nonlinear_fit_state()
        base = self.details_text().split("\n\nNonlinear fit:", 1)[0]
        self._set_details(base + ("\n\n" + result_summary(state) if state else ""))
        self._nonlinear_controller().sync_controls()
        self._install_nonlinear_diagnostics()
        self._apply_series_styles()
        self.workspaceChanged.emit()

    def _install_nonlinear_diagnostics(self):
        state = self.nonlinear_fit_state()
        if not state:
            residual = getattr(self, "_presentation_residual", None)
            if residual and residual[0].get("mode") == "raw":
                self._clear_residual_strip()
                self._presentation_residual = None
                self._presentation_residual_note = ""
                self._presentation_residual_ready = False
            return
        source, result = state["source"], state["result"]
        factor, unit = self._display_unit(source["y_name"], source.get("y_unit", ""))
        residual = {"centers": list(result["x"]),
                    "residual": (np.asarray(result["residuals"]) * factor).tolist(),
                    "mode": "raw", "unit": unit,
                    "note": f"Raw response residuals for {source['label']} ({unit or 'unitless'}); residual = observed minus fitted response.",
                    "reduced_chi2": None}
        self._clear_residual_strip()
        self._presentation_residual = (residual, self._scatter_logx.isChecked())
        self._presentation_residual_note = residual["note"]
        self._presentation_residual_ready = True
        if self._style.show_residuals:
            self._render_cached_residual(*self._presentation_residual)

    def open_nonlinear_fit(self):
        self._nonlinear_controller().open()

    def nonlinear_fit_state(self):
        return self._nonlinear_fitter.snapshot() if hasattr(self, "_nonlinear_fitter") else None

    def restore_nonlinear_fit_state(self, state):
        self._nonlinear_controller().restore(state)
        self._nonlinear_controller().sync_controls()
        if state:
            from .nonlinear_fit_dialog import result_summary
            summary = result_summary(state)
            if summary not in self.details_text():
                self._set_details(self.details_text() + "\n\n" + summary)
            self._install_nonlinear_diagnostics()
            self._apply_series_styles()

    def _overlay_scatter_fit(
        self, x: np.ndarray, y: np.ndarray, xcol: str, ycol: str, details: list[str],
        src: PlotSource | None = None, key: str = "",
    ) -> str | None:
        """Fit and draw the scatter best-fit line; return its summary (or None).

        The numbers go to the fit label as a :class:`fit_label.FitReadout` keyed by
        ``key`` (the source's), never to the title.

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
                self._scatter_fit_pending_after_snapshot = True
                details.append(
                    "Scatter-fit configuration restored; update the plot to recompute the "
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
        item._presentation_role = "category" if src is not None else "fit"
        self._tag_series(item, "fit", key or "fit", f"{src.label if src else 'Data'} fit",
                         colour=src.colour if src is not None else None)
        item._presentation_width_extra = 1
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
        # THE FIT LABEL'S NUMBERS ROW LEAVES OUT A STATISTIC, NEVER A REFUSAL. The first two
        # branches here are refusals -- "y does not vary" and the dof tag are the sentences
        # that say this R² is arithmetic rather than evidence -- and Details prints them
        # whatever the label shows. Only the plain, meaningful number is a field the label
        # may leave out, which is the one a clean paper figure has a reason to omit.
        if not np.isfinite(fit.r2):
            r2_txt = "R² undefined (y does not vary)"
        elif _dof_tag:
            r2_txt = f"R² = {fit.r2:.3f} ({_dof_tag})"
        else:
            r2_txt = f"R² = {fit.r2:.3f}"
        if fit.log_log:
            coeff = fit.meta["coefficient"] * fy / fx ** fit.meta["exponent"]
            head = f"y = {coeff:.4g}·x^{fit.meta['exponent']:.4g}{unit_txt}"
        else:
            head = f"y = {fit.slope * fy / fx:.4g}·x + {fit.intercept * fy:.4g}{unit_txt}"
        summary = "   ".join(p for p in (head, r2_txt, f"(n = {fit.n})") if p)
        # THE FIT LABEL'S COPY. The refusal ("undefined", or the dof tag on a number that
        # is arithmetic) is a refusal, not a field: the label prints it whenever R² is
        # asked for and no template or Numbers choice can drop it. Only the plain,
        # meaningful number is a field the Numbers row may leave out.
        parts = {"equation": head, "n": f"n = {fit.n}"}
        refusals: tuple[str, ...] = ()
        if not np.isfinite(fit.r2) or _dof_tag:
            refusals = (r2_txt,)
        else:
            parts["r2"] = f"R² = {fit.r2:.3f}"
        self._presentation_readouts.append(
            (key, FitReadout(parts, refusals, None, src.colour if src is not None else None)))
        item._presentation_fit_family = "power law" if fit.log_log else "linear"
        if src is not None:
            # By the FIGURE's name for the source (`_legend_name`, the one resolver),
            # as the series entry above it is: with two rows typed to "Apo" the
            # series read "Apo — -700 mV" / "Apo — -600 mV" and these two fit entries
            # read "Apo fit: …" twice in the same legend (A1-N2). No numbers: they are
            # in the fit label, one line per source in its colour.
            self._legend_add(item, self._fit_legend_label(item, self._legend_name(src), key),
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
