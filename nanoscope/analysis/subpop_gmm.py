"""Subpopulation quantification for a single recording: how many event classes, and how big.

This is the quantitative end of the Zn2-insulin waveform pipeline. The
exploratory end (per-scope HDBSCAN / k-means / t-SNE / UMAP over the 13
waveform features) answers *"is there structure?"*; this module answers the
harder question the owner actually asked of ``zn_30h`` -- **how many
subpopulations are there, what fraction of events is in each, and how sure are
we of both numbers**.

It takes a per-event feature table and returns a fitted Gaussian mixture, a
population table with bootstrap confidence intervals, and a crosstab against
whatever exploratory labels the caller already has. It does NOT detect events,
does NOT compute the 13 features, and does NOT add a column to the EventSet
schema -- the features are derived on demand upstream, exactly as
:mod:`nanoscope.analysis.event_shape` derives its ensembles.

Three decisions in here are load-bearing and none of them is the obvious one.

**1. The number of PCs is not chosen by a variance rule.**
The exploratory clustering keeps however many PCs reach 95 % of the variance.
That rule is fine for drawing a scatter and wrong for fitting a density: it
asks "where is the variance?" when the question is "where is the *structure*?".
Measured on ``zn_30h``, the 95 % rule stops at **4** PCs, and the same ``K = 4``
mixture fit in those 4 lands on a different partition -- 444/307/240/85 against
387/295/269/125 -- with the fraction of confidently assigned events falling from
**98.6 % to 79.0 %**. Same components, much blurrier boundaries: the separating
structure is sitting in PCs the variance rule threw away. So ``n_PCs`` and ``K``
are chosen **jointly**, by 5-fold cross-validated held-out log-likelihood -- see
:func:`select_pcs_and_k`.

**2. Held-out likelihood is not comparable across PC counts unless you pay for
the discarded axes.** A GMM in 4 dimensions and a GMM in 11 dimensions are
densities over different spaces; their log-likelihoods are in different units,
so ranking them directly ranks the dimensionality rather than the fit. The fix
is PPCA-style: score the model in the FULL scaled feature space, kept PCs
through the mixture density and discarded PCs through an isotropic Gaussian
whose variance is the mean of the *training* eigenvalues on those axes. Every
``(n_PCs, K)`` cell is then a density over the same 13-D space and the numbers
can be ranked.

Measured, the residual term is not a refinement -- it is the difference between
an answer and an artefact. Delete it and the same grid on the same data selects
``n_PCs = 13``, the very top of the range, because the last principal axes carry
almost no variance and a density over a nearly-degenerate axis has an enormous
log-density that has nothing to do with the mixture. The uncorrected p-profile
at ``K = 4`` climbs monotonically over the last three PCs (-2.48, 3.51, 9.50)
and would keep climbing if the grid were wider.

**3. BIC is reported and NOT used to choose.** On the owner's ``zn_30h`` the two
criteria disagree loudly -- cross-validation picks ``K = 4``, BIC picks
``K = 8``, the top of the grid. BIC's penalty counts parameters, and a full
covariance in 11 dimensions costs 77 of them per component; that penalty is
still not enough to stop it chasing the heavy tail of the long-dwell events with
extra components. The held-out likelihood, which pays for overfitting directly
rather than by proxy, stops at 4. :func:`bic_ladder` computes BIC anyway and
:attr:`SubpopFit.warnings` says so out loud, because "the other criterion said 8"
is exactly the kind of thing a reader of a four-population claim is owed.

**4. Nothing above measures whether the classes are APART, so a fourth thing
does.** Held-out likelihood, BIC and the posteriors are all properties of the
density. A single cloud cut into K contiguous wedges satisfies all three -- the
wedges predict unseen events, they pay for their parameters, and the posterior
inside each wedge is ~1 -- and it is still one population. On the owner's
apoferritin at -700 mV the pipeline chose ``K = 7`` with BIC agreeing and 84 % of
events above a 0.9 posterior, and the seven classes score a mean silhouette of
**0.098**: 27 % of events have a NEGATIVE silhouette and 31 % sit nearer another
class's centre. The best plain k-means at ANY searched K on the same coordinates
reaches only **0.224**, itself inside the same lowest band -- a CEILING on how
separated this cloud can be made to look, not a rival partition, and the
distinction matters because the ceiling is measured with the method the
silhouette is biased towards. :func:`measure_separation` computes that number, :class:`Separation`
carries it, and :meth:`SubpopQuantification.claim` makes it impossible to state
the class count without it. The rule the module now keeps is that **a headline
may not be more confident than its own caveats**: the count and the evidence for
the count travel together or neither travels.

Honesty about the bootstrap
---------------------------
:func:`bootstrap_fractions` reproduces the owner's resampling exactly, including
its Hungarian component matching, and the intervals it returns on ``zn_30h`` are
**wide and mutually overlapping** -- subpopulation 0's 95 % interval spans
0.182-0.638 around a point estimate of 0.360, and it swallows the point
estimates of two other subpopulations whole. That is not 3x uncertainty in the
fractions. It is residual label-switching surviving the mean-distance match: on
a resample where two components trade places, Hungarian matching on component
means picks the assignment that minimises total distance, which is not always
the one that preserves identity. :func:`_ci_warnings` detects the signature and
says so, so the interval never reaches a figure caption uncaveated. Reproduce
the numbers; do not present them as calibrated.

sklearn/scipy/numpy only. Qt-free, and safe on a worker thread: every loop long
enough to matter polls the ambient cancel token from
:mod:`nanoscope.analysis.cancel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA as _SKLearnPCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from nanoscope.analysis.cancel import CancelFlag, check_cancelled
from nanoscope.analysis.populations import cluster as _cluster
from nanoscope.analysis.stats import ci_wanted
from nanoscope.analysis.subpop_cluster import CLUST_FEATS, scale_features

__all__ = [
    "SUBPOP_FEATURES",
    "scale_features",
    "DEFAULT_K_GRID",
    "DEFAULT_N_BOOT",
    "DEFAULT_N_SPLITS",
    "CONFIDENCE_FLOOR",
    "TABLE_DECIMALS",
    "SEPARATION_FLOOR",
    "SEPARATION_WEAK_CEILING",
    "SEPARATION_MAX_N",
    "SEPARATION_REFERENCE_K",
    "SEPARATION_CEILING_K_GRID",
    "STABILITY_BAND",
    "STABILITY_ODDS_RATIO",
    "stability_band_for",
    "PCSelection",
    "SubpopFit",
    "FractionCI",
    "Separation",
    "SubpopQuantification",
    "select_pcs_and_k",
    "bic_ladder",
    "fit_subpopulations",
    "bootstrap_fractions",
    "measure_separation",
    "population_table",
    "label_crosstab",
    "quantify_subpopulations",
]


#: The 13 per-event waveform features, **in the order the mixture sees them**.
#:
#: An alias of :data:`nanoscope.analysis.subpop_cluster.CLUST_FEATS`, not a copy.
#: The exploratory clustering and this stage MUST see the same columns in the
#: same order or their labels cannot be crosstabbed, and a second literal list
#: is a second thing to forget to update.
#:
#: The order is not the feature table's column order, and the difference is not
#: cosmetic: ``entropy`` sits last here and in the middle there. PCA loadings do
#: not care about column order but the *scores* do -- a sign flip, or a swap of
#: two nearly-degenerate components, is enough to change which EM local optimum
#: the mixture lands in and therefore which events end up in which
#: subpopulation. Feed the features in this order or do not expect to reproduce
#: anybody's labels.
SUBPOP_FEATURES: tuple[str, ...] = CLUST_FEATS

#: Candidate component counts. ``1`` is in the grid on purpose -- "there is only
#: one population" has to be able to win, or a four-population answer is an
#: artefact of the search space rather than a finding.
DEFAULT_K_GRID: tuple[int, ...] = tuple(range(1, 9))

#: Folds for the held-out likelihood.
DEFAULT_N_SPLITS = 5

#: Bootstrap resamples for the fraction intervals.
DEFAULT_N_BOOT = 200

#: How often :func:`bootstrap_fractions` ticks its optional progress sink. Every
#: resample would put a queued Qt signal in front of a fit that takes a few tens of
#: milliseconds; every tenth is still ~20 updates over the default 200 resamples.
_BOOT_PROGRESS_STRIDE = 10

#: A posterior above this counts as a confident assignment. Reported as a
#: fraction of events, because a mixture that assigns most of its events at
#: 0.5 has found overlapping shoulders, not populations.
CONFIDENCE_FLOOR = 0.9

#: Decimal places for the reported population table, per column. Applied by
#: :func:`population_table` unless the caller passes ``decimals=None`` to get
#: the unrounded numbers.
TABLE_DECIMALS: dict[str, int] = {
    "fraction": 3,
    "frac_lo95": 3,
    "frac_hi95": 3,
    "rate_per_s": 3,
    "dwell_p50_ms": 3,
    "dwell_p90_ms": 3,
    "depth_max_pA": 1,
    "depth_sust_pA": 1,
    "f_sust": 4,
    "entropy": 3,
    "kurtosis": 2,
    "conf_med": 2,
    "early_frac": 2,
}

#: Profile columns of the population table: ``(output name, source column)``,
#: each reported as the per-subpopulation **median**. A median, not a mean:
#: ``dwell_ms`` spans four orders of magnitude within a single subpopulation and
#: its mean is a statement about the longest event, not about the class.
_PROFILE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("depth_max_pA", "depth_max_pA"),
    ("depth_sust_pA", "depth_sust_pA"),
    ("f_sust", "f_sust"),
    ("entropy", "entropy"),
    ("kurtosis", "kurtosis"),
)

_DWELL_COLUMN = "dwell_ms"
_TIME_HALF_COLUMN = "time_half"
_EARLY_LABEL = "early"

#: Below this, a "confident assignment" claim stops being true. See
#: :func:`_fit_warnings`.
_CONF_FRAC_FLOOR = 0.80

#: A 95 % interval wider than this multiple of its own point estimate is
#: reported as uninformative. 1.0 means "the interval is wider than the number
#: it brackets".
_CI_REL_WIDTH_FLOOR = 1.0

#: Mean silhouette at or below which the reported partition carries **no
#: substantial structure**.
#:
#: THIS NUMBER IS CITED, NOT TUNED. It is the top of Kaufman & Rousseeuw's
#: lowest band (<=0.25 "no substantial structure has been found"; 0.26-0.50
#: "weak, could be artificial"; 0.51-0.70 "reasonable"; >0.70 "strong"). It was
#: not chosen by looking at what the apoferritin runs score, and moving it to
#: make a particular recording pass or fail is the defect this constant exists
#: to prevent.
SEPARATION_FLOOR = 0.25

#: Top of the "weak, could be artificial" band; above it the partition is at
#: least reasonably separated. Same citation, same rule about not moving it.
SEPARATION_WEAK_CEILING = 0.50

#: The silhouette is O(n^2) in distances. Above this many events it is measured
#: on a deterministic subsample (seeded from the model's own ``random_state``)
#: and the report says how many events it was measured on. 4000 events is ~16 M
#: pairwise distances -- about a tenth of a second and ~128 MB in float64, which
#: is the last size at which "compute it every run" is honest about its cost.
SEPARATION_MAX_N = 4000

#: Reference partition the mixture's separation is quoted against: k-means at
#: this K, fitted **in the same coordinates the mixture was fitted in**, so the
#: two silhouettes are comparable. Three because it is the smallest K that can
#: express more than a split, and because it is the K the exploratory stage
#: already runs (:data:`subpop_cluster.KMEANS_K`) -- this is a YARDSTICK for
#: "how separated can these coordinates be made at all", never a recommendation
#: to report three subpopulations.
SEPARATION_REFERENCE_K = 3

#: K values the SEPARATION CEILING is searched over -- the mixture's own search
#: grid minus ``K = 1``, where a silhouette is undefined.
#:
#: WHY A CEILING AND NOT A RIVAL. Comparing this mixture's silhouette against ONE
#: k-means at ONE other K compares two numbers that are not comparable: the mean
#: silhouette falls with K for any cloud that is a continuum, so a K=3 partition
#: beating a K=7 mixture is the arithmetic, not a finding. Worse, k-means
#: minimises within-cluster scatter and the silhouette REWARDS within-cluster
#: scatter, so a k-means partition out-scoring a full-covariance mixture at the
#: SAME K is also the expected result -- measured here at K=7 on apo -700 mV
#: (0.166 vs 0.052) and at K=4 on the Zn2+ control (0.385 vs 0.110). A caveat
#: built on either comparison fires on every dataset and separates none of them.
#:
#: What a max over this grid DOES support is an existence statement, and an
#: existence statement needs no comparability across K: if the best partition the
#: method that is biased IN ITS FAVOUR can find, at ANY K in the search, is still
#: inside the lowest band, then no partition of these coordinates is separated
#: and the question "how many classes" has no separated answer here. That fires
#: on apo -700 mV (ceiling 0.223 at K=2) and stays silent on the Zn2+ positive
#: control (ceiling 0.678 at K=2), which is what a caveat has to do to be
#: measuring anything.
SEPARATION_CEILING_K_GRID: tuple[int, ...] = tuple(k for k in DEFAULT_K_GRID if k >= 2)

#: ``early_frac`` inside this band means "present through both halves of the
#: recording". Outside it a class is at least 3:1 concentrated into one half,
#: which is what the population table's own caption calls drift or fouling
#: rather than a class. The band is the caption's sentence turned into a test:
#: it is readable straight off the printed column, which is the point.
#:
#: It is the band for a STATIONARY recording, i.e. the special case
#: ``stability_band_for(0.5)``. On a recording whose own event supply is lopsided
#: it is the wrong band and :func:`stability_band_for` is the right one -- see
#: there for why comparing a class against 0.25-0.75 on a 90 %-front-loaded
#: recording condemns every class in it for being shaped like the recording.
STABILITY_BAND: tuple[float, float] = (0.25, 0.75)

#: The odds ratio :data:`STABILITY_BAND` encodes, derived from it rather than
#: written down twice: ``0.75/0.25 = 3``, i.e. "at least 3:1 into one half".
STABILITY_ODDS_RATIO: float = (STABILITY_BAND[1] / (1.0 - STABILITY_BAND[1]))

#: Decimals the stability check WORKS IN -- taken from the reported precision of
#: the column it reads so the two cannot drift apart.
#:
#: THE INVARIANT: **a printed number and the band printed beside it must be the
#: same number the gate used.** ``population_table`` rounds ``early_frac`` to
#: this many places before :func:`_stability_warnings` ever sees it, so a band
#: recomputed at full precision from those rounded values lands a few
#: ten-thousandths off its printed edge -- and a class printed at ``0.25``,
#: against a band printed ``0.25-0.75``, was condemned for being "outside" it.
#: The reader could not check the sentence against the table it sits under, and
#: was right not to believe it. Rounding to a fixed number of places is
#: monotone, so gating on the printed values can only ever WITHDRAW a flag that
#: the reader could not have verified -- never invent one.
_STABILITY_DECIMALS: int = int(TABLE_DECIMALS.get("early_frac", 2))


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------


@dataclass
class PCSelection:
    """The joint ``(n_PCs, K)`` choice, with the whole grid it was chosen from.

    The grid is kept, not just its argmax, because the argmax alone cannot tell
    a reader whether the choice was made by the data or by a rounding error.
    :meth:`k_profile` is the slice worth printing beside any claim about the
    number of subpopulations.

    Attributes:
        n_pcs: Principal components kept by the selection.
        k: Mixture components chosen.
        loglik: Mean held-out log-likelihood per event at ``(n_pcs, k)``, in the
            full scaled feature space (kept PCs + isotropic residual).
        grid: ``(len(p_grid), len(k_grid))`` of the same quantity. NaN where a
            cell could not be scored.
        p_grid: PC counts searched, in row order of ``grid``.
        k_grid: Component counts searched, in column order of ``grid``.
        n_splits: Cross-validation folds used.
        random_state: Seed shared by the fold split, the per-fold PCA and every
            mixture fit inside the loop.
        n_init: EM restarts per candidate inside the loop.
        warnings: Caveats about THIS selection -- a flat optimum, or a choice at
            the edge of the searched grid.
    """

    n_pcs: int
    k: int
    loglik: float
    grid: np.ndarray
    p_grid: tuple[int, ...]
    k_grid: tuple[int, ...]
    n_splits: int
    random_state: int
    n_init: int
    warnings: tuple[str, ...] = ()

    def k_profile(self) -> dict[int, float]:
        """Held-out log-likelihood against ``K`` at the chosen PC count."""
        row = self.p_grid.index(self.n_pcs)
        return {int(k): float(self.grid[row, i]) for i, k in enumerate(self.k_grid)}

    def p_profile(self) -> dict[int, float]:
        """Held-out log-likelihood against PC count at the chosen ``K``."""
        col = self.k_grid.index(self.k)
        return {int(p): float(self.grid[i, col]) for i, p in enumerate(self.p_grid)}

    def __repr__(self) -> str:
        return (
            f"<PCSelection n_pcs={self.n_pcs} k={self.k} "
            f"loglik={self.loglik:.2f}/event grid={self.grid.shape}>"
        )


@dataclass
class SubpopFit:
    """A fitted mixture over one recording, with components numbered by size.

    Components are renumbered so 0 is the largest -- EM's own numbering is an
    artefact of initialisation and would make two runs of the same pipeline
    disagree about which subpopulation is "component 1" while agreeing about
    every event.

    Attributes:
        labels: Subpopulation id per event, indexed like the source table.
        posteriors: ``(n_events, k)`` responsibilities, columns in the same
            size order as ``labels``.
        confidence: Per-event max posterior -- how sure the mixture is about the
            label it gave that event.
        conf_frac: Fraction of events assigned above :data:`CONFIDENCE_FLOOR`.
        conf_median: Median of ``confidence``.
        scores: ``(n_events, n_pcs)`` PCA projection the mixture was fit in.
        means: ``(k, n_pcs)`` component means, size-ordered.
        weights: ``(k,)`` mixing fractions from the model (not the empirical
            label counts -- see ``sizes`` for those).
        sizes: ``{subpop: n_events}`` by hard assignment.
        n_pcs: PCs kept.
        k: Components fit.
        order: ``order[i]`` is the raw EM component that became subpopulation
            ``i``. Needed to reorder anything else the raw model produced.
        explained_variance_ratio: Variance fraction on each kept PC.
        bic: ``{k: BIC}`` cross-check ladder, or ``None`` if not computed.
        bic_k: ``argmin`` of that ladder.
        warnings: Caveats about THIS fit -- a BIC that disagrees, low assignment
            confidence, a degenerate component.
        meta: Seeds and restarts, so a result can be re-derived from itself.
    """

    labels: pd.Series
    posteriors: np.ndarray
    confidence: np.ndarray
    conf_frac: float
    conf_median: float
    scores: np.ndarray
    means: np.ndarray
    weights: np.ndarray
    sizes: dict[int, int]
    n_pcs: int
    k: int
    order: np.ndarray
    explained_variance_ratio: np.ndarray
    bic: dict[int, float] | None = None
    bic_k: int | None = None
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        sz = ", ".join(f"{c}:{n}" for c, n in sorted(self.sizes.items()))
        return f"<SubpopFit k={self.k} n_pcs={self.n_pcs} sizes=[{sz}]>"


@dataclass
class Separation:
    """How separated the reported mixture's own classes are -- measured, not assumed.

    THE MISSING NUMBER. Everything else this module reports is a property of the
    *density*: held-out likelihood says the mixture predicts unseen events well,
    the posterior says the mixture is sure which component generated an event,
    BIC says the parameter count is paid for. None of them says the classes are
    APART. A single Gaussian cloud sliced into seven contiguous wedges scores
    well on all three -- the slices really do predict, the posteriors really are
    near 1 inside each wedge, and each wedge really does earn its parameters --
    and it is still one population. The silhouette is the statistic that can
    tell those two situations apart, and until this class existed the report
    printed one for the exploratory k-means and **none for the model it led
    with**.

    Measured in the space the mixture was FITTED in (``fit.scores``), not in the
    two drawn coordinates: a claim about seven classes in eleven dimensions has
    to be checked in eleven dimensions, and checking it in the scatter's plane
    would be answerable with "the picture is only a projection".

    Attributes:
        silhouette: Mean silhouette of the reported labels over ``scores``.
        neg_fraction: Fraction of events with a NEGATIVE silhouette -- events
            that are, on average, closer to another class than to their own.
        nearer_other_fraction: Fraction of events whose nearest component MEAN
            is not the mean of the class they were assigned to. The posterior
            can still be ~1 for these (a wide component wins on covariance), so
            this is not a restatement of ``conf_frac``.
        per_subpop: ``{subpop: (n, silhouette, nearer_other_fraction)}``.
        reference_k: K of the reference partition (:data:`SEPARATION_REFERENCE_K`).
        reference_silhouette: Silhouette of k-means at ``reference_k`` on the
            SAME coordinates -- the yardstick for how separated this cloud can
            be made at all. Not a recommendation.
        reference_profile: ``{K: silhouette}`` for plain k-means at every K in
            :data:`SEPARATION_CEILING_K_GRID` (plus the mixture's own K), all in
            the same coordinates. The point of the whole row is its MAXIMUM;
            individual entries are not comparable with each other, and reading a
            K off it as "the right number of classes" is the error the ceiling
            exists to avoid making.
        ceiling_k: ``argmax`` of ``reference_profile``, or ``None``.
        ceiling_silhouette: ``max`` of ``reference_profile``. THE CEILING: the
            most separated any partition of these coordinates looks, measured
            with the method the silhouette is biased towards. A mixture cannot
            be blamed for scoring under it, but a ceiling that is itself inside
            the lowest band says the cloud has no separated partition at any K.
        n_used: Events the silhouette was measured on (see
            :data:`SEPARATION_MAX_N`).
        n_events: Events in the fit.
        n_pcs: Dimension of the space it was measured in.
        subsampled: True when ``n_used < n_events``.
    """

    silhouette: float
    neg_fraction: float
    nearer_other_fraction: float
    per_subpop: dict[int, tuple[int, float, float]]
    reference_k: int
    reference_silhouette: float | None
    n_used: int
    n_events: int
    n_pcs: int
    subsampled: bool = False
    reference_profile: dict[int, float] = field(default_factory=dict)

    @property
    def ceiling_k(self) -> int | None:
        """``argmax`` of :attr:`reference_profile`."""
        prof = self._ceiling_profile()
        if not prof:
            return None
        return int(max(prof, key=lambda k: prof[k]))

    @property
    def ceiling_silhouette(self) -> float | None:
        """``max`` of :attr:`reference_profile` -- see the attribute docs."""
        prof = self._ceiling_profile()
        if not prof:
            return None
        return float(max(prof.values()))

    def _ceiling_profile(self) -> dict[int, float]:
        """The profile, falling back to the single reference where there is no row.

        A :class:`Separation` built by hand (or unpickled from before the profile
        existed) carries one reference silhouette and no row. ONE MEASURED K IS
        ONLY A LOWER BOUND on the true ceiling, so a caveat built on it could
        fire where a wider search would have cleared the floor. That is why the
        caveat names the Ks it actually measured instead of claiming "any K":
        the existence statement is only ever made about the row in hand.
        """
        prof = {int(k): float(v) for k, v in (self.reference_profile or {}).items()
                if v is not None and np.isfinite(v)}
        if not prof and self.reference_silhouette is not None:
            prof = {int(self.reference_k): float(self.reference_silhouette)}
        return prof

    @property
    def ceiling_substantial(self) -> bool | None:
        """Whether ANY searched partition of these coordinates clears the floor.

        ``None`` when nothing could be measured. ``False`` is the strong reading:
        not "this K is wrong" but "no K here is separated".
        """
        ceil = self.ceiling_silhouette
        if ceil is None:
            return None
        return ceil > SEPARATION_FLOOR

    @property
    def verdict(self) -> str:
        """A three-word reading of :attr:`silhouette`, on the cited bands."""
        if self.silhouette <= SEPARATION_FLOOR:
            return "NO SUBSTANTIAL STRUCTURE"
        if self.silhouette <= SEPARATION_WEAK_CEILING:
            return "WEAK, COULD BE ARTIFICIAL"
        if self.silhouette <= 0.70:
            return "REASONABLY SEPARATED"
        return "STRONGLY SEPARATED"

    @property
    def substantial(self) -> bool:
        """Whether the partition clears :data:`SEPARATION_FLOOR`."""
        return self.silhouette > SEPARATION_FLOOR

    def one_line(self) -> str:
        """The sentence that has to travel WITH the class count, everywhere.

        WRITTEN TO SURVIVE TRUNCATION. This is the only sentence that reaches
        every surface -- the report document, the report window's header and,
        since the status bar was routed through ``headline()``, the status bar
        too -- and the status bar is a single elided line. On the owner's
        -700 mV recording the bar's message is 425 characters and about 72 % of
        it fits at 1900 px, so a third of this sentence is read by nobody. That
        makes WORD ORDER a correctness property, not a style one: the verdict
        goes first, adjacent to the class count it qualifies, so the half that
        survives is the half that carries the caveat. It used to lead with the
        raw silhouette and reach ``NO SUBSTANTIAL STRUCTURE`` a clause later.

        AND WHAT IS NO LONGER HERE. It used to end "; k-means K=3 on the same
        coordinates scores 0.197" -- a bare rival number, on a line with no room
        to frame it, four caveats above the report's own sentence saying a
        number of exactly that kind "is a CEILING ... not a rival result" and
        quoting a DIFFERENT one (0.224, at K=2, the max over the searched grid).
        Two k-means silhouettes on one cloud, one framed and one bare, and the
        bare one was the one that travelled -- and the one truncation kept. The
        reference silhouette and the whole k-means-by-K row are still printed,
        once, in the report document's ``separation`` block, where the ceiling
        paragraph that disowns them is on the very next line.
        """
        on = "" if not self.subsampled else f", on {self.n_used} of {self.n_events} events"
        return (
            f"{self.verdict} — silhouette {self.silhouette:.3f} in the "
            f"{self.n_pcs}-PC space it was fitted in, "
            f"{100 * self.nearer_other_fraction:.1f} % of events nearer another "
            f"class's centre{on}"
        )

    def __repr__(self) -> str:
        return (
            f"<Separation silhouette={self.silhouette:.3f} ({self.verdict}) "
            f"nearer_other={self.nearer_other_fraction:.3f}>"
        )


@dataclass
class FractionCI:
    """Bootstrap intervals on the population fractions.

    Attributes:
        lo: ``(k,)`` 2.5th percentile of the resampled fractions.
        hi: ``(k,)`` 97.5th percentile.
        samples: ``(n_boot, k)`` resampled fractions; a row is all-NaN where the
            refit failed to converge and was skipped.
        n_ok: Resamples that produced a fit.
        n_boot: Resamples requested.
        seed: Seed of the RNG that drew the resampling indices.
        warnings: Caveats about THESE intervals -- above all the label-switching
            signature, which is present in the owner's own run.
    """

    lo: np.ndarray
    hi: np.ndarray
    samples: np.ndarray
    n_ok: int
    n_boot: int
    seed: int
    warnings: tuple[str, ...] = ()

    @property
    def computed(self) -> bool:
        """False when the bootstrap was skipped (``point_fits_only``/``n_boot=0``)."""
        return self.n_ok > 0

    def __repr__(self) -> str:
        return f"<FractionCI n_ok={self.n_ok}/{self.n_boot} k={self.lo.size}>"


@dataclass
class SubpopQuantification:
    """Everything :func:`quantify_subpopulations` produced, in one object.

    Attributes:
        selection: The ``(n_PCs, K)`` search.
        fit: The mixture at that choice.
        ci: Bootstrap intervals on the fractions.
        table: The reportable population table.
        crosstab: Subpopulation x exploratory-label contingency table, or
            ``None`` when no exploratory labels were supplied.
        features: Feature columns used, in order.
        n_events: Events quantified.
        separation: How far apart the reported classes actually are, or ``None``
            at ``K = 1``, where there is nothing to separate. **Quote this
            wherever ``fit.k`` is quoted** -- that is the whole reason it is a
            field on the same object rather than something a caller may compute.
        warnings: The union of every caveat below, de-duplicated in order. Read
            THIS one -- it is the only place that sees all of them.
    """

    selection: PCSelection
    fit: SubpopFit
    ci: FractionCI
    table: pd.DataFrame
    crosstab: pd.DataFrame | None
    features: tuple[str, ...]
    n_events: int
    separation: Separation | None = None
    warnings: tuple[str, ...] = ()

    def claim(self) -> str:
        """The class count and, inseparably, the evidence for it.

        THE INVARIANT THIS METHOD IS: a headline may not be more confident than
        its own caveats. Every surface that wants to say "K subpopulations" says
        it by calling this, so the qualifier cannot be dropped by a caller who
        only wanted the number -- which is exactly how the count reached a window
        header and a status bar with the separation statistic left in a document
        nobody scrolled to.
        """
        head = f"{self.fit.k} subpopulation(s) at n_PCs={self.fit.n_pcs}"
        if self.fit.k <= 1:
            # "and did not" is only worth saying when it COULD have. A caller who
            # searched k_grid=(1,) has been told what it asked for, and a claim
            # that the data chose one population would be this method committing
            # the very fault it exists to prevent.
            bigger = [k for k in self.selection.k_grid if k > self.fit.k]
            return (
                f"{head} — one population"
                + (
                    f"; the search could have chosen up to K={max(bigger)} and did not"
                    if bigger
                    else f"; K>1 was NOT searched, so this is the grid's answer, not the data's"
                )
            )
        conf = f"{100 * self.fit.conf_frac:.1f} % confidently assigned"
        if self.separation is None:
            return f"{head} ({conf}) — SEPARATION NOT MEASURED"
        return f"{head} — {self.separation.one_line()} — {conf}"

    def __repr__(self) -> str:
        return (
            f"<SubpopQuantification n={self.n_events} k={self.fit.k} "
            f"n_pcs={self.fit.n_pcs} warnings={len(self.warnings)}>"
        )


# ---------------------------------------------------------------------------
# feature matrix -- IMPORTED, not reimplemented
# ---------------------------------------------------------------------------
#
# `scale_features` is re-exported from `subpop_cluster`, which composes it from
# `populations.robust_scale`. There is deliberately no scaler in this file. The
# exploratory clustering and this stage must scale identically -- winsorise
# 1-99 %, then centre on the median and divide by the IQR OF THE CLIPPED column
# -- or the crosstab in item 12 is comparing labels from two different spaces,
# and nothing in either result would look wrong.
#
# What that scaler does NOT do is drop rows: a non-finite feature is excluded
# from its own column's percentiles and lands on the scaled median.
# `populations.feature_matrix` would drop the row whole, which is right for a
# scatter and wrong here -- an event dropped from the mixture is an event
# silently missing from a population FRACTION, and a fraction is the number this
# module exists to report.


# ---------------------------------------------------------------------------
# item 8a -- joint (n_PCs, K) selection by cross-validated held-out likelihood
# ---------------------------------------------------------------------------


def _residual_loglik(T_test: np.ndarray, eigenvalues: np.ndarray, p: int) -> np.ndarray:
    """Isotropic log-density of the test scores on the DISCARDED axes.

    The PPCA term that makes held-out likelihoods comparable across ``p``. The
    variance is the mean of the **training** eigenvalues on the discarded axes
    -- training, because using the test scores' own spread there would be
    scoring the model on information it was not given, and would reward large
    ``p`` for a reason that has nothing to do with fit quality.

    Args:
        T_test: ``(n_test, d)`` test scores in the full PCA basis.
        eigenvalues: ``(d,)`` training ``explained_variance_``.
        p: PCs handed to the mixture; axes ``p:`` are the discarded ones.

    Returns:
        ``(n_test,)`` summed log-density over the discarded axes; all zeros when
        nothing was discarded.
    """
    if p >= len(eigenvalues):
        return np.zeros(T_test.shape[0], dtype=np.float64)
    s2 = max(float(eigenvalues[p:].mean()), 1e-9)
    tail = T_test[:, p:]
    return (-0.5 * (tail ** 2 / s2 + np.log(2.0 * np.pi * s2))).sum(axis=1)


def _selection_warnings(
    grid: np.ndarray,
    p_grid: tuple[int, ...],
    k_grid: tuple[int, ...],
    n_pcs: int,
    k: int,
) -> tuple[str, ...]:
    """Caveats a caller must see before quoting ``(n_PCs, K)`` as a result."""
    out: list[str] = []
    row = p_grid.index(n_pcs)
    col = k_grid.index(k)
    best = float(grid[row, col])

    # A K chosen by a margin this small is a preference, not a finding. Scale is
    # nats per event, so 0.1 is a hundredth of the spread the grid covers on the
    # owner's data (7.4 -> 16.4 across K).
    k_row = grid[row, :]
    rivals = [float(v) for i, v in enumerate(k_row) if i != col and np.isfinite(v)]
    if rivals:
        gap = best - max(rivals)
        if gap < 0.1:
            runner = k_grid[int(np.nanargmax(np.where(np.arange(len(k_row)) == col, -np.inf, k_row)))]
            out.append(
                f"FLAT OPTIMUM in K: K={k} beat K={runner} by {gap:.3f} nats/event. The "
                f"held-out likelihood is not distinguishing these; do not report the "
                f"number of subpopulations as a measured quantity at this margin."
            )
    if k == k_grid[-1]:
        out.append(
            f"K AT THE GRID EDGE: the selection chose K={k}, the largest value searched, "
            f"so the true optimum may lie outside the grid. Widen k_grid before reading "
            f"this as the number of populations."
        )
    if n_pcs == p_grid[-1]:
        out.append(
            f"n_PCs AT THE GRID EDGE: the selection kept {n_pcs} PCs, every component "
            f"available, so no dimension was judged noise. Either the features carry no "
            f"redundancy or the residual term is not biting."
        )
    return tuple(out)


def select_pcs_and_k(
    Xs: np.ndarray,
    *,
    p_grid: Sequence[int] | None = None,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = 0,
    n_init: int = 2,
    cancel: CancelFlag | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> PCSelection:
    """Choose the PC count and the component count together, by held-out likelihood.

    For each fold: fit PCA on the training rows only (all components), project
    both halves, then for every ``(p, k)`` fit a full-covariance mixture on the
    training scores' first ``p`` axes and score the test scores as
    ``mixture(kept) + isotropic(discarded)``. The reported cell value is the
    mean over folds of the per-fold mean per-event log-likelihood.

    Why the PCA is inside the loop: fitting it once on everything leaks the test
    rows' covariance into the basis the model is scored in, which inflates every
    cell -- and inflates the large-``p`` cells most, because that is where the
    leaked directions live. The selection is the whole point here, so the leak
    would be selecting.

    Args:
        Xs: ``(n_events, n_features)`` **already scaled** feature matrix, e.g.
            from :func:`scale_features`. Must be finite.
        p_grid: PC counts to search. Default ``2 .. n_features``.
        k_grid: Component counts to search. Default :data:`DEFAULT_K_GRID`.
        n_splits: Cross-validation folds.
        random_state: One seed for the fold split, the per-fold PCA and every
            mixture in the loop -- there is deliberately not a second knob.
        n_init: EM restarts per candidate. Low on purpose: this loop runs
            ``n_splits * len(p_grid) * len(k_grid)`` fits, and the final model is
            refit properly by :func:`fit_subpopulations`.
        cancel: Cancel token; the ambient one from
            :mod:`nanoscope.analysis.cancel` is used when ``None``.
        progress: Optional ``(fraction, message)`` sink, ticked once per
            ``(fold, p)`` row of the search. This is the loop a GUI has to be
            able to watch -- it is ``n_splits * len(p_grid) * len(k_grid)``
            mixture fits (480 on the owner's zn_30h) and about a minute of the
            wall clock. It is a report channel only: nothing reads it back, so
            passing one cannot move a cell of the grid.

    Returns:
        A :class:`PCSelection`.

    Raises:
        ValueError: on a non-2-D or non-finite ``Xs``, an empty grid, a
            non-positive entry in either grid, a ``p`` exceeding the feature
            count, ``n_splits < 2``, ``n_init < 1``, or fewer rows than folds.
        OperationCancelled: if the cancel token fires.
    """
    X = np.asarray(Xs, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Xs must be 2-D (n_events, n_features), got shape {X.shape}")
    n, d = X.shape
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Xs contains non-finite values; scale with scale_features(), which "
            "maps them to the scaled median rather than leaving them to poison the fit"
        )
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n < n_splits:
        raise ValueError(f"only {n} event(s) for {n_splits} folds; need at least one per fold")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    ps = tuple(int(p) for p in (range(2, d + 1) if p_grid is None else p_grid))
    ks = tuple(int(k) for k in k_grid)
    if not ps or not ks:
        raise ValueError(f"empty search grid: p_grid={ps}, k_grid={ks}")
    if min(ps) < 1 or max(ps) > d:
        raise ValueError(f"p_grid entries must lie in 1..{d} (n_features), got {ps}")
    if min(ks) < 1:
        raise ValueError(f"k_grid entries must be >= 1, got {ks}")

    grid = np.full((len(ps), len(ks)), np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_rows = n_splits * len(ps)
    for fold, (tr, te) in enumerate(kf.split(X)):
        pca_f = _SKLearnPCA(random_state=random_state).fit(X[tr])
        T_tr = pca_f.transform(X[tr])
        T_te = pca_f.transform(X[te])
        ev = pca_f.explained_variance_
        for pi, p in enumerate(ps):
            if progress is not None:
                done = fold * len(ps) + pi
                progress(
                    done / n_rows,
                    f"Cross-validating (n_PCs, K): fold {fold + 1}/{n_splits}, "
                    f"{p} PCs, K={min(ks)}-{max(ks)}",
                )
            res = _residual_loglik(T_te, ev, p)
            for ki, k in enumerate(ks):
                check_cancelled(cancel)
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    n_init=n_init,
                    random_state=random_state,
                ).fit(T_tr[:, :p])
                ll = gm.score_samples(T_te[:, :p]) + res
                prev = grid[pi, ki] if fold else 0.0
                grid[pi, ki] = prev + ll.mean() / n_splits

    if not np.any(np.isfinite(grid)):
        raise ValueError("no (n_PCs, K) cell could be scored; every fit failed")
    pi, ki = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    n_pcs, k = ps[int(pi)], ks[int(ki)]

    return PCSelection(
        n_pcs=int(n_pcs),
        k=int(k),
        loglik=float(grid[pi, ki]),
        grid=grid,
        p_grid=ps,
        k_grid=ks,
        n_splits=n_splits,
        random_state=int(random_state),
        n_init=n_init,
        warnings=_selection_warnings(grid, ps, ks, int(n_pcs), int(k)),
    )


# ---------------------------------------------------------------------------
# item 8b -- the BIC cross-check, and the final fit
# ---------------------------------------------------------------------------


def bic_ladder(
    Z: np.ndarray,
    *,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    random_state: int = 0,
    n_init: int = 3,
) -> tuple[dict[int, float], int, tuple[str, ...]]:
    """BIC against ``K`` in the chosen PCA space -- the cross-check, not the choice.

    Delegates to :func:`nanoscope.analysis.populations.cluster`, which already
    *is* "GMM with auto-``k`` by minimum BIC" and carries the degenerate-cluster
    and unstable-``k`` warnings with it. Reimplementing the ladder here to save
    a DataFrame round-trip would be a second copy of a decision this package has
    already made once.

    Args:
        Z: ``(n_events, n_pcs)`` PCA scores.
        k_grid: Component counts. Must be a contiguous run starting at 1, since
            ``cluster``'s search is ``1..k_max``.
        random_state: Seed for every fit on the ladder.
        n_init: EM restarts per rung. ``3`` because a one-restart ladder can put
            its minimum at a ``k`` a second restart would have beaten -- see
            ``populations.cluster``.

    Returns:
        ``(bic_by_k, argmin_k, warnings)``.

    Raises:
        ValueError: on a non-2-D ``Z`` or a ``k_grid`` that is not ``1..k_max``.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D (n_events, n_pcs), got shape {Z.shape}")
    ks = tuple(int(k) for k in k_grid)
    if ks != tuple(range(1, len(ks) + 1)):
        raise ValueError(
            f"bic_ladder needs a contiguous k_grid starting at 1 (populations.cluster "
            f"searches 1..k_max); got {ks}"
        )
    cols = [f"pc{i + 1}" for i in range(Z.shape[1])]
    res = _cluster(
        pd.DataFrame(Z, columns=cols),
        cols,
        method="gmm",
        k=None,
        k_max=ks[-1],
        standardize=False,
        random_state=random_state,
        n_init=n_init,
    )
    bic = {int(kk): float(v) for kk, v in (res.bic or {}).items()}
    # Attribute the warnings HERE, at the only place that knows where they came
    # from. They describe the BIC ladder's own argmin fit -- a different K, and a
    # different component numbering, from the model this module reports. Merged
    # flat into the report's caveats they read as complaints about the reported
    # table: "cluster 5 (n=10)" printed under a table listing subpops 0-3 sends a
    # reader looking for a subpopulation that was never claimed.
    chosen = int(res.chosen_k)
    notes = tuple(
        f"BIC CROSS-CHECK (its own K={chosen} fit, NOT the reported model): {w}"
        for w in res.warnings
    )
    return bic, chosen, notes


def _fit_warnings(
    sizes: dict[int, int],
    n: int,
    conf_frac: float,
    k: int,
    bic_k: int | None,
) -> tuple[str, ...]:
    """Caveats about a fitted mixture, before its fractions are believed."""
    out: list[str] = []
    if bic_k is not None and int(bic_k) != int(k):
        out.append(
            f"CRITERIA DISAGREE: cross-validated held-out likelihood chose K={k}; BIC "
            f"chose K={bic_k}. BIC penalises parameter count, not overfitting directly, "
            f"and with a full covariance it keeps buying components to chase a heavy "
            f"tail. The CV answer is the reported one; quote both."
        )
    if conf_frac < _CONF_FRAC_FLOOR:
        out.append(
            f"LOW ASSIGNMENT CONFIDENCE: only {conf_frac:.1%} of events exceed a "
            f"{CONFIDENCE_FLOOR:.0%} posterior. The components are overlapping shoulders "
            f"of one cloud more than they are separated populations; per-event labels "
            f"from this fit are not safe to condition on."
        )
    floor = max(5, int(round(0.01 * max(n, 1))))
    tiny = {c: m for c, m in sorted(sizes.items()) if m < floor}
    if tiny:
        named = ", ".join(f"subpop {c} (n={m})" for c, m in tiny.items())
        out.append(
            f"DEGENERATE COMPONENT(S): {named} — below {floor} events (1% of {n}, floor 5). "
            f"A component this small is the mixture spending a density on outliers; its "
            f"fraction is not a population fraction."
        )
    return tuple(out)


def fit_subpopulations(
    Xs: np.ndarray,
    *,
    n_pcs: int,
    k: int,
    index: pd.Index | None = None,
    random_state: int = 0,
    n_init: int = 10,
    bic_check: bool = True,
    bic_k_grid: Sequence[int] = DEFAULT_K_GRID,
    bic_n_init: int = 3,
    cancel: CancelFlag | None = None,
) -> SubpopFit:
    """Fit the mixture at a chosen ``(n_pcs, k)`` and number its components by size.

    PCA is fit on all the data here -- unlike inside :func:`select_pcs_and_k`,
    where holding it out was the point. There is nothing being selected at this
    stage; the model is being *fit*, and fitting it in a basis derived from 80 %
    of the events would just be a worse basis.

    Args:
        Xs: ``(n_events, n_features)`` scaled feature matrix.
        n_pcs: PCs to keep.
        k: Mixture components.
        index: Index for the returned ``labels``. Defaults to a ``RangeIndex``.
            Pass the source table's index to join labels back on -- and pass the
            *original* index, not a reset one: a positional label array assigned
            into a label-indexed slice is correct only while the two agree.
        random_state: Seed for the PCA and the mixture.
        n_init: EM restarts for the final fit. Higher than the search's, because
            this one is the answer.
        bic_check: Compute the BIC ladder as a cross-check.
        bic_k_grid: Component counts for that ladder.
        bic_n_init: EM restarts per ladder rung.
        cancel: Cancel token; ambient when ``None``.

    Returns:
        A :class:`SubpopFit` with components renumbered largest-first.

    Raises:
        ValueError: on a non-2-D or non-finite ``Xs``, ``n_pcs`` outside
            ``1..min(n_events, n_features)``, ``k`` outside ``1..n_events``,
            ``n_init < 1``, or an ``index`` of the wrong length.
        OperationCancelled: if the cancel token fires.
    """
    X = np.asarray(Xs, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Xs must be 2-D (n_events, n_features), got shape {X.shape}")
    n, d = X.shape
    if not np.all(np.isfinite(X)):
        raise ValueError("Xs contains non-finite values; scale with scale_features()")
    n_pcs = int(n_pcs)
    k = int(k)
    limit = min(n, d)
    if not (1 <= n_pcs <= limit):
        raise ValueError(
            f"n_pcs={n_pcs} must lie in 1..min(n_events={n}, n_features={d})={limit}"
        )
    if not (1 <= k <= n):
        raise ValueError(f"k={k} must lie in 1..n_events={n}")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")
    if index is not None and len(index) != n:
        raise ValueError(
            f"index has {len(index)} entries for {n} events; labels would misalign"
        )

    check_cancelled(cancel)
    pca = _SKLearnPCA(n_components=n_pcs, random_state=random_state).fit(X)
    Z = pca.transform(X)

    check_cancelled(cancel)
    gm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        random_state=random_state,
    ).fit(Z)

    raw = gm.predict(Z)
    post = gm.predict_proba(Z)

    # Renumber largest-first. `argsort(-counts)` is stable, so two components of
    # equal size keep their EM order rather than swapping between runs.
    order = np.argsort(-np.bincount(raw, minlength=k), kind="stable")
    remap = {int(o): i for i, o in enumerate(order)}
    labels_arr = np.array([remap[int(v)] for v in raw], dtype="int64")
    post = post[:, order]
    conf = post.max(axis=1)

    idx = pd.RangeIndex(n) if index is None else pd.Index(index)
    labels = pd.Series(labels_arr, index=idx, name="subpop")
    sizes = {int(c): int(np.sum(labels_arr == c)) for c in range(k)}
    conf_frac = float((conf > CONFIDENCE_FLOOR).mean())

    bic: dict[int, float] | None = None
    bic_k: int | None = None
    bic_notes: tuple[str, ...] = ()
    if bic_check:
        check_cancelled(cancel)
        bic, bic_k, bic_notes = bic_ladder(
            Z, k_grid=bic_k_grid, random_state=random_state, n_init=bic_n_init
        )

    notes = _fit_warnings(sizes, n, conf_frac, k, bic_k) + bic_notes

    return SubpopFit(
        labels=labels,
        posteriors=np.asarray(post, dtype=np.float64),
        confidence=np.asarray(conf, dtype=np.float64),
        conf_frac=conf_frac,
        conf_median=float(np.median(conf)),
        scores=np.asarray(Z, dtype=np.float64),
        means=np.asarray(gm.means_[order], dtype=np.float64),
        weights=np.asarray(gm.weights_[order], dtype=np.float64),
        sizes=sizes,
        n_pcs=n_pcs,
        k=k,
        order=np.asarray(order, dtype=np.int64),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        bic=bic,
        bic_k=bic_k,
        warnings=notes,
        meta={
            "random_state": int(random_state),
            "n_init": n_init,
            "bic_n_init": int(bic_n_init) if bic_check else None,
            "n_events": int(n),
            "n_features": int(d),
        },
    )


# ---------------------------------------------------------------------------
# item 10 -- bootstrap CIs on the population fractions
# ---------------------------------------------------------------------------


def _ci_warnings(fractions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[str, ...]:
    """Detect the two ways these intervals lie, and say so.

    The first is arithmetic: an interval wider than the number it brackets is
    not measuring that number. The second is the label-switching signature --
    subpopulation A's interval containing subpopulation B's point estimate means
    some resamples matched A's component onto B's, and the spread is the swap,
    not the sampling.
    """
    out: list[str] = []
    f = np.asarray(fractions, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(f)
    if not np.any(ok):
        return ()

    wide = [
        int(i)
        for i in range(f.size)
        if ok[i] and f[i] > 0 and (hi[i] - lo[i]) > _CI_REL_WIDTH_FLOOR * f[i]
    ]
    if wide:
        named = ", ".join(
            f"subpop {i} ({f[i]:.3f}, CI {lo[i]:.3f}-{hi[i]:.3f})" for i in wide
        )
        out.append(
            f"UNINFORMATIVE INTERVAL(S): {named} — the 95% interval is wider than the "
            f"fraction it brackets. Quote the point estimate and the caveat, not the "
            f"interval as an error bar."
        )

    swaps: list[str] = []
    for i in range(f.size):
        if not ok[i]:
            continue
        caught = [
            int(j)
            for j in range(f.size)
            if j != i and ok[j] and lo[i] <= f[j] <= hi[i]
        ]
        if caught:
            swaps.append(f"subpop {i}'s CI contains subpop {caught}'s fraction")
    if swaps:
        out.append(
            "LABEL SWITCHING IN THE BOOTSTRAP: "
            + "; ".join(swaps)
            + ". Hungarian matching on component means minimises total distance, which "
            "on a resample where two components trade places is not the assignment that "
            "preserves identity. The width is the swap, not the sampling uncertainty; "
            "these intervals are NOT calibrated."
        )
    return tuple(out)


def bootstrap_fractions(
    Z: np.ndarray,
    reference_means: np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    n_init: int = 2,
    refit_seed_base: int = 0,
    point_fractions: np.ndarray | None = None,
    cancel: CancelFlag | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> FractionCI:
    """Percentile intervals on the mixing fractions, components matched by Hungarian.

    Resample the events with replacement, refit the mixture on the resample,
    then decide which refit component corresponds to which reference component
    by minimising the total Euclidean distance between component means
    (:func:`scipy.optimize.linear_sum_assignment`). Without the matching step
    the fractions would be averaged across arbitrary EM numberings and every
    interval would come out the same width.

    **Read the intervals with the warnings attached.** Hungarian matching is
    optimal for the assignment problem, not for identity: see :func:`_ci_warnings`.

    Skipped entirely inside :func:`nanoscope.analysis.stats.point_fits_only` --
    this is 200 mixture fits, and the module's standing rule is that a CI nobody
    reads must not be paid for. The returned object then has ``computed is False``
    and NaN bounds; nothing else about the fit changes.

    Args:
        Z: ``(n_events, n_pcs)`` scores the reference mixture was fit in.
        reference_means: ``(k, n_pcs)`` reference component means, already in
            the reported (size) order -- e.g. :attr:`SubpopFit.means`.
        n_boot: Resamples. ``0`` skips the bootstrap.
        seed: Seed of the RNG drawing the resampling indices. The draws are
            sequential, so this seed fixes the whole run.
        n_init: EM restarts per refit.
        refit_seed_base: Each resample ``b`` refits with
            ``random_state = refit_seed_base + b`` -- a *varying* seed on
            purpose, so the interval includes EM's own initialisation
            sensitivity rather than holding it fixed and understating the spread.
        point_fractions: ``(k,)`` point estimates, used only to detect the
            label-switching signature. Defaults to the reference weights being
            unavailable, in which case only the width check runs.
        cancel: Cancel token; ambient when ``None``.
        progress: Optional ``(fraction, message)`` sink, ticked every 10th
            resample -- often enough to see it move, rarely enough that a Qt
            queued signal per refit does not become the cost. A report channel
            only; nothing reads it back.

    Returns:
        A :class:`FractionCI`.

    Raises:
        ValueError: on a non-2-D ``Z``, ``reference_means`` whose second axis
            does not match ``Z``, ``n_boot < 0``, or ``n_init < 1``.
        OperationCancelled: if the cancel token fires.
    """
    Z = np.asarray(Z, dtype=np.float64)
    M = np.asarray(reference_means, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D (n_events, n_pcs), got shape {Z.shape}")
    if M.ndim != 2 or M.shape[1] != Z.shape[1]:
        raise ValueError(
            f"reference_means must be (k, n_pcs={Z.shape[1]}), got shape {M.shape}"
        )
    n_boot = int(n_boot)
    if n_boot < 0:
        raise ValueError(f"n_boot must be >= 0, got {n_boot}")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    k = M.shape[0]
    n = Z.shape[0]
    nan_k = np.full(k, np.nan, dtype=np.float64)

    if n_boot == 0 or not ci_wanted():
        return FractionCI(
            lo=nan_k.copy(),
            hi=nan_k.copy(),
            samples=np.empty((0, k), dtype=np.float64),
            n_ok=0,
            n_boot=n_boot,
            seed=int(seed),
        )

    rng = np.random.default_rng(seed)
    samples = np.full((n_boot, k), np.nan, dtype=np.float64)
    for b in range(n_boot):
        check_cancelled(cancel)
        if progress is not None and b % _BOOT_PROGRESS_STRIDE == 0:
            progress(b / n_boot, f"Bootstrapping fractions: resample {b + 1}/{n_boot}")
        idx = rng.integers(0, n, n)
        try:
            gb = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                random_state=refit_seed_base + b,
            ).fit(Z[idx])
        except Exception:
            # A refit that will not converge is a missing replicate, not a zero.
            # The row stays NaN and the percentiles below are nan-aware.
            continue
        D = np.linalg.norm(gb.means_[:, None, :] - M[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(D)
        samples[b] = gb.weights_[rows[np.argsort(cols)]]

    n_ok = int(np.sum(np.all(np.isfinite(samples), axis=1)))
    if n_ok == 0:
        return FractionCI(
            lo=nan_k.copy(),
            hi=nan_k.copy(),
            samples=samples,
            n_ok=0,
            n_boot=n_boot,
            seed=int(seed),
            warnings=(
                f"BOOTSTRAP FAILED: none of the {n_boot} resamples produced a mixture "
                f"that converged, so there are no intervals. The fractions are point "
                f"estimates with no uncertainty attached.",
            ),
        )
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)

    notes: list[str] = []
    if n_ok < n_boot:
        notes.append(
            f"BOOTSTRAP REPLICATES LOST: {n_boot - n_ok} of {n_boot} refits did not "
            f"converge and were dropped. The intervals rest on {n_ok} replicates."
        )
    if point_fractions is not None:
        notes.extend(_ci_warnings(np.asarray(point_fractions, dtype=np.float64), lo, hi))

    return FractionCI(
        lo=np.asarray(lo, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
        samples=samples,
        n_ok=n_ok,
        n_boot=n_boot,
        seed=int(seed),
        warnings=tuple(notes),
    )


# ---------------------------------------------------------------------------
# H4 -- is the reported partition SEPARATED, and does it hold still in time?
# ---------------------------------------------------------------------------


def measure_separation(
    fit: SubpopFit,
    *,
    random_state: int = 0,
    reference_k: int = SEPARATION_REFERENCE_K,
    ceiling_k_grid: Sequence[int] = SEPARATION_CEILING_K_GRID,
    max_n: int = SEPARATION_MAX_N,
) -> Separation | None:
    """Measure how far apart :class:`SubpopFit`'s own classes are.

    Reads nothing but the fit: the scores it was fitted in, the labels it
    assigned and the component means it assigned them from. It does not refit
    the mixture and cannot move a single number the mixture produced.

    Args:
        fit: The fitted mixture.
        random_state: Seeds the subsample (above ``max_n``) and the reference
            k-means, so two runs of the same report give the same silhouette.
        reference_k: K for the yardstick k-means quoted beside the mixture's own
            silhouette on the header.
        ceiling_k_grid: K values the SEPARATION CEILING is maximised over. The
            mixture's own ``K`` and ``reference_k`` are always included.
        max_n: Above this many events the silhouette is measured on a seeded
            subsample rather than on every pair.

    Returns:
        A :class:`Separation`, or ``None`` at ``K = 1`` / fewer than two
        distinct labels, where a silhouette is undefined and a zero would be a
        fabricated reassurance.
    """
    labels = np.asarray(fit.labels.to_numpy(), dtype=np.int64)
    scores = np.asarray(fit.scores, dtype=np.float64)
    n = int(labels.size)
    uniq = np.unique(labels)
    if uniq.size < 2 or n <= uniq.size:
        return None

    from sklearn.metrics import silhouette_samples

    # The nearest-centre check uses EVERY event -- it is O(n*k) in TIME, so there
    # is no reason to subsample it, and it is the number a reader can act on.
    # Chunked, though: the obvious `scores[:, None, :] - means[None, :, :]` is
    # O(n*k*p) in SPACE, which is 41 MB at 50 000 events and grows with the
    # recording. `pairwise_distances_argmin` never materialises the block.
    from sklearn.metrics import pairwise_distances_argmin

    nearest = pairwise_distances_argmin(scores, fit.means)
    nearer_other = nearest != labels

    idx = np.arange(n)
    if n > int(max_n):
        rng = np.random.default_rng(int(random_state))
        idx = np.sort(rng.choice(n, size=int(max_n), replace=False))
        # A subsample that lost a whole class cannot score it. Fall back to the
        # full set rather than silently reporting a silhouette for k-1 classes.
        if np.unique(labels[idx]).size < uniq.size:
            idx = np.arange(n)
    sil = silhouette_samples(scores[idx], labels[idx])
    sil_labels = labels[idx]

    per: dict[int, tuple[int, float, float]] = {}
    for c in uniq.tolist():
        in_c = labels == c
        in_c_sil = sil_labels == c
        per[int(c)] = (
            int(in_c.sum()),
            float(sil[in_c_sil].mean()) if in_c_sil.any() else float("nan"),
            float(nearer_other[in_c].mean()),
        )

    # The reference row. Every K in ONE loop, in the same coordinates and on the
    # same (possibly subsampled) events as the mixture's own silhouette, because
    # the ceiling is only a bound on the mixture's number if both were measured
    # on the same events. ~7 k-means fits and 7 silhouettes: about a second at
    # SEPARATION_MAX_N, and it is the difference between a caveat that fires on
    # every dataset and one that discriminates.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    profile: dict[int, float] = {}
    wanted = sorted({int(k) for k in ceiling_k_grid if int(k) >= 2}
                    | {int(reference_k)} | ({int(fit.k)} if int(fit.k) >= 2 else set()))
    for kk in wanted:
        if not (2 <= kk <= len(scores[idx]) - 1):
            continue
        km = KMeans(n_clusters=kk, n_init=10, random_state=int(random_state)).fit(
            scores[idx]
        )
        if np.unique(km.labels_).size >= 2:
            profile[kk] = float(silhouette_score(scores[idx], km.labels_))
    ref: float | None = profile.get(int(reference_k))

    return Separation(
        silhouette=float(sil.mean()),
        neg_fraction=float((sil < 0).mean()),
        nearer_other_fraction=float(nearer_other.mean()),
        per_subpop=per,
        reference_k=int(reference_k),
        reference_silhouette=ref,
        n_used=int(idx.size),
        n_events=n,
        n_pcs=int(fit.n_pcs),
        subsampled=bool(idx.size < n),
        reference_profile=profile,
    )


def _separation_warnings(sep: Separation | None, k: int) -> tuple[str, ...]:
    """Caveats that must fire BEFORE a class count is quoted as a finding."""
    if sep is None:
        return ()
    out: list[str] = []
    if not sep.substantial:
        out.append(
            f"NOT SEPARATED: the reported {k}-class partition scores a mean silhouette "
            f"of {sep.silhouette:.3f} in the {sep.n_pcs}-PC space it was fitted in. On "
            f"Kaufman & Rousseeuw's bands that is the lowest one -- NO SUBSTANTIAL "
            f"STRUCTURE (<= {SEPARATION_FLOOR:g}). {100 * sep.neg_fraction:.1f} % of events "
            f"have a NEGATIVE silhouette and {100 * sep.nearer_other_fraction:.1f} % sit "
            f"nearer another class's centre than their own. Held-out likelihood, BIC and "
            f"the posteriors all say the mixture DESCRIBES these events well; none of "
            f"them says the classes are apart, and this one says they are not. Contiguous "
            f"slices of one distribution score exactly like this."
        )
    elif sep.silhouette <= SEPARATION_WEAK_CEILING:
        out.append(
            f"WEAKLY SEPARATED: the reported {k}-class partition scores a mean silhouette "
            f"of {sep.silhouette:.3f} ({100 * sep.nearer_other_fraction:.1f} % of events "
            f"nearer another class's centre). That is Kaufman & Rousseeuw's 'weak, could "
            f"be artificial' band; the count is a hypothesis, not a measurement."
        )
    # THE COMPARISON THAT USED TO LIVE HERE compared this mixture's silhouette
    # against ONE k-means at ONE other K and read the difference causally. Two
    # things are wrong with that and both were measured, not argued: the mean
    # silhouette falls with K on any continuum, so a K=3 partition out-scoring a
    # K=7 mixture is arithmetic; and k-means minimises exactly the within-cluster
    # scatter the silhouette penalises, so it out-scores a full-covariance
    # mixture at the SAME K too (apo -700 mV K=7: 0.166 vs 0.052; Zn2+ control
    # K=4: 0.385 vs 0.110). Under either rule the caveat fires on every real
    # dataset in this project, including the positive control -- which is the
    # wallpaper failure the separation statistic was added to remove.
    #
    # What the reference row DOES support needs no comparability across K,
    # because it is an existence claim about the CLOUD rather than a verdict on
    # the count: if the best partition the biased-in-its-favour method can find
    # at ANY searched K is still inside the lowest band, no partition of these
    # coordinates is separated. That is silent on the Zn2+ control (ceiling 0.678)
    # and fires on apo -700 mV (ceiling 0.223), which is the whole point.
    ceil = sep.ceiling_silhouette
    if ceil is not None and ceil <= SEPARATION_FLOOR and not sep.substantial:
        at_k = sep.ceiling_k
        searched = sorted(sep._ceiling_profile())
        span = (f"K={searched[0]}"
                if len(searched) == 1
                else f"K={searched[0]}-{searched[-1]}"
                if searched == list(range(searched[0], searched[-1] + 1))
                else "K=" + ",".join(str(k) for k in searched))
        out.append(
            f"NOT THE BEST-SEPARATED PARTITION ON OFFER — AND NEITHER IS ANY OTHER: the "
            f"most separated partition these coordinates admit anywhere in the searched "
            f"range ({span}) is a plain k-means at K={at_k} scoring {ceil:.3f}, against this "
            f"mixture's {sep.silhouette:.3f} — and {ceil:.3f} is itself inside Kaufman & "
            f"Rousseeuw's lowest band (<= {SEPARATION_FLOOR:g}). k-means minimises exactly "
            f"the within-class scatter a silhouette penalises, so that number is a CEILING "
            f"on how separated any partition of this cloud can be made to look, not a rival "
            f"result -- and it is NOT a recommendation to report {at_k} subpopulations. "
            f"The honest reading is not that some other K in that range would have been "
            f"separated: it is that 'how many classes' has no separated answer in these "
            f"coordinates at any K the search looked at."
        )
    worst = [
        (c, n_c, s)
        for c, (n_c, s, _) in sorted(sep.per_subpop.items())
        if np.isfinite(s) and s < 0.0
    ]
    if worst:
        named = "; ".join(
            f"subpop {c} (n={n_c}, silhouette {s:.3f}, "
            f"{100 * sep.per_subpop[c][2]:.0f} % nearer another centre)"
            for c, n_c, s in worst
        )
        out.append(
            f"CLASS(ES) INSIDE ANOTHER CLASS: {named}. A negative mean silhouette means "
            f"the average member of that class is closer to a different class than to "
            f"its own; its row in the population table is a profile of a region of the "
            f"cloud, not of a population."
        )
    return tuple(out)


def stability_band_for(pooled: float) -> tuple[float, float]:
    """The ``early_frac`` band a class is "present throughout" in, on THIS recording.

    :data:`STABILITY_BAND` is the band for a recording that delivered half its
    events into each half of itself. It encodes an odds ratio, not a pair of
    numbers: 0.75 against 0.25 is "at least 3:1 into one half"
    (:data:`STABILITY_ODDS_RATIO`). On a recording that is itself lopsided, the
    ratio is the part that still means something and the endpoints are not --
    a recording that delivered 90 % of its events into its first half hands
    every class an ``early_frac`` near 0.90, and testing those against 0.25-0.75
    condemns all of them for being shaped exactly like the recording they came
    out of. That is the comparison :func:`_stability_warnings` documented and
    did not make.

    So the band travels with the recording: the class's odds of being early are
    compared against the RECORDING's odds of being early, and a class is flagged
    when its own odds are more than :data:`STABILITY_ODDS_RATIO` times, or less
    than a ``1/ratio`` of, the pooled odds.

    At ``pooled = 0.5`` this returns :data:`STABILITY_BAND` exactly, so the
    population table's caption still reads straight off the printed column in
    the ordinary case; it is only a lopsided recording that moves it.

    Args:
        pooled: The recording's own pooled first-half fraction, in ``[0, 1]``.

    Returns:
        ``(lo, hi)`` clipped into ``[0, 1]``. A degenerate ``pooled`` of 0 or 1
        (every event in one half) collapses the band onto that same endpoint,
        which is right: when the recording delivered nothing into one half, no
        class in it can be lopsided *relative to the recording*, and the
        recording-level caveat is the one that has something to say.
    """
    p = float(pooled)
    if not np.isfinite(p):
        return STABILITY_BAND
    p = min(max(p, 0.0), 1.0)
    r = float(STABILITY_ODDS_RATIO)
    if p <= 0.0:
        return (0.0, 0.0)
    if p >= 1.0:
        return (1.0, 1.0)
    odds = p / (1.0 - p)
    hi_odds, lo_odds = odds * r, odds / r
    return (lo_odds / (1.0 + lo_odds), hi_odds / (1.0 + hi_odds))


def _odds_clause(ef: float, pooled: float) -> str:
    """The quantity the gate actually compared, printed beside the class.

    The gate is an ODDS RATIO against the recording's own early odds, and until
    this clause existed the reader was given only the two fractions and a
    sentence claiming something else about them ("3:1 into one half"). Naming
    the measured ratio is what lets a reader see *how* lopsided the class is
    relative to its own recording, and see that a class at 0.50 on a 0.10
    recording is flagged for being 9x too early rather than for being
    concentrated in a half.

    Both inputs are the values as PRINTED (see :data:`_STABILITY_DECIMALS`), so
    the ratio recomputes from the numbers on screen.
    """
    if not (0.0 < pooled < 1.0):
        return ""
    if ef >= 1.0:
        return ", every one of them in the first half"
    if ef <= 0.0:
        return ", not one of them in the first half"
    r = (ef / (1.0 - ef)) / (pooled / (1.0 - pooled))
    if r >= 1.0:
        return f", early odds {r:.1f}x the recording's"
    return f", early odds 1/{1.0 / r:.1f} of the recording's"


def _stability_warnings(table: pd.DataFrame) -> tuple[str, ...]:
    """Read ``early_frac`` -- the one column that can see non-stationarity.

    It was computed, rounded, printed under a caption saying what an extreme
    value means, and then never compared against anything. This is that
    comparison, on the caption's own terms: at least 3:1 concentrated into one
    half of the recording (:data:`STABILITY_BAND`, generalised to a lopsided
    recording by :func:`stability_band_for`).

    Two different things can be true and they are reported separately, because
    the reader's response differs. If a *class* is lopsided while the recording
    as a whole is not, that class is a time slice. If the *recording* is
    lopsided, the event rate itself changed and no class in it is "present
    throughout" -- and the fair comparison for each class is then the pooled
    fraction, not 0.5. THAT SECOND COMPARISON IS THE ONE THIS FUNCTION MAKES:
    stating it in the recording-level caveat and then testing every class
    against 0.25-0.75 anyway is a note that contradicts itself on the next line.
    """
    if "early_frac" not in table.columns or "n" not in table.columns:
        return ()
    ef = pd.to_numeric(table["early_frac"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(table["n"], errors="coerce").to_numpy(dtype=float)
    ids = (
        table["subpop"].to_numpy()
        if "subpop" in table.columns
        else np.arange(len(table))
    )
    ok = np.isfinite(ef) & np.isfinite(n) & (n > 0)
    if not ok.any():
        return ()
    nd = _STABILITY_DECIMALS
    # EVERY NUMBER BELOW IS THE NUMBER THE READER SEES. `ef` arrived already
    # rounded from `population_table`; `pooled` and the band are rounded here,
    # once, and the comparisons are made on the rounded values. See
    # `_STABILITY_DECIMALS` for the sentence that could otherwise contradict the
    # column it was printed under.
    ef = np.round(ef, nd)
    pooled = round(float((ef[ok] * n[ok]).sum() / n[ok].sum()), nd)

    band_lo, band_hi = STABILITY_BAND

    out: list[str] = []
    if not (band_lo <= pooled <= band_hi):
        # `pooled` IS THE FIRST-HALF FRACTION (see `population_table`: early_frac
        # counts rows whose `time_half` is "early"). The sentence names a half, so
        # the number in front of it has to be that half's share -- printing the
        # first-half fraction in front of the word "second" turned a 90 %
        # back-loaded recording into the claim that 10 % of its events fall in
        # its second half. The pooled value still appears, once, as the thing
        # every early_frac is to be read against, where it IS a first-half
        # fraction and is labelled as one.
        back_loaded = pooled < band_lo
        half = "second" if back_loaded else "first"
        share = (1.0 - pooled) if back_loaded else pooled
        out.append(
            f"THE RECORDING IS NOT STATIONARY: {100 * share:.0f} % of all gated events "
            f"fall in the {half} half of it. No class below is 'present throughout the "
            f"recording' in the sense the population table's caption means, and every "
            f"early_frac (a FIRST-half fraction) must be read against this "
            f"{pooled:.{nd}f}, not against 0.50."
        )

    if not (0.0 < pooled < 1.0):
        # A DEGENERATE RECORDING HAS NO ODDS TO COMPARE AGAINST, and the whole
        # class-level sentence is about odds. When the recording delivered
        # (as printed) nothing into one of its halves, `stability_band_for`
        # collapses the band onto that endpoint -- as its own docstring says it
        # should, because "no class in it can be lopsided *relative to the
        # recording*, and the recording-level caveat is the one that has
        # something to say". The gate did not read that: it condemned every
        # class with a non-zero `early_frac` for being outside `0.00-0.00`,
        # with no ratio beside it because none exists. The recording-level
        # sentence above has already told the reader the whole of it.
        return tuple(out)

    # ...and here is where it is read against it. The band is the recording's
    # own, so a class is flagged for being lopsided RELATIVE TO the event supply
    # it was drawn from, not for being shaped like it: on a 90 % front-loaded
    # recording every class sits near 0.90 and the fixed 0.25-0.75 band condemned
    # all of them, one line under the sentence promising the opposite.
    lo, hi = stability_band_for(pooled)
    lo, hi = round(lo, nd), round(hi, nd)
    bad = ok & ~((ef >= lo) & (ef <= hi))
    if bad.any():
        frac = float(n[bad].sum() / n[ok].sum())
        named = "; ".join(
            f"subpop {ids[i]} (n={int(n[i])}, early_frac {ef[i]:.{nd}f}"
            f"{_odds_clause(float(ef[i]), pooled)})"
            for i in np.flatnonzero(bad)
        )
        # WHAT THE SENTENCE MAY CLAIM IS WHAT THE GATE MEASURED. The gate is
        # "this class's odds of being early, against THIS RECORDING's odds of
        # being early, by more than STABILITY_ODDS_RATIO" -- a relative test,
        # and on a lopsided recording it is not the same statement as "3:1 into
        # one half". It said the latter anyway, so at pooled 0.10 a class
        # sitting at exactly 0.50 -- one event in each half, the most balanced a
        # class can be -- was announced as "at least 3:1 concentrated into one
        # half of the recording", which is false of it.
        #
        # THE ABSOLUTE CLAIM IS A PROPERTY OF EACH CLASS, NOT OF THE RECORDING,
        # so it is checked on each class: it is made only when EVERY class named
        # in this sentence is also outside STABILITY_BAND, where "3:1 into one
        # half" is literally true of it. Gating it on the recording's own pooled
        # value instead is the same category error one level up -- at pooled
        # 0.34 a class at 0.70 is 2.33:1, and would have been announced as 3:1.
        absolute = (
            f", i.e. at least {STABILITY_ODDS_RATIO:g}:1 concentrated into one half of "
            f"the recording"
            if bool(((ef[bad] < band_lo) | (ef[bad] > band_hi)).all())
            else ""
        )
        out.append(
            f"TIME-UNSTABLE CLASS(ES): {named} — outside {lo:.{nd}f}-{hi:.{nd}f}{absolute}. "
            f"That band is this recording's OWN pooled early fraction of {pooled:.{nd}f} "
            f"widened by {STABILITY_ODDS_RATIO:g}:1 in the ODDS, so what is measured is "
            f"each class's odds of falling in the first half against the recording's "
            f"own — on a lopsided recording that is not the same statement as '3:1 into "
            f"one half', and the recording's own shape is not held against a class. "
            f"(On a stationary recording, pooled 0.50, the band is exactly the "
            f"{band_lo:g}-{band_hi:g} the population table's caption describes.) Together "
            f"{100 * frac:.1f} % of the events. The population table's own caption says "
            f"what that is: drift, fouling or a pore that changed, not an event class. "
            f"These recordings were also acquired as a voltage series, so a class that "
            f"lives in one half of one recording is confounded with elapsed time."
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# item 11 -- the population table
# ---------------------------------------------------------------------------


def population_table(
    df: pd.DataFrame,
    fit: SubpopFit,
    *,
    duration_s: float | None = None,
    ci: FractionCI | None = None,
    decimals: Mapping[str, int] | None = TABLE_DECIMALS,
) -> pd.DataFrame:
    """The reportable per-subpopulation summary: how many, how often, what shape.

    One row per subpopulation, ordered by size. Every profile column is a
    **median** over the events in that subpopulation.

    ``early_frac`` is the stability check and the most easily misread column: it
    is the fraction of a subpopulation's events that fell in the first half of
    the recording, so ``~0.5`` means the class is present throughout and a value
    near 0 or 1 means it is drift, fouling or a pore that changed -- not an
    event class. It needs a ``time_half`` column in ``df``; without one the
    column is NaN and the caller is told. **That sentence is only true of a
    ``time_half`` split on the recording's clock** -- see
    :func:`~nanoscope.analysis.event_features.time_half`, whose fallback median
    split makes the pooled value 0.50 by construction and this column a
    statement about event order rather than about elapsed time.

    Args:
        df: The per-event table the fit came from. Must be aligned with
            ``fit.labels`` (same index, same length).
        fit: The fitted mixture.
        duration_s: Recording duration, for ``rate_per_s``. ``None`` leaves that
            column NaN -- a rate needs a denominator and guessing one from the
            last event onset would understate it.
        ci: Bootstrap intervals for ``frac_lo95``/``frac_hi95``. ``None`` leaves
            them NaN.
        decimals: Per-column rounding, defaulting to :data:`TABLE_DECIMALS`
            (the reported precision). Pass ``None`` for unrounded values.

    Returns:
        A ``DataFrame`` indexed ``0..k-1`` by ``subpop``.

    Raises:
        ValueError: if ``df`` and ``fit.labels`` are not aligned, or
            ``duration_s`` is non-positive.
    """
    if len(df) != len(fit.labels):
        raise ValueError(
            f"table has {len(df)} rows but the fit labelled {len(fit.labels)} events"
        )
    if not df.index.equals(fit.labels.index):
        raise ValueError(
            "df.index and fit.labels.index differ; a positional label array assigned "
            "into a label-indexed frame misaligns every row silently. Pass the SAME "
            "frame the features were taken from, unfiltered and unsorted."
        )
    if duration_s is not None and not (float(duration_s) > 0):
        raise ValueError(f"duration_s must be > 0, got {duration_s!r}")

    labels = fit.labels.to_numpy()
    conf = fit.confidence
    n_total = len(df)
    has_time_half = _TIME_HALF_COLUMN in df.columns
    has_dwell = _DWELL_COLUMN in df.columns

    rows: list[dict[str, Any]] = []
    for k in range(fit.k):
        mask = labels == k
        sub = df.loc[mask]
        n_k = int(mask.sum())
        row: dict[str, Any] = {
            "subpop": k,
            "n": n_k,
            "fraction": (n_k / n_total) if n_total else np.nan,
            "frac_lo95": float(ci.lo[k]) if ci is not None and k < ci.lo.size else np.nan,
            "frac_hi95": float(ci.hi[k]) if ci is not None and k < ci.hi.size else np.nan,
            "rate_per_s": (n_k / float(duration_s)) if duration_s is not None else np.nan,
        }
        if has_dwell:
            d = pd.to_numeric(sub[_DWELL_COLUMN], errors="coerce")
            row["dwell_p50_ms"] = float(d.median()) if n_k else np.nan
            row["dwell_p90_ms"] = float(d.quantile(0.9)) if n_k else np.nan
        else:
            row["dwell_p50_ms"] = np.nan
            row["dwell_p90_ms"] = np.nan
        for out_col, src_col in _PROFILE_COLUMNS:
            if src_col in sub.columns and n_k:
                row[out_col] = float(
                    pd.to_numeric(sub[src_col], errors="coerce").median()
                )
            else:
                row[out_col] = np.nan
        row["conf_med"] = float(np.median(conf[mask])) if n_k else np.nan
        if has_time_half and n_k:
            row["early_frac"] = float(
                (sub[_TIME_HALF_COLUMN].astype(str) == _EARLY_LABEL).mean()
            )
        else:
            row["early_frac"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if decimals:
        for col, nd in decimals.items():
            if col in out.columns:
                out[col] = out[col].astype(float).round(int(nd))
    return out


# ---------------------------------------------------------------------------
# item 12 -- crosstab against the exploratory labels
# ---------------------------------------------------------------------------


def label_crosstab(
    fit: SubpopFit,
    other: pd.Series | Sequence[Any],
    *,
    name: str = "hdbscan",
) -> pd.DataFrame:
    """Contingency table of mixture subpopulations against exploratory labels.

    The point of the crosstab is agreement *between two methods that share no
    machinery* -- a density model and a density-based clusterer landing on the
    same partition is evidence; either one alone is a hypothesis. It is also
    where HDBSCAN's noise label (``-1``) becomes readable: a subpopulation that
    HDBSCAN mostly called noise is the one built out of the sparse tail.

    Args:
        fit: The fitted mixture.
        other: Exploratory labels, one per event. A ``Series`` is aligned by
            **index**; anything else is taken positionally and must be the same
            length as ``fit.labels``. Events whose exploratory label is missing
            are excluded from the table (pandas' own convention), so the row
            sums need not equal the subpopulation sizes -- compare against
            :attr:`SubpopFit.sizes` before reading a row as complete.
        name: Column-axis name for the table.

    Returns:
        A ``DataFrame`` of counts, subpopulation down the rows.

    Raises:
        ValueError: on a length mismatch, or a ``Series`` whose index does not
            cover every labelled event.
    """
    if isinstance(other, pd.Series):
        missing = fit.labels.index.difference(other.index)
        if len(missing):
            raise ValueError(
                f"{len(missing)} labelled event(s) have no entry in `other`; "
                f"first few: {list(missing[:5])}"
            )
        rhs = other.reindex(fit.labels.index)
    else:
        rhs = pd.Series(list(other), index=fit.labels.index)
        if len(rhs) != len(fit.labels):
            raise ValueError(
                f"`other` has {len(rhs)} labels for {len(fit.labels)} events"
            )
    return pd.crosstab(fit.labels.rename("subpop"), rhs.rename(name))


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------


#: Where the cross-validated selection's slice of :func:`quantify_subpopulations`'s
#: progress ends and the final fit's begins. Measured on the owner's zn_30h: the
#: 480-fit search is ~60 s of a ~75 s run, the final 10-restart fit and the BIC
#: ladder ~5 s, the 200-resample bootstrap ~10 s. A bar apportioned by *stage
#: count* rather than by measured cost would sit at 20 % for a minute and then
#: sprint, which reads as a hang.
_SELECT_PROGRESS_SPAN = 0.80

#: Where the bootstrap's slice starts -- the gap from :data:`_SELECT_PROGRESS_SPAN`
#: is the final fit plus the BIC ladder.
_BOOT_PROGRESS_START = 0.88


def _scaled_progress(
    progress: Callable[[float, str], None] | None,
    lo: float,
    hi: float,
) -> Callable[[float, str], None] | None:
    """Re-map a stage's own 0-1 progress onto ``[lo, hi]`` of the whole run.

    Returns ``None`` for ``None`` so the stage below keeps its "no sink, no
    call" fast path rather than paying for a wrapper that discards everything.
    """
    if progress is None:
        return None

    def _emit(frac: float, msg: str) -> None:
        progress(lo + (hi - lo) * float(frac), msg)

    return _emit


def _dedup(items: Sequence[str]) -> tuple[str, ...]:
    """De-duplicate while preserving order -- the same caveat can arrive twice."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def quantify_subpopulations(
    df: pd.DataFrame,
    *,
    duration_s: float | None = None,
    features: Sequence[str] = SUBPOP_FEATURES,
    exploratory_labels: pd.Series | Sequence[Any] | None = None,
    exploratory_name: str = "hdbscan",
    p_grid: Sequence[int] | None = None,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    n_splits: int = DEFAULT_N_SPLITS,
    n_boot: int = DEFAULT_N_BOOT,
    random_state: int = 0,
    cv_n_init: int = 2,
    fit_n_init: int = 10,
    bic_n_init: int = 3,
    boot_n_init: int = 2,
    bootstrap_seed: int = 0,
    cancel: CancelFlag | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> SubpopQuantification:
    """Scale, select ``(n_PCs, K)``, fit, bootstrap, tabulate, crosstab.

    The single entry point a panel or a script should call. Every stage is also
    a public function, so a caller that already has, say, the selection can skip
    straight to :func:`fit_subpopulations`.

    Determinism: ``random_state`` seeds the fold split, every PCA and every
    mixture; ``bootstrap_seed`` seeds the resampling draws. Two runs with the
    same arguments give identical output, including the labels.

    Args:
        df: Per-event feature table for ONE recording. Not filtered, not sorted,
            not re-indexed between here and reading the labels back.
        duration_s: Recording duration for ``rate_per_s``.
        features: Feature columns, in order. Defaults to :data:`SUBPOP_FEATURES`.
        exploratory_labels: Labels from an independent clustering, for the
            crosstab. ``None`` skips it.
        exploratory_name: Column-axis name for the crosstab.
        p_grid: PC counts to search. Default ``2 .. len(features)``.
        k_grid: Component counts to search.
        n_splits: Cross-validation folds.
        n_boot: Bootstrap resamples; ``0`` skips them.
        random_state: THE model seed.
        cv_n_init: EM restarts inside the selection loop.
        fit_n_init: EM restarts for the final fit.
        bic_n_init: EM restarts per BIC rung.
        boot_n_init: EM restarts per bootstrap refit.
        bootstrap_seed: Seed for the resampling indices.
        cancel: Cancel token; ambient when ``None``.
        progress: Optional ``(fraction, message)`` sink covering the whole run.
            The two long loops report inside themselves (the selection over
            0-:data:`_SELECT_PROGRESS_SPAN`, the bootstrap over the rest), so
            the bar moves during the minute the cross-validation takes rather
            than jumping once at the end. A report channel only.

    Returns:
        A :class:`SubpopQuantification`.

    Raises:
        ValueError: propagated from the stage that objected.
        OperationCancelled: if the cancel token fires.
    """
    cols = tuple(str(c) for c in features)
    Xs = scale_features(df, cols)

    selection = select_pcs_and_k(
        Xs,
        p_grid=p_grid,
        k_grid=k_grid,
        n_splits=n_splits,
        random_state=random_state,
        n_init=cv_n_init,
        cancel=cancel,
        progress=_scaled_progress(progress, 0.0, _SELECT_PROGRESS_SPAN),
    )
    if progress is not None:
        progress(
            _SELECT_PROGRESS_SPAN,
            f"Fitting the mixture at n_PCs={selection.n_pcs}, K={selection.k}",
        )
    fit = fit_subpopulations(
        Xs,
        n_pcs=selection.n_pcs,
        k=selection.k,
        index=df.index,
        random_state=random_state,
        n_init=fit_n_init,
        bic_check=True,
        bic_k_grid=k_grid,
        bic_n_init=bic_n_init,
        cancel=cancel,
    )
    point = np.array(
        [fit.sizes.get(i, 0) / max(len(df), 1) for i in range(fit.k)], dtype=np.float64
    )
    ci = bootstrap_fractions(
        fit.scores,
        fit.means,
        n_boot=n_boot,
        seed=bootstrap_seed,
        n_init=boot_n_init,
        point_fractions=point,
        cancel=cancel,
        progress=_scaled_progress(progress, _BOOT_PROGRESS_START, 1.0),
    )
    if progress is not None:
        progress(1.0, "Tabulating subpopulations")
    table = population_table(df, fit, duration_s=duration_s, ci=ci)
    # Reads the finished fit and refits nothing; see `measure_separation`. The
    # ceiling is searched over THIS call's own k_grid, not over the module
    # default: the caveat's claim is "no K the search looked at is separated",
    # and a caller that searched 1..12 must not be answered about 1..8.
    sep = measure_separation(
        fit,
        random_state=random_state,
        ceiling_k_grid=tuple(int(k) for k in k_grid if int(k) >= 2),
    )
    xt = (
        None
        if exploratory_labels is None
        else label_crosstab(fit, exploratory_labels, name=exploratory_name)
    )

    notes = (
        list(selection.warnings)
        + list(fit.warnings)
        # BEFORE the bootstrap's caveats, not after: these are about whether the
        # classes exist at all, and the ones after them are about how precisely
        # their fractions are known. A reader who stops early should hit the
        # first question first.
        + list(_separation_warnings(sep, fit.k))
        + list(_stability_warnings(table))
        + list(ci.warnings)
    )
    if _TIME_HALF_COLUMN not in df.columns:
        notes.append(
            f"NO TIME-STABILITY CHECK: the table has no '{_TIME_HALF_COLUMN}' column, so "
            f"early_frac is NaN and nothing here distinguishes a real event class from "
            f"drift or fouling that only occurred in part of the recording."
        )
    if duration_s is None:
        notes.append(
            "NO EVENT RATES: duration_s was not supplied, so rate_per_s is NaN. "
            "Fractions alone cannot say whether a subpopulation is rare or the "
            "recording was short."
        )

    return SubpopQuantification(
        selection=selection,
        fit=fit,
        ci=ci,
        table=table,
        crosstab=xt,
        features=cols,
        n_events=int(len(df)),
        separation=sep,
        warnings=_dedup(notes),
    )
