# Changelog

All notable NanoScope changes are recorded here. The project is currently in private beta; beta
tags identify tested development baselines and do not promise a stable file or API surface.

## Unreleased

### Added

- Schmitt detection offers Automatic, Local and Global noise thresholds with a visible noise
  window, versioned bounded local profiles, matching threshold overlays, and export metadata.
  Legacy saved runs retain their global policy; local thresholds do not masquerade as one
  scalar distribution-fit floor or change the event table's global SNR definition.

- Background comparison sources now include editable time/current anchors, separate
  regions for genuine baseline jumps, existing open-pore references, and an independent
  reference recording with explicit shift, scale, offset and optional linear resampling.
  Custom recipes are inspection/export only, persist with source/calibration/unit bindings,
  and export all samples with an atomic CSV and provenance JSON pair.
- Nonlinear X–Y fitting for Gaussian peaks, exponentials, 4PL/5PL, Hill, Boltzmann,
  stretched exponential, Lorentzian, Voigt, straight line, power law and safe custom
  equations. Scatter and Voltage dependence expose parameter starts/bounds/fixed values,
  sigma weights, robust losses, cancellable workers, residuals, uncertainty and JSON/CSV
  export with saved project results.

### Fixed

- A CUSUM run now counts what the same settings find in the opposite direction, and the three
  validation studies follow or name their detector. Null control runs the selected detector; its
  readout names the detector and its settings and is shown only under a table of that detector.
  Under CUSUM it leads with the new count: *CUSUM (k 0.75, h 28, low-pass 5 kHz): N
  opposite-direction detection(s) against M on the recording.* Sweep threshold and Recovery map
  are Schmitt only: both are disabled with CUSUM selected, and a recovery map is used only for
  fits of a Schmitt table (under any other table *use measured efficiency* is greyed with the
  reason and keeps its tick). When the opposite-direction count away from detected events is at
  least 5 and at least 10% of what the run found in the event direction, the Summary shows the
  advisory *N opposite-direction detection(s) against M in the event direction. CUSUM fires that
  often where no events are expected, so unless events really run both ways many of the shallow
  rows here may be noise. Raise h or k and re-run.* The count is an estimate, not a rate. With
  High-pass on it is not a noise estimate, because the far edge of each event also fires in the
  opposite direction: the advisory then says that instead, and a CUSUM Null control is refused
  until High-pass is off.
  CUSUM's `h` assumes independent samples, so low-pass filtering and 1/f noise both raise its
  noise firings, and nothing reported it. On a 60 s apo-ferritin recording the run now says so
  itself: 707 events with 810 opposite-direction detections in the same run; raising h to 224
  leaves 259 events and 17. With CUSUM selected, Null control ran the Schmitt study and showed
  its count under the CUSUM table.
  New CUSUM provenance keys: `n_opposite_events`, `n_opposite_isolated`, `n_opposite_reference`
  and `opposite_guard_s`. Existing CUSUM and Schmitt settings return the same event tables as
  before; at a very low `h`, where firings are dense, a CUSUM run can take up to about twice as
  long. Project format unchanged; study results are not saved. The generated manuals are not
  rebuilt.
- Automatic masked baselines use local noise-aware masks and robust flank anchors, preventing
  noisy intervals from becoming one-sided interpolation bias and isolated endpoint samples
  from biasing event depths. Stationary automatic gap filling uses the same robust anchors;
  versioned replay distinguishes this from fixed legacy ML preprocessing.
  Noise masking checks original and rough-residual signals together so dense events and
  accelerating drift do not inflate their own mask threshold. Manual ML preparation retains
  exact training arithmetic; historical baseline descriptions distinguish settings from replay.
- Signal Conditioning invalidates obsolete layout items when moving live controls, preserving
  widget ownership and callbacks. Baseline health prominently distinguishes numerical checks
  from an unverified physical open-pore reference.
- Baseline health and window ranking assess moderate-SNR long states separately from the
  stronger-event median, preventing a damaged event from excluding itself from the assessment.
  Unavailable median evidence remains unavailable and cannot win a ranking as zero intrusion.

- Baseline sample/window counts are stable for round-off-sized changes at integer boundaries.
  Numerical policy stamps guard exact replay; fixed ML training recipes retain legacy arithmetic.

- Runtime baseline correction reports event-attenuation and occupancy risks from the
  same fitted baseline used for correction. PCHIP bounds edge estimates by local data
  to prevent unsupported excursions while retaining observed drift. Invalid AsLS domains
  are refused, and changed endpoint algorithms invalidate exact replay of older results.
- Background comparisons reject results after source, sample-rate, calibration or recipe
  changes. Main-trace baseline drawings run in cancellable workers. Comparison output
  choices no longer silently change the main-trace display; other enabled processing
  stages are disclosed separately.
- macOS builds keep TensorFlow's Python extension and native dependents on one
  library image, preventing duplicate operation registration during ML startup.
  Unavailable-runtime messages now distinguish packaged-app repair from source
  environment dependency setup.
- Nonlinear fitting now verifies local numerical sensitivities before reporting uncertainty,
  withholds unrepresentable intervals, and reports ambiguous observation-noise prediction
  bands instead of letting duplicate X values change them with row order.
- Fit JSON/CSV exports and research exports stage complete replacements so write failures
  preserve earlier files. Invalid saved fit metadata and failed overlay replacement leave
  the prior applied fit intact; interval captions follow the requested confidence level.
- Baseline edits update current and dormant figure warnings without recomputing retained
  results. Empty updates retain their empty status, and managed comparison generation uses
  the same explicit update path as the figure toolbar.
- CUSUM open-pore autocorrelation now measures gaps outside the union of padded events.
- Packaged builds reject incompatible dependency sets. The validated macOS environment
  uses Qt 6.9.3 after native widget-lifecycle and rendering checks, with compatible
  TensorFlow/Keras dependencies and recorded reproducible package constraints.

- Nonlinear parameter uncertainty and curve bands respect parameter domains and local
  scales, including logistic midpoints far smaller than the full dose range. Invalid
  uncertainty is withheld with a warning instead of producing an unreadable saved fit.
- Nonlinear R² uses resolved response variation, preserving the statistic when a large
  constant baseline is added while continuing to identify constant responses.
- Histogram confidence bands and residual graphics now round-trip through project and
  figure snapshots without refitting. Distribution-fit bootstrap samples are persisted.
- CDF diagnostics use declared support instead of normalizing to the observed range;
  reference thresholds are explicitly descriptive. Truncated exponential component
  support and selected-range Gaussian-mixture likelihoods follow the observation window.
- Inspector fits run off the GUI thread, respect cancellation and reject stale inputs.
  Centred/scaled linear regression preserves slopes at large X offsets; I-V gains OLS
  intervals. Constant exponential responses are refused and log-response objectives
  are identified in Voltage dependence.

### Changed

- **Batch detect runs the detector selected in the Detection panel, Schmitt or CUSUM.** Until now
  it ran Schmitt whatever the detector box showed. This supersedes the earlier Schmitt-only rule
  (*Batch detect records the detector it ran*, under the 13.4 answers below). ML is still not
  offered.
  - With CUSUM selected, each recording is detected as a single CUSUM Detect with the same form
    would detect it, and gets a CUSUM table. With Schmitt selected the detector gets the same
    parameters and finds the same events as before, and the prompt and status texts are
    unchanged. The rules below hold for both detectors.
  - A recording that has only the other detector's table counts as missing, and that table and
    its curation are left alone. *Only missing* and *Re-run all* count and replace tables of the
    selected detector only.
  - The form's detector decides, whatever table is on screen.
  - **Each recording then shows the table the batch made.** On every recording it detects that
    is not on screen, the batch's table becomes the remembered table, as a single Detect's does:
    select the recording and that table is shown, with the detector box on its detector.
    *Export density figure…* therefore reads one detector across the recordings. Until now a
    recording that had already shown the other detector's table went on showing it. The recording
    on screen follows the single run's rule, so a Manual table you chose, or a Reconciled table
    built from that detector's table, stays on screen.
  - **Undo after a batch re-run restores the earlier tables with their curation.** Everything is
    committed to the undo history just before the batch starts, as before a single Detect.
    *Undo Re-run detection* on a recording brings back the table the batch replaced there, hand
    rejections made just before the batch included; those used to come back accepted.
  - **Voltage dependence follows the latest pass on a tie, and always names its detector.** The
    series still pools the detector stored on the most recordings that can become a point. When
    two are stored on equally many, it now pools the one whose tables were made most recently
    (the newest `detected_at` among each detector's tables in the series). A series with Schmitt
    on every recording that is then batch-detected with CUSUM plots CUSUM, and the reverse.
    Missing, unreadable or equal stamps fall back to Schmitt, then CUSUM, then ML. The Pooled
    provenance block prints its *Detector:* line always, *Detector: Schmitt* included.
  - The prompt, progress and summary name CUSUM when it is the detector: *N of M recording(s)
    already have a CUSUM detection.*, *Batch detecting N recording(s) with CUSUM…*, *Batch detect
    (CUSUM): N recording(s), E event(s).* and *Batch detect (CUSUM): nothing to do — every
    recording already has a CUSUM detection.* The Schmitt texts are unchanged. **Cancel** reads
    *Cancelled.* for either detector.
  - After a CUSUM batch the summary says how many recordings raised the opposite-direction
    advisory, when any did: *K recording(s) raised the opposite-direction advisory.* The count
    comes from the numbers each run recorded, by the rule that decides the advisory.
  - **Two new refusals, both before the prompt.** A Batch detect while a batch is running is
    refused in the run gate's sentence, *A batch detection is already running. This app runs one
    analysis job at a time, so the request for a batch detection was refused; nothing was
    started. Use Cancel in the window running that job, or wait for it to finish.* It used to
    replace the running batch without saying so. A maximum dwell that is on and does not exceed
    the minimum dwell is refused in the single run's sentence, *Batch detect refused — maximum
    dwell (0.5 ms) must exceed minimum dwell (1 ms). Increase the maximum or set it to 0 (off).*
    It used to fail every recording inside the detector.
  - **Provenance of a batch-made table.** It records the recording's effective acquisition
    bandwidth, one you declared included, as a single Detect does; it used to record the file
    header's. A recording whose open-pore windows were measured and confirmed under one detector
    keeps that confirmation when a batch of the other detector runs it at the same windows,
    instead of being marked as copied from the batch form.
  - The menu tooltip reads *Run the selected detector (Schmitt or CUSUM) with its current
    parameters on every open recording that has a current channel and a known sample rate, in the
    background. Detect the whole -1000..-200 mV series in one pass, then plot the voltage
    dependence. (ML detection is not offered here — it is too slow to batch over many full
    recordings.)*
  - The refusal under an open-pore reference you supplied and the refusal while another analysis
    is running still apply. Single-run Detect, the detection arithmetic and the project format
    are unchanged. The generated manuals are not rebuilt.

- **Plotting redesign, Phases 3–4 and full reference workflow (2026-11-03 development source).**
  Project Figures now expose New/Duplicate, multiple figures per type, freshness/dormant states,
  recipe input requirements and optional wording recall. Shared styles are pinned by revision and
  portable snapshot, with explicit project/local scopes and a default for plain New figures.
  - Format supports per-series ink/visibility and shared source-name/color defaults, inherited
    property markers and supported matplotlib text/series selection. Frame, palette, fractional
    metrics, synthetic preview and optional physical canvas/column widths share the style model.
    Existing scientific source/category identity colors and frozen Snapshots remain preserved.
  - Declared time quantities display in ms across inputs, axes, tables, readouts, Study and report
    figures. Inverse entry conversion and versioned saved-figure presentation preserve raw data,
    fit parameters and CSV keys in their scientific basis. Unknown-rate samples, normalized time,
    frequency/rates and charge keep their own units. Density conversion includes its inverse factor.
  - Snapshot/composition workflows preserve lineage and panel-letter overrides. Serial selected-
    figure export uses saved results and each pinned style's explicit sizing; missing named widths
    and per-file failures are reported without silent recomputation or guessed journal requirements.
  - Manual source covers the full workflow. Generated HTML/PDF/ZIP and final screenshot provenance
    are rebuilt only after the Phase 4 validation/capture gate, not by this source-only change.


- **Plotting redesign, Phase 2: independent named figures and project format v13 (2026-11-03).**
  Analysis Explorer retains multiple figures of each analytical type, with separate sources,
  pending/applied recipes, saved results, wording, appearance and session history. Create,
  duplicate, rename, delete and restore figures without replacing another figure of that type;
  inactive figures remain saved when their editor is unloaded. Snapshots and compositions retain
  their own data and lineage. Historical v11/v12 files retain their native fields/resources, with
  frozen compatibility defaults and named unsupported records instead of empty replacements.
  - Every new project save is v13. Earlier projects still open here; older NanoScope builds cannot
    open v13 projects. The first in-place conversion offers Save and convert / Save a copy / Cancel,
    retains a SHA-verified original backup without overwriting unrelated backups, and verifies the
    staged workspace before replacing the project. Recovery keeps its existing envelope.
  - Writer version/build identity is diagnostic and separate from the format gate. Unknown fields,
    foreign artifacts and unavailable-source state remain preserved. Generated manuals are deferred
    until the remaining plotting phases are complete.

- **Plotting redesign, Phase 1: one toolbar and a right inspector (2026-11-03).**
  All eight Analysis Explorer pages now use **Data & analysis** and **Format**. Source and recipe
  groups replace Section bands; the first two groups start open, later groups collapsed. The
  common toolbar holds **Undo**, **Redo**, **Update plot ▾**, **Style ▾**, **Reset figure** and
  **Export ▾**. Inspector width, visibility, tab and group states are remembered per user.
  - **Update plot** commits the page's pending operation, with its pending count beside it;
    Distributions commits sources, plot and fit choices together, while summaries/comparisons
    generate explicitly. **Refresh from recordings** is in its menu. Ctrl/⌘Return (including keypad
    Enter) commits the focused field first; Ctrl/⌘1 and Ctrl/⌘2 select the inspector tabs.
  - **Format** selects an element and shows its properties. Click editable figure text to select
    it and use the floating mini toolbar; double-click editing and right-click menus remain.
    Inheritance markers clear one override; **Reset element**, **Reset look**, **Copy format** and
    **Paste format** operate on styles without replacing wording. Axis Reset element also clears
    its pinned range and major tick in one Undo step. Reset look preserves ranges and positions.
    The later Phase 4 section above supersedes the initial matplotlib selection limitation.
  - **Style ▾** applies built-in or locally saved looks to this page or every open page, preserving
    hand-set text styles by default. Each page owns its Undo; **Undo style on N pages** can undo an
    undisturbed batch. Saving a look excludes typed words and numerical settings.
  - **Save to Figure Library…**, **Save figure bundle…** and **Open Figure Library** move into
    **Export ▾**. Home's read-only **Refresh availability** sits beside search. Distributions moves
    Lasso select and Reset view to its status line, templates to Recipes, and axis pins to Format.
  - Scientific calculations, Phase 0 quantity-change resets, template wording promises, figure
    fonts and project/export formats are preserved. Generated manuals and retained screenshots
    require a separate rebuild/capture before distribution.

- **Words you typed, and ranges you pinned, go back to automatic when a figure's quantity changes
  (plotting redesign, Phase 0, 2026-11-02).**
  - **The rule.** The Title, X label, Y label and the fit label's wording you typed belong to what
    the figure showed when you typed them, and so does the range pinned in **Axis properties** on
    each axis, with the major tick spacing typed beside it (*range* below means both). When the
    figure comes to show something else, the words and the range of each axis whose quantity moved
    go back to automatic, as one step of that plot's own Undo, and its status line names only what
    was reset (*Title and Y label reset: the plot now shows median depth.*, or *Title, Y label and Y
    range reset: …* when a range went too; a major tick spacing is named apart from the range, *Y
    range and Y major tick reset: …*, or *Y major tick reset: …* when only a spacing was typed). A Y
    range pinned for the capture rate in Hz used to keep framing median depths in pA, and a range
    pinned on a Box plot's value axis a new measurement. An axis whose quantity did not move keeps
    its range, and so does another view's unless its axis shows the same quantity (the two pairs of
    Distribution summaries share their measurement). A figure with nothing typed on it and no range
    pinned on an axis that moved records no step and says nothing; one with only a range pinned
    records the step and says *Y range reset: …*. If the status line still shows the sentence of an
    earlier change, that sentence is removed, because the figure has moved on since (any other
    message on the status bar stays). A title or label you hid (**Hide title**, or a single space in
    its box) counts as typed and goes back to automatic too: the window's automatic title or label
    comes back if it draws one (Voltage dependence and Subpopulations draw no automatic title, so
    there a hidden title coming back shows nothing). The look stays: fonts, sizes, colours, text
    styles, the legend, the ranges pinned on axes whose quantity did not move, and the fit label's
    numbers, position, size and colour. A caption or other annotation text, Voltage dependence's
    caption included, is not touched by this rule (Reset figure takes it back).
  - **Undo.** It brings the words and the ranges back over the quantity that is on screen and does
    not move the control. To return to the earlier quantity with its words, pick it first (nothing
    is typed any more, so that records no step) and then **Undo**; **Undo** first and the pick after
    is a new change, which clears the words again. Distributions is the exception: its
    **Configuration** Undo takes the plot type and the words back together, so press **Undo** alone
    there. On the pages that have an appearance Undo (Voltage dependence, Distribution summaries,
    Comparison figures, Event waveforms, Noise and Subpopulations) the **Undo** and **Redo**
    tooltips name the step by what it took back (*Undo: reset title and Y label*, *Undo: reset
    title, Y label and Y range*, *Undo: reset X range and Y range*), so it does not pass for a
    **Reset figure**, whose tooltip reads *Undo: reset figure* when it also takes the look back; a
    figure whose only customisation is typed words and ranges has nothing else for Reset figure to
    take, so there the two read alike. Distributions' **Configuration** Undo counts steps and names
    none (*Undo the last change to this plot's configuration (3 to step back).*).
  - **What counts, window by window.**
    - **Voltage dependence:** the **Metric**, which is the Y axis (the X axis is voltage whatever the
      metric, so its range stays), and the applied fit when **Apply fit** replaces one trend with
      another or takes the fit away (only the fit label's wording goes back and no axis moves; the
      first fit applied after *none* keeps wording typed ahead of it). Ticking or clearing **|V|**
      turns the X axis between the signed voltage and its magnitude, so it takes the X range and
      leaves the words (*X range reset: the plot now shows capture rate against |V|.*). **Log y** is
      a real log axis whose range is typed in the units it shows: both ranges stay.
    - **Distribution summaries and Comparison figures:** the measurement, summary or figure type (in
      Comparison also X, Y and the pair features), at the moment **Generate** commits it; until then
      the figure on screen is the old one and keeps its words. In Distribution summaries a new
      measurement is the value axis changing (X in ECDF and Survival, Y in Box and Violin), so it
      takes the value range of both pairs and the sentence says *value range* once; the ECDF's and
      Survival's fraction axis and the Box and Violin sources axis keep their range, and a new
      summary type takes none (each pair keeps its own, and the one you leave is waiting when you
      come back). A Comparison figure is a picture without Axis properties: it has no range to take.
    - **Event waveforms:** the heatmap's **X** (it takes the map's X range) and **Y** (the map's Y
      range), the switch between the Event-count heatmap and Ensemble tabs (each view keeps its own
      range, so it takes none), **Normalize time** (it takes the ensemble's X range) and **Align**
      (start, center, extremum), which is what t = 0 means on the ensemble's X axis once time is in
      seconds, so another alignment takes the words and that range too (*Title, X label, Y label and
      X range reset: the plot now shows the ensemble with t = 0 at each event's deepest sample.*);
      while time is normalised no alignment is drawn, the combo is greyed out and setting it takes
      nothing. **Log x** and **Log y** take the range of their own axis of the map and leave the
      words, because the map draws log10 values on a linear axis while one is ticked (*X range
      reset: the plot now shows median depth against log10 of dwell.*); changing **Log scale** in
      the map's Axis properties, though, keeps the range typed in that dialog (unticking it there to
      type a range in seconds is how the map's X axis, which opens on a log axis, is pinned). A
      range left as the dialog opened with it is carried across to the new units instead, log10 of
      it when Log scale is ticked and ten to the power when it is unticked, so the map keeps its
      frame; the major tick spacing the dialog opened with, which does not carry across, goes, and
      so does a range that cannot be logged (a minimum at or below zero); a spacing typed in the
      same dialog stays.
    - **Noise:** the **Show** view takes the words and no range (each view keeps its own, so
      Spectrogram and back brings the PSD's range back), and **Amplitude spectral density** on or
      off takes the words and the PSD's Y range, and leaves its frequency range.
    - **Subpopulations:** the **Embedding** takes the words and both axes of the scatter, not the
      waveforms'.
    - **Distributions:** the plot type and a column, which already took back the title, the axis
      labels and the typed range of an axis that moved (the axis of the column that changed, and the
      Y axis too while Normalize is *density*, whose unit is the column's; both axes on a plot-type
      change), and now take the fit label's wording too (wording typed for an exponential, *Single
      exponential: {equation}, {n}*, used to stay over a linear fit); and the applied fit (another
      model, another fixed number of components for a mixture, or Auto (BIC) switched on or off for
      one), which takes back only the wording, the first fit applied after *none* keeping wording
      typed ahead of it; and the **Normalize** you pick (count, density, probability per bin), which
      is what the histogram's Y axis counts, so it takes the words and the Y range in the step of
      the pick, and the status line says what the histogram is now set to show (*Title, X label, Y
      label, fit label wording and Y range reset: the plot is now set to show the histogram of
      dwell_s as density, fitted with exponential.*). The switch to density that ticking a second
      source makes by itself, and back to count when you are down to one, takes nothing, and neither
      does a Normalize set on a page that is not the histogram. While Normalize is *density* the Y
      axis is per the column's unit (*density (1/s)* over a dwell time, plain *density* over a
      fractional depth), so a new column takes the Y range too; with *count* or *probability per
      bin* the Y axis does not depend on the column and keeps its range. On the PCA page the
      **Features** you tick are what its two axes are made of, so another set takes the words and
      both ranges (PC1 and PC2 are other combinations of other columns), in the Configuration step
      of the tick that changed it; on any other plot type the same list is a clustering setting and
      takes nothing. The status line now names a range that went, words and ranges together (*Title,
      X range and Y range reset: the plot is now set to show the scatter of ycol against xcol.*),
      and a change that clears only a range says so instead of leaving the line empty. The reset is
      part of the same **Configuration** Undo step as the change that caused it. A column also moves
      on its own when you tick a source that lacks it (the column boxes offer only the columns every
      ticked source has; the manual-measurement table has no `depth_pA`), and the rule then runs as
      for a pick.
    - **I-V** has no such control and is unchanged; the study figure editor has none that changes
      what an axis shows.
  - **Normalize and Align are a choice made beyond the owner's explicit list** (plot type, column,
    metric, measurement and view), for the owner to veto. Neither is one of those, but each changes
    what an axis shows (what the histogram's Y axis counts; what t = 0 means on the ensemble's X
    axis), so the rule is built to run on both; a veto of either takes it out again and leaves the
    other.
  - **What does not run the rule.** Choosing another recording or ticking another source that has
    the plotted column, applying a preset, a refresh of the same quantity, a figure's first draw, a
    log axis, |V|, bin counts, a colormap, a colour-by choice, the band, a subset of events,
    Accepted only and the Welch settings keep the words, and so do Distributions' Time unit and the
    switch to density that ticking a second source makes by itself (a Normalize you pick takes them,
    above). None of them takes a range off either, except that |V| (Voltage dependence's X range)
    and Log x and Log y (the Event-count heatmap's range on that axis) change the numbers on the
    axis the range was typed in, so they take it and leave the words. Opening a project shows the
    words and ranges it was saved with (the rule acts on the next change), and recalling a template
    installs the title, labels and wording it was saved with (none, if it was saved without), and
    the ranges it was saved with, in place of the ones typed and pinned. A template recalled in
    Distribution summaries or Comparison figures brings its words with its measurement, so the
    Generate that commits it keeps them; in Voltage dependence the template's fit arrives selected,
    and the **Apply fit** that commits it keeps the wording that came with it. They are kept only
    while the words (and, in Distribution summaries, the ranges) are still the template's: **Undo**
    on the recall, a word or range typed or pinned after it, another quantity picked after it, or
    saving the project and opening it again before the Generate or **Apply fit** makes that an
    ordinary change of quantity, and they go back to automatic (one Undo step and the sentence, as
    for any change). Restoring a Subpopulations figure shows its saved words and no reset sentence.
  - **A project saved by an older build** whose appearance holds a range with no view (a bare *X* or
    *Y* range) takes it with the primary view's axis it belongs to (the PSD's in Noise, the
    Event-count map's in Event waveforms, the scatter's in Subpopulations, the ECDF's and Survival's
    in Distribution summaries), and no other view's.
  - **Where the sentence is read.** On the page's status line inside the Analysis Explorer: the
    status bar at the bottom of Voltage dependence, Distribution summaries, Event waveforms, Noise
    and Distributions (ten seconds), the status line at the top of Comparison figures, and the
    standing line under the report text in Subpopulations. It stands only while it is true: it says
    that what it names is automatic, so it comes off (the status bar, or in Comparison figures and
    Subpopulations the line it stands on) as soon as one of those words or ranges is typed or pinned
    again, as by the **Undo** that brings them back, a title or label typed since, or a range pinned
    again. A change of look, **Reset figure** and its **Undo** and **Redo** leave it, since nothing
    it names is typed or pinned then, and **Redo** takes the words and ranges away again without
    bringing the sentence back.
  - **The Appearance boxes show the automatic text they would draw** in Voltage dependence and
    Distribution summaries. The X label placeholder follows |V| (*Applied |voltage| (mV)*), the Y
    label's names the metric and Log y (*Median depth (pA) (log)*), and Distribution summaries'
    Title, X label and Y label name the figure on screen, so a reset box reads *Automatic: Median
    depth (…)*.
  - **The Title, X label and Y label boxes say what a typed title covers** in Noise, Event waveforms,
    Comparison figures and Subpopulations. They read *Title (this view)*, *X label (this view)* and
    *Y label (this view)* (*(both figures)* in Subpopulations), where they said *(all views)*, and so
    does the right-click entry that adds a title on a figure: **Add title (this view)…** (**Add title
    (both figures)…** in Subpopulations). Words typed there belong to what is on screen and go back to
    automatic when the figure comes to show something else (another view, metric or embedding); the
    boxes' tool tips say so, and that the status line names what went and Undo brings it back. The
    old wording promised a title that stayed across the views.

- **One short font list, with Arial as the default (plotting redesign, Phase 0, 2026-11-02).**
  - **Every font control is the same list:** *Arial (default)*, *Helvetica*, *Times New Roman*,
    *Courier New*, then *Other font…*. It replaces the 180-family lists in the Appearance section of
    every plot page (Distributions' included), the **Text style…** font box and the composer's
    panel-label font. *Other font…* opens a list of every installed font with a filter box and a
    sample line (*Aa τ μ Δ Å 0123*) in the highlighted font; Cancel keeps the previous font. It
    leaves out the faces of symbols (Symbol, Webdings, Wingdings, Wingdings 2 and 3, Zapf Dingbats,
    Apple Symbols), which no figure's text is set in. A font outside the list, picked there or
    stored by an older project (a symbol face included), is shown as one extra row under its own
    name (*name — not on this computer* when this computer lacks it) and is kept as saved; the row
    of an installed font stays so it can be picked again. In **Text style…** the first row reads
    plain *Arial*, because there "no font" is the *Custom font* box being off (the font box's tool
    tip says so). Choosing a font is one Undo step in every plot page, and the font box no longer
    takes typed text (type the name into the filter box of *Other font…* instead).
  - **A figure with no font chosen is drawn in Arial** instead of the application font, which on
    macOS is a private system face no journal has: title, axis labels, tick numbers, legend entries,
    the fit label, Voltage dependence's caption and the numbers that mark the levels of Noise's
    All-points histogram, in Distributions, Voltage dependence, I-V, Noise spectrum, Event waveforms
    (colour bar included), Distribution summaries and Subpopulations. The fit label and the legend
    also follow a font chosen for the figure now; they used to stay in the application font. Comparison
    figures and the study figure editor, which matplotlib draws, are Arial too (see the exports below).
  - **A font this computer lacks falls back to a metric-compatible one** (Arial to Helvetica,
    Liberation Sans, Arimo…; Times New Roman to Times, Liberation Serif, Tinos…; Courier New to
    Courier, Liberation Mono, Cousine…; any other font to Arial's list) instead of to the system UI
    font. With none of them installed, the labels and the tick numbers use the same face of that
    kind (a sans-serif, a serif for Times New Roman, a fixed-width one for Courier New). The figure
    keeps the name you chose; only what is drawn, and so what an SVG names first, is the substitute.
    The font list says so: on Windows, which has no Helvetica, the row reads *Helvetica (drawn in
    Arial on this computer)*; on a Linux computer with only the Liberation fonts, *Arial (default;
    drawn in Liberation Sans on this computer)*, and likewise for Times New Roman and Courier New.
    The row still stores the family, so a figure styled on a Mac opens unchanged; a computer that
    has the four fonts shows the plain rows.
  - **Greek letters** are ordinary Unicode characters (τ, µ, Δ) drawn in the figure's font; no symbol
    font is on the list. A character the font lacks, such as the subscript zero of *I₀* or the
    exponent in *10⁻³* in Times New Roman, is drawn in another installed font and no longer in the
    system's own interface face, which a PDF export of the plot embedded. Labels now name only the
    fonts this computer has (no generic *serif* or *sans-serif*, which Qt drew in the application
    font).
  - **Compose multi-panel figure:** panel labels default to *Arial (default)*, so they match the
    panels of figures saved by this build that have no font chosen, which are Arial now. A figure
    saved to the Figure Library by an earlier build is a frozen picture and keeps the font it was
    drawn in (the application font unless one was chosen), so its panels do not match the Arial
    labels (on a Mac, where the application font is a private face, the list has no row for it;
    where it is an ordinary family such as Segoe UI, Other font… lists it, so it can be picked to
    match): generate it again in its Analysis tool and save it again to get Arial panels. A
    composition keeps the family it was saved with: one an earlier Mac build saved in the
    application font shows it on a row of its own, *Application font (older figure)*, so its letters
    keep matching its panels in the PNG and the PDF, and its SVG writes Arial for those letters;
    choosing Arial changes the letters everywhere.
  - **Nothing in a saved project changes:** an older project's font (`Menlo`, `Times New Roman`, or
    none) opens exactly as saved, and one that names none is now drawn in Arial. An older build draws
    a figure with no font chosen in its application font, and opens a project that names a font as
    before.

- **Distributions: the fit label's Numbers row chooses a fit's numbers, Title weight is gone, and
  Reset appearance is now Reset figure, in the page toolbar (plotting redesign, Phase 0, owner
  decision 9, 2026-11-02).**
  - **The fit label's Numbers row replaces three Style boxes.** *Fit values on plot*, *R² on
    figure* and *KS statistic on figure* are gone from Distributions' Style column. Tick BIC, KS
    and R² in **Appearance ▸ Fit label ▸ Numbers**: a new label still shows Equation, R² and n, and
    ticking BIC and KS draws the text *Fit values on plot* drew. The Style column's **Residual
    strip** box still adds the strip's χ²/dof after the equation. The six Numbers boxes now sit
    three to a line, so BIC and KS no longer fall off the right edge of Distributions' 344 px
    controls column (which no longer scrolls sideways in Appearance); in a wide strip such as
    Voltage dependence's or I-V's they keep that spacing instead of spreading a third of the strip
    apart. The R² and KS boxes have tool tips that say what they do.
  - **Unticking R² takes the R² refusal off the label with it.** *R² undefined (y does not vary)*
    and *exact by construction* are printed, at the end of the label or where `{r2}` stands, while
    R² is ticked or `{r2}` is in your words, as in every other window; unticked, with no `{r2}` in
    your words, the label makes no claim about R² and Details still says. *R² on figure* used to
    drop the number and keep the refusal. The scatter fit's R² is now always in Details' summary
    line.
  - **Title weight is gone;** the title's bold and italic are its **Text style…** (right-click the
    title).
  - **Older files open with the figure they had, with the two differences just named** (the
    refusal tag, and Details' summary line). A project, Distributions template or plot
    configuration file saved with those switches takes its label's numbers and the title's bold
    and italic from them: a label with nothing of its own saved (never dragged, retyped or changed
    in the Fit label group) shows Equation, R² and n plus BIC and KS if *Fit values on plot* was
    ticked, any other label keeps the numbers it had, and *R² on figure* / *KS statistic on
    figure* unticked leave R² / KS out, with a dash where your own wording names `{r2}` or `{ks}`,
    as the older build drew it; Bold and Italic go to the title's Text style where it has not set
    that one itself. The import note no longer lists those keys as settings not taken. This build
    writes none of the old settings itself: a page saved after an edit holds none of them (a
    project saved again with no edit to the page, and an older Distributions template stored on
    this computer, are carried as they were and open the same way), and the project format is
    still 12. An older build opens a project saved by this one and draws the same figure, because
    the label's numbers and the title's weight are kept in fields it already reads.
  - **Reset appearance is Reset figure, a button in the page toolbar,** just before **Export ▾**, on
    every Analysis Explorer page, so it is there from every Section. It puts the title, axis labels,
    fit-label wording, any caption or other annotation text you typed (Voltage dependence's caption,
    the words on Distributions' detection-limit line) and the look back to their defaults in one
    Undo step; ⌘Z brings it all back (on the other pages the Appearance section's Undo button names
    the step from what it took back: *Undo: reset figure* when a title, a label and a look went back
    together, *Undo: title ‘Capture vs voltage’ → automatic* when only the title did, and for a
    figure whose only customisation is typed words and ranges the words of the reset a change of
    quantity makes, *Undo: reset title and Y label*; Distributions' Undo counts steps and names
    none). A fit label whose wording a change of quantity took back is the untouched label again
    unless its numbers, position, size or colour were set, so Reset figure pressed straight after
    such a change, on a figure with nothing else customised, finds nothing to take back and records
    no step. A source you renamed on its legend row (Distributions, Distribution summaries) keeps
    its name. The study figure editor, which has no such toolbar, keeps it in its Appearance
    section. The Undo and Redo buttons stay in the Appearance section, side by side. In
    Distributions it now also returns the label's numbers to Equation, R² and n and the title's bold
    and italic to plain, and its tool tip says what it leaves as it is: the **Axis range** boxes,
    **Font size (body)**, the legend's position, entries and (n = …) switch, source names,
    **Confidence band**, **Residual strip**, **Provenance caption** and **Heatmap colormap**. On the
    other pages that have Axis properties, a range or tick spacing typed there goes with the look,
    and the button's tool tip says so; Event waveforms leaves its **Event-count heatmap colormap**
    and Noise its **Spectrogram colormap** and **colour limits**, and their tool tips say so too. It
    puts the font back to *Arial (default)* along with the rest of the look.

- **Exports name the figure's font with a fallback list, and report PDFs hold TrueType (plotting
  redesign, Phase 0, export fonts, 2026-11-02).**
  - **SVG exports name a fallback list for every font, and never the private system face.** Qt wrote
    one font-family on each text element, even for a font that is not installed and with nothing
    behind it, and a figure that was never given a font was written in the application's own face,
    which on macOS is the private `.AppleSystemUIFont`. Every text element now names the font the
    figure is drawn in, followed by its stand-ins (Arial, then Helvetica, Liberation Sans, Arimo,
    TeX Gyre Heros, Nimbus Sans and sans-serif; Times New Roman and Courier New have lists of their
    own that end in serif and monospace, and so does the stand-in that a computer without them draws
    them in (Liberation Serif, Liberation Mono, Tinos…): it is named first and followed by its
    family's list, so a viewer without it still gets a serif or a fixed-width face; any other family
    is followed by Arial's). A font this computer lacks is written as the substitute the figure was
    drawn in, so an SVG exported here for a Calibri figure names Arial first. Text that has no font
    of its own, such as the grey message a plot shows instead of data, is written as Arial when the
    application's own font is a private system face (a name starting with a dot, as on a Mac); a
    family someone picked is named first, the application's own font included. The font-family Qt
    also put on the groups around the text, which named the private face too, is dropped. Text stays
    text and no font is embedded. This covers **Export current figure…** (also at a physical size),
    **Save figure bundle…**, Figure Library snapshots taken from now on and pyqtgraph's own
    right-click Export. A figure saved to the Figure Library, or into a project, by an earlier
    version keeps the SVG it was captured with, so its SVG export still has the single font family
    per text element (and, for a figure whose font was never chosen, the application's own face).
  - **Comparison figures and the study figure editor draw in the chosen font's list, and in Arial
    when none is chosen.** The chosen family is drawn as that family, then those of its stand-ins
    that are installed, then matplotlib's own sans-serif, serif or monospace face, then DejaVu Sans.
    A family that is not installed falls back quietly; the bare name used to be logged as *not found*
    on every text it drew, and the figure fell back to DejaVu Sans. A comparison figure with no font
    chosen was drawn in DejaVu Sans, and a study figure in the editor in the font the study had drawn
    it in. Their SVG and PDF still hold the text as outlines. On a Mac matplotlib reads only one
    face of Helvetica, so bold or italic text set in Helvetica (a study figure's bold title, y tick
    labels and bar text) is drawn in the regular weight there; pick Arial when you need bold.
  - **A study figure with saved styling is drawn in Arial unless a font was chosen, also when its
    figure group is regenerated.** This holds whatever else the styling holds, even nothing; a study
    figure with no saved styling stays in the DejaVu Sans the study draws it in, so one regenerated
    study can hold both. **Reset all figure styling** in the study figure editor leaves the figure in
    Arial, no longer in the study's own font.
  - **Matplotlib figures draw in a list that starts with Arial.** The report figures' font is the
    list Arial, Helvetica, Liberation Sans, Arimo, TeX Gyre Heros and Nimbus Sans (those installed),
    then matplotlib's sans-serif and DejaVu Sans; it was the word sans-serif, which names a single
    face. A character the first face lacks, such as the subscript digits and the superscript minus
    that Arial does not have, is taken from the next face that has it, and last from DejaVu Sans, so
    τ, µ, Δ and the ₀ of ΔI/I₀ still draw on a computer with none of the others.

- **Plot text belongs to you; a fit's numbers go in a fit label (plot UX round, 2026-11-01).**
  - **A fit no longer replaces the plot title.** In Distributions (scatter and histogram fits), I-V and
    Voltage dependence, the equation and numbers (R² with its degrees-of-freedom or *undefined* tag,
    n; BIC and the KS statistic when ticked in its Numbers row) are shown in a **fit label** on the plot,
    one line per source in the source's colour. A typed title keeps them. Legend fit rows read *Apo
    fit (linear)*; **In legend instead** (Distributions) puts the numbers there on a second line.
    Voltage dependence states what was pooled (merge state, baseline, gain, duration) in a caption
    you can drag, retype or hide. The PCA title no longer repeats the explained variance (the PC axis
    labels state it; Details keeps the sentence).
  - **Typed titles, axis labels and legend names are shown exactly as typed:** `I < 5 pA & τ > 1 ms`
    no longer loses characters. Use `_{…}` and `^{…}` for sub- and superscripts. A title saved with
    HTML tags such as `<sub>` now shows the tags literally; retype it with `_{…}`.
  - **Appearance presets keep the look only** (fonts, sizes, colours, styles); they no longer carry
    or overwrite titles and axis labels, and a preset carried by an older project is read the same
    way. **Window templates** keep titles, axis labels and text styles, but not legend names.
  - **The Appearance section starts with Title, X label and Y label**, whose placeholders show the
    automatic text; *Original* is now *Automatic* everywhere.
  - **Voltage axes read in millivolts:** *Applied voltage (mV)* with ticks in mV; ranges typed in
    Axis properties are in mV.
  - **Multi-panel composition with shared axes** uses a frozen picture for panels carrying a fit
    label, a caption or text styles it cannot redraw, instead of redrawing them without it.
  - **Distributions' Features list is a checklist** with Select all, Clear and Default (the preferred
    measurements); event_id, start_idx and end_idx are listed last under *Identifiers* and left out
    of Default.
- **Results ▸ Columns… stays open while you tick several columns**, lists them in the table's order
  with the header's own names, and adds a filter, Show all / Hide all / Default columns and Default
  order; right-click a column header to hide it. The table can no longer be emptied: Hide all keeps
  *accepted*, and a saved view that hid every column is ignored. Restoring the scientific column
  order no longer also resets which columns are shown: **Default order** and **Default columns** are
  separate.
- **Comparison figures:** a click anywhere on a pair-matrix feature ticks it, and the two-to-five rule
  is shown and enforced as you pick; a generated figure opens fitted to the canvas (also when a
  project reopens) instead of scrolled into it.
- **Figure Library:** a saved figure is named after its tool and plot type (*I-V / conductance*,
  *Distributions · Histogram*) instead of its page id (*Iv*), and an export suggests a file name with
  *-* for the separators; the list shows recording file names, with the full path in the tooltip.
- **Compose multi-panel figure:** panel spacing is in whole pixels and the label colour is a colour
  swatch.
- **Analysis Explorer:** a tool that cannot open yet says why in a banner inside the Explorer (from a
  double-click, Enter or Explain requirements alike) instead of on the main window's status bar, and
  Analysis home greys out tools that cannot open yet, with the reason first in their tooltip.
- **A new look for the Measure mask.** Its edges are brackets, `[` for the start and `]` for the
  finish, in a magenta-violet no other trace mark uses, over a dark outline, with grip tabs near the
  bottom of the pane instead of circles on the trace. A fitted edge has a ring on the sample it
  snapped to. An edge that could not be fitted is dashed and tagged *not fitted*. A span being drawn
  is dashed, and S alone shows a single bracket. The pointer over an edge is ↔, and the crosshair
  stays on top.
- **Clear mask keeps Measure on**, as X does.

- **The owner's answers to the editing foundation's open questions (design 13.4).**
  - **Polarity is judged on its own.** It was compared with the run only inside the Baseline drift
    stage, so with that stage off a changed polarity left the table looking current, although the
    detector reads it. It now reads *⚠ Settings changed: polarity auto → blockade*, the amber
    line's Details lists it on the Thresholds row, and Revert puts it back as one step. With the
    drift stage on, the stage names it, in the notice and on the Details list's Signal chain row. A
    table whose run did not record its polarity is not compared on it.
  - **A recording's first table keeps settings you changed before it and did not run.** When a
    batch run or a hand-made event gives a recording its first table, the form keeps your edit:
    marked edited, named by the ⚠ button, saved with the project, and taken back and forward by
    Edit ▸ Undo and Redo. The next Detect runs it, and one Ctrl+Z takes back that run together with
    the edit.
    - A batch counts as having run your edit only when it was launched from that same form with
      nothing changed since, ML and open-pore reference included.
    - The minimum dwell and merge gap the app set to the other detector's defaults follow the new
      table, so a Schmitt table no longer shows CUSUM's 1 ms and 500 µs as settings you changed. A
      value you typed stays, even one equal to the other detector's default.
  - **Switching the detector box is no longer an undo step,** and neither is the minimum dwell and
    merge gap it moves to the chosen detector's defaults. Ctrl+Z goes to the edit before the
    switch, and an edit typed just before or after it is its own step. A step the switch leaves
    with nothing to change is withdrawn, and one whose change it altered is named for what it now
    changes.
  - **An emptied Title, X label or Y label box means the automatic label in every plot window**,
    Comparison figures and the study figure editor included, which drew no label for it. To hide a
    label, type a single space. This now works in every window, and a hidden title leaves no gap
    above the plot. A Comparison or Study appearance saved with an emptied box draws the automatic
    label when reopened.
  - **The Plots window's STALE mark follows a Baseline-form edit as you make it.**
    - It is redrawn once, about 0.3 s after the last knob step, and only when a mark on a column
      the plot draws changes (or the clustering's amplitude note does).
    - Your zoom and pan are kept.
    - A saved snapshot on screen is left as it is.
  - **⌘Z / ⇧⌘Z in a focused text field with nothing typed in it reach the window's Undo and
    Redo,** as they already did in a spin box: a plot's Title box, a calibration
    field, a search box. A field you have typed in keeps them for its typing, and a table cell you
    are editing keeps them while its editor is open (press Esc first).
  - **A Plots-window fit over a typed Fit range is never the report's fit.** Such a fit records its
    range and is not saved when the window closes, and Generate Report figures never draws a saved
    fit that records one. The report uses the table's saved whole-column fit if it can, or fits
    that histogram afresh, and the window's FIT RANGE line says which. A project saved by an
    earlier version keeps the fit it saved.
  - **Batch detect records the detector it ran,** which is now the one the detector box shows,
    Schmitt or CUSUM (see *Batch detect runs the detector selected in the Detection panel*
    above). A batch then always ran Schmitt, and before this fix one made while the box showed
    CUSUM recorded `detect.cusum` on its Schmitt table, which moved the box when that table came
    on screen. The stale line's Re-run always replays the table's own detector, so projects saved
    before this are safe too.

- **A recording's samples are read-only in memory.** Every channel array is flagged read-only when
  the recording is built (the flag, not a copy), and so is each calibrated array the session
  caches. At an identity calibration every panel and worker reads the recording's own array, so one
  in-place write anywhere would have changed the recording for every surface, silently, for the
  rest of the session; it now raises at the line that tried it. No code in the app wrote in place;
  the rule is for the next line that would.
- **The detection's thresholds are judged with its chain.** A `k_in`, `k_out`, CUSUM `k` / `h`, min
  or max dwell or merge-gap edit made after a Schmitt or CUSUM run left the table on screen looking
  current. The chain notice now names it first — *⚠ Settings changed: k_in 5 → 6* — and Revert puts
  it back with the chain. The table's chip reads **⚠ Settings changed** then, and Revert and the
  chip's menu speak of settings (*Revert to detection's settings*, *Revert form to this
  detection's settings*).
- **A dwell or depth fit is judged by the events it fitted, not only by how many.** `fit_dwell` and
  `fit_depth` record a digest of the fitted column over the rows they were handed — each row named
  by its start, so a merge's renumbering cannot move it — with the table's sample rate and
  open-pore estimator stamp. The Fits tab and Generate Report figures then stop treating a fit as
  current when an accepted event was swapped, resized or re-measured at an unchanged count, which
  the count alone could not see. A fit made before a re-measure or a resize (an edge drag
  re-measures that event's depth) therefore reads as out of date, and one made before a merge of
  rejected events stays current; a fit saved by an earlier build has no digest and keeps the count
  rule.
- **Deleting a protocol, a saved ML preset or a plot template asks first**, naming it, with Cancel
  as the default and the escape button. All three are preferences kept on this computer and
  outside Edit ▸ Undo (a project saved earlier keeps its own copy of a plot template, but deleting
  it here is not undone), and each was one click with nothing asked. Deleting the ML preset the
  tab is on also says which default preset the tab falls back to.
- **A frequency clamped to a slower recording's Nyquist is announced, and comes back.** The status
  bar names the clamp — *Low-pass 20 kHz is above this recording's Nyquist (12.5 kHz); the form now
  reads 12.4999 kHz.* — for a stage that is on and a value someone set, never a box's built-in
  default, and the value you set returns on a recording that can hold it. A look at a 25 kHz
  recording used to leave a 20 kHz low-pass at 12,499.9 Hz on the way back.
- **Distributions and Generate Report figures ask whether a table is current, as the Results panel
  does.** Distributions names the columns of a source whose sample rate changed after it was
  detected, and of the table on screen when its open-pore estimator did, as it already did for a
  calibration change (*⚠ STALE: dwell_s*, and in Details *STALE: dwell_s on this plot is not what a
  fresh run would measure now …*). Generate Report figures refuses a table whose sample rate is
  stale, as the event exports do, and names a table stale on its calibration or estimator on the
  status line.
- **Ctrl+M files an event; Ctrl+Shift+M no longer adds (it now turns Analysis ▸ Measure event on and
  off), and *Mark selection as missed event* is gone.** The Measure
  tab's *Add to list* became **Add event** and the old measurement list is a group, **Old
  measurements (N)**, shown only while it holds rows (Go to, Remove, Export CSV; no longer added to
  or edited). The *measure on* Raw/Processed choice is hidden whenever the Analysis panel supplies the
  table's own signal, which is always in the app.
- The rise / fall search window of an event pads its edges by max(64 samples, 1.28 ms): bit-identical
  at 50 kHz and below, wider above it, so an edge fitted mid-ramp at 250 kHz no longer loses its 10 %
  point to a window that was a fixed 64 samples (0.26 ms there).
- The event table, its Summary / Fits / Quality tabs and the Merge sub-events strip moved out of
  the Inspector into a new **Results** dock under the trace (View ▸ Panels ▸ Results, ⇧⌘R; close
  it to give the trace its full height — nothing is lost). The Inspector keeps only the parameter
  forms, so they get its full height back (Detection page 256 → 592 px at 1512x900). The Results
  dock opens tall enough for six table rows, capped at a third of the window (at least four whole
  rows at 1512x900), and its two header rows have a fixed height: the toolbar (Table selector,
  Columns…, Merge…, the table's constant columns, advisories) and one status line — the event
  count, or the stale notice, which is now one amber line (verdict, action, Details for the full
  sentence) instead of a wrapped banner up to 254 px tall. When the action is unavailable the
  line starts its reason, giving up the verdict first on a narrow panel. Table rows are 24 px on
  a Mac (one text line + 8 px). A short panel (or Merge… open) scrolls the Events page instead of
  squeezing the table. ⇧⌘R hides the panel with the focus parked on the trace and, shown again,
  hands it back to the table, so a/r/f keep working.
  Saved layouts keep their dock widths and gain the Results dock; the layout is now saved as
  version 1, so a downgraded build resets the layout once and names `windowState` on its info bar.
- Keep scientific warnings and fit verdicts in Details/readouts and saved figure
  provenance, not on plot titles, fit legends or Event Gallery waveform captions.
  Optional fit statistics, units, curves and numerical exports are unchanged.
- Standard report PDF/PNG files retain diagnostic context in document/image
  metadata. New Figure Library captures include full Details text; existing frozen
  figures and previously generated guides are not rewritten.
- **View ▸ Zoom Back and Zoom Forward are greyed out when there is nowhere to go.** A zoom you have
  just made counts at once, so Ctrl+← straight after a dive still steps back.

### Added

- **Edit any text in a plot from the plot itself.** Double-click the title, an axis label, a legend
  entry or the fit label to retype it (Return keeps it, Esc cancels; empty = automatic, a space hides
  it). Right-click it for **Text style…** (font, size, bold, italic, colour, title alignment) and, on
  an axis, **Axis properties…** (label, range with Auto, log where the window has one, tick label
  size, major tick spacing where the window keeps one). Every change is one Undo step and is saved
  with the project. In Distributions, I-V, Voltage dependence, Noise, Event waveforms (including the
  colour-bar and marginal *count* labels), Distribution summaries (a legend rename sets that source's
  display label) and Subpopulations. Where several views share one X and Y label (Noise, Event
  waveforms, Subpopulations) the label is renamed in Appearance; on the figure it takes a text style.
- **Fit label.** A fit's result is shown in its own label on the plot: on by default, draggable (the
  position is saved), with your own words around live numbers (`{equation}` `{r2}` `{n}` `{ci}`
  `{bic}` `{ks}` `{source}`). An R² refusal such as zero degrees of freedom is always shown when R²
  is asked for. Noise shows a single curve's 1/f exponent α in it.
- **Annotate mode on the trace** (E, Ctrl+Shift+E, the **✎ Annotate (E)** button under the trace and
  on the Measure tab): a drag that fits both edges adds the event at once, one Edit ▸ Undo step, and
  a double-click on an event adds it.
- **A right-click menu on the trace** (not on its axes): Add event here, Add the selection as an event, Discard the
  selection, Delete this hand-made event, and the Annotate switch, with the key for each.
- **Analysis ▸ Measure event has a shortcut, Ctrl+Shift+M (⌘⇧M).**
- **Plot pages: Ctrl+1–4 (⌘1–4) switch Section** to Data, Analysis, Appearance or Export (not yet in
  Distributions).
- **Right-click the tick numbers of a plot axis for their own menu:** **Text style…** sets the tick
  labels' font, size, bold, italic and colour, and **Axis properties…** is there too. A right-click
  on the axis label still gives the label's menu. While a label is hidden, the tick numbers' menu
  offers **Show label**.
- **Add title…**: right-click an empty part of a plot whose title is empty or hidden and type the
  title in place. Return keeps it as one Undo step; Esc, or Return with nothing typed, leaves the
  figure as it was. A hidden title also offers **Show title**. Where the views share one title
  (Noise, Event waveforms, Subpopulations) the entry reads **Add title (this view)…** (**Add title
  (both figures)…** in Subpopulations): the title goes back to automatic when the view changes.
- **Distributions: every legend row has Text style… and Hide entry** from a right-click (one Undo
  step, saved with the plot and the project, not carried by templates or presets); a fit's row
  renames its source on a double-click. The *below detection limit* words can be retyped, styled or
  hidden from the figure; while they are hidden, a right-click on the plot offers **Show
  detection-limit label**.

- **Reset view, zoom history and a pointer readout in the plot windows.** Distributions, Noise
  spectrum, I-V / conductance, Voltage dependence, Event waveforms, Distribution summaries and
  Subpopulations share them (§10, *A consistent plotting workflow*).
  - **Reset view** returns the zoom and pan to the plot's own framing: the fit to its data, the
    category slots of a box or violin, or Distributions' typed Axis range. It does not re-plot,
    restyle, change a setting or add an Undo step.
  - **Zoom back / Zoom forward** step through the places you zoomed or panned to, on the plot or
    on a strip linked to it. They are greyed out when there is nowhere to go. The history starts
    again when the plot is drawn anew, and an appearance edit keeps it.
  - **The right-click menu is the app's own:** Reset view, Zoom back, Zoom forward and the page's
    exports. pyqtgraph's generic menu is gone, with its log, FFT and invert transforms, which
    changed the plot outside its saved settings, and its Export… dialog. Subpopulations, which had
    no menu, now has this one; it has no toolbar button, since it shows two figures at once. The
    colour bars of the event-count heatmap and the spectrogram lost their colour-map menu, which
    recoloured them behind the page's saved colormap control.
  - **A status line shows the values under the pointer** in the units the axes show: the real value
    on a log axis, Distributions' Time unit, the source over a box or violin, and the count or log₁₀
    PSD over Event waveforms' density map, Distributions' Event-count heatmap or a spectrogram.
    On a log-frequency spectrogram it gives the true frequency and the log₁₀ PSD of the original
    frequency bin under the pointer, not of the display row. On a Distributions Scatter or PCA it
    also names the nearest point's source and event_id. It is never part of an exported figure.
- **Go to an event by its ID, or to a time, on the main trace.**
  - **View ▸ Go to event…** (Ctrl/⌘+J) takes an `event_id` of the event table on screen, selects
    it and zooms the trace to it, as a click on its row does. It frames the event again even when
    it is already the selected one. An ID the table does not have (IDs have gaps after a merge or
    a delete) moves nothing and the status bar gives the table's ID range.
  - **View ▸ Go to time…** (Ctrl/⌘+Shift+J) takes a time as the trace's x axis reads it (the
    instrument clock on a recording, the cut's own clock on a segment), and the prompt quotes the
    axis label. Where the axis and the event table's `t_start_s` differ (t₀, an alignment shift) it
    gives the number to add to a `t_start_s`. The view pans to centre the time without changing the
    zoom, and the crosshair is put on the nearest sample; the status bar says when the view could
    not move (full-trace zoom) or stopped at the end of the recording. A time outside the trace is
    refused with the trace's range.
  - **The Event inspector's ID… button** (Ctrl/⌘+J in its own window) goes to an event by its ID.
    The Event box beside ◀ ▶ stays a position in the filter, and its tooltip now says it is not the
    ID. An ID outside the current filter moves the filter to All, so the box shows a true position,
    and the progress line says why. Docked, Ctrl/⌘+J is View ▸ Go to event…, which moves the
    inspector the same way.
- **The trace crosshair names the sample index.** A third line reads *sample 5,000 (event-table
  index)*: the snapped sample in the numbering the event table's `start_idx` / `end_idx` use, from
  the first loaded sample. A partial load adds the file's own sample number, and a segment its
  parent's. On a recording with no sample rate, where the x axis already is that index, it is given
  once, in place of the time.
- **View ▸ Show detection thresholds draws what the detector compared the signal with.** Off by
  default. For a Schmitt table, the enter level (dashed) and the exit level (dotted): an event is a
  run beyond the exit level, so it starts and ends there, and it counts only if it reaches the enter
  level. For a CUSUM table, the drift level k·σ (dash-dot), since h is an amount of evidence and not
  a current. They are tracks, not flat lines: each is the detector's own moving open-pore baseline
  plus its threshold, rebuilt from the table's record and checked against the noise and baseline it
  stamped. A step in that baseline is drawn at its own sample. They follow the baseline-subtracted
  view, change no number and keep your zoom. Nothing is drawn, and the status bar says why, for a
  stale table, an ML, hand-made or reconciled table, or a table whose detection signal is not at the
  trace's level (high-pass, baseline-drift removal or Normalize in the chain). Hover the menu item
  to read the sentence again.
- **Edges moved by hand are marked on the trace.** A boundary that is no longer where the detector
  put it (an Event inspector mask drag, a merge) gets a dashed vertical line; an event drawn by hand,
  including one placed and fitted with the trace keys, gets dotted lines at both edges. Revert
  removes the mark. Tables saved before the app recorded the detector's own bounds mark only the
  hand-drawn events and the merges that took one in. A reconciled row is marked by the event its
  span came from. The Events tab's tooltip says what the marks mean.
- **View ▸ ML confidence track.** The ML detector now keeps the model's probability with its table,
  and this option draws it in a thin **ML p** pane under the trace while an ML table is on screen:
  0–1 over the analysed range, with the run's enter and exit thresholds as dashed lines, on the
  trace's time axis, and **p** at the crosshair. The curve is drawn with the trace's peak-preserving
  decimation, and samples no model window reached are left blank instead of drawn as 0. It is
  display only: turning it on or off changes no event and keeps the x and y zoom. The probability is
  held at one byte per analysed sample, rounded to 1/255, and only for the session that ran the
  detector. Edit ▸ Undo of an ML re-run brings back the earlier run's probability with its table,
  and each earlier run's probability counts toward the 128 MiB undo memory budget. It is not saved
  in projects; a reopened ML table says so in the pane and needs a re-run to draw it. Turning the
  option on while the ML table's channel is hidden says so on the status bar. Off at every launch.
- **A KDE overlay on Distributions histograms.** Tick **KDE overlay** on the Histogram page and
  press **Plot / Apply**: each drawn series — each ticked source, or each cluster of a split
  histogram — gets one Gaussian KDE with Scott's bandwidth (the violins' rule), drawn dotted in its
  colour over its observed range and scaled to the bars' y quantity (count: `n × bin width × f`;
  density: `f` per the column's stored unit; probability per bin: `bin width × f`). On a log x axis
  it is estimated on log10 of the column, matching the log-spaced bins. **Details** names each
  series' bandwidth, the legend has a *KDE (Gaussian, Scott)* row, and **Export plot data** writes
  the curves as `kde` rows. The setting is saved with undo, templates, the configuration JSON, the
  project and Figure Library captures; opening a project does not recompute it (Plot shows ●). A
  bin, normalization or log change before **Plot / Apply** redraws the bars without drawing a ticked
  overlay, and a Figure Library capture records the overlay only once it is on the picture.
- **An Event-count heatmap plot type in Distributions.** Any two numeric columns binned into a 2-D
  grid of event counts, with a colormap and a colour bar labelled *Event count*, using the same
  binning as Event waveforms' heatmap (`event_shape.density_2d`): Bins per axis, Log x (on by
  default) and Log y, the Time unit on time axes. It shows the first ticked source only and says
  so in the title and Details. Details carries the same comparability and amplitude warnings as
  Scatter for the same columns. The **Heatmap colormap** (Appearance) recolours it without
  re-binning or moving the zoom; an imported colormap this build does not offer opens as viridis
  and is named in the import note. **Export plot data** writes one `cell` row per bin with its
  edges and count; the exported figure carries the colour bar. Lasso and point-click selection are
  not available on it.
- **Event waveforms ▸ Ensemble aligns on the extremum.** **Align ▸ extremum** puts each event's
  deepest sample at t = 0, read with the event table's polarity, so a blockade that lowers the
  current and one that raises it are both aligned on their deepest point. The window reaches the
  longest dwell plus the padding to both sides, so every drawn event lies wholly inside it. The
  polarity used, and where it came from, is in the ensemble's Full details and the CSV. An event
  with no finite sample to align on is counted apart from the edge skips. After a project reopen
  the CSV's skip counts come from the count saved with the draw, as the title's do.
- **A variability band in the ensemble.** **Band** shades mean ± 1 SD around the mean or the
  25–75 % interquartile range around the median (default None). It is worked out from the traces
  on screen, so it never re-reads the recording or resets the zoom. It uses the same events as
  the mean and median, and is left out with them where an event's window holds a non-finite
  sample; Full details says where. The CSV adds the band and `n_events_at_point` (the events
  behind that row's mean and median), and the project saves the band.
- **The ensemble can use the Event gallery's subsets.** **Events** offers All accepted (the
  default), Unreviewed, Unlabelled, each label, and Hand-made or Detected only, always within the
  accepted events. The title and the CSV name the subset, and the project saves it. Subpopulation
  subsets are not offered yet.
- **Spectrogram colour limits.** **Spectrogram colour limits** is Auto (the 2nd–99.8th
  percentile, as before) or Manual, with Low and High in the colour bar's log₁₀ PSD. Changing them
  re-levels the image without recomputing it or moving the zoom. High always stays above Low.
  The caption gives the limits and their linear values in either mode, and the project saves them.
- **A log-frequency spectrogram, labelled as a display resampling.** **Log frequency (display
  resampling)** averages the evenly spaced bins, weighted by overlap, onto 512 log-spaced rows;
  the 0 Hz bin is not drawn. The axis and caption say so, and **Export data (CSV)…** keeps the
  original linear grid. The time range is kept, a narrowed frequency range is carried into the new
  axis, and a whole-spectrogram view is fitted again. Zoom back and forward start again, since
  the earlier views were on the other axis. The project saves the choice. Before the first
  spectrogram, or after one could not be computed, the page is blank, on a linear axis, with
  nothing for the pointer readout or Reset view to read.
- **Plot configuration files are version 3.** They add the KDE overlay, the heatmap's settings and
  its colormap. Version 1 and 2 files still open; a build that reads only up to version 2 refuses a
  version 3 file by name instead of dropping those settings. (That refusal covers the sidecar
  files; inside a project the same settings are protected by the project version, next.)
- **Projects are version 12.** A project now carries the Distributions KDE overlay and heatmap
  settings, the Noise spectrum's spectrogram colour limits and frequency axis, and the Event
  waveforms ensemble's band, extremum alignment and Events subset. A version 11 build (the
  2026-10-31 and 2026-11-01 releases) would have opened such a project, shown a heatmap as a
  histogram and an extremum-aligned ensemble as *start* without a word, and dropped those settings
  on its next save; with the version at 12 it refuses the project with *needs a newer NanoScope*
  instead, and it lists a crash-recovery copy written by this build as one it cannot read, and
  keeps it. Every older project, version 1 to 11, still opens here as before.
- **Templates in Noise spectrum, Voltage dependence, Event waveforms, Distribution summaries and
  Comparison figures.** Each page's Export section has a **Templates** group: a template combo,
  **Save as template…**, **Rename** and **Delete**. A template keeps the page's analysis settings
  and its appearance, never its sources, the Noise spectrum's time span or any result. A Noise
  spectrum template includes the spectrogram's colour limits and frequency axis; an Event
  waveforms template includes the ensemble's alignment (extremum too), band and Events subset.
  - Recalling one changes the controls, and does only what changing those same controls by hand
    would do. In Distribution summaries and Comparison figures it is a draft, marked unapplied,
    until you press **Generate**; a template that matches the controls marks nothing. In Voltage
    dependence the fit arrives selected and waits for **Apply fit**. Noise spectrum recomputes
    only when the template's Welch settings differ, and Event waveforms redraws a view only when
    one of its settings differs.
  - A template that differs only in appearance or style keeps your zoom and recomputes nothing,
    so recalling one on a project whose recording is missing keeps the saved spectrum or count
    grid.
  - The template's appearance is one step of that plot's appearance Undo.
  - **Rename** moves the saved settings unchanged. **Delete** asks first, with Cancel as the
    default. A template saved by a newer NanoScope, or for another window, is refused with a
    message that says so.
  - I-V / conductance has no template: its only settings are appearance, which **Save appearance
    preset…** already keeps.
- **A saved project carries copies of your plot presets and templates.** Appearance presets,
  Distributions templates and the new window templates are kept on the computer that saved them.
  Save Project now also writes a copy into the project. Opening the project adds the ones this
  computer does not have, and the banner names them.
  - A preset or template on this computer is never replaced. One of the same name with different
    settings comes in as *Name (from project)*, and the banner says so.
  - One this version cannot read is not added. It stays in the project and is saved back with it.
    A damaged one (for example holding NaN) is treated the same way, and the project still opens.
  - A carried Distributions template with the name of a shipped one comes in as *Name (from
    project)*, so it can be picked.
  - A computer with no presets or templates writes no copy. The copy is an optional section of
    the analytical workspace and needed no format change of its own, but it is saved in a version
    12 project, which a version 11 build refuses as a whole (*Projects are version 12*, above).
- **Edit ▸ Undo takes back analysis settings, on the recording on screen.** A third undo stack,
  `core/history.SettingsHistory`, beside the event tables and the measurement list, covers each
  event table's Detection, ML and Baseline forms, a channel's calibration, a recording's time base
  and a channel's acquisition bandwidth. Its steps are named — *Undo Change k_in 5 → 6*, *Undo
  Recall protocol ‘Zn2-insulin (zn_30h)’*, *Undo Apply calibration*, *Undo Reset calibration*, *Undo
  Declare 25,000 Hz*, *Undo Use the file's own time base*, *Undo Change acquisition bandwidth* — and
  an undo re-measures nothing: the table's stamps match again, so its amber line goes away. Undoing
  a calibration puts `<recording>.npcal.json` back exactly as it was before the step, byte for byte
  and without asking — a file the step created is removed, a file a Reset deleted comes back with
  its note — and the status line names the file; a file that is no longer as the step left it is
  left alone, and the line says so. A press never changes a recording that is not on screen: when
  that is all there is, the menu says so (*Undo — nothing on this recording* when no event table
  holds any history either, *Undo — nothing to undo on this event table* otherwise), and redo is
  kept per table and per setting, so a calibration change does not end a curation's Redo, nor the
  other way round. An edit still waiting out its 700 ms pause is committed before any other step is
  recorded, so a press takes the newer step first, and that form's Redo is withheld while it waits.
  Each setting keeps 30 steps, and a step dropped at that ceiling is announced as a table's is. Undo
  history is still not saved with a project.
- **Undoing a run takes back the settings that led to it, in one press.** After *change settings,
  run detection* the menu reads *Undo Re-run detection and 2 setting changes*, and redo replays the
  settings, then the run. Only that table's setting changes made just before the run join it; a
  re-measure joins the same way.
- **The Event gallery has a Redo** beside its Undo. It mirrors Edit ▸ Redo, names the step in its
  tooltip, and answers ⇧⌘Z.
- **Every plot window has its own appearance Undo and Redo** — Event waveforms, Noise spectrum,
  I-V, Voltage dependence, Distribution summaries, Comparison figures, Subpopulations and the study
  figure editor — in the Appearance section (beside **Reset figure** in the study figure editor; the
  other pages carry that button in the page toolbar), with ⌘Z / ⇧⌘Z whatever the page's Section
  shows (the study figure editor, which has no Sections, answers while its Appearance controls are on
  screen). An edit, a preset, a saved appearance, a colour and a reset are each a step, and edits
  within 700 ms join one; 200 steps, with the tooltip naming the step (*Undo: title size 12 → 14
  pt*) and counting any dropped at the ceiling. It is kept apart from Edit ▸ Undo, and ends when a
  project opens, a saved state is restored or **Reset all figure styling** is pressed.
  Distributions keeps its one recipe stack, which already held its appearance.
- **Signal Conditioning: Revert stage and Reset stage.** The stage list marks each stage that
  differs from the run on screen (bold, in the chain notice's amber). Under the stage form,
  **Revert stage** puts back that stage's settings from the run, and **Reset stage** restores its
  default settings and leaves it on or off; each is one named undo step (*Undo Revert Notch stage*,
  *Undo Reset Notch stage*).
- **Open-pore baseline: Back to default in every mode, and Clear reference.** Back to default
  resets the estimator alone — its windows, their source and the *ACCEPTED OVER* mark — and now
  shows under a user-provided reference too, beside a caption that reads *Automatic estimator (not
  in use): 0.5 s / 0.1 s*; a press there leaves a table measured under that reference current. The
  new **Clear reference** puts the recording back on Automatic and keeps the windows.
- **The amber line's Details lists every input a table was made from** — sample rate, calibration,
  open-pore estimator, signal chain and thresholds — each judged on its own, as recorded and as it
  stands now, with its own way forward (Re-run, Re-measure depths, Revert). A table stale on two
  counts used to name only the first. One pure judge, `analysis/lineage.py`, gives the verdicts.
- **Hand-made events.** **Analysis ▸ Add event from cursors** (Ctrl+M) files the region selected on
  the trace as an event row in the table on screen (Schmitt, CUSUM, a study table), or
  in a new per-channel **Manual** table that the Table selector always offers ("Manual — no events
  yet") and the first add creates. One door, `AnalysisPanel.add_manual_event`, is behind Ctrl+M, the
  Measure tab's **Add event** button and the trace keys; the row is measured under the table's own
  provenance by `recharacterize_span`, is accepted, `added` and `reviewed`, joins the table in time
  order, and is one undo step. Every refusal changes nothing and names the way forward: no sample
  rate, a non-current channel, a stale table, a span outside what the detector searched, a span that
  overlaps an event, a Reconciled table. The table on screen is remembered per channel, however it got
  there, and saved with the project (workspace key `event_table_choices_v1`; no format bump). Rows carry `note`, `origin`,
  `bounds_source` (`drawn`, `placed`, `fit:step`) and the span as drawn.
- **Trace keys, draw mode and fitted edges.** With the trace focused: **C** catches the event under the
  pointer and fits both edges, **S** / **F** place an edge at the pointer fitted to its step
  (**Shift+S** / **Shift+F** exactly), **D** adds, **Shift+D** adds and goes to the next window, **G** /
  **Shift+G** step windows, **X** discards the mask, **E** turns **Annotate mode** on or off (also the
  **✎ Annotate (E)** button under the trace and on the Measure tab, and **View ▸ Annotate: drag adds an
  event**, Ctrl+Shift+E). In it, a drag across an event fits both edges and, when both fit, adds the
  event at once as one Edit ▸ Undo step; a double-click on an event catches and adds it. A drag that
  does not fit both edges, or that holds more than one event, adds nothing and says why; the span
  stays: **D** adds it (press it twice when an edge is not fitted, to add it as placed). Shift at
  release keeps the span to adjust, and **Option**-drag box-zooms. Outside the mode, Option-drag draws
  a span to look at before adding it with **D**. The mode is not remembered between launches. Return / Enter / Esc are aliases that act
  only after a click or the pointer arriving, never after a dock parked focus on the trace. New
  `analysis/edge_fit.py` fits the step with gates checked against correlated noise. Keyboard focus
  follows the pointer onto the trace, with Measure on or off, unless a table, text field, menu or the
  docked Event inspector holds it, a horizontal swipe
  or notch pans **time**, and **Shift+↑ / Shift+↓** zoom about the pointer.
- **Measure tab: preview and unstored statistics.** A preview line shows the row the add would store
  (measured like the chosen table's own events, on that table's signal), refusals before the click,
  and the channel's hand-made counts. The statistics beside it are measured on the same signal against
  the table's baseline and noise, so the baseline shown is the baseline stored.
- **Hand-made events across the app.** The gallery marks them **M**, the inspector says so and offers
  **Delete**, the table and gallery menus delete them, and the source filters gain *Hand-made only*,
  *Detected only* and *Merged (hand + detected)*. **Re-running detection keeps them**: each is
  re-measured under the new run, labels move onto a detection of the same event, and the status line
  says what was kept and what was left out. **Analysis ▸ Copy hand-made events to…**,
  **Include hand-made events in exports** (events and training exports) and **Export event
  statistics…** (mean, median, std, variance, min, max, peak-to-peak, RMS, skewness, kurtosis, area,
  amplitude and SNR for one table, every table on the channel, or every Manual table in the project,
  with a `signal_basis` column).
- **Analysis ▸ Convert old measurements to events…** turns the saved manual measurements you tick into
  events in the channel's Manual table (re-measured as events; the dialog says which numbers change),
  hides the converted originals from the Distributions plot source, and is one undo step.
- **Inspector ▸ File Info ▸ Sample rate / time unit…** declares a recording's time base when the
  file does not say, or says it wrongly: the unit its time column is written in (the rate and t₀
  move together) or the sample rate it was acquired at (the only option for a legacy TDMS, which
  declares none). A column the 1 Hz floor refuses keeps the interval it measured, so its unit
  can be declared the same way. The declaration is saved with the project under a new
  workspace key (no format bump of its own; a version 11 build kept it, and refuses the version 12
  project this build now writes), never written beside the raw
  file, shown in File Info as `100,000 Hz · user (file: 100 Hz · measured)` (the status bar
  prints the declared rate and its provenance), and followed by the recording's segments.
  Detections made at the old rate are marked stale (*Times stale — sample rate changed*) and
  offered a re-run — refused before the click when a stored cutoff is above the new Nyquist —
  and a saved manual measurement gets the same mark on its row; nothing is re-scaled in place.
- A CSV time column whose **header names its unit** — `time_ms`, `t_us`, `Time (ms)`, `time_s` —
  is read in that unit, with a Note in File Info saying so. A bare `time` is still read as
  seconds.
- The **Event inspector can dock beside the trace** (opt-in; it still opens as its own window by
  default): **Dock beside trace** / **Own window** on its toolbar, or **Window ▸ Dock Event
  inspector beside the trace**, moves the same window — event, filter and context kept — into a tab
  beside the Inspector panel (left or right column only), and the choice is remembered. Its
  minimum fell from 1107×827 to about 452×550 px (452×440 docked): the toolbar is two rows in a
  wide window and five short ones in a column, the amplitude row is two, the waveform's floor is
  180 px, and the readout scrolls and collapses under **Details ▾ / ▸** (hidden when docked, shown
  in a window, each remembered), with the Label row in the fixed bar under the waveform. The
  staleness sentence is a one-line chip (all of it in its tooltip, ⚠ on the Details button).
  Docked, ⌘S / ⇧⌘S / ⌘Z / ⇧⌘Z
  are the main window's own items (so neither copy goes dead as an ambiguous pair), a click anywhere
  in it or on its tab takes the keyboard, a tab sent behind hands the keyboard to the trace, and Tab
  never lands in the panel behind. In a short column the part under its toolbar scrolls, so no
  amplitude method, note or Details state pushes the main window past a 1280×800 screen. The
  waveform's title no longer widens the plot past its widget: a narrow plot draws a shorter form
  that keeps depth and dI/I0 whole (all of it in the tooltip), and a short axis reads **I (pA)**.
  The amplitude note reserves its height, so a staleness flip no longer moves the waveform.
- A **Window** menu: Bring All to Front; the main window, Event inspector, Event gallery,
  Analysis Explorer (and its open pages), Figure Library and the other windows NanoScope has
  opened, checked while on screen; Dock Event inspector beside the trace; and **Reset Window
  Layout…** (asks first). Child windows now reopen at their remembered size and place, validated
  like the main window's and brought back onto the current screen.
- Curation throughput in the Event inspector and the Event gallery: **Advance after decision**
  (off by default, one remembered choice for both windows) makes the A / R / Space / label keys
  move on to the next event of the filter — in the gallery only for a one-tile selection, turning
  pages; buttons never move on and decision keys no longer auto-repeat. New **Unreviewed**,
  **Unlabelled** and **Labelled: key=value** filters; **First unreviewed** (Ctrl/⌘+U); a
  fixed-height progress line (*Reviewed 312 / 870 · accepted 801 · labels: …*); a Label row
  (Apply / Clear) for the inspector's current event; ⌘Z / ⇧⌘Z in the inspector on the main
  undo history. Edit ▸ Undo / Redo no longer reset the inspector to All / the first event or move
  the gallery to tile 0 when the restored table holds the same events. An inspector readout line
  too long for the window now ends in "…" instead of widening the window, and in the gallery a
  decision that takes the focused tile out of the Show subset leaves the cursor beside it (the
  next ← / → selects its neighbour, not the page's first tile).
- Signal Conditioning window (Analysis ▸ Signal conditioning…, Ctrl+Shift+F; also the
  Preprocess card's Open button, the Filter step and the baseline workspace's Advanced
  signal-chain settings): the classical
  detectors' stage editor, moved out of the Inspector into one built-once window with an
  execution-order stage list, notices below the form, the full-height response view, protocol
  Import/Export (versioned JSON, never overwriting) and Copy as Python. The Preprocess tab keeps
  a fixed-height card. When the form's chain differs from the Schmitt/CUSUM detection on screen,
  the event table shows "⚠ Chain changed" with Revert / Run / Open, and the card and window name
  the differences. Ctrl+S / Ctrl+Shift+S save the project from the window, and its Run
  detection… follows the Detect button's gate. Recalling a protocol now sets every stage
  (stages it does not name, and the
  low-pass, are turned off), and its name stays on the combo only while the form equals it —
  a value the recording cannot hold (a cutoff above its Nyquist limit) drops it to "— manual —"
  and the status bar names what changed.
- Signal Conditioning previews of the main trace's visible window, computed on one background
  thread 250 ms after the last edit (a newer edit cancels the older preview): Time (raw against
  processed — Overlay, Split, Raw − processed — split onto the processed trace's real unit when
  Normalize or a high-pass changes it), Spectrum (Welch before/after with a deterministic line
  finder and Send to notch through the form's own checks) and Response (Magnitude plus Phase,
  Group delay and Step, applied zero-phase against the single-pass design). The status line says
  "Window estimate" while baseline drift or Normalize (or a stage settling longer than the 40 s
  padding, which it names) make a window inexact; Compute exact runs the whole recording on
  demand only, and the status line leads with what is happening (Computing exact…, Exact pass
  failed…, Updating…) before the current chain's verdict. A mains line is snapped onto its
  harmonic only when Δf ≤ 1 Hz. Send to notch on an off notch that keeps a list adds to that list
  and leaves the notch off. The window keeps its one preview thread from first open to quit.
- Opt-in Comparison figures page: shared-axis per-recording facets, feature-pair matrices with
  common-bin count histograms, and strip/swarm individual-event plots with optional descriptive
  boxes. Seaborn renders prepared observations; generation is explicit, source identities and
  exclusions are retained, and large/crowded inputs are refused instead of silently sampled.
  Appearance and workspace restore use frozen geometry. Native vector export, publication preview,
  Figure Library capture and bundles preserve exact tables and provenance; warnings stay in Details.
- Publication-workflow hardening: revise saved compositions from their embedded original panels,
  restore layout controls, preview and save as a new figure with portable revision lineage. Nested
  panels and absent source entries are supported; project-generation guards prevent modal saves
  into another project, including an identical reopened project.
- Extended detached shared ranges to exact built-in readable-axis formatters, including Population
  millisecond/log axes and PSD notation, while preserving uniform point styling. Versioned models
  prevent older readers from silently dropping unit-aware formatting. Unknown overrides and varying
  point styles remain unsupported; actual PNG dimensions are bounded before decoding.
- Next-phase plotting: frozen multi-panel composition with ordering, panel labels, grid layouts,
  preview-before-save and compatible shared ranges on detached Cartesian graphics. Each original
  panel's exact tables and resources remain in composition bundles; unsupported plots retain
  independent axes instead of silently losing annotations or image content.
- Selection-only lasso for Population Scatter/PCA with exact event IDs, modifier toggles,
  explicit multi-source choice, gesture cancellation and unchanged curation/analysis results.
- Opt-in Distribution summaries page with validated ECDF, uncensored empirical survival,
  linear-quartile box plots and Scott-KDE violins. Explicit generation, exclusion/comparability
  disclosures, presentation-only styling, frozen workspace restore and Figure Library capture
  are separate from existing detection, baseline, calibration and fitting behavior.
- Completed the core plotting-workflow additions: shared Data / Analysis / Appearance / Export
  sections, reusable appearance-only presets, persistent warning summaries with full details,
  and explicit event-count heatmap versus KDE terminology.
- Project-owned Figure Library, separate from Event Gallery, with named thumbnails, exact frozen
  data/configuration/provenance, source status, duplication, renaming, export and reveal-file actions.
- Graphical publication-export previews, explicit residual-panel inclusion and self-contained,
  checksummed figure ZIP bundles containing PNG/SVG, CSV/Parquet and analytical resources.
- Extended in-place Population styling to colours, markers, lines, legends and fit decorations,
  including appearance-only undo/redo and preservation of pending analytical settings.
- Opt-in event amplitudes: whole-event median, mean, peak and histogram mode; explicit
  selected-interval/state median or mean; and an editable equivalent level in Curate.
  Original median `depth_pA`, fractional depth, SNR, detection masks and state fits are unchanged.
- Measurement-domain Inspector traces, explicit units/reference labels, amplitude-only undo,
  project/CSV audit fields, guarded batch application and stale-choice invalidation after
  mask, merge or baseline changes. Plot pickers distinguish selected amplitudes from median
  depth and disclose missing values and non-comparable measurement choices.
- Shared opt-in plot typography, I-V/voltage-series appearance presets, and project-saved
  presentation overrides. Title/label/color-map edits in event/noise plots and text/canvas
  edits in Distributions update existing items without a new analysis or zoom reset.
- A shared **Export current figure** dialog with PNG/PDF/SVG, native or mm/in sizing,
  aspect lock, 85/180 mm width presets, and PNG DPI metadata. Standard report generation
  is labelled separately. Existing explicit-path exports and numerical CSVs are unchanged.
- Implemented project-v11 complete analytical projects: resumable plot and source state; frozen
  PCA, clustering and subpopulation results with preprocessing and seeds; PSD, density, I-V and
  voltage-dependence snapshots; fingerprinted ML-model references; and live analysis-panel state.
- Restored analytical snapshots lazily without reopening windows or silently rerunning detectors,
  solvers, fits or spectrum kernels.
- Added stable recording/channel/result lineage, missing-source and missing-model carry-through,
  and opaque future/plugin artifact preservation.
- Added explicit backward compatibility for v1-v10 projects and refusal by older readers rather
  than allowing them to strip v11 analytical data.

### Fixed

- **The first edit in Appearance no longer takes off what a project, template or import had put on
  the figure.** After one of those installs, Distributions' Appearance controls kept showing what
  they had last been told, so the next edit (a few letters typed in Title) took the fit label's
  numbers and the title's bold off the figure, and Reset figure did nothing because the controls
  believed they were already at their defaults. The controls now show the style that was installed.
- **Report-figure PDFs hold TrueType text, not Type 3.** *Analysis ▸ Generate Report figures…* and
  *Export density figure…* saved outside the NanoScope style, so matplotlib's default Type 3 glyph
  drawings were written (an IEEE conference's paper page says they are not accepted; Nature asks for
  TrueType, Type 42) although the style asks for 42. The PDF now embeds TrueType (Type 42) fonts.
- **The report figures' text is printed in the face it was laid out in.** It was laid out in
  Helvetica and printed in DejaVu Sans. The PDF and the PNG now both draw it in Arial, and a
  character Arial does not have, such as the subscript zero of the fractional-blockade axis ΔI/I₀ or
  the superscript minus of s⁻¹, in the next face of its list that has it (see the exports above).
  The text of these figures therefore changes face. Greek and subscripts written as math text
  (τ, µs, t_min) are still set in DejaVu Sans, which the PDF embeds as well.
- **Distributions: Undo on a project you opened goes back to what the project was saved with.** The
  Undo history began at the blank window, so the first change after opening (the plot type, say)
  was undone to the built-in defaults: the typed title, the font, the ranges you had typed and the
  fit label's numbers did not come back, and **Undo** with nothing changed wiped the page. The
  history now starts when the project is opened, at the state it was saved in. An **Undo** or
  **Redo** also counts as a change to the project: it marks it unsaved, and a project saved after
  one holds what is on screen (it held the state from before the Undo, so a **Reset figure** and an
  **Undo** in an opened project lost the saved look in the next save).
- **A composition with shared axes keeps the figure's font in the title and the legend.** A panel
  re-drawn with Share X range / Share Y range or One shared legend came back with its title and
  legend rows in the application font (a private face on macOS, which the composed PDF embedded)
  beside axis labels and ticks in Arial or the chosen font; with Times New Roman chosen the title
  was affected too. A figure saved by an earlier version has no font for its legend rows in its
  stored model, so those rows are still drawn in the application font; save the figure again.
- **Comparison figures: ⌘Z answers from every Section.** Reset figure sits in the toolbar and the
  reset when **Generate** commits another figure is made from whichever Section is showing, but the
  Undo keys answered only while Appearance was on screen. They now answer from every Section, as on
  the other plot pages (the study figure editor, which has no Sections, is unchanged).
- **Subpopulations: opening a project no longer carries the last project's title, labels and ranges
  into its first report.** The page is kept when a project is opened and was blanked without its
  appearance, so a project saved with no subpopulation report drew its first report under the title
  and the embedding range typed on the previous project's scatter, and a project saved while that
  empty page was open stored them in its own file. The page now starts from the defaults, as it does
  for a project saved without any; a project that holds a report still opens with its own words and
  ranges.
- **Distributions: a reopened project draws a histogram fit's label with every number it was saved
  with** (all parameters, n, BIC, the KS statistic and verdict), a typed fit-label template keeps its
  `{n}`, the legend's fit row stays "fit (…)" without numbers, and with **In legend instead** the
  row keeps its numbers through a reopen and every redraw. A Fit colour or line width chosen after a
  reopen restyles the restored curve too. An applied scatter fit, which opening a project does not
  recompute, now shows **Plot ●** until Plot draws it.
- **The fit label's source names** (the line prefix and the `{source}` token) draw `_{…}` and `^{…}`
  as sub- and superscripts, as the legend does; a legend name such as `τ_{1}` showed its braces in
  the fit label and in exported figures.
- **Return in an in-figure text editor without changing the text changes nothing** (Return on a
  hidden label's empty box used to bring the automatic label back). The editor opens inside the
  view.
- **Voltage dependence: the fit's numbers read in the axis's own unit.** A linear trend's slope is
  given per mV with the metric's unit (*slope = −0.07331 Hz/mV*; it read *slope = −73.31*, a
  per-volt number with no unit, beside an axis in mV), and the capture-rate model's V₀ is in mV
  (*V₀ = −500 mV*), in the headline, the fit label, Fit details and exported figures. The data
  export keeps the fit's parameters in volts, like its voltage_V column, and names their units in a
  new fit_params_units_json column. I-V's data export names the units of fit_slope and
  fit_intercept (new fit_slope_units, fit_intercept_units).
- **I-V, Voltage dependence, Noise spectrum, Event waveforms and Distribution summaries: a range
  typed in Axis properties stays in view** after a redraw — Refresh, Generate, a new setting, an
  Explorer tab switch, a saved view restored into the page (it was dropped from view while Axis
  properties still showed it) — until the axis comes to show another quantity, when it goes back to
  automatic with the typed words (see the reset rule, above). An appearance change still keeps a
  zoom made by hand. With a range pinned, the I-V and Voltage dependence empty-page message (*No
  usable recordings…*) stays in view; the range returns when there are points again.
  In Noise spectrum, Event waveforms and Subpopulations the range belongs to the view whose axis
  it was typed on: the other views keep framing their own data and their Axis properties show Auto
  (a range typed on the PSD had become every view's, so the Spectrogram and, from the ensemble, the
  Event-count map were drawn blank, and one typed on the Subpopulations scatter left the waveforms
  beside it almost blank). A range saved before this, with no view, is the PSD's in Noise spectrum,
  the Event-count map's in Event waveforms and the scatter's in Subpopulations. In Distribution
  summaries a range belongs to what
  the axes show: ECDF and Survival share theirs (value on x), Box and Violin theirs (sources on
  x), so a Type change between the two pairs starts on the page's own framing at Generate and
  Reset view, and changing back brings the range back (an x range typed on the ECDF had left
  a Box or Violin out of view). A range saved before this is ECDF and Survival's. Window
  templates keep each view's range, as they keep a one-view page's; presets keep no range.
- **⌘Z / Ctrl+Z (and ⇧⌘Z) undo and redo a plot's appearance whatever the page's Section shows**, in
  I-V, Voltage dependence, Noise spectrum, Event waveforms, Distribution summaries and
  Subpopulations, so an edit made in the figure can be undone from the keyboard. In the Analysis
  Explorer only the page in front answers.
- **Voltage dependence: dragging the caption is one Undo step** (*caption moved*) and its position is
  saved with the figure — projects, window templates and Undo keep it; presets leave it where it is.
- **Distribution summaries: renaming a legend entry in the figure is one Undo step.** The name still
  lives in Source names and colors, which Reset figure and templates leave alone.
- **Annotate: a loose drag fits the event.** A drag that overshot an event by more than about 12
  pixels was refused as "not significant against the noise", even for a step dozens of times the
  noise, and with the step just past the search window it could add the event with a start or
  finish tens of samples off. Each edge of a drawn span is now looked for across nearly half the
  span, at any zoom. An edge drawn so far out that no step is near it says *no step near the drawn
  edge* and suggests drawing closer or C. C, S, F and Measure fit exactly as before. A double-click
  on the trace's axes no longer catches an event. A tight drag on one event beside a neighbour
  nobody had annotated yet, a few pixels away at about 12 samples per pixel or coarser, was refused
  as *more than one event*: when the span drawn holds one event and its fit reaches a neighbour,
  the edges are now fitted again without crossing the baseline before that neighbour, so the drag
  adds the event drawn. That refit is kept only when it holds one event; a drag that reaches close
  to the neighbour can still be refused, and then a drag closer to the event's edges adds it.
- **Results ▸ Columns… remembers across tables.** Hiding or showing a column on one table, then
  changing columns on a table without it (amplitude columns before Curate ▸ Apply, ML or hand-made
  columns), forgot the first choice: the hidden column came back, and a default-hidden column you
  had shown was hidden again. **Default columns** still resets every column, those included, and
  keeps the order. **Default order** places them in the scientific order too.
- **Distributions: typing in the Title box made its first keystroke an Undo step of its own** (the
  appearance section spells a colour `#0072b2`, the window `#0072B2`); typing then Return is one step.
- **Distributions: text with < or & in a title, axis label or legend name lost characters**, and the
  *Features (clustering & PCA)* title showed a mnemonic underline instead of the &.
- **Distribution summaries: a source name containing < or & lost characters in the legend.**
- ***Applied voltage (V) (x0.001)*** on the I-V and Voltage dependence axes.
- **The Measure tab's status line no longer outlives the mask it describes.**

- **Crash recovery can no longer lose work by itself or through another window.** A user lost a
  session to a crash with only an export left; the investigation found these gaps, now closed:
  - **A second NanoScope window** could offer a running window's recovery as a crashed session,
    and Discard deleted it. That window's later checkpoints were then dropped without a word. Each
    window now holds a lock on its own recovery folder, released only when it exits or crashes,
    and a locked folder is never offered, discarded or pruned.
  - **Checkpoints are taken sooner:** after 5 s without an edit (was 15 s), at most every 20 s
    (was 60 s), and at once when you export events, when a detection or batch run starts, and
    when you switch away. An export could be newer than the last checkpoint.
  - **Discard is no longer permanent:** the copies go to `Recovery/Discarded` for 7 days.
  - **The prompt names the work:** recordings, event tables with accepted counts, the last edit
    and the build.
  - **Nothing is pruned or hidden silently.** A checkpoint holding work is kept 90 days and is
    never pruned to make room. A damaged `retirement.json` no longer hides a valid one. Copies
    this build cannot read are reported at launch.
  - **A failed checkpoint stays visible:** a banner remains until one succeeds, the status bar
    shows when the last one was saved, and NanoScope retries every 30 s.
  - **A missing recording no longer reads as lost work:** the banner says its tables were kept
    and will be saved with the project.
  - **Scripts stay out of your recovery folder:** a `MainWindow()` built by a script or test no
    longer writes there; only the app's own launch turns recovery on. Each checkpoint records the
    build that wrote it.
- **A crash can be traced afterwards.**
  - The app log records the build id and process id at each start and a *clean exit* line at each
    normal quit, so a crash can be told from a quit.
  - A crash (segmentation fault or abort) leaves a Python stack trace in the log.
  - An error the app survives, in a button, menu or background thread, is logged with its time
    and thread.
- **The trace crosshair's readout is drawn.** It was placed above the top of its pane, where the
  plot clips it, so the time and value it was meant to show never appeared. It now hangs from the
  pane's top edge beside the cursor line, on a translucent backing, and moves to the line's left in
  the right half of the view so it is never cut off. It keeps to the top edge when the view is
  zoomed or panned under a still cursor, and draws nothing where there is no sample under the line.
- **The crosshair's time on a segment.** On a cut, whose axis is its own clock, the crosshair took
  the axis value for the instrument time and printed a negative time *in*: 0.2010 s into a cut
  made 0.4 s into its parent read *-0.1990 s in · 0.2010 s instrument*. It now reads *0.2010 s in
  · 0.6010 s instrument*, which agrees with the *parent sample* line under it.
- **One cursor across the trace after a rebuild.** Switching Overlay / Stacked or toggling a
  channel left the trace panes' cursor line at 0 with no readout, while the ML p pane kept its
  line and *p* at the cursor. The trace panes now put their line and readout back on the cursor
  too.
- **Re-measure keeps an old table's run.** On a table from a v1 or v2 project (no stored settings,
  its run known from the detector's own stamp), **Re-measure depths** stored only the open-pore
  estimator's settings as the table's run.
  - The chain notice then read *⚠ Differs from the Schmitt run: low-pass off → on*, and Revert
    turned the run's low-pass off.
  - A polarity edit went unnoticed.
  - A later Re-run replaced the table with a different detection.

  The re-measured table now keeps the settings in its stamp, under the new estimator. A project
  saved with only the estimator's settings stored reads its run from the stamp again when opened.
- **A Re-measure is always named *Re-measure depths*** in Edit ▸ Undo. With the estimator
  unchanged (Analysis ▸ Compare estimators keeps *Re-measure now* enabled), it was named *Re-run
  detection*, and the status bar said the curation was not carried over although every
  accept/reject was kept.
- **Details names every Baseline drift setting that changed.** A polarity, mask, dilate or
  percentile edit read the same on both sides of the Signal chain row (*Baseline drift block median
  · 0.5 s → Baseline drift block median · 0.5 s*); the row now names it.
- **Compose with shared axes no longer draws a blank title band** over a panel whose figure had no
  title, or a hidden one. A titled and an untitled panel still line up.
- **⌘Z and ⇧⌘Z work after a spin box's arrows or a pick from a list.** A spin box changed with its
  arrows or Page Up/Down, or an editable list (the plot windows' Font was one, before the short
  font list replaced it), kept the key while it had focus and did nothing with it, so the plot
  windows' appearance Undo/Redo, Edit ▸ Undo/Redo and the Signal Conditioning window's ⌘Z were
  dead until focus moved. The control now keeps the key only while it has typing of its own to
  undo (`gui/app.py`, `_UndoKeyGuard`).
- **The plot appearance size boxes step from Default straight to the smallest size a figure draws**
  (6 pt for title, axis, tick and legend text) and back, instead of through 1–5 pt, which changed
  nothing.
- **A calibration Apply that changes no number writes nothing when the file already holds those
  values** (a version-1 `.npcal.json` is no longer re-encoded by it), so a Redo kept for another
  channel can still put its step back. An Apply that does change the file without moving a number
  (after opening a project whose calibration differs from the file, or the Apply the "left as it
  is" line advises) is now an undo step, and Undo puts the file back byte for byte, note and all.
  Before, it had no Undo, and later undos could leave calibrations you had undone in the file, to
  rescale the recording on its next open.
- **Generate Report figures no longer draws a saved fit of another quantity.** A Plots-window
  Gaussian fit of `depth_pA` was drawn over the dI/I0 histogram as "your saved fit" — a flat
  curve, with μ in pA. A saved fit that records another table of the recording is refused too,
  even when the accepted counts agree. Such a histogram is refit afresh and the status line says
  why (*saved fit is of depth_pA, not the frac_depth this figure draws*; *saved fit belongs to the
  CUSUM table*). A fit that records no table is used as before when its calibration and count
  agree.
- **Undo and Redo keep their promises in five more cases.**
  - Detect after a Baseline-only change is one ⌘Z, named *Undo Re-run detection and 1 setting
    change*. It could read *Undo Resize N event masks* and need a second press.
  - ⌘Z pressed while a run is in flight no longer leaves a Redo that does nothing once the run
    lands.
  - A step on another recording no longer changes the order in which this recording's undone
    steps are redone.
  - A batch *Re-run all* over a table that already exists no longer takes an un-run edit made on a
    detector with no table: the edit's draft and its undo steps stay with that form. Only a
    recording's first table takes them, as before.
  - A project draft holding a value of the wrong type is not applied (the open says so), and a
    draft the form cannot take is dropped without leaving the form half written.
- **Revert is withheld when the run's frequency is above the recording's Nyquist now.** After a
  slower sample rate was declared, Revert, the chip menu's Revert and a stage's Revert stage stayed
  enabled for a low-pass (or high-pass or notch) the run recorded above the new Nyquist; each press
  said "reverted" and changed nothing. They are now disabled, and the notice says why (*Revert
  cannot restore the run's low-pass 10 kHz, which is above this recording's Nyquist (8 kHz)*).
  Reset stage on a recording slower than twice a default frequency no longer resets again on
  every press.
- **The amber line's Details list agrees with the controls.**
  - It no longer offers Revert for the signal chain or the thresholds when every Revert control is
    disabled; it says *Re-run to apply*.
  - A rate-only declaration no longer lists the open-pore estimator as changed, since its
    decimation is derived from the rate. The estimator row's action is then Re-run, never the
    Re-measure the panel refuses for such a table.
- **The Fits box is redrawn when the calibration changes or is undone,** so its advice (*press Fit*
  or *re-run detection*) agrees with the amber line.
- **⇧⌘Z redoes in the main window on a Mac.** Edit ▸ Redo registered the standard Redo key and
  Ctrl+Shift+Z, which resolve to the same key on a Mac; Qt answers a key registered twice with
  "Ambiguous shortcut overload" and triggers nothing, so only the menu item redid, while every test
  (offscreen, where Redo is Ctrl+Y) passed. Each resolved key is now registered once, through one
  helper, `gui/key_sequences.py`, shared with the Plots window, Signal Conditioning and the gallery.
- **Edit ▸ Undo follows a Table-selector or detector switch.** It went on showing the old table's
  state until the Edit menu was opened — offering a step the table on screen did not have, or
  staying disabled over a table that had history, where ⌘Z does nothing.
- **⌘Z and ⇧⌘Z work in the Signal Conditioning window**, where the chain is edited. They did
  nothing there; a spin box you are typing in still keeps ⌘Z for its typing.
- **Switching recordings or tables no longer throws away settings you have not run.** The form was
  re-seeded from the new target's stored run, so a low-pass edited on recording A was gone after a
  look at B, and with no stored run on A the form kept B's values. Each event table now keeps its
  own Detection, ML and Baseline form, edits included, and the edits are saved with the project
  under a new workspace key, `analysis_form_drafts_v1` (no format bump). An edit of the ML form,
  which marked nothing unsaved before, now does.
- **A CUSUM table from an older build is judged, seeded and re-run on its own `k` and `h`.** A table
  that kept its settings only in its provenance (a project older than v3 stored no params) lost the
  CUSUM thresholds on the way to the form: the chain notice never compared them, the form kept its
  own `k` and `h` instead of the run's, and the amber line's Re-run replayed the run without them.
  The notice, the form and the Details list now read one record, the detector's own stamp.
- **Calibration ▸ Reset to default no longer puts gain 0.1 on a voltage channel that opened at
  1.0**, and no longer clears the channel's note and its "Treat this channel as" choice. It
  installed the default asked again without the channel's median, which cannot tell the
  10×-inflated encoding from a host-guest channel reading its bias directly: 10× on every voltage,
  conductance and V-dependence figure. It now restores the gain, offset and units the channel
  opened with and keeps the note and the choice. The sidecar compares with the same opened default,
  so a write no longer stores the untouched voltage channel beside the one you calibrated, and a
  0.1 typed on such a channel is now stored instead of being taken for the default.
- **Revert puts back only what differs.** *Revert to detection's chain* rewrote all 37 chain keys
  and the polarity, filling any key the run had not recorded with its default. It now puts back
  only the stages and thresholds that differ, and when a difference is in a setting the run did
  not record it is withheld, with the reason (*Revert cannot restore a setting the run did not
  record*). A Revert is one undo step (*Undo Revert chain to Schmitt's run*).
- **Re-measure depths refuses a table whose calibration has changed.** It measured under the live
  calibration and kept the table's old calibration stamp, so the depths had moved while the amber
  line went on saying *Depths stale — calibration changed*. It now says *Nothing was re-measured.
  Re-run first — the calibration changed since this table was detected.*
- The User Guide quoted an Edit-menu ceiling message the app never shows ("… N of your steps have
  been evicted"); §9b now quotes the status-bar line and the Edit-menu wordings the app does show.
- **An ML notch line typed as `inf` or `1e999` is dropped like any other entry that is not a
  frequency.** It parsed and was positive, and the pipeline's own check then refused it, so reading
  the ML form raised from the one box that is meant to be lenient: Save project refused and the Run
  ML button did nothing. An empty result still falls back to the trained 60 / 120 / 180 Hz.
- **The amber line's Details popup is no longer cut off at the bottom.** It kept the height of its first
  open, so a longer sentence, or a list of inputs that had grown since, ran past the end of the popup.
  Every open now measures it again.
- **A project can be saved after "Close only this recording".** Once a segment's parent recording was
  closed, every save failed with "Could not save the project:" and the parent's path, and wrote
  nothing, not even other recordings' work; the crash-recovery copy stopped updating too ("Could not
  update the crash-recovery copy: KeyError ..."). The save now reaches the "N segment(s) cannot be
  saved" question, **Save without them** writes the rest, and the recovery copy keeps the dropped
  segments until they are saved. Broken since 2026-09-12.
- **The Event inspector can be dragged narrow again.** The Detection page's Controls row (Essential /
  Advanced and **Baseline reference…**) kept the docked inspector at least about 330 px wide since
  2026-09-11; it now sits in a pinned strip and the dock narrows to its tab bar as before.
- **Curate in main workspace** (study workflows) no longer opens an event the study accepted but that
  touches the first or last sample of the exported trace as accepted beside `clipped_by_boundary`, and an
  imported `dwell>max_dwell_s` rejection keeps its reason (and stays rejected) instead of arriving as a
  rejection with no reason.
- **Saving a study recipe** refuses a read-only recipe instead of replacing it, and keeps the file's
  permissions and extended attributes; it had reset them to 0600 and dropped the attributes.
- Study windows recognise a recording opened under another capitalisation of its path as the same
  file (macOS volumes ignore case), when reopening a review copy and when saving curation back.
- **Save figure bundle** and **Save study recipe** ask through the app's own save dialog, with its
  overwrite question, like every other export.
- Closing a study figure editor while its styling was still being saved to an Acid study reported a
  failure after a save that had succeeded; the editor now hides on close and is freed later.
- The Plots window's Data sources buttons (**Save name as project default**, **Project source
  color…**) are stacked, so the controls column fits its 360 px budget again.
- Loading or clearing an ML model, or a channel override, no longer moves the ML page's preset row and
  its **Save as…** / **Delete** by 38 px: the gate's note keeps its reserved row again, as it did
  before the Inspector restructure of 2026-09-04.
- The User Guide names **Analysis ▸ Noise spectrum of the measured region** and **Add event from
  cursors**, the menu's own labels.
- **Every add files the same row.** Ctrl+M and the Measure tab's **Add event** button now add through
  the same path as **D**: the same record of how each edge was made (`bounds_source`, the span drawn,
  `fit_json`), the same second press for an edge that is not fitted ("Press again to add it as
  placed"), and the mask cleared afterwards with Measure left on. They had stored a caught event as
  drawn, without its fit, and kept the mask, so a second Ctrl+M was refused as an overlap. The
  second press now confirms only the mask it was asked about: a new mask made within the ten
  seconds was filed by its first press. A table that takes no hand-made row (Reconciled, a stale
  table) is refused on the first press, before any question about the mask.
- **S and F move one edge of any mask.** On a mask placed by a double-click, Snap to view, Snap to
  selected event, Go to measurement or a drag, S or F threw the other edge away and collapsed the
  mask to one sample. Now the other edge stays; only the mask Measure lays down when it is turned on
  is no instruction until it is moved. With one edge still unset (S or F alone) an add is refused and
  names the key that sets it, and the Measure tab's preview says so first; **D** had filed the
  one-sample stand-in as an event. `fit_json` keeps each edge's own fit, where it had described only
  the latest fit and could call an edge stored as `fit:step` unfitted.
- An event cut off by the recording's end ends at the last sample and is rejected automatically as
  `clipped_by_boundary`; it was filed one sample short and accepted. **Shift+D** at the last window
  keeps the add's confirmation and adds that there is no next window, instead of replacing it.
- **An add goes to the table on screen.** With Reconciled on screen, Ctrl+M filed into the source
  table remembered from before, and **Edit ▸ Undo** then reverted the Reconciled table instead. The
  remembered table now follows the screen (the selector, a new run, Reconcile), a saved project
  reopens on it, and Reconciled on screen refuses an add with the way forward. A **Batch detect**,
  or a Run that finished after you moved to another channel of the same recording, no longer puts
  that channel's own older table of the same detector on screen in place of the one you were
  working in.
- **Export event statistics** no longer mixes gains: a table measured before a calibration change is
  refused before the save dialog when it is the table on screen, and left out and named when the
  scope is every table on the channel or every Manual table, as a table on another sample rate is.
- The docked Event inspector lost A, R, Space and the label keys whenever the pointer crossed the
  trace, because the Inspector panel beside it had been registered as the trace's key owner. With
  Measure off, how a recording opens, focus did not follow the pointer onto the trace, **C** did
  nothing and **Shift+↑ / Shift+↓** ignored the pointer; the trace keys now act on the selected
  recording's current channel with Measure on or off.
- After a dock closes and parks the keyboard on the trace, a click on the trace wakes Return, Enter
  and Esc, as the User Guide says. The click did not count, because the trace already had the
  keyboard: Return added nothing and Esc neither discarded the mask nor cancelled a draw, while
  **D** added, until the keyboard left the trace and came back. The pointer merely arriving still
  wakes nothing after a park: the trace grows under a pointer resting on the closed panel, and a
  habitual Return there must not add.
- **Treat this channel as** (Calibration tab) takes the trace keys to the new current channel. When
  it made another channel the recording's current one, the analysis moved there, but the keys, the
  Navigator and the Scan strip stayed on the old channel until the recording was selected again:
  with Measure off, focus followed the pointer onto the old channel's pane and not the new one's,
  **Shift+↑ / Shift+↓** ignored the pointer on the new one, and the label on the trace stayed on the
  old pane. A mask on the old channel is still cleared, as selecting another channel clears it.
- **Delete** of a hand-made event from the inspector dropped its *Hand-made only* filter and put both
  curation windows on a detected event. Each window now keeps its filter and lands, quietly, on the
  next event of its own list. **Edit ▸ Undo** of an add the windows were showing does the same.
- **An add moves a curation window only when its filter shows the new event.** Under *Detected only*,
  or *Accepted only* for an event the recording's end cut off (stored rejected), an add put the
  inspector on the new event outside its filter, so the next A or R ruled on it, and left the gallery
  with no cursor and no selection, so A, R and Space acted on nothing. Both now stay on the event
  they showed. A filter a Delete emptied stays chosen, and when the Undo, the next add, the undo of a
  curation step or a reject in the Results table fills it again, each window shows its first event;
  the inspector had gone back to *All* on the first event (also when the recording was selected
  again), and the gallery had left the restored tile unselected (or, when the Delete had emptied a
  Manual table, announced it as a click).
- The Measure tab's preview went blank and **Add event** greyed out on a span holding a NaN or Inf
  sample, while Ctrl+M stored the row; the preview now shows that row, marked *would be rejected:
  non_finite*, and **Add event** is available whenever there is a mask.
- Re-running detection kept only one note when several hand-made events landed on one detection; every
  note is kept now, joined in time order with `; `. With **Include hand-made events in exports** off,
  the training export's sidecar counted the switched-off rows as withheld. After a re-run with another
  low-pass, the edge fit used the old run's open-pore tau.
- **A merge keeps what every fragment said.** Merge all, Merge events in selection and the Event
  inspector's Merge with next kept only the first fragment's `note`, `origin` and
  `legacy_measurement_ids`. The merged event now joins every fragment's note in time order with `; `,
  each once, and lists every fragment's origin and old-measurement id once, separated by commas
  (`12,15`). Converted events merged into one had left the old list offering every measurement but
  the first as unconverted, and **Copy hand-made events to…** called merged copies overlaps instead
  of already copied. Undo brings back each fragment as it was.
- A merged event says how each of its edges was made: `bounds_source`, the drawn bound and the fit
  in `fit_json` come from the first fragment for the start and from the fragment that ends last for
  the end. It had kept the first fragment's record for both, so its end read as that fragment's end,
  a drawn bound inside the span and its fit. An edge from a detected fragment records nothing.
- Event inspector and Event gallery: A, R and Space no longer take the key from the control that
  has the keyboard. Space on a Tab-focused button (Show in trace, Gallery…, Accept all / Reject
  all, Accept selected / Reject selected, Apply…) or check box presses it instead of toggling
  the event on screen or flipping every selected tile. The letters, Space and page keys typed
  into an open Filter, Show or other dropdown, or the gallery's Select menu, go to the list
  instead of accepting, rejecting or paging (typing "r" in the open Show list rejected the
  selected tiles, and PgDn turned the page and dropped the selection). Advance after decision
  (both windows), Dock beside trace, the Label combo with Apply and Clear, and Details had been
  taken out of the Tab chain so Space could not hit them; Tab reaches them again and Space
  presses them, while a mouse click on them still leaves the keyboard on the waveform or tiles.
  A, R and Space still curate from the waveform, the tile view and the window background.
- Pressing **Detect** (or any other analysis) while a **Merge all**, **Re-measure** or **Compare
  baseline estimators** was running replaced that job, and from then on the Manual study's and
  the Acid study's "prepare review" actions and the study import refused with "Wait for
  main-workspace analysis and editing operations to finish first." although nothing was running;
  selecting another recording, a second Detect or closing every recording did not clear it. The
  job it had replaced left its bookkeeping behind, and the same left "Measure this recording"
  reading "Measuring…" (and refusing a second click) and the Baseline-correction workspace's
  Refresh reading "Refreshing comparison…" until the recording changed. The analysis controller
  now announces a replaced job and the panel forgets everything it recorded about it, releasing
  the busy button that job owned (a job that needs the same button, such as an export replacing
  a comparison, keeps it); a comparison that was replaced says so in its status line instead of
  "Calculating…". The study guard still refuses while a job really runs. Also fixed: the answer
  or failure of a job that ended just as another replaced it could be taken for the NEW job's,
  ending it and losing its result (about a third of replacements in a tight loop); a replaced
  job's worker is now kept until its queued signals have been delivered and dropped.
- After **Inspector ▸ File Info ▸ Sample rate / time unit…**, a hand-added event (Ctrl+Shift+M
  or the Measure tab's button), a span edge dragged or Reverted in the Event inspector, and the
  three merges (Merge all, Merge events in selection, the inspector's Merge with next) wrote
  rows on the NEW clock into the detection table the amber line had just called entirely the old
  one: on a 2 s, 50 kHz slice declared to be 40 kHz a 300-sample event got a dwell of 7.5 ms
  beside nine neighbours measured at 50 kHz, and Merge all left four old-clock and four
  new-clock rows under a line that still said every time in the table was computed at the old
  rate. Stored detections keep the rate they were measured at (never silently re-scaled), so
  these doors now refuse such a table before reading or writing anything, so no row, no stamp
  and no Undo step moves. The status bar says what did not happen (*Nothing was added.* /
  *Nothing was merged.* / *The event's bounds were not changed.*) followed by the amber line's
  own sentence, whose Re-run is the way forward; Analysis ▸ Compare baseline estimators and
  Re-measure depths say it the same way (*Nothing was re-measured.*) instead of opening a
  `ValueError` box. A table that is not stale (no declaration, the declaration taken back, a
  fresh run under the declared rate, or a table with no rate stamp) is edited exactly as before.
  One rule, `analysis.events.stale_rate_hz`, now decides this for the amber line, the editing
  doors and the re-measure kernel's own guard.
- **Analysis ▸ Export events…**, **Export events for training…** and **Export sub-state
  levels…** no longer write a table the amber line has marked stale. After a sample rate or time
  unit was declared, the table kept the rate it was detected at, yet the export went ahead: the
  CSV's times stayed on the old clock while the `.meta.json` took `fs` from the table (the old
  rate) and `t0_s` with its "add t₀ to find it on the trace" sentence from the new one. For 40
  kHz declared over a 50 kHz detection that sentence put the first event 2599 samples off, and a
  time-unit declaration moved `t0_s` (10000.0 → 10.0) under an unchanged CSV, with nothing in
  either file saying a declaration existed. The three exports now refuse such a table before
  asking for a file, write nothing, and say on the status bar to re-run detection under the
  declared rate, naming the rate the table was detected at and the one the recording reads now;
  after the re-run the export goes through and its `t0_s + t_start_s` lands on the trace sample.
  A table that is current, one with no rate stamp, and one whose recording has no rate any more
  export exactly as before.
- A pass made by Signal Conditioning ▸ **Compute exact** at one sample rate is no longer shown
  as the exact trace of another. After a rate was declared (Inspector ▸ File Info ▸ Sample rate
  / time unit…) and Signal Conditioning opened again, the cache of whole-recording passes found
  the pass made at the file's 50,000 Hz under the key of the chain at the declared 25,000 Hz and
  drew it under *Whole recording (exact)* with the button reading *Done: the preview shows this
  chain's whole-recording (exact) samples* — 26 pA from what the chain makes of the same samples
  at the declared rate, on a recording whose noise is 4.5 pA. The cache key now names the rate
  the pass was run at (the chain's signature never did: it uses the rate only to drop notch
  lines above Nyquist), so a pass belongs to one rate: after a declaration, or after taking it
  away, the window says *Window estimate … · press Compute exact for the true trace* with the
  button enabled until the pass has been made at the rate in force, and a pass made earlier at
  the other rate is found again when the declaration is taken away. Calibration edits still drop
  the pass, and the cache still holds the two most recent.
- In the Results table, ⌘A, ⌘R, ⌘F and every Ctrl / Option / Shift chord on a, r, f, j or k were
  read as the bare curation key: ⌘A accepted the selected row instead of selecting all of them,
  ⌘R rejected it and ⌘F flagged it (Edit ▸ Undo then read "Undo Curate event", the status bar
  said nothing), and Select All never selected anything. The same happened to a chord pressed on
  a control inside the table's panel, such as ⌘R on the focused Columns… button. Only an
  unmodified a / r / f / j / k curates now: ⌘A selects every row (the caption reads "N of N rows
  selected · A accepts / R rejects all N"), the other chords do nothing in the table, and ⇧⌘F
  and ⇧⌘R still reach their menu items.
- A sample rate or time unit declared under **Inspector ▸ File Info ▸ Sample rate / time
  unit…**, or taken away again with *Use the file's own time base*, now reaches an open **Signal
  Conditioning** window. It used to go on previewing at the old rate: after a CSV whose time
  column is in milliseconds was declared (50 Hz → 50 kHz) its Time tab spanned 0–2000 s against
  the trace's 0–2 s under "Exact for these stages · window 2000.00 s", its Spectrum tab stopped
  at the old Nyquist, a 10 kHz low-pass the form had accepted was refused as "at or above
  Nyquist (25 Hz) for fs=50 Hz", and a detection run afterwards left the window holding two
  rates at once (source 50,000 Hz, chain 25,000 Hz on a recording declared at 25 kHz). Only
  closing and reopening the window or re-selecting the channel healed it, and nothing said it
  was stale. The window now re-reads its source and its chain together whenever the recording it
  shows is re-declared (a segment follows its parent's declaration), drops a preview that was
  still coming back at the old rate, cancels a running *Compute exact*, and re-frames a Time
  plot you had panned; a stage that no longer fits (a notch list above the new Nyquist, a
  baseline-drift window longer than the declared recording) is refused in the new rate's words.
  The source's identity now includes its sample rate, so a *Compute exact* pass computed at the
  old rate is no longer served as the exact trace of the new one when the window is reopened (26
  pA off a 4.5 pA noise sigma, labelled "Whole recording (exact)").
- The **Navigator strip** stayed on the old clock after **Inspector ▸ File Info ▸ Sample rate /
  time unit…** declared a recording's time base (or *Use the file's own time base* took the
  declaration away): the trace re-framed, the strip kept its curve, extent and shaded-window
  bounds. Declaring a CSV's millisecond column left it reading **Local context · 9× view** over
  a 2 s recording, drawing none of its 100,000 samples, and letting a drag of the shaded window
  to 13–15 s take the trace there too, past the last sample, so the pane went blank; a
  sample-rate-only declaration left the curve filling half the strip, or the window unable to
  reach the end of a recording it had made longer; a refused column declared in microseconds
  kept the strip in samples; Zoom-to-full repaired none of it, only re-selecting the recording.
  The strip now re-reads the recording whenever a drawn recording's time base is declared or
  removed — curve, extent, window bounds, the nine-view lens, its label and the Measure band —
  whichever of it and the trace hears the change first; a declaration on a recording that is not
  drawn leaves it alone.
- The Results dock's **Fits** tab now shows a fit only beside the table it was made on. The
  result store keeps one dwell fit and one depth fit per recording and channel, so a recording's
  Schmitt, CUSUM and Reconciled tables shared one slot and the tab printed whatever it held:
  Schmitt's `n = 316` under CUSUM's 370 candidates, the same fit beside a merged table (246
  accepted), a re-run (44 events), a table with one event rejected and the table an Undo
  restored — while Export figure refused that very fit as "saved fit is stale" — and a Fit whose
  dwell half failed left one table's dwell fit beside another's depth fit. Each fit now records
  the table it was made on (`detection_module_id`, a new key in the fit's saved metadata; no
  project format change of its own, and a version 11 build read it as before, though it refuses the
  version 12 project this build now writes). The tab prints a fit only
  when that table is the one on screen, with the calibration and accepted events it was fitted
  to (the figure export's two checks plus the table). Otherwise it says which table the fit
  belongs to, or that the table has changed since (curated, merged, re-run or undone), or that
  the calibration differs, and to press **Fit**. A Fit press replaces the pair: a half that
  fails drops its predecessor. A fit that records no table — saved by an earlier build, or filed
  by the Plots window — stays in the project and is still used by Export figure exactly as
  before, but is not printed beside any table until you press **Fit**. The accepted count still
  stands in for "the same events" (as it does for the export), so swapping one accepted event
  for another leaves a fit standing.
- Switching the Results dock's **Table** selector (or the Detection tab's detector) now takes
  the open **Event inspector**, docked or in its own window, and the **Event gallery** with it.
  They used to keep the table the dock had just stopped showing, so one R rejected the event in
  that table **and**, by event number, a different event of the table on screen: two tables
  written, one Edit ▸ Undo step that took back only the second (the first kept a "reviewed"
  rejection that travelled into the saved project and every export, and the menu read "Undo —
  nothing on this table; the threshold detector (1 step) on this channel"), and a click on a
  Results row showed the other table's event under the same number. Each surface now shows the
  table on screen from its first event (the inspector back to All, the gallery on its first
  tile, the trace and the Results selection left where they were), one keypress writes one table
  as one undo step, and a surface that is ever out of step with the dock keeps its decision on
  its own table, says so in the status bar and re-syncs instead of mirroring it onto the dock's
  table by event number. A CUSUM sub-state fit stays with the table it was measured on rather
  than being drawn under another table's events.
- A CSV time column in **milliseconds** loaded silently at a rate 1,000× too low: 10 µs steps
  written as `0.01` per row read as 100 Hz, `Provenance.MEASURED`, duration 552 s for 0.5 s of
  samples, with an empty Issues box — the only guard was the 1 Hz floor that catches microsecond
  columns, and no sample-rate override existed anywhere to correct it. An undeclared time column
  (a CSV's, or an HDF5 `/time`) whose measured rate is below 1 kHz now raises an Issue naming the
  step, the rate as read, what the rate is if the time base is in milliseconds or microseconds,
  and how to settle it (for a CSV, a unit in the column header; for either, the File Info
  declaration); the measured
  rate is kept, nothing is converted by a guess, and a header-declared unit never raises it. The
  user guide's Issues row claimed every non-seconds column was refused; it now says what is
  caught, what is asked, and what is read as declared.
- A saved window size and place from a differently sized display (a laptop and an external
  monitor more than about a fifth apart in width) is no longer reported as "a value this app did
  not write": the main window, the child windows and Signal Conditioning take their saved size
  and place, brought onto the current screen (a maximized main window stays maximized), and a
  window's remembered place is no longer deleted. Only bytes that really are damaged are named.
  Signal Conditioning is now listed in the **Window** menu. **Window ▸ Reset Window Layout…**
  also puts a window that is hidden back at its default size and place.
- A dock that is floating when you quit (Results, Inspector or File manager) now comes back
  floating and visible; quitting used to close it before the layout was saved. **Window ▸ Reset
  Window Layout…** now also forgets the Signal Conditioning window's size, place and splitter,
  and no longer lets a window you opened and hid earlier write its old place back at quit.
  A hand-added event (Ctrl+Shift+M) over an event the detection already has is refused with a
  status-bar line instead of counting the event twice, and a hand-add (or Merge all, or their
  undo) with the Event gallery open no longer jumps the main trace to the gallery's first tile.
- Reopening a project with ML detections no longer shows a false "open-pore estimator
  changed" banner when the stored windows differ from the defaults. The ML branch of the
  stored-detection seeding restored only the ML tab and left the baseline form on the app
  default, so the banner appeared and its "Re-measure depths" button would have re-measured with
  the defaults instead of the windows the table was measured with. ML results saved without
  stored parameters (pre-v3 projects) are unchanged. The baseline form is shared by every
  detector on a channel, so a table of another detector measured with different windows still
  reads as stale until it is selected.
- The ML preprocessing median (`_decimated_median`) is a rolling order statistic instead of
  ndimage's O(n*w) rank filter: about 40-50x faster at the 2501-tap Host-guest / Zn windows
  (6.2 s to 0.16 s at 3M samples; 31 s to 0.6 s at 1.5M samples after decimation). Values are
  identical on 240 synthetic cases and the 35 real-recording corpus tests (the sign of a zero
  can differ when -0.0 and 0.0 are mixed). Non-finite input and windows longer than the signal
  keep the previous path. The pinned md5 of `ml_preprocess.py` was re-pinned with this
  verification; the median's arithmetic now also depends on
  `acid_incubation_numerics.median_filter_1d`, which has its own tests.
- Voltage dependence and its menu gate read tables from any primary detector (Schmitt, CUSUM,
  ML): the one stored on the most recordings that can become a point is pooled for the whole
  series, recorded in saved workspaces, and named in the Pooled provenance block. An ML-only
  project previously had the action greyed out. As first made, this fix sent a tie to Schmitt
  and named the detector only when it was not Schmitt; a tie now goes to the latest pass and the
  block always names the detector (see *Batch detect runs the detector selected in the Detection
  panel* under Changed).
- Event inspector: opens on the event selected in the table (and, re-opened, where you were)
  instead of event 0; keyboard focus starts on the waveform instead of the "Event N" spin box,
  so A/R/T and label digits reach their shortcuts; the first event no longer draws against a
  0..1 pA placeholder range when the stale-measurement note arrives after the first draw.
- Analysis > Event labels... fits a laptop screen: its explanation line wraps instead of
  forcing a minimum width wider than the screen (1777 px at the app's 13 pt font).
- The trace view gives the current channel more height than the command-voltage pane (5:2
  instead of equal rows).
- "Known open-pore current" accepted 0 pA (and anything that underflows to 0 in the
  float32 pipeline that reads it, e.g. 1e-300) with no error, then silently produced a
  NaN fractional depth on every event and a NaN Summary median with no explanation.
  Refused at the editor, at the precision the analysis pipeline actually reads it under.
- Three numeric boxes (open-pore current, notch lines, acquisition bandwidth cutoff)
  each guarded only against PEP 515 underscore grouping (`1_20` -> 120.0); none guarded
  against the non-ASCII digit scripts bare `float()` also accepts (`١٢٠` -> 120.0). A
  shared strict-ASCII parser now backs all three.
- The per-event local baseline-intrusion advisory could report "assessed" with a
  near-zero fraction on exactly the two canonical Automatic-mode baseline failures
  (isolated long events; high-duty recordings), because it compared a fragment's two
  flanks to each other rather than to a wider, already-computed reference. Gated on the
  recording-level baseline-health verdict baseline.py already produces; advisory only,
  no detector output changes.
- The HDF5 reader read t0 from a `/time` column it had already declared unusable for a
  sample rate (all-NaN or reversed), producing a NaN or backwards origin with no issue
  naming the problem. Returns the documented "unavailable" sentinel with an explicit
  issue instead.
- `recharacterize_span` accepted a negative sample rate (a wrong-signed row, no error)
  and raised the wrong exception type for zero, infinite, or `None`; now raises
  `ValueError` naming `fs`, matching `detect_events`'s existing convention.
- Opening Analysis > I-V on a large, never-before-medianed recording blocked the GUI
  thread for the whole whole-channel median (~300 ms observed) with no busy state. The
  first gather now runs on a background thread once a channel exceeds ~500,000 samples;
  the raw arrays it needs are resolved on the GUI thread first and handed to the worker
  as plain data, never a live `Session` reference, after an earlier version of this fix
  was found and rejected in review for exactly that hazard.
- A project reset/switch deleted the I-V window directly, bypassing the retirement step
  every other background-worker-holding window goes through at teardown; a gather still
  in flight was left running with nothing left to deliver its result to.
- Practice-recording duplicate detection compared paths with `==` instead of the one
  function that answers "is this file already open" -- the literal pattern an
  architectural guard exists to forbid -- and carried a latent case-sensitivity defect
  on a case-insensitive volume as a result.
- Two dialogs (Help > About; the open-pore reference preview) set `WA_DeleteOnClose`,
  the one lifetime pattern `nanoscope/gui` otherwise avoids everywhere: the app's
  teardown sweep calls `close()` on every owned window as a stop mechanism, which is a
  hide for every other window and would have been a delete under a live Python wrapper
  for these two.
- Several architectural AST guards had gone stale as legitimate shared infrastructure
  (plot-appearance/workflow-bar/label modules, a lasso-selection primitive, a
  publication-export dialog) was adopted by multiple analysis windows without the
  guards' exact-import-set assertions being updated to match. Each newly-allowed module
  was individually verified -- no widget of its own, no further reach beyond the same
  shared, non-panel set -- before being added, rather than the assertions being loosened
  wholesale.

### Validation

- 2026-09-13: all six items confirmed still open from the 2026-09-12 beta sweep (a
  zero open-pore current; non-ASCII digit acceptance in three boxes; the baseline-
  intrusion advisory's false negative; the HDF5 t0 defect; `recharacterize_span`'s
  fs validation; the I-V first-open stall) are fixed and independently reviewed;
  root-cause fixes throughout, not input-specific patches. Nine other items from that
  sweep, including the acquisition-metadata save failure, were found already resolved
  by intervening work and are not claimed here. 358+ targeted tests re-verified across
  measurement quality, the reader contract, characterize/span/flag-invariant guards
  and I-V; 455 of 457 tests across the sibling-import guard files (the two remaining
  failures are a pre-existing, unrelated geometry pin).
- 2026-09-13: root cause found for the intermittent PySide6 signal-manager crash noted
  below -- once a Qt wrapper's C++ side is destroyed, its Python object becomes
  permanently invisible to Python's own cyclic garbage collector. Confirmed on PySide6
  6.11.1 and 6.11.2 both, and confirmed independent of this suite's `gc.disable()`
  policy (a separate, unrelated deadlock avoidance; re-enabling it does not free these).
  Measured: ~734 leaked `QObject`s and ~3.6 MB RSS per `MainWindow` construct/destroy
  cycle, linear with no plateau across 40 cycles. This is a PySide6 characteristic, not
  fixed here; the manual full-regression CI job now runs the `gui` shard as 16 smaller
  processes instead of one, bounding rather than eliminating the accumulation a single
  long process would otherwise reach.
- Comparison-figure source pass: 571 focused tests passed across rendering, page interaction,
  workspace persistence, Library capture, native vector/export composition and scientific
  regressions. All 37 recorded analysis-module hashes are unchanged. Read-only apo-ferritin
  QA used all 2,104 saved accepted events for the three new views, checked PNG/PDF/SVG exports,
  and left original event tables and project bytes unchanged. This source pass preceded the
  packaged comparison release below.
- 2026-09-06 comparison packaging: rebuilt the Apple Silicon app and replaced both the local
  and SanDisk DMGs. All 16 applicable in-bundle checks passed, including Seaborn facets, pair
  matrices, strip/swarm views, source-free restore and twelve PNG/PDF/SVG exports; three optional
  real-recording/model checks were skipped because no inputs were supplied. All 27 packaging
  regressions passed. Strict ad-hoc signature verification and both DMG integrity checks passed;
  the images have matching SHA-256 `3ffe9e901c7032169496b3303b4dc27ebec56e3eb8c0c9c70fad52b79c9a48d6`.
  All 125 embedded NanoScope modules plus the packaged self-test match current source bytecode,
  apart from normalized filenames. The installed `/Applications` copy was not replaced in this pass.
- 2026-09-06 packaging: rebuilt the macOS application and replaced the local and SanDisk
  DMGs. All 15 applicable packaged smoke checks passed; three optional real-recording/model
  checks were skipped because no inputs were supplied. Strict ad-hoc signature verification
  and both DMG integrity checks passed, and the DMGs have matching SHA-256 hashes. All 122
  embedded NanoScope modules match the current source code apart from rewritten filenames.
  The installed `/Applications` copy was left unchanged.
- Publication-hardening pass: 88 composition/revision checks passed, including real Population
  linear/log millisecond axes, real PSD replay, original snapshot/resource preservation, model
  version guards and project-switch cancellation. Additional targeted runs passed 108 export/
  capture checks, 74 existing axis/selection checks and 79 Library/Explorer/workspace checks
  (these groups overlap). Native and physical-size shared-panel exports and revision layout
  were visually verified. No scientific kernels or analysis defaults changed in this pass.
- Next-phase verification covered numerical references and unit-change invariants for all four
  new summaries, exact frozen restore/export, invalid-input refusal, composition resources and
  shared-axis safety, real lasso gestures and event-table routing, and unchanged automatic zoom.
  The existing baseline, detection, characterization, amplitude and workspace regression group
  passed all 394 checks; Explorer/workspace integration passed 101. Offscreen visual QA covered
  all four summary types and the multi-panel composer. These are targeted runs, not a full suite.
- Core plotting verification: 182 targeted checks passed across frozen previews/exports,
  appearance invariance, Figure Library persistence/provenance, all-page capture and staged-write
  safety. Broader plotting, numerical, workspace and responsive-layout suites also passed in
  targeted runs. Visual QA covered publication previews and smaller Explorer windows.
- Long combined GUI runs can intermittently hit the existing PySide6 6.11.1 signal-manager
  crash seen before this work; this is not claimed as a clean full-suite run.
- 870 distinct focused amplitude, detector, characterization, persistence, curation and
  plotting regression checks passed; offscreen visual checks verified editable-level and
  selected-interval layouts. These source checks preceded the packaged build described above.
- 8,114 tests passed and 8 corpus/platform tests skipped in the complete Stage 3 regression.

## 0.1.0 Beta 1 — 2026-09-03

### Added

- Interactive ABF, HDF5, TDMS, CSV, and text-trace viewing and calibration.
- Schmitt, CUSUM, ML, and manual event detection with provenance-preserving reconciliation.
- Event inspection, batch curation, structured labels, and supervised-learning exports.
- Histograms, density and scatter plots, fitting, PCA, clustering, PSD, I–V, and voltage-dependence
  analysis.
- Project persistence with guarded atomic saves and external-change conflict detection.
- Crash-recovery autosaves after 15 seconds idle, bounded by a 60-second maximum, with two atomic
  generations and Recover, Discard, and Show in Finder startup actions.
- Adaptive local-context trace navigation for efficient manual annotation.

### Validation

- 8,060 tests passed and 6 corpus/platform tests skipped in the complete Stage 2 regression.
- The packaged macOS application passed all 18 bundle self-tests, strict ad-hoc signature
  verification, and DMG checksum verification.

### Distribution limits

- The beta build targets Apple Silicon and macOS 15 or later.
- The application is ad-hoc signed but not Apple-notarized; Gatekeeper may require the documented
  first-launch workaround.
- Raw recordings are referenced by path and are not embedded in projects or recovery snapshots.
