# Manual Cu-insulin incubation workflow

Use **Analysis → Incubation study…** and choose **Cu incubation**. Create a new study and add the five ABFs listed in the delivered Cu report. Every stage is separately editable and saved; the fixed Zn recipe is not used. The Cu StudentInference notebook's current primary route is model-free Schmitt v4, so no trained model is needed for this route.

## One incubation entry for current and future studies

Open **Analysis → Incubation study…**. For this reference choose **Cu incubation**, enter **Study name** and optionally **Sample**, then click **Create study…** and select a new, empty folder. That action creates the workspace and opens its stages; no second New workspace operation is needed. To restore a completed study, use **Open saved study…** and select its whole workspace folder. To resume one already open, select it in the open-study list and click **Continue selected study**. Opening another study preserves the other open studies and their drafts. Use one editing session per folder.

General, Zn and Cu share **1 Sources**, **2 Prepare**, **3 Detect / review**, **4 Compare**, **5 Features / populations** and **6 Figures**, with curation in the main app's Event table, Gallery and Inspector. The Zn profile starts with reflected median edges and Cu starts with shrinking edges. A profile supplies starting settings; enter every reference parameter below rather than assuming the profile has set them all. Opening an existing study restores its scientific method and saved settings without applying new-profile defaults.

For future samples or conditions choose **General incubation**, give the study an explicit name and sample, and enter the actual recording metadata. On **1 Sources**, edit **Study name**, **Sample / material** and **Condition / treatment label**, then **Apply study details**. A trained model is optional: prepared data can use signal detection or the main workspace's compatible detectors. Changing the sample does not validate the Zn or Cu scientific parameters for it. The Acid profile uses the same launcher and main curation tools while preserving its calibrated CUSUM and family-analysis method.

This is an analysis protocol for acquired recordings. The supplied files and metadata do not specify a complete laboratory incubation recipe: temperature, pH, metal concentration, preparation and agitation must come from the experimental record. Do not infer them from this interface or substitute the recording concentration for an undocumented incubation concentration.

## Sources and preparation

Use channel 1, downward events, notebook float32 scaling, 100 pM and -200 mV. Set incubation times 2, 10, 18 and 30 h, with nominal pore labels 16, 16, 18 and 18 nm. Give the primary recordings separate pore identities.

- `cu_2h`, `cu_18h`: retain the entire supplied file.
- `cu_10h`: exclusions `0:20;370:380`, in source seconds. Both endpoints are excluded.
- `cu_30h`: use the new Desktop ABF ending `Baseline adjuested.abf`. Add channel 2 to channel 1. Enable **Additional channel values are pA (override header)**: this particular processed file's removed-current baseline is mislabeled mV. This is a deliberate interpretation of this file, not a general voltage-to-current conversion. Keep 300 to 900 s; the file ends at 899.84982 s.
- `cu_30h_excerpt`: use the old `10-15 min.abf` file. Retain it whole and enable **Supplementary recording (exclude from pooled series)**. It is a recording-state comparison, not a fifth incubation timepoint.

Choose **Shrinking (Cu convention)** median edges in Sources, then Apply recording details. In Prepare choose event-masked median, rough window 0.5 s, baseline window 0.5 s, decimation 10, mask 3 MAD sigma, dilation 5 ms, notch off and despike 0. Click Prepare selected recording.

The trace preview and native review copy use retained acquisition time. Exported events also contain `original_onset_s` and `original_end_s` mapping back to the ABF. Cu signal figure overviews use original source times. The 30-hour population-onset figure uses retained time: add 300 s to recover source time. Baseline calculation operates on concatenated retained samples, matching the notebook; detections cannot merge across a removed source interval.

## Detection, controls and curation

In Detect / review, use **Detect and measure signal events** with mean window 5 samples, enter 4 MAD sigma, minimum enter 8 pA, exit 1 sigma, merge 5 ms, minimum duration 0.2 ms, plateau duration 1 ms, plateau sustained depth 3 sigma, local raw-I0 rolling window 5 s and decimation 50. Depth is measured on the unsmoothed trace. Short spikes pass the automatic rule; longer regions need sustained depth of at least 3 sigma. `shallow_long` rows remain inspectable but start excluded.

Run signal null controls with 5 global-phase seeds, 3 block-phase seeds and 10 s blocks. Thresholds remain pinned to the real trace. Null counts are diagnostic controls, not a measured false-discovery rate.

For the reference reproduction keep automatic acceptance. For independent curation, click **Show all candidates**, use **All candidates**, **Accepted only**, or **Rejected only** to find events, inspect their full-resolution waveforms, then **Accept selected** or **Reject selected**. These decisions control downstream event selection; `passes_gate` retains the automatic rule's decision.

To use the main app's curation tools, click **Curate in main workspace…**, curate in its event table, Gallery and Inspector, then choose **Save curation to study** and **Return to study** in the main study bar. This captures acceptance and edited bounds while recalculating the Cu measurements with the same Schmitt measurement policy. Later native edits require **Save curation to study** again.

The main study bar identifies the exact recording. Wait for the save to complete successfully before returning; **Return to study** alone does not save edits. Reentering compatible curation preserves unsaved main-workspace edits. **Save Project** preserves the main workspace separately.

When ready, click **Use reviewed events for analysis** for each recording you intend to analyze. Confirm all four primary recordings before comparisons, feature extraction, or population fitting; review the supplementary excerpt separately as well. Confirmation is required even when retaining automatic acceptance for the reference reproduction. It saves approval of the current selection without changing any acceptance choices or claiming every row was individually inspected. After acceptance, boundary, preparation, or measurement changes, inspect and confirm again, then recompute dependent comparisons, features, populations and figures.

## Comparisons and populations

After confirming the primary recordings, compute recording comparisons: time blocks 20, bootstrap draws 4000, seed 0. For Schmitt data this generates event-type, conductance-ratio, drift and monotonicity tables; the Zn single-depth-floor gate-transfer tables do not apply. Supplementary sources stay outside the pooled series.

Extract reviewed-event waveform features. Use these 12 columns in order:

`log_dwell,depth_max_pA,depth_sust_pA,mean_pA,std_pA,min_dev_pA,skewness,kurtosis,area_pA_ms,t_peak_frac,f_sust,snr`

Run pooled exploratory clustering with K-means K=3, PCA target 0.95, maximum PCs 10, HDBSCAN minimum samples 10, seed 0 and descriptive t-SNE enabled. K-means partitions the observations even when density clustering finds no clusters. Do not infer three biological species from that partition.

For the Cu 30-hour quantitative appendix, select `cu_30h`, the same 12 features, PCA grid `2,3`, K grid `1,2,3,4`, 5 folds, 200 bootstrap fits, seeds 0 and minimum accepted events 244. The sample floor follows the notebook's 5 training events per free parameter at the largest 3-PC, 4-component full-covariance model. The fresh fit reaches the search-grid boundary and has weak separation. Its components are exploratory descriptions, not established molecular populations.

## Review whole clusters before continuing

Individual and whole-cluster accept/reject actions record which events you explicitly reviewed, even when the acceptance value stays the same. This evidence returns to the main analysis workspace and its reviewed-event training exports. **Reset review to automatic gate** clears the current human review status while restoring automatic acceptance. **Use reviewed events for analysis** confirms the selection for analysis; it does not mark every event as individually inspected.

This is an optional curation branch after the first clustering or mixture fit. Retain automatic acceptance for the reference reproduction; copy the saved workspace before developing a different selection.

1. At the top of **5 Features / populations**, choose **Pooled clustering** or **Selected-recording mixture** as the review source. New runs load their memberships automatically. For an older saved result, click **Load saved clusters for review**.
2. Choose **K-means**, **HDBSCAN**, or **Mixture populations**, as available. Check the summary table's recording domain and event counts. Pooled review covers its snapshot of primary recordings; the supplementary `cu_30h_excerpt` stays outside that domain. Mixture review affects only its fitted target, such as `cu_30h`.
3. Select one or several cluster rows and click **Show events in selected clusters** to inspect source traces. The **Unassigned / no waveform features** bucket contains events without membership in this result. HDBSCAN **Noise (-1)** is distinct: those events received the density method's noise label and remain accepted until you change their selection.
4. Use **Accept selected clusters** or **Reject selected clusters** for those groups. **Keep only selected clusters** accepts the selected groups and rejects all other events within the review snapshot's recording domain, including unassigned events unless that bucket is selected. It leaves recordings outside that domain untouched.
5. Refine individual events on **3 Detect / review** if needed. Individual and cluster actions edit the same `accepted` decisions; the last action wins for each affected event. Neither changes the original automatic `passes_gate` decision.
6. Click **Use cluster-reviewed events for analysis** to confirm the current selection for the displayed recording domain. Confirm any other included recording separately if its review is pending. Then run **Compute recording comparisons** and/or **Extract reviewed-event waveform features**, followed by the desired clustering or mixture fit, and regenerate the affected figure groups.

Acceptance edits clear dependent results while retaining the saved memberships for review, allowing you to accept a previously rejected group again. New measurements, detections, or a main-workspace import retire old memberships. A new clustering or mixture run replaces that source's review memberships. A fit on fewer accepted events must still satisfy its selected minimum-event requirement.

Cluster numbers are local to the saved method and run. Refitting a subset can change the assignments and their numbering; they are not molecular identities. Saved figure styles persist through curation and regeneration. Check the colors after refitting because a new cluster number may describe a different group.

## Figures and saved workspace

In Figures, generate the selected-recording group once for each of the five recordings (5 each). Generate comparisons (5), pooled clustering (2) and the quantitative appendix (3): 35 figures total. Generate all groups after the final curation and analysis stages. PNGs and their numerical sidecars are in the workspace's `runs` directories. Save, close and reopen the workspace to inspect every committed stage.

To style a plot:

1. Choose it in the saved-figure selector and click **Style selected figure…**.
2. Adjust the shared appearance controls. Use **Panel titles and axis labels** for individual panels and **Linked series colors → Choose series color…** to recolor matching series, fills and legend keys together.
3. Click **Apply appearance**, inspect the preview, then **Save styling to study**. These settings persist on reopening and are reapplied when regenerating that figure group. Styling preserves event-review approval and does not rerun detection, features, clustering, or mixture fitting.
4. Use **Export image…** or **Save figure bundle…** for an individual figure. For publication composition, click **Save to Figure Library…**, then **Save Project** in the main window to retain the entry. Use the main Figure Library's composition tools for the saved panels.

Custom gallery captions stay attached to the same event and trace when panels are rearranged. If curation replaces an event, or its bounds or plotted measurements change, the panel uses its automatic event caption until you style that event again. Fonts, colors, and general plot styling remain available. Older gallery captions saved without an event identity need to be entered again in the figure editor.

An older completed workspace remains viewable without an event-review confirmation. Before starting new downstream analyses, inspect the event selection and click **Use reviewed events for analysis** for each included recording. Reopening alone does not approve its saved acceptance decisions.

The delivered report includes all 35 figures, exact source filenames, measured outcomes and verification evidence. It separates fresh results from the older `CSVData_mm3` cache: raw values match, but that cache's prepared baselines differ from the current notebook functions. Accepted counts match, although a few boundaries and measurements differ. The fresh NanoScope preparation and detections were checked exactly against independent execution of the current notebook definitions on the raw ABFs.
