# Reproduce the Zn-insulin study manually in NanoScope

This guide starts with the four supplied ABF recordings and the trained `Zn2student_zn_30h.keras` model. You choose the files, enter each recording's settings, inspect intermediate results, and run preparation, inference, event selection, comparisons, clustering, and figure generation separately. No Jupyter execution or saved study recipe is required.

The supplied model is an input to this inference study. Training a replacement teacher or student is a separate experiment. Likewise, the ABF filenames identify acquisition exports already marked “Process” or “Lab view corrected”; this guide does not recreate undocumented processing before those files were exported. It follows `Zn2Insulin_DataPreprocessing.ipynb` and the original `Zn2Insulin_StudentInference.ipynb`, not the additional calibrated CUSUM workflow.

The optional [automated Zn reference recipe](INCUBATION_STUDIES.md) remains available through **Advanced: original Zn reference recipe…** for subsequent runs from prepared CSVs. Use the manual workspace below when learning the methods, inspecting each decision, or changing parameters. It also connects to the main NanoScope event table, gallery, inspector, and analysis tools. [Section 5a](#5a-curate-with-the-main-analysis-tools-before-comparing-recordings) explains how to edit detections there before the study, or use existing Schmitt, CUSUM, ML, or reconciled detections instead of running the supplied student.

## One incubation entry for current and future studies

Open **Analysis → Incubation study…**. For this reference choose **Zn incubation**, enter **Study name** and optionally **Sample**, then click **Create study…** and select a new, empty folder. That action creates the workspace and opens its stages; no second New workspace operation is needed. To restore a completed study, use **Open saved study…** and select its whole workspace folder. To resume one already open, select it in the open-study list and click **Continue selected study**. Opening another study preserves the other open studies and their drafts. Use one editing session per folder.

General, Zn and Cu share **1 Sources**, **2 Prepare**, **3 Detect / review**, **4 Compare**, **5 Features / populations** and **6 Figures**, with curation in the main app's Event table, Gallery and Inspector. The Zn profile starts with reflected median edges and Cu starts with shrinking edges. A profile supplies starting settings; enter every reference parameter below rather than assuming the profile has set them all. Opening an existing study restores its scientific method and saved settings without applying new-profile defaults.

For future samples or conditions choose **General incubation**, give the study an explicit name and sample, and enter the actual recording metadata. On **1 Sources**, edit **Study name**, **Sample / material** and **Condition / treatment label**, then **Apply study details**. A trained model is optional: prepared data can use signal detection or the main workspace's compatible detectors. Changing the sample does not validate the Zn or Cu scientific parameters for it. The Acid profile uses the same launcher and main curation tools while preserving its calibrated CUSUM and family-analysis method.

This is an analysis protocol for acquired recordings. The supplied files and metadata do not specify a complete laboratory incubation recipe: temperature, pH, metal concentration, preparation and agitation must come from the experimental record. Do not infer them from this interface or substitute the recording concentration for an undocumented incubation concentration.

## 1. Collect the inputs and create a workspace

Use an updated NanoScope build that has **Analysis → Incubation study…**. Quit an older running NanoScope instance before opening the updated app.

On the original study computer, inputs are in `/Users/abrahamcampos/Desktop/Zn2IonCondition/`; the model is inside its `CSVData` folder. On another computer, keep the same files in a folder of your choice and select their local paths.

| Recording tag | Exact ABF filename |
|---|---|
| `zn_2h` | `100 pM_Zn_2 h_Insulin_16 nm_50 percent 35k PEG_0000_-200 mV_Conductive_Good_0-4 min_Lab view corrrected.abf` |
| `zn_10h` | `100 pM_Zn_10 h_Insulin_18 nm_50 percent 35k PEG__0021_-200 mV_Conductive_ Very Good_Process_Lab view corrected.abf` |
| `zn_18h` | `100 pM_Zn_18 h_Insulin_18 nm_50 percent 35k PEG__0005_-200 mV_Conductive_Very good_Process_Lab view corrected_118 s.abf` |
| `zn_30h` | `100 pM_Zn_30 h_Insulin_20 nm_50 percent 35k PEG_0008_-200 mV_Conductive_Good_Process_225s.abf` |

1. Choose **Analysis → Incubation study…**, select **Zn incubation**, and enter **Study name** and optionally **Sample**.
2. Click **Create study…** and select a new, empty folder, for example `Zn-insulin-manual`. The study is created and opens immediately.
3. Click **Add raw ABF…** and select the four recordings.
4. Use the recording selector at the top to select one file. The **1 Sources** page shows its full path, channels, sample rate, and sample count. Check the path before assigning its tag.

Completed operations save automatically to this workspace. **Save workspace** also preserves current settings. The right side provides **Trace / event**, **Results table**, **Saved figure**, and **Methods / history** views. Drag the central divider to give the controls or plots more room, and scroll inside the left panel to reach all settings. Wait for an operation to complete before starting another. **Cancel operation** requests a cooperative stop; a running library calculation may need to finish before cancellation takes effect.

## 2. Enter the recording identities

On **1 Sources**, enter the following values for each selected recording, then click **Apply recording details**. A physical pore ID is a label for the actual pore, not just its nominal diameter. The tag itself is a suitable unique ID for this study because each recording used a different pore.

| Field | 2 h | 10 h | 18 h | 30 h |
|---|---:|---:|---:|---:|
| Recording tag | `zn_2h` | `zn_10h` | `zn_18h` | `zn_30h` |
| Incubation (h) | 2 | 10 | 18 | 30 |
| Physical pore ID | `zn_2h` | `zn_10h` | `zn_18h` | `zn_30h` |
| Pore diameter (nm) | 16 | 18 | 18 | 20 |
| Concentration (pM) | 100 | 100 | 100 | 100 |
| Voltage (mV) | -200 | -200 | -200 | -200 |
| Current channel (1-based) | 1 | 1 | 1 | 1 |

For all four, select **Event orientation: Downward in source** and **ABF numeric scaling: Notebook float32 convention**. Channel 1 is `IN11`, in pA; channel 2 is `IN14`, in mV. All recordings have a 50,000 Hz sample rate. Keep the complete recordings:

| Tag | Samples | Duration, seconds |
|---|---:|---:|
| `zn_2h` | 11,943,488 | 238.86976 |
| `zn_10h` | 10,219,519 | 204.39038 |
| `zn_18h` | 5,923,839 | 118.47678 |
| `zn_30h` | 11,269,119 | 225.38238 |

**Notebook float32 convention** reproduces the original reader's numerical ADC arithmetic. It does not multiply current by an experimental gain factor. Do not apply an additional factor of ten or infer an inversion from the negative voltage. The prepared raw current remains in its native sign for later open-pore measurements.

Complete steps 3-5 for each recording before proceeding to cross-recording analysis. Selecting another recording preserves the completed stages and that recording's own settings.

## 3. Prepare and inspect the selected raw recording

Open **2 Prepare** and enter these common settings:

| Control | Value |
|---|---|
| Apply sequential notch | Checked |
| Notch frequencies (Hz, comma separated) | `60,120,180,240,300,360,420,540,660` |
| Notch Q | `30` |
| Round notch output to float32 | Unchecked |
| Baseline method | Event-masked median |
| Mask: rough window (s) | `0.5` |
| Mask: baseline window (s) | `0.2` |
| Median-grid decimation (samples) | `10` |
| Despike kernel (samples; 0 disables) | `0` |

Set these two controls individually:

| Tag | Mask threshold (MAD sigma) | Mask dilation (ms) |
|---|---:|---:|
| `zn_2h` | 3.0 | 0.2 |
| `zn_10h` | 3.5 | 2.0 |
| `zn_18h` | 3.0 | 2.0 |
| `zn_30h` | 3.0 | 2.0 |

The **Robust median / SavGol window (s)** and **SavGol polynomial order** fields belong to other baseline methods; they do not replace the two mask-window values above. The reference preparation uses no additional low-pass, high-pass, moving average, Savitzky-Golay baseline, or despiking.

Click **Prepare selected recording**. Each notch frequency receives its own zero-phase filter pass. The baseline then masks downward excursions, interpolates across those gaps, estimates a slow median reference, and subtracts it. The output remains in pA. Model normalization occurs later.

Use the preview selector and **Show full recording** to inspect:

- **Raw current:** the original calibrated current, including its open-pore level.
- **Filtered current and baseline:** the notched current and the actual estimated baseline. Check that the baseline follows slow drift rather than diving into brief events.
- **Prepared current:** the baseline-subtracted signal used for inference. Zoom into short events as well as quiet regions. The overview preserves extrema; selecting an event later shows its original samples.

**Methods / history** records the parameters, source identity, and preparation quality checks. These checks describe noise, residual baseline, drift, and upward/downward excursions; they are not tuning targets that should be forced to match an expected count.

Optional: click **Export prepared CSV and metadata…** and select a destination folder. NanoScope writes `<tag>.csv.gz` and `<tag>_metadata.json`, retaining time, raw current, prepared current, baseline, and provenance. Existing outputs are refused rather than silently replaced. This export is useful for the automated shortcut, but the manual workspace can continue directly without reopening a CSV.

The 10 h historical preparation has a documented numerical exception; see section 10. Use the settings above and retain the newly calculated result.

To detect and curate with the main NanoScope tools instead of the supplied student, continue directly to [section 5a](#5a-curate-with-the-main-analysis-tools-before-comparing-recordings). Preparation is enough to open the recording there; model inference and candidate measurement are not prerequisites.

## 4. Run the student and inspect its probabilities

Open **3 Detect / review**, click **Choose trained model…**, and select `Zn2student_zn_30h.keras`. Use this same model for all four recordings and verify its path after switching recordings. It was trained on the 30 h recording; the other incubation times test transfer to other recordings.

| Control | Value |
|---|---:|
| Robust normalization: clip and divide (sigma) | 30 |
| Model window (samples) | 2048 |
| Window stride (samples) | 1024 |
| Inference batch size | 64 |
| Window coverage | Exclude exact last start (reference) |

Click **Run model inference**. This produces a saved probability for each covered sample; it does not yet select events. The normalization estimates a quiet median and MAD twice, clips at ±30 noise units, and divides by 30. It does not overwrite the prepared pA signal.

To inspect the output, return to **2 Prepare**, choose **Model probability**, and click **Show full recording**. Compare suspicious time intervals with **Prepared current**. A high probability identifies a potential event; amplitude acceptance is a separate test below.

The reference window policy deliberately leaves a short uncovered tail at zero probability. **Methods / history** records its sample count and coverage boundary. These samples are unmeasured by the model, not demonstrated event-free. The other coverage choices are available for a deliberate alternative analysis; keep the reference choice for this reproduction. A recording shorter than the model window receives no padded inference window.

## 5. Find, measure, and review events

Return to **3 Detect / review**. Enter:

| Candidate control | Value |
|---|---:|
| Enter probability | 0.4 |
| Exit probability | 0.15 |
| Probability median kernel (samples) | 3 |
| Merge gap (ms) | 0.05 |
| Minimum event duration (ms) | 0.1 |

Click **Find candidate events**. This reads the saved probabilities; it does not rerun the model. A contiguous region at or above the exit threshold becomes a candidate if it reaches the enter threshold. Nearby regions merge before the minimum-duration test. No onset snapping or offset trimming is applied in this workflow.

The minimum duration rounds up to a whole number of samples. For example, at 10 kHz, a 0.29 ms minimum requires three samples (0.30 ms). The reference 50 kHz / 0.1 ms setting is exactly five samples. **Methods / history** records the effective sample threshold; older saved candidate stages retain their original rounding provenance until you rerun **Find candidate events**.

Then enter:

| Measurement control | Value |
|---|---:|
| Sustained-depth acceptance (quiet MAD sigma) | 4 |
| Quiet raw I0 block (s) | 0.1 |
| I0 median smoothing window (s) | 5 |

Click **Measure and gate candidates**, then **Show all candidates**. Every fired candidate remains in the table. The automatic `passes_gate` column is true when its median depth is at least four times the recording's event-excluded noise MAD. The `accepted` column initially copies that decision and is the selection used downstream.

Click a row to inspect its full-resolution waveform. The shaded region marks the measured event, and the dot marks its actual minimum sample. `start_idx` and `end_idx` are both inclusive. Duration is therefore `(end_idx - start_idx + 1) / sample_rate`; the exported `end_s` is the exclusive end time.

For the reference reproduction, inspect without changing acceptance. For a separately documented curation, use **All candidates**, **Accepted only**, or **Rejected only** to find events, select rows and click **Accept selected** or **Reject selected**. **Reset review to automatic gate** restores the detector's acceptance choices. Curation leaves `passes_gate` unchanged, so a hand decision remains distinguishable from the automatic gate. The workspace records those actions and clears dependent comparisons, fits, and current figure selections.

When the selection is ready, click **Use reviewed events for analysis**. Do this for each of the four recordings before comparisons, feature extraction, or population fitting, including when retaining automatic acceptance for the reference reproduction. This saves your approval of the current event selection without changing its decisions. It does not claim that every event was individually inspected. Acceptance, boundary, preparation, or measurement changes require confirmation again before a new downstream analysis can run.

Click **Export candidate and accepted tables…** to save both `<tag>_candidates.csv` and `<tag>_accepted.csv`. The complete provenance remains in the workspace. The table can be wider than the display; scroll horizontally for the measurement and acceptance columns.

After repeating preparation, inference, decoding, and measurement for all four files, these are the expected checks with automatic acceptance:

| Tag | Fresh raw candidates | Accepted events | Accepted rate, per second |
|---|---:|---:|---:|
| `zn_2h` | 408 | 108 | 0.452129 |
| `zn_10h` | 3,943 | 110 | 0.538186 |
| `zn_18h` | 1,647 | 194 | 1.637452 |
| `zn_30h` | 11,004 | 1,076 | 4.774109 |
| Total | 17,002 | 1,488 | Do not average the per-recording rates as independent replicates |

The historical prepared-CSV route has 3,945 candidates at 10 h and 17,004 overall. The two-candidate difference is explained in section 10; all accepted intervals remain the same in the independently checked 10 h comparison.

## 5a. Curate with the main analysis tools before comparing recordings

Complete your review before **4 Compare**, feature extraction, and population fitting. The main workspace provides the existing event table, Event Gallery, Event Inspector, waveform and distribution tools, labels, and Undo. The study uses the event boundaries and acceptance decisions you explicitly bring back from that workspace. The counts above describe the unchanged reference analysis; curated or alternative detections can give different counts and fitted populations.

### Send a prepared study recording to the main workspace

1. Select the recording at the top of the manual study window and finish **Prepare selected recording**. You can proceed immediately, without a model or candidates. If you already measured study detections, those events will accompany the recording.
2. On **3 Detect / review**, click **Curate in main workspace…**. NanoScope opens a saved copy of that recording's prepared trace. Check the active recording before continuing.
3. To create detections there, choose Schmitt, CUSUM, or ML and run it over the **full recording**. If the copied study events are already the detections you want to review, select that event table and keep them.
4. Inspect events through the main **Event table**, **Event gallery**, and **Event Inspector**. Accept or reject detections, add labels, adjust boundaries, merge split events, or mark a missed event using the existing controls. Use **Edit → Undo** to reverse a main-workspace edit. [The user guide](USER_GUIDE.md#9b-undoing-a-change-to-the-event-table) describes the curation controls and their restrictions.
5. Leave the intended recording and event table active in the main workspace. Choose **Save curation to study** in the main study bar, then **Return to study**. The bar keeps the exact study recording attached to its events.
6. Click **Show all candidates** and inspect several edited events. Check the `accepted` decisions and event boundaries, then click **Use reviewed events for analysis**. Repeat for the other recordings before continuing at **4 Compare** below.

The review copy contains the prepared current, so its near-zero baseline is not the original raw open-pore current. On import, NanoScope recalculates incubation depths, dwell times, and fractional blockades from the study's prepared and raw arrays. Original main-workspace columns, including measurements and labels, remain available with a `source_` prefix in the study candidate table. An amplitude calculated with another main-workspace estimator remains a source measurement; it does not silently replace the incubation amplitude definition.

The prepared review trace is an HDF5 file inside the study workspace. Save a normal NanoScope project to retain the main-workspace analysis, and keep the whole study workspace available with it: that project needs its referenced HDF5 recording. Importing detections does not edit the original ABF or the main-workspace result.

### Exact main-window event review controls

After **Curate in main workspace…**, keep the matching recording and detection table active. The persistent **Incubation study** bar identifies the exact study recording. Choose **Analysis → Event inspector…**. Use **◀ Prev** and **Next ▶** to navigate, then **Accept** or **Reject** for the current event; these actions do not advance automatically. The **Accepted** checkbox reflects its state. The A/R shortcuts perform the same decisions, and the arrow keys navigate.

For a boundary correction, drag the event mask edge in Event Inspector and wait for the edit and measurement to finish. **Revert mask**, when available, restores the recorded original bounds. The highlighted interval in the study's own trace preview is for inspection and cannot be dragged. The main Event table, Gallery and Inspector also provide their ordinary labeling, missed-event, merge and Undo tools; Reconciled tables retain their source-editing restrictions.

Click **Save curation to study** in the main study bar and wait for successful completion, then **Return to study**. Inspect the imported bounds, measurements and decisions, then click **Use reviewed events for analysis**. Saving a main project preserves that workspace but does not commit event changes to the study. Later edits require **Save curation to study** again. Reentering compatible curation retains unsaved main-workspace edits; if upstream study data or settings changed, return to the study and open its current events because an outdated copy is refused.

### Use detections you already curated in NanoScope

You can use this route without choosing a student model or running model inference in the study.

1. In the main workspace, open the original ABF, select its intended current channel, and run your chosen detector over the **full recording**. Schmitt, CUSUM, ML, and compatible Reconciled tables are supported. Curate the result with the usual tools.
2. In **Analysis → Incubation study…**, add that same ABF, apply its recording identity and channel, and complete **2 Prepare**. The study still needs its raw and prepared currents for consistent incubation measurements.
3. Make the curated recording and the intended **Event table** active in the main workspace. In the manual study, select the matching recording, open **3 Detect / review**, set the measurement controls, and click **Use existing main-workspace detections**. This imports and measures the selected table without running the student.
4. Review **Show all candidates** and click **Use reviewed events for analysis**. Repeat for each included recording before continuing at **4 Compare**.

A Reconciled table can be imported after review. Its existing restrictions still apply: edit boundaries, merge, or add missed events in its source detector tables, then reconcile again before importing the new result.

### Keep curation and saved analyses in sync

**Save curation to study** and the advanced **Use existing main-workspace detections** import each take a saved snapshot. Later edits in the main workspace do not change the study automatically; click it again to bring those changes into the selected study recording. Each import replaces that recording's study event selection and clears dependent comparisons, features, clustering, mixtures, and current figures. Inspect the imported result, click **Use reviewed events for analysis** again, then recompute those stages after the last import.

The imported `accepted` column controls comparisons and waveform features. The separate `passes_gate` column reports what the study's sustained-depth gate would decide; a disagreement does not overwrite your imported decision. **Reset review to automatic gate** deliberately replaces review decisions with that gate, so use it only when you intend to discard the current selection. Re-running the study's model or candidate detector starts a new detection path; import the curated table again if that is the path you intend to analyze.

NanoScope refuses a different recording or channel, an outdated result, or a partial or masked analysis range. Full-recording coverage is required because incubation rates use the full recording duration. Select a current compatible table or rerun detection over the complete source before importing; changing a recording tag does not make another source compatible.

Stay in the manual workflow for the remaining analyses. **Run study** in the automated incubation window runs its frozen student recipe and does not use these curated snapshots. That window's **Curate detections / manual workflow…** button opens this manual workflow.

## 6. Compare recordings and test the common floor

Once every included recording is measured and confirmed with **Use reviewed events for analysis**, open **4 Compare**. All included recordings participate, with their individual durations and pore identities.

| Control | Value |
|---|---:|
| Time blocks | 20 |
| Bootstrap draws | 4000 |
| Random seed | 0 |
| Fractional floor: start | 0.02 |
| Fractional floor: stop | 0.20 |
| Fractional floor: step | 0.01 |
| Sensitivity block counts | `10,20,40` |

Click **Compute recording comparisons**. Rate is accepted events divided by the full analyzed duration. The 95% interval resamples counts from contiguous time blocks, retaining evidence of within-recording clustering. **Time blocks** controls that partition; **Sensitivity block counts** repeats the calculation with the listed alternatives.

Use the comparison-table selector and **Show comparison table**:

| Selection | What to check |
|---|---|
| Summary | Candidate and accepted counts, durations, rates and intervals, medians, pore IDs, and uncovered tails |
| Accepted events | Pooled selected events with source identities and explicitly named amplitude columns |
| `rate_block_sensitivity` | How intervals and Fano factors change across 10, 20, and 40 blocks |
| `gate_transfer_sweep` | Number and rate remaining above each fractional floor, for each amplitude definition |
| `gate_transfer_pairs` | Pairwise rate ratios; both the requested native common floor and the evaluated grid floor |
| `distribution_distances` | KS distribution distances, or an explicit insufficient-data status |

Three fractional amplitudes are used; select interpretations by column, not just the word “amplitude”:

| Quantity | Table column | Meaning |
|---|---|---|
| Sustained / local I0 | `fracblock` | Quiet-reference minus event median, divided by the absolute time-local raw open-pore current |
| Peak / global I0 | `aggregation_fracblock_global` | Quiet-reference minus event minimum, divided by the absolute whole-recording float32 raw median |
| Sustained / global I0 | `f_sust_global` | The sustained depth divided by that global raw median |

Local I0 comes from quiet raw-current block medians, smoothed over the selected window and interpolated at event onsets. It is not the near-zero prepared baseline. Peak depth emphasizes a brief minimum; sustained depth emphasizes the event body.

The sweep applies an additional floor to the already accepted events. Its native gate reference uses preparation QC noise divided by global I0; the table also records the detector's distinct event-excluded noise. Pair ratios evaluate the nearest sweep-grid value to each pair's larger native floor. KS comparisons use the exact largest native floor among recordings with at least 50 accepted events, and require at least 20 above-floor events in each arm. The common-floor KDE uses the largest native floor and the sustained/global definition. These are explicit reference-method choices, not interchangeable thresholds.

The 10 h and 18 h recordings share a nominal 18 nm diameter but use different physical pores. Fractional normalization only partly adjusts current scale. The intervals quantify uncertainty within recordings; they do not create independent pore replicates or prove incubation caused the differences.

## 7. Extract features and run pooled exploratory clustering

After confirming the event selection for all four recordings, open **5 Features / populations** and click **Extract reviewed-event waveform features**. This measures the accepted waveforms without rerunning inference. **Show waveform feature table** displays the saved result. Expect 1,488 rows with unchanged reference acceptance.

For **Pooled clustering features (comma separated)**, use these 12 columns in this order:

```text
log_dwell,depth_max_pA,depth_sust_pA,mean_pA,std_pA,min_dev_pA,skewness,kurtosis,area_pA_ms,t_peak_frac,f_sust,snr
```

| Control | Value |
|---|---:|
| Clustering seed | 0 |
| K-means components | 3 |
| PCA retained variance target | 0.95 |
| Maximum retained PCs | 10 |
| HDBSCAN minimum samples | 10 |
| Compute descriptive t-SNE | Checked |

Click **Run pooled exploratory clustering**. The result includes PCA coordinates, HDBSCAN labels, k-means labels, and descriptive t-SNE coordinates. HDBSCAN's noise label is `-1`; it does not mean those accepted events vanished.

Use **Show saved clustering assignments** to display these results again after saving or reopening, without recalculating them.

This feature method intentionally differs from the candidate catalog: it recomputes waveform features against the whole prepared-trace median/MAD and a global float64 raw-current median. It clips feature tails at the 1st and 99th percentiles, centers/scales by median/IQR, and maps remaining non-finite scaled values to zero. HDBSCAN's minimum cluster size is `max(20, number_of_rows // 100)`. These method choices, loadings, variance, and transformation values are saved with the analysis.

Manual feature extraction retains the events-down orientation established during preparation, including when the original source events point upward. If an older workspace contains upward-event features calculated before this correction, opening it explains which feature and population stages must be rerun. The original raw data, preparation, probabilities, candidates, and measurements remain available; the old disk checkpoint is unchanged until you save. The downward-event Zn reference does not require this migration.

PCA compresses correlated features; the variance target is limited by **Maximum retained PCs**. K-means partitions every point into the chosen number of groups. HDBSCAN can leave points unclustered. t-SNE is a qualitative view of the retained PCs; distance or apparent separation in that plot is not an independent statistical test or molecular identification.

## 8. Fit the separate 30 h quantitative mixture

Remain on **5 Features / populations**. Select **`zn_30h`** in the mixture-target selector. For **Mixture features (comma separated)** use the same 12 features plus `entropy` at its reference position:

```text
log_dwell,depth_max_pA,depth_sust_pA,mean_pA,std_pA,min_dev_pA,skewness,kurtosis,area_pA_ms,t_peak_frac,f_sust,snr,entropy
```

| Control | Value |
|---|---|
| PCA dimension grid (blank = automatic) | `2,3,4,5,6,7,8,9,10,11,12,13` |
| Mixture K grid | `1,2,3,4,5,6,7,8` |
| Cross-validation folds | 5 |
| Mixture bootstrap replicates | 200 |
| Mixture selection / fit seed | 0 |
| Mixture bootstrap seed | 0 |

Click **Fit selected-recording mixture**. This searches the specified PCA/component grid by held-out likelihood, fits the selected model, and bootstraps its mixture weights. The 30 h fit uses its 1,076 accepted events. Pooled 12-feature clustering and this 13-feature fit are separate analyses; their labels need not agree.

To inspect saved results later, choose a **Saved mixture table** and click **Show saved mixture table**. Available tables include event assignments, population quantification, the cross-validation grid, mixture-weight bootstrap results, the BIC ladder, and the exploratory crosstab. This reads the saved results without fitting again.

The previously verified reference selected 11 PCs and four components, with assigned counts 387, 295, 269, and 125. Treat these as checks after fitting, not parameters to force. Component numbers can permute between equivalent fits. Inspect the selection grid, event membership, source geometry, and diagnostics rather than accepting a count match alone.

The population table distinguishes **assigned event fraction** (hard class count / 1,076), **assigned event rate** (hard class count / recording duration), and **mixture weight** (the fitted soft weight). Bootstrap intervals describe mixture weights, not uncertainty on the displayed hard counts. Preserve weak-separation, label-switching, and calibration warnings. Four fitted components are not evidence for four molecular species. The reference is weakly separated; its silhouette was approximately 0.11.

Time-half diagnostics use the recording's clock midpoint. Historical onset-median halves are retained only as a legacy comparison, because splitting at the median event onset would manufacture equal event counts.

## 8a. Review whole clusters and continue with the selected events

Individual and whole-cluster accept/reject actions record which events you explicitly reviewed, even when the acceptance value stays the same. This evidence returns to the main analysis workspace and its reviewed-event training exports. **Reset review to automatic gate** clears the current human review status while restoring automatic acceptance. **Use reviewed events for analysis** confirms the selection for analysis; it does not mark every event as individually inspected.

Use this optional branch to curate by waveform group after an initial clustering or mixture fit. Keep automatic acceptance unchanged to reproduce the reference counts and figures above. For an alternative analysis, first copy the saved workspace so that the original result remains available for comparison.

1. At the top of **5 Features / populations**, choose the review source: **Pooled clustering** for groups across the included recordings, or **Selected-recording mixture** for the fitted target recording only. Newly completed runs load their memberships automatically. For an older saved result, click **Load saved clusters for review**.
2. Choose the saved method: **K-means**, **HDBSCAN**, or **Mixture populations**, as available for that source. Check the recording domain and counts in the summary table. Cluster numbers belong to this saved method and run; a K-means number does not identify the same group as a mixture number.
3. Select one or several rows in the summary table. Use **Show events in selected clusters** to inspect their events against the source trace before deciding. **Unassigned / no waveform features** is an explicit bucket for events without a membership in this result. HDBSCAN **Noise (-1)** is a separate assigned label; it does not automatically reject those events.
4. Click **Accept selected clusters** or **Reject selected clusters** to change only those groups. **Keep only selected clusters** accepts the selected groups and rejects the other events within this snapshot's recording domain, including the unassigned bucket unless you select it. A selected-recording mixture decision affects only its target recording; recordings outside the review snapshot are unchanged.
5. Inspect the resulting counts and any individual events that need a different decision. Cluster actions and the individual controls on **3 Detect / review** edit the same `accepted` selection. The last individual or cluster action wins for each affected event. The original automatic `passes_gate` decision remains unchanged.
6. Click **Use cluster-reviewed events for analysis** to confirm the current event selection for the displayed recording domain. Confirm any other included recording separately on **3 Detect / review** if its review is still pending.
7. Recompute the analyses you need from this selection: **Compute recording comparisons** on **4 Compare**, and/or **Extract reviewed-event waveform features** on **5 Features / populations**, followed by the desired pooled clustering or selected-recording mixture fit. Then regenerate the relevant figure groups on **6 Figures**.

Acceptance edits invalidate the dependent analysis results but retain the saved cluster memberships for review, so you can select a rejected group and accept it again without fitting it first. Fresh measurement, detection, or a main-workspace import retires the old memberships because the event data may have changed. A new clustering or mixture run replaces that source's review memberships.

Refitting a selected subset can change both group membership and cluster numbering. Interpret a number only within its saved method and run; these groups are not established molecular identities. Saved figure styles persist and are reapplied when figures are regenerated. Review the series colors after a refit because the new labels may describe different groups.

## 9. Generate and inspect all 32 figures

Finish the scientific operations above before collecting the final figures. A newly completed upstream analysis clears dependent current figures; previous saved files remain in the workspace's history folders.

Open **6 Figures**, select a group, and click **Generate selected figure group**. Generate these groups:

| Figure group | Action | Expected figures |
|---|---|---:|
| Selected recording inspection (5 figures) | Select each of the four recordings at the top and generate once per recording | 20 |
| Recording comparisons | Generate once after comparisons | 7 |
| Pooled exploratory clustering | Generate once with descriptive t-SNE enabled | 2 |
| Selected-recording mixture | Generate once after the 30 h mixture | 3 |
| Total | All four groups completed | 32 |

The saved-figure selector displays each generated figure. Use **Fit figure to view** for an overview, then zoom to read dense galleries. Tall 50-event galleries are intended for zoomed inspection.

Each recording's five figures are trace/deepest-excursion context, an all-candidate summary, four typical accepted events, a 50-event peak-depth ladder, and a 50-event sustained-depth ladder. “Typical” uses proximity to median log-dwell/peak-depth coordinates with temporal separation. The ladders sample amplitude ranks; they are not random samples. Their local display recentering does not replace the saved scientific amplitudes. The all-candidate summary deliberately follows the notebook's separate full-trace-median peak-depth definition and rounding; it does not show only the accepted events.

The seven comparison figures are the three amplitude-definition distributions, the nominally 18 nm pair, rates/medians, gate-transfer sweep, and common-floor KDE. The pooled figures show PCA and t-SNE with exploratory labels. The mixture figures show model selection/weights/waveforms, representative events per component, and assignments on the trace/timeline.

For the last assignment figure, verify a few examples against their table rows: the upper dots sit at actual within-event current minima; the lower marks sit at event onsets; both use the same recording clock. The overview preserves short excursions with a minimum/maximum envelope. This is the corrected plotting convention, resolving the earlier misleading onset dots and strided overview.

Figure files and their numerical plotting sidecars are written inside the workspace's timestamped `runs` folders. Figure generation consumes completed results; it does not secretly rerun the student or refit the mixture.

Comparison captions describe the actual physical pore identities and whether the selected events come from reviewed acceptance or the automatic gate. If you change pore identities or acceptance decisions, recompute comparisons and generate their figures again.

### Style the saved figures

1. In **6 Figures**, choose a saved figure and click **Style selected figure…**.
2. Use the shared appearance controls for fonts, line widths and markers. In **Panel titles and axis labels**, select a panel and edit its title or labels. In **Linked series colors**, select a series and click **Choose series color…**; matching series, fills and legend keys stay together.
3. Click **Apply appearance** to preview, then **Save styling to study** to keep the styled figure and its settings in this workspace. Saved styles return on reopening and apply again when you regenerate that figure group. Styling preserves event-review approval and does not rerun detection, feature extraction, clustering, or mixture fitting.
4. Use **Export image…** for an individual image or **Save figure bundle…** for a portable figure with its data and provenance. For multi-panel composition in the main app, click **Save to Figure Library…**, then **Save Project** in the main window to persist that library entry. Compose the saved entries through the normal Figure Library tools.

## 10. Understand the historical 10 h exception

Fresh raw preparation matches every stored raw-current sample for all four recordings. It also matches every historical baseline/prepared sample for 2 h, 18 h, and 30 h. For 10 h, 230,078 of 10,219,519 baseline/prepared samples differ, by at most 0.333267 pA. Independent direct NumPy median checks agree with the stable calculation now used by NanoScope at all checked positions; the historical saved baseline disagrees at all 128 sampled disputed positions.

A fresh side-by-side downstream audit found:

- Raw reconstruction: **3,943 candidates / 110 accepted**; historical prepared CSV: **3,945 / 110**.
- All 110 accepted start/end samples and dwell times are identical.
- Twenty-eight rejected intervals shift; five historical rejected intervals disappear and three new rejected intervals appear. No shared-boundary gate decision flips.
- Accepted sustained depth changes by at most **0.064265 pA**, or **0.5345%** for the most affected event. Median change is **-0.000636 pA**. Local and global I0 remain identical; the largest sustained-fraction change is **0.0003973**.

The manual historical-CSV detector rerun matches the earlier saved study's candidate geometry and gate decisions exactly. The discrepancy therefore belongs to historical preparation, not to a different intended student threshold. Do not tune the new pipeline until it recreates the two extra rejected candidates. Keep the new raw results and document the numerical difference. The original prepared files remain useful for exact historical replay through the automated shortcut.

## 11. Save, reopen, and deliberately change the analysis

Click **Save workspace** before closing. Later choose **Analysis → Incubation study… → Open saved study…** and select the complete workspace folder. NanoScope restores saved prepared arrays, probabilities, candidate tables, acceptance decisions, comparisons, fitted results, figures, settings, and history without recalculating them. File hashes detect missing or changed saved data.

Custom gallery captions stay attached to the same event and trace when panels are rearranged. If curation replaces an event, or its bounds or plotted measurements change, the panel uses its automatic event caption until you style that event again. Fonts, colors, and general plot styling remain available. Older gallery captions saved without an event identity need to be entered again in the figure editor.

An older workspace may contain completed analyses without the newer event-review confirmation. You can inspect its saved tables and figures immediately. Before running new comparisons, features, clustering, or mixture fits, review each included recording and click **Use reviewed events for analysis**. Opening a workspace does not automatically approve its event choices.

Finder metadata and unrelated files added to the workspace do not become required scientific outputs. Previously saved Finder metadata is ignored when opening an older checkpoint, while recorded scientific outputs remain protected against missing or changed files. A malformed checkpoint is refused without replacing the workspace already open.

If another NanoScope session saves the same folder, this session refuses to overwrite that newer revision. Its current state and settings remain in the window; reopen the folder to load the latest saved revision before continuing. Keep one editing session per workspace. Applying recording details without changing any field preserves completed analyses and figures.

Keep `workspace.json`, `arrays`, and `runs` together. Copy the whole folder when sharing completed results. Original raw files and the model are external inputs: they are not required simply to reopen the completed checkpoint, but they are required to rerun dependent operations. On another computer, the most direct reproducibility test is to create a new workspace and select the copied ABFs and model using this guide.

Editable fields describe proposed settings until their operation completes. The pending-settings message identifies differences from the saved calculation. **Restore completed settings** returns the fields to the saved result's values. To run a deliberate alternative:

| Change | Operation to repeat first | Then repeat |
|---|---|---|
| Raw channel, orientation, reader arithmetic, or preparation | Prepare selected recording | Inference, candidates, measurement, review confirmation, downstream analyses |
| Model, normalization, stride, or coverage | Run model inference | Candidates, measurement, review confirmation, downstream analyses |
| Probability thresholds, smoothing, merge, or duration | Find candidate events | Measurement, review confirmation, downstream analyses |
| Sustained gate or I0 estimator | Measure and gate candidates | Review confirmation, comparisons, features/fits, affected figures |
| Acceptance decisions | Accept/Reject selected | Review confirmation, comparisons, features/fits, affected figures |
| Whole-cluster acceptance | Accept/Reject selected clusters or Keep only selected clusters | Use cluster-reviewed events for analysis, comparisons and/or feature extraction followed by desired fits, affected figures |
| Main-workspace boundaries, acceptance, or selected event table | Save curation to study (or the advanced Use existing main-workspace detections import) | Review confirmation, comparisons, features/fits, affected figures |
| Bootstrap or floor settings | Compute recording comparisons | Comparison figures |
| Pooled features or clustering settings | Run pooled exploratory clustering | Pooled figures |
| Mixture settings or target | Fit selected-recording mixture | Mixture figures |
| Figure appearance, labels, or series colors | Style selected figure… → Apply appearance | Save styling to study; no scientific stage needs rerunning |

Prior result files remain available in timestamped `runs` folders. Use a separately copied or newly created workspace for an alternative you want to compare directly with the reference. A manual workspace is the reproducibility record; an exported PNG alone is not.

## Validation status

The manual interface has been exercised through its real controls in an offscreen Qt session, beginning with the four supplied ABFs and trained model. Each scientific operation ran separately, without the fixed study runner or injected cached analysis stages. The completed reference workspace contains 1,488 accepted events and all 32 figures, with saved stages restored successfully on reopening. These were GUI button-driven checks, not a claim that a person operated the packaged app by mouse.

An independent scientific audit reopened that checkpoint, checked source/model and saved-file identities, and matched all 1,488 accepted event intervals to the earlier study. It verified the preparation arrays, comparison calculations, waveform features, clustering, mixture results, and all 1,076 assigned-trace markers. The historical 10 h preparation exception and its small measurement changes are quantified above.

A second fresh run completed the whole workflow in one uninterrupted process with normal cyclic garbage collection: 35 recorded actions, 33 completed stage operations, all 32 figures, and save/reopen in 5 minutes 14 seconds. All 20 saved scientific stage representations and 244 scientific array files matched the first completed workspace exactly; all 32 PNG files were byte-identical. Both runs retained the expected 387/295/269/125 mixture component counts. Runtime varies with the computer.

One earlier instrumented test run had a native crash during figure generation. Its checkpoint recovered successfully, and the subsequent uninterrupted run passed. The cause remains unestablished; these results do not establish that the crash has been fixed. Packaged-application and installation receipts are recorded separately from the development-source GUI checks above.
