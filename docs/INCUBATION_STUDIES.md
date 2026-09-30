# Incubation studies

Edition 2026-09-29, NanoScope build **40.16.59**, with the persistent study library.
The shared incubation and curation workflow was introduced in build **40.15.59**.
The final sections preserve the original Zn reference method and its scientific
definitions; historical counts in the recording-specific guides are not a new
detector benchmark. See the [Zn guide](MANUAL_ZN_WORKFLOW.md),
[Cu guide](MANUAL_CU_WORKFLOW.md), and [illustrated manual](manual.html).

## One entry for a new or saved study

Open **Analysis → Incubation study…** and select your study in **Saved studies**.
Choose **Open selected study**, or double-click its row. The list survives closing
and reopening NanoScope. **Show study folder** opens the selected folder in the
system file browser. A missing folder remains listed with an explanation.

The standard library folder is **Documents/NanoScope/Incubation Studies**.
Your existing Zn, Cu and Acid studies can appear here through named shortcuts;
the original data stay in their existing folders. Library names do not rename
the saved study or change its scientific settings. Newly created or successfully
opened studies are remembered automatically. Only workspaces directly inside the
library folder are discovered automatically; the app does not search unrelated
folders or reports.

Use **Open saved study…** to browse for another workspace. The picker starts at
the last available study folder. **Create study…** starts in the library folder;
create and select a new empty subfolder for each study.

For a new study, select a profile and enter the **Study name** and optional
**Sample** before choosing **Create study…**. Existing study folders contain `workspace.json`.
The saved schema selects its original method; a new-study profile does not change
the saved preparation, detector, measurements or review decisions. **Continue
selected study** returns to an open study. Other open studies and their drafts
remain available when another study is opened.

| Profile | Starting method |
|---|---|
| General incubation | Editable stages, reflected median-window edges initially; suitable for future sample and condition labels |
| Zn incubation | The same general stages, with reflected median-window edges |
| Cu incubation | The same general stages, with shrinking median-window edges |
| Acid incubation | Calibrated Acid preparation, CUSUM, measurement and family-analysis definitions |

General/Zn/Cu share one workflow and do not require ML. Signal detection and
compatible main-workspace detections can be used without a model. Select a trained
model only when its training and preprocessing contract is appropriate. Profiles
do not infer chemistry, channels, polarity, excluded intervals or physical metadata
from a filename. Older markerless Zn/Cu workspaces may display General while retaining
their saved scientific settings. The Acid profile remains a distinct calibrated
method, not a replacement numerical recipe for arbitrary samples.

## Prepare and detect explicitly

For General/Zn/Cu, the stages are **1 Sources**, **2 Prepare**, **3 Detect / review**,
**4 Compare**, **5 Features / populations** and **6 Figures**.

1. Add source recordings. In **1 Sources**, enter the study name, sample and
   condition label and choose **Apply study details**. Enter the actual recording
   time, physical pore ID, nominal diameter, concentration, voltage, current channel
   and orientation, channel combination and retained/excluded intervals; choose
   **Apply recording details**. Defaults are not experimental measurements.
2. Set the recording's preparation in **2 Prepare** and choose **Prepare selected
   recording**. Inspect raw, filtered/baseline and prepared traces before detection.
3. In **3 Detect / review**, choose a suitable signal or optional model workflow,
   or use the main app's detection tools on a compatible full recording. Inspect
   candidates and measurement definitions before review.

Acid has separate **3 Detect** and **4 Review**, then **5 Analyze** and **6 Figures**.
Apply its preparation/detection settings, prepare each source, and choose
**Calibrate and detect selected recording**. Inspect calibration methods and the
candidate table before accepting results. Its source form allows unknown nominal
diameter to remain blank; General/Zn/Cu require a positive value, so do not present
their initial placeholder as a measured diameter.

## Save main-workspace curation back to the study

1. Select the intended study recording. Choose **Curate in main workspace…** from
   **3 Detect / review** (General/Zn/Cu) or **4 Review** (Acid).
2. Check the recording named by the main window's **Incubation study** bar. The
   prepared trace and candidates are available in the ordinary event table,
   Event Gallery and Inspector. **Show study events** returns to that source if
   another main recording is currently selected.
3. Accept/reject events, edit bounds, add missed events, merge, annotate and use
   undo/redo with the main tools. Opening the same current review again retains
   pending edits. Review the trace rather than assuming every automatic candidate
   or group is a molecular event.
4. Choose **Save curation to study** and wait for the completion message. The study
   commits the event selection and uses its own saved measurement settings. Edited
   event geometry is remeasured under those definitions; main prepared-trace
   amplitudes and calibrated study amplitudes need not be identical.
5. Choose **Return to study**. This changes the view; it does not save edits,
   approve review or recompute dependent results.
6. Confirm the selection for every included recording: **Use reviewed events for
   analysis** in General/Zn/Cu, or **Confirm event selection for analysis** in Acid.
   Confirmation approves the current selected population, not a claim that every
   row was individually inspected. Explicit human review remains distinct from
   automatic acceptance and rejection.

Main **Save Project** and **Save curation to study** have different destinations.
Save Project preserves the main workspace, while the curation action updates the
incubation checkpoint. Later edits need another study save and confirmation. Keep
the complete study folder with its arrays, outputs, manifest and prepared review
trace; a saved main project may reference that trace. Undo history does not replace
a saved checkpoint or scientific audit record.

General/Zn/Cu also offer **Use existing main-workspace detections** for a compatible original
recording/channel with full-recording Schmitt, CUSUM, ML or Reconciled detections.
This imports a snapshot with acceptance and source provenance. Later main edits
need a further explicit save/import. Mismatched identity, stale inputs and partial
or masked exposure are refused; read the reason and correct the source or settings.

## Recalculate and review populations

After changing accepted events or geometry, confirm again and run the needed
dependent stages. For General/Zn/Cu, choose **Compute recording comparisons**;
for waveform populations, **Extract reviewed-event waveform features** before
**Run pooled exploratory clustering** or **Fit selected-recording mixture**.
Then use **Generate selected figure group**. Existing saved output may remain
viewable; it is not proof that a canceled or failed new calculation completed.

Cluster review edits the same accepted selection as individual review. Inspect
events before **Accept selected clusters**, **Reject selected clusters** or **Keep
only selected clusters**. Keep-only also rejects other events in the displayed
saved recording domain, including unassigned events unless selected. Use **Use
cluster-reviewed events for analysis**, confirm other pending recordings, and
recompute the dependent analyses. A fresh fit may change the meaning of a cluster
number; group labels are not molecular identities.

In Acid, confirm each primary recording, then optionally **Build event families
for review**. Inspect family events, make group decisions, and confirm the revised
selection. Use **Analyze confirmed event selections**, followed by **Generate study
figures**. Acceptance-only review retains frozen family assignments; changes to
bounds, additions, deletions and merges invalidate affected family analysis and
require rebuilding. The calibrated detector, amplitude, polarity and full exposure
definitions remain those of the Acid method.

**Style selected figure…** changes appearance without detection, review confirmation,
feature extraction or fitting. Save styling to the study, regenerate figures after
numerical changes, and inspect labels and colors again after a refit. Save main
Figure Library entries with the main project separately.

## When a step is unavailable

- Check the selected study recording and the main curation bar, then use **Show
  study events** if the main workspace has another recording active.
- Wait for a running operation to stop. A cancellation request may need a native
  calculation to return before it can complete.
- Commit intended settings before opening or saving a review. If the study changed
  after events were opened, return to it and open its current events.
- Save curation before confirming. Confirm all included recordings before a new
  comparison or population calculation. Recompute stale results before new figures.
- If a window was closed, return to the study and reopen its review as needed.
  Failed or canceled saves keep the previous completed checkpoint; main edits remain
  separate until a successful study save.

## Advanced: original Zn reference recipe

The remainder of this page documents **Advanced: original Zn reference recipe…**
in the study launcher. This preserves the fixed original student workflow and
its saved result bundles for reproducibility.

The automated **Run study** action below always runs the frozen student recipe.
It does not consume acceptance decisions or edited boundaries from the main
workspace. After curation, run comparisons, features, population fitting, and
figures as separate steps in the manual workflow.

### Set up and run the reference recipe

1. Choose the folder containing the prepared recordings. Each recording needs a
   CSV (optionally gzip compressed) with `time_s`, `current_raw_pA` and
   `current_detr_pA`, and its `*_metadata.json` file. The sample rate, polarity,
   sample count and duration in the metadata must describe that CSV.
2. Review each recording's incubation time, physical pore identity, pore diameter,
   concentration and voltage. Choose the trained student model explicitly.
3. Select the recording whose quantitative subpopulations should be fitted. For
   the Zn study this is `zn_30h`, the recording on which the student was trained.
4. Choose a new output directory and save the recipe if you want to rerun it later.
5. Run the study. Progress identifies the recording and analysis stage. Cancel
   requests a cooperative stop; completion of a native library call may be needed
   before cancellation finishes.

The study reads the prepared current traces directly. It does not apply another
filter or baseline subtraction. Preserve the metadata from the preprocessing
that produced the CSV files. This matches the inputs to the inference notebook.
The reader accepts the preprocessing notebook's float32 timestamp export and
records its precision in provenance. Event times use the sample index divided by
the metadata sample rate, avoiding drift from rounded CSV timestamps.

### Review and reopen reference results

The study window displays the summary, event table, figures and methods report.
Use the figure selector to browse comparisons, galleries and population plots;
drag to pan and use the wheel or buttons to zoom. Open a saved study directory to
review its exported results without rerunning the model or mixture fitting.

A saved recipe identifies the input files and settings. A completed study bundle
contains the result tables, figures and recorded source/model identities. Keep
the bundle together when copying it to another location. Rerunning the recipe
requires its source recordings and model; reviewing the saved output does not.

### Reference measurement definitions

The versioned student recipe reproduces the incubation section shared by
`Zn2Insulin_StudentInference.ipynb` and its CUSUM variant. The calibrated CUSUM
section inserted into the latter is a separate analysis.

All student candidates are retained in the per-recording catalogs, with the
sustained-depth acceptance decision recorded separately. Incubation comparisons
use the recorded amplitude definition: sustained depth with local open-pore
normalization and the notebook's separate global-reference comparisons are not
interchangeable. Tables and methods retain their definitions.

The current notebook's exploratory clustering uses twelve features. NanoScope's
expanded quantitative subpopulation analysis uses thirteen, including entropy.
Their outputs are identified separately; their cluster labels need not agree.
Event families are descriptive groups, not molecular identifications.

The population table separates the fraction of events assigned to each component
from its fitted mixture weight. Bootstrap intervals describe the mixture weights;
they are not uncertainty intervals for the hard assigned event counts or rates.

The four Zn incubation conditions were measured on different physical pores.
Fractional blockade partly adjusts for current scale, but does not remove this
experimental difference. Rate intervals describe uncertainty within a recording,
not independent experimental replication.

For reference compatibility, recipe v1 retains the notebook's inference-window
coverage: a short tail may have no model predictions. The actual uncovered sample
count is saved in each recording's provenance. A corrected coverage policy would
be a different recipe and must be compared explicitly.

### Headless reference reproduction

The desktop window and command line use the same study engine:

```sh
python -m nanoscope.incubation_cli --manifest study.json --output new-result
```

The command emits progress as JSON lines and returns a nonzero exit status on a
failed or cancelled run. It requires NanoScope's analysis dependencies and the
TensorFlow runtime used by its ML detector. Existing study output is not silently
replaced.
