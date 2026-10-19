# NanoScope — User Guide

A specialist walkthrough from opening a recording to exporting a curated event population.
Edition 2026-09-29, NanoScope build `40.16.59`, with the persistent study library
and the unified incubation and curation workflow introduced in build `40.15.59`.
Other main-analysis chapters retain their 2026-09-11 review against source `703a96a`,
unless a later date is stated. The illustrated manual, HTML, PDF and control reference
are maintained with this guide. Older illustrations and historical results have not
all been recaptured or rerun; they are not current benchmark or calibration guarantees.

For click-by-click plotting instructions, start with the [illustrated manual](manual.html):
**Analysis and plotting: navigation guide**, then **Your first plot: a worked histogram**.
Continue with **Current population plots: recipes and interpretation** for the individual
analysis recipes, and the analysis troubleshooting guidance when a control, source or result
does not match what you expect. The sections below provide the companion reference.

If you know the open-pore current or can identify open-pore intervals, see
[Recording-specific open-pore references](OPEN_PORE_REFERENCE.md). This optional
workflow accepts your evidence for one recording/channel, previews candidate events
without replacing the table, and requires deliberate fresh detection to apply it.
The Automatic workflow below remains available and unchanged.

For the 2026-09-12 source update, see [Measurement-quality and acquisition-bandwidth
advisories](MEASUREMENT_RELIABILITY.md): how to inspect baseline/polarity warnings in
the Results panel's Summary tab, enter acquisition metadata in File Info, and distinguish unassessed
hardware resolution from the existing software-filter flag. The detailed addendum
remains a separate companion document; this incubation-focused edition does not
claim a new audit of all measurement-advisory behavior.

> ## Before your first detection — establish the reference
>
> Every depth in this app is measured against an **open-pore baseline**, and the window that
> estimates it is a property of **your pore**, not of the app. Across four reference recordings on
> this bench the right window ranges from **0.03 s** (a gating pore) to **2.0 s** (a pore with a
> 521 ms partial clog) — a factor of 67. The app ships a default of **0.5 s / 0.1 s**, and on the
> gating pore that default is **3.3× worse than the estimator it replaced, with every health check
> reading `ok`.**
>
> For the Automatic estimator: open the **Preprocess** tab, press **"Measure this recording"**, and inspect the proposal before
> deciding whether it is appropriate. Only then press **"Use these values"** if you want to apply it,
> and inspect the updated overlay afterwards. The existing overlay before applying the proposal does
> not show that proposal's baseline. Measurement usually takes a few seconds — measured 2 s on a 3 M-sample recording
> and 6 s on 9 M samples — and the button reads "Measuring…" while it runs, so a two-second wait is the
> click landing, not the click failing. §5a explains what the two windows mean and how to read the
> proposal.
>
> Alternatively, use an independently known signed open-pore current or identified open-pore
> intervals through the recording-specific workflow above. These are explicit user assertions,
> not automatically verified truth. No health score alone establishes which plateau is open pore.

## Contents

1. [The workspace](#1-the-workspace)
2. [Loading recordings](#2-loading-recordings)
3. [Calibration](#3-calibration)
4. [Viewing the trace](#4-viewing-the-trace)
5. [Filtering](#5-filtering)
5a. [The open-pore baseline — **measure this recording**](#5a-the-open-pore-baseline--measure-this-recording)
6. [Event detection](#6-event-detection)
7. [The event table & characterization](#7-the-event-table--characterization)
8. [Manual measurement](#8-manual-measurement)
9. [The event inspector](#9-the-event-inspector)
9a. [When the open-pore baseline changes](#9a-when-the-open-pore-baseline-changes)
9b. [Undoing a change to the event table](#9b-undoing-a-change-to-the-event-table)
9c. [Reconciling detectors & building training labels](#9c-reconciling-detectors--building-training-labels)
10. [Population plots & machine learning](#10-population-plots--machine-learning)
11. [Noise spectra (PSD)](#11-noise-spectra-psd)
11a. [Reopening a project](#11a-reopening-a-project)
11b. [Recovering unsaved work after a crash](#11b-recovering-unsaved-work-after-a-crash)
11c. [Complete analytical projects — Stage 3](#11c-complete-analytical-projects--stage-3)
12. [Exporting](#12-exporting)
13. [Keyboard shortcuts](#13-keyboard-shortcuts)
14. [Known limitations & caveats](#14-known-limitations--caveats)

---

## 1. The workspace

The main window has four regions:

- **Left dock — File Manager.** A tree of open recordings grouped by applied voltage. Each recording
  expands into its channels; a checkbox next to each channel toggles it in the viewer, and a colour
  swatch sets its trace colour. Dead channels are greyed out.
- **Centre — the workflow strip** (`Open › Calibrate › Filter › Detect › Curate › Analyze`, showing
  where you are), then the **analysis launcher** row, then **the trace view** with a **navigator**
  strip beneath it (a full-extent overview with a draggable region box).
- **Right dock — Inspector**, a tabbed panel: **Calibration**, **File Info**, and **Analysis**.
- **Bottom dock — Results**, under the trace: a toolbar (the **Table** selector, **Columns…**,
  **Merge…**, the table's constant columns such as `recording = …`, and the advisories link), a
  one-line status (the event count, or the amber stale
  notice), then **Events** / **Summary** / **Fits** / **Quality**. Close it (its ✕,
  **View ▸ Panels ▸ Results**, or **⇧⌘R**) to give the trace its full height — nothing in it is
  lost; drag its top edge for more rows. It floats or docks on any side, and its place and height
  are remembered.

The **analysis launcher** starts with **Analysis home**, followed by shortcuts to
**Distributions**, **Event waveforms**, **Noise spectrum**, **I-V / conductance**,
**Voltage dependence**, **Report figures** and **Event gallery**. Full tool names match the
Analysis menu and Explorer pages. Short labels or glyphs appear when space is limited; hover
to see the full name. A shortcut you cannot use yet stays on the row, greyed;
hover it to read what the surface does and why it is not available, or click it to have the reason
posted in the status bar. Each button opens exactly what the matching **Analysis** menu item does.
The interactive surfaces share one **Analysis Explorer** instead of opening a pile of overlapping
windows. Its analytical pages are **Distributions**, **Distribution summaries**, **Comparison
figures**, **Event waveforms**, **Noise spectrum**, **I-V / conductance**, **Voltage dependence**
and **Subpopulations**. The header keeps the selected recording, result freshness
(`CURRENT`, `STALE`, or `SAVED SNAPSHOT`), and **Refresh current** visible while you move between
pages. Pages are created only when first used: a missing tool tab usually means that tool has not
been opened, not that the feature is absent. Use **Analysis home** to find it. Returning to a live
page can refresh its inputs or calculation; the opt-in summary/comparison pages only offer current
sources until **Generate**. A saved project snapshot remains intact until explicit refresh or
generation. The header names the active recording, but a multi-source plot uses the sources
checked inside its own **Data** section, not just that header recording.

### Incubation studies: shared entry and main-workspace curation

**Analysis → Incubation study…** is the entry point for all incubation studies.
Your **Saved studies** list remains available after restarting the app. Select a
study and choose **Open selected study**, or double-click its row. **Show study
folder** reveals its location. The standard library is
**Documents/NanoScope/Incubation Studies**, with named shortcuts for existing
studies so their data can stay in place. Successfully opened and newly created
studies are remembered. **Open saved study…** starts browsing at the last available
study directory; **Create study…** starts in the library folder.

Choose **General incubation**, **Zn incubation**, **Cu incubation**, or **Acid
incubation**, give the study a name and optional sample, and choose **Create study…**
with a new empty folder. Use **Open saved
study…** to restore an existing study with its saved method and settings, or
**Continue selected study** to return to an open study. Opening another study keeps
other open studies and their drafts available.

General, Zn and Cu use the same editable preparation, detection/review, comparison,
features/populations and figure stages. General supports future samples and conditions;
study name, sample and condition label are editable. Zn starts with reflected median
window edges; Cu starts with shrinking edges. Enter each recording's actual time,
pore diameter and identity, concentration, voltage, channels and retained intervals.
Profiles supply starting settings, not automatic identification of an experiment.
Saved workspaces retain their existing method and settings. Older markerless Zn/Cu
workspaces may display General while retaining their scientific settings. The app
does not infer the profile from filenames. Apply study metadata in **1 Sources**
with **Apply study details**; apply each recording's metadata separately. General/Zn/Cu
require a positive nominal diameter: replace placeholders with supported values;
the Acid source form allows an unresolved diameter to remain blank.
Use signal detection or the main workspace's detectors without a study model; a
trained model is optional. Acid retains its calibrated CUSUM and family-analysis
method within the same study entry point.

Choose **Curate in main workspace…** on **3 Detect / review** (General/Zn/Cu) or
**4 Review** (Acid). The prepared recording and its events open in the main event
table, Event Gallery and Inspector. Accept or reject events, edit boundaries, add
missed events, merge, annotate, and undo with the usual controls. The persistent
**Incubation study** bar names the exact recording. Choose **Save curation to study**
to commit edits using that study's measurement definitions; changed geometry is
remeasured. Then choose **Return to
study**. Returning is navigation, not another save or review confirmation.
Reentering the same current review retains unsaved main-workspace edits. Use
**Show study events** if the main window is displaying another recording. Confirm the event
selection before running dependent comparisons, populations, or figures. Saving
the main project preserves its workspace; later edits require saving to the study
again.

The General/Zn/Cu workspace also retains **Use existing main-workspace detections** for
importing a compatible full-recording Schmitt, CUSUM, ML or Reconciled result on
the original ABF and channel after preparation. The **Advanced: original Zn
reference recipe…** entry reproduces the fixed student notebook and opens its old
result bundles; its automated Run study action uses that recipe. See
[Incubation studies](INCUBATION_STUDIES.md) for this reference method.

Imported acceptance decisions drive the incubation comparisons and features.
The study recalculates its own measurements and retains the original table's
columns with a `source_` prefix. Each import is a snapshot: later main-workspace
edits require **Save curation to study** again (or another explicit import). On **3 Detect / review**,
use **All candidates**, **Accepted only**, or **Rejected only** to find events,
then **Accept selected** or **Reject selected**. When the selection is ready,
click **Use reviewed events for analysis** for each included recording before
comparisons, waveform features, clustering, or mixture fitting. This approves
the current selection without changing its decisions; it does not claim that
every row was individually inspected. Repeat confirmation after acceptance,
boundary, preparation, or measurement changes, then recompute dependent analyses
and figures. Older workspaces' saved results remain viewable immediately, but
new analysis requires this explicit confirmation.

For Acid, use **Confirm event selection for analysis** on **4 Review** for every
primary recording. If reviewing families, choose **Build event families for review**,
inspect and accept/reject the groups, then confirm the revised selections. Choose
**Analyze confirmed event selections** on **5 Analyze**, then **Generate study
figures** on **6 Figures**. Acceptance-only review retains frozen family assignments;
boundary edits, additions, deletions and merges require the study's remeasurement
and invalidate affected family analysis. Rebuild before interpreting a changed
population. Saving a main prepared-trace measurement does not replace the Acid
method's calibrated amplitude, polarity or exposure definitions.

To curate whole waveform groups after a first fit, use the cluster review controls
at the top of **5 Features / populations**:

1. Choose **Pooled clustering** or **Selected-recording mixture**. New runs load
   their memberships automatically; use **Load saved clusters for review** for an
   older saved result. Select **K-means**, **HDBSCAN**, or **Mixture populations**,
   as available, and check the recording domain in the summary table.
2. Select one or several cluster rows. **Show events in selected clusters** lets
   you inspect their events against the source trace. **Unassigned / no waveform
   features** contains events absent from the saved assignment map; HDBSCAN
   **Noise (-1)** is a distinct label, not an automatic rejection.
3. Use **Accept selected clusters** or **Reject selected clusters** to change the
   selected groups. **Keep only selected clusters** accepts them and rejects the
   other events in that snapshot's recording domain, including unassigned events
   unless their bucket is selected. Pooled actions affect only the snapshot's
   recordings; mixture actions affect only the fitted target recording.
4. Refine individual decisions if needed, then click **Use cluster-reviewed
   events for analysis** to confirm the displayed recording domain. Confirm any
   other included recording whose review is pending. Run **Compute recording
   comparisons** and/or **Extract reviewed-event waveform features**, followed
   by the desired clustering or mixture fit, then regenerate the affected figures.

Cluster and individual actions edit the same `accepted` selection: the last action
wins for each affected event, while the original automatic `passes_gate` remains
unchanged. Acceptance edits invalidate dependent results but retain the review
memberships, so rejected groups can be accepted again. Fresh measurement,
detection, or a main-workspace import retires those memberships; new clustering
or mixture results replace the corresponding review map. Cluster numbers belong
to one saved method and run, may change on a subset refit, and are not molecular
identities. Saved plot styles persist when regenerating figures; inspect colors
again after a refit because a new label may describe a different group.

Importing different sources, outdated results, or partial or masked analysis
ranges is refused. A prepared review trace is saved inside the study workspace;
keep that folder available when saving or reopening a main project that uses it.
Follow the [curation click path](MANUAL_ZN_WORKFLOW.md#5a-curate-with-the-main-analysis-tools-before-comparing-recordings)
or [the complete manual Zn-insulin guide](MANUAL_ZN_WORKFLOW.md) to reproduce the
four recordings using the supplied student model. The [manual Cu-insulin guide](MANUAL_CU_WORKFLOW.md)
uses the Cu notebook's model-free Schmitt route and its own measurement settings.

In **6 Figures**, select a saved figure and choose **Style selected figure…**.
The editor uses NanoScope's shared appearance controls, with **Panel titles and
axis labels** and **Linked series colors** for the study's multi-panel plots.
Click **Apply appearance** to preview, then **Save styling to study** to preserve
the figure and settings. Saved styles return on reopening and apply again when
regenerating the figure group. Styling keeps review approval and does not rerun
detection, waveform features, clustering, or mixture fits. **Export image…** and
**Save figure bundle…** export an individual figure. **Save to Figure Library…**
adds it to the main app for composition; use **Save Project** in the main window
to persist that library entry as part of the project.

### Analysis home: find a tool before opening it

Open **Analysis home** from the launcher, the **Analyze** workflow step, or the **Analysis** menu.
It remains the first Explorer tab, with a header button available from every analytical page.
You can open it without a recording or any detected events.

The catalog lists all eight analytical pages, **Event gallery**, **Report figures**, **Figure
Library**, and **Guided first analysis**, including tools you have never used. Search by tool,
plot type or input, such as `ECDF`, `conductance` or `events`. Selecting a name only shows its
purpose, required inputs and current availability. Choose **Open…**, double-click the list item,
or press Enter on it to request that tool. An unavailable tool explains its requirements and
does not open. Availability is checked again when you act.

**Browsing, searching and refreshing availability do not construct analytical pages or run
calculations.** **Refresh availability** on the home page is different from **Refresh current**
on an analytical page, which can update that page's result. Noise and pore-characterization
tools have their own input requirements; event detection is not a prerequisite for opening
Analysis home or examining a noise spectrum.

**Three different kinds of navigation:** the Inspector's **Analysis** tab contains preparation,
detection and measurement controls; the separate Explorer has one tab for each opened analytical
tool; inside a tool, the **Section** dropdown selects **Data**, **Analysis**, **Appearance** or
**Export** controls. These are not four more tool tabs. Changing Section only changes what is
visible. For a histogram or fitted overlay choose **Distributions**; for ECDF, survival, box or
violin choose **Distribution summaries**; for facets, a pair matrix or strip/swarm choose
**Comparison figures**. The last two are available through Analysis home or the Analysis menu,
even though they do not have individual buttons in the short launcher row.

Opening a tool is a distinct action from browsing its description: a live Distributions,
Event waveforms or Noise spectrum view may calculate immediately. On the two opt-in pages,
**Refresh current** is another way to execute **Generate**, not a read-only availability check.
Read the displayed source and result state before using it.

### Guided first analysis

Choose **Guided first analysis…** from **Help**, **Analysis**, or the Analysis home catalog.
The guide is modeless: keep it beside the workspace and move between its seven steps while
using the normal controls.

1. **Open a recording.** Select the current channel and identify the active recording.
2. **Verify acquisition and units.** Check sampling, gain, offset and current units against
   acquisition records rather than trusting a non-default setting.
3. **Inspect noise and baseline.** Examine a quiet region, measure the open-pore reference,
   inspect any proposal, and apply it only after deciding it is appropriate.
4. **Detect candidates.** Inspect the actual settings and active processing chain, then run
   detection yourself.
5. **Review the events.** Compare boundaries, baseline and waveform shape with the trace.
   Acceptance by an analysis mask is not a record of human review.
6. **Make a figure.** Choose the sources and metric, generate the intended result, then adjust
   appearance and inspect its units and interpretation.
7. **Export and save.** Distinguish exporting a picture, saving a Figure Library snapshot or
   bundle, and saving the resumable project with its required source files.

Each guide button opens the relevant existing control or explicitly requested action. Opening
the guide, changing steps and checking **I checked this step** never sets scientific parameters,
applies calibration, starts detection or fits a model. The checkmarks are learning notes, not
certification; they reset when the selected recording context changes.

Opening another Inspector tab may mark the project unsaved because the existing workspace
saves which tab you were viewing. This is view metadata, not a change to scientific settings,
results or review decisions.
An explicit tool-opening click can still compute that tool's normal live view, for example a
Noise spectrum or Distributions view. The no-computation guarantee concerns reading and navigating
the lesson, not every tool action it offers.

For practice, step 1 offers **Create synthetic practice recording…**. Creation is an explicit
choice with a file destination. The generated CSV has a required **SYNTHETIC-** filename prefix and is
opened alongside existing recordings, not substituted for experimental data. It contains noise,
slow drift and four artificial blockades. Its generator settings are explained in the guide;
they are not promises about detector output or a validation benchmark. The synthetic file's
current values are already pA, but CSV units still need declaring when necessary. Never copy
the practice calibration to a real recording without checking the acquisition conversion.
The final normalized filename is checked for overwrite conflicts, and a practice recording
cannot replace a recording that is currently open.

The Inspector's **Analysis** tab is where the work happens, and it has its own four-tab strip:
**Preprocess** (choose **Signal chain** — whose stages you edit in the Signal Conditioning window, §5 — or **Open-pore baseline** from its section selector, §5a) · **Detection** ·
**ML** · **Measure**. Preprocess configures the signal and reference; Detection
and ML produce results shown in the **Results** panel under the trace, on its **Events** tab;
it also has **Summary**, **Fits** and **Quality** tabs. Essential/Advanced changes control
visibility only.
Manual measurements have their own measurement list. **Add measurement to list** appends there;
**Mark selection as missed event** instead adds a reviewed candidate to the active source detector
table. These are different destinations. When the recording holds more than one detector
result, the **Event table** selector — **Table**, at the left of the Results panel's toolbar —
chooses which stored table is shown; changing that selector changes only the view and does not
replace or merge any result.

### Essential and Advanced controls

The Inspector's **Controls** selector changes presentation complexity only. **Advanced** is
the default when no preference has been saved; choosing **Essential** hides less-frequent
tools and collapses advanced settings without resetting, disabling or replacing their values.
Enabled filters and important retained settings are disclosed in the Essential summary.
Configured or unresolved ML models remain discoverable. Use **Show advanced tools…** or select
**Advanced** to inspect the complete controls again. **Baseline reference…** remains a direct
route to the open-pore reference controls.

Your explicit selector choice is remembered as a UI preference. Guide navigation can switch
the view without overwriting that preference. Neither view is a scientific preset, and switching
between them does not rerun analysis, change results or alter review history.

### What the status indicators mean

**Current** means a displayed result matches its tracked source and settings, not that its
calibration, event review or scientific interpretation has been validated. **Stale** means the
source or settings changed after the visible result; inspect the difference before refreshing.
**Saved snapshot** identifies a frozen result displayed without recomputation. These definitions
are available in the Explorer status tooltip and on Analysis home.

The calibration chip distinguishes **Default gain settings** from **Gain settings entered**.
Neither certifies the amplifier conversion, physical units or open-pore reference. Defaults can
be appropriate for already calibrated input, and an entered gain can still be wrong.

The workflow's event counts distinguish candidates accepted by the analysis mask from rows
with a recognized human-review record. Unknown legacy review status stays unknown rather than
being counted as reviewed. The Curate completion mark requires review records for all candidate
rows and no unresolved conflicts; it still does not establish molecular identity or detection
accuracy. Guide checkmarks are separate from these event-review records.

These additions are intended to make the workflow easier to discover and explain. No formal
newcomer usability study has yet established a learning-time or error-rate improvement.

Menus: **File** (Open, Open/Save Project, Create segment from selection, Close document, Quit),
**Edit** (Undo / Redo — §9b), **View** (Overlay/Stacked, Zoom Back/Forward, Reset zoom, drag mode,
Panels), and **Analysis**, grouped into **Detect** (Run detection, Batch detect, ML model),
**Curate** (Event inspector, Measure event, Add measurement to list, Compare baseline estimators —
§9a), **Explore this recording** (Distributions, Distribution summaries, Comparison figures,
Event waveforms, Event gallery, Reconcile detector results, Event labels, Noise spectrum),
**Compare recordings** (I-V / conductance, Voltage dependence) and **Export** (events, training
data, figures). **Analysis home** and **Guided first analysis** also have menu entries; the guide
is available from **Help**. **Window** lists the windows NanoScope opens — the main window,
Event inspector, Event gallery, Analysis Explorer (with one indented entry per open page), Figure
Library, Signal Conditioning, and — once opened this session — Guided first analysis, About
NanoScope, the Incubation study launcher and the Original Zn reference recipe — with a check mark
beside each one on screen; choosing one opens it or brings it forward through the same door as its own menu
item. It also holds **Bring All to Front**, **Dock Event inspector beside the trace** (§9) and
**Reset Window Layout…**. Child windows reopen where you last left them (size and place,
remembered between sessions and brought back onto the screen you have now).

Closed a dock by accident? Reopen it from **View ▸ Panels ▸ File Manager / Inspector / Results**
(**⇧⌘R** toggles Results), or put everything back with **Window ▸ Reset Window Layout…**, which
asks first, returns the panels to their opening places and sizes and the Event inspector to its own
window, and forgets remembered window sizes (your analysis is not changed). While the Results panel
is closed, a new detection or a table turning stale is announced on the status bar; the panel is
never re-opened for you.

---

## 2. Loading recordings

**File ▸ Open…** (Ctrl+O), or drag a file onto the window. Multiple files can be open at once and
compared. Supported formats:

| Format | Notes |
|---|---|
| **Axon ABF** (`.abf`) | Read via `neo`. These are ABF‑v1 float recordings; the standard `pyabf` cannot read them. A different `CPYA` variant is detected and refused with a clear message. |
| **HDF5** (`.h5`, `.hdf5`) | Flat `/current`, `/voltage`, `/conductance` datasets. Units are **not declared** in the file, so they are reported as unknown and you set them in Calibration. Traces are stored as float32; a float64 file whose live values sit below float32 range is not silently flushed to zeros — the downcast is audited and the Issues box names the flushed channel ("…nonzero but below float32 range"). |
| **TDMS** (`.tdms`) | Modern LabVIEW layout (sample rate known) and a legacy layout (rate **unknown** — the x‑axis is then in samples and detection is disabled honestly). Channels stored as raw **Volts** are flagged as such. |
| **Text** (`.csv`, `.txt`) | Trace tables: column 0 is the time base, every other column is a signal. Quoted fields are read the way Excel writes them, so a header like `time,"current, pA"` and fully quoted numbers are ordinary input, not damage. A column literally named `current`/`voltage`/`conductance` sets the channel *kind* (so a CSV's current column is analysable); units still stay unknown until you set them. |

**A file with the wrong extension is told what it is.** Open an HDF5 recording named `.abf`, an ABF
named `.csv`, or either of them under a name this app has never heard of — `.dat`, `.bin`, or no
extension at all, which is what an acquisition script or an archive tool leaves behind — and the
message names the format its bytes really are and which extension reads it here, instead of the
decoder's own words for a file that is not damaged at all. A `CPYA` ABF is named the same way but
without the rename advice, because renaming it will not make this build read it. (A file of plain
*text* under a binary extension is the one case left unnamed: "these bytes are text" is not "these
bytes are a trace table", and this app will not tell you to rename your notes.)

**A file that cannot be opened at all says which of the four things went wrong**, in the same
words whatever the file is called: *is no longer there — it was moved, renamed or deleted, or it
lives on a drive that is not mounted*; *is a folder, not a file*; *cannot be read: permission
denied*; or — the one failure that lives in the process rather than on the disk — the app has run
out of file descriptors (EMFILE/ENFILE), which is named as exactly that instead of blaming the
recording as damaged. Being a folder is reported ahead of a permission problem even though opening a locked
folder is technically a permission error, because `chmod` will not turn a folder into a recording —
the message you are given is one that stays true after you act on it. None of these is described as
a format problem: the app does not tell you what is inside a file it never managed to read.

**Channels.** For a typical ABF the reader finds a dead channel (all-quiet), a voltage channel, and
a current channel; the **current channel is selected automatically** on open, so detection and
measurement point at the right signal from the start.

**Large files.** Recordings up to ~15 M samples/channel (≈300 s at 50 kHz) open and pan smoothly.
Above that soft ceiling the app warns and offers to **load a sample range** instead of the whole
file. Trace rendering uses peak-preserving min/max decimation, so a one-sample spike is never hidden
when zoomed out. A partly loaded recording is labelled in the Files dock — *(partial: samples
30,000,000–31,000,000)*.

**A partial load carries the window's own experimental time.** Load samples 30 M–31 M of a 50 kHz
recording and the trace starts at **600 s**, not at 0 — because that is when those samples were
acquired. In builds before this one only the HDF5 reader got this right; ABF, TDMS and CSV windows
all reported the *file's* origin, so every experimental time from a windowed load was `start / fs`
seconds early — **600 s early** on the window above. **Whole-file loads are unaffected and always
were**, so a number you wrote down from a complete load needs no revisiting; a number taken from a
partial load in an older build does.

For a CSV the window's start time is **read out of the file's own time column** at that row, not
computed from the sample rate — so a recording whose acquisition paused, or that was concatenated
from segments, is still placed where the file says it was. (It is computed only when that row's
time cannot be read, and a **Note** says so when it happens.)

**Two clocks, and the app labels both.** The app does *not* put every time on one clock, on purpose,
and reading a number off one surface into a control that wants the other is how a headline noise
figure came out 5.7× wrong on these recordings. The two are:

| Clock | Zero at | Where you see it |
|---|---|---|
| **Instrument clock** | the experimental time the acquisition software wrote into the file | the trace's x-axis and its cursor crosshair — labelled *s (instrument clock — this recording starts at t₀ = …)* — and **t₀** in Inspector ▸ File Info |
| **Recording clock** | the **first sample that was loaded** | the event table's `t_start_s (t₀=0)` column, a hand measurement's `t_start`, and any window a panel takes in seconds |

So on the 30 M–31 M window above, the crosshair reads ≈600 s while an event 2 s into the window
reads `t_start_s = 2`, not 602. A **segment** cut from that window starts its own clock at 0 as
well, and names its parent time in its label — *samples 300,000–340,000 · t₀=606.1s*. Every control
that shows or takes a time says which clock it is in; when in doubt, read the label rather than
assuming, and hand a selection between surfaces by dragging it on the trace rather than by typing
the number.

### Read the Issues box

**Inspector ▸ File Info** ends in **Notes** and **Issues**. Notes are advisories (a fallback text
encoding, a channel stored in raw Volts). Issues are the ones that matter: *this recording is not
what it looks like.* The box is not decorative, and a file that shows nothing there was read clean —
it is never "not checked".

What turns up there:

| What happened | What the Issues box says |
|---|---|
| **A truncated TDMS** — the acquisition or the copy stopped mid-file | Which segment it stops inside, how many samples survived, and that *every duration, event rate and capture rate computed here belongs to those samples, not to the recording you started*. Do not pool it with complete files. |
| **A CSV or HDF5 whose loaded timestamps are invalid or change interval** | Every loaded timestamp is checked, including chunk boundaries. An invalid or nonuniform clock leaves the sample rate **unknown**, uses a sample-index axis, and disables seconds-based analysis. Clean selected windows remain usable when their own clock is valid. The quick opening probe is preliminary; it cannot establish that the middle of a file is sound. |
| **A channel that exists and was refused** | The dataset's name and the reason, so a missing channel is never indistinguishable from a file that never had one. |
| **A non-UTF-8 text file** | Which encoding was fallen back to and why — Latin-1, or UTF-16, which is what Excel's "Unicode Text" and LabVIEW write by default. A clean fallback is a **Note**; it becomes an **Issue** only when the decode was lossy, i.e. characters in your file did not survive. Either way the interval and span checks above still run on a UTF-16 file, which they did not always. |
| **A load window that outran the file** | What was asked for, where the file actually ends, and the consequence that matters: *if this window came from a saved project, the recording is not the length it was when that project was saved.* |
| **An HDF5 `/time` longer or shorter than the signal** | How far past (or short of) the last sample the file's own clock runs, and that the samples that exist are what every duration here is computed from. |
| **A time base that cannot be in seconds** — an HDF5 `/time` or a CSV time column stepping by 100 or 20,000 per sample, which is what an exporter logging **microseconds or nanoseconds** writes | The step it measured, the rate that would imply (`0.01 Hz` — less than one sample per second), and that the unit is declared nowhere in the file, so it is **not guessed**: the sample rate is left *unknown*, the x-axis is in samples and detection is disabled, rather than every duration and dwell time being wrong by a factor of a million in silence. Re-export the time base in seconds. |
| **A TDMS whose channels declare different start offsets** | Each channel's declared `wf_start_offset`, and which one the recording's t₀ was taken from. Dwell times are unaffected; only the absolute experimental time this recording is stamped with. (Different *sample rates* across channels are refused outright instead — see the TDMS row above.) |
| **A calibration that destroyed the channel** | A gain large enough to overflow the 32-bit sample format, named with the number of samples it turned to `inf` (§3). This one is damage the *app* did, and it is listed beside the load's own findings. |
| **An HDF5 `/time` that exists but cannot be used** — fewer than two samples, or not monotonic | That it is present and unusable, with the reason (*it holds only 1 sample(s), and at least 2 are needed to measure a sample interval*), and that the sample rate is left **unknown** rather than guessed. |

**An integer `/time` dataset in an HDF5 file is read as whole seconds, and File Info says so** — as
a **Note**, not an Issue, because the file may well be right: *"/time is stored as int32 — an
integer time base, read as whole seconds (50,000 Hz). If this file's clock is in milliseconds,
microseconds or nanoseconds, every duration and rate here is wrong by that factor: re-export /time
as float seconds."* This is the one advisory in the Notes list worth reading like an Issue. The file
opens, every shape is right, and nothing on screen looks out of range — the numbers are simply off
by 10³, 10⁶ or 10⁹. LabVIEW and hand-written exporters are the usual sources. The app will not guess
the unit for you; re-export `/time` as float seconds. (A clock so far out that it cannot be seconds
at all is a different case and *is* an Issue — the row above about microseconds and nanoseconds.)

---

## 3. Calibration

The **Calibration** tab applies a linear transform per channel: `calibrated = raw × gain + offset`,
with a target-units label. It shows a live **raw-vs-calibrated** table (median / min / max / span)
and a **command-voltage check** that reconciles the measured voltage against the nominal applied
voltage.

- For ABF recordings the **voltage channel** defaults to a gain of `0.1` (the raw values are stored
  10× too large); you can see this reconcile in the command-voltage check (e.g. raw −5999 mV × 0.1 =
  −599.9 mV vs a nominal −600 mV).
- The **current channel** defaults to a gain of `1.0`. **This is the key calibration to set for your
  rig.** Until it is correct, the current axis is in raw units — detection and dwell are unaffected,
  but absolute blockade depths (pA/nA) and conductance are not physically scaled. Edits apply live
  and re-render the trace without a reload.
- **Treat this channel as** is the other half of the same correction. The units box says what the
  numbers *mean*; this combo says what the channel *is* — leave it at *"as the file declares"*
  unless the header is wrong or silent. It changes no number and reaches no sample (it is not part
  of the calibration fingerprint, so there is no live preview of it): it decides what the channel
  **is to the whole app** — which **pane** it draws in and which **overlay group** it shares, and,
  because the whole pipeline resolves a channel's kind through this one override, which channel
  **Detect**, **Analysis ▸ Run detection**, **Batch detect**, **Measure this recording**,
  **Noise spectrum** and **I-V / conductance** act on, and what the calibration chip and the workflow strip
  say — once *Apply* commits it. They all ask the same resolver, so *current* on an undeclared
  channel makes it analysable the moment Apply lands (the Detect button goes from refused to
  runnable and the status-bar chip stops saying *No current channel*, with no reload), and *voltage*
  on a channel the file calls current switches detection off for it. A refusal your own override
  caused says so instead of quoting the header back at you: *Detection runs on CURRENT channels;
  this is a voltage channel (your Calibration-panel override; the file says current).* Without the
  override, a channel you had corrected to `pA` still sorted below the voltage, so two recordings of
  one experiment opened with their panes in opposite orders; and a TDMS whose channels are *both*
  raw Volts stays one group, sharing one pane and one y-axis, until something says which one is the
  current. The override is saved with the project; a project written by an older build carries none,
  which reads as "the header stands" and leaves the sidecar's own value alone.

**Where a calibration is stored, and its note.** *Apply* writes `<your recording>.npcal.json` next to
the raw file, so the same recording opens calibrated next time. That file carries a free-text
**note** — what the gain was measured against, when the electrode was replaced — and it is the one
field that only lives there. Projects saved by this build record the note too, so reopening a project
and pressing *Apply* no longer replaces it with an empty string. A project written by an older build
says nothing about the note, and is read that way: whatever the sidecar already holds is left alone.
Setting every channel of a recording back to its default deletes that file, so *Apply* and *Reset to
default* both **ask first, naming the file and the note**, and the status line afterwards says what
actually happened to it — including the operating system's own reason when a write fails. Clearing
**one** channel while another keeps the file alive asks the same question at the smaller scale: the
file stays, that channel's entry does not, and the note is shown in full before it goes. Changing a
gain carries the note forward and asks nothing.

**A gain of zero is refused,** at the box you typed it into, with the reason. Zero is not a small
gain: it flattens the whole channel to a single constant, and every number downstream — RMS noise,
baseline, depth — becomes a `0` that looks exactly like a measurement. This is not a rule against
small gains (`1e-30` is honoured); it is a rule against the one value that erases the recording, and
a value that *underflows* to zero in the 32-bit sample format is the same erasure by another route.
A **negative** gain is normal (an inverting headstage) and is accepted. If a hand-edited sidecar or a project holds a zero
or non-finite gain, that channel keeps its default and the app names it rather than refusing to open.

**A calibration that ruins a channel says so.** A gain large enough to overflow the 32-bit sample
format turns samples into `inf`, and that damage did not exist when the file was loaded. It is now
reported in **File Info ▸ Issues** alongside anything the load itself found, naming the gain and how
many samples it affected. An empty Issues box means the calibration added nothing, never "not
checked". Likewise the PSD panel now distinguishes *this channel has no noise in this band* from
*this channel is a flat line*, instead of printing "Total RMS noise: 0 pA" for both.

---

## 4. Viewing the trace

- **Panes.** Channels with different kinds/units get their own y-axis. In **Overlay** mode, channels
  of the same kind+units share a pane (so the current channels from several voltage runs overlay for
  comparison); in **Stacked** mode each layer gets its own pane. All panes share the x (time) axis.
- **Adaptive Navigator.** At broad zooms the strip under the trace shows the full recording. At
  event scale it automatically becomes a **Local context · 9× view** lens: the main window occupies
  about one ninth of the strip, with nearby waveform on either side, so its shaded region remains
  easy to grab even in a long recording. Drag the shaded region's body to pan without changing the
  zoom, or drag an edge to zoom. When **Measure event** is active, the blue band in the Navigator is
  a read-only mirror of the boundary cursors; the shaded view region remains the navigation control.
- **Zoom.**
  - *Box-zoom* (default): drag a rectangle on the trace to dive into it.
  - *Mouse wheel*: moves the signal **vertically only** without changing either axis span. Scroll
    up to move the waveform up and down to move it down.
  - The **− / +** buttons in the **Scan** strip zoom out / in one step around the centre. Unlike
    Reset, **−** keeps you in local context instead of jumping straight back to the full recording.
  - **Full** in the Scan strip returns to the complete trace. **Reset zoom** (Ctrl+0) and
    trace-focused **Home** do the same.
  - **Zoom Back / Forward** (Ctrl+← / Ctrl+→) step through your view history — dive into an event,
    reject it, and pop back to where you were browsing.
  - **View ▸ Drag draws zoom box** toggles between box-zoom and pan.
- **Scan at the current zoom.** The **◀ / ▶** buttons beside the zoomed-in scrollbar move to the
  previous / next view while preserving the current scale. Each move advances by 90% of the window,
  leaving **10% overlap** so an event at a boundary is not lost. If the Measure cursors are active,
  they move forward with the view, ready to bracket the next event. **Shift+← / Shift+→** perform the
  same scan, and **Shift+↑ / Shift+↓** zoom in / out, while focus is on the trace.
- **Crosshair.** Hovering shows the time and the value at the nearest sample, in the channel's units.

---

## 5. Filtering

Filters are **pre-detection signal conditioning**. For the classical Schmitt and CUSUM detectors,
the orange trace is the exact **candidate-localization** signal: every enabled stage of the chain,
then the low-pass. Each stage is zero-phase, so it introduces no phase delay; filtering can still
reshape an event and move a threshold crossing. This matters especially for high-pass filtering.

Each filter can be enabled independently, and the fixed order is **notch → median → high-pass →
baseline drift → running mean → Savitzky-Golay → Normalize → low-pass**. The low-pass is
applied last inside the detector. The table below is ordered for reading, not for execution.

High-pass creates one deliberate exception to the single-chain model. A new Schmitt or CUSUM run
uses the orange, high-passed branch only to find candidate boundaries. NanoScope then rebuilds the
same chain with only high-pass disabled and measures the rows on that
**characterization branch**. High-pass itself therefore does not erase its DC reference; every
other enabled transform still applies. In particular, Normalize still recenters the branch and
makes physical-current and fractional-blockade interpretation invalid. With high-pass off,
localization and characterization receive the same samples.

### The Signal Conditioning window

The stage controls live in their own window, **Signal Conditioning**. Open it in any of four
ways: **Analysis ▸ Signal conditioning…** (**Ctrl+Shift+F**), the **Open Signal Conditioning…**
button pinned at the top of **Preprocess ▸ Signal chain**, the **Filter** step on the workflow
strip, or **Advanced signal-chain settings** in the open-pore baseline workspace (which opens it on
the Baseline drift stage). There is one window: closing it hides it, and every setting stays on the form. It opens
against the right edge of the main window, over the Inspector, so the main trace stays in view;
**Show main trace** on its toolbar brings the main window back in front, and NanoScope remembers
where you put it. **Ctrl+S** / **Ctrl+Shift+S** save the project from this window too.

The window edits the same chain the Preprocess tab always held — it moves the controls, it does
not copy them — so detection, **Measure ‘Processed signal’**, **Export processed trace** and a
saved project all read exactly what it shows. A grey line under its toolbar says so: the chain is
*used by Schmitt/CUSUM detection, Measure ‘Processed signal’ and Export processed trace — not by
the ML detector (ML tab ▸ Model preprocessing)*. **ML chain…** shows that other chain.

On the left, **Stages, in execution order** lists the eight stages the way they run — notch,
median, high-pass, baseline drift, running mean, Savitzky-Golay, normalize, then the low-pass,
marked *applied inside detection*. Each row shows the stage's current setting, then its role
(when a narrow column cuts a row short, the role goes first; the tooltip has it all); with a
sample rate, the low-pass row's tooltip adds its true −3 dB corner. Tick a row to enable the stage; select it to
edit it in the form below. Notices about the chain (high-pass is for localization only,
Normalize is on, a refused detrend method) appear **below** the form, so enabling a stage never
moves the control you just clicked.

### Previewing the chain: Time, Spectrum and Response

The tabs on the right preview the chain on the **window the main trace shows** (with **Follow main
trace** on; at most 2,000,000 samples — 40 s at 50 kHz — and the centre of a wider view, which the
status line says). Panning or zooming a preview turns Follow off and previews that range instead;
tick Follow again to rejoin the main trace. The previews are computed off the user interface on one
background thread, a quarter-second after your last edit, and a newer edit cancels an older
preview; nothing is computed while the window is closed. **Ctrl+1 / Ctrl+2 / Ctrl+3** pick the
three tabs.

**Time** draws the raw (calibrated) trace in grey and the processed trace in blue, with a legend:
**Overlay** puts them on one axis, **Split** on two linked axes, and **Raw − processed** shows what
the chain removed. When the chain changes the unit (Normalize) or removes DC (high-pass) the view
is forced to Split, and the processed axis names its real unit — *processed (σ, whole-trace
z-score)* or *processed (pA, DC removed by 100 Hz high-pass)* (shortened on an axis too short for
it; hover the axis for the full label). The legend sits above each plot, never over the trace.

**Window estimate versus exact.** The window is padded on each side by the chain's longest settling
or window length, so for the local stages — notch, median, high-pass, running mean,
Savitzky-Golay and the low-pass — the preview is the same samples as the whole recording,
conditioned and sliced: the status line reads *Exact for these stages · window 2.00 s, padded
±1.47 s*. **Baseline drift** and **Normalize** use statistics of the whole recording, so on a
window they are only an estimate: the status line says *Window estimate — Baseline drift/Normalize
use whole-recording statistics* and the legend says *processed — window estimate*. The padding is
capped at 40 s a side, so a very slow high-pass or notch can be an estimate too, and then the
status line names it (*Window estimate — High-pass settles over 400 s, longer than the ±40 s
padding*). A view wider than the preview limit is previewed at its centre, and the status line
says so next (*centre 40.0 s of the 250.1 s view*). **Compute exact** conditions the whole
recording (seconds on a long file, in the background) and shows its exact samples (*Whole
recording (exact)*); it runs only when you press it, and an edit of the chain cancels it. While
it runs the button reads **Computing…** and the status line starts with *Computing exact (whole
recording)…*; if it fails it starts with *Exact pass failed: …* and the button offers **Retry
exact** (for that chain only). Right after an edit, until the new preview arrives, the status line
starts with *Updating…* and already gives the verdict of the chain you now have, and the legend
calls the curve still on screen *processed — previous chain*. The status line keeps three fixed
lines, what is happening first, so the plot never moves; hover it for the whole sentence. The
Phase, Group delay and short Step views are computed at once, so they never wait for an exact
pass; only the step of a very slow stage does, and its note says so.

**Spectrum** shows Welch spectra of the same window before (grey) and after (blue) the chain on a
log-log plot, with the frequency resolution above it (*Δf 6.10 Hz (nperseg 8,192) on a 2.00 s
window*); resolving mains lines needs Δf ≤ 1 Hz, which takes about a 10 s window at 50 kHz. With
Normalize on, the *after* curve leaves Normalize out (it only rescales) so both curves share one
unit. Below the plot, a list names the narrow lines in the raw spectrum — at least 10 dB above a
running-median floor a quarter-octave wide, from 20 Hz up to the low-pass cutoff (or Nyquist) —
tagging mains harmonics (*120.0 Hz · +24.0 dB · mains 60×2*). A line is placed on the harmonic
itself only when Δf ≤ 1 Hz; at a coarser resolution it is listed where it was measured
(*240.3 Hz · +24.0 dB · near mains 60×4*), because a line 5 Hz off a harmonic still falls within
one 6 Hz bin of it, and a notch on the harmonic would leave most of it. Tick lines and press
**Send to notch** (or double-click one, or press Return on it) to add them to the notch's explicit
line list through the form's own checks: the notch is switched on, a line the notch already
removes (at least 20 dB off) is marked *notched* and skipped, the list never grows past 32 lines,
and a line whose frequency the spectrum cannot place within the notch's own width (*Δf too
coarse*) cannot be sent — widen the window first. If the notch is off but keeps a list of its own,
the lines join that list and the notch **stays off** (*Notch still OFF: added … — tick Notch to
apply them*): switching it on would bring back lines you switched off, and replacing the list
would lose them. The outcome is said in the two lines under the button, and on the status bar.

**Response** has four views of the selected stage — or, with **show combined response of enabled
linear stages**, of the whole linear chain. **Magnitude** is the applied |H(f)|. **Phase** and
**Group delay** show the applied response as a solid line — every stage NanoScope applies is
zero-phase, so it is flat at 0 (a 180° step is a smoother's sign flip, not a delay) — and, dashed,
the same IIR design run once causally, which is what a hardware filter of that design would do.
**Step** pushes a unit step through the linear stages and reads off the 10–90 % rise, overshoot and
pre-shoot (a zero-phase filter rings symmetrically before and after an edge). The median, baseline
drift and Normalize are listed as *not representable as a linear response* rather than drawn.

The line headed **Active execution order** is the recipe NanoScope will actually run, including the
detector-owned low-pass in its real last position. It stays on the Preprocess card in the dock.
When high-pass is on, it instead names the **Candidate-localization order** and says that
quantitative measurement omits high-pass. The response graph says **ON · included in execution
chain**, **ON · candidate localization only** for high-pass, or **OFF · design preview only**. An
off-stage design is grey and dashed so inspecting a useful-looking curve cannot be mistaken for
enabling it.

Tick **show combined response of enabled linear stages** to replace that curve with the cumulative
response of the active linear cascade. NanoScope multiplies the zero-phase responses on one shared
frequency grid and names the stages it cannot honestly put in a fixed transfer function: median,
baseline estimation and normalization are nonlinear or data-dependent. When high-pass is on, both
this combined response and the orange trace show the **localization branch**, not the no-high-pass
branch used for quantitative measurement.

**Protocols.** The **Protocol** combo at the top of the window recalls a saved chain. A protocol is
the complete chain: picking one sets every stage, and turns off any stage it does not name — the
low-pass included. The combo shows the protocol's name only while the form still is that protocol;
the moment a setting differs it reads **— manual —**. That includes a value this recording cannot
hold — a cutoff above its Nyquist limit, say: the form clamps it, the combo reads **— manual —**
and the status bar names what changed. **Export protocols…** writes your own
protocols to a `.json` file; **Import protocols…** adds the protocols of such a file to your
library. Import never overwrites: a name you already use comes in as *name (imported)*, a protocol
identical to one you have is skipped, and a file that is not a NanoScope protocol file, or a
protocol naming a setting that is not part of the signal chain, is refused by name.

**Copy as Python** puts a short script on the clipboard that rebuilds the processed trace outside
NanoScope: it opens the recording with NanoScope's own reader, applies the channel's calibration
(gain and offset), runs `nanoscope.analysis.conditioning.conditioned_trace` with the chain on the
form, and slices the window the main trace shows (**Follow main trace** on; untick it to keep the
last range — selecting another recording or channel drops that range, and the script then takes
the whole channel).

**When the form no longer matches the table.** If the event table on screen came from a Schmitt
or CUSUM detection and you change the chain, an orange **⚠ Chain changed** button appears in the
table's toolbar, the Preprocess card and the window's status line name the differences
(*low-pass 10000 → 5000 Hz; notch Q 30 → 20*), and **Revert to detection's chain** is enabled.
The rows are not marked stale: they are a faithful record of the chain they were detected with,
and nothing changes until you run detection again (**Run detection…** in the window, enabled
exactly when the Detect button is, or the button's menu). A table from the ML detector or a reconciled table is labelled as not produced by
this chain.

Open **Preprocess ▸ Signal chain** for the compact card: the comparison with the table on screen,
any warning, the protocol name, the scope line and the execution order, with **Open Signal
Conditioning…** and **show processed trace on the plot** pinned above it. The baseline estimator is
the separate **Open-pore baseline** subsection.

| Filter | What it does | When to use it |
|---|---|---|
| **Low-pass** (8-pole Bessel) | Removes high-frequency noise above a cutoff. Applied inside the detector so rise-time and filter-limited flags stay correct. | Almost always — set the cutoff to your acquisition bandwidth. |
| **High-pass** (Butterworth/Bessel) | Removes very low frequencies / DC drift from the **candidate-localization branch only** for new Schmitt/CUSUM runs. Baseline, depth, fractional depth, mean/max deviation and SNR are measured on the otherwise identical branch with high-pass omitted. High-pass still reshapes the candidates: start/end, dwell and rise/fall remain boundary-dependent, and ECD uses high-pass-derived integration limits. | Locating candidates in special cases only; prefer *Baseline drift* when you need drift removal without high-pass boundary distortion. |
| **Notch** | Removes mains hum at a frequency (default 60 Hz) and its harmonics. | When line pickup contaminates the trace. |
| **Savitzky-Golay** | Polynomial smoothing that preserves peak shape better than a moving average. | Gentle smoothing without a hard low-pass. |
| **Running mean** | Simple moving average. | Fast, coarse smoothing (blurs edges). |
| **Median** | Sliding median. **Removes impulse/spike noise while preserving square event edges.** | Despiking a noisy trace before detection. |
| **Baseline drift** | Estimates the slow open-pore level and *subtracts the drift*, keeping event depths intact (not a high-pass). Methods: block-median, percentile, ASLS. | Long recordings with electrode/temperature/clogging drift. |
| **Normalize** | Rescales the whole conditioned trace onto a **dimensionless** scale — `z-score`, `robust z` (÷ the same 1.4826·MAD sigma the detector thresholds on), `min–max` → [−0.5, +0.5], or the clipped `model input` transform. Applied **last**, after every other stage. **It divides the picoamps out:** current-derived amplitudes, baselines, thresholds and pA/pA·s export fields retain physical-unit labels while no longer carrying physical units. Centering makes `frac_depth`/dI/I0 invalid; model-input clipping can also move boundaries or counts. Index, time, source, review and status fields keep their ordinary meanings. Off by default; while it is on an amber banner stays up under every stage of the Signal Conditioning window, and the Preprocess card says so. | Feeding a model, or putting recordings on a common scale — **never** to measure a conductance. |

Notes:
- The **Baseline drift** window must be *much longer than your longest event* (it flattens drift, not
  events). Of its methods, **block-median** and **percentile** flatten drift effectively; **ASLS** is
  a stiff global fit and is weak at default settings.
- The low-pass and Savitzky-Golay are both smoothers — usually you use one or the other.
- A result restored from an older NanoScope build has no dual-signal marker. If high-pass was on,
  its baseline and amplitudes really were measured on the zero-DC trace. NanoScope keeps those old
  rows reproducible, labels them candidate-only, and tells you to re-run; it does not silently
  reinterpret them as measurements from the new branch.
- A narrow low-frequency notch can require a very long end-settling pad. The Notch response note
  reports the exact seconds and samples per side before you run. NanoScope refuses a pad above its
  2,000,000-sample safety limit rather than attempting a surprise allocation; remove or raise the
  lowest-frequency notch line, or reduce Q.
- Rise-time measurement is likewise refused above a 2,000,000-sample impulse probe. This protects
  the app from an accidental resource spike; choose a physically meaningful bandwidth/order rather
  than interpreting an unbounded probe as extra precision.
- **The preview says why it is empty.** A ticked box that draws nothing used to be the only symptom
  of an over-Nyquist cutoff, a detrend window longer than the trace, or a legacy TDMS with no sample
  rate. Now an amber line under the toggle names the reason — *processed-trace preview unavailable:
  sample rate unknown — the filter stages are designed in Hz and seconds, so the processed trace
  cannot be computed for this recording (detection is disabled for the same reason)* — or repeats
  the failing stage's own complaint. With a known rate the three frequency spins are also capped at
  that recording's Nyquist, so a 100 kHz cutoff carried over from a faster file is pulled down
  instead of poisoning the next run.

---

## 5a. The open-pore baseline — measure this recording

In **Preprocess**, choose **Open-pore baseline** from the section selector. This is not signal conditioning
and it is not cosmetic: it is the estimate of `I₀(t)`, the current your pore carries with nothing in
it, and **every depth in the app is measured against it.** It also feeds the detector's threshold, so
it decides where events *start and stop*, not only how deep they are. When high-pass is enabled, the
same estimator settings are applied separately to the localization branch for thresholding and to
the no-high-pass characterization branch for the values written into the event table.

The estimator is a two-pass masked median with two windows you set:

| Control | What it decides | The failure it causes when wrong |
|---|---|---|
| **Rough window** | Pass 1 — **which excursions count as events** and get masked out before the baseline is measured. | It must be several times the longest state you want treated as an event. A rough window shorter than the state it is sitting in *becomes* that state and nothing gets masked. On the −600 mV apo file, the 521 ms partial clog is **15 % masked at 0.5 s and 99.96 % masked at 2.0 s**. |
| **Base window** | Pass 2 — **how fast the baseline may follow the trace.** | It must be shorter than the shortest genuine conductance level dwell, or a gating pore's steps get smoothed *into* the baseline instead of being measured against it. |
| **Decimation** | Derived from the base window, shown but never set by hand. | It is the cost throttle (cost goes as 1/decimation²), not a quality knob. |

### Press "Measure this recording"

The button probes the **whole channel on the quantitative characterization branch** — the current
filter recipe and low-pass, but with high-pass omitted even when the orange localization preview
includes it. It usually takes a few seconds: measured 2 s on a 3 M-sample recording and 6 s on 9 M
samples, with the button reading "Measuring…" meanwhile. It measures three things from that trace:
a robust noise σ, the distribution of excursion dwells (with short gaps closed, or
the apo file's shallow 521 ms clog fragments into ≤ 26 ms runs and is missed entirely), and whether
the pore is **gating**, plus the mean interval between its conductance steps. It then shows a
**proposal with its evidence**.

The rule's own derivation is at the top; under it is one **`Ranked:`** row per window the probe
actually estimated, and the winner is marked **`— CHOSEN`**. The ranking can and does override the
rule — when it has, the reason line says so in capitals (*"…IS THE RULE'S WINDOW AND NOT WHAT THE
APP OFFERS … the ranking that follows SUPERSEDES it with …"*, naming both windows), so the two
windows above the table are not always the two the app is offering.

**Every candidate is judged on two numbers, and the second one is the important one.** The first is
the *median* share of each state's own depth the tracked baseline follows — the same statistic
Baseline health condemns above 3 %. The second is the **worst single state**: a window whose
baseline followed more than **50 %** of any one state's excursion has *ridden into* that state, and
loses to any window that stays out, whatever the medians say.

The second criterion exists because a median over hundreds of states cannot see one catastrophe. On
the −600 mV apo file at 1 s / 300 ms the median reads **exactly 0.0** while that window's baseline
rides 56 % into a 473 ms state — and on the 521 ms partial clog, that window does not detect the
clog **at all** and invents about thirty shallow events on the collapsed local baseline instead,
with every health check green. Both numbers are printed on every row for that reason: a row showing
only the median would show a shorter, cleaner-looking window losing for no visible reason.

Ties go to the **shorter** window — a longer one tracks slower drift less well, so the app takes the
least smoothing it can get away with. Windows the search never estimated are counted and named, and
the reason line is explicit that a stop rule is not a measurement of them.

It proposes; it does not apply. **"Use these values"** is the confirmation — the app never applies a
measured parameter on its own — and **"Back to default"** returns to 0.5 s / 0.1 s.

A proposal the estimator's own health checks condemn is **refused on the first press**, with the
reason: *REFUSED — 2 of the checks in Baseline health CONDEMN the windows proposed above (…)*. The
button then re-arms, reading **"Use these values anyway"**, and a second press takes them. It is a
two-step and not a refusal because the probe is deterministic — re-running it will offer the same
windows — so the only remedies are to widen the windows by hand or to accept them knowingly.

A line under the windows always says where the current numbers came from: *app default (this
recording was not measured)*, *measured on this recording, and confirmed*, *set by hand*, *restored
from the stored analysis*, or — on a recording that **Batch detect** ran with windows measured or
typed for another one — *COPIED from the batch form: set for a DIFFERENT recording — the one
selected when the batch form was filled in — and this recording was never measured*. Batch copies
one form onto every open recording: the
recording the form was filled in for keeps its own source, every other one is stamped `batch`, keeps
the never-measured advisory, and reads that last line here rather than claiming a confirmation
nobody gave. The line travels with the analysis into `provenance` (`baseline_param_source`), so a
stored result records not only which windows measured it but **whether anyone ever looked at this
recording.** A recording too short for the windows you asked for does *not* change it — the source
is still whoever chose the windows, and the never-measured advisory still fires when nobody did —
but a second line appears in amber underneath: *The detection on screen actually ran at rough
0.08 s / base 0.016 s, not the 0.5 s / 0.1 s above — CLAMPED to this recording's length…* (the
short-segment case below; the clamp is recorded beside the source, as `baseline_fallback`, never in
place of it). A project written by an older build can still carry that build's stamp for a clamped
run, which reads *restored from an older analysis whose estimator fell back on this trace (legacy
stamp; the windows were not chosen by anyone)* — a fresh run never produces it.

**There is one more reading, and it is the one to look for.** If you accepted a condemned proposal,
the same line adds the other half of the truth in amber: *measured on this recording, and confirmed
— **ACCEPTED OVER 2 FAILED HEALTH CHECKS** (mask_fraction; baseline_intrusion). Depths and dI/I₀
measured with these windows are not certifiable — see Baseline health below.* The reassuring first
half stays, because it is the provenance. The mark is attached to the **window values**, not to the
click: select another recording and come back — which re-seeds the same windows out of the stored
detection — and the mark comes back with them. It ends when the windows themselves move, by a hand
edit, by **Back to default**, or by a clean re-proposal.

The buttons stay clickable when they cannot act. Click one that is blocked and it tells you the first
real reason — no current channel selected, no sample rate, a probe already running — rather than
sitting greyed and unexplained.

### Why no default can be right

Measured on the four reference recordings this estimator was built against, the correct rough window
is **0.03 s** on `ferritin_data3` (a gating pore) and **2.0 s** on `…_0006_Process_-600 mV.abf` (the
521 ms partial clog) — and 2.0 / 0.5 s is exactly what the probe proposes for that second file.

The shipped default of 0.5 s / 0.1 s is a defensible **floor** for a recording nobody has looked at:
on the apo file it holds the median stamped-baseline error to **1.56 pA**, and the fraction of events
whose stamped baseline is more than 10 pA off to **0.97 %**, against the old 10 ms block median's
**80.6 %**. It is also **provably wrong for some pores** — on the gating pore it is **3.3× worse than
the estimator it replaced.**

**And the health checks below will still all read `ok` while that happens.** They catch catastrophe —
a mask that found nothing, a mask that swallowed the recording, a rough pass measuring drift instead
of events, a tracking window too narrow for the states it exists to ignore. A 12 pA
error on a trace whose noise floor is 4.6 pA is an 18 % depth error that looks entirely publishable,
and no check fires on it. That is why the app instead warns about the thing it *can* know for
certain: any table measured at the untouched default carries the warning **"these depths were
measured with the app's DEFAULT open-pore window; this recording was never measured."**

### Baseline health

After a run, the **Baseline health** block reports each check with its measured value and its
expected range, worst first. Two are worth understanding:

- **Mask fraction.** Too low has *two opposite readings* and the check says so: either there really
  are no events (fine), or the estimator has **inverted** — past ~50 % occupancy the mask lands on
  the open pore instead of on the events and the baseline sits at the blockade level. Measured: duty
  0.60 on a −100 pA pore gives a −295 pA baseline, **and the reported depths still look perfect.**
  Compare the baseline against the trace's modal current; the check cannot tell these apart for you.
  Too high means whole regions are being *bridged* rather than masked — shorten the rough window.
- **Overshoot.** The one QC scalar that resists a too-short window. The other three improve
  monotonically as you shrink it, so **never tune on them.**
- **Tracked level vs the recording's own histogram.** The check that exists because none of the
  others can see an *inverted* estimate — the one case where every other row is green and the
  baseline is sitting on the blockade. It does not condemn; it **stops certifying**, and the result
  then carries the standing advisory described in the next subsection.

### The open pore is chosen, not certified

Under the Summary table on the **Detection** tab is an amber block headed **"⚠ About these
numbers"**. It carries every caveat the detection was produced under, and one of them fires on
almost every real recording: **"The open pore here is NOT CERTIFIED, only chosen."**

The estimator's whole rule is that the open pore is the level the pore is at *most of the time*.
Nothing in the health block checks that this is physically the open pore. When the trace holds a
second conductance level for more than **2.5 %** of the recording — which is true of all four
reference apo files, whose second level holds 0.054 / 0.099 / 0.087 / 0.077 — or when the tracked
level holds less than 80 % of the recording, the app says so, names both currents and what share of
the recording each holds, and prints ΔI/I₀ **both ways**: against the tracked baseline and against
the other level. The depth in pA is the same number either way; only the denominator changes. Read
the two. If ΔI/I₀ against the *other* level is the one that matches the pore you know you have, the
estimator has settled on the blockade and every depth in the table is referenced to the wrong
current.

Measured on a synthetic pore blocked 81 % of the time (open 100 pA, blocked 60 pA): the baseline
lands at 60.04 pA, median ΔI/I₀ reads 0.665 against a true 0.400, the events are reported as
*enhancement* — and **all the health checks read `ok`, overshoot included.** At 90 % and 95 %
blocked the same fixture reads the same way. A recording 21 % off-level and a recording 79 % blocked
with its open-pore returns masked are the *same samples*, so nothing in the block can separate them.
This is why the app says the thing it can know rather than pretending to a check it does not have.

The block has no dismiss button, and that is deliberate: the only way to clear it is to remove the
condition — measure the recording, confirm the windows, detect again. **Compare the baseline against
the open-pore current you know for this pore** — the current just after the voltage step, or an
unclogged stretch — and analyse that stretch instead if the two disagree.

A warning here is not a claim that your data is wrong. It is a statement that this particular
question was not answered by any measurement, which is why it fires on ordinary recordings too.

### Short recordings and short segments

If the thing being analysed is **shorter than the base window** — a segment you cut from a trace, a
tiny test export, or an ML window handed to the characteriser — the app cannot run the estimator at
the width you set. At the default 0.1 s base window and 50 kHz, that means anything under **5,000
samples**.

What it does is **fit the same estimator to the length it has**: `rough_s` is clamped to the whole
recording and `base_s` is scaled to keep your own ratio between the two. Nothing is swapped for a
different estimator, and the result says which happened — the provenance carries
`baseline_fallback = windows_clamped_to_recording`, and the run carries a warning saying a shorter
segment is a worse measurement. The Open-pore baseline box keeps its *These values* line as it was
(the clamp chooses no window for you, so the source — and the never-measured advisory, if that is
the source — stand) and adds the amber *actually ran at… CLAMPED* line under it, quoting the widths
it used beside the ones the form asks for. Measured on a synthetic trace with a known −35 pA baseline and a
known 20 ms, 88 pA blockade: at 3,072 samples the clamped estimator reports **1.02×** the true
depth. From about **2,000 samples up** it reproduces the whole-recording answer to within a few
percent; below that it degrades, and that is real, not hidden — the same reason a rough window has
to be several times the longest state you want masked applies to a segment that is only a few
events long.

**So the practical rule: do not cut a segment down to a handful of events and expect the depths to
match the parent recording's.** Cut it wide, or measure on the parent. If you must work on a very
short segment, read the warning the baseline block posts and compare its `baseline_pA` against the
parent's.

> **Historical comparison:** an older build substituted a10ms block median and severely
> understated depth on this fixture. That is not a description of the current installed build.
> Use **Help → About NanoScope…** to identify the executable you are actually running.

---

## 6. Event detection

**Analysis ▸ Run detection** (Ctrl+D). Choose either classical detector in the Detection controls.
**Schmitt** is a two-threshold trigger: an event starts when the signal crosses
`baseline ± k_in·σ` and ends when it returns within `baseline ± k_out·σ`, where σ is the **MAD**
noise estimate. **CUSUM** accumulates one-sided evidence sample by sample and reports an estimated
changepoint when that evidence crosses `h`. Both operate over the whole current channel and return
the same event-table schema.

Polarity (blockade vs enhancement) is **measured from the data**, not assumed. In an Auto run with
high-pass enabled, the no-high-pass characterization branch owns the physical polarity decision. If
the high-pass branch votes for the opposite direction because its step response rings, NanoScope
discards that first candidate pass and repeats localization once with the physical direction.

That `baseline` is the open-pore estimator from **§5a**, so it affects where events *begin and end*,
not only how deep they are — which is why §5a comes before this section. With high-pass on, the
threshold uses the localization branch's baseline/noise while the row's baseline, depth,
fractional depth, mean/max deviation and SNR use the independently measured no-high-pass branch.
The candidate boundaries are deliberately retained, so start/end, dwell and rise/fall remain
high-pass-dependent; ECD is integrated from the no-high-pass samples but over those same inherited
limits. If you have not pressed
**"Measure this recording"** yet, do that first; a re-detection afterwards is one Ctrl/⌘Z away from
being undone (§9b), but a dwell time measured against the wrong baseline is not recoverable by any
later step.

Detection is enabled only on a **current channel with a known sample rate**; otherwise the button is
greyed with an honest reason (e.g. a legacy TDMS file has no rate → dwell would be undefined).

### Parameters (Detection controls)

| Parameter | Effect | To reduce over-detection |
|---|---|---|
| **Detector** | **Schmitt** uses threshold crossings; **CUSUM** accumulates evidence and estimates a changepoint. | Start with Schmitt for the familiar, fast threshold detector; try CUSUM for shallow persistent events. |
| **Schmitt k_in** (enter, in σ) | Threshold to *start* an event. The primary Schmitt knob. | Raise it (5 → 6, 7). |
| **Schmitt k_out** (exit, in σ) | Threshold to *end* an event; must be `< k_in` (hysteresis). | Keep well below k_in (default 2.0). |
| **CUSUM k** (drift, in σ) | Minimum sustained deviation that can accumulate evidence. | Raise it to ignore shallow noise; lower it to reach shallow events. |
| **CUSUM h** (threshold, in σ·samples) | Accumulated evidence required to fire; larger values trade sensitivity for delay. The reported onset is the estimated changepoint, not the crossing sample. | Raise it to require more sustained evidence. |
| **min dwell** | Events shorter than this are dropped. | Raise it to kill brief noise blips. |
| **max dwell** | Events longer than this are flagged (kept, not accepted). | Set to exclude clogs/wander. |
| **merge gap** | Events closer than this are merged into one, transitively along a chain. In **milliseconds** (0–1000 ms; `0` = off, and this is the one merge control where 0 really does mean off). A run whose merged span would be mostly open pore is **refused** rather than merged — see *Merging fragmented events* below. | Raise to combine fragmented events. |
| **Low-pass cutoff** | Smooths before thresholding. | Enable / lower to suppress noise crossings. |
| **Polarity** | auto / blockade (−1) / enhancement (+1). Beside it, a **measured** line reports what the last detection actually found — *blockade (down): 18 down / 0 up (confidence 1.00, measured)*, or *not yet measured* on a recording nothing has been run on. | Leave on auto; read the measured line before overriding it. A wrong override does not raise an error, it simply detects almost nothing. |

After a run the **Summary** shows counts and rates, the event table fills, and events appear as a
single batched **span overlay** on the current pane.

### Merging fragmented events

A noisy or multi-level translocation that briefly recrosses `k_out` is reported as several adjacent
rows, each with a fraction of the true dwell. **Merge sub-events** — shown above the event table
when you press **Merge…** in the Results panel's toolbar — stitches them back together *after*
detection, so you do not have to re-tune thresholds and lose the genuinely short events:

- **Window** — the merge window in **milliseconds** (0.01–100 ms, default 0.15 ms). Two events merge
  when `next.start − current.end ≤ window`, transitively along a chain. There is no `0` here: the
  minimum is 0.01 ms, which still stitches fragments that touch exactly. (`0` on the Detection tab's
  **merge gap** *does* mean off — merging is an optional step inside a detection run; here the button
  beside the field only ever means merge.)
- **Merge all** — every run of fragments in the whole detection, in one click, with the before/after
  counts in the status bar. On a large table it runs in the background with a Cancel.
- **Merge events in selection** — every event overlapping the on-trace Measure selection becomes one,
  whatever the gap. An explicit drag overrides the *window*.
- **Merge with next** in the event inspector does the same for one event and its successor.

Merged events are **re-characterized** over the new span, not concatenated, so depth, dwell, ECD and
SNR are real measurements of the whole passage. A merge is reversed by **Edit ▸ Undo (Ctrl+Z)**,
which restores the exact table including your curation — **never** by re-running detection, which
replaces the whole table and takes every accept/reject with it.

> **A merge can be refused, and "nothing merged" is not "nothing to merge".** The gap rule is
> transitive and has no brake, so a window wider than the spacing between genuinely separate events
> chains them all into one row: 40 isolated 0.4 ms / 300 pA events spaced 8 ms apart, merged at a
> 10 ms window, became **one** event of dwell 312.40 ms and depth 0.70 pA — accepted, with a blank
> flag, because the depth is a median over a span that is 96% open pore. So every run the window
> proposes is asked a second question: what fraction of the merged span do its fragments actually
> cover? A genuinely shattered event is over 90% occupied; below **50%** the open pore is the
> majority of the row, and the run is left as separate events and reported as a refusal —
> *"Nothing merged: N run(s) within X ms are mostly open pore between their events…"*, or, alongside
> a partial merge, *"…N further run(s) were left alone."* Read that as *this window is too wide for
> this recording*, not as *the merge is broken*. The check applies on `Merge all`, on a region merge,
> on `Merge with next`, and inside detection's own `merge gap`.

### Threshold sweep & controls

Next to **Detect** are three tools:

- **Sweep threshold…** — re-runs detection across a range of `k_in` and shows how the event count
  varies, so you can pick a value on the stable plateau rather than guessing.
- **Null control…** — the false-positive question. **Read all of this before you quote a number
  from it,** because what it prints has changed and older numbers are not salvageable.

  It works by removing your detected events, building an **AAFT surrogate** of what is left (same
  spectrum, phases randomised, so it should contain no events at all), and running the detector over
  it. Anything found is by construction a false positive — *if* the surrogate really is the same
  noise.

  **On real nanopore recordings it is not.** Phase randomisation is a valid null only for a trace
  that is a stationary spectrum plus Gaussian noise, and a real pore is neither: it drifts, and its
  noise is heavy-tailed (event-free kurtosis ≈ 10–11 against 3.0 for Gaussian). The surrogate comes
  back with **more power than the original in every band** — by up to 2.2× on the apo file, 6.4× and
  14.2× on the other two — and the detector duly finds **30–70× more events in it than in the very
  noise it was built from.** The reported rate saturated at **100 %** on a recording with 315
  plainly real events. Fixing the spectrum does not fix it: an IAAFT implementation that drives the
  worst band's error from +0.78 to +0.009 in log makes the count *worse*. The dominant term is the
  open-pore drift, and a surrogate that is a genuine null of a real nanopore recording is a research
  question, not a patch.

  **So the control now checks its own premise and refuses to publish a rate it cannot stand
  behind.** Where it used to print a percentage, the Statistics readout says (the count is your
  recording's; 31 is the apo file's):

  > false-positive rate **NOT estimated** — 31 detection(s) on this recording's own event-free noise
  > (a lower bound, measured with no surrogate).

  with the reason as a tooltip. **`n_event_free` is the number to quote.** It is the same detector
  run on your own trace with its events excised and no surrogate anywhere — a real, conservative
  **lower bound** on your false positives. The **Sweep** dialog stops labelling the surrogate curve
  "false +" when it is not a null, too (on the apo file that curve sits ~88× *above* the real one).

  **But it is one draw from a wide distribution, not a count.** The excised stretches are refilled
  by drawing from your recording's own quiet noise, and that draw moves. Measured before the run was
  seeded: four identical calls on the −600 mV apo recording, same slice, same detector settings,
  gave **52 / 55 / 66 / 52** event-free detections against a `n_real` of 409 that did not move at
  all; a second harness on the same file gave 21 and 42 where an earlier run had given 34.
  **`n_real` was always deterministic; `n_event_free` was not.**

  It is now **seeded from the recording's own bytes and the settings being measured**, so pressing
  the button again reproduces the same integer, and the seed is stamped into the result and reported
  under Statistics: *seed = 1722390092 — pressing "Null control…" again on this recording with these
  settings reproduces these counts exactly.* Seeding it off the bytes rather than off the path
  matters: the same recording copied to three names used to produce three different "reproducible"
  numbers. That makes the count **citable, not meaningful.** Quote it as a lower bound, and do not
  read a change in it as a change in the pore.

  > **Any FDR percentage produced by a build older than this one should be discarded, not
  > rescaled** — including anything already written into a draft. There is no factor that converts
  > it, because the number it came from had no meaning.
- **Recovery map…** — a detection-efficiency map used to correct dwell fits for missed short events.

### One analysis at a time

Only one background analysis runs at a time, across the whole app. Press **Detect** while a
**Batch detect** is running and the status bar says which run is in the way and that nothing was
started — press **Cancel** to stop the batch, or wait. It works in both directions, and it covers
Detect, Run ML, the three surrogate studies, the open-pore probe, a re-measure, an estimator
comparison (§9a) and a batch pass.

The button is left enabled on purpose. A control greyed out by the batch could not tell you it was
the batch — you would be left with a dead button and a guess. The click is allowed to land, and it
is answered by name. Nothing you had running is stopped by a refused click.

Opening a file is *not* blocked by an analysis: the two touch different data.

---

## 7. The event table & characterization

Each detected event is characterized into a row of the event table. Key columns:

| Column | Meaning |
|---|---|
| `dwell_s` | Duration = `(end − start) / fs`. |
| `depth_pA` | Original blockade depth = the **median** signed in-event deviation from the samplewise baseline. Includes the full detected interval, including transitions; this definition is unchanged. |
| `amplitude_pA` | Optional, explicitly selected amplitude; its method, interval, units and audit record are saved separately. See §9. No one estimator is appropriate for every event shape. |
| `frac_depth` | `depth / |baseline|`, i.e. ΔI/I₀. **Values above 1 are a real message, not a scaling error** — see below. |
| `ecd_pA_s` | Equivalent charge deficit (integral of the deviation). |
| `max_dev_pA` | Peak deviation (honest for spikes; *not* used as the depth). |
| `snr` | `depth / noise RMS`. |
| `rise_s`, `fall_s` | 10–90 % entry / 90–10 % exit times. |
| `n_levels` | Resolved sub-states (1 = simple square event). |
| `filter_limited` | The event is within ~2× the filter rise time — its depth is attenuated and dwell biased long. |
| `accepted` | Curation flag (checkbox) — but only half of it is yours; see below. Exports and plots honour it. |

The `accepted` cell is an **AND of two halves**: `~(too long | clipped by the edge of the span |
measured from a non-finite sample)` — a statement about whether the row is a measurement at all —
and your own curation. Unchecking always works, because a person may always reject an event the app
was willing to keep, and it drops the event from the summary count and the trace overlay.
**Checking a row the detector flagged does not.** The box snaps back, nothing is written, and a
line under the table says so and stays there. (Column names keep the `_pA` suffix as a fixed
contract; the values are in the channel's units.)

### "Accept all" does not accept everything, and it tells you what it left

Three detector flags make an event unmeasurable — `clipped_by_boundary`, `dwell>max_dwell_s` and
`non_finite` — and no bulk accept will override them: not the event table's **Accept all**, not the
Event Gallery's **Accept all shown**, not the Event Inspector's batch buttons. A grey line under the
buttons counts what was declined, with a different verb for each of three different things:

- **`3 events kept rejected: flagged by the detector as not measurable`** — the rows were already
  rejected and stay rejected. Nothing moved.
- **`2 events corrected back to rejected`** — those rows were sitting *accepted* beside a rejecting
  flag (reachable from any table written by an older build) and this gesture put them back.
  Something moved: your accepted count went **down** on a click that said "accept".
- **`7 events NOT accepted: this table has no 'flag' column, so the detector's own rejections
  cannot be read — refusing rather than guessing`** — nothing at all was accepted. The app fails
  closed rather than accepting rows it cannot check, because a silence read as success ships a
  curation that never happened.

The line is a statement about the **frame**, not about the last click. Reject one ordinary event
afterwards and the clause stays; it clears only when a new detection replaces the events it names.
The event table, the Event Inspector and the Event Gallery share one ledger, so all three carry the
same sentence — the gallery on its status line, counting tiles: *314 of 317 accepted · 3 tiles kept
rejected*. And remember that **Accept all shown** acts on the gallery's **current filter**, not on
the whole detection unless the filter is *All*.

### `frac_depth > 1` — what it actually means

`frac_depth` is `depth_pA / |baseline_pA|`, a **ratio**, so **the current gain cannot move it**: set
a gain of 1000× and every `frac_depth` in the table is unchanged (the app asserts this). So a table
whose `frac_depth` sits above 1 — the apo `…_0006_` recording's median is **3.1** — is not an
uncalibrated-gain artefact and cannot be fixed in Calibration.

What ΔI/I₀ > 1 says is that the current **magnitude grows** during the event. Two readings, and only
you can tell them apart:

1. **The events really are conductive rather than blocking** — a category your own corpus names in
   its filenames.
2. **The amplifier has a DC offset**, so `baseline_pA` is not the true open-pore current and the
   ratio is measured against the wrong denominator.

Settle it before reading any of those numbers as blockade fractions: check the sign of the excursion
against the open pore, and check the amplifier offset. The re-measure report says the same thing
whenever it produces such rows.

---

## 8. Manual measurement

For events you want to measure by hand (or to check the detector), use the **Manual measurement**
section of the Analysis tab.

1. Zoom to a useful inspection scale with box-zoom or the **− / +** buttons in the
   Scan strip. At event scale the Navigator changes from **Full recording** to a **9× local-context**
   lens, so you can still pan precisely without zooming all the way out.
2. **Double-click the trace at the event**. This activates Measure and places a mask centred at the
   click, clamped fully on screen and no wider than **20% of the visible view**. Alternatively use
   **Measure event (cursors)**, **Snap to view**, or **Snap to selected event**.
3. Drag either circular edge handle to resize the mask, or drag its shaded body to move the whole
   mask; everything recomputes live. The blue band in the Navigator
   mirrors this working selection, while the shaded region shows the visible trace window.
4. Choose the **amplitude method**: `median`, `mean`, `min` (deepest/peak), `max`, or `mode`. Amplitude
   is `baseline − level`, where the baseline is the robust median of the current just *outside* your
   cursors, and *level* is the chosen statistic of the in-cursor samples.
5. The **Live statistics** read out: dwell, amplitude, ΔI/I₀, level, baseline, mean, median, std, min,
   max, RMS, skewness, **kurtosis** (excess), area/ECD, SNR, and n samples.
6. Record the selection in the right place:

   - **Add to list** (or **Ctrl+M**, from any tab) saves an independent manual measurement.
   - If a **source detector table** is selected and the detector missed this event, **Mark selection
     as missed event** (or **Ctrl+Shift+M**, from any tab) adds it to that event table as a reviewed,
     accepted, hand-added candidate. The shortcut stays enabled whenever a recording is selected:
     if no source result exists or Reconciled is selected, it gives a clear refusal telling you to
     run or select a source detector rather than failing silently. A span that overlaps an event
     the table already has is refused too (the status bar names the event): select that event to
     curate it instead, so nothing is counted twice.

7. Continue without losing your scale: press **Scan ▶** (or **Shift+→**) to move forward by 90% of
   the current view. The 10% overlap keeps boundary events visible, and active Measure cursors move
   with the view. Refine the cursors around the next event and repeat. **Scan ◀** / **Shift+←** moves
   back; **+ / −** or **Shift+↑ / Shift+↓** adjusts the working zoom incrementally.

**Export CSV…** saves the manual-measurement list (the full 32-column measurement schema, provenance
stamps included); **Remove selected** drops a row, after asking.

**A saved row records what it was measured under.** *Add to list* stamps each row with the channel's
`units`, the calibration `cal_gain` and `cal_offset` (the same fingerprint a detection stamps), and
`t0_s`, the recording's instrument-clock origin. Edit the calibration afterwards and the affected
rows are **marked, never rescaled** — the saved numbers are the record of what was measured, exactly
as the analysis panel treats a stale detection.

The mark has **three** states, not two:

| Mark | Meaning |
|---|---|
| *(nothing)* | Checked against the live calibration, and current. |
| **⚠** amber | Measured under a calibration that has since changed. Double-click the row to return to the span, then **Add to list** for a row under the current calibration. |
| **?** | No verdict. The row's recording is not open (or the row predates the stamp columns), so the app **cannot** compare. It is *not* known to match. |

The `?` is the one worth knowing about: without it, "nobody can check this" would be drawn exactly
like "checked, and current." And a row that was already **⚠** when you closed its recording **keeps**
its ⚠, with a tooltip saying the check can no longer be repeated — the numbers did not change when
the document closed, so neither does what was true about them.

One more thing worth knowing: the live readout may be a strided preview on a very large selection
(and says *approx* when it is), but *Add to list* re-measures the span over **every** sample before
it stores a row. Nothing approximate is ever saved.

Manual measurements are kept as their own list, separate from the auto-detected events — but
**Edit ▸ Undo reaches them too** (§9b): deleting rows here is recoverable.

---

## 9. The event inspector

**Analysis ▸ Event inspector…** opens a window for stepping through detected events one at a time.

**Its own window or docked beside the trace.** It opens as its own window. **Dock beside trace**
(bottom-right of its toolbar), or **Window ▸ Dock Event inspector beside the trace**, puts it in
the main window instead, as a tab beside the Inspector panel right of the trace, so the trace stays
in view while you curate; **Own window** puts it back. The choice is remembered. Docked, it can be
dragged to the left column (not above or below the trace), and its tab's ✕ closes it — Ctrl+I
brings it back. A click anywhere in the docked inspector, or on its tab while you were typing in the
Inspector panel, gives it the keyboard; when its tab goes behind the Inspector panel the keyboard
moves to the trace, and Tab skips whichever of the two is behind, so a key never curates an event
or edits a setting that is not on screen. The toolbar is two rows in a wide window and five short
rows in the docked column (or a narrow window); the buttons keep their order. In a short column the
part under the toolbar scrolls rather than pushing the main window past the screen.

Open the Inspector before choosing the target event, or reselect that event after opening it.
A new Inspector can start at the first event regardless of an earlier table selection. The
Event spinner is a position in the filtered subset, not `event_id`; verify the ID and sample bounds.
For the current −700mV gallery table, position147 in All corresponds to ID146.

- **Waveform.** The current around the event in the recorded **measurement domain**, replaying
  conditioning and low-pass filtering where recorded. The event region is shaded; the baseline
  and equivalent amplitude level use that same domain. The readout distinguishes original median
  depth from an optional selected amplitude, alongside dwell, ΔI/I₀, SNR, rise/fall, level count,
  and filter-limited flag. Normalized traces are labelled dimensionless, not pA.
- **Details.** **Details ▾ / ▸**, under the waveform, shows or hides the readout of this event's
  numbers. Hidden, the waveform takes the room; in a narrow column its title keeps depth and dI/I0
  whole and drops the event number, then the dwell, before them (hover the title for all of it).
  Shown, drag the bar between them to share the height; docked, the readout leads with median depth
  and dI/I0. It starts hidden when docked and shown in a window, and each is remembered. The button
  and the waveform's top-left never move when it is toggled. When the numbers are stale the button
  reads **⚠ Details** and its tooltip carries the sentence; the readout's first line is that
  sentence as a one-line chip (hover it for all of it), and each stale number keeps its own ⚠. On a
  short waveform the current axis is labelled **I** rather than "Measurement current".
- **Navigate.** ◀ / ▶, an "Event *i* of *N*" index, and a **Filter** — All, Accepted only,
  Rejected only, Unreviewed, and (once a label palette exists or any event carries a label)
  Unlabelled and one **Labelled: key=value** entry per label. Keyboard: **←/→** move, **A/R**
  accept/reject, **Space** toggles. See *A curation pass* below for Unreviewed, **Advance after
  decision** and **First unreviewed**.
- **Context.** A `Context:` combo sets how much recording is *loaded* either side of the event —
  **± 3 ms** (the default), **± 30 ms**, **± 300 ms** or **Whole trace** — and the choice persists as
  you step, because it is a working preference rather than a per-event one. Below ±3 ms was once the
  hard-coded pad and therefore a ceiling: panning inside a 6 ms slice cannot answer *"was the pore
  quiet before this?"*, only loading more samples can.
- **Show in trace** (`T`) centres the main trace on the event you are looking at — the same jump as
  right-click ▸ **Go to event in trace** on its tile in the gallery — and **Gallery…** opens the
  contact sheet. The three event surfaces all link to each other now; none of them is a dead end.
- **Label.** While an event is on screen, the bar under the waveform holds the gallery's
  Label idiom for it (it stays there when Details is hidden): pick a palette entry, **Apply** writes
  it, **Clear** removes that entry's *key* (other labels and flags stay). Both record that a person
  reviewed the event, both are one Undo step, and neither moves on. A readout line too long for the
  window ends in "…" (hover the readout for all of it) rather than widening the window.
- **Curate.** Accepting/rejecting writes to the shared event table, so it flows to the table
  checkbox, the trace span overlay, the summary counts, the plots, and exports — all at once, in both
  directions. Navigating also zooms the main trace to the event.
- **Save without leaving the window.** The status bar names the current project and says whether it
  has unsaved changes. Press **Save Project** (use its arrow for **Save Project As…**), **Ctrl/⌘+S**,
  or **Ctrl/⌘+Shift+S**. These controls use the same guarded project writer as the main File menu;
  cancelling or failing a save leaves the unsaved warning in place. Docked, the main window's title
  carries the unsaved mark and the keys are File ▸ Save Project's own — the same writer.
- **Undo without leaving the window.** **Ctrl/⌘+Z** and **Ctrl/⌘+Shift+Z** in the inspector run the
  main window's Edit ▸ Undo / Redo (there is one undo history; docked, they are Edit ▸ Undo / Redo
  themselves). After an undo or redo the inspector
  keeps its filter and the event on screen, and the gallery its selection, as long as the restored
  table holds the same events; an undo that renumbers events (a merge) starts again from the first.

### A curation pass: Unreviewed, Advance after decision and progress

Both the inspector and the gallery offer the same tools for working through a detection.

- **Unreviewed** (Filter / Show) holds the events no person has ruled on yet — no accept, reject or
  label. Events with **no review record** at all (detections saved before review tracking, or from a
  detector that does not stamp it) count as unreviewed here; the progress tooltip says how many.
- **First unreviewed** (**Ctrl/⌘+U**, or the button) goes to the first unreviewed event of the
  current filter — in the gallery, the first such tile in the current sort, turning the page. If the
  filter has none it says how many remain elsewhere.
- **Advance after decision** (a tick box, off by default and shared by both windows) makes the
  *keys* — A, R, Space and the label keys — move to the next event of the current filter once the
  decision is written. In the gallery it moves only a **one-tile** selection, turning the page when
  needed; a swept selection never moves. Buttons, the Accepted box and the Label row never move on.
  A refused accept stays on the event (in the gallery under Unreviewed, where the refusal's review
  stamp takes the tile out, the selection goes on to the next tile instead of to nothing). At the
  end of the filter the window stays and the progress line says so (*End of Unreviewed · 12
  earlier*). With the option on, **Ctrl/⌘+Z** in the window also goes back to the event the last
  advance moved away from, when the undo changed that event. Decision keys never auto-repeat, so
  holding R cannot reject a run of events.
- **The progress line** — beside **First unreviewed** in each window's status bar — reads
  *Reviewed 312 / 870 · accepted 801 · labels: type=collision 40,
  type=translocation 260*. It counts the whole detection, not the current filter; hover it for the
  breakdown (unknown review records, labels outside the palette, merge conflicts). It shares the
  status bar with the project label by what each says: when both are long, the file name gives way
  first and the save state stays readable.

| Key | Event inspector | Event gallery |
|---|---|---|
| ← / → | Previous / next event of the filter | Native: move the focused tile on the page. After a decision takes the focused tile out of the Show subset, nothing is selected and the first ← / → selects the tile beside it |
| A / R / Space | Accept / reject / toggle the event on screen (moves on with Advance) | Same on every selected tile (moves on only for one tile with Advance) |
| 1–9, custom label keys | Label the event on screen (moves on with Advance) | Label every selected tile (moves on only for one tile with Advance) |
| T | Show the event in the main trace | — |
| Ctrl/⌘+U | First unreviewed event of the filter | First unreviewed tile of the Show subset, in sort order |
| Ctrl/⌘+Z | Undo (main window's history) | Undo (main window's history) |
| Ctrl/⌘+Shift+Z | Redo | — |
| PgDn / PgUp, Home / End | — | Next / previous page, first / last page |
| Ctrl/⌘+A, Esc, Ctrl/⌘+I | — | Select all on the page / none / inverse |
| Ctrl/⌘+S, Ctrl/⌘+Shift+S | Save / Save As | Save / Save As |

### Choosing an amplitude and editing the red level

An amplitude is a measurement choice, not a universal description of every analyte. The original
`depth_pA` remains the median over the complete detected interval. The **Amplitude** controls in
the Curate event inspector save a separate `amplitude_pA`; they never overwrite `depth_pA`,
ΔI/I₀ (`frac_depth`), SNR, event boundaries, acceptance decisions or fitted states.

1. Open the event inspector and navigate to the event. Check the waveform, event boundaries,
   measurement units and baseline reference before choosing an estimator.
2. Choose the method from **Amplitude**:

   | Method | What it measures |
   |---|---|
   | Median depth (whole event) | Median of signed, samplewise baseline deviations over the full event. |
   | Mean depth (whole event) | Mean of those deviations; includes transitions and all occupied states. |
   | Peak excursion (whole event) | Largest signed deviation in the interval. Sensitive to noise, sampling and filtering; not automatically a better estimate. |
   | Histogram mode (whole event) | Centre of the most populated deviation bin. Bin-dependent; may represent the most occupied state, not the deepest state. |
   | Median / Mean of selected interval or state | The chosen statistic over only the explicit interval. No transition trimming is performed automatically. |
   | Manual level | A user-specified equivalent current level, converted to amplitude against the whole-event mean baseline. This is an annotated choice, not a fitted result. |

3. For a selected interval, type the absolute sample bounds or drag the **purple** interval
   handles. Start is inclusive and stop is exclusive. These controls do not move the event mask.
   **Use fitted state** copies the bounds of an already fitted state; it does not rerun a fit or
   use a fitted height as the amplitude. Adjusting those bounds makes it an ordinary selected
   interval again.
4. Press **Apply** to save the choice for this event. The readout and red level update together.
   For manual adjustment, check **Edit red level**, then drag the red horizontal line and release
   to save, or type an **Equivalent level** and press **Apply**. Moving the line across the
   baseline in the wrong event-polarity direction is refused. Switching methods or events locks
   dragging again; choosing Manual level alone does not unlock it.
5. Use **Reset amplitude** to clear only the optional choice and return the line to the stored
   median-depth equivalent. **Edit ▸ Undo (Ctrl/⌘Z)** reverses an amplitude change or reset.

For event polarity `p` (+1 for upward, −1 for downward events), the horizontal line is
`mean baseline over the measurement interval + p × selected amplitude`. With baseline drift,
this is an **equivalent horizontal reference**, not a fitted plateau or the pointwise baseline.
For legacy results without reproducible baseline provenance, the saved event baseline is used
as an explicitly labelled constant reference. If the measurement signal cannot be reconstructed,
quantitative line editing is disabled; it does not silently measure another signal.

**Apply to all…** is available for whole-event median, mean, peak and mode. Its confirmation
states that it replaces existing optional choices, including manual choices and rejected rows.
All rows must validate before any are written, and one Undo restores the previous choices.

Save the project to retain the choices. Event-table CSVs also carry `amplitude_method`, selected
sample bounds, `amplitude_units`, the equivalent level and baseline, `amplitude_status`, and
`amplitude_provenance_json` (processing, baseline and calibration references). The `_pA` suffix
is a column identifier; consult `amplitude_units` for normalized measurements. Mask resizing,
merging and re-measurement that change the reference invalidate affected optional numbers and
retain their audit history as **stale**. Reapply or reset them deliberately. Re-detection replaces
the event table and warns about losing these choices; Undo can restore the prior table.

In plotting controls, choose **Selected amplitude** / `amplitude_pA` to use the new measurement.
Existing depth plots continue to use median `depth_pA`. Missing/stale optional amplitudes do not
fall back to median depth. Warnings flag incomplete coverage and mixed methods, intervals, units
or processing references—even within one recording. Selected intervals or manual levels do not
establish that every event represents the same physical state; assess that before pooling.
None of these methods deconvolves filtering or recovers an unresolved event's unfiltered height.
Density plot-data exports include the per-event amplitude audit while the live source is being
plotted. A restored frozen amplitude map exports its saved bins and units without attaching
possibly changed live event choices; use the saved event-table CSV for the per-event audit.

---

## 9a. When the open-pore baseline changes

Every depth in the table is measured against an **open-pore baseline** — the current the pore
carries with nothing in it — and which estimator produced that baseline is stamped on the result,
next to the calibration stamp.

If a stored detection's stamp does not match the estimator now set in **Preprocess ▸ Open-pore
baseline** (or if the table predates the stamp entirely), the Results panel's status line turns
amber — *Depths stale — estimator changed* (or *Depths stale — predates the current estimator*) —
with **Re-measure depths** and **Details**, which shows the full sentence naming the old
estimator; and the affected columns — `baseline_pA`, `depth_pA`, `frac_depth`,
`ecd_pA_s`, `mean_dev_pA`, `max_dev_pA`, `snr`, `rise_s`, `fall_s` — carry a ⚠ in the table header
and are shown in amber. `rise_s` and `fall_s` are on that list because 10 %/90 % are fractions of
each event's *own* depth, so a changed depth moves them; they were once documented as unaffected,
and that was wrong. **Nothing is re-measured.** The values on screen are exactly the values that
were stored.

- **Analysis ▸ Compare baseline estimators…** measures, on *this* recording, what a re-measure
  would change: old vs new `depth_pA` per event against the y = x line, the ratio distribution, and
  how many events have a stamped baseline that moves more than 10 pA. On a recording without the
  pathology this shows ~1.0× everywhere. Opening it changes nothing.
- **Re-measure depths** (on that line, or from inside that report) rewrites the amplitude columns
  against the configured estimator. It does **not** re-detect: every event span keeps its
  boundaries, and every accept/reject and hand-written flag survives — including ones you make
  while it is running. A calibration change is different: its line (*Depths stale — calibration
  changed*) offers **Re-run**, which re-detects, and the current curation does not survive that.
  When the offered action cannot run (for example the channel is overridden to voltage), the
  button is greyed and the line goes on *· Re-run unavailable: …* (on a narrow panel it gives up
  the verdict first, never the start of the reason); the whole reason is in **Details**.

Optional selected amplitudes are not silently reinterpreted under the new baseline: affected
values become stale and must be reapplied. If an amplitude choice is edited while re-measurement
is running, the late result is refused rather than overwriting that edit.

Old projects always open. They are marked, badged, and otherwise fully readable, plottable and
exportable.

### Re-detect, do not re-measure — for anything that uses the span

This is the catch in the paragraph above, and on the reference recording it is **larger than the
depth correction it is fixing.**

A re-measure keeps every event boundary exactly where it was. That is what protects your curation —
and it is also the limitation, because the old estimator did not only mis-stamp the depths. It fed
the detector's threshold too, and a baseline that dives into each blockade **rises and falls with the
event**, so the Schmitt trigger re-crosses inside it and one long blockade is emitted as a string of
short fragments. Detected fresh on the apo `…_0006_` recording, identical detector settings, one
estimator swapped:

| estimator | events | median dwell |
|---|---|---|
| block median, 10 ms (the old default) | 1641 | 0.44 ms |
| block median, 200 ms | 327 | 17.48 ms |
| masked median, 0.5 / 0.1 s (the app default) | 315 | 17.56 ms |
| masked median, 2.0 / 0.5 s (what the probe proposes for this file) | 275 | 18.18 ms |

The event count falls **5×** and the median dwell rises **40×**. Neither is a mask or a merge
artefact: the 200 ms block median uses no event mask and no dilation at all and already gives 327
events at 17.48 ms. Total blocked time goes 1.80 s → 4.60 s.

So a re-measured table has **correct depths measured against dwell times the old estimator got
wrong.** For a depth histogram that is fine. For **a dwell-time fit, an event rate, `ecd_pA_s`, or a
depth-vs-dwell scatter** — anything that integrates or bins over the span — **re-detect**, and expect
a different number of events. Since a re-detection is now recoverable with a single Ctrl/⌘Z (§9b),
this is no longer the costly choice it used to be.

`start_idx`, `end_idx`, `t_start_s` and `dwell_s` are bit-identical before and after a re-measure.
That is exactly why. And for the same reason, **never pool a re-measured table's dwells with a fresh
detection's** — on this file they differ by 40×. Depths measured under different baseline
configurations are not poolable either; carry the configuration with the number, which is what the
`baseline_fingerprint` in `provenance` and in every export's `.meta.json` sidecar is for.

---

## 9b. Undoing a change to the event table

**Edit ▸ Undo (Ctrl/⌘Z)** walks the event table on screen back through every change made to it in
this session, and the menu names the step it will undo — *"Undo Re-run detection"*, *"Undo Merge
events"*, *"Undo Resize event mask"*, *"Undo Edit event amplitude"*, *"Undo Accept all events"*.
**Redo** is Ctrl+Shift+Z / ⇧⌘Z.
Everything that can change an event table is covered: a re-detection, a merge, an accept/reject, a
batch Accept-all / Reject-all, a mask resize in the Event Inspector, a hand-added missed event and
a re-measure, and amplitude choices, red-level edits and resets. An undo restores the table
**exactly** — the accept/reject decisions, optional amplitude audit fields, any flag token
you typed, and the hand-added markers.

**And any edit that reaches the table by a path nobody announced.** That is deliberate, and it is
what makes the guarantee worth relying on: an edit that changed the table without telling the undo
stack is committed anyway, the next time anything asks — when you open the **Edit** menu, when you
**save the project**, when you press **Ctrl+Z**, or when you **quit**. It lands with a derived label
("Edit events") instead of a named one. A forgotten door costs you a word, never a row — and
because a quit and a save are two of those four moments, an unannounced edit also reaches the dirty
flag and therefore the save prompt.

Four things to know:

- **The history is per event table.** Undo acts on the detection you are looking at, so Ctrl+Z can
  never silently repair a recording that is not on screen. When there is history but none for this
  table, the menu says so instead of going quiet.
- **Re-running detection replaces the table, and now says so.** The status bar reports what the
  replaced table held ("6 events, 2 rejected by hand and 1 hand-added") and points at Undo.
  Curation is still not carried into a fresh detection — undo is what gets it back. The counts are
  the ones **you** made: a rejection the detector itself made (too long, clipped by the edge of the
  span, measured from a non-finite sample) is re-derived by the next run and is not quoted, because
  nothing about it is lost. This holds when you press Detect a second time with the settings
  unchanged, which is the commonest case and the one that used to pass in silence.
- **Closing a recording is the one remaining one-way door.** A closed document takes its results
  *and its history* with it, so closing a recording that carries detected events asks first and names
  them. For a **segment** it says the thing that matters: a segment is held in memory, not on disk,
  so there is nothing to reopen and nothing to detect again. Save the project first. **Opening a
  project** clears the session and therefore the stack, for the same reason. Nothing else in the
  app is one-way: deleting a hand measurement, which used to be the second door, is now a step like
  any other — see below.
- **Hand measurements are covered too, by the same Ctrl+Z.** Deleting rows from the
  Manual-measurement list (§8) is now a step like any other — *"Undo Delete 3 measurements"*. There
  is one Undo, not two: when both the event table and the measurement list have history, a press
  takes back **the most recent thing you did**, then the one before it. The confirmation before
  deleting stays, because knowing there is something to recover is a different job from being able
  to recover it.

The history keeps up to 30 steps **per event table**, within a 128 MiB budget for the whole
session, and is cleared when the recording closes or a project is opened. When the ceiling evicts
steps, the Edit menu says so by name (*"Undo history for “&lt;name&gt;” is at its 30-step ceiling — N of
your steps have been evicted"*) instead of showing the never-edited wording. The per-table depth
matters when several recordings are open: curating one recording no longer uses up another's
history, and undoing on one no longer ends the other's redo branch. If a step is ever dropped
because the memory budget bound first, the app says so rather than losing it silently.

**Keyboard curation on the event table.** Click a row, then `j` / `k` move down / up, `a` accepts,
`r` rejects and `f` marks the row with your own flag. `f` **adds** your mark to whatever reasons the
row already carries (`clipped_by_boundary`, `dwell>max_dwell_s`, …) rather than replacing them, and
your mark survives a merge, a mask resize and a hand-add.

---

## 9c. Reconciling detectors & building training labels

NanoScope can be the annotation and export front end for a supervised-learning dataset. It keeps
the detector proposals, your positive and negative review decisions, structured event labels, and
the provenance needed to reproduce the regions. It does **not** train a model inside the app; use
the exported table and sidecar in the StudentModel notebook or another external training pipeline.

### Build a reconciled candidate table

1. On one recording and current channel, normally run **at least two detector types** — for example
   Schmitt, CUSUM, or ML. Agreement (`support > 1`) requires the **same analysed range**;
   a union (`support = 1`) can combine compatible source tables with different fully recorded
   coverage. When coverage is unknown, non-nominal, inconsistent, or a hand-added event falls
   outside a detector's analysed span, NanoScope keeps the rows but suppresses the event rate rather
   than guessing at an exposure time. Reconciliation does not treat two
   runs of one detector as two independent votes. There is one useful single-source exception: a
   detector table containing both automatic candidates and `added=True` missed events is eligible,
   so a manual mark can converge with an overlapping detector candidate.
2. Use the **Event table** selector above the table to choose a **source detector table**. If a
   detector missed a real event, turn on **Analysis ▸ Measure event**, bracket it on the trace, and
   press **Mark selection as missed event**. Do this *before* reconciliation. A Reconciled table is
   a derived, curation-only view: add, merge, resize/revert a mask, or re-measure a baseline in a
   source table, then reconcile again. **Add to list / Ctrl+M** writes to the separate manual
   measurement list; it is not a reconciliation source. Use **Mark selection as missed event /
   Ctrl+Shift+M** in a detector table when the region is intended to become a reviewed
   detector-training candidate. The shortcut remains callable with a recording selected so it can
   explicitly tell you when there is no source detector or Reconciled is the current table.
3. Choose **Analysis ▸ Reconcile detector results…** and tick the source detection(s). The command
   is available with either two source detector tables, or one table containing both automatic and
   hand-added candidates. Multiple sources must agree on calibration, baseline measurement
   contract, units, sample rate, event polarity, and (when agreement is required) analysed range;
   Preview names a mismatch instead of combining unlike measurements.
4. Set **Minimum temporal overlap**. Two candidates may join when their shared time is at least
   this percentage of the **shorter** candidate. The 50% default tolerates modest boundary
   disagreement without joining events that merely touch.
5. Set **Required detector support**:

   - `1` is a **union**: keep a candidate found by any selected detector.
   - A value between 1 and the number of sources is **agreement**: require at least that many
     independent detector tables.
   - The number of selected sources is an **intersection**: every selected detector must support
     the candidate.

   With the single mixed source/manual exception, support stays at `1`; it is deduplication, not a
   claim of cross-detector agreement.

6. Leave **Keep manual-only events** on when building supervised-learning labels. It retains an
   event you added by hand even if no automatic detector found it. Turn it off only when the
   dataset intentionally requires automatic support for every row.
7. Press **Preview**. Check the input, overlap-group, output, converged, filtered, manual-only, and
   unresolved-label-conflict counts. Nothing is stored until you press **Create reconciled table**.

The result is stored as a separate **Reconciled** event table. The source detector tables and all
of their curation remain unchanged. Use the **Event table** selector to switch among them; switching
changes the table being viewed, not the data that are stored. Reconciliation keeps the whole row
from one canonical contributor rather than averaging scientifically incompatible measurements,
and records every contributing source in its provenance. Because canonical rows may come from
different detector conditioning chains, baseline re-measurement, event insertion, and merging are
done in a source table followed by a new reconciliation; mask resize/revert controls are blocked on
the derived table for the same reason. Fit-substate-level and Event-subpopulations analyses also
rebuild events from one waveform, so run those on a source detector table; plots and distribution
fits over already-stored Reconciled columns remain available.

The Reconciled table is a provenance-bearing snapshot. If a selected source is later rerun, edited,
removed, or otherwise no longer matches its stored fingerprint, a stale-lineage line in the
Results panel (*Source table changed*, with **Details**) keeps the snapshot viewable and exportable
but tells you to inspect the source tables and reconcile again.

### Review and label the result

Open **Analysis ▸ Event gallery…** for bulk work or **Analysis ▸ Event inspector…** for one event at
a time. Accept/reject decisions write to the selected Reconciled table. Define structured labels in
**Analysis ▸ Event labels…** and apply its `key=value` entries in the gallery or inspector; applying
a label records that the row was human-reviewed. The label palette, table, and review state are
saved in the project. Both windows keep the project name and Saved/Unsaved state visible at the
bottom, and both accept **Ctrl/⌘+S** or **Ctrl/⌘+Shift+S**, so a long labeling pass never requires
returning to the main window just to save.

If contributors carry different values for the same label key, Preview reports an **Unresolved
label conflict** instead of guessing. The row's `consensus_label_conflicts` column names the affected
key, while `flag` preserves the competing values with a `|` marker (for example
`molecule=apo|holo`). Inspect the row and apply the correct value. The training-export warning
defaults to **Cancel**; resolve the conflict before training unless an external cleanup step is
deliberate.

If overlapping contributors carry contradictory human accept/reject decisions, NanoScope likewise
does not choose a class silently. The row is marked in `consensus_curation_conflict` as
`accepted|rejected` and is excluded from **Reviewed decisions** until you explicitly press Accept or
Reject in the table, gallery, or inspector. That explicit action clears only the decision conflict;
label conflicts remain until the affected structured label is corrected.

Digits `1`–`9` always select palette positions. A custom label may also use any unique, single
character hotkey except the reserved curation/navigation keys (`A`, `R`, `T`, Space, case
insensitive); the same hotkey works in the gallery and inspector. The label editor highlights an
invalid or duplicate hotkey and will not save it.

### Export the labelled examples

Choose **Analysis ▸ Export events for training…**, then choose exactly which population to write:

- **Reviewed decisions (recommended)** writes human-reviewed positives **and negatives**. It
  excludes untouched detector candidates and purely automatic rejections, making it the safest
  starting point for supervised learning.
- **Accepted candidates** writes accepted rows, including rows nobody has reviewed yet. This is a
  candidate set, **not curated truth**.
- **All raw candidates** writes every row, including automatic detector rejects. Use it for an
  explicit downstream triage or cleanup workflow, not as ready-made labels.

CSV and Parquet exports include event regions, the accepted/added state, dynamic columns rebuilt
from the structured labels, and a `.meta.json` provenance sidecar. For a Reconciled table that
sidecar also carries the reconciliation membership and matching contract. Keep each table beside
its sidecar.

Save a `.nanoscope` project before closing. A project preserves the source detections, the separate
Reconciled table, curation, labels, and label palette, so the labeling session can be resumed.
NanoScope is not an annotation database and currently has no dataset-split manager, annotation
import/round-trip, reviewer identity or timestamps, edit audit log, or inter-annotator workflow.
Those steps need an external dataset/versioning pipeline.

---

## 10. Population plots & machine learning

**Analysis ▸ Distributions…** opens the **Distributions** page of the Analysis Explorer over the per-event
tables the app holds:
**one source per (recording, channel, detector result) with at least one row** — Schmitt, CUSUM, ML,
and Reconciled tables are offered separately, segments included — plus your **manual measurements**.
**Use accepted events only** is ticked by default; untick it to plot the rejected rows alongside the
kept ones.

The page opens with **Data** showing source selection and the plot recipe. The fixed toolbar
keeps **Plot / Apply** and **Export ▾** reachable. Use the **Section** dropdown for **Analysis**
(plot/fit controls, PCA, clustering and axis limits), **Appearance**, and **Export**; only the chosen
section is visible. These are control sections within Distributions, not separate Explorer tabs.
A short **Result** and first actionable
warning stay under the canvas. Expand **Details — methods and provenance** when you need the full
audit trail.

### First use: sources, controls, then commit

1. Choose **Section → Data**. Check the source rows you intend to analyze and verify their
   recording, channel and detector identity. A highlighted row is only the row selected for
   editing its name or source colour; its checkbox decides whether its events enter the plot.
2. Choose **Histogram**, **Scatter**, **KDE (density)** or **PCA** and the appropriate measurement
   columns. Check **Use accepted events only**, units and normalization deliberately.
3. Press **Plot / Apply** in the fixed toolbar. The local **Plot** button and the **Apply fit**
   buttons perform the same commit: all pending source, plot-type, column and fit proposals are
   applied together. Do not assume Apply fit commits only the fit while leaving a proposed source
   change unapplied. A **●** indicates a pending proposal; the displayed picture may still be the
   previous result.
4. Read **Result** and expand **Details — methods and provenance**. Only then choose
   **Section → Appearance** to style the figure, and export or save a named snapshot.

A live Distributions page normally draws when first populated. Its **Refresh** button in Data,
and the Explorer's **Refresh current**, repull stored event/manual sources and can redraw; they
are not just list updates. Existing checked sources are retained where possible, but previous
PCA/clustering results are cleared when new source tables are supplied. Refreshing a saved
analytical snapshot explicitly returns that view to current inputs. Save a Figure Library copy
first if you need to retain the old picture and numbers.

Not every numerical control waits for Plot: bin count, normalization and some log/geometry
controls redraw an already active plot. Source, plot-type, column and fit proposals use the
explicit commit described above. Appearance-only edits restyle the existing result without
running those analyses or applying pending proposals. After changing any numerical control,
inspect the result and its state before assuming the canvas is unchanged.

### Comparing two of anything

The source picker is a **checkable list**, not a single choice. Tick two or more and the histogram
and scatter **overlay** them, each in its own colour-blind-safe colour, with a **legend** naming
each. That is the two-pore comparison and the two-segment comparison; a segment named identically
to its parent still gets its own row, because sources are identified by their stable project key
rather than by their display name.

Three behaviors help organize the comparison; none establishes scientific comparability:

- **Shared bins + normalization.** All drawn sources use ONE set of bin edges, and the y quantity is
  selectable — **count**, **density** or **probability per bin**. It switches to
  <!-- Renamed from "density (area = 1)" by S6-03: the heights are per the column's STORED
       unit (per second for a dwell) while the x ticks follow the Time-unit combo, so the
       drawn area was 1000 on the default ms axis and 23.3 under the shipped log-x dwell
       template, where no constant factor can repair it. The y axis now names the unit —
       `density (1/s)`, `density (1/pA)` — and Details says where the drawn area is not 1. -->
  *density* as soon as a second source is shown, unless you have already selected a normalization.
  This removes the simple event-count scaling that would otherwise differ by 25-fold between
  5,000-event and 200-event recordings of identical shape. It does not correct calibration,
  detection efficiency, acquisition duration or sampling bias. Density is counts divided by
  `n × bin width` in the column's stored unit; its integral is one in that unit, not necessarily
  on rescaled millisecond ticks or a logarithmic display. Check the printed y-axis units.
- **One fit per source.** With a fit enabled each source is fit **separately, using its own
  detector's resolution** — its own `min_dwell_s`, its own filter-limited exclusion, its own
  right-truncation state. Two sources are never pooled into one number, and one recording's
  detection floor is never applied to another's dwell times. Every τ (with n) is named by source in
  the legend and in **Details**; if the sources were detected at different resolutions, the window
  says so.
- **Colour has one owner.** With more than one source drawn, colour means **source**, and the
  cluster split is disabled (and says why). With a single source, colour means **cluster**.

> **Two recordings on one axis are not automatically comparable.** The app checks tracked
> conditions such as merge state, baseline/reference settings, signal domain and current
> calibration for the measurements being drawn. Read the short warning below the canvas, then
> expand **Details — methods and provenance** for the full comparison verdict. Warnings are
> deliberately not printed in the plot title or artwork. Exporting a clean picture does not
> resolve those warnings; retain the methods and caveats in your figure legend, project or bundle.
>
> Histogram density/probability normalization only changes the count scale. Fractional depth
> `depth_pA / |baseline_pA|` can cancel a common multiplicative gain applied consistently to depth
> and reference, but it does not rescue an incorrect open-pore reference, inconsistent offsets,
> different signal domains or detection/curation bias. Do not treat the app's suggested alternative
> axis as a calibration guarantee. In the historical apo example, re-detecting the −700 mV
> recording using the −600 mV recording's baseline windows moved median fractional depth by
> 11.3%; that sensitivity is a reason to inspect each recording, not to share its settings blindly.

### Plot types

- **Histogram** — any numeric column, adjustable bins, linear/log x, linear/log y, a **Normalize**
  choice (`count` / `density` / `probability per bin`), a **Colour by** choice
  (`None` / `Cluster`), and an optional **fit overlay**. Five families: **exponential** (the classic
  escape model — monotonic linear-time density, with no interior mode), **gamma (peaked)** and **inverse-gaussian (first
  passage)** (densities *with* a mode, for a dwell histogram that peaks away from zero),
  **exponential + gamma (bimodal)**, and **gaussian-mixture**. The four dwell families are all
  truncated at the detector's `min_dwell_s`; compare BIC only over the same fitted observations and range.
  A peak in log-spaced-bin counts alone does not rule out an exponential. Choosing a family is a **proposal**: press
  **Apply fit** beside the combo to draw it (the button carries a ● until you do). It used to fit and
  draw the moment you picked, so walking the list merely to read the options ran five maximisations,
  one of them a nine-second optimisation. Beside the
  family: **Fit components** 1–4 with an **Auto (BIC)** option (exponential and gaussian-mixture
  only — the other three have a fixed parameterisation), a **Fit range** min/max in the axis unit
  shown — a *truncation*, not a crop: the likelihood is renormalised to the range, so BIC is only
  comparable between fits over the same range — and **include filter-limited events**, off by
  default, which excludes events whose dwell is within ~2× the filter rise time (depth attenuated,
  dwell biased long) and reports the excluded count in **Details**.
- **Scatter** — any X vs any Y (e.g. amplitude vs dwell), per-axis log, colour-by-cluster or
  colour-by-source, and an optional **best-fit line** per source: **linear** (`y = m·x + b`) or
  **power law (log-log)** (`y = a·x^b`), with **R²** and `n`. The same confirmation applies —
  pick the family, then press **Apply fit**. The printed equation is converted to
  the **axes' own display units** and names them — `y = 2.5·x + 5   (x in ms, y in pA)` — so
  plugging a value read off the axis into it gives the right answer; the fit itself still runs on
  the stored seconds. Two degenerate cases are called out rather than printed bare: a fit through
  two points reports `R² = 1.000` as arithmetic and not evidence (a line has two free parameters, so
  there are no residual degrees of freedom left), and a flat response reads
  `R² undefined (y does not vary)` instead of `nan`. In a log-log fit, points with `x ≤ 0`, `y ≤ 0`
  or a non-finite value have no logarithm; they are excluded and the count is said in **Details**.
  Click a point to select that exact source row in the event table and focus its span on the trace.
  Ctrl, Command, or Shift-click adds/removes points without losing the multi-row selection. The
  handoff preserves the plotted recording and detector identity, so two detector results from the
  same channel cannot send you to one another's table. Manual-measurement points remain visible but
  correctly report that they are not detector-table rows.
- **KDE (density)** — 2-D kernel-density contours of any two numeric columns; one contour set per
  ticked source, in that source's colour, on shared axes. It is a **plot type here beside the others**,
  with the same source overlay, styling and **Export current figure…** as the rest of the window — not a separate
  surface. It defaults to the signature axes, `dwell_s` (log x) against `frac_depth`, and a
  **Contours** spinbox sets the lines per source. Each source is contoured against its **own** peak
  density, so a rare population keeps its shape beside an abundant one. It needs at least 5 usable
  points per source; anything dropped is disclosed in **Details**, along with each source's bandwidth
  factor. (**Analysis ▸ Export density figure…** is still the separate publication export — the same
  view with marginal distributions, written as PNG + PDF + a CSV of the plotted points.)
- **PCA** — projects a chosen feature set to two components, coloured by cluster (with a legend),
  and shows the explained variance. Its points have the same click-to-event handoff as Scatter,
  including the exact kept-row mapping after non-finite rows are removed.

### Clustering

Pick a **feature set** (multi-select of numeric columns), choose **GMM** or **k-means**, set `k`
(fixed, or **Auto (BIC)** for GMM), and **Run clustering**. The scatter, the PCA **and the
histogram** recolour by cluster — the histogram draws one curve per cluster over shared bins, so you
can finally see the two populations' dwell distributions apart — and a summary shows the cluster
count, sizes, and per-feature means. This is the standard "how many populations are in my data"
tool. Clustering runs on the **first ticked** source.

### Styling & axes

- **Time unit** — `s` / `ms` / `µs`, default **ms**, applied to every time column (`dwell_s`,
  `rise_s`, `fall_s`, `t_start_s`, `t_end_s`). It rescales the **axis only**: the data, and every
  fit computed on it, stay in seconds, so the dwell fit keeps the detector's own `min_dwell_s`.
  Every parameter the window prints — τ, the mixture's μ, their confidence intervals, the
  truncation notes — is shown in the unit the axis is showing; a parameter that is not a time (a
  gamma's dimensionless shape `k`, a depth in pA) is shown in its own. The **Fit range** boxes are
  read in the axis unit too. Axis-range entries follow their displayed unit labels.
- **Axis range** — type explicit X/Y limits in the displayed units (stored in data units and converted to log internally on a log axis)
  that persist across redraws; **Auto** releases them. The plot is also mouse zoom/pan-able.
- **Appearance** — a custom title and axis labels, font family + size, data-series and fit colours,
  marker size, line width, grid and background. These controls, legend edits and fit-decoration
  toggles update the existing picture without recalculating bins, KDE, PCA or fits, collecting
  source data, or resetting manual zoom. Appearance-only Undo/Redo has the same guarantee.
  Pending analytical changes stay pending until **Plot / Apply** or **Apply fit**.
- **Legend** — a **Show legend** toggle, a **Legend position** (upper right, the default, / upper
  left / lower right / lower left; the legend is drawn over the data, so move it off whichever
  quadrant your points occupy), a legend font size and a **(n = …)** toggle. The legend **labels are
  editable**: double-click a row in the source list and type, and that series is renamed everywhere it
  is named — legend, fit lines, Details and the plot title. The real name is kept underneath, so
  clearing the edit back to empty restores it rather than legending a blank.
  Highlight that source and choose **Save name as project default** to reuse its edited name in
  newly generated figures. **Project source color…** saves its identity color for subsequent
  source-colored plots; press **Plot/Generate** explicitly to use it. The current frozen picture,
  single-series data color, fit color and cluster palette are not silently recolored.
- **Export ▾** keeps the outputs together: **Export current figure…** writes the live plot as PNG, SVG, or
  vector PDF; **Export plot data…** writes the exact plotted values as CSV; and **Export
  configuration JSON…** saves the reproducible plot recipe. Figure export opens a shared format/size
  dialog first: choose native size or physical dimensions in mm/in, optionally lock the aspect
  ratio, and set PNG resolution. A frozen graphical preview shows the current figure. If a residual
  panel is displayed, explicitly tick **Include residual panel** to place it beneath the fitted
  plot; it is excluded by default. Physical dimensions apply to all three formats. The 85 mm and
  180 mm presets are starting widths, not promises of compliance with a particular journal.
  At a physical size, **Print check** estimates the smallest measurable text and scalable stroke
  in points. Expand its details before relying on the estimate: outlined text, embedded images,
  nested panels and cosmetic strokes may need manual inspection. The default 8 pt / 0.5 pt targets
  are editable preferences, not journal rules. **Enlarge to measured targets** enlarges the whole
  frozen canvas, preserving its aspect ratio; it does not independently reflow fonts at a fixed
  column width. For a fixed width, increase typography in Appearance and capture/export again.
  **Approximate print-size preview…** uses the screen's reported resolution; it is uncalibrated,
  not a physical ruler. PNG DPI improves raster sampling, not the physical size of text.

### A consistent plotting workflow

Analytical pages use the **Section** dropdown for **Data → Analysis → Appearance → Export**.
The selector only changes which controls are visible; it does not change their values or run an
analysis. It is distinct from the Explorer's tool tabs. The primary **Analyze**, **Generate** or
**Plot / Apply** action stays reachable independently of appearance controls. In Distributions,
**Apply fit** is another entry point to the same full Plot / Apply commit, not a separate fit-only
commit. Where a report consumes an already-computed result, its Data/Analysis
sections explain the source and route an explicit new run to the existing analysis dialog.

The shared Appearance controls include typography, titles/axis labels, canvas/grid, colours,
line width, marker size and legend visibility/size. Controls that do not apply to the current
plot are disabled with an explanation. Category-coloured plots retain category identity;
heatmap colour scales have their own palette controls. Use **Save appearance preset…** to reuse
your style across plot types. Presets are local application preferences and contain appearance
only—not recordings, event filters, bins, axis scaling, fitting choices or detector settings.
Reset restores that plot's existing defaults. Appearance used by a figure is saved in its project.

Warnings remain beside the figure as a short summary and count. Expand **Full details** (or
**Details — methods and provenance** in Distributions) for the complete warnings and audit text.
An **Event-count heatmap** displays counts in two-dimensional bins; **KDE probability density**
is a smoothed density estimate. They are different quantities, not interchangeable names for the
same plot. Counts, probability-per-bin and density retain their existing numerical definitions.

### Figure Library

Use **Analysis Explorer ▸ Save to Figure Library…** to name and freeze the current figure.
The library is separate from the Event Gallery: it stores figures, not curated event selections.
**Save Project** writes the library into the `.nanoscope` project, including:

- A named thumbnail and full PNG/vector SVG picture of the saved view.
- Exact plotted data, source-recording identities, analysis settings, appearance and provenance.
- The associated analytical tables and resources, including saved fitted or derived results.

Open **Analysis ▸ Figure Library…** without opening any analytical page. Selecting a thumbnail
does not rerun analysis, even if the recording is missing. **Duplicate** makes an independent
frozen entry, not an editable analytical page. Reopen the appropriate Analysis tool and generate
from stored event data for an editable variant. **Rename** changes only its name. **Export saved figure…** offers the same publication
sizing/preview for PNG, SVG and PDF. **Save figure bundle…** exports its complete frozen data and
configuration; **Reveal file** selects its most recent exported file in Finder. These actions do
not overwrite another library entry or replace its saved numbers with live results.

Each entry has a source-status label:

- **Current:** its recorded source/calibration/result lineage still matches. The picture continues
  to use its *saved* analysis settings and appearance, not any new settings on another open plot.
- **Outdated:** source data, curation, parameters or calibration changed, or the plot was already
  stale when captured. The old picture and numbers remain intact.
- **Saved snapshot:** source lineage is unavailable or was not verified at capture. This is not a
  claim that a missing recording has been checked. Restored analytical snapshots are captured
  conservatively in this state.

Status is checked when the library opens, when project edits occur while it is visible, or with
**Check status**. Source checks combine a content hash of the immutable loaded samples (cached
once per load without copying the recording), the existing bounded on-disk fingerprint, and
result/metadata checks. No detection, baseline estimation or fitting is run. Appearance edits do not make
scientific source lineage outdated. A snapshot taken with pending controls records those controls
separately from the visible numerical result; Distributions also records its applied plot recipe.

### Multi-panel composition

First save the individual plots to the Figure Library. In **Analysis ▸ Figure Library…**, choose
**Compose multi-panel figure…**. Check 2–12 figures, edit their panel labels (up to 32 characters),
and use **Move up / Move down** to arrange them in reading order. Choose 1–4 columns. The **Rows**
and **Cols** cells set each panel's span (1–4 cells); a column span must fit the grid. Panels are
automatically packed without overlap and fitted without cropping. Use a wider span for a trace
above smaller summary panels. **Panel spacing** and the label font, size, boldness and color are
editable; sizes are native canvas pixels and scale with publication export. **Update
preview** builds a frozen composition; changing the selection, labels or layout invalidates that
preview until you update it again. **Save to Figure Library** creates a new entry and does not
modify any component figure. Save the project to retain it.

**Share X range / Share Y range** uses the union of the saved view ranges on detached copies of
compatible Cartesian plots. It does not stretch an independently scaled picture and call that a
shared axis, and it does not change live zoom ranges, observations or fitted parameters. Matching
axis labels, units, scale, linear/log mode, direction and tick formatting are required. NanoScope's
built-in readable axes retain their exact notation and unit conversion, including linear and
logarithmic dwell axes displayed in milliseconds and the PSD's power-of-ten labels. Some plots
cannot be replayed faithfully with new ranges—such as heatmaps, unknown/custom axis formatters or ticks, error bars,
annotations, per-point palettes or customized legends. Those figures still compose with their
original independent axes. Compatible simple I–V, Population Scatter and PSD plots can share
ranges. Uniform point styling is retained; differing point styles are not flattened.
Older snapshots without a Cartesian model also use independent axes. A shared-axis refusal is a
reason to keep independent axes, not to remove scientific annotations needed to interpret a plot.
Save a new Figure Library snapshot to capture the extended unit-aware model. Older NanoScope
versions refuse to replay this newer model, while retaining its frozen picture and independent-axis composition.

New snapshots can preserve supported standard legends when sharing ranges. **One shared legend**
requires every panel to have the same ordered labels and matching symbol/line styles in a faithfully
replayable Cartesian model. It renders that legend once and removes the matching duplicates from
detached panel copies. Different names, counts, colors or custom legend items are refused, not
silently merged. Keep independent legends in those cases. Older snapshots remain usable as frozen
panels; saving a new source snapshot captures the extended legend model.

Export the composition through the usual Library publication controls (PNG, SVG, PDF, mm/inches
and DPI). A composition's bundle stores a panel index at the top level; each panel's **original,
exact** numerical tables, configuration, provenance and rendering resources remain under its
`panels/panel_NNN/` folder. No concatenation or rescaling changes the original numbers. Compositions
are labeled **Saved snapshot** rather than claiming that all component analyses were refreshed.

To revise an existing composition, select it in the Library and choose **Revise composition…**.
The composer restores its panel order, labels, column count, spans, spacing, label styling and
shared-range/legend choices. Older layouts retain their equal-grid defaults. Reorder or
uncheck panels, edit labels, change the layout and select **Update preview**. A revision can retain
1–12 panels. **Save as new figure** creates a separate Library entry; the selected composition and
all original panels remain unchanged. Changing layout after preview disables saving until a new
preview is generated. Cancel saves nothing.

Revision uses the **embedded original snapshots**, not today's similarly named figures or live
analyses. It therefore works even if separate source figures and recordings are unavailable.
Nested compositions stay intact as single panels; they are not automatically flattened. Changing
shared-range choices starts again from the original saved ranges, avoiding repeated rescaling.
Revision lineage is retained in the project and portable figure bundle. Save Project to keep both
versions. A project switch during editing cancels the save, even if the incoming project contains
identical figures; ordinary background checkpoints do not cancel it. Unsupported composition
versions or damaged panel resources block revision without replacing the stored figure.

### Lasso selection in Scatter and PCA

In the **Distributions** page, generate a Scatter or PCA plot, then enable **Lasso select** in the
toolbar. Drag an outline inside the plotting area and release to select the enclosed displayed
events. Selection uses event identifiers, not point positions or table row numbers; it follows the
displayed logarithmic coordinates when applicable. Ctrl/Cmd/Shift toggles the enclosed events in
the selection. A plain outline around no points clears the current plot selection.

The detector event table shows one source at a time. If the outline contains points from several
sources, a dialog lists their names and counts so you can choose one; Cancel leaves selection
unchanged. Manual-measurement points can be highlighted but are not detector event-table rows.
Esc or right-click cancels the gesture and returns to pan/zoom. A redraw or canvas change cancels
an unfinished outline. The temporary outline is not included in figure exports. Selecting events
does **not** accept/reject them, change their masks, remeasure amplitudes or run an analysis.

### Optional distribution summaries

Open **Analysis ▸ Distribution summaries…** to compare existing measurements without changing
the established histogram, fitting or detection workflows. This page is opt-in: opening it, changing
its controls or returning to its tab does not calculate a new result.

You can also find it through **Analysis home → Distribution summaries → Open Distribution
summaries**. It is not the Histogram option inside Distributions. Use the page's **Section**
dropdown to alternate between the following control groups. Its **Generate** button and the
Explorer's **Refresh current** both calculate a new summary from current inputs; the latter is
not simply refreshing the list. Unapplied controls leave the previous figure visible.

1. In **Data**, check the desired sources. Each stays a separate distribution; values are not pooled.
   **Accepted events only** is on by default. Choose dwell time, median depth, fractional depth or
   the explicitly selected amplitude. Values keep their recorded units and measurement domain.
2. In **Analysis**, choose a summary and press **Generate**:
   - **ECDF:** the unweighted fraction of finite observations less than or equal to each value,
     `F(x) = count(X ≤ x) / n`. Repeated values share one right-continuous jump.
   - **Empirical survival (uncensored):** `P(X > x) = count(X > x) / n`. This is **not Kaplan–Meier**:
     it makes no censoring, detection-efficiency or missed-event correction and must not be
     interpreted as a censoring-corrected lifetime analysis.
   - **Box:** Q1, median and Q3 use linear-interpolated quantiles. Whiskers reach the outermost
     observations inside the 1.5-IQR fences; circles retain observations outside those fences.
   - **Violin:** Gaussian KDE with Scott bandwidth. The grid control changes where density is
     evaluated, not the bandwidth rule. The shape is drawn only across the observed range;
     omitted tails are not renormalized. Each source has the same peak display width, regardless
     of its event count. For fewer than three observations or constant data, observed values are
     shown without inventing a violin. Large inputs are refused rather than silently subsampled.
3. Read the counts and **Full details**. Non-finite or nonnumeric measurements are excluded and
   counted, not imputed. Missing acceptance flags, sparse data, differing calibration/baseline/
   normalization and incompatible selected-amplitude choices are disclosed. In particular, the
   stored `depth_pA` column can describe a normalized signal rather than literal picoamps.
   Boolean, complex and typed date/time measurements are refused, as are integer measurements
   beyond ±2⁵³; they are not silently converted or rounded. Long display names are abbreviated
   in plot labels, with complete names retained in details and exported numerical provenance.
4. Use **Appearance** to style the committed graphics without recomputing the summary or resetting
   zoom. Data/Analysis edits leave the previous result visible and marked as unapplied until Generate.
   **Source names and colors → Apply to figure** gives each source a short condition label and
   a local color override. **Use as project source default** also saves it for that exact source
   identity (recording/segment, channel and detector) for subsequently generated figures. Save the
   project to persist defaults. Other frozen figures are not rewritten. Clearing an override
   restores the figure's captured source style; original names remain in tables and Details.
   ECDF and survival ticks retain probabilities between 0 and 1, without automatic SI scaling.
   Measurement ticks stay in the recorded units stated on the axis.
5. Export the committed picture or plotted-coordinate CSV. Save to the Figure Library to retain
   source observations, summary values, draw coordinates, method settings and provenance together.
   Project reopening paints the stored coordinates; it does not rerun ECDF, quantiles or KDE.

These are descriptive summaries of the observations that were actually recorded and selected.
They do not change baseline removal, event detection, amplitude estimation, existing fit defaults,
calibration or previous numerical exports.

Method references: [SciPy ECDF / empirical survival](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html),
[NumPy linear quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html),
[SciPy Gaussian KDE / Scott bandwidth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html),
and [Matplotlib's 1.5-IQR box-plot convention](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html).

---

### Comparison figures: facets, pair matrices and individual events

Open **Analysis ▸ Comparison figures…**. This opt-in page displays measurements already stored
in event tables; opening it never detects events, recalibrates currents or fits a model.

The equivalent catalog route is **Analysis home → Comparison figures → Open Comparison figures**.
The **Figure** choice is under **Section → Data**; its detailed controls are under
**Section → Analysis**. Opening or returning to this page does not generate new comparison
geometry. Press **Generate** deliberately; **Refresh current** in the Explorer executes the same
generation path. Checkmarked sources are included; simply highlighting a row does not include it.

1. In **Data**, check the recordings to compare and choose the figure type. **Accepted events
   only** starts enabled. Sources remain separately identified, even when their names or event
   IDs repeat. Missing sources can still be viewed in a saved snapshot but cannot be regenerated.
2. In **Analysis**, configure the selected view, then press **Generate**:
   - **Faceted comparisons:** choose X and Y measurements and the number of columns. One scatter
     panel is drawn per source with shared numerical axis limits. Optional logarithmic axes
     exclude nonpositive values explicitly; they do not transform the stored measurements.
   - **Pair-plot matrix:** select two to five features. Off-diagonal panels show individual-event
     scatter plots; diagonal panels show separate-source count histograms with common bin edges.
     There is no KDE, regression, correlation estimate or automatic confidence interval. Missing
     values are handled separately for each panel, so one missing feature does not remove an
     event from every other comparison. Axes retain their individual feature labels and units.
   - **Individual-event overlays:** choose a measurement and **Strip** or **Swarm**. Strip uses
     deterministic horizontal spreading. Swarm computes a collision-avoiding layout once on
     Generate. Horizontal position is a display aid, not another measurement. Optional boxes
     use linear quartiles and 1.5-IQR sample whiskers. All retained observations are drawn,
     including outliers. Increasing marker size later can create overlap without moving events.
3. Read **Details — methods and provenance** for selected and excluded counts, calibration and
   measurement-domain warnings, method settings and complete source names. Warnings are not
   printed on the figure. The `depth_pA` field is the original median depth; **Selected amplitude**
   is the separate, explicitly curated amplitude. Neither is recalculated by this page.
4. Use **Appearance** for typography, background, grid, marker size and line width. Source colours
   retain recording identity. Styling uses saved geometry; it never recalculates bin counts, box
   statistics or swarm offsets. Data/Analysis changes remain drafts until Generate. **Fit canvas**,
   scrolling and dragging magnify/pan the canvas only; exports retain the complete figure.
   **Source names and colors** offers the same figure-local labels/colors and explicit project
   defaults as distribution summaries. Short labels appear on facets and legends; full identities
   and overrides remain in Details and the saved configuration. Duplicate names are disambiguated,
   not treated as one source. Saved geometry and numerical exports do not change.
5. Use **Export current figure…** for PNG, PDF or SVG and the shared physical-size/preview controls.
   **Export plotted values CSV…** writes committed drawing values with primitive type, source and
   measurement labels. Use **Save to Figure Library…** or **Save figure bundle…** to retain source
   observations, panel exclusions, exact geometry, configuration and full provenance together.
   Project reopening draws the saved snapshot without requiring the original recordings.

Limits are explicit: at most eight sources and 50,000 selected rows; pair matrices allow 10,000
selected rows and 250,000 plotted observations. Swarm allows at most 500 rows per source and
2,000 overall and may refuse a crowded layout. Use Strip for larger sets. Nothing is silently
subsampled. Figures can be used as frozen panels in compositions; detached shared-axis editing
is not offered for these already-multi-panel canvases. These comparisons do not change baseline
removal, detection, calibration, amplitude estimation, fitting defaults or existing numerical exports.

---

## 11. Noise spectra (PSD)

**Analysis ▸ Noise spectrum…** opens the **Noise spectrum** page of the Analysis Explorer: the
log-log current-noise spectrum every
nanopore paper uses to grade a pore. It runs on the **raw calibrated trace** (no detection needed,
no filter chain applied), over a chosen quiet open-pore segment, and separates the three noise
regimes: the low-frequency **1/f (flicker)** power law (α and the 1 Hz magnitude are fit and the
line overlaid), a **mid-log-frequency median proxy** (a heuristic summary band, not proof
of a white-noise plateau), and the high-frequency **dielectric /
capacitive rise**. It reports the **band-integrated RMS noise** — a total, the standard
`<1 kHz / 1-10 kHz / 10 kHz-Nyquist` split, and a live read-out over a band you set.

**The window never invents a unit.** A current channel whose file declares none — the default state
of every fresh HDF5 and CSV — is reported as `(units unknown)` on every read-out and in the legend,
the y axis reads *PSD (units unknown — set them in Calibration)*, and the summary carries a note
saying every magnitude above it is a bare number in the file's own scale, not a current in any named
unit. Set the channel's units (and gain) in **Calibration** to make the noise figures physical. In
builds before this one the blank was filled in with `pA`, which printed a 4.2 fA noise floor for a
recording whose own loader note said the magnitudes suggest nA.

**Setting the window: don't type the numbers.** *Start* and *End* are seconds from the start of the
recording, while the trace's x-axis is on the instrument clock (§2), so a recording with a non-zero
t₀ will not agree with what you read off the trace. Bracket the quiet stretch with the Measure
tool's cursors and use **Analysis ▸ Power spectrum of the measured region** instead — the selection
is carried as sample indices, so the two clocks cannot be confused.

### Comparing two pores

Pick the **primary** recording in *Source*, then tick any other open recordings (segments included)
in **Compare with**. Their spectra are overlaid on the same axes, each in its own colour with a
**legend** giving its name and total RMS. Every curve uses the same Welch settings, so the
comparison is like-for-like, and every curve is **reported separately**: its own noise summary
(α, mid-log median proxy, band RMS), its own sample rate and Nyquist, its own units, and its own band-RMS
read-out. A comparison recording shorter than the chosen window is analysed over what it has, and
says so.

> The **shapes** of two spectra are always comparable (α, where the floor sits, where the capacitive
> rise starts). Their absolute **heights** are not, until the recordings share a real current
> calibration — a PSD value is a squared current, so it carries the amplifier gain. The window
> prints that caution, and calls out curves that are in different units.

### Spectrogram timing and large selections

The spectrogram caption distinguishes four quantities that are easy to confuse: the FFT source-block
duration, overlap hop, drawn column step, and nominal temporal support after display columns are
merged. When a long spectrogram must be narrowed to the screen, NanoScope uses equal-width fractional
column averaging: every complete source column contributes, the mean is preserved, and a short leftover
group cannot make the final column appear to have a different duration. Standard Welch segmentation can
leave fewer than one hop of trailing requested samples outside a complete block; the caption states that
tail explicitly. **Export data (CSV)…** records each
drawn bin's start/stop time, contributing source-column count, block duration, hop, drawn step and
nominal support, plus the analysed and uncovered-tail sample counts, so the image can be reconstructed
without guessing its time geometry.

Very large FFTs and spectrogram matrices are refused *before* their full float64/SciPy work arrays are
allocated. The message gives the requested size and asks you to analyse a shorter **Segment** or the
**Visible view**. This is a resource guard, not silent downsampling of scientific output.

---

## 11a. Reopening a project

A project stores **references**, not signal: each recording's path plus the sample range you
loaded, and everything you cannot cheaply recreate (calibrations, layer view state and stable
recording colours, curated event tables, manual measurements, segments). Reopening it reopens those
files from disk.

**If a source file has changed, the app says so.** A path is a *name*, not a recording — re-exporting
from Clampex over the same filename leaves the name identical and the data completely different. The
project records what each file contained when you saved it (its sample count, its size and a content
digest), and on reopen a mismatch raises a banner naming what differs and **how many of the restored
events now lie past the end of the recording**. The project still opens; nothing is thrown away.
Re-detect on the new file, or point the project at the original.

**A damaged project file is refused as damaged, with the fix named.** A `.nanoscope` that begins
like a project but is cut off (an interrupted copy or sync) is refused as *"damaged or truncated…
Restore it from a backup, or copy the original again"*; corruption caught mid-read fails the
archive's integrity check with the same advice. Only a file that never was a project is called
*"not a NanoScope project (not a ZIP archive)"*.

**A save never silently overwrites newer work.** If the project file changed on disk since this
window last read or saved it — another NanoScope window, another machine, a sync tool — Save asks
first, with Cancel as the default and *Save Project As…* suggested to keep both. Nothing is
reverted without the question. NanoScope compares the file's contents, not only its timestamp and
size, and keeps the final comparison and replacement inside one cross-process lock. An overwrite
answer approves only the exact revision shown: if a third writer saves while the question is open,
NanoScope asks again. The same protection catches two first-time saves that choose the same new
project name.

**If you cancel a restore, it says what it did not load.** Pressing *Cancel* while a project is
reopening stops the recording in flight *and* the ones still queued behind it — so a restore you
cancel is a partial one. The banner names how many recordings were not loaded and does not claim
the project was restored, and the project stays marked unsaved so a later Ctrl+S cannot quietly
write the shortfall back out. Those recordings are kept in the project file, not dropped; reopen it
to load them. A source file that was found but could not be *read* is reported separately, with the
reader's own reason (an unsupported ABF variant, a saved window that no longer fits a re-exported
file, permissions) rather than as a file that could not be found.

The check is deliberately quiet when it has no evidence: a project saved by an older build carries no
such stamp, and a file whose modification time moved but whose contents did not (a copy, a sync
folder) is not a change. A warning you can trust is one you only see when something really happened.

**The app never analyses a recording it is not drawing.** A project can legitimately be saved showing
a deliberate multi-recording overlay, and reopening it reproduces that view exactly. If the restored
detection belongs to a recording the saved view did not include, that recording is **added** to the
view — nothing already on screen is hidden — and the banner says so. This is also why *Detect over
the visible view* refuses, with a reason, on a channel that is not on screen instead of silently
running over the whole recording.

**A run in flight belongs to the recording it started on.** Close a recording while a detection is
running and the answer is discarded and reported, rather than reappearing on the next thing you open
under that name — including a re-opened load of the same file, which is a different recording as far
as your numbers are concerned.

**Saving writes the file you named.** Every file this app writes over your own data — the project,
the calibration sidecar, exported events, exported figures — is written to a temporary file beside it
and moved into place in one step, so an interrupted or failed save leaves the previous version
completely intact rather than a truncated one. Three consequences worth knowing:

- **A symlink is followed.** If `current.nanoscope` is a link into a dated folder, the file it points
  at is the one written and the link still points at it. The save says so the first time — the file
  that changed is not the one you clicked — and then stops repeating itself.
- **Finder tags, colours and comments survive**, as do the file's permissions and owner. A project
  you marked read-only (`chmod 400`) is **refused** with a message instead of being overwritten.
- **A hard link cannot be preserved.** Replacing a file gives it a new identity on disk, so any other
  name for the same file keeps the *previous* contents. The save names those other names when there
  are any; nothing else in the app can tell you this after the fact.

---

## 11b. Recovering unsaved work after a crash

NanoScope keeps a private emergency checkpoint while your analysis is unsaved. It takes one after
about **15 seconds without another edit**; if you keep curating continuously, a **60-second maximum**
prevents each new click from postponing the checkpoint forever. The checkpoint contains the same
restorable analysis state as a project — calibrations, view state, detector tables, curation,
labels, fits, segments and manual measurements. As with a project, recordings are path references;
their raw signal is not duplicated.

After NanoScope or the computer stops unexpectedly, the next launch presents **Recover unsaved
NanoScope work**. Read the project name (or *unnamed analysis*) and revision, then choose:

- **Recover** — open that checkpoint. The title begins with `•` and Gallery/Inspector say
  **Unsaved changes**. This is intentional: the private checkpoint has not become a durable project.
  Press **Ctrl/⌘+S** and choose a `.nanoscope` file to keep it. If the recovered work came from a
  named project, that path is suggested, but NanoScope does not silently overwrite it.
- **Discard** — permanently remove both checkpoint generations for only the session named in the
  question. If another crashed NanoScope window also left work, it is offered separately.
- **Show in Finder** — reveal the private checkpoint folder, then return to the same Recover/Discard
  question. This does not consume or modify the checkpoint.
- **Not now** (or close the question) — leave the checkpoint untouched and continue with an empty
  window. It is offered again on a later launch.

NanoScope retains two complete generations. If the newest file was interrupted or damaged, the app
validates it and offers the preceding good generation instead. A successful project save removes
only checkpoint revisions represented by the project that actually reached disk; an edit made while
that save was finishing remains unsaved and recoverable. A cancelled or failed save removes none.

Treat recovery as a safety net, not as file management or a backup. Its retention is bounded, it
does not include undo/redo history, and the private files are not designed for sharing. Save a named
`.nanoscope` project whenever you reach a point you need to reproduce, move, or archive.

---

## 11c. Complete analytical projects — Stage 3

Project format v11 makes the ordinary `.nanoscope` file a complete analytical project. It saves the
durable session underneath the analysis windows and the project-owned analytical workspace: live
analysis-panel configuration, plot/source choices and styling, exact PCA and clustering outputs
with their preprocessing and random seeds, complete subpopulation reports, PSD and comparison
analysis, density-figure choices, and the selected ML-model path plus a content fingerprint. Raw
recordings and model bytes remain external files rather than being copied into the project.

The important distinction is between a **saved result** and a **redrawn view**:

| After a v11 reopen | What happens |
| --- | --- |
| PCA, clustering, subpopulation, PSD, I-V and voltage-dependence numbers | Restored from the saved snapshot. NanoScope does not silently run them again. |
| Histogram bins, scatter/KDE geometry, contours, fitted-line drawing, captions and plot widgets | Redrawn from saved structured data and settings. |
| A missing or changed recording/model | Named as missing, changed, or stale. The saved result is retained where it can stand alone; no replacement analysis starts automatically. |
| **Refresh** or **Run…** | Explicitly computes a new result and marks the project unsaved. |

Population Plots preserves a KDE's recipe but does not store its sampled density grid. After a
reopen it shows **Press Plot to recompute** instead of silently running the KDE. The Event Density
view does store and restore its frozen count grid and physical bin edges.

Older projects will continue to open with empty defaults for the new analytical blocks. The reverse
is intentionally refused: an app limited to project v10 must tell you a v11 project needs a newer
NanoScope instead of opening it and stripping fields on the next save. The full implementation and
test contract is in [STAGE_3_ANALYTICAL_PROJECTS.md](STAGE_3_ANALYTICAL_PROJECTS.md).

---

## 12. Exporting

- **Analysis ▸ Export events…** — the current detection's events to CSV (accepted rows), plus a
  `<name>.meta.json` sidecar naming the detector, the preprocessing chain, the **open-pore
  estimator** (`baseline_method`, its windows, and a `baseline_fingerprint`) that produced the
  amplitudes, **and the clock the times are on.** The 20-column analysis schema is frozen; the
  standard exported CSV carries 25 columns — those 20 plus `orig_start_idx`,
  `orig_end_idx`, `added`, `source` and `review`. Optional curation fields may add columns;
  the sidecar's `columns` list names the columns actually written. A new column is
  therefore not where the origin can go, so it goes in the sidecar: a top-level `t0_s` plus a
  `clock` block giving `columns_clock`, the columns it applies to, `t0_s_known`, and the same
  sentence the on-screen column header shows — *"add t₀ = 1154.9909 s to a number here to find
  it on the trace."* That matters on a windowed load:
  `t_start_s` is seconds from the first *loaded* sample, so two exports taken from two windows of one
  recording date the same physical event differently, and `t0_s` is what reconciles them. There is no
  import path that can re-attach a clock afterwards — the CSV is a one-way handoff. Where the writing
  surface had no origin to give, `t0_s_known` is `false` and the note says the times cannot be placed
  on the trace axis; it is never silently filled in with 0. The CSV and its sidecar land together or
  not at all — a table from today beside a sidecar from last month is not a partial export, it is an
  export with a confidently wrong provenance record. CSVs exported before that sidecar existed cannot
  be repaired in place — see `docs/RELEASE_NOTE_BASELINE.md` for the note to attach to them.
- **Manual measurement ▸ Export CSV…** — the manual-measurement list, **32 columns**, including
  kurtosis, the amplitude method, and four provenance stamps that make a pooled table safe to read
  downstream: `units`, `cal_gain` and `cal_offset` (the calibration each row was measured under — a
  table may legitimately mix pA and nA rows from different recordings, and these are what tell them
  apart) and `t0_s` (the instrument-clock origin of that row's recording, so `t_start_s + t0_s` is
  the instant the trace axis showed). For a segment, `t0_s` is the **parent** recording's origin,
  not the 0 the app rebased the segment's own axis to. `NaN` in any of them means *unknown* — a row
  saved before the column existed, or a recording with no time base — and never 0. An older list,
  restored from a project written before a column existed, is coerced onto the current header on the
  way out rather than exported short.
- **Analysis ▸ Export events for training…** — the current event table as regions/labels for the
  StudentModel notebook or another external trainer: one row per selected event with accepted,
  added, and structured `label_*` columns, plus a provenance sidecar that records the selection and
  makes the preprocessing that produced the regions recoverable. The first dialog offers **Reviewed
  decisions (recommended)** (human positives and negatives), **Accepted candidates** (which may
  include untouched rows), and **All raw candidates** (including automatic detector rejects).
  A Reconciled export carries its membership/matching provenance too, and unresolved `|` label
  conflicts raise a warning whose default is Cancel.
- **Analysis ▸ Export sub-state levels…** — the fitted sub-state levels, one row per event level,
  plus a provenance sidecar naming the CUSUM parameters. Enabled once a level fit has been run.
- **Analysis ▸ Export density figure…** — every open recording's detected events as 2-D KDE contours
  (dwell vs amplitude) with marginal densities: the comparison figure for several molecules or
  several voltages on one pair of axes. Amplitude can be depth in pA or fractional blockade ΔI/I₀,
  which is gain-independent and therefore comparable across recordings.
- **Analysis ▸ Generate Report figures…** — standard figures (matplotlib) for the current analysis, each as
  PDF + PNG with a source-data CSV sidecar. **Each figure's three files land together or not at
  all**, so a figure can never disagree with the numbers beside it. So does the export as a whole
  against anything environmental: a permission or disk error while staging, or a failure at the
  moment of commit, writes nothing and tells you the folder is exactly as it was. What is *not*
  all-or-nothing is one bad figure — a figure that fails while being rendered (an invalid label
  carried out of the data file, say) is withdrawn from the staged set and named with its real error,
  and the healthy figures still commit. In builds before this one one bad title sank the whole export
  and the dialog blamed the environment. Either way, if a figure could not be built or rendered and a
  file of that name is already on disk, you are told **which files under this base name are left over
  from an earlier export**, with their dates — including when *every* figure failed, which is the
  case where the most is stale.
- **Analysis Explorer ▸ Export current figure…** — the plot that is actually on
  screen, including companion axes such as density marginals and spectrogram colour bars. The
  Distributions page exposes the same choices from **Export ▾**. This is distinct from the standard
  report generator above: it keeps your current view and styling. Distributions exports its main
  axes by default; explicitly enable **Include residual panel** in the preview dialog to include
  the displayed residual strip. Physical PNG exports embed resolution metadata,
  PDF uses a vector page, and SVG records mm dimensions without changing its viewBox. Sizing scales
  the entire canvas, including text and lines: inspect readability at the final printed size.

- **Analysis Explorer ▸ Save figure bundle…** — one atomic ZIP containing the current PNG and
  vector SVG, `data.csv` (17-digit numerical precision), lossless `data.parquet` (including dtypes,
  missing values and index), `config.json`, `provenance.json` and a checksummed `manifest.json`.
  Ordinary numerical Parquet columns remain native numbers. Heterogeneous object fields (such as
  integer event IDs mixed with empty summary cells) use explicitly tagged scalars identified by
  the `nanoscope.figure_table_codec` Parquet schema metadata; NanoScope reconstructs their exact
  types. Python users can use `nanoscope.core.figure_tables.decode_figure_table` on the Parquet
  bytes to restore tagged fields without starting the GUI. The CSV remains directly readable.
  A checksummed `rendering/figure.qpic` preserves the
  frozen vector drawing commands for faithful PNG/PDF rendering, including clipping and backgrounds.
  Additional analytical tables/resources referenced by the configuration are included. Nothing
  is recomputed. Bundles are self-contained exports, not raw-recording archives or project files;
  use `.nanoscope` to resume work. The Figure Library can export the same bundle from a saved
  snapshot, without live recordings. Residual inclusion is explicit when capturing a live figure;
  a saved figure keeps the panels captured with it.

**Presentation controls.** Every analytical page exposes the shared **Appearance** section with
opt-in built-in and reusable user presets. Shared settings are saved with project workspaces;
style edits and reopening saved figures do not rerun fits or analyses. Existing automatic units,
source/category colours and full scientific warnings are retained. The main preview is scaled to
fit the dialog. An optional approximate print-size view uses reported screen DPI but is not calibrated.
The physical-size print check estimates measurable SVG text/strokes and explicitly lists unsupported
content; it is not a guarantee against clipping, hidden annotations, or journal noncompliance.
Interactive exports freeze their picture
before showing the preview, so a background result finishing during the dialogs cannot change the
approved image. Large marker plots may take time to prepare their vector snapshot; a warning allows
you to cancel without dropping points. Inspect the exported figure at its final
printed size; journal-specific requirements are still your responsibility.

The numeric-data export commands remain separate:

- **Analysis Explorer ▸ Export data (CSV)…** — the exact numeric values behind the
  visible plot, rather than a recomputation from its controls. Depending on the page this includes
  histogram bin edges, scatter/event identities, ensemble traces, spectral grids, fitted curves,
  excluded points, source identity, settings, warnings, and analysis provenance. An empty view is
  refused instead of producing a plausible-looking header-only CSV. Event ensembles and
  spectrograms are exact long-form tables; above 250,000 rows NanoScope refuses the interactive
  export before allocating it, explains the event×point or frequency×time size, and tells you to
  lower **Ensemble › Points** or shorten the analysed **Segment**. It never silently samples the
  displayed result. The Distributions page calls this **Export plot data…** under **Export ▾**.

Every export replaces its target the same way a project save does: staged beside it and moved into
place, so a failure leaves the previous file whole rather than truncated.

**Because a replace installs a new file, the app tells you what that changed about the *file*.** Two
things an atomic replace cannot preserve, and it says so at the moment it happens — the only moment
it can, because the replace leaves no trace of either afterwards:

- **Hard links.** *"events.csv had 1 other name(s) on this disk (hard links). Saving replaces the
  file rather than editing it in place, so those other names still hold the PREVIOUS contents. Delete
  and re-create them if you rely on them."* If you hard-linked an exported CSV into a manuscript
  folder, curated two events and re-exported, the manuscript copy is now the **old** table.
- **Symlinks.** *"'current.nanoscope' is a link to /analysis/2026-08-24/run.nanoscope; that file was
  written, and the link still points at it."* The write landed on the destination, not on the name
  you typed.

This is now the default for every writer, not an opt-in the project save alone used: the events CSV
and its sidecar, every exported figure, and image/data export on all five live plotting panels. The
note appears as a small dialog titled after the command — *Export events*, *Export figures*, *Save
plot image* — scoped to the files that command just wrote, and said once per file per session. An
ordinary save of an ordinary file says nothing at all; silence here means *nothing unusual*, not
*not checked*.

And because nothing is written until rendering or CSV generation succeeds, a failed plot export no longer claims
"the file was not created" over a file that is sitting there. It tells you the file in that folder is
the **earlier** export, with the date it carries, untouched but now older than the figure on screen.

**The overwrite prompt names every file, not just the one you typed.** A figure export writes up to
twelve files under one base name, and an events export writes a `.meta.json` beside the CSV; the
prompt lists whichever of them already exist. **Cancel is the default** — pressing Return or Escape
keeps your files, and replacing them takes a deliberate click on **Replace**. In builds before this
one Return meant Replace.

---

## 12a. A complete worked example

The manual's historical **Walkthrough 5 — ML detection, replicated step by step** records a full analysis
on the three apo-ferritin reference recordings — open-pore measurement, ML detection with a Keras
student model, scatter, KDE, histograms with fits, PSD, PCA and GMM/t-SNE/UMAP substructure — with
the exact numbers this bench produced, screenshots of every surface, and the three traps a
first-time user hits (wait for the proposal before Apply; detect before switching recordings; read
the fit's KS verdict in Details before quoting its parameters). Its model, preprocessing and
dependencies must match before its counts are treated as replication checks. The current local
environment records a TensorFlow/Keras requirement mismatch; this manual refresh does not resolve
or validate those ML dependencies.

The updated manual also gives a current Schmitt apo-ferritin recipe with recording-specific
Automatic proposals and explicit10ms post-detection merging:881/239/984 accepted events at
−500/−600/−700mV, respectively. Its nine native population figures, editable amplitude example,
and generator-assisted reference examples distinguish numerical replication from seven custom
Matplotlib layouts. These were generated by current engines/offscreen renderers, not a desktop
GUI benchmark; retain the exact input files and unverified instrument-scale qualification.

## 13. Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+O | Open file |
| Ctrl+S | Save project |
| Ctrl+Shift+S | Save project as… |
| Ctrl+W | Close current document |
| **Ctrl/⌘+Z** | **Undo the last change to the event table on screen (or to the measurement list)** |
| **Ctrl+Shift+Z / ⇧⌘Z** | **Redo it** |
| Ctrl+D | Run detection |
| **Ctrl+I** | **Open the Event inspector** |
| **Ctrl+G** | **Open the Event gallery** |
| **Ctrl+M** | **Save the current cursor region as an independent manual measurement (works from any tab; not a detector-training candidate)** |
| **Ctrl+Shift+M** | **Write the current cursor region as a reviewed, accepted, hand-added event in the selected source detector table; loudly refuses without a source or on Reconciled** |
| **Ctrl+Shift+F** | **Open the Signal Conditioning window (the classical detectors' signal chain)** |
| **Signal Conditioning:** Ctrl+1 / Ctrl+2 / Ctrl+3 | Time / Spectrum / Response preview tab |
| Ctrl+0 / trace-focused Home / **Full** button | Reset zoom to the complete trace |
| Ctrl+← / Ctrl+→ (or Alt+← / Alt+→) | Zoom Back / Forward (view history) |
| Ctrl+E | Centre the view on the current selection |
| **Trace focused: Shift+← / Shift+→** | **Scan to the previous / next view at the current zoom, with 10% overlap; carry active Measure cursors with it** |
| **Trace focused: Shift+↑ / Shift+↓** | **Zoom in / out one incremental step around the centre** |
| Double-click trace | Place/activate a Measure mask at the click, capped at 20% of the visible view |
| Mouse wheel on trace | Move the signal vertically only (up moves up; down moves down) |
| **Event inspector:** ←/→ | Previous / Next event |
| **Event inspector:** A / R / Space | Accept / Reject / toggle (then the next event of the filter when **Advance after decision** is on) |
| **Event inspector:** T | Centre the main trace on the event on screen |
| **Event inspector & Event gallery:** Ctrl/⌘+U | First unreviewed event / tile of the current filter |
| **Event inspector:** Ctrl/⌘+Z / Ctrl/⌘+Shift+Z | Undo / redo on the main window's history, without leaving the inspector (docked: Edit ▸ Undo / Redo themselves; an advance is still walked back) |
| **Event inspector & Event gallery:** Ctrl/⌘+S | Save the complete project through the same writer as File ▸ Save Project (docked inspector: File ▸ Save Project itself) |
| **Event inspector & Event gallery:** Ctrl/⌘+Shift+S | Save the complete project under a new name |
| Ctrl/⌘+Shift+R | Show / hide the Results panel (event table, Summary, Fits, Quality) |
| **Event table:** j / k | Next / previous row |
| **Event table:** a / r / f | Accept / Reject / flag the current row |
| **Event gallery:** A / R / Space | Accept / Reject / toggle every selected tile (a single tile moves to the next one when **Advance after decision** is on) |
| **Event gallery:** PageDown / PageUp | Next / previous page |
| **Event gallery:** Home / End | First / last page |
| **Event inspector & Event gallery:** 1–9 | Apply the matching label from the palette (**Analysis ▸ Event labels…**) — same digits, same meanings, in both windows |
| **Event inspector & Event gallery:** custom one-character label hotkey | Apply that palette label (reserved and duplicate keys cannot be saved) |

---

## 14. Known limitations & caveats

- **Current event-count heatmap display limitation:** the Event waveforms image can transpose
  bin coordinates relative to the correctly indexed counts/marginals. Do not read dwell-depth
  coordinates from it until a verified fix is installed. Use numerical tables, scatter or the
  validated histogram. This is distinct from KDE probability density. Documentation does not
  repair the display code.
- **Background sub-state fits use frozen inputs.** Cancellation or a changed source, event table,
  calibration, sample rate, parameters or provenance can prevent a pending fit being installed.
  Review the message; do not assume a finished worker implies a current result.
- **Reference/filter restore safety is deliberate.** A malformed or missing applied reference
  recipe blocks analysis rather than silently using Automatic. An unreplayable stored filter
  does not silently remeasure raw samples. Resolve the reported source problem; original
  measurements and curation remain the record of the saved analysis.

- **Reconciliation Preview currently runs synchronously.** The legal candidate and edge caps keep
  ordinary previews bounded, but there is not yet a truthful mid-computation Cancel button. Review
  the source sizes before reconciling an unusually dense detector output.

- **No universal reference choice.** For Automatic, inspect a recording-specific proposal;
  for supplied modes, document the independent current or identified intervals. Neither a
  passing health check nor user confirmation certifies physical open-pore identity.
- **An event sitting within one base window of a genuine conductance step gets an overstated depth,
  silently.** Measured on a synthetic gating pore: 205 pA against a true 120 pA (1.71×), with an
  empty `flag` and every one of the nine Baseline health checks reading `ok`. Every gating pore has
  these. Detecting the step itself is not implemented; on a gating pore, check depths near the steps
  by hand.
- **The open pore is *chosen*, not certified** (§5a). The estimator calls the level the recording
  spends most of its time at the open pore. That is its whole rule, and no check in the Baseline
  health block reads the physics that would confirm it — so a pore that is *blocked* most of the
  time is measured upside down. On a synthetic pore blocked 81 % of the time (open 100 pA, blocked
  60 pA): a baseline of 60.04 pA, a median ΔI/I₀ of 0.665 against a true 0.400, and the events
  reported as "enhancement"; 90 % and 95 % blocked read the same way. A recording that visits two
  conductance levels raises a standing advisory naming both levels, how much of the recording each
  holds, and the implications for ΔI/I₀. These are interpretations, not a certification of the
  open pore. A deliberate new reference/detector run can change noise, polarity, boundaries,
  depth and counts. **Compare with the current you independently know for this recording**;
  supported Schmitt/CUSUM workflows can use that signed current or selected open-pore intervals.
  Preview first, then Detect explicitly and retain the previous project for comparison.
- **The false-positive *rate* is not estimated at all** (§6). The AAFT surrogate is not a valid null
  on a real pore, so the control refuses a percentage and reports the event-free count instead.
  Discard any FDR percentage from an older build rather than rescaling it. The lower bound it does
  report, `n_event_free`, is **one draw from a wide distribution, not a count** — four identical
  unseeded runs on the −600 mV recording returned 52, 55, 66 and 52. It is now seeded from the
  recording's own bytes and the settings being measured, so pressing the button again reproduces the
  same number and the seed is stamped into the result. That makes it citable, not meaningful: quote
  it as a lower bound, and do not read a change in it as a change in the pore.
- **A very short segment gets a worse baseline, and says so** (§5a). Below about 2,000 samples the
  clamped estimator degrades; below the base window (5,000 samples at the default and 50 kHz) it is
  clamped at all, and the provenance records that. Do not compare depths from a few-event segment
  against the parent recording's.
- **Six `.keras` models cannot be loaded** — not by this app and not by any Keras 3 install: their
  `Lambda` layers are refused by `safe_mode`. Re-export them from the training notebook the way the
  working ones were re-exported.
- **Current-gain calibration is deferred.** Absolute pA/nA (depths, conductance) are only physically
  correct once you set the current-channel gain in Calibration. Detection, dwell, and distribution
  *shapes* are unaffected.
- **Crash recovery is not a project save or a backup.** NanoScope keeps a private, bounded emergency
  checkpoint (§11b), but a durable or shareable analysis still requires **File ▸ Save Project**
  (⌘S). Closing and choosing **Discard** removes unsaved recovery data. A project written by this
  build cannot be opened by a build older than it — every older project still opens here.
- **`CPYA`-variant ABF files** cannot be read; they are refused with a specific message.
- **Legacy TDMS** files declare no sample rate → the x-axis is in samples and detection is disabled.
- **Raw-Volts TDMS** channels are stored as Volts; they need a gain to become current, so they are
  not auto-classified as a current channel.
- **High-pass is localization-only in new Schmitt/CUSUM runs, not consequence-free.** The event
  table's baseline, depth, fractional depth, mean/max deviation and SNR come from the equivalent
  no-high-pass branch, but start/end, dwell and rise/fall retain high-pass-derived boundaries, and
  those boundaries set ECD's integration limits. Review candidates when the cutoff approaches the
  event bandwidth. Legacy high-pass results remain candidate-only and should be re-run. Use
  Baseline drift when you need quantitative drift removal without this boundary dependence;
  **ASLS** baseline flattening is weak at default parameters — prefer block-median/percentile.

---

*NanoScope is developed for solid-state nanopore / resistive-pulse work (not Oxford Nanopore DNA
basecalling). See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.*
