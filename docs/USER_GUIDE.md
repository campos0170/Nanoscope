# NanoScope — User Guide

A complete walkthrough of the application, from opening a recording to exporting a curated event
population. This is a *draft* and tracks the app as of the current build.

> ## Before your first detection — measure the recording
>
> Every depth in this app is measured against an **open-pore baseline**, and the window that
> estimates it is a property of **your pore**, not of the app. Across four reference recordings on
> this bench the right window ranges from **0.03 s** (a gating pore) to **2.0 s** (a pore with a
> 521 ms partial clog) — a factor of 67. The app ships a default of **0.5 s / 0.1 s**, and on the
> gating pore that default is **3.3× worse than the estimator it replaced, with every health check
> reading `ok`.**
>
> So: open the **Preprocess** tab, press **"Measure this recording"**, look at what it proposes, and
> press **"Use these values"**. It usually takes a few seconds — measured 2 s on a 3 M-sample recording
> and 6 s on 9 M samples — and the button reads "Measuring…" while it runs, so a two-second wait is the
> click landing, not the click failing. §5a explains what the two windows mean and how to read the
> proposal.
>
> The app cannot do this for you — it can only tell you afterwards that nobody did it, which it does,
> on every table measured at the untouched default. This is the single largest source of wrong
> numbers in the program, and it is not a bug.

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

The main window has three regions:

- **Left dock — Files.** A tree of open recordings grouped by applied voltage. Each recording
  expands into its channels; a checkbox next to each channel toggles it in the viewer, and a colour
  swatch sets its trace colour. Dead channels are greyed out.
- **Centre — the workflow strip** (`Open › Calibrate › Filter › Detect › Curate › Analyze`, showing
  where you are), then the **analysis launcher** row, then **the trace view** with a **navigator**
  strip beneath it (a full-extent overview with a draggable region box).
- **Right dock — Inspector**, a tabbed panel: **Calibration**, **File Info**, and **Analysis**.

The **analysis launcher** is the Analyze stage spelled out: one button per analysis surface —
**Populations**, **Density / ensemble**, **Power spectrum**, **I-V / conductance**,
**V-dependence** and **Export figures**. A button you cannot use yet stays on the row, greyed;
hover it to read what the surface does and why it is not available, or click it to have the reason
posted in the status bar. Each button opens exactly what the matching **Analysis** menu item does.

The Inspector's **Analysis** tab is where the work happens, and it has its own four-tab strip:
**Preprocess** (the filter chain *and*, at the top, the open-pore baseline — §5a) · **Detection** ·
**ML detector** · **Manual measurement**. All four are ways of producing events, so they feed the
**shared event-table surface** below the strip. When the recording holds more than one detector
result, the **Event table** selector chooses which stored table is shown; changing that selector
changes only the view and does not replace or merge any result.

Menus: **File** (Open, Open/Save Project, Create segment from selection, Close document, Quit),
**Edit** (Undo / Redo — §9b), **View** (Overlay/Stacked, Zoom Back/Forward, Reset zoom, drag mode,
Panels), and **Analysis**, grouped into **Detect** (Run detection, Batch detect, ML model),
**Curate** (Event inspector, Measure event, Add measurement to list, Compare baseline estimators —
§9a), **Explore this recording** (Plots, Event plots, Event gallery, Reconcile detector results,
Event labels, Power spectrum), **Compare recordings** (I-V, Voltage dependence) and **Export**
(events, training data, figures).

Closed a dock by accident? Reopen it from **View ▸ Panels ▸ Files / Inspector**.

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
| **A CSV whose sample interval changes mid-file** | Both halves: the interval near the end against the interval at the start (e.g. `0.001 s against 0.0001 s (10×)`), and the file's true span against the span that rate implies (`0.649 s where that rate implies 0.1999 s (3.247×)`). The rate is *reported*, never silently corrected — so every dwell time from that file is wrong by about that factor. |
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
  **Power spectrum** and **I-V** act on, and what the calibration chip and the workflow strip
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
  - *Mouse wheel*: zooms the **time axis only**.
  - The **− / +** buttons in the **Scan** strip zoom out / in one step around the centre. Unlike
    Reset, **−** keeps you in local context instead of jumping straight back to the full recording.
  - *Double-click*: resets to the full trace.
  - **Zoom Back / Forward** (Ctrl+← / Ctrl+→) step through your view history — dive into an event,
    reject it, and pop back to where you were browsing.
  - **Reset zoom** (Ctrl+0). **View ▸ Drag draws zoom box** toggles between box-zoom and pan.
- **Scan at the current zoom.** The **◀ / ▶** buttons beside the zoomed-in scrollbar move to the
  previous / next view while preserving the current scale. Each move advances by 90% of the window,
  leaving **10% overlap** so an event at a boundary is not lost. If the Measure cursors are active,
  they move forward with the view, ready to bracket the next event. **Shift+← / Shift+→** perform the
  same scan, and **Shift+↑ / Shift+↓** zoom in / out, while focus is on the trace.
- **Crosshair.** Hovering shows the time and the value at the nearest sample, in the channel's units.

---

## 5. Filtering

Filters are **pre-detection signal conditioning**: they clean the trace fed to the detector, and the
**"show processed trace on the plot"** toggle previews exactly the signal the detector sees — every
enabled stage of the chain, then the low-pass. All filters are **zero-phase** (they do not shift
event timing). Each can be enabled independently, and whatever you enable runs in one fixed order —
notch, median, high-pass, baseline drift, running mean, Savitzky-Golay, **Normalize** — with the
low-pass applied last, inside the detector. The table below is ordered for reading, not for execution.

| Filter | What it does | When to use it |
|---|---|---|
| **Low-pass** (8-pole Bessel) | Removes high-frequency noise above a cutoff. Applied inside the detector so rise-time and filter-limited flags stay correct. | Almost always — set the cutoff to your acquisition bandwidth. |
| **High-pass** (Butterworth/Bessel) | Removes very low frequencies / DC drift. **Warning:** distorts square translocation events (droops the flat top). | Special cases only; prefer *Baseline drift* for flattening a wandering baseline. |
| **Notch** | Removes mains hum at a frequency (default 60 Hz) and its harmonics. | When line pickup contaminates the trace. |
| **Savitzky-Golay** | Polynomial smoothing that preserves peak shape better than a moving average. | Gentle smoothing without a hard low-pass. |
| **Running mean** | Simple moving average. | Fast, coarse smoothing (blurs edges). |
| **Median** | Sliding median. **Removes impulse/spike noise while preserving square event edges.** | Despiking a noisy trace before detection. |
| **Baseline drift** | Estimates the slow open-pore level and *subtracts the drift*, keeping event depths intact (not a high-pass). Methods: block-median, percentile, ASLS. | Long recordings with electrode/temperature/clogging drift. |
| **Normalize** | Rescales the whole conditioned trace onto a **dimensionless** scale — `z-score`, `robust z` (÷ the same 1.4826·MAD sigma the detector thresholds on) or `min–max` → [−0.5, +0.5]. Applied **last**, after every other stage. **It divides the picoamps out:** depths, baselines, thresholds and every exported column downstream are still *labelled* pA and are no longer in pA, and nothing downstream can put them back. Off by default; while it is on an amber banner stays up on every page of the Signal-filters group. | Feeding a model, or putting recordings on a common scale — **never** to measure a conductance. |

Notes:
- The **Baseline drift** window must be *much longer than your longest event* (it flattens drift, not
  events). Of its methods, **block-median** and **percentile** flatten drift effectively; **ASLS** is
  a stiff global fit and is weak at default settings.
- The low-pass and Savitzky-Golay are both smoothers — usually you use one or the other.
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

At the top of the same **Preprocess** tab is **Open-pore baseline**. This is not signal conditioning
and it is not cosmetic: it is the estimate of `I₀(t)`, the current your pore carries with nothing in
it, and **every depth in the app is measured against it.** It also feeds the detector's threshold, so
it decides where events *start and stop*, not only how deep they are.

The estimator is a two-pass masked median with two windows you set:

| Control | What it decides | The failure it causes when wrong |
|---|---|---|
| **Rough window** | Pass 1 — **which excursions count as events** and get masked out before the baseline is measured. | It must be several times the longest state you want treated as an event. A rough window shorter than the state it is sitting in *becomes* that state and nothing gets masked. On the −600 mV apo file, the 521 ms partial clog is **15 % masked at 0.5 s and 99.96 % masked at 2.0 s**. |
| **Base window** | Pass 2 — **how fast the baseline may follow the trace.** | It must be shorter than the shortest genuine conductance level dwell, or a gating pore's steps get smoothed *into* the baseline instead of being measured against it. |
| **Decimation** | Derived from the base window, shown but never set by hand. | It is the cost throttle (cost goes as 1/decimation²), not a quality knob. |

### Press "Measure this recording"

The button probes the **whole channel** — usually a few seconds: measured 2 s on a 3 M-sample
recording and 6 s on 9 M samples, with the button reading "Measuring…" meanwhile — and measures three things from
the trace itself: a robust noise σ, the distribution of excursion dwells (with short gaps closed, or
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

> **This behaviour is newer than the shipped `NanoScope.app`.** The bundle on the DMG still carries
> the old answer, which was to substitute a 10 ms block median: on the same synthetic test it
> reports **0.0094×** the true depth at 3,072 samples — a 100-fold understatement — and it refuses
> outright below 500 samples with a message naming a `window_s = 0.01 s` you never set. Anything
> at or above 5,000 samples is identical in both, which is every ordinary recording.

---

## 6. Event detection

**Analysis ▸ Run detection** (Ctrl+D). Detection is a **two-threshold (Schmitt) trigger** on the
current channel: an event starts when the signal crosses `baseline ± k_in·σ` and ends when it
returns within `baseline ± k_out·σ`, where σ is the **MAD** noise estimate. Polarity (blockade vs
enhancement) is **measured from the data**, not assumed.

That `baseline` is the open-pore estimate from **§5a**, so it sets where events *begin and end*, not
only how deep they are — which is why §5a comes before this section. If you have not pressed
**"Measure this recording"** yet, do that first; a re-detection afterwards is one Ctrl/⌘Z away from
being undone (§9b), but a dwell time measured against the wrong baseline is not recoverable by any
later step.

Detection is enabled only on a **current channel with a known sample rate**; otherwise the button is
greyed with an honest reason (e.g. a legacy TDMS file has no rate → dwell would be undefined).

### Parameters (Detection controls)

| Parameter | Effect | To reduce over-detection |
|---|---|---|
| **k_in** (enter, in σ) | Threshold to *start* an event. The primary knob. | Raise it (5 → 6, 7). |
| **k_out** (exit, in σ) | Threshold to *end* an event; must be `< k_in` (hysteresis). | Keep well below k_in (default 2.0). |
| **min dwell** | Events shorter than this are dropped. | Raise it to kill brief noise blips. |
| **max dwell** | Events longer than this are flagged (kept, not accepted). | Set to exclude clogs/wander. |
| **merge gap** | Events closer than this are merged into one, transitively along a chain. In **milliseconds** (0–1000 ms; `0` = off, and this is the one merge control where 0 really does mean off). A run whose merged span would be mostly open pore is **refused** rather than merged — see *Merging fragmented events* below. | Raise to combine fragmented events. |
| **Low-pass cutoff** | Smooths before thresholding. | Enable / lower to suppress noise crossings. |
| **Polarity** | auto / blockade (−1) / enhancement (+1). Beside it, a **measured** line reports what the last detection actually found — *blockade (down): 18 down / 0 up (confidence 1.00, measured)*, or *not yet measured* on a recording nothing has been run on. | Leave on auto; read the measured line before overriding it. A wrong override does not raise an error, it simply detects almost nothing. |

After a run the **Summary** shows counts and rates, the event table fills, and events appear as a
single batched **span overlay** on the current pane.

### Merging fragmented events

A noisy or multi-level translocation that briefly recrosses `k_out` is reported as several adjacent
rows, each with a fraction of the true dwell. **Merge sub-events**, in the Analysis panel's controls
column, stitches them back together *after* detection, so you do not have to re-tune thresholds and
lose the genuinely short events:

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
| `depth_pA` | Blockade depth = the **median** in-event deviation from baseline (median, not max — max is biased by dwell). |
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

1. Zoom to a useful inspection scale with box-zoom, the mouse wheel, or the **− / +** buttons in the
   Scan strip. At event scale the Navigator changes from **Full recording** to a **9× local-context**
   lens, so you can still pan precisely without zooming all the way out.
2. Click **Measure event (cursors)**. A draggable region appears on the current-channel trace — its
   **two edges are your boundary cursors**. (Or use **Snap to view** / **Snap to selected event**.)
3. Drag the edges to bracket the event; everything recomputes live. The blue band in the Navigator
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
     run or select a source detector rather than failing silently.

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

- **Waveform.** The raw current around the current event, with the event region shaded, the baseline
  and in-event level lines drawn, and a readout of dwell, depth, ΔI/I₀, SNR, rise/fall, level count,
  and filter-limited flag.
- **Navigate.** Prev / Next, an "Event *i* of *N*" index, and a filter (All / Accepted / Rejected).
  Keyboard: **←/→** move, **A/R** accept/reject, **Space** toggles.
- **Context.** A `Context:` combo sets how much recording is *loaded* either side of the event —
  **± 3 ms** (the default), **± 30 ms**, **± 300 ms** or **Whole trace** — and the choice persists as
  you step, because it is a working preference rather than a per-event one. Below ±3 ms was once the
  hard-coded pad and therefore a ceiling: panning inside a 6 ms slice cannot answer *"was the pore
  quiet before this?"*, only loading more samples can.
- **Show in trace** (`T`) centres the main trace on the event you are looking at — the same jump as
  double-clicking its tile in the gallery — and **Gallery…** opens the contact sheet. The three event
  surfaces all link to each other now; none of them is a dead end.
- **Curate.** Accepting/rejecting writes to the shared event table, so it flows to the table
  checkbox, the trace span overlay, the summary counts, the plots, and exports — all at once, in both
  directions. Navigating also zooms the main trace to the event.
- **Save without leaving the window.** The status bar names the current project and says whether it
  has unsaved changes. Press **Save Project** (use its arrow for **Save Project As…**), **Ctrl/⌘+S**,
  or **Ctrl/⌘+Shift+S**. These controls use the same guarded project writer as the main File menu;
  cancelling or failing a save leaves the unsaved warning in place.

---

## 9a. When the open-pore baseline changes

Every depth in the table is measured against an **open-pore baseline** — the current the pore
carries with nothing in it — and which estimator produced that baseline is stamped on the result,
next to the calibration stamp.

If a stored detection's stamp does not match the estimator now set in **Preprocess ▸ Open-pore
baseline** (or if the table predates the stamp entirely), the analysis panel shows an amber banner
naming the old estimator, and the affected columns — `baseline_pA`, `depth_pA`, `frac_depth`,
`ecd_pA_s`, `mean_dev_pA`, `max_dev_pA`, `snr`, `rise_s`, `fall_s` — carry a ⚠ in the table header
and are shown in amber. `rise_s` and `fall_s` are on that list because 10 %/90 % are fractions of
each event's *own* depth, so a changed depth moves them; they were once documented as unaffected,
and that was wrong. **Nothing is re-measured.** The values on screen are exactly the values that
were stored.

- **Analysis ▸ Compare baseline estimators…** measures, on *this* recording, what a re-measure
  would change: old vs new `depth_pA` per event against the y = x line, the ratio distribution, and
  how many events have a stamped baseline that moves more than 10 pA. On a recording without the
  pathology this shows ~1.0× everywhere. Opening it changes nothing.
- **Re-measure depths** (on the banner, or from inside that report) rewrites the amplitude columns
  against the configured estimator. It does **not** re-detect: every event span keeps its
  boundaries, and every accept/reject and hand-written flag survives — including ones you make
  while it is running. A calibration change is different: its banner offers **Re-run**, which
  re-detects, and the current curation does not survive that.

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
events"*, *"Undo Resize event mask"*, *"Undo Accept all events"*. **Redo** is Ctrl+Shift+Z / ⇧⌘Z.
Everything that can change an event table is covered: a re-detection, a merge, an accept/reject, a
batch Accept-all / Reject-all, a mask resize in the Event Inspector, a hand-added missed event and
a re-measure. An undo restores the table **exactly** — the accept/reject decisions, any flag token
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
steps, the Edit menu says so by name (*"Undo history for “<name>” is at its 30-step ceiling — N of
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
removed, or otherwise no longer matches its stored fingerprint, a stale-lineage banner keeps the
snapshot viewable and exportable but tells you to inspect the source tables and reconcile again.

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

**Analysis ▸ Plots…** opens an interactive plotting window over the per-event tables the app holds:
**one source per (recording, channel, detector result) with at least one row** — Schmitt, CUSUM, ML,
and Reconciled tables are offered separately, segments included — plus your **manual measurements**.
**Use accepted events only** is ticked by default; untick it to plot the rejected rows alongside the
kept ones.

### Comparing two of anything

The source picker is a **checkable list**, not a single choice. Tick two or more and the histogram
and scatter **overlay** them, each in its own colour-blind-safe colour, with a **legend** naming
each. That is the two-pore comparison and the two-segment comparison; a segment named identically
to its parent still gets its own row, because sources are identified by their stable project key
rather than by their display name.

Three things happen automatically so the comparison actually means something:

- **Shared bins + normalization.** All drawn sources use ONE set of bin edges, and the y quantity is
  selectable — **count**, **density** or **probability per bin**. It switches to
  <!-- Renamed from "density (area = 1)" by S6-03: the heights are per the column's STORED
       unit (per second for a dwell) while the x ticks follow the Time-unit combo, so the
       drawn area was 1000 on the default ms axis and 23.3 under the shipped log-x dwell
       template, where no constant factor can repair it. The y axis now names the unit —
       `density (1/s)`, `density (1/pA)` — and Details says where the drawn area is not 1. -->
  *density* as soon as a second source is shown, because overlaid raw counts from a 5000-event and
  a 200-event recording are two decades apart and the short one vanishes. (Pick a normalization
  yourself and your choice sticks.)
- **One fit per source.** With a fit enabled each source is fit **separately, using its own
  detector's resolution** — its own `min_dwell_s`, its own filter-limited exclusion, its own
  right-truncation state. Two sources are never pooled into one number, and one recording's
  detection floor is never applied to another's dwell times. Every τ (with n) is named by source in
  the legend and in **Details**; if the sources were detected at different resolutions, the window
  says so.
- **Colour has one owner.** With more than one source drawn, colour means **source**, and the
  cluster split is disabled (and says why). With a single source, colour means **cluster**.

> **Two recordings on one axis are not automatically comparable, and the window says so once.**
> Before it draws, it asks one question about the ticked sources and prints one verdict: are they in
> the same **merge state**; were they detected under the same **baseline window**; and — for an
> absolute-current column (`depth_pA`, `amplitude`, `ecd_pA_s`, …) — under the same **current
> calibration**? Each failing half is spelled out in **Details**, most fundamental first, and the
> plot's own title carries `⚠ NOT COMPARABLE across N sources: <columns>` — in the title because the
> title is the only thing **Save image…** keeps.
>
> The advice is reconciled rather than stacked. `frac_depth` (dI/I₀) survives a *gain* disagreement,
> so it is offered then. But dI/I₀ is `depth_pA / |baseline_pA|`, so a *baseline* disagreement makes
> it exactly as incomparable as the depth, and it is dropped from the sentence — what is left is
> `dwell_s, or a count`. On this bench, re-detecting the −700 mV apoferritin recording under the
> −600 mV recording's baseline windows moved its median `frac_depth` by 11.3 %.

### Plot types

- **Histogram** — any numeric column, adjustable bins, linear/log x, linear/log y, a **Normalize**
  choice (`count` / `density` / `probability per bin`), a **Colour by** choice
  (`None` / `Cluster`), and an optional **fit overlay**. Five families: **exponential** (the classic
  escape model — monotonic, cannot peak), **gamma (peaked)** and **inverse-gaussian (first
  passage)** (densities *with* a mode, for a dwell histogram that peaks away from zero),
  **exponential + gamma (bimodal)**, and **gaussian-mixture**. The three dwell families are all
  truncated at the detector's `min_dwell_s`, so their BIC values compare directly. Choosing a family is a **proposal**: press
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
- **KDE (density)** — 2-D kernel-density contours of any two numeric columns; one contour set per
  ticked source, in that source's colour, on shared axes. It is a **plot type here beside the others**,
  with the same source overlay, styling and **Save image…** as the rest of the window — not a separate
  surface. It defaults to the signature axes, `dwell_s` (log x) against `frac_depth`, and a
  **Contours** spinbox sets the lines per source. Each source is contoured against its **own** peak
  density, so a rare population keeps its shape beside an abundant one. It needs at least 5 usable
  points per source; anything dropped is disclosed in **Details**, along with each source's bandwidth
  factor. (**Analysis ▸ Export density figure…** is still the separate publication export — the same
  view with marginal distributions, written as PNG + PDF + a CSV of the plotted points.)
- **PCA** — projects a chosen feature set to two components, coloured by cluster (with a legend),
  and shows the explained variance.

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
  read in the axis unit too. **Axis range** below is the exception: it is in data units.
- **Axis range** — type explicit X/Y limits (data units, converted to log internally on a log axis)
  that persist across redraws; **Auto** releases them. The plot is also mouse zoom/pan-able.
- **Style** — a custom title, font family + size, data-series and fit colours, marker size, line
  width, a Dark/Light background (Light = white canvas for publication), and an **export size**.
- **Legend** — a **Show legend** toggle, a **Legend position** (upper right, the default, / upper
  left / lower right / lower left; the legend is drawn over the data, so move it off whichever
  quadrant your points occupy), a legend font size and a **(n = …)** toggle. The legend **labels are
  editable**: double-click a row in the source list and type, and that series is renamed everywhere it
  is named — legend, fit lines, Details and the plot title. The real name is kept underneath, so
  clearing the edit back to empty restores it rather than legending a blank.
- **Save image…** exports the current plot as a PNG at the chosen resolution.

---

## 11. Noise spectra (PSD)

**Analysis ▸ Power spectrum (noise / PSD)…** opens the log-log current-noise spectrum every
nanopore paper uses to grade a pore. It runs on the **raw calibrated trace** (no detection needed,
no filter chain applied), over a chosen quiet open-pore segment, and separates the three noise
regimes: the low-frequency **1/f (flicker)** power law (α and the 1 Hz magnitude are fit and the
line overlaid), the mid-decade **thermal / white floor**, and the high-frequency **dielectric /
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
(α, white floor, band RMS), its own sample rate and Nyquist, its own units, and its own band-RMS
read-out. A comparison recording shorter than the chosen window is analysed over what it has, and
says so.

> The **shapes** of two spectra are always comparable (α, where the floor sits, where the capacitive
> rise starts). Their absolute **heights** are not, until the recordings share a real current
> calibration — a PSD value is a squared current, so it carries the amplifier gain. The window
> prints that caution, and calls out curves that are in different units.

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
  exported CSV carries 23 columns — those 20 plus three export-only columns, `orig_start_idx`,
  `orig_end_idx` and `added` — and the sidecar's `columns` list names all 23. A new column is
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
- **Analysis ▸ Export figure…** — publication figures (matplotlib) for the current analysis, each as
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
- **Plots ▸ Save image…** — the current interactive plot as a PNG.

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
and its sidecar, every exported figure, and **Save image…** on all five plotting panels. The note
appears as a small dialog titled after the command — *Export events*, *Export figures*, *Save plot
image* — scoped to the files that command just wrote, and said once per file per session. An
ordinary save of an ordinary file says nothing at all; silence here means *nothing unusual*, not
*not checked*.

And because nothing is written until the render succeeds, a failed image export no longer claims
"the file was not created" over a file that is sitting there. It tells you the file in that folder is
the **earlier** export, with the date it carries, untouched but now older than the figure on screen.

**The overwrite prompt names every file, not just the one you typed.** A figure export writes up to
twelve files under one base name, and an events export writes a `.meta.json` beside the CSV; the
prompt lists whichever of them already exist. **Cancel is the default** — pressing Return or Escape
keeps your files, and replacing them takes a deliberate click on **Replace**. In builds before this
one Return meant Replace.

---

## 12a. A complete worked example

The manual's **Walkthrough 5 — ML detection, replicated step by step** reproduces a full analysis
on the three apo-ferritin reference recordings — open-pore measurement, ML detection with a Keras
student model, scatter, KDE, histograms with fits, PSD, PCA and GMM/t-SNE/UMAP substructure — with
the exact numbers this bench produced, screenshots of every surface, and the three traps a
first-time user hits (wait for the proposal before Apply; detect before switching recordings; read
the fit's KS verdict in Details before quoting its parameters). Follow it to verify your install
end to end.

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
| Ctrl+0 / Home | Reset zoom |
| Ctrl+← / Ctrl+→ (or Alt+← / Alt+→) | Zoom Back / Forward (view history) |
| Ctrl+E | Centre the view on the current selection |
| **Trace focused: Shift+← / Shift+→** | **Scan to the previous / next view at the current zoom, with 10% overlap; carry active Measure cursors with it** |
| **Trace focused: Shift+↑ / Shift+↓** | **Zoom in / out one incremental step around the centre** |
| Double-click trace | Reset to full extent |
| Mouse wheel on trace | Zoom time axis |
| **Event inspector:** ←/→ | Previous / Next event |
| **Event inspector:** A / R / Space | Accept / Reject / toggle |
| **Event inspector:** T | Centre the main trace on the event on screen |
| **Event inspector & Event gallery:** Ctrl/⌘+S | Save the complete project through the same writer as File ▸ Save Project |
| **Event inspector & Event gallery:** Ctrl/⌘+Shift+S | Save the complete project under a new name |
| **Event table:** j / k | Next / previous row |
| **Event table:** a / r / f | Accept / Reject / flag the current row |
| **Event gallery:** A / R / Space | Accept / Reject / toggle every selected tile |
| **Event gallery:** PageDown / PageUp | Next / previous page |
| **Event gallery:** Home / End | First / last page |
| **Event inspector & Event gallery:** 1–9 | Apply the matching label from the palette (**Analysis ▸ Event labels…**) — same digits, same meanings, in both windows |
| **Event inspector & Event gallery:** custom one-character label hotkey | Apply that palette label (reserved and duplicate keys cannot be saved) |

---

## 14. Known limitations & caveats

- **Reconciliation Preview currently runs synchronously.** The legal candidate and edge caps keep
  ordinary previews bounded, but there is not yet a truthful mid-computation Cancel button. Review
  the source sizes before reconciling an unusually dense detector output.

- **No open-pore baseline default can be right, and no code change can make one.** The correct
  window is a property of the pore. The app ships a floor, warns when a table was measured at it,
  and cannot measure the recording for you. **Press "Measure this recording" (§5a) before you trust
  a depth.** This is the largest item on this list.
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
  holds, and the ΔI/I₀ each one implies — every depth is the same number either way, and only ΔI/I₀
  moves. The warning is the whole defence, because the two readings are the same samples:
  **compare the baseline with the open-pore current you know for this pore** — the current just
  after the voltage step, or an unclogged stretch — and analyse that stretch instead if the two
  disagree.
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
- **High-pass** distorts square events (use Baseline drift instead for flattening drift); **ASLS**
  baseline flattening is weak at default parameters — prefer block-median/percentile.

---

*NanoScope is developed for solid-state nanopore / resistive-pulse work (not Oxford Nanopore DNA
basecalling). See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.*
