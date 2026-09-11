# Recording-specific open-pore references

NanoScope can use information you supply about the **signed open-pore current** for
one recording and channel. This is optional: **Automatic (existing estimator)**
retains the previous estimator, detection algorithms, noise calculation and defaults.

An open-pore reference is the unblocked current against which events are detected
and measured. It is **not** the red event-level line: that line represents the level
inside an individual event. Changing the reference can change event boundaries,
counts and measurements; editing an event level is a separate curation action.

## Step by step

1. Open the recording and select its current channel. Verify the current gain,
   offset and **pA** units in Calibration. Merely changing a unit label does not
   convert the numerical data. Use the reference for the same voltage, pore state
   and recording conditions under which it was measured.
2. Open **Preprocess → Open-pore baseline** and choose a reference mode:
   - **Known open-pore current:** enter a signed value in pA, for example `-120`
     when the unblocked current is −120 pA. Do not enter its absolute magnitude.
     This value is held constant throughout the recording.
   - **Selected open-pore intervals:** select an unblocked, reasonably stable region
     with the trace's **Measure** cursors, then press **Add current trace selection**.
     Alternatively, press **Add interval** and enter start/end times. Repeat as needed.
     Double-click a time to edit it; select a row and use **Remove selected** to remove it.
3. For selected intervals, choose how to construct the reference:
   - **Constant reference (stable recording)** pools all selected samples and takes
     their median. Longer intervals contribute more samples; it is not a median of
     equally weighted interval medians.
   - **Linear between interval anchors (drift)** needs at least two intervals. It
     interpolates their median currents at their sampled centre times. Before the
     first centre and after the last, it holds the nearest level constant. Select
     anchors throughout the recording; interpolation does not establish what
     happened in long unobserved gaps or across abrupt pore changes.
4. Add optional source/notes, such as the notebook measurement or why these intervals
   are believed to be open pore. Leave event polarity on **Auto**, or explicitly
   choose the direction appropriate to this recording. The sign of the reference
   does not itself force event polarity.
5. Press **Preview reference and detection…**. This runs the selected Schmitt or CUSUM
   detector with the current settings without replacing the event table. Inspect
   the conditioned trace, reference and event-level markers, zoom as needed, and
   expand **Details** for warnings and construction assumptions. The trace is
   min/max-decimated for display; numerical detection uses the full recording.
6. Close the preview and press **Detect** to apply the settings deliberately. A
   changed reference requires a fresh detection, including fresh residual noise and
   Auto polarity, not a depth-only remeasurement. If the existing table contains
   hand curation, NanoScope asks before replacing it. Cancel keeps that table;
   **Edit → Undo** restores a replaced curated table.
7. Review the resulting events, then save the project. The analysis stores the
   reference recipe, notes, fingerprint and construction details. Reopening restores
   the recipe; selecting a different recording/channel does not copy it there.
   A reference entered into the form but not used for detection is only a draft.

Interval times are seconds from the start of the currently loaded recording,
independent of displayed clock offsets or trace alignment. Start is included and
end excluded. Intervals must be ordered, non-overlapping and within the recording.
Each needs at least five samples; that is a numerical minimum, **not evidence of
adequate duration or statistical confidence**. Invalid ranges are refused, not clipped.

## Supported combinations and limits

- Initially supported for **Schmitt and CUSUM**, on calibrated pA channels.
- Runtime baseline subtraction, high-pass filtering and normalization must be
  disabled explicitly. Those stages alter the current's reference/domain. NanoScope
  does not turn them off silently. Other supported filters can remain enabled;
  interval medians are then measured on the same conditioned trace used by detection.
- Automatic baseline-window proposals, Batch detect, randomized null controls,
  threshold sweeps and recovery studies do not accept user-reference mode. A
  reference for this recording is not automatically a reference for another trace
  or a randomized surrogate. ML use has not been validated for this mode.
- The older Baseline Correction comparison/export workspace is a separate workflow;
  use the dedicated reference preview with these modes. Event/project exports retain
  the recorded reference provenance.
- The supplied current and selected intervals are **user assertions**, not verified
  ground truth. A quiet blocked plateau can still be mistaken for open pore. A
  constant reference cannot follow drift; linear anchors cannot establish nonlinear
  drift between them. Noise remains the detector's existing whole-residual estimate,
  not a new local or selected-interval noise estimator.
- This feature addresses reference ambiguity when the user has additional evidence.
  It does not guarantee recovery of low-SNR, bandwidth-limited, overlapping or other
  difficult events. Review results and benchmark the settings appropriate to the data.

Warnings belong in the surrounding details/readouts, not printed inside plots.
