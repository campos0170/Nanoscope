# Measurement-quality and acquisition-bandwidth advisories

Source implementation update: 2026-09-12. These controls require the updated source/app;
they are not yet in the previously built application or illustrated PDF manual.

## What changes—and what does not

Schmitt and sequential-CUSUM detection now record additional advisory evidence about
baseline/polarity consistency and possible local baseline-following during events.
This is not a new detector or an automatic validation/rejection rule. Detection
parameters, event boundaries, acceptance, amplitude estimators, existing numerical
columns and fit-selection defaults are unchanged by these additions.

No SNR cutoff is introduced. No merge gap is changed. A short or low-SNR candidate
is not rejected merely because of these checks. The user must still establish the
physical open-pore reference for the particular recording.

## Read the quality information

1. Select the recording/current channel and establish calibration and the open-pore
   reference. Use the existing Automatic, known-current, or selected-interval workflow
   appropriate to the evidence you have.
2. Run detection normally.
3. In Analysis, open **Results → Summary**. The analysis-warning link also opens this
   summary. Read the warning details below the summary table; the advisories are not drawn into plots.
4. A baseline/detector polarity disagreement means the polarity-dependent baseline
   construction and the final event direction disagreed. Inspect the trace before
   accepting amplitudes. Agreement is not evidence that either selected the true
   open-pore plateau.
5. Local baseline-following above the 10% advisory threshold flags a suspected bias
   relative to nearby baseline estimates. This is a diagnostic threshold, not a
   calibrated 10% error bound against ground truth. It uses the existing intrusion
   statistic with two-sided flanks (5 ms, separated by a 0.2 ms pad).
6. If those flanks are incomplete, overlap detected events/excluded regions, or do
   not support a defined depth, the check says **not assessed**. Undetected events can
   still contaminate apparently available flanks. A zero-event result is also explicitly
   unassessed, not a clean bill of health.
7. Known-current and selected-interval references remain user assertions. Their
   correctness is not automatically certified. Check the sign, current calibration,
   timing, and suitability for the selected recording/channel.

Quality status is separate from the existing accepted/rejected and human-review states.
Accept/reject edits alone do not remeasure the data. Geometry, amplitude or reference
changes invalidate old quality evidence; the details and exported provenance identify
it as stale. Rerun detection to obtain a fresh report when appropriate. Re-detection
is a deliberate analytical action, not something triggered by presentation edits.

Per-event diagnostic records are stored in the `measurement_quality` provenance
block, keyed by event ID and sample boundaries. The report also carries recording-level
messages when the event table is empty. Legacy and unsupported paths without the new
assessment remain explicitly unassessed; they are not retroactively certified.

## Enter acquisition bandwidth

1. Select the recording and open **Inspector → File Info**.
2. Review **Acquisition low-pass (metadata only)**, then click **Acquisition bandwidth…**.
3. Choose the exact channel in the dialog.
4. Review the original file declaration. Its tooltip retains reader notes and raw
   header evidence. Choose one of the following:

   - **Use original file declaration**: remove a user override and use the header.
   - **User-provided cutoff**: enter the known positive cutoff in **Hz**, optionally
     specifying the filter type and order if genuinely known.
   - **Unknown / not verified**: explicitly decline to assert a cutoff.
   - **Reported stage bypassed**: state that the particular filter stage was bypassed,
     without claiming that the whole acquisition chain was unfiltered.

5. Record the amplifier/settings source or uncertainty in **Provenance note**.
6. Press **Apply** for that channel. Apply before switching channels if you want to
   retain its edits. Close the dialog and **save the project** to retain the override.

This is metadata only: it does not filter samples, change calibration, rerun detection,
or update existing results retroactively. Newly created results snapshot both the
effective declaration and the original file header. Already-created results retain
their original snapshot; Summary identifies it as the value recorded at detection.
No sidecar is written beside the raw recording. Derived segments inherit the parent's
current declaration at creation and can subsequently have their own project override.

ABF2 metadata is read per channel from the fields exposed by Neo. If multiple hardware
stages are reported, the narrowest reported cutoff is disclosed as a stage setting,
not as the combined effective response. Postprocessing metadata is retained separately
and is not represented as hardware filtering. Unsupported/missing/invalid metadata is
unknown. The current Neo ABF1 reader does not expose the needed cutoff fields, so these
files normally remain unknown until the user supplies reliable information.

## Temporal-resolution limits remain explicit

The existing `filter_limited` flag retains its software-conditioning meaning. It does
not now include a fabricated hardware contribution. A corner frequency alone does not
specify the filter's transfer function, rise time, multi-stage response, or attenuation
of a particular event. Hardware-limited resolution is **not assessed** in this pass,
even when the declared type/order are available. Do not use a missing flag to conclude
that an event's full amplitude or true duration is resolved.

## Figure-coordinate correction

Event-count heatmaps and time-frequency spectrograms now pin their image-axis order
explicitly in both live and saved-snapshot views. This repairs a display mapping that
could inherit the wrong global orientation. Numerical grids, bin edges, marginals and
exported table values are unchanged. Reopening a saved grid uses the corrected mapping;
previously exported image files are not rewritten automatically.
