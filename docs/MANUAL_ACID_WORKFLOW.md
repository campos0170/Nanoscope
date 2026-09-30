# Acid-insulin study: reproduce and curate in NanoScope

Use **Analysis → Incubation study…** in NanoScope **40.15.59** and select **Acid incubation** for a new study. Enter **Study name** and optionally **Sample**, click **Create study…**, and choose a new empty folder. Use **Open saved study…** to restore an existing study, or the open-study list and **Continue selected study** to resume one already open. Acid uses the common launcher and main event-curation tools while retaining its calibrated model-free CUSUM and family-analysis method. General, Zn and Cu share the same launcher and have their own editable preparation and detection settings.

## Reopen the supplied study and change the event selection

1. From **Analysis → Incubation study…**, choose **Open saved study…**, then select the supplied `manual-study` folder containing `workspace.json`. The saved method is selected from the workspace; it is not overridden by the launcher profile. Open the folder, not an individual PNG or ABF.
2. Choose a recording from the top-right selector. Open **4 Review → Show candidate events**. **All candidates** includes candidates rejected by the automatic gate, so those can also be accepted manually.
3. For the main app's full curation tools, click **Curate in main workspace…** on **4 Review**. The prepared trace and events open in the main Event table, Event Gallery and Event Inspector. Accept/reject events, adjust boundaries, add missed events, merge, annotate or Undo using those controls. Check the exact recording named in the persistent **Incubation study** bar. Click **Save curation to study**, wait for successful completion, then **Return to study**. The study preserves its saved Acid measurement definitions and remeasures edited geometry; source annotations remain separate. Quick review also remains available on **4 Review**: select a row and click **Inspect selected event**, or double-click it, then use **Accept selected events** or **Reject selected events**. The shaded interval and table preserve sample bounds and polarity.
4. Click **Confirm event selection for analysis** for every primary recording whose current selection is pending, including when automatic acceptance is unchanged. Confirmation approves the current selection; it does not assert that every event was personally inspected.
5. For family curation before downstream analysis, click **Build event families for review** after confirming all primary recordings. This fits/transfers the reference family model without the later bootstrap and comparison analyses. Then choose a recording and **Show event families**. Select a family and **Show events in selected families** to inspect its members. Use **Accept selected families**, **Reject selected families**, or **Keep only selected families**. Keep-only rejects other families and unassigned events in that recording.
6. Confirm each changed recording again. Open **5 Analyze → Analyze confirmed event selections**. This recomputes results from the selected events. Membership remains tied to the saved reference model, so a rejected family can subsequently be accepted again without silently redefining its labels.
7. Open **6 Figures → Generate study figures**. Choose a figure, then **Style selected figure…**. The ordinary NanoScope figure editor supports titles, labels, fonts, line widths, linked colors, background, legend, image export and Figure Library. Click **Apply appearance**, then **Save styling to study**. **Reset all figure styling** restores the underlying scientific figure. Styling does not change events or results.
8. Click **Save workspace** before leaving. Completed operations also checkpoint automatically. Reopen the same folder to recover committed stages, selections and figure styles. Figure Library entries belong to the main NanoScope project; save that project separately.

The saved baseline study uses automatic acceptance and has no human curation. Your rejection/acceptance decisions intentionally produce different results. Preserve the supplied baseline folder if you want to compare it with a separately curated copy; copy the entire workspace folder, including `arrays` and `runs`, before opening the copy.

## Start again from the raw ABFs

1. Open **Analysis → Incubation study…**, select **Acid incubation**, enter the study name and optional sample, then **Create study…** and choose a new empty folder. Click **Add raw ABF…** and add the four primary acid recordings from `OnlyAcidCondition`.
2. In **1 Sources**, set each recording’s metadata, then **Apply recording details**. Use the following tags exactly when reproducing this study, so the reference setting identifies the correct acquisition. All four use current channel **1** (pA), concentration **100 pM**, and nominal voltage **−300 mV**. Give every distinct physical acquisition its own pore identity unless laboratory records establish a shared physical pore.

| Raw recording | Recording tag | Incubation time | Nominal pore diameter |
| --- | --- | ---: | ---: |
| Whole 30 h ABF, 1200 s | `acid_30h_whole` | 30 h | Leave blank: filename says 16 nm, notebook labels 26 nm; unresolved |
| 18 h conductive ABF | `acid_18h` | 18 h | 18 nm |
| 10 h conductive ABF | `acid_10h` | 10 h | 20 nm |
| 2 h conductive ABF | `acid_2h` | 2 h | 18 nm |

The 30 h Part 3 excerpt is an alternative scope and must not be counted as an independent timepoint alongside its whole recording. The short 26 nm baseline-only recording is not an additional incubation timepoint. If adding either for a separate check, mark it **Supplementary; exclude from primary series**.

3. For each primary recording, open **2 Prepare**. The reference defaults use sinusoid mains removal at nominal 60 Hz, minimum prominence 5, 1 s fit windows, excursion mask 3.5 sigma, exact 2 s rolling median baseline evaluated every 10 samples, and conditional fourth-order causal Bessel matching to 0.273 ms when the measured roll-off exceeds 2500 Hz. Click **Apply preparation settings** only if you edited them, then **Prepare selected recording**. Inspect **Preview prepared trace** and the source/method metadata. This whole-record overview samples the trace and can omit brief events. Use **4 Review → Inspect selected event** for exact event samples and interval bounds, and the saved min/max trace overview for full-record event visibility. The algorithm may retain the signal unchanged by the mains step when no qualifying line is detected.
4. Open **3 Detect**. Keep the reference defaults: both pulse/level CUSUM scales, opposite-polarity detection enabled, primary set `cusum`, timescale-calibrated peak gate at 4 sigma using the local baseline, reported peak amplitude using the global baseline, and the saved injection/grid settings. Click **Apply detection settings** after editing, then **Calibrate and detect selected recording**. Repeat preparation and detection for all four recordings. The engine records any fallback to detector-matched events when calibration does not meet its budget; inspect this in methods and calibration figures.
5. Follow event/family curation steps 2–6 above. To reproduce the supplied automatic baseline, leave acceptance unchanged and confirm the default selection for each recording. The interactive confirmation remains a selection approval, not an assertion of detailed human review.
6. In **5 Analyze**, the reference recording is `acid_30h_whole`; reference refitting defaults off after the first saved fit, burst exclusion is on, validation uses 8 time blocks and 200 block-bootstrap draws, and modeling requires at least 100 selected events. Long-event morphology is enabled with 0.5 ms minimum dwell and 100-event minimum. Small selected sets are explicitly reported as insufficient for the relevant modeling stages. Apply any edited settings, analyze, then generate/style/export figures as above.

**Save curation to study** is an explicit snapshot. **Return to study** alone does not save edits, and **Save Project** preserves the main workspace separately. Later main-workspace changes require saving to the study again. Compatible reentry preserves unsaved curation; if the study changed upstream, open its current events again. After saving changed events, confirm each affected selection, rebuild any retired event families, rerun analysis and regenerate the affected figures. Keep the entire workspace with the main project because it references the study's prepared review trace.

Changing preprocessing or detection settings retires incompatible event decisions, family assignments and downstream outputs. Changing scientific analysis settings retires the affected analysis/family results. Unapplied control drafts block scientific execution; they are saved separately and must be applied explicitly. Appearance edits preserve the scientific selection.

## Future studies and laboratory metadata

For a different sample or incubation condition, choose **General incubation** unless the calibrated Acid method is the intended documented analysis. Enter the actual time, concentration, voltage, pore identity, channels and intervals. General supports signal or main-workspace detection with an optional trained model. Profiles do not identify chemistry or establish the suitability of a scientific method.

These instructions describe analysis of existing recordings, not a complete laboratory incubation recipe. Incubation temperature, acid composition/pH, preparation and agitation must be taken from the experimental record; they are not established by these settings.

## Interpretation limits

These are exploratory descriptive comparisons across acquisitions, with physical pore and recording effects confounded with incubation time. A family/component number is not a validated molecular species. Matching to a Bessel response is an analysis assumption; it does not establish the complete original acquisition bandwidth. Whole 30 h declares a 1000 Hz signal-conditioner stage in the header, while the amplifier/combined response is unresolved. The other processed ABFs do not provide complete acquisition-bandwidth information through the reader.

The current acid workflow implements the latest model-free notebook route. It does not run the older supplied student network. Scientific settings, source identities, selected-event fingerprints, frozen family model and saved numerical figure recipes are recorded in the workspace.

## Reading example-trace labels

The descriptive text attached to a family gives its median properties. An individual example trace can differ from those medians. Shading shows that event’s actual detected bounds; use its event-table row for the individual dwell and amplitude. Distribution figure eligibility requires at least five events per family, and density curves require at least 25 events. Smaller eligible groups can appear as points in joint plots without a density curve.
