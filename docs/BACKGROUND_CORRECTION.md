# Background correction workflow

Open a current recording and select its current channel. In **Inspector → Analysis →
Preprocess**, open the **Preprocess workspace** and choose **Background comparison**.
The workspace has three separate jobs: **Filters & drift**, **Event reference**, and
**Background comparison**. Its target header identifies the recording, channel, units and
sample rate. Main-trace display and automatic-drift controls stay available when changing
comparison sources. Original recordings are retained.

## Automatic comparison and runtime correction

1. In **Background comparison**, choose **Automatic estimator** as the background source.
2. Choose the calculation by its estimator: **Event-reference residual (zero)** uses the
   automatic event-reference estimator and its Rough/Base windows;
   **Drift-stage correction (offset retained)** uses the separately configured automatic
   drift estimator. These are different fits, not one fit with a different vertical offset.
3. Use **Edit event reference…** to adjust the reference windows, or **Edit drift settings…**
   to adjust the Baseline drift method and its method-specific parameters in **Filters & drift**.
   The reference and drift windows remain separate settings even when their values match.
4. Press **Update comparison**. Inspect raw current, fitted baseline and corrected output.
   Read the advisories before deciding whether the settings suit the events of interest.
5. To use automatic drift correction for subsequent classical detection and processed export,
   enable the automatic drift stage explicitly. Run detection deliberately to produce new
   event measurements. Neither calculating a comparison nor exporting one replaces saved rows.

The comparison-output choice changes only comparison/export. The main-trace checkboxes
separately control the reference overlay, reference-subtracted display and processed-chain
overlay. The baseline-only comparison excludes other signal-chain stages; inspect the
**Filters & drift** preview for the full chain. A changed recipe, source, sample rate or
calibration makes an earlier comparison stale. Its persistent **Out of date** indicator
remains until a matching comparison is calculated, even after a successful export.

Custom comparison backgrounds do not replace the automatic runtime drift stage. If drift
is already enabled, it stays enabled and its switch remains usable while inspecting a
custom background. The **Event reference** editor is shared with future event measurements,
including ML event characterization; ML model-input preprocessing remains separate in **ML**.

Short windows can follow and erase long events. A warning identifies measured attenuation
or occupancy risk, but cannot decide whether a sustained level is an event or deliberate
gating. Inspect the original signal, try a longer rough window where appropriate, and compare
event amplitudes. A quiet residual alone is not evidence of a valid correction.

The block-based estimators record their endpoint algorithm in provenance. Existing event
rows made with an older endpoint policy remain saved, but their processed signal cannot be
claimed as an exact reconstruction under the new policy. Review the stale notice and rerun
detection explicitly when updated measurements are needed.

## Editable anchors and discontinuities

1. Choose **Editable time/current anchors**. Values are bound to the current source, units,
   calibration and time base. If those change, review the numbers and press
   **Confirm current source and units** before calculating.
2. Press **Add region**. Enter a unique region name, start and end in milliseconds from the
   recording start. Regions must cover the whole recording without gaps or overlaps.
   Ends are exclusive; the next region begins at the preceding end.
3. Press **Add anchor**. Enter the region name, time in milliseconds and the baseline current
   in the displayed physical unit. Repeat for the remaining anchors.
4. Choose **Linear between anchors**, **PCHIP between anchors**, or
   **Constant (one anchor per region)**. Linear and PCHIP require at least two ordered
   anchors per region. Constant requires exactly one.
5. Choose **Refuse outside anchor support** or deliberately choose
   **Hold nearest endpoint within region**. Held edges are assumptions and are disclosed.
6. For a genuine baseline jump, add adjoining regions and choose
   **Separate regions; preserve boundary jumps**. Each region has its own anchors;
   interpolation never bridges the jump. **One continuous region** accepts exactly one region.
7. Choose the comparison output, then press **Update comparison**.

Example: for a two-second trace with a jump at one second, use regions `A: 0–1000 ms`
and `B: 1000–2000 ms`. With constant interpolation, one anchor in A at 100 pA and one
in B at 80 pA subtract those two levels independently. Anchors are user assertions;
NanoScope does not certify that the chosen levels are open-pore observations.

## Existing open-pore reference

1. Open **Event reference** and set **Known open-pore current** or **Selected open-pore
   intervals**. Supply the known current or selected intervals and their settings.
   This is the shared reference for future event measurements, including ML characterization.
2. Choose **Current open-pore reference** as the background source.
3. Choose an output and press **Update comparison**.

The comparison reuses the existing reference contract, including its interval support and
endpoint assumptions. Selecting this background or calculating it does not edit the shared
reference or enable runtime correction. Editing that reference in **Event reference** does
configure future measurements; saved event rows keep their recorded settings until an
explicit new analysis.

## Independent reference recording

1. Open both the target and independent reference recording. Select the target current channel.
2. Choose **Independent reference recording**, then its current channel in
   **Open reference recording / current channel**.
3. Enter **Reference time shift (ms)**, **Reference scale**, and **Reference current offset**.
   Positive shift moves the reference later. Alignment is relative to each recording's start:
   `background(t) = scale × reference(t − shift) + offset`.
4. Leave **Reference resampling** at **None — equal rates and aligned samples** unless
   interpolation is intended. **Linear resampling (explicit)** permits a different rate or
   fractional-sample shift. It changes noise correlations and is not anti-alias filtering.
5. Choose an output and press **Update comparison**.

Both channels must have matching physical units, declared sample rates and complete reference
coverage of the shifted target. Partial overlap, missing sources, unknown units, and unsupported
alignment are refused. There is no automatic unit conversion, trimming, extrapolation or
alignment optimization. Independent noise is not automatically removed by subtraction.

## Export and save

**Export current recipe…** recalculates from the current controls and writes
every sample, regardless of preview decimation. The CSV contains time, raw current, background
and corrected current; its `.meta.json` sidecar records the recipe, source/calibration,
units and warnings. Custom-source exports additionally record numerical hashes and the
subtraction equation. Both files are replaced together;
a failed write preserves any previous pair. Export completion is separate from preview
freshness: an older plotted comparison remains marked **Out of date** until updated.

For custom sources, zero-centred output is `raw − background`; retained-offset output adds
the median background level. These custom results are inspection/export only and do not
change detector input. Save the project to preserve the editable recipe and reference source
selection, including unfinished text. A future or malformed recipe is preserved without being
partially applied. Reopen missing reference recordings and calculate again before using them.

## Numerical repeatability and noise-aware detection

Automatic baseline windows now resolve sample counts consistently when a rate is only a few
floating-point steps from a sample boundary. The recording's declared rate is retained.
Saved event rows remain available; exact baseline replay checks the algorithm policy and asks
for an explicit new analysis when the saved policy is obsolete. Fixed ML training recipes
retain their historical sample-count arithmetic and identify that policy separately.

Background subtraction cannot remove independent noise, and a quieter residual can still
contain false events. After correction, **Detection → Schmitt → Noise thresholds → Automatic**
checks for strong local changes in noise. Inspect **View → Show detection thresholds** and the
result advisories. **Local** allows an explicit noise window; **Global** preserves one threshold
scale for the full recording. Event-table SNR remains a global measurement-noise ratio.

Automatic masked-median baselines now interpolate across masked events using robust nearby
unmasked flank medians, with support limited at resolved baseline jumps. This prevents a
single noisy endpoint from biasing the full event span. Unsupported edges retain endpoint
holding, and insufficient baseline support remains a limitation. The change is versioned;
fixed ML training recipes keep their historical interpolation arithmetic.

For automatic event masks, local noise is checked against both the original current and
its rough-baseline residual. A local threshold rises only where both support the increase;
it remains at least as large as the global residual-noise threshold. This prevents rough
baseline movement through dense events, or smooth physical drift, from being mistaken for
extra random noise. Flank anchors use bounded nearby support to keep large recordings responsive.

Passing baseline health checks does not independently establish the physical open-pore level.
When two levels are plausible, the health panel explicitly marks the reference as unverified.
Use a known open-pore current or a verified unclogged interval to resolve that ambiguity.

Baseline health also checks long events separately from its median intrusion statistic.
A moderate-SNR event can lose most of its depth even while the median across stronger events
looks acceptable. The long-state check reports that loss explicitly; when the headline median
cannot be measured, it remains unavailable rather than being reported as zero. An unsuitable
manually selected window can therefore warn even when other windows pass.
