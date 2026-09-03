# Sweep-7 fix batch — verification register (2026-09-01/02)

Wave 1 (6 agents) fixed the 24 register findings + 8 skeptic-found items; wave 2 (2 agents) wired the seams and fixed the subpopulation findings. Then 11 skeptics re-drove everything through the real widgets, 20 fixers repaired what they found, and 50 fresh skeptics re-verified each repair.

**Group verdicts on the original findings: 69 FIXED, 19 PARTIAL, 8 not re-driven by that skeptic (covered by another).**
**Skeptics raised 53 new defects: fixers closed 50, declined/deferred 3. Re-verification: 45 confirmed, 5 partial.**

## Original findings, by group

### A1 — Plots window (S7-FIX1, S7-SK1, S7-FIX2, S7-SK2, S7-FIX3, S7-JRN2 Plots half)
- **FIXED** — S7-FIX1
- **FIXED** — S7-SK1
- **FIXED** — S7-FIX2
- **PARTIAL** — S7-SK2
    - repro_S7-SK2_two_curves_one_name.py: tree exit 0 (rows ['-700 mV','Apo'], canvas ['-700 mV (n = 982)','Apo (n = 236)'], SVG has one 'Apo'), snapshot exit 7. Two rows typed to one name are told apart in the SERIES entries ('Apo — -700 mV' / 'Apo — -600 mV'), Details carries the LEGEND note, SVG carries two names. BUT the resolver was applied at five call sites, not at the invariant: the scatter FIT legend entry (plots_panel.py:11464), the in-figure provenance caption (:7819) and export_data's fallback series names (:12064) still read _legend_label, so the same saved SVG reads 'Apo fit: y = -3.6
- **FIXED** — S7-FIX3
- **PARTIAL** — S7-JRN2
    - S7-JRN2_rejected_fit_on_figure.py (cwd scratch/JRN): tree exit 0 — at all four settings of show_gof x fit_values_on_plot the title reads 'τ = 20.41 ms (n = 275) — KS REJECTED (1%)' and the legend 'fit: τ = 20.41 ms…KS REJECTED (1%)'; snapshot exit 7. The rewritten test drives a real window (title/legend/values box/SVG). Numbers: export_data CSV numeric columns and histogram arrays byte-identical snapshot vs tree on 5 (recording, column) cases; only the fit rows' `series` label of a REJECTED fit now carries the tag (the CSV is 8.4 kB larger; a passing fit's CSV is identical). BUT the tag lives

### A2
- **FIXED** — S7-JRN3 (dwell fit: tau printed with a 95% CI for a KS-rejected model, no verdict on the exported PDF/PNG)
- **FIXED** — S7-JRN3 baseline-advisory half (the 'open pore NOT CERTIFIED' advisory in provenance['baseline_warnings'] reaches none of the four figures)
- **FIXED** — MISSED-1 (exported scatter y axis says 'blockade depth ΔI/I₀' and 'median ΔI/I₀ = 2.524' bare over an enhancement population)
- **FIXED** — MISSED-2 (exported depth figure's Gaussian fit also KS-rejected, printed 'μ = 2.363 [2.265, 2.455]' with no verdict)

### B — Event gallery + Event inspector (S7-POP1, S7-POP2, S7-POP5, S7-POP6, S7-POP7, skeptic MAJOR "gallery multi-tile selection wiped by any curation on another surface")
- **FIXED** — S7-POP2 — selecting an event the gallery's filter/sample excludes leaves a different tile current; the next R rejects it
- **PARTIAL** — Skeptic MAJOR — the gallery's multi-tile selection is wiped by any curation on another surface (reflect_acceptance's bare refilter)
    - MISS-1c on the tree: sweep [3..8] survives a table checkbox tick ('gallery now holds: [3,4,5,6,7,8] current 8'), 'Reject selected' rejects exactly [3..8]; the inspector's R path is covered by the fixer's test (4/4 pass, real mouse sweep + model.setData). The one-reset invariant is real. BUT the new reset calls _focus_event(current) BEFORE _restore_selection (event_gallery.py:1746-1747), and _focus_event turns pages: under 'Accepted only' with the cursor on the first tile of page 2 and a five-tile sweep, a table rejection of an EARLIER event turns the gallery to page 1 under the pointer and col
- **FIXED** — S7-POP1 — the stale-depth badge stops at the events table; gallery and inspector repeat the numbers unmarked (B's seams + C1/C2 wiring)
- **FIXED** — S7-POP5 — the depth caption is the only sort caption without its unit
- **FIXED** — S7-POP6 — the status line advertises '1–9 to label' on an empty palette; pressing 1 says nothing
- **FIXED** — S7-POP7 — the range readout prints '0–-1 of 0' on an empty subset

### C-subpops
- **FIXED** — S7-POP3 (report buttons live after a merge; 'fitted no mixture' false reason)
- **FIXED** — S7-POP4 (report not marked when the curation under it changes)
- **FIXED** — Skeptic MAJOR: merely LOOKING at another open recording permanently retires the assignment

### C-wiring
- **FIXED** — S7-POP1 end-to-end wiring (Preprocess knob / project open -> gallery captions + inspector readout badged)
- **PARTIAL** — S7-NOTCH1 replay guard in analysis_panel._run_detection (+ status-bar forward of the box's own refusal)
    - Guard holds (verify/C-wiring/probe_notch_replay.py, real MainWindow, owner's 0004 ABF, real detection with an explicit 2-line list, stored dict flooded to 33 lines, gain edit -> banner 'Re-run'): Re-run refused with 'Detection refused — the notch list this run would replay was not accepted: the list held 33 lines and the box accepts at most 32 ... Fix the list on the Preprocess tab (“use an explicit line list”) and run again.', same sentence on the status bar, banner stays, _pending None, stored table untouched, nothing started. Real key input: a 33-line paste into the Schmitt box + Tab -> not
- **FIXED** — ML-detector baseline source stamp in ml_modules (MLDetect.run through modules._resolve_run_baseline)
- **FIXED** — S7-CAL1 Detect gate through the session (analysis_panel resolver/_gate_for/_regate + MainWindow action sync)

### D
- **FIXED** — S7-CAL1 — 'Treat this channel as: current' honoured by Batch detect and refused by the Detect button
- **FIXED** — Skeptic MAJOR — toolbar chip and workflow rail deny the corrected channel
- **PARTIAL** — Skeptic minor — Calibration panel contradicts itself (title / hidden conductance boxes)
    - Title is fixed: 'Untitled · current (file says unknown)' immediately after the real Apply. The partner boxes are NOT: after Apply 'treat as current' on the selected ch0 the Open-pore-conductance box stays hidden (_cur_ch None) until the user clicks another row and back; after Apply 'treat as voltage' on ch1 the Command-voltage box stays hidden until reselect; and in the reverse direction (ch0 re-declared voltage, or IN15 on the ABF) the conductance box keeps printing I0 = -122.8 mV / G — from the channel just declared a voltage (ABF: I0 -144.2 pA, G 0.1442 nS) until reselect. Mechanism: _on_ca
- **FIXED** — S7-CAL3 — min/max/span from a strided subsample printed as plain numbers
- **FIXED** — S7-CAL4 — float64 preview vs float32 apply; chip certifies the ruined channel
- **PARTIAL** — S7-CAL5 — Apply on a moved recording reports 'saved to', writes into the old folder, value gone on reopen
    - repro_S7-CAL5.py (copies under docs/sweep7/repros/cal5_*): after the move, Apply 0.9 -> 'Applied · not saved beside the recording: pore.abf is no longer at …/cal5_old; the calibration lives only in this session (and in a saved project). Move the recording back, or save a project, to keep it.'; the orphan in cal5_old is NOT rewritten (still 0.75), nothing invented in cal5_new; recording_moved() answers False for segments and never-on-disk docs and reverts when the file is moved back. UNADDRESSED, by the fixer's own report: step 4 of the same repro — reopening the recording where it now lives gi
- **FIXED** — S7-CAL2 section B — the reverse override must disable Run detection

### E — detection provenance (S7-DET1, S7-DET2, S7-DET3, S7-DET4, DET MISSED-1, DET MISSED-2, S7-JRN1 tooltip half)
- **PARTIAL** — S7-DET1 — Batch detect stamps 'proposed' on never-measured recordings and silences the never-measured advisory
    - The filed path is fixed and holds on the real widgets: S7-DET1 repro (cwd scratch/DET) on the tree gives -1000 mV baseline_param_source='batch', copied-windows advisory present on the result and in Stats > About these numbers, panel 'These values: COPIED from the batch form -- …'; the snapshot (PYTHONPATH pre-sweep7) gives 'proposed', advisory absent, panel 'measured on this recording, and confirmed'. v_det1_export: exported .meta.json 'batch' (snapshot 'proposed'). My p1: the stale banner's Re-run on the 'batch' result reproduces 'batch'; a project save+reopen in a fresh MainWindow keeps 'pro
- **FIXED** — S7-DET2 — Detector score publishes precision/recall/F1 on an uncurated table because auto-rejections count as user FPs
- **FIXED** — S7-DET3 — a clamped short segment loses the never-measured advisory ('default' overwritten to 'fallback')
- **FIXED** — S7-DET4 — Merge all leaves provenance['n_events'] at the pre-merge count; subpop report counts merges as curation
- **FIXED** — DET MISSED-1 — 'Re-measure depths' deletes every detector-authored advisory
- **FIXED** — DET MISSED-2 — Re-measure re-derives n_clipped_boundary but not n_filter_limited; sidecar disagrees with the table
- **FIXED** — S7-JRN1 (tooltip half) — 'Measure this recording' tooltip promises about a fifth of a second

### F
- **FIXED** — S7-NOTCH1 (Schmitt notch-line box: no cap, silent truncation at 32,767 chars, manufactured 54.0 Hz line)
- **FIXED** — Skeptic 'notch typo policy' item (parse_notch_lines dropped '-120' / 'abc' silently)
- **FIXED** — S7-IV1 (I-V Details header 'current in ' with the unit word missing)
- **PARTIAL** — S7-PSD1 (RMS-in-band label rounded to 4 significant digits; zero-width band beside a non-zero RMS)
    - The reported symptoms are gone: repro on the tree prints (24999-25000 Hz) beside 0.007632 pA and (12345-12355 Hz) for 12345-12355 (snapshot: 25000-25000 / 12340-12360); my psd_probe.py on the owner's real -1000 mV excerpt, a 20k-sample synthetic (24 Hz bins) and a two-curve 50k/20k overlay found no exponent notation, no float noise, no zero-width band beside a non-zero RMS, and the empty band keeps its words. BUT the new _fmt_band_edges tolerance (half a PSD bin) still restates a typed 0.1-Hz edge at less precision than the band the integral used: on the owner's file (0.76 Hz bins) typed 12345
- **PARTIAL** — S7-JRN1 ('about a fifth of a second' in four places)
    - All four stale claims are gone and consistent: tooltip (baseline_panel.py:300-305), USER_GUIDE.md:16 and :312, analysis_panel.py:665 wiring comment and _on_probe_requested docstring (1.97 s / 6.38 s), manual.md:137/2464, inventory.json:4567 all say 'usually a few seconds - measured 2 s on a 3 M-sample recording and 6 s on 9 M samples'; tests/test_docs_merge_and_format_agree_with_app.py 10 passed; only the known-stale built HTML still says 'fifth of a second'. The button figure holds within load: JRN1 repro (exit 7 by design - it still tests the old 0.2 s claim) measured 2.49 s / 7.6 s medians
- **PARTIAL** — MISSED-3 (ProbeResult.elapsed_s docstring's measured table 233 ms / ~70 ms per M)
    - baseline_probe.py:445-457 and probe_and_propose (2028-2036) rewritten: stage-only, dated, cold caveat, and an explicit hedge ('treat these as this bench on this date'); baseline_panel.py:24-27 module docstring agrees (343 ms / 1,073 ms). But the replacement table is again a 'measured' claim I could not reproduce on the same file: three warm presses gave 634 ms median (probe_elapsed.py; bare probe_timescales and the button's proposal.probe.elapsed_s agree with each other in probe_elapsed2.py), i.e. ~210 ms per M, not 115-190; ratio to the whole button 29% vs the docstring's 17%, matching the sw
- **FIXED** — S7-JRN7 (doc column counts 20/21 vs measured 23)

### X-combos (cross-feature skeptic, sweep-7 fix batch)
- **FIXED** — Chain 1 — detect → Merge all → Re-measure → Undo/Redo (Edit menu + Ctrl+Z through core/history.EventHistory) → Export events + subpop report (S7-DET4, MISSED-1, MISSED-2 across every step)
- **FIXED** — Chain 2 — two recordings → probe+confirm on −600 mV → Batch detect → Voltage dependence caveats → project save → reopen in a fresh window (S7-DET1 end to end)
- **PARTIAL** — Chain 3 — calibration override on the undeclared CSV (copy; real Apply) → Detect → PSD, I-V, Event plots, Plots, chip, rail, File info must agree; override back to voltage → every gate refuses naming the override (S7-CAL1 + MISSED rail/chip + MISSED panel + C2 wiring)
    - chain3_override_every_surface.py. FIXED half: after 'Treat as current' + pA + 2.5 the chip reads '✓ Gain set: ×2.5 → pA', rail CALIBRATE done / FILTER+DETECT available, Calibration-panel title 'Untitled · current (file says unknown)', Analysis-panel title the same shape, Detect button enabled with an empty gate, Analysis▸Run detection / Batch / PSD / I-V actions enabled (menu resynced), Detect stored 15 events on ch0 in pA, rail DETECT 'Events detected.' / CURATE '15 accepted event(s)', Plots window lists the recording. Reverse (back to voltage): Detect gate 'Detection runs on CURRENT channels
- **FIXED** — Chain 4 — Plots window in the real MainWindow: import colleague JSON (refused normalization + untakeable bins + untakeable fit.kind) → undo → redo → bins → time unit → template → undo back → export JSON; rename source A to B's label → SVG (S7-FIX1, S7-SK1, S7-FIX2, S7-SK2, A1's origin rule)
- **FIXED** — Chain 5 — gallery multi-select → curate from the table → curate from the inspector → Accepted only → show_event on a rejected one → R → stale badge after a Preprocess knob → Re-measure clears gallery AND inspector (S7-POP2, MISSED selection wipe, S7-POP1 wiring, POP5/POP6/POP7 lines)
- **FIXED** — Chain 6 — subpop run → click another recording → back → Colour the trace → merge → buttons retired with the true reason → curation → report marked (S7-POP3, S7-POP4, skeptic MAJOR 'looking retires the map')
- **FIXED** — Chain 7 — notch: paste 40 lines → refused; type 60,-120 → refused; stale banner's Re-run with a stored 40-line list → refused by name, not run; stored list replayed into the form on re-selection (S7-NOTCH1, typo policy, C2's replay guard)
- **PARTIAL** — Chain 8 — 0.08 s segment → Detect at defaults → advisory present + CLAMPED line; Batch detect with the segment open, then after a probe+confirm on the parent (S7-DET3 + S7-DET1 on a segment)
    - chain8_segment_clamp_batch.py on a session.create_segment(100000..104000) of −600 mV. Detect at defaults: source stays 'default', warnings carry 'never measured', the 4000-sample CLAMPED sentence, the 29.9 % masked sentence and NOT CERTIFIED; Stats shows the advisory and CLAMPED; the form line reads 'The detection on screen actually ran at rough 80 ms / base 16 ms, not the 0.5 s / 0.1 s above -- CLAMPED…'. Batch detect with the segment open (dialog '1 of 2 recording(s) already have…' answered Re-run all): segment 'default' + advisory + CLAMPED line, parent 'default' with no CLAMPED line. After

### X-numbers (bit-identity skeptic, sweep-7 fix batch)
- **FIXED** — NUMBERS INVARIANT — the 20 frozen EventSet columns at every stage (Detect at defaults, Merge all 10 ms, Measure → Use these values → Re-measure depths, Batch detect) on the real −600 mV and −1000 mV ABFs, through the real widgets
- **FIXED** — S7-DET4 / MISSED-1 / MISSED-2 — merge restamps n_events; re-measure keeps the detector's warnings and re-derives row counts; nothing numeric moved
- **FIXED** — S7-DET1 — Batch detect stamps 'batch' on the recording the windows were not measured on, and nothing else differs
- **FIXED** — S7-DET2 — scoring rows (screen-only) are the only thing that moved in the Detector score box
- **FIXED** — S7-JRN3 + MISSED-1/2 (figures) — Export figure… sidecars and every number on the axes unchanged; only the CI is withdrawn for rejected fits and the verdict/advisory lines added
- **FIXED** — S7-JRN2 — Plots window on the shipped dwell template: exported data + JSON
- **FIXED** — S7-PSD1 / S7-IV1 — PSD window (2 s segment) and I-V window over two recordings
- **FIXED** — S7-CAL3 / S7-CAL4 — Calibration panel readouts (no Apply pressed, no sidecar)
- **FIXED** — S7-DET3 — short segment: 'fallback' → 'default' with the advisory, clamp arithmetic unchanged
- **FIXED** — Ferritin/apo corpus oracles and bit-identity tests

### X-surfaces (every sentence, badge, tooltip, status line and label the sweep-7 fix batch added or changed)
- **FIXED** — S7-CAL1 — 'Treat this channel as: current' honoured by Batch detect and refused by Detect
- **PARTIAL** — S7-DET1 — Batch detect stamps 'proposed' on unmeasured recordings and silences the advisory
    - App surface: BaselinePanel._SOURCE_LABELS now renders 'These values: COPIED from the batch form -- proposed or confirmed for a DIFFERENT recording; this recording was not measured' (P5 §1) and the 'fallback' label became a legacy wording; batch stamping itself was not re-driven here (E's repro/tests). Docs were NOT updated: docs/inventory.json:4561 (rendered into manual.md:2463) still enumerates the 'These values' readout as ending in 'fell back -- the estimator could not run on this trace' and never mentions the batch label — E's own report listed 'Manual prose for the new batch source label'
- **FIXED** — S7-DET2 — Detector score publishes precision/recall/F1 on an uncurated table
- **NOT_APPLICABLE** — S7-DET4 — Merge all leaves provenance['n_events'] at the pre-merge count
    - Not re-driven by this audit (no on-screen string changed except the report's CURATION line, which E's repro covers).
- **NOT_APPLICABLE** — S7-FIX1 / S7-SK1 — Undo / Time-unit erase the import disclosures
    - Not re-driven here; the sentences themselves are unchanged (A1 repros FIX1/SK1 report 7->0).
- **FIXED** — S7-FIX2 / S7-SK2 — retyping the original name strands the override; two curves under one legend name
- **FIXED** — S7-JRN2 — rejected fit reported as tau with no verdict; tooltip promised otherwise
- **PARTIAL** — S7-JRN3 — exported PDF/PNG carries a 95% CI for a rejected model and no verdict/advisory
    - P4 (Analysis > Export figure… on the -600 mV recording after probe/confirm/detect): dwell box 'τ = 20411.6 µs (rejected fit) / n = 275 / t_min = 20.0 µs / KS D = 0.397 > 0.098 (1%): fit REJECTED / — parameters are not measurements / open-pore baseline not certified / — see the events CSV sidecar'; depth 'μ = 2.363 (rejected fit)' + verdict + advisory; scatter median line + '(ENHANCEMENT: > 1, not a blockade)' + advisory; trace: advisory above the axes; no interval anywhere; a synthetic passing fit prints 'KS D = 0.029 ≤ 0.081 (1%): consistent with these data' with its interval. BUT the seven-l
- **FIXED** — S7-NOTCH1 — Schmitt notch-line box silently truncates a long paste
- **FIXED** — S7-POP2 — selecting an event the gallery's subset excludes leaves another tile current
- **NOT_APPLICABLE** — S7-POP3 — report buttons stay live after a merge; 'fitted no mixture' false reason
    - Verified by C-subpops; not re-driven here. One new sentence it introduced names a 'Files panel' that does not exist — the dock is 'File Manager' (X-SURF-12).
- **FIXED** — S7-CAL3 — calibration preview prints strided min/max/span as plain numbers
- **FIXED** — S7-CAL4 — overflowing gain previews finite numbers and the chip certifies the ruined channel
- **FIXED** — S7-CAL5 — Apply after the file moved reports 'saved to …'
- **FIXED** — S7-DET3 — a clamped short segment loses the never-measured advisory
- **PARTIAL** — S7-FIX3 — density note calls a picoamp density 'dimensionless'
    - X6/P3: the dimensionless clause is now kept only for frac_depth/snr (declared '-'), depth_pA reads 'a density in 1/pA'. The replacement clause is false on the very table the finding was about: amplitude/level/baseline/n_samples on the manual-measurement table now read '— a density per unit of amplitude; this table does not declare what that unit is.' while that table's own `units` column reads 'pA' and MEASUREMENT_SCHEMA declares the column (X-SURF-7). Details-only, not exported.
- **FIXED** — S7-IV1 — I-V Details header 'current in ' with no unit
- **PARTIAL** — S7-JRN1 — 'about a fifth of a second' in four places
    - Button tooltip now reads 'Look at this recording and propose a rough and base window from its own timescales, then run the health checks on what it proposes. Usually a few seconds -- measured 2 s on a 3 M-sample recording and 6 s on 9 M samples (the button says Measuring... meanwhile; it has not missed the click). Nothing is applied until you press Use these values.' (P1); USER_GUIDE lines 16/310 rewritten. But docs/inventory.json:4567 (rendered as manual.md:2464) quotes a DIFFERENT tooltip — 'Look at this recording (usually a few seconds — measured 2 s …) and propose…' — F's report asked to r
- **FIXED** — S7-POP1 — stale-depth badge stops at the events table
- **NOT_APPLICABLE** — S7-POP4 — report not marked when the curation under it changes
    - Verified by C-subpops (C-SUB-1 filed there); not re-driven here.
- **PARTIAL** — S7-POP5 — gallery depth caption has no unit
    - Caption now '91 · 139pA' (P1, P5). But the unit is resolved once at feed time (event_gallery.py:1602 self._units = self._resolve_units()) and never again: after an in-session gain edit to nA (write_sidecar=False) the gallery still reads '91 · 139pA ⚠' while the inspector reads 'depth: 138.85 nA ⚠' and the table header 'depth_pA ⚠' for the same event 91, and a curation gesture does not refresh it (X7 repro exit 7; X-SURF-9).
- **FIXED** — S7-POP6 — '1–9 to label' advertised with an empty palette
- **FIXED** — S7-POP7 — '0–-1 of 0' on an empty subset
- **FIXED** — S7-PSD1 — RMS band label rounded to 4 significant digits
- **FIXED** — MISSED (CAL, major) — chip and workflow rail deny the corrected channel
- **PARTIAL** — MISSED (CAL, minor) — Calibration panel title 'unknown' and hidden conductance boxes on a corrected recording
    - Title now 'Untitled · current (file says unknown)' immediately after Apply (P2 §E). The Open-pore-conductance box stays hidden after the override and appears only when the channel is re-selected (P2 §E cond_box visible=False right after Apply; P5 §5 / X8: hidden after set_calibration, visible with I0 -122.8 pA after set_selection away and back; same for the Command-voltage box after ch1 -> voltage) — X-SURF-10.
- **NOT_APPLICABLE** — MISSED (POP, major) — gallery multi-selection wiped by curation on another surface
    - Not re-driven here (B's tests).
- **NOT_APPLICABLE** — MISSED (POP, major) — looking at another recording retires the assignment
    - Verified by C-subpops; not re-driven here.
- **PARTIAL** — MISSED (JRN-1) — exported scatter labels its y axis 'blockade depth' over an enhancement cloud
    - P4: scatter y label 'ΔI/I₀ (ENHANCEMENT: median > 1, not a blockade)', median line 'median ΔI/I₀ = 2.524 (ENHANCEMENT: > 1, not a blockade)', same helper as the depth histogram; a synthetic median-0.4 table keeps 'blockade depth ΔI/I₀'. But that box now extends past the left edge of the scatter axes (x0 = -0.07 of the axes width) and across the data column — X-SURF-1.
- **FIXED** — MISSED (JRN-2) — exported depth fit also rejected
- **NOT_APPLICABLE** — MISSED (JRN-3) — probe elapsed_s docstring table
    - Docstring, not a user surface; F rewrote it with dated numbers.
- **FIXED** — MISSED (SPEC) — notch box drops unparseable tokens silently
- **NOT_APPLICABLE** — MISSED (DET-1 / DET-2) — Re-measure deletes detector advisories / n_filter_limited stale
    - Not re-driven here (E's repros).
- **FIXED** — Data contamination (sidecars beside the owner's recordings)

## New defects the skeptics found in the fixes, and what happened to them

| id | sev | file | title | fixer | re-verify |
|---|---|---|---|---|---|
| A1-N1 | major | `nanoscope/gui/plots_panel.py` | A typed title + Show legend off ships the KS-rejected curve with no verdict on the saved image | FIXED | FIXED |
| B1 | major | `nanoscope/gui/event_gallery.py` | The stale sentence pushed onto the gallery status line re-fits the page under the user and re-cuts the represe | FIXED | FIXED |
| B4 | major | `nanoscope/gui/event_gallery.py` | An inbound curation now turns the gallery's page under the user and collapses the swept block to one tile, cal | FIXED | FIXED |
| C1 | major | `nanoscope/gui/main_window.py` | Another detector's run on the same recording parks the map in 'away' with a recovery instruction that cannot w | FIXED | FIXED |
| CW-2 | major | `nanoscope/gui/event_inspector.py` | The S7-POP1 badge push moves the Event Inspector under the user: the plot shrinks 71–84 px and the readout box | FIXED | FIXED |
| E1 | major | `nanoscope/gui/baseline_panel.py` | A second Batch detect from the never-measured recording re-stamps the measured recording 'batch' and says it w | NOT_FIXED | — |
| F-SK-1 | major | `nanoscope/gui/panels.py` | Notch-line refusal is erased by the run it accompanies; on the Ctrl+D path nothing survives | FIXED | PARTIAL |
| X-SURF-1 | major | `nanoscope/plot/figures.py` | Exported dwell/depth/scatter figures draw the new verdict + baseline + ENHANCEMENT lines over the data, and th | FIXED | FIXED |
| X1 | major | `nanoscope/gui/psd_panel.py` | PSD window denies the channel the app just calibrated and detected on: Analysis ▸ Power spectrum is enabled on | FIXED | FIXED |
| X2 | major | `nanoscope/analysis/iv.py` | I-V window denies the corrected channels: analysis/iv.py resolves the current channel and the voltage channel  | FIXED | FIXED |
| X4 | major | `nanoscope/gui/analysis_panel.py` | A clamped short segment raises a permanent, false 'The open-pore estimator changed since this detection ran' b | FIXED | FIXED |
| A1-N2 | minor | `nanoscope/gui/plots_panel.py` | Two sources under one name still ship in the scatter fit entries and the in-figure provenance caption | FIXED | FIXED |
| A1-N3 | minor | `nanoscope/gui/plots_panel.py` | The new density note says the manual-measurement table 'does not declare' a unit its units column declares as  | FIXED | FIXED |
| A1-N4 | minor | `nanoscope/gui/plots_panel.py` | Two colleague imports, then Undo onto the first file's own state: both of its disclosures are gone and the re- | FIXED | FIXED |
| A2-NEW-1 | minor | `nanoscope/plot/figures.py` | The KS verdict and baseline caveat the fix added are written across the histogram peak on both exported histog | FIXED | FIXED |
| B2 | minor | `nanoscope/gui/event_gallery.py` | Under the representative sample the gallery now says a tile was 'still curated' when nothing curated it | FIXED | FIXED |
| B3 | minor | `nanoscope/gui/event_inspector.py` | The stale sentence inserted into the inspector readout shrinks the waveform plot by 84 px on every staleness f | FIXED | FIXED |
| B5 | minor | `nanoscope/gui/event_gallery.py` | An edit on another surface re-selects the cursor tile the user had just deselected; 'Reject selected' then rej | FIXED | FIXED |
| C2 | minor | `nanoscope/gui/main_window.py` | The S7-POP4 standing line is derived from the accepted COUNT, so it clears while the accepted SET is not the r | FIXED | PARTIAL |
| C3 | minor | `nanoscope/gui/subpop_window.py` | The report window's standing line is a fixed two-line label, and the AWAY reason plus the S7-POP4 sentence nee | FIXED | FIXED |
| C4 | minor | `nanoscope/gui/main_window.py` | After the away-and-back round trip the map returns but the gallery's chosen detrended trace (and the represent | FIXED | FIXED |
| CW-1 | minor | `nanoscope/gui/analysis_panel.py` | The notch replay refusal prescribes a fix (the Preprocess box) that the door it refuses (the banner's Re-run)  | FIXED | FIXED |
| CW-3 | minor | `nanoscope/gui/event_gallery.py` | The S7-POP1 badge push appends the whole ~720-char banner sentence to the gallery's word-wrapped footer, shrin | FIXED | FIXED |
| CW-4 | minor | `docs/USER_GUIDE.md` | 'Treat this channel as' is still documented (tooltip, USER_GUIDE, manual/inventory) as deciding only the pane  | FIXED | PARTIAL |
| D1 | minor | `nanoscope/gui/calibration_panel.py` | Calibration panel partner boxes are re-derived only on reselect, not on the Apply that changed the kind | FIXED | FIXED |
| D2 | minor | `nanoscope/gui/analysis_panel.py` | Stale banner's Re-run stays live after the reverse override and clicking it does nothing, silently | FIXED | FIXED |
| D3 | minor | `nanoscope/gui/workflow_status.py` | After the reverse override the calibration chip's tooltip denies the header without naming the override, while | NOT_FIXED | — |
| D4 | minor | `nanoscope/gui/calibration_panel.py` | 'Treat this channel as' tooltip, code comment and USER_GUIDE still say the override only picks a pane | NOT_FIXED | — |
| E2 | minor | `nanoscope/gui/baseline_panel.py` | The 'actually ran at … not the X s / Y s above' line goes on quoting form values the form no longer shows | FIXED | FIXED |
| E3 | minor | `nanoscope/analysis/subpop_report.py` | The subpopulation report's 'curated out (not below)' line calls the merged table 'detected region(s)' | FIXED | FIXED |
| E4 | minor | `nanoscope/analysis/baseline.py` | Hand-set ('user') windows reach the other recordings in a batch described as 'proposed or confirmed for a DIFF | FIXED | FIXED |
| E5 | minor | `docs/USER_GUIDE.md` | USER_GUIDE.md still lists the pre-sweep-7 Open-pore source labels: 'fell back — the estimator could not run on | FIXED | FIXED |
| E6 | minor | `nanoscope/gui/baseline_panel.py` | A 'batch'-seeded form carries the copied windows onto the next recording selected, and the interactive Detect  | FIXED | FIXED |
| F-SK-2 | minor | `nanoscope/gui/panels.py` | Refusal note grows and moves the control below it under the pointer; the click is lost | FIXED | FIXED |
| F-SK-3 | minor | `nanoscope/gui/panels.py` | Spreadsheet column/row paste (newline, tab, space separated) refused with the wrong diagnosis | FIXED | FIXED |
| F-SK-4 | minor | `nanoscope/gui/psd_panel.py` | PSD band label still restates a typed 0.1-Hz edge at less precision than the integrated band (up to half a bin | FIXED | FIXED |
| X-SURF-10 | minor | `nanoscope/gui/calibration_panel.py` | After 'Treat this channel as: current' the Calibration panel's title says current but its Open-pore-conductanc | FIXED | FIXED |
| X-SURF-11 | minor | `docs/USER_GUIDE.md` | USER_GUIDE.md still says a short segment's baseline source reads 'fell back — the estimator could not run on t | FIXED | FIXED |
| X-SURF-12 | minor | `nanoscope/gui/main_window.py` | The report window's new 'away' sentence tells the user to select the recording 'in the Files panel' — the dock | FIXED | FIXED |
| X-SURF-2 | minor | `nanoscope/plot/figures.py` | Every exported figure says '— see the events CSV sidecar' but the figure export writes no such file, and the f | FIXED | FIXED |
| X-SURF-3 | minor | `docs/inventory.json` | docs/inventory.json (and the manual Control reference generated from it) quotes five surfaces sweep 7 rewrote: | FIXED | PARTIAL |
| X-SURF-4 | minor | `nanoscope/gui/baseline_panel.py` | The new 'The detection on screen actually ran at … not the 0.5 s / 0.1 s above' line keeps naming the old form | FIXED | FIXED |
| X-SURF-5 | minor | `nanoscope/gui/calibration_panel.py` | The float32-overflow refusal is written into the un-wrapped formula line and pushes the Calibration panel's mi | FIXED | FIXED |
| X-SURF-6 | minor | `nanoscope/gui/calibration_panel.py` | On a channel that carries an infinity the calibration box prints '—' for min/max/span while the new coverage l | FIXED | FIXED |
| X-SURF-7 | minor | `nanoscope/gui/plots_panel.py` | The rewritten density note says 'this table does not declare what that unit is' on the manual-measurement tabl | FIXED | FIXED |
| X-SURF-8 | minor | `docs/manual-markdown/manual.md` | manual.md prose quotes the old gallery hint and a tooltip string that is not the button's | FIXED | FIXED |
| X-SURF-9 | minor | `nanoscope/gui/event_gallery.py` | After a calibration edit the gallery's new depth caption keeps the unit it was fed with while the inspector pr | FIXED | FIXED |
| X3 | minor | `nanoscope/gui/file_info_panel.py` | File info's Channels table prints the header kind ('unknown') after the override, while the Calibration panel  | FIXED | FIXED |
| X5 | minor | `nanoscope/gui/main_window.py` | The subpopulation report's Export… writes the run's header and fractions with nothing saying the accepted set  | FIXED | FIXED |
| X6 | minor | `nanoscope/gui/main_window.py` | Edit ▸ Undo of a plain curation retires the subpopulation assignment with 'retired when the detection changed' | FIXED | FIXED |
| X7 | minor | `nanoscope/analysis/subpop_report.py` | After a merge the exported subpopulation report calls the merged table 'detected region(s)' two lines above th | FIXED | FIXED |
| X8 | minor | `nanoscope/gui/workflow_status.py` | On an undeclared recording set back to 'Treat as voltage', the workflow rail and the chip say 'No current chan | FIXED | PARTIAL |
| X9 | minor | `nanoscope/gui/file_dock.py` | Tab out of the Preprocess ▸ notch 'lines' box walks into the Files dock's per-row offset spin boxes and silent | FIXED | FIXED |

## Not fully closed — carry-over

### C2 [minor] — fixer FIXED, re-verify PARTIAL
The S7-POP4 standing line is derived from the accepted COUNT, so it clears while the accepted SET is not the run's

REPORTED CASE: FIXED, and at the invariant. Original repro /private/tmp/claude-501/-Users-abrahamcampos-Desktop/6f43f43e-d8d3-4c35-89b0-a20f09cbc171/scratchpad/verify/C-subpops/repro_C2_curation_note_count_not_set.py now exits 0 on ABF 0004: after accept-A-back + reject-B the line stands as "accepted now: 109 of 110; this run's fractions describe the 109 accepted at run time - re-run to update" with accepted-set == run's-set False. Code read: main_window.py:4986-4990 snapshots frozenset(event_id of accepted rows) into _subpop_accepted_ids, cleared at main_window.py:5118 in _drop_subpop_assignment, and _subpop_curation_note (main_window.py:5316-5350) compares sets, falling back to the count only when the frame has no event_id column or run_ids is None. One resolver, one writer - the fixer's structural claim holds.

REAL WIDGETS (my probe /private/tmp/.../reverify/C2/probe_R2_real_doors_swap.py, exit 0): events-table checkbox rejects A before the run (n_fired 109); line silent straight after the run; Event Inspector's real Accept button puts A back -> line stands (110 of 110); gallery R key rejects B -> count is 109 again, set differs (in-run-rejected-now [7], accepted-now-not-in-run

### CW-4 [minor] — fixer FIXED, re-verify PARTIAL
'Treat this channel as' is still documented (tooltip, USER_GUIDE, manual/inventory) as deciding only the pane and overlay group, now that it gates Detect, Run detection, Batch, the rail, the chip and the probe

The two edits the register assigned are real and hold up; the third surface the defect title names (manual/inventory) is untouched, and the register's reason for excluding it is false.

WHAT IS FIXED (driven through real widgets, offscreen, tree at /Users/abrahamcampos/Desktop/nanoscope)

1) Tooltip, /Users/abrahamcampos/Desktop/nanoscope/nanoscope/gui/calibration_panel.py:176-183 (was :151). Read off the live CalibrationPanel._treat_as: "What this channel IS, to the whole app: which pane it draws in and how it is grouped with the others, AND which channel the analysis acts on — Run detection, Batch detect, Power spectrum and I-V, the calibration chip and the workflow strip all follow it. 'current' makes an undeclared channel analysable; 'voltage' on a channel the file calls current switches detection off for it. ..." Snapshot (cd / && PYTHONPATH=…/pre-sweep7-fixes-2026-09-01) still has the layout-only string.

2) Guide, /Users/abrahamcampos/Desktop/nanoscope/docs/USER_GUIDE.md:204-213. diff vs snapshot USER_GUIDE.md:202-203 shows the CW-4 edit is confined to that bullet (the other hunks are E5/X-SURF-11/probe-timing). Snapshot: "it decides which pane the channel gets and which ove

### D3 [minor] — fixer NOT_FIXED, re-verify —
After the reverse override the calibration chip's tooltip denies the header without naming the override, while the strip beside it names it

Only consumers of CalibrationStatus.issue are status_chip.set_status (DAMAGED-guarded) and the DAMAGED branch of the CALIBRATE stage reason, so carrying the sentence on NO_CHANNEL changes no other surface. The two-line consumer change is minimal because the chip shows `issue` verbatim exactly as it already does for DAMAGED ('two surfaces, one description').

### D4 [minor] — fixer NOT_FIXED, re-verify —
'Treat this channel as' tooltip, code comment and USER_GUIDE still say the override only picks a pane

EXACT CHANGE NEEDED in docs/USER_GUIDE.md (the 'Treat this channel as' bullet, ~lines 201-205). Replace: 'it decides which **pane** the channel gets and which **overlay group** it shares, once *Apply* commits it.' with: 'it decides which **pane** the channel gets and which **overlay group** it shares — and, since the whole pipeline resolves a channel\'s kind through this same override, which channel *Analysis ▸ Run detection*, *Batch detect…*, *Power spectrum…* and *I-V* act on and what the calibration chip and the workflow strip report (*current* makes an undeclared channel analysable; *voltage* on a channel the file calls current switches detection off for it) — once *Apply* commits it.' The clause MUST sit before 'once *Apply* commits it.' because the repro's non-greedy regex stops at that phrase; text appended after it would not be seen. With that one edit the repro exits 0.

### E1 [major] — fixer NOT_FIXED, re-verify —
A second Batch detect from the never-measured recording re-stamps the measured recording 'batch' and says it was never measured

The runner change was PROTOTYPED as a monkeypatch (tree untouched) and driven through the real widgets on the owner's own -600 mV / -1000 mV recordings: /private/tmp/claude-501/-Users-abrahamcampos-Desktop/6f43f43e-d8d3-4c35-89b0-a20f09cbc171/scratchpad/fixup/nanoscope_gui_baseline_panel_py/proto_E1_runner_half.py, output out/proto_E1_tree.txt -> 'RUNNER HALF RESOLVES E1: True'. With it: re-batch launched from the -1000 form leaves -600 at 'proposed' (panel 'These values: measured on this recording, and confirmed', Stats COPIED advisory False, exported .meta.json baseline_param_source 'proposed'), -1000 stays 'batch'; row counts unchanged 276 / 98 before and after; and a re-batch at OTHER windows typed on -600 ('user' 1.5/0.5) stamps 'user' at home and 'batch' on -1000, i.e. the keep is gated on identical windows, not on the recording. Numbers did not move anywhere. Note the exported CSV/meta for -600 written by the prototype is out/proto_E1_600_after_rebatch.* in scratch, not beside the owner's data.

### F-SK-1 [major] — fixer FIXED, re-verify PARTIAL
Notch-line refusal is erased by the run it accompanies; on the Ctrl+D path nothing survives

NOTE SURFACE: FIXED, measured. Original repro on the tree (cwd /Users/abrahamcampos/Desktop/nanoscope/docs/sweep7/scratch/JRN, PYTHONPATH=.) exits 0. Menu/Ctrl+D road on the owner's -1000 mV excerpt: right after trigger box '60, 120', refusal painted True; after the run the note still reads 'Explicit spelling: 2 line(s) - 60, 120 Hz ...\n⚠ Refused: 1 entry is not a positive frequency in Hz - ‘abc’. The box was restored to its previous contents (60, 120). Nothing was dropped and nothing ran.', detection stamped notch_freqs [60.0, 120.0]. The spy prints the exact call the defect named - analysis_panel.py:914 _seed_form_from -> panels.py:3924 set_params -> set_notch_lines([60.0, 120.0]) - now "cleared the refusal: False". Output: /private/tmp/claude-501/-Users-abrahamcampos-Desktop/6f43f43e-d8d3-4c35-89b0-a20f09cbc171/scratchpad/reverify/F_SK_1/tree_repro.txt.

MY OWN RE-DRIVE through the real widgets, 11 doors (script probe1_widgets.py, output probe1.txt in the same dir): P1 typo refused via editingFinished -> exactly ONE notchLinesRefused emission, under=[60.0,120.0]; P2 set_notch_lines([60,120]) (the replay) KEEPS it; P3 set_notch_lines([60,180]) clears it; P4 retyping the same lis

### X-SURF-3 [minor] — fixer FIXED, re-verify PARTIAL
docs/inventory.json (and the manual Control reference generated from it) quotes five surfaces sweep 7 rewrote: the probe tooltip, the 'These values' label set, the score box's 'eight rows', the gallery status line and the '(sampled)' calibration box

ALL FIVE inventory.json claims are genuinely fixed, verified widget-first; the finding's stated reach into the SHIPPED manual.md is untouched, which is why this is PARTIAL and not FIXED.

REPRO (unmodified original), cwd=/, QT_QPA_PLATFORM=offscreen: all 8 checks [ok], exit 0. But that exit is weak evidence — 5 of the 8 are negative substring tests and the 2 manual.md checks belong to X-SURF-8, so I re-drove everything myself.

BEFORE ORACLE, two independent ways:
- /private/tmp/.../reverify/X_SURF_3/repro_X2_BEFORE_inventory.py (the original repro with only its INV path pointed at scratchpad/fixup/docs_inventory_json/inventory.BEFORE.json, live app): all five inventory lines [BAD] -> "REPRODUCED [probe-tooltip, these-values, score-rows, gallery-status-1-9, raw-vs-calibrated]".
- Snapshot tree (cd / && PYTHONPATH=/Users/abrahamcampos/Desktop/nanoscope-reports/pre-sweep7-fixes-2026-09-01, INV=<snapshot>/inventory.json) with my own probe: 16 [BAD]. Importantly the snapshot's tooltip check, its 8-row check and its "Raw vs calibrated (sampled)" title check all PASS there — the old doc matched the old app — so my oracle is not punishing the old tree for the old app. The one exception is

### X8 [minor] — fixer FIXED, re-verify PARTIAL
On an undeclared recording set back to 'Treat as voltage', the workflow rail and the chip say 'No current channel' without naming the override — only the Detect gate names it

RAIL HALF: FIXED and verified on real widgets. CHIP HALF: unchanged — the second of the two guided surfaces X8 names still never says "override".

1) Original repro on the tree — /private/tmp/.../scratchpad/verify/X-combos/repro_X8_rail_chip_reverse_override_unnamed.py exits 0 (was 7). But its criterion is an AND (gate names it AND rail does not AND chip does not), so it flips to "fixed" as soon as the RAIL names the override, even with the chip untouched. It printed exactly that: chip 'No current channel' | tip 'This recording has no live current channel to calibrate.'

2) My own real-widget drive, /private/tmp/claude-501/-Users-abrahamcampos-Desktop/6f43f43e-d8d3-4c35-89b0-a20f09cbc171/scratchpad/reverify/X8/p1_undeclared_reverse_override_surfaces.py (real MainWindow, real Calibration-panel combo + Apply on a COPY of the CAL fixture, real Detect run = 15 events, then back to voltage):
- CALIBRATE/FILTER/DETECT reasons AND the WorkflowStrip tooltips all read "... (the file says Untitled is unknown and your Calibration-panel override treats it as voltage)."; clicking each locked stage chip puts the same sentence in the status bar (stageActivated -> main_window.py:1276).
- Detect ga

## Full suite after the batch (2026-09-02)

Run in three invocations (never `tests/gui` and `tests/lifetime` together — the lifetime conftest
then shadows the gui one and 8 modules fail to collect):

| half | result |
|---|---|
| `tests` minus gui/lifetime | 3636 passed, 5 skipped, 1 teardown error |
| `tests/gui` | 3535 passed, 1 xfailed, 1 teardown error |
| `tests/lifetime` | 3 passed |

**7174 passed, zero real failures** (6793 before the batch: +381 tests).

BOTH teardown errors are the SAME false positive, and neither is a code failure.
`tests/conftest.py:198 _real_settings_store_is_never_written` hashes
`~/Library/Preferences/com.nanoscope-local.NanoScope.plist` at session start and re-checks it after
every test. Something outside the suite rewrote that file at 02:49:45 while both halves were running,
so each half flagged whichever test happened to be in teardown at that moment
(`test_fup_subpop_messages_gates.py::test_the_ceiling_gate_fires_on_one_cloud_cut_into_wedges` and
`test_ml_preprocess_gui.py::test_the_notebook_tail_convention_blinds_the_end_of_an_analysed_sub_range`).
Evidence it is not contamination: the plist holds only Abraham's genuine `recentFiles`/`lastDir`
entries with NO pytest tmp paths (the signature the guard exists to catch), and both files re-run
green in isolation (35 passed / 64 passed) leaving the plist hash `b225b034a5445ddd` unchanged.
The guard is working; it simply cannot tell an external writer from the suite.

## Carry-over closure (2026-09-02)

| id | sev | outcome |
|---|---|---|
| E1 | major | FIXED + re-verified. The keep is now gated on the batch form holding the windows a recording was CONFIRMED at, not on which recording the batch was launched from. Landed in `gui/analysis_worker.py` (`_CONFIRMED_STAMPS`, `_stored_stamp_to_keep`, per-target `BATCH_SCOPE_KEY`), not in `baseline_panel.py`, because the prototype patched `BatchDetectController.start`. The fixer ALSO caught a false claim in the prescribed prototype: it kept a stored stamp even for the recording the form is about, which would have re-labelled a hand-typed 'user' window as 'proposed' — so the keep returns None when the scope already names this target. |
| C2 | minor | FIXED + re-verified. The subpopulation curation line compares the accepted SET (frozenset of event_id), falling back to the count only when the frame has no event_id column. |
| D4 | minor | FIXED + re-verified. The USER_GUIDE bullet now says the override also decides which channel Run detection, Batch detect, Power spectrum and I-V act on, and what the chip and strip report. |
| D3 + X8 | minor | FIXED by the coordinator in `gui/status_chip.py` — the file no fixer owned. `CalibrationChip.set_status` was gated on `state is DAMAGED`, so the override sentence `workflow_status` already put on `CalibrationStatus.issue` reached the strip's three locked reasons and not the chip. The guard is gone; the remedy suffix is chosen per state ("lower the gain" only for DAMAGED, since it is nonsense advice for an override). `repro_X8_v2_chip_half_asserts_positively.py` (each of the three surfaces asserted POSITIVELY, unlike the original AND-of-negatives) now exits 0 with all three naming the override. |

**Tests:** `tests/gui/test_s7_close_chip_names_the_override.py`, 6 tests. Proved failing against the
pre-edit behaviour with an out-of-tree pytest plugin that restores the DAMAGED-guarded
`set_status` (never by editing source back and forth): 3 fail, and the 3 controls — the generic
tooltip when no override is involved, the chip's unchanged text and neutral colour, and the DAMAGED
case keeping its own sentence and its own remedy — pass in BOTH states, which is what stops the fix
over-reaching. The pre-sweep-7 snapshot is the wrong oracle here: it predates the `issue` field
entirely, so all six error there for a reason that has nothing to do with this change.
57 chip-related tests pass (6 files).

**Two items stay open, both by the fixer's own measurement, both minor and both documentation:**
CW-4 and X-SURF-3/X-SURF-8 reach `docs/manual-markdown/manual.md` and the inventory-fed Control
reference, which this batch deliberately left alone. The manual therefore still describes "Treat
this channel as" as a layout-only override and still quotes five surfaces sweep 7 rewrote.
