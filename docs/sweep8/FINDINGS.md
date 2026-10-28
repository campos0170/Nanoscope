# Sweep 8 — the October additions (2026-10-27)

**Method.** Eleven finders — seven by area (Signal Conditioning, the Results dock, the Event inspector and Window menu, geometry/layout/quit, the time-unit declaration, cross-surface interactions, documentation) and four by lens (data safety, concurrency and lifetime, numeric honesty, keyboard and focus) — each driving the real application offscreen, with preferences and recovery redirected. The area finders worked on the owner's recordings directly; one of them wrote a stray sidecar (see Housekeeping), so the lens finders were started on copy-on-write clones behind a write guard. Every area-round finding was then attacked by two independent SKEPTICS whose brief was to refute it: default REFUTED when it will not reproduce, `not-a-defect` when the app already discloses it where it matters. A finding survives only when BOTH reproduced it. The lens round was interrupted by usage limits before its skeptic stage (see the tiers below). Every defect in both tiers was reproduced by running the application; anything found by reading code alone is marked as such under the leads. The lens finders were told what the area finders had already found, so they raised only new defects.

**Result: 41 distinct defects — 10 major and 31 minor — in two evidence tiers.**

- **Tier A (area round, 22 defects: 4 major, 18 minor).** 28 reports were raised by the seven area finders; 27 survived both skeptics (merged into 22 distinct defects), 1 did not. Each was reproduced by the finder, by two independent skeptics, and again by the coordinator's re-run.
- **Tier B (lens round, 19 defects: 6 major, 13 minor).** The four lens finders (data safety, concurrency and lifetime, numeric honesty, keyboard and focus) worked for about 70 minutes and were then cut off by the account's session limit before writing anything up; the workflow received no results from them. What they left on disk was salvaged: 19 reproducible scripts, each of which the finder ran to a REPRODUCED verdict and the coordinator re-ran on the clean tree (the concurrency finder left probes, not scripts; two of its measured defects were turned into scripts by the coordinator, S8-THREAD1 and S8-THREAD2). **No skeptic attacked a tier B defect**, so each rests on one finder's reading plus a mechanical re-run; the severity of tier B is the coordinator's call. Treat them as confirmed reproductions awaiting an independent read.
- The skeptics additionally found 6 defects the finders missed (unverified, listed after the tiers).
- **4 open leads are not counted** (listed after the skeptics' finds). The one that matters most is S8-LEAD1: a storm of random open / close / detect operations segfaults the application intermittently, and only once the Signal Conditioning window has been created. It is not attributed to a line of code and is not shown to be reachable at human speed.

**Context.** The territory is the October additions that sweep 7 never saw (`git log 1aeb651..a53bf94`): the Results dock, the Signal Conditioning workspace, the docked Event inspector and Window menu, faster curation, the quit / layout-reset / geometry fixes, and the CSV time-unit fix with its sample-rate / time-unit declaration (`132b42b`) — which had been code-reviewed three times but never swept through the real application. The tree under test was `a53bf94`, unmodified. Sweep 7's own territory was not re-swept. No source file was changed by the sweep; fixing is a separate step. The repro scripts live in `docs/sweep8/repros/` and, like sweep 7's, are ignored by git (`.gitignore`: `/docs/sweep*/repros/`). Each repro was re-run by the coordinator on the clean tree under a write guard (see **Re-run** under each finding). The first run stopped when the account's usage credits ran out after the area round; the lens round was resumed with the area results reused unchanged, and was cut off again by the session limit (see Tier B above).

| ID | Severity | Tier | Defect |
|---|---|---|---|
| S8-SC3 | major | A | A time base declared in File Info never reaches an open Signal Conditioning window |
| S8-RD1 | major | A | Results Fits tab shows one fit per recording under any table, unmarked after merge or re-run |
| S8-TU1 | major | A | Navigator strip stays on the old clock after a sample-rate / time-unit declaration |
| S8-XS1 | major | A | Table selector does not re-feed the Event inspector: one R keypress curates two tables |
| S8-DATA1 | major | B | After a declaration, hand-add, edge-drag and Merge all write new-clock rows into a table marked stale |
| S8-DATA2 | major | B | Export events writes a stale table with a sidecar that splices the old rate onto the new clock origin |
| S8-THREAD2 | major | B | A replaced Merge-all or Re-measure leaves bookkeeping that blocks the study windows |
| S8-NUM1 | major | B | Signal Conditioning's exact-pass cache ignores the sample rate; the shown trace is 26 pA off |
| S8-KEYS1 | major | B | Results table reads Cmd+A, Cmd+R and Cmd+F as the bare curation keys |
| S8-KEYS2 | major | B | Inspector and gallery Space / A / R take the key from the focused button or dropdown |
| S8-SC1 | minor | A | Spectrum note says 'widen the window to ≥ 10 s' on a 10.00 s window; the threshold is 10.49 s |
| S8-SC2 | minor | A | Step view: the 'causal (reference)' curve is already at 37 % of the step one sample before it |
| S8-SC4 | minor | A | 'Copy as Python' after a declaration writes a script that cannot rebuild the trace |
| S8-RD2 | minor | A | Reconciled table's amber line keeps asking to refresh sources that were already re-run |
| S8-EI1 | minor | A | Event inspector window narrower than ~800 px: status-bar widgets overlap |
| S8-EI2 | minor | A | Window menu entries do nothing for a minimized gallery, Figure Library or Signal Conditioning window |
| S8-EI3 | minor | A | Reset Window Layout keeps an open child window's size and re-remembers it on close |
| S8-EI4 | minor | A | Inspector toolbar does not keep its button order when it re-flows, contrary to the docs |
| S8-EI5 | minor | A | With Advance off, A / R leave the Event spinner on the next event while the readout says 'filtered out' |
| S8-GEO2 | minor | A | Child windows left on an external monitor reopen on the laptop screen at every launch |
| S8-GEO4 | minor | A | A floating Results, File manager or Inspector dock comes back hidden after a macOS-native quit |
| S8-TU3 | minor | A | Frequency controls clamped by the misread rate keep their values and carry to the next recording |
| S8-TU4 | minor | A | One recording exported in ms and in s detects 9 vs 10 events |
| S8-XS3 | minor | A | After a declaration the stale line's Re-run is named 'Merge events (4066 → 6)' in Undo and the status bar |
| S8-XS4 | minor | A | Study windows from the Incubation launcher are missing from the Window menu |
| S8-DOC2 | minor | A | USER_GUIDE says a new Event inspector may start at the first event; the app opens on the selected one |
| S8-DOC3 | minor | A | 'Show in trace' / T does nothing once the event is already the table's selected row |
| S8-DOC4 | minor | A | docs/inventory.json quotes a Response-tab note the app no longer prints |
| S8-DATA3 | minor | B | Cmd+Z in the Event inspector stops at the 30-step ceiling without saying so |
| S8-DATA4 | minor | B | Compare baseline estimators on a stale table opens a raw Python ValueError modal |
| S8-THREAD1 | minor | B | Detect during 'Measure this recording' leaves the probe button stuck on 'Measuring…' |
| S8-NUM2 | minor | B | Step readout cuts off notch ringing and reports a fraction of the real overshoot |
| S8-NUM3 | minor | B | 'Send to notch' snaps to the nominal 60 Hz and leaves a line 0.44 Hz away 15 dB down |
| S8-NUM4 | minor | B | Spectrum lines list is silently capped at 24 and the note reads as a count |
| S8-NUM5 | minor | B | A segment drops its parent's question about an undeclared time unit |
| S8-NUM6 | minor | B | The chain sentence prints '(order 8)' for a Gaussian and a window the masked median ignores |
| S8-NUM7 | minor | B | Notch note 'deepest N dB' is a plot-grid artifact, not a property of the filter |
| S8-KEYS3 | minor | B | Tab cannot leave the Results dock's tables |
| S8-KEYS4 | minor | B | Cmd+I in the gallery inverts the tile selection instead of opening the inspector |
| S8-KEYS5 | minor | B | The first Cmd+G of a session replaces the selected event with event 0 |
| S8-KEYS6 | minor | B | Signal Conditioning's toolbar cannot be reached from the keyboard |

**An incident of the sweep itself is recorded under Housekeeping** — a calibration sidecar written beside the owner's real `-600 mV` recording. It has been quarantined, and it repeats the one sweep 7 recorded.

---

## MAJOR

### S8-SC3 - A time base declared in File Info never reaches an open Signal Conditioning window

> A time base declared in File Info never reaches an OPEN Signal Conditioning window: it keeps conditioning and drawing at the old rate, says 'Exact for these stages · window 2000.00 s' over a 2 s recording, and refuses a 10 kHz low-pass as 'above Nyquist (25 Hz) for fs=50 Hz' while the form beside it accepts it at 50 kHz

*Signal Conditioning workspace* · also raised as S8-TU2, S8-XS2 · finder: major · skeptics: major/major

**What the user sees.** The user opens a CSV whose bare 'time' column is in milliseconds (the new Issue says it read as 50 Hz), opens Signal Conditioning (⇧⌘F), then corrects the time base with Inspector ▸ File Info ▸ Sample rate / time unit… → milliseconds. The status bar, File Info, the Detection form and the main trace all move to 50 kHz and 0–2 s. The Signal Conditioning window does not: its preview still spans 0–2000 s against the trace's 0–2 s, its status still reads 'No stage is on, so processed = raw · window 2000.00 s' / 'Exact for these stages · window 2000.00 s', and when the user now turns on the 10 kHz low-pass the form accepts it while the preview prints 'Preview unavailable: cutoff_hz=10000 Hz is at or above Nyquist (25 Hz) for fs=50 Hz. There is nothing to filter; this is a data or units error.' — the units error the user has just corrected. Nothing on the window says it is stale. Removing the declaration again while the window is open leaves it the other way round (window at 50 kHz, document at 50 Hz).

**Why it matters.** The declaration is the headline of 132b42b ('followed by segments, results marked stale') and this is the one live surface of the recording it never reaches. The window is the place the user goes to judge the chain on the trace, and every sentence it prints after the declaration is false for the recording the rest of the app now shows: the exactness verdict, the window length, the x axis, and a refusal that quotes the wrong Nyquist. A chain edit after the declaration still runs at the old rate (measured); only re-selecting the channel (a fresh source push) or closing and reopening the window heals it, and the window gives no hint that either is needed.

**Measured.** Fixture: 100,000 samples (2.0 s) of the -600 mV recording written as CSV with a bare 'time' header in ms → opens at 50.000 Hz (Provenance.MEASURED, Issue 'this file's time column steps by 0.02 per sample. Read as seconds that is 50 Hz — below the 1,000 Hz…'). Signal Conditioning open: source fs 50, status 'No stage is on, so processed = raw · window 2000.00 s'. Session.set_sample_rate_declaration(doc, SampleRateDeclaration('time_unit', time_unit='ms')) (the dialog's apply() call): document 50000 Hz, Detection form sample_rate() 50000 Hz, main trace visible_x_range (0.0, 1.99998); SC window source fs still 50, preview raw_curve.xData spans (0.0, 1999.98), status unchanged. det.set_params({'filter_cutoff_hz': 10000.0}) → form holds 10000.0; SC status 'Preview unavailable: cutoff_hz=10000 Hz is at or above Nyquist (25 Hz) for fs=50 Hz…'. ws.close() + Analysis ▸ Signal conditioning… → source fs 50000, 'Exact for these stages · window 2.00 s, padded ±0.01 s'. Declaration removed while open → document 50 Hz, SC source fs 50000. Heals: re-selecting the same channel (src fs 50000, x span 2.0) or reopen; does not heal: a chain edit (p09/p11 probes). Sites: gui/main_window.py:7427-7445 _wire_signal_conditioning connects calibrationChanged and documentClosed but not Session.sampleRateChanged; analysis/conditioning_preview.py:99-103 ConditioningSource.key = (doc, channel, gain, offset, n) carries no rate, and gui/signal_conditioning.py:1222-1235 set_source returns early on an equal key, so even a push would not reschedule. Overlaps S8-TU2 (same defect reached from the time-base side).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-SC3_declared_rate_leaves_open_window_stale.py`; `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-TU2_signal_conditioning_old_clock.py`; `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-XS2_signal_conditioning_ignores_time_base_declaration.py`

**Re-run.** `S8-SC3_declared_rate_leaves_open_window_stale.py` → reproduced (exit 7); `S8-TU2_signal_conditioning_old_clock.py` → reproduced (exit 0); `S8-XS2_signal_conditioning_ignores_time_base_declaration.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, major). Reproduced exactly as reported, and it survives every variation I tried through the real door (SampleRateDialog ▸ Apply, the dialog File Info opens): the ms-column CSV declared as ms, the owner's -600 mV ABF re-declared at 25 kHz, and docs/sweep7's neutral-header -700 mV CSV declared at 100 kHz. In all three the document, the Detection form and the main trace move to the declared rate while the open Signal Conditioning window keeps its ConditioningSource at the old fs: the Time tab's x axis and 'window N s' sentence, the Spectrum tab's frequency axis (stops at 25 Hz, the stale Nyquist) and every refusal it prints are for the old clock, and no label on the window mentions the declaration or says it is stale. […]

**Skeptic 2** (confirmed, major). Reproduced with the finder's repro (exit 7) and with my own variation that goes through the real SampleRateDialog widget (mode combo → "The time column is in…", unit → ms, the Apply button). The dialog answers "Declared: 50,000 Hz…", the document, the Detection form and the main trace move to 50 kHz / 0–2 s, and the open Signal Conditioning window does not: source fs stays 50, the preview x axis spans 0–2000 s, the status still says "window 2000.00 s", the legends still say "processed — exact", and the Spectrum note says "Δf 0.01 Hz … on a 2000.00 s window". Nothing discloses it: […]

**Sites.** `nanoscope/gui/main_window.py:7427-7445 (_wire_signal_conditioning: no sampleRateChanged feed)`; `nanoscope/analysis/conditioning_preview.py:99-103 (ConditioningSource.key has no sample rate)`; `nanoscope/gui/signal_conditioning.py:1222-1235 (set_source: same key → no reschedule)`; `nanoscope/gui/analysis_panel.py:6133-6165 (_on_sample_rate_changed refreshes the form, the baseline panel and the filter preview — not the window)`

**Same root, also found by a skeptic.** Signal Conditioning window's source fs and chain fs disagree after a declaration (or a revert) while it says 'Exact for these stages' — Same root cause as S8-XS2 but a distinct wrong state: with SC open, declare 25,000 Hz on the -600 mV ABF, then run a detection (which pushes the chain at the form's new rate). The window now holds source fs 50,000 (time and spectrum previews) and chain fs 25,000 (response previews / stage design) at once, and its status reads 'Exact for these stages · … 59.9 s view · window 40.00 s' over a recording the app now says is 119.75 s long. […]


### S8-RD1 - Results Fits tab shows one fit per recording under any table, unmarked after merge or re-run

> The Results dock's Fits tab shows one stored fit per (recording, channel) under whichever table the Table selector names, and keeps it unmarked through a merge and a re-run — the export's own guard refuses the same fit as stale

*Results dock* · finder: major · skeptics: major/major

**What the user sees.** On the -600 mV recording the user presses Fit under the Schmitt table (316 accepted): the Fits tab reads "n = 316". The user switches the Table selector to CUSUM (370 candidates, never fitted): the Fits tab still reads Schmitt's fit, n = 316, under a table of 370. The user presses Fit under CUSUM: the dwell fit refuses (4 events below t_min) but the depth fit lands on CUSUM's 369 events; switching back to Schmitt the tab now prints Schmitt's dwell fit (n = 316) and CUSUM's depth fit (n = 369) as one readout under the Schmitt table. Merge all (317 → 247 rows, 246 accepted): the tab still prints n = 316. Re-run detection: still n = 316. The same fit then shows under the Reconciled table (443), and a fit made on the Reconciled table (n = 442) shows under CUSUM (370). Nothing on the tab, its label or its tooltip ("Dwell and depth fits of the accepted events.") says the fit belongs to another table or predates the one on screen.

**Why it matters.** The Results dock put the Table selector and the Fits tab in one strip and wrote the tooltip that claims the fits are of the accepted events — the events of the table beside them. The store is keyed (doc_id, channel, 'fit.dwell'/'fit.depth') with no detector lineage, so three tables share one fit slot and a half-failed Fit leaves a chimera (one detector's dwell fit beside another's depth fit) with no sentence anywhere. The app already knows the rule: MainWindow._stored_fit refuses the identical fit for Export figure with "saved fit is stale (the accepted events changed since it ran)" and on a calibration-fingerprint mismatch — the tab applies neither guard. tau, mu and their 95% CIs are the numbers that end up in a methods section, read off the tab beside a table they do not describe. Screen-only (the export is guarded), which is the one mitigation.

**Measured.** Schmitt 317 rows / 316 accepted, Fit -> Fits text 'exponential (left-truncated MLE) n = 316 ...' + 'Gaussian mixture ... n = 316'; tab tooltip 'Dwell and depth fits of the accepted events.'. Selector -> 'CUSUM — 370 candidates' (369 accepted, no fit): Fits text identical (== Schmitt's). Fit under CUSUM: dwell 'dwell fit failed: 4 event(s) are shorter than the stated detection limit t_min=0.001 s', depth 'n = 369'; back on 'Schmitt — 317 candidates': Fits = dwell n = 316 + depth n = 369 (chimera). Merge all at 10 ms -> 247 rows / 246 accepted: Fits heads unchanged (n = 316 / n = 369); win._stored_fit(current_detection, 'fit.dwell') -> (None, 'saved fit is stale (the accepted events changed since it ran)'). Re-run -> 317 rows: Fits unchanged, stored fit.dwell still present. Reconciled 443: Fits shows n = 316; Fit there -> n = 442; selector -> CUSUM 370: Fits shows n = 442. Visible labels on the Fits page besides the text: []. Stored fit: module_id 'fit.dwell', keyed doc/channel only; payload.meta has n_input_events=316 and no detector id. Also seen (p15): a k_in=50 re-run (44 events) keeps the n = 316 fit on the tab, and Undo of that run puts the 44-event fit under the 317 table. Sites: nanoscope/gui/analysis_panel.py:5506-5533 _refresh_fit_text (reads results.get(doc_id, channel, 'fit.dwell'/'fit.depth'), no module/n/fingerprint check); :5778-5820 _on_fit_requested (fits self._stored_detection() of the moment, add_result under the shared keys; a failed dwell fit leaves the previous one in the store while the depth fit is replaced); :1372-1377 the Fits tab and its tooltip (new in e2046a2); nanoscope/gui/main_window.py:10154-10179 _stored_fit (the export's fingerprint + n_input_events guard the tab lacks).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-RD1_fits_tab_ignores_table.py`

**Re-run.** `S8-RD1_fits_tab_ignores_table.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, major). The finder's repro reproduces exactly as filed (exit 7), and my own variations through the plainest user paths reproduce it too: a single rejection, a k_in re-run, and Edit > Undo all leave a fit on the Fits tab that describes a different accepted set than the table beside it, with no mark anywhere on the tab, the Summary tab or the amber stale line, while MainWindow._stored_fit refuses that identical fit for the figure export as 'saved fit is stale (the accepted events changed since it ran)'. The Undo case is worse than the filing's main line: Undo restores the 317-row table but leaves the 44-event fit (tau from 44 events) printed under it. Not-a-defect does not apply: […]

**Skeptic 2** (confirmed, major). Reproduced myself, both the finder's script and my own variations through the real widgets. The Fits tab text is written by AnalysisPanel._refresh_fit_text from results.get(doc_id, channel, 'fit.dwell'/'fit.depth') with no check of which table, which run or which accepted set the fit came from, and nothing on the page, the tab tooltip, the dock's amber stale line or the Fit gate label marks it. The app itself holds the opposite rule for the same stored object: MainWindow._stored_fit refuses it for the figure export the moment n_input_events differs from the accepted count or the calibration fingerprint moved. The simplest path is worse than the finder's multi-table one: […]

**Sites.** `nanoscope/gui/analysis_panel.py:5506-5533 (_refresh_fit_text)`; `nanoscope/gui/analysis_panel.py:5778-5820 (_on_fit_requested)`; `nanoscope/gui/analysis_panel.py:1372-1377 (Fits tab + tooltip, e2046a2)`; `nanoscope/gui/main_window.py:10154-10179 (_stored_fit — the guard the export has)`


### S8-TU1 - Navigator strip stays on the old clock after a sample-rate / time-unit declaration

> The Navigator strip stays on the old clock after a sample-rate / time-unit declaration: it flips to 'Local context · 9× view' over a 2 s recording, draws none of its samples, and drags the trace past the end of the data

*The time-unit declaration through the real app* · finder: major · skeptics: major/minor

**What the user sees.** Open a bare 'time,current' CSV written in milliseconds (reads as 50 Hz, t = 10,000–12,000 "s"), then Inspector ▸ File Info ▸ Sample rate / time unit… ▸ "The time column is in… milliseconds" ▸ Apply. The status bar says 50,000 Hz · user, File Info says '50,000 Hz · user (file: 50 Hz · measured)', the trace re-frames to 10–12 s. The navigator strip under the trace does not: its label flips to 'Local context · 9× view', its curve is empty (the 100,000 samples are still plotted at x = 10,000–12,000 while the strip shows 10–28 s), and dragging its shaded region to 13–15 s moves the main trace to [13, 15] s — past the last sample at 11.99998 s — so the pane goes blank. Zoom-to-full does not repair it; only selecting another recording in the File Manager and coming back does.

**Why it matters.** The navigator is the one always-on map of 'where am I in this recording'. After the declaration it is silently wrong in both directions: it misreports the view as a deep zoom into a 2,000 s file, and its region bounds (min(old extent, view) … max(old extent, view)) let the user pan the main trace thousands of seconds off the data. The same happens in reverse on 'Use the file's own time base', and on a refused (sample-axis) recording declared in microseconds (strip extent (0, 99999) samples under a 10–12 s trace). Nothing on screen says the strip is stale; the fix is a re-selection nobody is told to make.

**Measured.** S8-TU1 repro output (real MainWindow, offscreen, bare_ms CSV = 2 s slice of the -600 mV ABF): before: label 'Full recording', extent (10000.0, 11999.98), strip x (1e4, 1.2e4), 100000 samples drawn inside the strip. After declaring ms: trace x-range [10.0, 11.99998], axis 's (instrument clock — this recording starts at t₀ = 10.0000 s)'; navigator label 'Local context · 9× view', local_mode=True, extent still (10000.0, 11999.98), strip x (10, 28), curve x (10000, 12000), samples drawn inside the strip = 0, region (10.0, 12.0). Drag the region to (13, 15): main trace x-range [13.0, 15.0] while data end at 11.99998 s. Scan bar Full: still 'Local context · 9× view', strip (10, 28), 0 samples drawn. Re-select another recording and back: label 'Full recording', extent (10.0, 11.99998), 100000 samples drawn. Same on bare_us.csv declared µs: extent (0.0, 99999.0) samples, context (2.00008, 19.9999) under a 10–12 s trace (scratch explore3 E). Sites: nanoscope/gui/navigator.py:111-114 (bind connects layersChanged, documentClosed, calibrationChanged — not Session.sampleRateChanged); nanoscope/gui/trace_view.py:618 (_on_sample_rate_changed rebuilds the view but nothing re-feeds the strip); nanoscope/core/session.py:420 (sampleRateChanged.emit).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-TU1_navigator_stale_after_declaration.py`

**Re-run.** `S8-TU1_navigator_stale_after_declaration.py` → reproduced (exit 0) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, major). The finder's repro reproduces exactly as shipped, and it reproduces through the real dialog's Apply button, in the reverse direction ('Use the file's own time base'), and on the owner's -600 mV ABF with a Hz declaration. The Navigator's bind() connects layersChanged / viewModeChanged / documentClosed / calibrationChanged but not Session.sampleRateChanged, so after a declaration the strip keeps the old-clock curve and extent while the trace is on the new clock. On the 2 s ms-CSV the strip flips to the false label 'Local context · 9× view', draws 0 of its 100,000 samples, and its region (bounds = min(old extent, view)..max(old extent, view)) pans the trace to [13, 15] s past the last sample at 11.99998 s. […]

**Skeptic 2** (confirmed, minor). Reproduced exactly as reported, and also through the real Inspector ▸ File Info ▸ "Sample rate / time unit…" button (modal exec(), driven by a timer), not only the finder's hand-built dialog. Navigator.bind (nanoscope/gui/navigator.py:111-114) connects layersChanged, viewModeChanged, documentClosed and calibrationChanged but not Session.sampleRateChanged; the only listeners of that signal are main_window (dirty flag, status label), file_dock, measure_panel, analysis_panel, trace_view and file_info_panel (grep over nanoscope/). So after a declaration the strip keeps its curve, extent and region bounds on the old clock while the trace, scan bar, status bar and File Info move to the new one. Nothing on the surface discloses it: […]

**Sites.** `nanoscope/gui/navigator.py:111-114 (bind: no sampleRateChanged connection)`; `nanoscope/gui/navigator.py:185-245 (_refresh: the only path that re-reads x_axis / extent)`; `nanoscope/gui/trace_view.py:618 (_on_sample_rate_changed: the trace re-frames, the strip is not told)`; `nanoscope/core/session.py:420 (Session.set_sample_rate_declaration emits sampleRateChanged)`


### S8-XS1 - Table selector does not re-feed the Event inspector: one R keypress curates two tables

> Switching the Results panel's Table selector does not re-feed the Event inspector, so one R keypress curates TWO tables and Edit > Undo restores only the one on screen

*Cross-surface interactions* · finder: major · skeptics: major/major

**What the user sees.** The user has a Schmitt table (317 rows) and a CUSUM table (370 rows) on the -600 mV recording, the Event inspector docked beside the trace on Schmitt event 159 (the README: 'Docked, its selection stays in sync with the Results table'). The user switches the Results panel's Table selector to CUSUM; the Results dock and its summary now say CUSUM, the inspector still shows Schmitt event 159 with its progress line 'Reviewed 0 / 317'. The user presses R. Schmitt event 159 is rejected in the frame the inspector holds, CUSUM event 159 is rejected through the panel ('370 events · 369 accepted' becomes '368 accepted'), and the inspector jumps to CUSUM event 0. Edit > Undo reads 'Undo Curate event'; pressing it puts CUSUM back and leaves the Schmitt rejection in place, and the menu then reads 'Undo — nothing on this table; the threshold detector (1 step) on this channel'. The same happens with the inspector in its own window.

**Why it matters.** A curation gesture on an event the user is looking at writes a table the user is not looking at — and ALSO the one the user was looking at, under a name and count the inspector does not show. The Schmitt rejection then carries 'reviewed' and travels into the saved project and every export; the one Undo the user is offered does not reach it. The selector moved into the new Results dock (e2046a2) and the docked inspector is advertised as staying in sync with that dock, so the two surfaces disagree exactly where the user was told they agree. (S8-RD1 reports the same root — the selector changes the view, the surfaces do not follow — for the Fits tab; this is the curation-writing half.)

**Measured.** docs/sweep8/repros/S8-XS1_table_selector_curates_two_tables.py, both hosts. Dock: inspector on Schmitt event 159, progress 'Reviewed 0 / 317 · accepted 316'; Schmitt accepted=True, CUSUM accepted=True. Selector -> 'CUSUM — 370 candidates', panel.current_detection().module_id = detect.cusum, summary '370 events · 369 accepted'; inspector._df IS the Schmitt frame (True), progress line unchanged, event 159. QTest R on the inspector -> Schmitt accepted=False (reviewed), CUSUM accepted=False (reviewed), Results summary '370 events · 368 accepted', inspector now detect.cusum event 0. Edit > Undo 'Undo Curate event' -> Schmitt False, CUSUM True; menu 'Undo — nothing on this table; the threshold detector (1 step) on this channel'. Switching back to Schmitt and undoing there restores it. Window host: identical on event 162. Sites: gui/analysis_panel.py:1697 _show_result_choice (shows the chosen table, emits nothing MainWindow refreshes on) and :1633 _set_active_detection; gui/main_window.py:160-200 SURFACE_MUTATION_SIGNALS (no 'shown table changed' signal; _refresh_event_inspector at :7817 runs only on mutations); gui/main_window.py:6392 insp.acceptanceChanged -> analysis_panel.set_event_accepted (:5465, writes the CURRENT table by event_id) while gui/event_inspector.py:2939 set_accepted_current writes the inspector's own frame in place; examples/apo-ferritin-new-features/README.md §6 ('Docked, its selection stays in sync with the Results table').

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-XS1_table_selector_curates_two_tables.py`

**Re-run.** `S8-XS1_table_selector_curates_two_tables.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, major). The finder's repro reproduces exactly as reported (exit 7, both hosts), and my own variations through the real widgets reproduce it under more realistic gestures and in more places. Switching the Results panel's Table selector (handler on QComboBox.currentIndexChanged, so a keyboard Key_Down on the combo is the same path as a mouse pick) changes what the Results dock shows but emits nothing MainWindow re-feeds the inspector or gallery on; both keep a reference to the previously shown table's frame and nothing on their surface names the detector. […]

**Skeptic 2** (confirmed, major). Reproduced exactly as reported, through the real widgets, in both hosts, and it survives every variation I tried to refute it with. The Results panel's Table selector (`_show_result_choice`, analysis_panel.py:1697) re-seeds the panel and emits nothing MainWindow refreshes on; `_refresh_event_inspector` (main_window.py:7817) runs only on the SURFACE_MUTATION_SIGNALS, so an open inspector keeps the previous table's frame by reference. The inspector's `acceptanceChanged(event_id, accepted)` (event_inspector.py:2998) carries no table identity and is wired to `analysis_panel.set_event_accepted` (main_window.py:6392 -> analysis_panel.py:5465), which writes the CURRENT table by event_id. […]

**Sites.** `nanoscope/gui/analysis_panel.py:1697`; `nanoscope/gui/analysis_panel.py:1633`; `nanoscope/gui/analysis_panel.py:5465`; `nanoscope/gui/main_window.py:160`; `nanoscope/gui/main_window.py:6392`; `nanoscope/gui/main_window.py:7817`; `nanoscope/gui/event_inspector.py:2939`; `examples/apo-ferritin-new-features/README.md:§6`

**Same root, also found by a skeptic.** Event gallery shares the root: a Table-selector switch leaves the open gallery on the previous table, and R on a tile curates both tables; Undo restores only the one on screen — Schmitt and CUSUM tables on the -600 mV recording; Event gallery open (Analysis > Event gallery) on Schmitt tile 21; Results panel's Table selector moved to CUSUM; R on the gallery's selected tile. Schmitt event 21 and CUSUM event 21 are both rejected with reviewed set; Results summary drops to '370 events · 367 accepted'; Edit > Undo 'Undo Curate event' restores CUSUM 21 only.

**Same root, also found by a skeptic.** After a Table-selector switch, clicking a Results row shows a DIFFERENT event in the still-open inspector under the same event id (sync by id, not by event) — Inspector docked on Schmitt; Table selector moved to CUSUM; user clicks CUSUM row 161 in the Results table. The row highlights CUSUM event 161 (t_start 25.9445 s, dwell 14360 us, depth 280.49 pA); the inspector navigates its stale Schmitt frame and shows Schmitt event 161 (t_start 34.5206 s, dwell 15160 us, depth 349.5 pA) with readout 'Event 161 [162 of 317 · All]'. Same root as S8-XS1 (the selector does not re-feed the inspector); […]

**Same root, also found by a skeptic.** After a Table selector switch, selecting a Results row shows the inspector the OLD table's event under that id (README §6 'selection stays in sync with the Results table' false in the read direction too) — Inspector docked and holding Schmitt (fed when Schmitt was on screen). User switches the Results Table selector to CUSUM and clicks CUSUM row event_id 7 (what panel.select_event does). The inspector does not re-feed: it shows Schmitt event 7 — readout 'Event 7 [8 of 317 · All]', dwell 24560 µs, start_idx 102219 — while the Results dock's selected CUSUM event 7 has start_idx 102217, dwell 25160 µs, and the dock says 370 events. Same root as S8-XS1 (no re-feed on the selector switch; […]


### S8-DATA1 - After a declaration, hand-add, edge-drag and Merge all write new-clock rows into a table marked stale

> after a time-base declaration, hand-add, edge-drag and Merge all write rows on the NEW clock into the table the amber line says is entirely on the OLD one

*Lens: data safety and persistence* · **tier B** — found by the data-safety lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). after a time-base declaration, hand-add, edge-drag and Merge all write rows on the NEW clock into the table the amber line says is entirely on the OLD one. Inspector > File Info > Sample rate / time unit... marks every detection made at the old rate stale and says (CHANGELOG, USER_GUIDE section 2) that stored detections "keep the rate they were measured at ... never silently re-scaled". The three in-place editing doors do not honour that: ``add_current_selection_as_event`` (Shift+Cmd+M), the Event inspector's span drag (``_apply_edited_bounds`` -> ``_recharacterize_row``) and Merge all (``merge_by_gap`` on ``_active_signal``) all read the LIVE rate (``ch.sample_rate_hz`` / ``self._sample_rate()``) and write ``t_start_s`` / ``dwell_s`` at it, into a table whose provenance still says ``fs`` = the old rate and whose banner still says every time in it was computed at the old rate. One table, two clocks. (The fourth door that re-measures, Re-measure depths / Compare baseline estimators, DOES refuse with exactly this reason -- see S8-DATA4 -- so the kernel knows the rule; these three do not check it.) A 2 s, 50 kHz slice of the -600 mV recording (copied to a temp dir first) is detected at 50 kHz and then declared to be 40 kHz.

**See also.** S8-DATA4 (the one door that re-measures DOES refuse a stale table) and S8-DATA2; the stale-table family is also S8-XS3 and S8-RD2.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DATA1_edits_write_declared_clock_into_stale_table.py`

**Re-run.** `S8-DATA1_edits_write_declared_clock_into_stale_table.py` → reproduced (exit 7, 15.2 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
detected 10 events at 50,000.00 Hz (table stamp fs = 50,000.00)
declare -> Declared: 40,000 Hz. Save the project to retain it; detections already made on this recording are marked stale.
stale line: The sample rate changed since this detection ran (50,000 Hz → 40,000 Hz): every start time, dwell, rise/fall time and ECD in this table was computed a
  [after the declaration             ] rows=10  on the table's own clock (50,000 Hz)=10  on the DECLARED clock (40,000 Hz)= 0  table stamp fs=50,000  stale line shown=True
  hand-added event 10: dwell_s=0.0075000 for 300 samples -> 40,000 Hz (the table's other rows: 50,000 Hz)
  [after hand-add                    ] rows=11  on the table's own clock (50,000 Hz)=10  on the DECLARED clock (40,000 Hz)= 1  table stamp fs=50,000  stale line shown=True
  [after dragging one event's edge   ] rows=11  on the table's own clock (50,000 Hz)= 9  on the DECLARED clock (40,000 Hz)= 2  table stamp fs=50,000  stale line shown=True
  [after Merge all (200 ms)          ] rows= 8  on the table's own clock (50,000 Hz)= 4  on the DECLARED clock (40,000 Hz)= 4  table stamp fs=50,000  stale line shown=True
status bar: Merged 3 fragment(s): 11 → 8 events.
The amber line still says: The sample rate changed since this detection ran (50,000 Hz → 40,000 Hz): every start time, dwell, rise/fall time and ECD in this table was computed at the old rate, and 
RESULT: REPRODUCED
```


### S8-DATA2 - Export events writes a stale table with a sidecar that splices the old rate onto the new clock origin

> Analysis > Export events writes a table the app itself has flagged stale, with a sidecar that splices the OLD rate onto the NEW clock origin, and nothing in either file says a declaration exists

*Lens: data safety and persistence* · **tier B** — found by the data-safety lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). Analysis > Export events writes a table the app itself has flagged stale, with a sidecar that splices the OLD rate onto the NEW clock origin, and nothing in either file says a declaration exists. After Inspector > File Info > Sample rate / time unit... the events table on screen is marked stale ("Times stale -- sample rate changed") and Export events goes ahead anyway: no dialog, status "Wrote 9 accepted event(s) ...". The CSV's t_start_s / dwell_s are, correctly for a stale table, still on the old clock. The ``.meta.json`` is built from two different places: ``fs`` comes from the table's own provenance (old rate) but ``t0_s`` and the ``clock`` note come from the LIVE recording meta (``scoring.clock_stamp(meta=...)``), i.e. the declared one. The note then tells the reader to "add t0 to a number here to find it on the trace", which for these numbers lands on the wrong sample. For a unit declaration the same CSV exported before and after differs only in ``t0_s`` (10000.0 -> 10.0). Neither file says the rate was declared by the user, that the table was stale, or what the live rate is.

**See also.** S8-DATA1 and S8-DATA4 (the same stale table, reached from the export door).

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DATA2_export_of_stale_table_splices_live_origin.py`

**Re-run.** `S8-DATA2_export_of_stale_table_splices_live_origin.py` → reproduced (exit 7, 17.0 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
A  before the declaration: sidecar fs=50000.0  t0_s=10.0  first event t_start_s=0.259920
   declare -> Declared: 40,000 Hz. Save the project to retain it; detections already made on this recording are marked stale.
   stale line on screen: True
   after:  sidecar fs=50000.0  t0_s=10.0  first event t_start_s=0.259920
   dialogs shown by the export: []   status: Wrote 10 accepted event(s) of 10 to A_after.csv (+ provenance A_after.meta.json).
   sidecar clock note: Time seconds from the start of this recording: 0 is the first sample. The trace x-axis is on the OTHER clock — add t₀ = 10.0000 s to a number here to find it on the trace.
   following that note: t0 + t_start_s = 10.259920 s;  that event's first sample is at 10.324900 s on the trace now (-64.98 ms off, 2599 samples)
   sidecar mentions of the declaration / staleness / the live rate: []
B  as read (50 Hz): sidecar fs=50.0000  t0_s=10000.0  first t_start_s=0.220000
   declare -> Declared: 50,000 Hz. Save the project to retain it; detections already made on this recording are marked stale.
   after:  sidecar fs=50.0000  t0_s=10.0  first t_start_s=0.220000   CSV identical to the first export: True
   -> the same CSV now carries t0_s 10.0 instead of 10000.0, while fs and every time in it are still the 50 Hz reading
RESULT: REPRODUCED
```


### S8-THREAD2 - A replaced Merge-all or Re-measure leaves bookkeeping that blocks the study windows

> after Detect replaces a running Merge-all or Re-measure, the abandoned job's bookkeeping stays set, so the Manual and Acid study windows refuse to prepare a review for the rest of the session ("Wait for main-workspace analysis and editing operations to finish first.") while nothing is running

*Lens: concurrency and lifetime* · **tier B** — found by the concurrency-and-lifetime lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). after Detect replaces a running Merge-all or Re-measure, the abandoned job's bookkeeping stays set, so the Manual and Acid study windows refuse to prepare a review for the rest of the session ("Wait for main-workspace analysis and editing operations to finish first.") while nothing is running. Detect REPLACES a running analysis job ("Replaced baseline_remeasure with detect.schmitt. The previous run's result will not be applied."). The panel records what a Merge-all / Re-measure job needs in AnalysisPanel._merge_job / _remeasure_target and clears them only when that job finishes, fails or is cancelled - none of which a replacement does. study_event_bridge._assert_idle (the guard in front of the Manual study's and Acid study's "prepare review" actions, and the main window's study-import path) treats either one as "an operation is running". Measured: the guard raises after the replacement settles, after selecting another recording, after a second Detect, and after closing every recording - nothing the user can do clears it short of running the same kind of job to the end or restarting the app. What it does (one child process per job kind, so each boots a clean app): the -500 mV recording (a COPY) is opened and detected; job A (merge or remeasure) is started; Detect is pressed; everything settles; the controller and panel are read and _assert_idle is called; then a second Detect and a recording change are tried.

**See also.** S8-THREAD1 (the same cause for the baseline probe: Detect replaces a running job and the replaced job's own bookkeeping is never cleared).

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-THREAD2_replaced_job_leftovers_wedge_the_study_bridge.py`

**Re-run.** `S8-THREAD2_replaced_job_leftovers_wedge_the_study_bridge.py` → reproduced (exit 7, 47.6 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
[merge] job A started (running); pressing Detect
[merge] status: 'Replaced merge_all with detect.schmitt. The previous run's result will not be applied.'
[merge] settled:                 merge_job=SET; remeasure_target=None; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[merge] other recording selected: merge_job=SET; remeasure_target=None; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[merge] after a second Detect:    merge_job=SET; remeasure_target=None; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[merge] WEDGED: the guard refuses with nothing running
[remeasure] job A started (running); pressing Detect
[remeasure] status: 'Replaced baseline_remeasure with detect.schmitt. The previous run's result will not be applied.'
[remeasure] settled:                 merge_job=None; remeasure_target=SET; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[remeasure] other recording selected: merge_job=None; remeasure_target=SET; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[remeasure] after a second Detect:    merge_job=None; remeasure_target=SET; controller busy=False; pending=None; study bridge REFUSES ('Wait for main-workspace analysis and editing operations to finish first.')
[remeasure] WEDGED: the guard refuses with nothing running
RESULT: REPRODUCED - wedged after a replaced job: ['merge', 'remeasure']
```


### S8-NUM1 - Signal Conditioning's exact-pass cache ignores the sample rate; the shown trace is 26 pA off

> the Signal Conditioning window's cache of "Compute exact" passes is keyed WITHOUT the sample rate, so after a time-base declaration it serves the pass computed at the OLD rate as the exact trace of the NEW one: the status line says "Whole recording (exact)", the button says "Done: the preview shows this chain's whole-recording (exact) samples", and the trace drawn is 26 pA away (noise sigma 4.5 pA) from what the conditioning chain really produces at the declared rate

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). the Signal Conditioning window's cache of "Compute exact" passes is keyed WITHOUT the sample rate, so after a time-base declaration it serves the pass computed at the OLD rate as the exact trace of the NEW one: the status line says "Whole recording (exact)", the button says "Done: the preview shows this chain's whole-recording (exact) samples", and the trace drawn is 26 pA away (noise sigma 4.5 pA) from what the conditioning chain really produces at the declared rate. What it does (the -600 mV apo-ferritin recording, a COPY of it, preferences/recovery in a temp dir): 1. IN15, main trace on 30-40 s, Signal Conditioning open, chain = baseline drift (block median, 0.5 s, polarity -1) + 5 kHz Bessel low-pass; press Compute exact; the exact pass is cached. 2. Inspector > File Info > Sample rate / time unit... : declare 25,000 Hz (the real dialog's Apply). 3. Re-open the window the way the user does (hide it, Analysis > Signal conditioning... again): the window now holds the 25 kHz source and chain, finds the cached pass under the same key and shows it as the exact trace. It then conditions the whole calibrated channel at 25 kHz with the app's own conditioned_trace and compares the two arrays.

**See also.** S8-SC3 (the same window's source key also omits the rate; SC3 is the open window that is never told, this one is the cache that survives re-opening it).

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM1_exact_pass_cache_ignores_sample_rate.py`

**Re-run.** `S8-NUM1_exact_pass_cache_ignores_sample_rate.py` → reproduced (exit 7, 14.6 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
1. before: Window estimate — Baseline drift uses whole-recording statistics · press Compute exact for the true trace
   after Compute exact: Whole recording (exact) · showing 10.00 s · the pass took 0.2 s
2. declared: Declared: 25,000 Hz. Save the project to retain it; detections already made on this recording are marked stale.
3. window source fs = 25000.0  status: Whole recording (exact) · centre 80.0 s of the 119.8 s view (preview limit) · the pass took 0.2 s
   exact button enabled: False | tooltip: Done: the preview shows this chain's whole-recording (exact) samples.
   preview came from an exact pass: True  flagged exact: True
4. declared rate 25000 Hz; noise sigma 4.52 pA
   max |shown - conditioned_trace at 25000 Hz| = 25.990 pA
   max |shown - the pass cached at 50 kHz|        = 0.000e+00 pA
RESULT: REPRODUCED
```


### S8-KEYS1 - Results table reads Cmd+A, Cmd+R and Cmd+F as the bare curation keys

> The Results table reads every modifier chord as the bare curation key: Cmd+A accepts, Cmd+R rejects, Cmd+F flags

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). The Results table reads every modifier chord as the bare curation key: Cmd+A accepts, Cmd+R rejects, Cmd+F flags. WHAT IT SHOWS. EventTable._handle_curation_key(key) is handed event.key() ONLY - the modifiers are never looked at (panels.py, EventTable.eventFilter -> _handle_curation_key). So with the keyboard in the Results table - which Shift+Cmd+R hands it to on purpose ("so a/r/f keep working") - the standard Mac chords Select All (Cmd+A), Find (Cmd+F) and Reload (Cmd+R), and every Ctrl/Option/Shift chord on a, r, f, j, k, act as the bare key: the selected rows are accepted / rejected / flagged ("Undo Curate event"), the status bar says nothing, and Select All never selects anything. The one chord main_window.py guards is Shift+Cmd+F (its comment: "the event table, whose curation keys ignore modifiers"); Shift+Cmd+R is an action, so it wins before the table sees it.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS1_table_keys_ignore_modifiers.py`

**Re-run.** `S8-KEYS1_table_keys_ignore_modifiers.py` → reproduced (exit 7, 16.2 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
Results rows: 317 | dock visible: True
Cmd+A  (Qt: Ctrl+A)  - Select All    focus=QTableView  accepted False->True  flag ''->''  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Cmd+R  (Qt: Ctrl+R)                  focus=QTableView  accepted True ->False flag ''->''  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Cmd+F  (Qt: Ctrl+F)  - Find          focus=QTableView  accepted True ->True  flag ''->'flagged'  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Ctrl+A (Qt: Meta+A)                  focus=QTableView  accepted False->True  flag ''->''  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Option+R                             focus=QTableView  accepted True ->False flag ''->''  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Cmd+Shift+A                          focus=QTableView  accepted False->True  flag ''->''  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
Option+F                             focus=QTableView  accepted True ->True  flag ''->'flagged'  selected rows 1->1  Edit>Undo='Undo Curate event'  status=''
A Mac user expects Cmd+A to select all 317 rows and Cmd+F / Cmd+R to do nothing here.
Measured: the chord is read as the bare a/r/f key and writes curation; Select All never happens (selected rows stay at 1).
RESULT: REPRODUCED
```


### S8-KEYS2 - Inspector and gallery Space / A / R take the key from the focused button or dropdown

> The Event inspector's and Event gallery's Space / A / R shortcuts take the key from the control that has the keyboard: a focused button is never pressed, and a letter typed into an open Filter dropdown curates instead of choosing

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (major)

**What it shows** (the finder's own account, from its repro). The Event inspector's and Event gallery's Space / A / R shortcuts take the key from the control that has the keyboard: a focused button is never pressed, and a letter typed into an open Filter dropdown curates instead of choosing. WHAT IT SHOWS. EventInspectorWindow._install_shortcuts and EventGalleryWindow (the A / R / Space loop) register A, R and Space as Qt.WidgetWithChildrenShortcut: they fire whenever the keyboard focus is anywhere in the window - including inside an open combo popup, which is a Qt.Popup child - and neither a QPushButton nor a QListView accepts ShortcutOverride for those keys. Measured, on the -600 mV detection: A. Tab to the inspector's "Show in trace" / "Gallery..." / "Reject all" and press Space (the only key that activates a focused button outside a dialog): the button's clicked never fires and the event on screen is TOGGLED (accepted True -> False). B. Open the inspector's Filter dropdown and type "r" to jump to "Rejected only" (or "a" for "Accepted only"): the filter does not change and the event on screen is rejected (or accepted); Space toggles it instead of choosing the highlighted item. C. Gallery: Cmd+A selects the page's 15 tiles; Tab to "Accept selected" / "Reject selected" and Space flips the selected tiles (13 of 15 here; accepted tiles become REJECTED under "Accept selected"), and typing "r" in the open Show dropdown rejects the 15 selected tiles instead of choosing "Rejected only". D. The other half of the same conflict: the controls the inspector took out of the Tab chain so Space cannot hit them (Advance after decision, Details, the Label combo with Apply and Clear, Dock beside trace) have no keyboard route at all. A CONTROL (a plain QPushButton in the same harness) is clicked by Space, so the harness is sound.

**See also.** S8-KEYS1 (the Results table's half of the same family: a shortcut taking a key that belongs to something else).

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS2_curation_keys_take_the_focused_controls_keys.py`

**Re-run.** `S8-KEYS2_curation_keys_take_the_focused_controls_keys.py` → reproduced (exit 7, 25.8 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
CONTROL  plain QPushButton, Space -> clicked 1x   (style 'fusion')
A. Event inspector, event 5, accepted=True
   Tab x14 -> focus on 'Show in trace' (QPushButton); Space -> button clicked 0x; event 5 accepted True->False; total accepted 316->315
   Tab x15 -> focus on 'Gallery…' (QPushButton); Space -> button clicked 0x; event 5 accepted True->False; total accepted 316->315
   Tab x18 -> focus on 'Reject all' (QPushButton); Space -> button clicked 0x; event 5 accepted True->False; total accepted 316->315
B. Event inspector Filter dropdown, items ['All', 'Accepted only', 'Rejected only', 'Unreviewed']
   popup open=True; key r  (meant: jump to 'Rejected only'): event 5 accepted True->False; filter 'All'->'All'
   popup open=True; key a  (meant: jump to 'Accepted only'): event 5 accepted True->True; filter 'All'->'All'
   popup open=True; key Space  (meant: choose the highlighted item): event 5 accepted True->False; filter 'All'->'All'
C. Event gallery: Cmd+A selected 15 tiles; accepted total 315
   Tab x15 -> focus on 'Accept selected' (QPushButton); Space -> button clicked 0x; total accepted 315->302  (13 tiles flipped)
   Tab x16 -> focus on 'Reject selected' (QPushButton); Space -> button clicked 0x; total accepted 315->302  (13 tiles flipped)
   Show dropdown open=True, 15 tiles selected; typing 'r' (meant: 'Rejected only'): accepted 315->302; Show 'All'->'All'
D. Event inspector controls with NO keyboard focus at all (mouse only):
   QCheckBox      'Advance after decision'   keyboard equivalent: none
   QToolButton    'Dock beside trace'        keyboard equivalent: none
   _ElidingCombo  ''                         keyboard equivalent: none
   QToolButton    'Apply'                    keyboard equivalent: none
   QToolButton    'Clear'                    keyboard equivalent: none
   QToolButton    'Details ▾'                keyboard equivalent: none
   QToolButton    'First unreviewed'         keyboard equivalent: Ctrl+U
RESULT: REPRODUCED
This plugin does not support grabbing the keyboard
This plugin does not support grabbing the keyboard
This plugin does not support grabbing the keyboard
This plugin does not support grabbing the keyboard
```


## MINOR

### S8-SC4 - 'Copy as Python' after a declaration writes a script that cannot rebuild the trace

> 'Copy as Python' on a recording whose time base was declared writes FS = <declared> and a script that re-reads the file at its as-read rate, so the copied script stops with 'expected 50000 Hz, the file says 50 Hz' and cannot rebuild the trace the window shows

*Signal Conditioning workspace* · finder: minor · skeptics: minor/minor

**What the user sees.** On the same ms-column CSV, after declaring the unit (and with a fresh Signal Conditioning window showing the right 50 kHz preview), the user presses Copy as Python — status bar: 'Copied a 74-line Python script that rebuilds this processed trace to the clipboard.' The script has PATH = …csv and FS = 50000.00000000106; run, it exits 1 with 'expected 50000.00000000106 Hz, the file says 50.000000000001066 Hz'. The declaration is nowhere in the script, so no edit short of re-implementing it yields the samples NanoScope shows.

**Why it matters.** The button's tooltip promises 'the same samples NanoScope shows' and the README calls it a standalone rebuild of the previewed window; for any recording that needed the new File Info declaration (exactly the recordings whose files say the wrong rate) the deliverable refuses by design — its own guard fires against its own FS line. It fails loudly, so nothing wrong is computed, but the feature is dead for the files the declaration exists for, and the script carries no note of the declaration to tell the reader why.

**Measured.** Document 50000 Hz after SampleRateDeclaration('time_unit', 'ms'); SC source fs 50000; copy_chain_as_python(ws.current_window()) → script lines 'PATH = …/ms_column_bare_header_600mV_2s.csv', 'FS = 50000.00000000106 # Hz'; subprocess run of the script: exit 1, stderr 'expected 50000.00000000106 Hz, the file says 50.000000000001066 Hz'. Control on the -600 mV ABF (no declaration): the copied script runs (rc 0) and its processed_view equals the window's exact preview to 0.0 and the detector's own detection_signal to 0.0 (notch+median+lowpass chain to 2.3e-4 pA, float32 rounding).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-SC4_copy_as_python_after_declaration_refuses.py`

**Re-run.** `S8-SC4_copy_as_python_after_declaration_refuses.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced exactly as filed, and through the real toolbar action (ws.act_copy_python.trigger() -> clipboard) rather than the panel method: on a recording whose time base was declared, the copied script writes FS = <declared rate> but re-reads the file with open_document, which yields the as-read rate, so its own guard (conditioning_recipes.py:993-994) stops it. It fails in BOTH declaration modes (time_unit=ms and sample_rate=50000 Hz) and on a microsecond column whose rate the reader refused (fs None -> 'expected 50000.0 Hz, the file says None Hz'). […]

**Skeptic 2** (confirmed, minor). Reproduced through the real toolbar action (ws.act_copy_python.trigger() → clipboard), not only the direct call. For a recording whose time base was declared in File Info, the copied script writes FS = <declared rate>, re-reads the file with open_document (which yields the as-read rate) and stops on its own guard 'expected 50000 Hz, the file says 50 Hz'. It does so for both declaration modes (time_unit=ms and sample_rate=50000) and for a segment cut from the declared recording; with the declaration reverted the same script runs (rc 0), so the declaration is the sole cause. The script carries no comment, constant or note about the declaration (grep for 'declar'/'provenance' finds nothing; […]

**Sites.** `nanoscope/analysis/conditioning_recipes.py:926-1015 (python_script: FS from meta.sample_rate_hz, the rate check at :993, no declaration)`; `nanoscope/gui/analysis_panel.py:6694-6740 (copy_chain_as_python passes meta.sample_rate_hz, not the time_base_as_read / declaration)`


### S8-SC1 - Spectrum note says 'widen the window to ≥ 10 s' on a 10.00 s window; the threshold is 10.49 s

> The Spectrum note tells the user on a 10.00 s window to 'widen the window to ≥ 10 s' — the real threshold is 10.49 s, printed with :.0f, so the advice is already satisfied by the window it describes and stays wrong up to 10.48 s

*Signal Conditioning workspace* · also raised as S8-DOC1 · finder: minor · skeptics: minor/minor

**What the user sees.** With the -600 mV recording and the main trace on a 10.00 s window, Signal Conditioning ▸ Spectrum reads 'Δf 1.53 Hz (nperseg 32,768) on a 10.00 s window — resolving mains needs Δf ≤ 1 Hz: widen the window to ≥ 10 s'. The user widens to 10.2, 10.4, 10.48 s: the same sentence. At 10.50 s it becomes 'Δf 0.76 Hz (nperseg 65,536) on a 10.50 s window'.

**Why it matters.** The sentence exists to tell them how far to widen so the line finder can place a mains notch; it names a number the user has already met, so following it changes nothing and the lines stay 'Δf too coarse'. The computed threshold (window_for_resolution_s = 8 segments × 65,536 / 50,000 = 10.486 s) is right; only its rounding is wrong, and the rounding is in the one direction that makes the advice false.

**Measured.** fs 50,000.0 Hz; window_for_resolution_s(fs, 1.0) = 10.48576 s. Views 9.00 / 10.00 / 10.20 / 10.40 / 10.48 s → nperseg 32,768, Δf 1.526 Hz, note '…widen the window to ≥ 10 s' on all five; 10.50 s → nperseg 65,536, Δf 0.763 Hz, no advice. Bisection: Δf ≤ 1 Hz first reached between 10.4857 and 10.4858 s of view. Overlaps S8-DOC1 (same sentence, found from the docs side).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-SC1_df_advice_rounds_down.py`; `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DOC1_spectrum_widen_advice.py`

**Re-run.** `S8-SC1_df_advice_rounds_down.py` → reproduced (exit 7); `S8-DOC1_spectrum_widen_advice.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced through the real widgets exactly as the finder described, and it generalises: the Spectrum note prints window_for_resolution_s with ":.0f" (round-to-nearest), so whenever the true threshold's fraction is below .5 the advice names a window the user has already met. On the owner's -600 mV recording (50 kHz) the threshold is 10.48576 s of view; the note says "widen the window to ≥ 10 s" at 9.99, 10.00, 10.485 and 10.4857 s, all with nperseg 32,768 and Δf 1.526 Hz, and only at 10.4858 s (one more sample past 8 × 65,536) does nperseg become 65,536, Δf 0.763 Hz, and the advice disappear. The same false sentence appears at 20 kHz and 40 kHz ("≥ 13 s" while 13.00 and 13.10 s still give Δf 1.221 Hz; […]

**Skeptic 2** (confirmed, minor). Reproduced exactly as filed through the real widgets: on the owner's -600 mV ABF (fs 50,000 Hz) the Spectrum note on a 10.00, 10.20, 10.40 and 10.48 s window reads "… resolving mains needs Δf ≤ 1 Hz: widen the window to ≥ 10 s" while Δf is 1.526 Hz (nperseg 32,768); the first window that resolves 1 Hz is window_for_resolution_s = 8 × 65,536 / 50,000 = 10.486 s, and `f"{need:.0f} s"` prints it as 10. The tooltip is the same text, so nothing on the surface discloses the real threshold; the user guide says "about a 10 s window", which is hedged where the app's sentence is not. […]

**Sites.** `nanoscope/gui/signal_conditioning.py:1854-1858 (f"{need:.0f} s")`; `nanoscope/analysis/conditioning_preview.py:257-263 (window_for_resolution_s)`


### S8-SC2 - Step view: the 'causal (reference)' curve is already at 37 % of the step one sample before it

> In the Response tab's Step view the dashed curve labelled 'single pass, causal (reference)' — 'the same design run once, causally (a hardware filter)' — is at 37 % of the step 20 µs BEFORE the step whenever the chain holds a running mean or Savitzky-Golay, and its 'Single pass: rise … · overshoot 0.0 %' readout describes that non-causal hybrid

*Signal Conditioning workspace* · finder: minor · skeptics: minor/minor

**What the user sees.** Chain Notch → Running mean (0.4 ms) → Savitzky-Golay (1 ms) → Low-pass (10 kHz Bessel), Response ▸ Step with the whole-chain toggle on. The legend says 'applied (zero-phase)' and 'single pass, causal (reference)'; the note says 'Dashed: the same design run once, causally (a hardware filter)' and 'Single pass: rise 463.8 µs · overshoot 0.0 %'. The dashed curve is already at 0.37 twenty microseconds before the step. With only IIR stages in the chain (notch + low-pass) the dashed curve is exactly 0 before t = 0.

**Why it matters.** The reference exists to show what a hardware filter of the same design would do to an edge, so the user can see the zero-phase pre-shoot for what it is; a 'causal' curve that rises before the edge contradicts its own label, and the rise/overshoot readout attached to it is of a filter nobody can build (centred smoothers followed by a causal cascade). Nothing downstream reads it, so the blast radius is the Response tab and what a user infers from it.

**Measured.** chain_step: reference max before t = 0 is 0.370 at t = −20 µs (step 0 → 1 at t = 0); applied pre-shoot 1.8 %; legend ['applied (zero-phase)', 'single pass, causal (reference)']; note 'Applied: 10–90 % rise 461.9 µs · overshoot 1.9 % · pre-shoot 1.8 %. Single pass: rise 463.8 µs · overshoot 0.0 %. … Dashed: the same design run once, causally (a hardware filter).' Chain with notch + hp + rmean + savgol + low-pass: reference max before t = 0 = 0.314. Chain notch + low-pass only: 0.000e+00. Mechanism: filter_views.chain_step pushes the probe through the centred smoothers (fir) and only then sosfilt (:339-344), as its docstring says; the legend and note on the window do not.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-SC2_causal_reference_rises_before_step.py`

**Re-run.** `S8-SC2_causal_reference_rises_before_step.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces verbatim on the real Signal Conditioning window (offscreen, settings/recovery redirected), and on the −700 mV recording as well: with the whole-chain toggle on and a centred smoother in the chain, the on-screen dashed curve under the legend entry 'single pass, causal (reference)' is already at 0.34–0.38 of a unit step 20 µs before the edge, while the note under the plot says 'Dashed: the same design run once, causally (a hardware filter)' and prints a 'Single pass: rise … · overshoot …' readout for that curve. A chain of IIR stages only (notch + Bessel low-pass, or a single-stage Bessel) gives exactly 0.000 before t = 0, so the label is true there and false exactly when a centred kernel is present. […]

**Skeptic 2** (confirmed, minor). Reproduced through the real Signal Conditioning window: with the chain Notch → Running mean (0.4 ms) → Savitzky-Golay (1 ms) → Low-pass (10 kHz Bessel), Response ▸ Step, whole-chain toggle on, the curve the legend calls 'single pass, causal (reference)' and the note calls 'the same design run once, causally (a hardware filter)' is non-zero from t = −640 µs and at 0.370 at t = −20 µs, on a plot whose x-axis spans −2.17 … +2.15 ms, so the pre-step rise occupies a third of the left half of the picture. […]

**Sites.** `nanoscope/analysis/filter_views.py:321-356 (chain_step: reference = sosfilt(sos, fir) after the centred smoothers)`; `nanoscope/gui/signal_conditioning.py:1055 (legend 'single pass, causal (reference)')`; `nanoscope/gui/signal_conditioning.py:1983-1984 and 2022-2024 (note text and the 'Single pass: rise …' readout)`

**Same root, also found by a skeptic.** Step view's legend and note announce a dashed 'single pass, causal (reference)' curve when the chain has no IIR stage and nothing dashed is drawn — Whole-chain toggle on with a chain of centred kernels only (running mean alone with the low-pass absent; or a Gaussian low-pass alone), or the toggle off with a single Running mean row selected. ChainStep.reference is None, the dashed PlotDataItem is set to [] (signal_conditioning.py:2008-2009), yet the legend still lists 'single pass, causal (reference)' (added once at :1055, never hidden) and the note still ends 'Dashed: […]


### S8-RD2 - Reconciled table's amber line keeps asking to refresh sources that were already re-run

> After a sample-rate declaration, the Reconciled table's amber line keeps saying "Refresh sources first … refresh the stale source results" after both source tables have been re-run at the new rate and are clean

*Results dock* · finder: minor · skeptics: minor/minor

**What the user sees.** The user declares 25 kHz on the -600 mV recording (Inspector ▸ File Info ▸ Sample rate / time unit…). All three tables go "⚠ Times stale — sample rate changed"; the user re-runs Schmitt (261 candidates, line clears) and CUSUM (418, line clears) as told, then selects the Reconciled table. Its line still reads "⚠ Times stale · unavailable: A Reconciled table cannot b…" with a greyed button labelled "Refresh sources first"; Details says "Switch to each source detector table, refresh the stale source results, then reconcile again." and "Refresh the source detector results, then reconcile again." Neither source is stale. The same line survives a project save and reopen.

**Why it matters.** The line's action and its reason name a step that no longer exists, so the user is sent back to two clean tables looking for something to refresh before the sentence's last clause (reconcile again) tells them what to actually do. _judge_staleness judges the rate stamp first and, for a reconciled table, always offers "Refresh sources first" with _RECONCILED_STALE_REFUSAL; the source-change judge that would say the honest thing here ("Source table changed … Review sources first", measured in the Schmitt-only case) never gets its turn while the rate stamp differs. Not a wrong number — the snapshot is correctly marked stale and its time columns badged — hence minor.

**Measured.** Tables before: ['Schmitt — 317 candidates', 'CUSUM — 370 candidates', 'Reconciled — 443 candidates']. Declare 25 kHz -> Reconciled: hidden=False kind='sample_rate' btn='Refresh sources first' enabled=False, painted '⚠ Times stale · unavailable: A Reconciled table cannot b…'. Schmitt re-run -> 'Schmitt — 261 candidates', line hidden=True; CUSUM re-run -> 'CUSUM — 418 candidates', line hidden=True. Reconciled (stamped fs 50000.0012, live 25000.0): hidden=False, kind='sample_rate', button 'Refresh sources first' enabled=False, refusal 'Refresh sources first is unavailable: A Reconciled table cannot be refreshed directly because its rows can come from different source detectors and preprocessing chains. Switch to each source detector table, refresh the stale source results, then reconcile again.', message '... Re-run detection under the declared rate. Refresh the source detector results, then reconcile again.'. Badges ('t_start_s','dwell_s','rise_s','fall_s','ecd_pA_s'). Project save -> reopen (p11): the same line on the reopened Reconciled table. Contrast (p03b D, no rate change): after a Schmitt re-run alone the reconciled line reads '⚠ Source table changed · Review sources first unavailable: This Reconciled table is a preserved snapshot ...'. Sites: nanoscope/gui/analysis_panel.py:5857-5880 (_judge_staleness rate branch: action_label 'Refresh sources first' if reconciled, message + ' Refresh the source detector results, then reconcile again.'); :2086-2090 (_stale_action_refusal -> _RECONCILED_STALE_REFUSAL for every non-source kind); :190-194 (_RECONCILED_STALE_REFUSAL text).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-RD2_reconciled_line_after_sources_refreshed.py`

**Re-run.** `S8-RD2_reconciled_line_after_sources_refreshed.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces as written (exit 7), and my own variations through the real widgets reproduce it on the -600 mV recording at 25 kHz and on the -700 mV recording at 20 kHz: after both source tables are re-run at the declared rate and their own lines clear, the Reconciled table's line still carries the greyed button "Refresh sources first" and the refusal "Switch to each source detector table, refresh the stale source results, then reconcile again." while nothing is left to refresh; the source-change judge, asked directly in that state, would say the honest thing ("predates changes to its source results (changed: CUSUM, Schmitt). Review or restore those source tables, then reconcile again."). […]

**Skeptic 2** (confirmed, minor). Reproduced with the finder's script (exit 7) and with my own variation through the real widgets. After declaring 25 kHz and re-running Schmitt (261) and CUSUM (418) so both source lines are hidden, the Reconciled table's line still paints '⚠ Times stale · unavailable: A Reconciled table cannot b…' with a disabled 'Refresh sources first' button, and both Details sentences tell the user to refresh stale source results that no longer exist. At that same instant panel._reconciliation_source_staleness(res) already returns the honest sentence ('… predates changes to its source results (changed: CUSUM, Schmitt). Review or restore those source tables, then reconcile again.'), so the judge's rate-first ordering is what hides it. […]

**Sites.** `nanoscope/gui/analysis_panel.py:5857-5880 (_judge_staleness, rate branch)`; `nanoscope/gui/analysis_panel.py:2086-2090 (_stale_action_refusal, reconciled branch)`; `nanoscope/gui/analysis_panel.py:190-194 (_RECONCILED_STALE_REFUSAL)`


### S8-EI1 - Event inspector window narrower than ~800 px: status-bar widgets overlap

> In the Event inspector's own window narrower than ~800 px the status bar's widgets overlap: the project label and the Save Project button are painted over the curation progress line, and "First unreviewed" is cut to "Fir…wed"

*Event inspector docking, Window menu, curation speed* · finder: minor · skeptics: minor/minor

**What the user sees.** The user shrinks the inspector window below ~800 px wide — the width the new five-row toolbar explicitly invites ("five short rows in the docked column (or a narrow window)", USER_GUIDE §9). The toolbar re-flows cleanly, the waveform keeps its 180 px, but the status bar at the bottom turns into a pile-up: at 600 px the line reads "First ...wed | Reviewed 0 / 3 | Project: Unsaved changes ... | Save Project", with "Project: Unsaved changes …" drawn straight over the progress numbers; at 452 px the progress line shows 75 of its 330 px, "First unreviewed" is "Fir…wed", and the Save Project button's menu arrow is cut at the right edge. Nothing is elided — the widgets are simply stacked on top of each other. Docked, where the project label is hidden, the same bar is clean.

**Why it matters.** The progress line ("Reviewed 312 / 870 · accepted 801 · labels: …") is the one place a curation pass reports where it stands, and the guide promises exactly this bar shares its width by content ("when both are long, the file name gives way first and the save state stays readable", USER_GUIDE line ~1385). In a narrow own window the numbers are hidden under the label that is supposed to give way first, so the user reads a truncated count or none, and the Save control is partly off-screen. The dock host got the fix (the label is hidden and the floor drops to 150 px); the equally advertised narrow own window did not.

**Measured.** Own window with Details shown, 317-event -600 mV detection, 13 pt system font, 1512x900 offscreen screen. Progress line minimum 330 px (_PROGRESS_MIN_W; 250 px without a palette), project label minimum/maximum 233 px, Save button 126 px, First-unreviewed 140 px hint. At 452 px: progress rect x=83 w=330 (visible 75 px), project label at x=164 w=233 and Save button at x=326 w=126 both intersect it; First-unreviewed squeezed to 75 px ("Fir…wed"). At 520 px: progress visible 92 px, label overlapping. At 600 px: visible 112 px, label at x=238 overlapping. At 700 px: visible 181 px, label at x=335 overlapping. At 800 px and above: no overlap, no cut. Screenshots of the bar at every width are written by the repro (statusbar_<w>.png) and in work/p03_details_status/out/p03_statusbar_452/520/600/700.png in the scratch dir. Sites: event_inspector.py:1758-1759 (addPermanentWidget label with stretch 0, then the button), :1784 (insertPermanentWidget(0, curation_status, 1)), :2099-2117 _share_status_bar (docked branch hides the label and lowers the floor to 150; the window branch keeps floor 330 / label 233 whatever the bar's width), :1059-1069 _status_share, :146 _PROGRESS_MIN_W = 330; the claims docs/USER_GUIDE.md:1309-1310 and ~1384-1388.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-EI1_statusbar_overlap.py`

**Re-run.** `S8-EI1_statusbar_overlap.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced independently through the real Event inspector, offscreen at the Mac's 13 pt system font and also at the offscreen default font. In the own window narrower than 800 px (window floor is exactly 452 px at 13 pt, so every width from 452 to 799 is reachable by a user drag) the status bar is over-committed: its layout minimum is 769 px with the "Unsaved changes" label (665 px "Saved", 706 px "No project file"), but that minimum never binds the window's minimum. The bar gives the curationStatus container less than its minimum (156 px vs 396 at 452 px; 230 vs 396 at 600 px; […]

**Skeptic 2** (confirmed, minor). Reproduced through the real EventInspectorWindow, offscreen, with the owner's -600 mV ABF (317 events). In an own window narrower than ~800 px (13 pt system font; ~700 px at the default 9 pt font) the status bar's box layout squeezes the `curationStatus` container below its content minimum, so the progress line keeps its 247-250 px inside a 156-327 px container and is clipped at the container's edge with no ellipsis, "First unreviewed" is squeezed from 140 to 75 px and paints "Fir…wed", and the 233 px project label plus the 126 px Save button are laid beside the cut line. […]

**Sites.** `nanoscope/gui/event_inspector.py:1758-1759 (_install_project_save_controls: addPermanentWidget label/button)`; `nanoscope/gui/event_inspector.py:1784 (_install_curation_status: insertPermanentWidget)`; `nanoscope/gui/event_inspector.py:2099-2117 (_share_status_bar)`; `nanoscope/gui/event_inspector.py:1059-1069 (_status_share)`; `nanoscope/gui/event_inspector.py:146 (_PROGRESS_MIN_W = 330)`; `docs/USER_GUIDE.md:1309-1310 and ~1384-1388 (the narrow-window and the sharing claims)`


### S8-EI2 - Window menu entries do nothing for a minimized gallery, Figure Library or Signal Conditioning window

> Window ▸ Event gallery, Window ▸ Figure Library and Window ▸ Signal Conditioning (and ⌘G, ⇧⌘F, the inspector's Gallery… button) do nothing for a minimized window, while the same menu's Event inspector, Analysis Explorer and Guided first analysis entries bring one back

*Event inspector docking, Window menu, curation speed* · finder: minor · skeptics: minor/minor

**What the user sees.** The user minimizes the Event gallery to the Dock, later opens the Window menu: "Event gallery" carries a check mark (it counts as on screen), the user chooses it — and nothing happens; the gallery stays in the Dock. Same for Figure Library and Signal Conditioning, for Analysis ▸ Event gallery… (⌘G), Analysis ▸ Signal conditioning… (⇧⌘F) and for the inspector's own Gallery… button. Choosing "Event inspector", "Analysis Explorer" or "Guided first analysis" for a minimized window restores it, so the menu teaches that choosing an entry brings the window back, and then breaks that for three of its six windows.

**Why it matters.** The Window menu's whole purpose (USER_GUIDE §2: "choosing one opens it or brings it forward through the same door as its own menu item") is to get a window back; a checked entry that does nothing is a menu item that does nothing. The app already knows show() is not enough for a minimized window — _present_event_inspector, _show_analysis_explorer, _bring_main_to_front, _bring_child_to_front and _bring_all_to_front all call showNormal() first — but the three openers the menu reuses for the gallery, the Figure Library and Signal Conditioning never got that line, and _on_screen() deliberately counts a minimized window as on screen ("which the user CAN restore without passing through an opener", main_window.py:539-551), so the check mark is drawn and the opener returns early into show()/raise_()/activateWindow() with the WindowMinimized state intact.

**Measured.** Offscreen, read through Qt's own window state after each entry (isMinimized()). showMinimized() on each window, then its Window-menu entry (checked=True for all six): Event inspector → restored; Event gallery → STILL MINIMIZED; Figure Library → STILL MINIMIZED; Signal Conditioning → STILL MINIMIZED; Analysis Explorer → restored; Guided first analysis → restored. inspector Gallery… button on a minimized gallery → still minimized; Analysis ▸ Event gallery… (⌘G) → still minimized; Analysis ▸ Signal conditioning… → still minimized; Analysis ▸ Event inspector… (⌘I) on a minimized inspector → restored. Caveat stated honestly: the measurement is Qt's window state on the offscreen platform; on macOS show()/orderFront on a miniaturized NSWindow likewise leaves it in the Dock, which is why the other five openers in this file call showNormal(). Sites: main_window.py:7415-7417 (_open_event_gallery: show/raise_/activateWindow, no showNormal), :4701 (_open_figure_library: show), :7492-7494 (_open_signal_conditioning: show/raise_/activateWindow), versus :6549-6550 (_present_event_inspector showNormal), :7002-7003 (_show_analysis_explorer), :6869-6870 (_bring_child_to_front), :7024-7025 (_bring_all_to_front); :6958-6990 _sync_window_menu checks with _on_screen; :539-551 _on_screen.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-EI2_minimized_windows.py`

**Re-run.** `S8-EI2_minimized_windows.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced at a53bf94 with the finder's repro and again on a different recording through every door into the three windows. With the Event gallery, Figure Library or Signal Conditioning window minimized, its Window-menu entry is drawn checked and choosing it (or any other door to the same opener: Analysis ▸ Event gallery… ⌘G, the inspector's Gallery… button, Analysis ▸ Figure Library…, the Analysis Explorer's figureLibraryRequested, Analysis ▸ Signal conditioning… ⇧⌘F, the detection panel's openSignalConditioningRequested, the Filter stage chip) leaves isMinimized() True and prints nothing on the status bar; […]

**Skeptic 2** (confirmed, minor). The finder's repro reproduces exactly as written (exit 7), and my own variations through the real widgets extend it: Analysis ▸ Signal conditioning… (⇧⌘F) and Analysis ▸ Figure Library… also leave a minimized window minimized, while Window ▸ Event inspector / Analysis Explorer / Guided first analysis restore one. The offscreen caveat does not save it: QWidget::show() is a no-op on a window that is visible-but-minimized by design (Qt's own source comments "Don't call showNormal() as not to clobber Qt::Window(Max/Min)imized"), raise_()/activateWindow() map to orderFront/makeKeyWindow on Cocoa, which do not deminiaturize, and the five other openers in this very file call showNormal() first precisely for that reason — so the shipped macOS […]

**Sites.** `nanoscope/gui/main_window.py:7415-7417 (_open_event_gallery)`; `nanoscope/gui/main_window.py:4693-4701 (_open_figure_library)`; `nanoscope/gui/main_window.py:7476-7494 (_open_signal_conditioning)`; `nanoscope/gui/main_window.py:6549-6550, 7002-7003, 6869-6870, 7024-7025 (the openers that do call showNormal)`; `nanoscope/gui/main_window.py:539-551 (_on_screen counts a minimized window as on screen)`; `nanoscope/gui/main_window.py:6895-6935 (_populate_window_menu wires the entries to those openers)`; `docs/USER_GUIDE.md:377-381 (the 'brings it forward' claim)`


### S8-EI3 - Reset Window Layout keeps an open child window's size and re-remembers it on close

> Window ▸ Reset Window Layout… keeps an OPEN child window's dragged size and re-remembers it at the window's next close, while a CLOSED window goes back to its default

*Event inspector docking, Window menu, curation speed* · also raised as S8-GEO3, S8-SC5 · finder: minor · skeptics: minor/minor

**What the user sees.** With the Event gallery open and dragged down to 862x420 the user chooses Window ▸ Reset Window Layout…, reads "Put every panel and window back where it starts? … every remembered window size and place is forgotten", presses Reset. The panels go back, the inspector returns to its own window at 1100x860 — and the gallery is only re-centred, still 862x420. The user closes it; that close writes windows/event_gallery/geometry straight back with the dragged size, so every later open (this session or the next) is 862x420 again, as if the reset had never happened. Had the gallery been closed at the time of the reset it would reopen at its 940x700 default.

**Why it matters.** This is a state surviving a layout reset that the dialog and the guide ("forgets remembered window sizes", USER_GUIDE line 388) say it does not survive, and it is inconsistent inside one gesture: the inspector is special-cased back to its default, a hidden window is resized to nanoscopeDefaultSize (the 61cdea3 fix), but a visible gallery / Signal Conditioning / Figure Library window keeps whatever size it was dragged to and the next Hide event re-remembers it. A user who resets precisely because a window ended up at a bad size has to know to close it first. (Overlaps the GEO and SC areas' repros S8-GEO3 / S8-SC5, found independently; the coordinator may merge.)

**Measured.** 1512x900 offscreen screen. Gallery default (nanoscopeDefaultSize) 940x700, opened at 940x700; dragged to (42,42,862,420). _reset_window_layout with the gallery OPEN: geometry (327,216,862,420), size == default False, settings key windows/event_gallery/geometry removed (False). Gallery closed afterwards: key present again (True, the Hide path's _remember_child_geometry). Reopened: (327,216,862,420). Control: gallery closed at 700x420 before the reset → key removed → reopened at (288,76,940,700) == default. Also seen in P10 with Signal Conditioning open at 1000x700: after the reset (258,76,1000,700), size kept; the quit afterwards saved windows/event_gallery/geometry. Sites: main_window.py:7047-7091 _reset_window_layout (visible branch :7080-7082 only re-places; hidden branch :7083-7089 resizes to default), :6770-6800 _place_child_window (`if key == _ROLE_INSPECTOR: window.resize(*_WINDOW_DEFAULT_SIZE)` at :6781 — the one child window the reset resizes), :6855-6864 _remember_child_geometry via :6859 (Hide), :7034-7044 _confirm_reset_window_layout text.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-EI3_reset_keeps_open_window_size.py`; `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-GEO3_reset_layout_keeps_open_window_sizes.py`; `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-SC5_reset_layout_keeps_open_window_size.py`

**Re-run.** `S8-EI3_reset_keeps_open_window_size.py` → reproduced (exit 7); `S8-GEO3_reset_layout_keeps_open_window_sizes.py` → reproduced (exit 7); `S8-SC5_reset_layout_keeps_open_window_size.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced with the finder's repro (exit 7) and with my own variation driven through the real Window-menu action (_populate_window_menu() then _act_reset_window_layout.trigger()) on the -600 mV ABF with a 317-event detection. With the Event gallery, Signal Conditioning and Figure Library all OPEN at dragged sizes, the reset removed every geometry key but only re-centred the three windows at their dragged sizes (gallery 862x420 vs default 940x700, SC 1000x600 vs 760x560, Figure Library 791x413 vs 1120x740); the inspector, open in its own window at 640x661, was the one window put back to its 1100x860 default. […]

**Skeptic 2** (confirmed, minor). Reproduced through the real Window ▸ Reset Window Layout… QAction (menu populated via aboutToShow, confirm seam returning True), not just the method. With the Event gallery, Figure Library, Signal Conditioning and the own-window Event inspector all open and dragged to non-default sizes, the reset re-centred the first three but left their dragged sizes (gallery 862x420 vs default 940x700, Figure Library 791x413 vs 1120x740, Signal Conditioning 1000x700 vs 760x560); only the inspector went back to its 1100x860 default. […]

**Sites.** `nanoscope/gui/main_window.py:7047-7091 (_reset_window_layout; visible vs hidden branches at 7080-7089)`; `nanoscope/gui/main_window.py:6781 (_place_child_window resizes only the inspector)`; `nanoscope/gui/main_window.py:6855-6864 and 6859 (_remember_child_geometry on Hide re-remembers the kept size)`; `nanoscope/gui/main_window.py:7034-7044 (the dialog's sentence)`; `docs/USER_GUIDE.md:386-388 ('forgets remembered window sizes')`


### S8-EI4 - Inspector toolbar does not keep its button order when it re-flows, contrary to the docs

> The Event inspector's toolbar does not keep its buttons' order when it re-flows from two rows to five, although the CHANGELOG and USER_GUIDE §9 say "the buttons keep their order"

*Event inspector docking, Window menu, curation speed* · finder: minor · skeptics: minor/minor

**What the user sees.** In a wide window the second row ends with Merge with next and the first row ends with Dock beside trace; docked (or below ~1050 px wide) Merge with next is the 9th control (row 2), Dock beside trace is the very last (row 5), Gallery… and Show in trace swap places, and Advance after decision moves from before Gallery… to after Reject all. The guide the user reads to learn the docked layout says the order is kept.

**Why it matters.** A doc claim that is false in the one sentence that describes the re-flow; a user who learned the strip in a window looks for Merge with next at the end and finds Accept all / Reject all / Advance there instead. The regrouping itself is defensible (per-event gestures together), so the fix is either the sentence or the narrow spec — but one of them must change.

**Measured.** From the window's own two arrangement specs (the lists _arrange_toolbar lays out), flattened: two rows = [◀, Event N, of N, ▶, ⟵, ⟶, context, Accept, Reject, Accepted, Revert mask, Dock beside trace, Filter:, filter combo, Advance after decision, Gallery…, Show in trace, Accept all, Reject all, Merge with next]; five rows = [◀, Event N, of N, ▶, Accept, Reject, Accepted, Revert mask, Merge with next, ⟵, ⟶, context, Show in trace, Gallery…, Filter:, filter combo, Accept all, Reject all, Advance after decision, Dock beside trace]. 13 of 20 controls change position: Merge with next 20→9, Dock beside trace 12→20, Advance 15→19, Gallery… 16→14, Show in trace 17→13, Accept/Reject/Accepted/Revert 8-11→5-8. The real re-flow was driven too (1100 px → _toolbar_is_wide True; 600 px → False). Sites: event_inspector.py:1475-1492 (_toolbar_wide / _toolbar_narrow specs), :2646-2668 (_arrange_toolbar), :2693-2699 (_sync_toolbar: need 1035 px for two rows); claims CHANGELOG.md:52-53 and docs/USER_GUIDE.md:1309-1310.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-EI4_toolbar_order_claim.py`

**Re-run.** `S8-EI4_toolbar_order_claim.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced through the real widgets, not just the specs. USER_GUIDE §9 (docs/USER_GUIDE.md:1310) says the toolbar "buttons keep their order" when it re-flows to five rows; the rendered toolbar (each control's real row layout and x position) shows a different reading order in every narrow state I drove: own window at 600 px, docked beside the trace via the real 'Dock beside trace' button, and after 'Own window' + re-dock. 16 of 20 controls change rank; the relative order is not even a subsequence (Merge with next last → 9th on row 2; Dock beside trace row 1 → row 5; Gallery…/Show in trace swap; Advance after decision moves behind Reject all; Accept/Reject/Accepted/Revert move ahead of the pan buttons and context combo). […]

**Skeptic 2** (confirmed, minor). Reproduced through the real widgets, not just the spec lists: the on-screen geometry of the 20 toolbar controls (mapTo(window), rows by y, order by x) in a 1100 px own window (two rows) versus a 600 px own window and versus the inspector actually docked beside the trace via set_host("dock") (five rows). The two arrangements order the controls differently, and the docked strip equals the narrow-window strip. Wide: row 1 ends with 'Dock beside trace', row 2 is Filter / combo / Advance after decision / Gallery… / Show in trace / Accept all / Reject all / Merge with next. Narrow/docked: row 1 ◀ Event of ▶ Accept Reject; row 2 Accepted / Revert mask / Merge with next; row 3 ⟵ ⟶ context / Show in trace / Gallery…; […]

**Sites.** `nanoscope/gui/event_inspector.py:1475-1492 (_toolbar_wide / _toolbar_narrow)`; `nanoscope/gui/event_inspector.py:2646-2668 (_arrange_toolbar)`; `docs/USER_GUIDE.md:1309-1310 ('the buttons keep their order')`; `CHANGELOG.md:52-53`


### S8-EI5 - With Advance off, A / R leave the Event spinner on the next event while the readout says 'filtered out'

> Under the Unreviewed filter (and any filter a decision moves the event out of) with Advance after decision off — the default — A / R leave the toolbar's "Event N of M" spinner at a position that now names the NEXT event, while the readout under it says the event on screen is "filtered out"

*Event inspector docking, Window menu, curation speed* · finder: minor · skeptics: minor/minor

**What the user sees.** Filter = Unreviewed, Advance off (the shipped default), the 6th unreviewed event on screen ("Event 6 of 317"). The user presses A. The event stays on screen (by design: the decided event does not move away), the count becomes "of 316" — and the spinner still says "Event 6". Position 6 of the filter is now a different event: typing the 6 the user sees + Return jumps to event 6, → from there goes to event 7 "7 of 316". The readout beneath says "Event 5 [filtered out of Unreviewed (316 shown)]" — so within one window the toolbar names position 6 and the readout says the event has no position. Under Unreviewed every decision takes the event out of the filter, so this is the state after every key press of a pass with Advance off.

**Why it matters.** A silently stale display in the control that is the only index into the filter: the number shown belongs to another event. It is small per gesture, but it is the common case of the new curation pass (Unreviewed + the default Advance off), and the two surfaces of one window disagree. _sync_controls deliberately leaves the spinner alone when the current event is no longer a member (`if pos is not None: setValue`), whereas _refresh_readout already has the honest wording; the spinner could say the same (or show the position the next ▶ will land on).

**Measured.** 317-event -600 mV detection, Unreviewed filter, Advance off. Before: event 5 shown as 'Event 6 of 317'. decide_accept(True) (the A key): on screen event 5; toolbar 'Event 6 of 316'; readout first line 'Event 5 [filtered out of Unreviewed (316 shown)]'; _subset_ids()[5] == 6 (position 6 is now event 6). Typing '6' + Return into the spinner → event 6. next_event() → event 7 'Event 7 of 316'. The same shape under Accepted only + R (pre-existing); under Unreviewed it follows every decision. Sites: event_inspector.py:4383-4402 _sync_controls (:4402 `self._index_spin.setValue(pos + 1)` only when pos is not None), :4059-4070 _refresh_readout (:4070 'filtered out of …'), :2939-2960 set_accepted_current ('Does NOT navigate').

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-EI5_index_spinner_stale.py`

**Re-run.** `S8-EI5_index_spinner_stale.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced with the finder's repro (exit 7) and with real key presses through the shown inspector. Under Unreviewed with Advance off (the shipped default), A and R leave the "Event N of M" spin box at a value that is no longer the position of the event on screen, while the readout under it says "filtered out of Unreviewed". The stale number is a live index into the filter: the spin box's own ▲ arrow then goes to position N+1 and skips the event that the ▶ button (same gesture, "next") lands on — measured ▶ -> event 6, spinner ▲ -> event 7 from the same state. At the last position the range shrink clamps the value one down, so ▼ skips the event ◀ reaches. […]

**Skeptic 2** (confirmed, minor). Reproduced through the real A/R keys, not only the finder's decide_accept() call, and in all three hosts/states. Under Unreviewed with Advance off (the shipped default) a decision leaves the toolbar spinner at its old value, which after the subset shrinks is the 1-based position of the NEXT member: "Event 6 of 316" over event 5, while the readout says "Event 5 [filtered out of Unreviewed (316 shown)]" and the plot title says "Event 5". Two qualifications that keep it MINOR and that the finder's why-it-matters overstates: (1) the stale number is exactly where ▶ lands and where a retyped number lands (measured: ▶ from the stale state -> event 7 when the spinner named 7; […]

**Sites.** `nanoscope/gui/event_inspector.py:4383-4402 (_sync_controls; the spinner left stale at 4402)`; `nanoscope/gui/event_inspector.py:4059-4070 (_refresh_readout 'filtered out of')`; `nanoscope/gui/event_inspector.py:2939-3008 (set_accepted_current, no navigation)`

**Same root, also found by a skeptic.** After deciding the LAST event of Unreviewed (Advance off), ▶ stays enabled although no later member exists; _step then snaps to members[-1], i.e. 'next' lands on an EARLIER event (the same one ◀ reaches) — Filter Unreviewed, Advance off, on the last unreviewed event ('Event 315 of 315'), press A. The ▶ button is enabled; clicking it goes to the previous unreviewed event (event 315), the same target as ◀.

**Same root, also found by a skeptic.** ◀ on a just-decided FIRST member of the filter is enabled and moves FORWARD; ▶ on a just-decided LAST member is enabled and moves BACKWARD (the _step opposite-end fallback) — Unreviewed, Advance off. Show the first unreviewed event (event 0, 'Event 1 of 314', ◀ disabled). Press A: the event leaves the filter, ◀ becomes enabled; press ◀ (or the ← key) and the window moves to event 1 at position 1 — a LATER event, from a button whose tooltip says 'Previous event in the current filter (←)'. Mirror image at the end: show the last member (event 316, 'Event 313 of 313', ▶ disabled), press A, ▶ becomes enabled and lands on event 315, an EARLIER event. Borderline: […]


### S8-GEO2 - Child windows left on an external monitor reopen on the laptop screen at every launch

> With a laptop screen and an external monitor both connected, the Event gallery, Event inspector and Signal Conditioning windows left on the external monitor reopen on the main window's (laptop) screen at every launch

*Geometry, layout reset, quit, hand-add* · finder: minor · skeptics: minor/minor

**What the user sees.** The user keeps the main window on the laptop and the Event gallery (2000,300 1200x900), the Event inspector (2200,200 1300x1000) and Signal Conditioning (2400,400 1100x1200) on the 3840x2160 external monitor, and quits. The next launch, with the same two displays, opens all three on the laptop screen over the main window — gallery at (310,80), inspector at (210,2) 1300x978, Signal Conditioning at (412,2) 1100x982 — with no line on the info bar or status bar. The user drags them back to the big monitor every morning.

**Why it matters.** This is the setup 61cdea3 was written for ('a laptop and an external monitor'), and its claims are false there: 'the child windows and Signal Conditioning take their saved size and place, brought onto the current screen', the CHANGELOG's 'Child windows now reopen at their remembered size and place … brought back onto the current screen' and the user guide's 'Child windows reopen where you last left them … brought back onto the screen you have now'. Qt's own restoreGeometry had put them right (screen index and width match); the move is the app's own fit. Mechanism: `_fit_to_available_screen` asks `window.screen()` of a never-shown child window, and Qt answers with the PARENT's screen (QWidget::screen → associatedScreen → windowHandle(Closest) = the main window's QWindow), so the restored rect is fitted into the main window's screen. Signal Conditioning is clamped to `self._available_screen_geometry()` — the main window's screen — by construction, so it moves even when the fit would not.

**Measured.** Two offscreen screens: 'laptop' (0,0,1512,982) and 'external' (1512,-600,3840,2160). Save: main (10,30,1400,850) screenAt=laptop; gallery (2000,300,1200,900), inspector (2200,200,1300,1000), SC (2400,400,1100,1200), all screenAt(center)=external. Relaunch on the same two screens: gallery (310,80,1200,900), inspector (210,2,1300,978), SC (412,2,1100,982), all screenAt=laptop; main unchanged; info bar '' and no 'could not be used' line. Mechanism: a never-shown child window at (2000,300,800,600) answers `screen()`='laptop' while `screenAt(center)`='external' (and 'external' once shown). The earlier variant with children straddling the screen boundary gave the same three moves. Sites: nanoscope/gui/main_window.py:6773 (`self._fit_to_available_screen(window)` after a successful restore in `_place_child_window`), :6803 (`screen = window.screen() or self.screen() or primaryScreen()`), :7524-7530 (`avail = self._available_screen_geometry()` … `ws.setGeometry(clamp_rect(ws.geometry(), avail))`), :7541 (`_available_screen_geometry` = the main window's screen); docs/USER_GUIDE.md:382-383.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-GEO2_children_pulled_onto_main_screen.py`

**Re-run.** `S8-GEO2_children_pulled_onto_main_screen.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces exactly as written (exit 7), and my own four variations through the real MainWindow/_open_event_gallery/_open_event_inspector/_open_signal_conditioning paths agree: with both displays still present, every child window left on the OTHER display from the main window reopens on the main window's display. It is not an offscreen-plugin artefact of primaryScreen: in the mirror case (main window on the external, children on the laptop) a never-shown child answers screen()='external' while screenAt(center)='laptop' and primary='external'; in the finder's layout it answers 'laptop' with primary='external' — i.e. […]

**Skeptic 2** (confirmed, minor). Reproduced with the finder's repro (exit 7) and with four variations of my own through the real widgets: every two-screen layout (external at y=-600, at y=0, to the left, and the mirrored case with the main window on the external and the children on the laptop) pulls the Event gallery, Event inspector and Signal Conditioning onto the MAIN WINDOW's screen at the next launch, while the one-screen control keeps all three rects exactly. The mechanism the finder named is what I measured: Qt's own restoreGeometry puts the never-shown gallery back at its saved rect (2000,300,1200,900) on the external, but because the child is parented to the main window, QWidget.screen() answers the main window's screen ('laptop'; […]

**Sites.** `nanoscope/gui/main_window.py:6773`; `nanoscope/gui/main_window.py:6803`; `nanoscope/gui/main_window.py:7524`; `nanoscope/gui/main_window.py:7530`; `nanoscope/gui/main_window.py:7541`; `docs/USER_GUIDE.md:382`


### S8-GEO4 - A floating Results, File manager or Inspector dock comes back hidden after a macOS-native quit

> A floating Results (or File manager / Inspector) dock still comes back hidden after a quit through macOS's own quit path (Dock icon ▸ Quit, logout, shutdown): QApplication.closeAllWindows() closes the dock before the main window on about half of the launches, and 3cc3f6b's fix covers only File ▸ Quit and the close box

*Geometry, layout reset, quit, hand-add* · finder: minor · skeptics: minor/minor

**What the user sees.** The user has pulled the Results panel out as a floating window (the event table, Summary/Fits/Quality, the Merge strip). The user quits from the Dock icon, or logs out. On some launches — it varies from one launch to the next — NanoScope comes back with no Results panel anywhere: the dock is restored floating AND hidden, and the table is only reachable through View ▸ Panels ▸ Results (⇧⌘R). File ▸ Quit, ⌘Q from the application menu and the close box are fine every time.

**Why it matters.** 3cc3f6b: 'A dock that is floating when you quit (Results, Inspector or File manager) now comes back floating and visible; quitting used to close it before the layout was saved.' The fix removed QDockWidget from the teardown's close protocol, which is the main window's own quit. The native quit reaches Qt as a QCloseEvent to the application (qcocoaapplicationdelegate.mm applicationShouldTerminate: → canQuit → QApplication::event → tryCloseAllWindows → tryCloseAllWidgetWindows), which closes every visible top-level widget in QApplication.topLevelWidgets() order — a QSet hashed on pointers — so on some launches the floating dock (a top-level window) is closed, i.e. hidden, before the main window, and the main window's closeEvent then saves the layout with it hidden: the defect 3cc3f6b fixed, through a door it did not cover, nondeterministically. Same for a floating Inspector or File manager dock and for the docked Event inspector's dock when floating.

**Measured.** Seven save/relaunch pairs on private settings (repro): one ordered `dock.close(); window.close()` and six `QApplication.closeAllWindows()` — the loop the native path runs. Close order ['QDockWidget','MainWindow'] in 4 of 7 → every one relaunched with `_results_dock.isHidden()=True, isFloating()=True`; order ['MainWindow','QDockWidget'] in 3 → dock visible and floating. An earlier sample of six closeAllWindows() quits (quit_scenarios.py): 2 of 6 lost the dock (toplevel order printed each time). File ▸ Quit (`self.close()`) with the same floating dock: visible on relaunch, 1/1, and the tests' LC-1 holds. Sites: nanoscope/gui/main_window.py:1287 (File ▸ Quit → `self.close()`, the only quit the app owns), :13716-13732 (`_stop_protocols` excludes QDockWidget — teardown only), :13848 (`closeEvent` → `_teardown` → `_save_settings` after whatever Qt already closed); no handler for QEvent.Close on the application / applicationShouldTerminate.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-GEO4_native_quit_hides_floating_dock.py`

**Re-run.** `S8-GEO4_native_quit_hides_floating_dock.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced myself, and through a door closer to the native quit than the finder's: instead of calling QApplication.closeAllWindows() by hand I sent the application Qt's own QEvent.Quit — the event the Cocoa plugin's applicationShouldTerminate: hands Qt 6.11 via QWindowSystemInterface::handleApplicationTermination (the 6.11 source does not call closeAllWindows or send a QCloseEvent itself; the window-closing is QGuiApplication::event(Quit) → tryCloseAllWidgetWindows over topLevelWidgets(), a list filtered from QWidgetPrivate::allWidgets). With a floating dock, that path closes the dock BEFORE the main window on a varying fraction of launches, and the main window's closeEvent → _teardown → _save_settings then records it hidden; […]

**Skeptic 2** (confirmed, minor). Reproduced by the finder's repro and by my own variations through the real widgets, and the "door" is established from the Qt 6.11.1 sources rather than inferred: on macOS a Dock-icon Quit / logout reaches applicationShouldTerminate: -> QWindowSystemInterface::handleApplicationTermination -> QGuiApplicationPrivate::processApplicationTermination, which sends QEvent::Quit to the QApplication; QApplication::event(QEvent::Quit) calls closeAllWindows() -> tryCloseAllWidgetWindows(), which iterates QApplication::topLevelWidgets() (built from the QWidgetSet allWidgets; Qt's own closeAllWindows docs say "in random order"). […]

**Sites.** `nanoscope/gui/main_window.py:1287`; `nanoscope/gui/main_window.py:13716`; `nanoscope/gui/main_window.py:13729`; `nanoscope/gui/main_window.py:13848`


### S8-TU3 - Frequency controls clamped by the misread rate keep their values and carry to the next recording

> The frequency controls clamped by the misread rate keep their clamped values after the rate is declared, and carry them to the next recording: low-pass 24.9 Hz, high-pass 24.99 Hz and a 24.9 Hz 'mains' notch on a 50 kHz recording

*The time-unit declaration through the real app* · finder: minor · skeptics: minor/minor

**What the user sees.** Open the ms CSV (read as 50 Hz): the Preprocess tab's low-pass cutoff drops from 10,000 to 24.9 Hz, the high-pass from 100 to 24.99 Hz and the notch from 60 to 24.9 Hz (the Nyquist clamp). Declare the column in milliseconds: the ranges go back up to 24,999.9 Hz but the values stay 24.9 / 24.99 / 24.9 Hz. Tick low-pass and notch and press Detect: the run uses a 24.9 Hz Bessel and a 24.9/49.8/74.7 Hz notch (the status line warns only that the notch needs 3.5 s to settle) and finds 8 events where the same slice opened in seconds finds 10 with the default 10 kHz. Then select the seconds CSV: it too now shows 24.9 / 24.99 / 24.9 Hz, where a fresh open shows 10,000 / 100 / 60 Hz.

**Why it matters.** The clamp was designed for the reverse case ('a stale 100 kHz cutoff carried over from a faster recording is pulled down to something legal') and is harmless there; here the values were pulled down by a reading the user has just declared wrong, and nothing restores them. A notch at 24.9 Hz is not a mains line, and a 24.9 Hz low-pass on a 50 kHz recording removes the events the user is looking for. The boxes do show the numbers, but a user who ticks the checkbox trusting the default is silently running a different chain — on this recording and on every recording selected after it.

**Measured.** S8-TU3 repro output: fresh open of the seconds CSV: 'low-pass 10000 Hz (max 24999.9) · high-pass 100 Hz · notch 60 Hz (max 24999.9)'. Open of the ms CSV read as 50 Hz: 'low-pass 24.9 Hz (max 24.9) · high-pass 24.99 Hz · notch 24.9 Hz (max 24.9)'. After declaring ms (50 kHz): 'low-pass 24.9 Hz (max 24999.9) · high-pass 24.99 Hz · notch 24.9 Hz (max 24999.9)'. Detect with low-pass + notch ticked ran {'filter_cutoff_hz': 24.9, 'notch_freq_hz': 24.9, 'notch_harmonics': 3}, 8 events (10 on the seconds CSV with the 10 kHz default; RuntimeWarning: notch lines [24.9, 49.8, 74.7] Hz). Then select the seconds CSV: 'low-pass 24.9 Hz · high-pass 24.99 Hz · notch 24.9 Hz'. Sites: nanoscope/gui/panels.py:3885-3899 (set_recording) and :3920-3950 (_apply_nyquist_ranges: 'Qt clamps an out-of-range CURRENT value as a side effect of setRange, which is the intent'); nanoscope/gui/analysis_panel.py:6148 (_on_sample_rate_changed → set_recording(fs), ranges only).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-TU3_nyquist_clamp_survives_declaration.py`

**Re-run.** `S8-TU3_nyquist_clamp_survives_declaration.py` → reproduced (exit 0) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced exactly as reported with the shipped repro, and in two variations of my own through the real widgets: (1) the owner's -600 mV ABF opened FIRST, then the ms CSV, the rate declared by SAMPLE RATE (50000 Hz) instead of by unit — the ABF's form is poisoned too (24.9 / 24.99 / 24.9 Hz), and a Detect on the owner's own 59.9 s recording with low-pass ticked ran filter_cutoff_hz=24.9 and found 240 events (265 at the 10 kHz default), with 238/240 flagged filter-limited; (2) ticking the stages after the declaration shows the clamped chain in the Preprocess readouts ('Active execution order: notch 24.9 Hz ×3 → low-pass bessel 24.9 Hz (order 8)', 'rise time: […]

**Skeptic 2** (confirmed, minor). Reproduced in the finder's repro and in my own variations through the real widgets. Opening a bare 'time,current' CSV in milliseconds is read as 50 Hz; DetectionPanel._apply_nyquist_ranges (panels.py:3920-3951) lets setRange clamp the CURRENT values of the three frequency spins to 24.9 / 24.99 / 24.9 Hz. Declaring the column in milliseconds (the real SampleRateDialog, Apply -> 'Declared: 50,000 Hz…') routes through AnalysisPanel._on_sample_rate_changed (analysis_panel.py:6133-6150), which deliberately only re-ranges ('NOT a re-selection, which would re-seed the form'): ranges go back to 1-24999.9 Hz, values stay 24.9 / 24.99 / 24.9. Because the form is global and a selection without a stored run never re-seeds (verified: […]

**Sites.** `nanoscope/gui/panels.py:3920-3950 (_apply_nyquist_ranges clamps the values, never restores them)`; `nanoscope/gui/panels.py:3885 (set_recording)`; `nanoscope/gui/analysis_panel.py:6148 (_on_sample_rate_changed pushes the new rate: ranges move, values do not)`


### S8-TU4 - One recording exported in ms and in s detects 9 vs 10 events

> One recording exported twice — time column in milliseconds (declared) and in seconds — detects 9 vs 10 events: the two text columns measure rates 1e-10 apart and the detector floors 0.5·fs and 1e-3·fs to whole samples (a detector-side sensitivity surfaced by the new reader; for triage)

*The time-unit declaration through the real app* · finder: minor · skeptics: minor/minor

**What the user sees.** The same 2 s slice of the -600 mV recording, bit-identical samples, written once with a ms time column (4 decimals) and once with a seconds column (7 decimals). The ms file, declared in milliseconds through the dialog, measures 49,999.999998908606 Hz and Detect finds 9 events; the seconds file measures 50,000.000001892884 Hz and finds 10 — one 14-sample tail (samples 76,872–76,886) is its own event under one rate and merged into its neighbour under the other, and seven of the shared rows differ in depth/baseline. The user sees two different event tables for one recording with no clue why. By contrast the HDF5 in ms and the refused-microsecond CSV, declared, match the seconds CSV to 4e-11 relative in every time column — the declaration arithmetic itself is exact.

**Why it matters.** A rate measured from a text column is only as precise as the column's decimals, and the baseline estimator converts seconds to samples with int(): int(0.5·fs) = 24,999 vs 25,000 and int(1e-3·fs) = 49 vs 50 move the noise RMS 4.1818 → 4.1955 pA and both thresholds, enough to split or merge a marginal event. Neither count is wrong, but a user who re-exports or declares a unit will get a different event set from the same samples and the app offers no explanation. Not caused by 132b42b (the kernel does the same on any two rates an ulp apart); reported because the territory's question — do dwell/t_start/ECD agree with the seconds re-export — is 'yes to 4e-11, except when the measured rate's last digits flip a floor'.

**Measured.** S8-TU4 repro output: samples identical: True. ms CSV declared: fs=49999.999998908606 events=9; s CSV: fs=50000.000001892884 events=10. Spans (ms): …(75933, 76887)…; spans (s): …(75933, 76871), (76872, 76886)…. noise_rms_pA 4.181768586730957 vs 4.195477931213379; threshold_in_pA 20.908842933654785 vs 20.977389656066894; threshold_out_pA 8.363537173461914 vs 8.390955862426758; baseline_median_pA -33.26416015625 vs -33.55693054199219. int(0.5*fs) = 24999 / 25000 / 25000 and int(1e-3*fs) = 49 / 50 / 50 for fs = 49999.999998908606 / 50000.0 / 50000.000001892884. The bare detect_events kernel on the SAME array: 9 events at 49999.999998908606, 10 at 50000.0. Scratch explore3: HDF5-ms declared vs seconds CSV max rel diff 3.78e-11 on t_start_s/dwell_s/ecd_pA_s (start_idx/end_idx exact); µs CSV declared vs seconds CSV 3.79e-11. Sites: nanoscope/analysis/baseline.py:2095, :3268, :3760 (int(dilate_ms·1e-3·fs)), :1942 (int(round(win·fs))), :1820/:1875 (int(window_s·fs/decim)); nanoscope/io/text.py (rate measured from the column's decimals).

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-TU4_two_exports_nine_vs_ten_events.py`

**Re-run.** `S8-TU4_two_exports_nine_vs_ten_events.py` → reproduced (exit 0) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces exactly as filed (ms CSV declared: fs=49999.999998908606, 9 events; s CSV: fs=50000.000001892884, 10 events; identical samples). My variations show it is general and larger than filed, not a 2 s-slice curiosity: on the owner's FULL -600 mV recording the native ABF detects 317 events and a plain seconds CSV of the same samples (8 decimals, t0=0 — sweep 7's own fixture format, nothing from 132b42b needed) detects 307, with File Info reading "50,000 Hz" for both; the -1000 mV recording moves 1201 -> 1181 the same way. The entire gap is one floor: […]

**Skeptic 2** (confirmed, minor). The finder's repro reproduces exactly as filed through the real widgets (ms CSV declared in ms: fs=49999.999998908606, 9 events; seconds CSV: fs=50000.000001892884, 10 events; identical samples; noise RMS 4.1818 vs 4.1955 pA; the same int(0.5·fs)=24999/25000 and int(1e-3·fs)=49/50 floors). The measured consequence follows: two event tables for one recording, and no surface lets the user see why — File Info prints "50,000 Hz · measured" for every variant, the dialog previews "Would read as 50,000 Hz" for both the unit declaration (49999.999998908606) and a Hz declaration (50000.0), and only the exported .meta.json carries the full-precision fs. […]

**Sites.** `nanoscope/analysis/baseline.py:2095 (iterations = int(dilate_ms * 1e-3 * fs): 49 vs 50)`; `nanoscope/analysis/baseline.py:3268 and :3760 (the same floor)`; `nanoscope/analysis/baseline.py:1820, :1875, :1942 (window_s * fs floored / rounded to samples)`; `nanoscope/io/text.py:_scan_impl (the rate is the column's own decimals: 49999.999998908606 from 4-decimal ms, 50000.000001892884 from 7-decimal s)`

**Same root, also found by a skeptic.** Stale guard is coarser than the detector: re-declaring 50,000 Hz over a unit declaration says 'detections already made on this recording are marked stale' but no stale line appears, and a re-run gives 10 events instead of the 9 still on screen — Open the bare ms CSV, declare 'The time column is in… milliseconds' (rate 49999.999998908606 Hz), Detect (9 events). Reopen Sample rate / time unit…, choose 'Declare the sample rate (Hz)' = 50000, Apply. The dialog prints 'Declared: 50,000 Hz. … detections already made on this recording are marked stale.' The Results dock shows no stale line (gui/analysis_panel.py _sample_rate_staleness returns None because |live − stamped| ≤ 1e-9·live), the 9-row table stays current, and pressing Detect under […]


### S8-XS3 - After a declaration the stale line's Re-run is named 'Merge events (4066 → 6)' in Undo and the status bar

> After a time-base declaration, the stale line's Re-run on an uncurated table is named 'Merge events (4066 → 6)' in Edit > Undo, the status bar and Redo

*Cross-surface interactions* · finder: minor · skeptics: minor/minor

**What the user sees.** The user detects on a CSV read at 50 Hz (4066 rows), declares the column is in milliseconds, and presses the amber line's Re-run; the detector re-runs under 50 kHz and lands 6 rows. Edit > Undo reads 'Undo Merge events (4066 → 6)'; pressing it says 'Undone: Merge events (4066 → 6).' and Redo offers 'Redo Merge events (4066 → 6)'. The user never merged. With three hand rejections on the same table the identical gesture is named 'Undo Re-run detection' and announced ('Detection replaced the previous table…').

**Why it matters.** The Edit menu is the one place that tells the user what the last destructive step was, and here it names a gesture the user did not make for the most destructive button in the app. The sample-rate re-run is new (132b42b) and is the only re-run whose stored params are byte-identical to the previous result's while the row count moves, so the history's own re-detection override never fires; it also means the replacement is announced on the status bar only when curation was at stake, since _on_step_recorded keys on the label. No data is lost (Undo restores the 4066 rows) — the sentence is false.

**Measured.** docs/sweep8/repros/S8-XS3_declaration_rerun_named_merge.py. No curation: 4066 rows / 4066 accepted at 50 Hz; declared ms -> 50,000 Hz, stale line '⚠ Times stale — sample rate changed'; Re-run -> 6 rows, dict(before.params) == dict(after.params) True; Edit > Undo 'Undo Merge events (4066 → 6)'; status messages since the click: []; after Undo 4066 rows, status 'Undone: Merge events (4066 → 6).', Redo 'Redo Merge events (4066 → 6)'. Three hand rejections: same path -> 'Undo Re-run detection', status 'Detection replaced the previous table (4066 events, 3 rejected by hand). Your curation is not carried over — Edit ▸ Undo puts it back.' Also seen in s05/s08 as 'Undo Merge events (52 → 0)' and '(6 → 0)'. Sites: core/history.py:375-377 describe_change ('fewer rows -> a merge'); core/history.py:884-905 _record (detection override = params moved outside baseline OR curation reset OR row count moved AND params differ — none true here); gui/analysis_panel.py:2910-2932 _on_rerun replays the result's stored params by contract; gui/main_window.py:13278-13308 _on_step_recorded announces only for 'Re-run detection'/'Detection'.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-XS3_declaration_rerun_named_merge.py`

**Re-run.** `S8-XS3_declaration_rerun_named_merge.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces exactly as written (exit 7), and it reproduces through the real Edit menu's aboutToShow (main_window.py:13364-13369 is the same _checkpoint_everything + _sync_undo_actions pair the repro's helper calls), on a second recording (-700 mV) and under the other declaration mode (a rate typed in Hz, not a time unit). The step is pushed at the moment the re-run lands (stepRecorded fires 'Merge events (4066 → 6)' with n_hand_rejected 0), so the false name is not an artefact of when the menu is opened. Mechanism as the finder says: […]

**Skeptic 2** (confirmed, minor). Reproduced through the real widgets, not only with the finder's shortcut. On the 2 s ms-column CSV (reads at 50 Hz, 4066 Schmitt rows, no curation), declaring the column as milliseconds and pressing the amber line's Re-run lands 6 rows with byte-identical stored params; the Edit menu's own aboutToShow (the path the user hits) then reads 'Undo Merge events (4066 → 6)', the status bar after the undo says 'Undone: Merge events (4066 → 6).' and Redo offers the same name. Nothing on the surface says a re-detection happened (status messages since the click: []; the stale line simply hides). The app does not disclose it anywhere, and the codebase's own stated invariant for this label (core/history.py describe_change: […]

**Sites.** `nanoscope/core/history.py:375`; `nanoscope/core/history.py:884`; `nanoscope/gui/analysis_panel.py:2910`; `nanoscope/gui/main_window.py:13278`

**Same root, also found by a skeptic.** Plain Detect button after a time-base declaration is also named 'Merge events (4066 → 6)' — Same fixture and state as XS3 but instead of the amber line's Re-run the user presses Detect with the form untouched after declaring ms (a detector whose resolved params are unchanged by the declaration): the table shrinks 4066 -> 6 and Edit > Undo reads 'Undo Merge events (4066 → 6)'. Widens XS3's scope from the stale line's button to any re-detection with identical stored params that shrinks an uncurated table; […]


### S8-XS4 - Study windows from the Incubation launcher are missing from the Window menu

> A Manual or Acid study window opened from the Incubation study launcher is missing from the new Window menu, unregistered and unremembered, while the launcher, the legacy Manual study shortcut and the Zn reference are listed

*Cross-surface interactions* · finder: minor · skeptics: minor/minor

**What the user sees.** Analysis > Incubation study…, create (or open) a study with the General profile and one with the Acid profile: two study windows are on screen ('XS general — Incubation study — NanoScope', 'Incubation study — Acid — NanoScope'). The Window menu lists 'Incubation study' (the launcher, checked) and nothing for either study window; with a study behind the main window there is no menu entry to bring it back. Reset Window Layout… leaves them where they are, their size and place are not remembered between sessions (no windows/<key>/geometry), and Bring All to Front's own loop skips them. The guide says Window 'lists the windows NanoScope opens … with a check mark beside each one on screen', the CHANGELOG 'the other windows NanoScope has opened'.

**Why it matters.** The study windows are where the incubation workflow (a4e185a) is actually done, and they are the long-lived ones; the menu built to list 'every NanoScope window' knows the launcher that spawned them and not them. The registry is a fixed five-key list (_CHILD_WINDOW_KEYS) and the launcher creates its windows through its own _window_for/_remember without _adopt_child_window, so the claim is false for exactly the windows that matter.

**Measured.** docs/sweep8/repros/S8-XS4_study_windows_missing_from_window_menu.py. created 'general' -> ManualStudyWindow on screen=True; 'acid' -> AcidStudyWindow on screen=True. Window menu (after _sync_window_menu): Bring All to Front, [x] NanoScope, Event inspector, Event gallery, Analysis Explorer, Figure Library, Signal Conditioning, [x] Incubation study, Dock Event inspector…, Reset Window Layout… — no study entry. registered in win._child_windows: ManualStudyWindow False (geometry key None), AcidStudyWindow False (None). Bring All to Front raised ['IncubationStudyLauncher', 'AcidStudyWindow'] (the acid raise comes from the launcher's own refresh, the manual study window is not raised). Reset Window Layout: study windows stay at (302, 302, 900, 600). Settings after a quit-save: windows/event_inspector/host, windows/incubation_launcher/geometry only. By contrast (s04) _open_manual_study / _open_incubation_reference windows are listed ('Manual study', 'Original Zn reference recipe', checked), placed by Reset and written to windows/manual_study/geometry, windows/incubation_reference/geometry. Sites: gui/incubation_launcher.py:185-200 (_remember / _window_for: no _adopt_child_window); gui/main_window.py:419-429 (_CHILD_WINDOW_KEYS / _CHILD_WINDOW_LABELS fixed list), :5925-5935 (_open_incubation_study adopts only the launcher), :6937-6940 (menu entries for the fixed keys), :6735 _register_child_window; docs/USER_GUIDE.md:376-380; CHANGELOG.md Unreleased 'A Window menu' paragraph.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-XS4_study_windows_missing_from_window_menu.py`

**Re-run.** `S8-XS4_study_windows_missing_from_window_menu.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces at a53bf94 (exit 7) and my own variations through the real widgets agree: a ManualStudyWindow and an AcidStudyWindow created or opened from the Incubation study launcher (Analysis ▸ Incubation study…, the menu action itself, then create_study / open_study — the code the Create study… / Open selected study dialogs land in) are top-level child windows of the main window (isWindow True, parent is main) yet are never registered (_adopt_child_window is only called for the launcher; […]

**Skeptic 2** (confirmed, minor). Reproduced through the real widgets. After Analysis ▸ Incubation study… creates a General and an Acid study (ManualStudyWindow / AcidStudyWindow, both top-level children of MainWindow, both on screen), the Window menu lists only the launcher ('Incubation study', checked); neither study window is registered in win._child_windows, so Bring All to Front raises neither (my run: raised == [] for both; the finder's 'AcidStudyWindow' raise was a side effect of launcher refresh timing, not the menu), a quit-save writes no windows/<key>/geometry for them, and Reset Window Layout leaves them where they are. Meanwhile the legacy Manual study shortcut and the Zn reference window are listed, checked and remembered. […]

**Sites.** `nanoscope/gui/incubation_launcher.py:185`; `nanoscope/gui/incubation_launcher.py:193`; `nanoscope/gui/main_window.py:419`; `nanoscope/gui/main_window.py:5925`; `nanoscope/gui/main_window.py:6937`; `docs/USER_GUIDE.md:376`


### S8-DOC2 - USER_GUIDE says a new Event inspector may start at the first event; the app opens on the selected one

> USER_GUIDE §9 tells the user a new Event inspector 'can start at the first event regardless of an earlier table selection' and to 'reselect that event after opening it' — the app opens on the selected event, as the CHANGELOG says it now does

*Documentation against the live widgets* · finder: minor · skeptics: minor/minor

**What the user sees.** §9 reads: 'Open the Inspector before choosing the target event, or reselect that event after opening it. A new Inspector can start at the first event regardless of an earlier table selection. The Event spinner is a position in the filtered subset…'. In the app, selecting event 219 in the Results table and pressing Ctrl+I opens the inspector on event 219 (not event 0). The CHANGELOG's Fixed section says the opposite of the guide: 'Event inspector: opens on the event selected in the table (and, re-opened, where you were) instead of event 0'.

**Why it matters.** The guide prescribes a workaround for a defect this release fixed, so a user following the guide re-selects the event needlessly and distrusts the inspector's opening position; the two documents in the same tree contradict each other on the same behaviour. Measured through the real Ctrl+I action on the owner's −600 mV recording after a CUSUM detection.

**Measured.** 650 CUSUM events; panel.selected_event_id() == 219 after select_event(219); win._act_event_inspector.trigger() → EventInspectorWindow.current_event_id() == 219. docs/USER_GUIDE.md still carries both sentences (the repro greps for them). Sites: docs/USER_GUIDE.md §9 lines ~1310-1312; nanoscope/gui/main_window.py:6353-6372 _open_event_inspector ('selected = self._analysis_panel.selected_event_id(); if selected is not None: insp.show_event_quietly(selected)'); CHANGELOG.md Unreleased ▸ Fixed 'Event inspector: opens on the event selected in the table…'.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DOC2_guide_inspector_first_event.py`

**Re-run.** `S8-DOC2_guide_inspector_first_event.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The finder's repro reproduces as written (exit 7: 650 CUSUM events, select 219, Ctrl+I opens a NEW own-window inspector on 219, and docs/USER_GUIDE.md lines 1313-1314 still carry both sentences). My own variations through the real widgets could not produce the behaviour the guide describes in any host or state: a real QTableView mouse click on the row for event 300 then Ctrl+I opened on 300 (spin 301 of 650, filter All); docked via the real Window ▸ "Dock Event inspector beside the trace" action, dock closed, event 42 selected, Ctrl+I re-opened the docked inspector on 42; a fresh window with NO selection opened on event 0 (the only case the first event shows, and there is then no "earlier table selection" to lose); […]

**Skeptic 2** (confirmed, minor). The finder's repro reproduces (exit 7) and every variation I drove through the real widgets agrees: a NEW Event inspector opens on the event selected in the Results table in both hosts and via every route, and a re-opened one returns to the current selection. The only 'first event' start is after a re-run installs a new table (selection correctly cleared), which is not what the guide describes. docs/USER_GUIDE.md §9 (lines 1310-1312) still says 'Open the Inspector before choosing the target event, or reselect that event after opening it. A new Inspector can start at the first event regardless of an earlier table selection.' That sentence was written in 5d4dda8; […]

**Sites.** `docs/USER_GUIDE.md:1310-1312`; `nanoscope/gui/main_window.py:6353-6372`; `CHANGELOG.md (Unreleased ▸ Fixed, Event inspector bullet)`


### S8-DOC3 - 'Show in trace' / T does nothing once the event is already the table's selected row

> 'Show in trace' / T in the Event inspector and the gallery's 'Go to event in trace' do nothing once the event is already the table's selected row — which it always is after inspector navigation — so after panning the trace away the documented jump is dead

*Documentation against the live widgets* · finder: minor · skeptics: minor/minor

**What the user sees.** USER_GUIDE §9: 'Show in trace (T) centres the main trace on the event you are looking at — the same jump as right-click ▸ Go to event in trace on its tile in the gallery'; §13: 'T | Centre the main trace on the event on screen'; inventory.json (Show in trace): 'Centre the main trace on the event you are looking at … A pure VIEW move'. On the −600 mV recording after a CUSUM detection: open the inspector, go to event 300 (the trace zooms to 89.797–89.800 s and the table's selection becomes 300), pan the trace to 62.303–63.303 s, press the 'Show in trace' button — the view stays at 62.303–63.303 s. Press T — same. Call the gallery's slot — same. Only when a different event (310) is selected in the table does the same button jump to 300.

**Why it matters.** Navigating the inspector always makes its event the table's current row, so the one state in which a person reaches for Show in trace — 'I scrolled the trace away, take me back to this event' — is exactly the state in which the button, the T key and the gallery's context-menu item are no-ops, with no message. The jump rides on the Results table's currentChanged (AnalysisPanel.select_event → EventTable.select_event), which does not fire for the row that is already current; clicking that same row in the table does not re-zoom either, so the user has no direct way back to the selected event's span except selecting another event first.

**Measured.** After insp.show_event(300): visible range (89.79749, 89.79969) contains t=89.79804 s, table selection 300. After tv.zoom_x_to(1–2 s): range (62.30338, 63.30338). _trace_btn.click() → (62.30338, 63.30338); QTest T → (62.30338, 63.30338); win._goto_event_in_trace(300) → (62.30338, 63.30338). Control: panel.select_event(310), insp.show_event_quietly(300), pan away, button → (89.79749, 89.79969). Sites: nanoscope/gui/main_window.py:7616-7639 _goto_event_in_trace (self._analysis_panel.select_event(int(event_id)) then lower/raise only); nanoscope/gui/analysis_panel.py:5459-5464 select_event ('highlight the matching row, which drives the trace zoom + marker'); nanoscope/gui/panels.py:8453 EventTable.select_event / :8927 _on_current_changed; nanoscope/gui/event_inspector.py:2187 (T → _on_show_in_trace) and :1418 _trace_btn.

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DOC3_show_in_trace_noop.py`

**Re-run.** `S8-DOC3_show_in_trace_noop.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). Reproduced exactly as reported, on two recordings and through every real user path into the state. 'Show in trace' / T in the Event inspector and the gallery's 'Go to event in trace' route to MainWindow._goto_event_in_trace, whose only view move is AnalysisPanel.select_event -> EventTable.select_event -> _select_proxy_row, and the zoom rides on the table's currentChanged, which does not fire when the requested row is already current. Inspector navigation (Right key, Next button, show_event), a gallery tile click and a table row click all make the event the current row, so after the user pans the trace away the documented 'centre the main trace on this event' does nothing, silently. Nothing on the surface discloses it: […]

**Skeptic 2** (confirmed, minor). Reproduced through the real widgets in every documented form. With event 300 the table's current row (which inspector navigation always makes it), panning the trace to 62.303–63.303 s and then (a) QTest-clicking the inspector's "Show in trace" button, (b) pressing T, (c) picking the gallery's real context-menu item "Go to event in trace" on tile 300 (through _on_context_menu with only the blocking exec seam patched) each leave the view at 62.303–63.303 s and the status bar empty. The same holds with the inspector DOCKED beside the trace (new in 35096e3), where the slot's only other effect — lowering the inspector window and raising the main one — no longer exists, so the gesture is a complete no-op. […]

**Sites.** `nanoscope/gui/main_window.py:7616-7639`; `nanoscope/gui/analysis_panel.py:5459-5464`; `nanoscope/gui/panels.py:8453-8467, 8927`; `nanoscope/gui/event_inspector.py:1418, 2187`; `docs/USER_GUIDE.md §9 'Show in trace' bullet and §13 'T' row`; `docs/inventory.json 'Show in trace' entry`

**Same root, also found by a skeptic.** Gallery status line tells the user to 'Double-click the tile to bring the trace forward', but double-click opens the Event Inspector window on top and leaves the trace where it was (pre-territory sentence, unverified by a saved repro) — After a CUSUM detection on the -600 mV recording, open Analysis ▸ Event gallery…, pan the main trace to 62.303-63.303 s, click a tile, read the status bar, then double-click the tile as it says.


### S8-DOC4 - docs/inventory.json quotes a Response-tab note the app no longer prints

> docs/inventory.json quotes a Response-tab note the app no longer prints: 'the applied response … 7800 Hz (0.78×)' where the widget says 'the designed response … 7600 Hz (0.76×)' at the same nominal 10000 Hz

*Documentation against the live widgets* · finder: minor · skeptics: minor/minor

**What the user sees.** inventory.json, entry 'response note (line under the plot)' (Signal Conditioning ▸ Response tab): 'One sentence saying what the selected filter really does, e.g. "At the nominal 10000 Hz the applied response is -6.0 dB — zero-phase, so |H| is SQUARED. The true -3 dB corner is 7800 Hz (0.78× the number in the box)."'. The widget, with the 8-pole Bessel at 10000 Hz on the owner's 50 kHz recording, prints: 'Low-pass is ON and included in the execution chain. Low-pass: at the nominal 10000 Hz the designed response is -6.0 dB — zero-phase, so |H| is SQUARED. The true -3 dB corner is 7600 Hz (0.76× the number in the box).' At 5000 Hz it prints 3619 Hz (0.72×), which is the README's number.

**Why it matters.** The control reference is the document that is supposed to quote the widgets byte for byte; this quote names a different quantity ('applied' vs 'designed' — a distinction the window itself makes between the applied zero-phase response and the single-pass design) and a different corner (7800 vs 7600 Hz) for the same box value, so a reader checking the app against the reference would conclude the filter changed. Numbers only; the guide and README are consistent with the widget.

**Measured.** det.response_note.text() with filter_cutoff_hz=10000, order 8, bessel, fs 50 000 Hz: 'Low-pass is ON and included in the execution chain. Low-pass: at the nominal 10000 Hz the designed response is -6.0 dB — zero-phase, so |H| is SQUARED. The true -3 dB corner is 7600 Hz (0.76× the number in the box).'; at 5000 Hz: '… 3619 Hz (0.72× the number in the box).' inventory quote parsed from the entry by the repro. Sites: docs/inventory.json entry label 'response note (line under the plot)' (group 'Signal Conditioning ▸ Response tab'); nanoscope/gui/panels.py:5862-5870 (the sentence builder: 'zero-phase, so |H| is SQUARED' / ' The true -3 dB corner is {true_corner:.0f} Hz ').

**Repro.** `QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DOC4_inventory_response_note.py`

**Re-run.** `S8-DOC4_inventory_response_note.py` → reproduced (exit 7) — by the coordinator on a53bf94, under a write guard.

**Skeptic 1** (confirmed, minor). The inventory's quoted example sentence is not one the app prints and cannot be produced by any combination the app offers. Through the real Signal Conditioning Response tab I measured the note on three recordings (the -700 mV ABF, the March_25 ATP ABF, and the undeclared_700mV.csv, all ~50 kHz) with the 8-pole Bessel at a nominal 10000 Hz: every one prints 'the designed response is -6.0 dB ... The true -3 dB corner is 7600 Hz (0.76× the number in the box)'. A Qt-free sweep of filters.response over fs in {25k, 50k, 100k, 200k, 250k, 500k} × order {4, 8} × {bessel, butter} never yields 7800 Hz / 0.78× (the nearest is the 4-pole Bessel at 50 kHz: 7694 Hz, 0.77×). […]

**Skeptic 2** (confirmed, minor). Reproduced through the real widget and not disclosed anywhere. docs/inventory.json's entry 'response note (line under the plot)' quotes a sentence the app has not printed since 3ba94ab ('the applied response is … 7800 Hz (0.78×)'); the Signal Conditioning Response tab prints 'the designed response is -6.0 dB … The true -3 dB corner is 7600 Hz (0.76× the number in the box)' for the same 8-pole Bessel at 10000 Hz on the owner's 50 kHz recording. The app's number is the correct one (scipy direct: 7589 Hz, 0.759×), and no sample rate, order or kind makes the app print 7800 Hz / 0.78× at a nominal 10 kHz, so the inventory's figure is not an example from another rate; it is a leftover from the initial commit's wording. It is in the territory: […]

**Sites.** `docs/inventory.json ('response note (line under the plot)')`; `nanoscope/gui/panels.py:5862-5870`

**Same root, also found by a skeptic.** The Response-tab note labels the APPLIED (zero-phase, squared) figure 'the designed response', contradicting the tab's own legend vocabulary — Signal Conditioning ▸ Response tab, Low-pass 8-pole Bessel at 10000 Hz on any 50 kHz recording. The note reads 'at the nominal 10000 Hz the designed response is -6.0 dB — zero-phase, so |H| is SQUARED.' In the same tab the magnitude view is captioned '|H(f)| as applied (zero-phase IIR stages are squared)', the legend entries are 'applied (zero-phase)' vs the 'single-pass causal design', and the step view says 'Solid: as applied ... Dashed: the same single-pass design'. […]


### S8-DATA3 - Cmd+Z in the Event inspector stops at the 30-step ceiling without saying so

> Cmd+Z in the Event inspector walks back 30 steps and then does nothing, and says nothing; the sentence that names the ceiling exists only in the main window's Edit menu

*Lens: data safety and persistence* · **tier B** — found by the data-safety lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Cmd+Z in the Event inspector walks back 30 steps and then does nothing, and says nothing; the sentence that names the ceiling exists only in the main window's Edit menu. The CHANGELOG's "Faster curation" makes a long sprint the normal use of the inspector ("Reviewed 312 / 870", A / R with Advance after decision, Cmd+Z / Shift+Cmd+Z "on the main undo history"). core/history.py bounds one table to MAX_DEPTH = 30 steps and, since IF-03, discloses the loss -- in ``MainWindow._act_undo``'s text ("Undo -- this table's oldest history was dropped at the 30-step ceiling (20 steps lost)"). The Event inspector in its own window has no menu bar: its Cmd+Z emits ``undoRequested`` -> ``MainWindow._undo_events``, which at the ceiling returns without a word (``_undo_target()`` is None -> ``_sync_undo_actions()``). The main window's status bar keeps the previous "Undone: Curate event." and the inspector shows nothing, so press #31 and every later one looks like it worked while 20 decisions stay put. 50 decisions on a 10-event table (each A/R press flips one event), then 55 presses of the inspector's Cmd+Z.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DATA3_inspector_undo_past_the_ceiling_is_silent.py`

**Re-run.** `S8-DATA3_inspector_undo_past_the_ceiling_is_silent.py` → reproduced (exit 7, 21.3 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
history ceiling MAX_DEPTH = 30 steps per event table; 10 events
50 decisions made; Edit > Undo reads: 'Undo Curate event'
  Cmd+Z press #30: the table changed=True
       main window status bar : 'Undone: Curate event.'
       inspector progress line: 'Reviewed 10 / 10 · accepted 10'
       inspector status bar   : ''
       Edit > Undo (main menu): "Undo — this table's oldest history was dropped at the 30-step ceiling (20 steps lost)" enabled=False
  Cmd+Z press #31: the table changed=False
       main window status bar : 'Undone: Curate event.'
       inspector progress line: 'Reviewed 10 / 10 · accepted 10'
       inspector status bar   : ''
       Edit > Undo (main menu): "Undo — this table's oldest history was dropped at the 30-step ceiling (20 steps lost)" enabled=False
  Cmd+Z press #55: the table changed=False
       main window status bar : 'Undone: Curate event.'
       inspector progress line: 'Reviewed 10 / 10 · accepted 10'
       inspector status bar   : ''
       Edit > Undo (main menu): "Undo — this table's oldest history was dropped at the 30-step ceiling (20 steps lost)" enabled=False
presses that undid something: 30 of 55; the last one that did was press #30
after all of them the inspector still shows 0 rejected / 10 reviewed events that Cmd+Z can no longer reach
press after the ceiling: main status bar still says 'Undone: ...' (a stale success), inspector says nothing: True | inspector mentions the ceiling: False
RESULT: REPRODUCED
```


### S8-DATA4 - Compare baseline estimators on a stale table opens a raw Python ValueError modal

> Analysis > Compare baseline estimators... (and the baseline banner's Re-measure) on a table a time-base declaration made stale opens a modal that is a raw Python exception

*Lens: data safety and persistence* · **tier B** — found by the data-safety lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Analysis > Compare baseline estimators... (and the baseline banner's Re-measure) on a table a time-base declaration made stale opens a modal that is a raw Python exception. The guard itself is right -- ``remeasure_events`` refuses to re-measure at a rate other than the one the table was detected at, because that "would silently rewrite the geometry this function exists to preserve" -- but nothing in front of it gates the action, so the user, who has just declared a rate and sees an amber line saying "Re-run detection under the declared rate", clicks the enabled menu item and reads, as the whole text of a modal: "ValueError: these events were detected at 50000.000001892884 Hz but this re-measure was handed 40000.0 Hz. t_start_s and dwell_s are derived from the rate, ... Pass the recording's own rate." -- an exception class, an unrounded float, and an instruction addressed to a caller of a Python function. The status bar says only "Re-measure discarded: it could not finish. The table is unchanged." Neither sentence mentions the declaration or that Re-run is the way forward. (The sibling editing doors -- hand-add, edge drag, Merge all -- have no such guard at all: S8-DATA1.)

**See also.** S8-DATA1 (the editing doors that do not refuse) and S8-XS3.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-DATA4_compare_estimators_on_stale_table_prints_a_valueerror.py`

**Re-run.** `S8-DATA4_compare_estimators_on_stale_table_prints_a_valueerror.py` → reproduced (exit 7, 13.9 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
detected 10 events at 50,000.0000 Hz
declare -> Declared: 40,000 Hz. Save the project to retain it; detections already made on this recording are marked stale.
amber line: The sample rate changed since this detection ran (50,000 Hz → 40,000 Hz): every start time, dwell, rise/fall time and ECD in this table was computed at the old rate, and the filters and dwel
Analysis > Compare baseline estimators... enabled: True | text: &Compare baseline estimators…
MODAL TEXT : ValueError: these events were detected at 50000.000001892884 Hz but this re-measure was handed 40000.0 Hz. t_start_s and dwell_s are derived from the rate, so re-measuring at a different one would silently rewrite the geometry this function exists […]
MODAL BUTTONS: (OK)
status bar  : Re-measure discarded: it could not finish. The table is unchanged.
the sentence names the declaration or the Re-run remedy: False | starts with a Python exception class name: True | speaks to a programmer ('Pass the recording's own rate.'): True
RESULT: REPRODUCED
```


### S8-THREAD1 - Detect during 'Measure this recording' leaves the probe button stuck on 'Measuring…'

> press Detect while "Measure this recording" is running and the probe button reads "Measuring..." for ever: the button is dead, the controller is idle, and only selecting another recording clears it

*Lens: concurrency and lifetime* · **tier B** — found by the concurrency-and-lifetime lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). press Detect while "Measure this recording" is running and the probe button reads "Measuring..." for ever: the button is dead, the controller is idle, and only selecting another recording clears it. The open-pore probe ("Measure this recording" on the Baseline reference card) is an analysis job. The app runs one analysis job at a time and Detect REPLACES a running job ("Replaced baseline_probe with detect.schmitt. The previous run's result will not be applied."). The replaced probe never reports back, so the Baseline panel's own busy flag - set when the button was clicked - is never cleared: after everything has finished (controller idle, nothing pending) the button still says "Measuring...", is flagged busy, and a second click starts nothing. The state is cleared only by a recording change (BaselinePanel.set_recording), so a user who pressed Detect a second too early has a probe button that looks like it is working and is not, with no message and no way to cancel. What it does: the -500 mV recording (a COPY) is opened and detected; "Measure this recording" is clicked; Detect is pressed 120 ms later; everything is allowed to finish; the button and the controller are read; the button is clicked again; the -600 mV recording is then opened (the recording change that clears it).

**See also.** S8-THREAD2.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-THREAD1_probe_replaced_by_detect_leaves_button_stuck.py`

**Re-run.** `S8-THREAD1_probe_replaced_by_detect_leaves_button_stuck.py` → reproduced (exit 7, 20.6 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
before: button 'Measure this recording' enabled=True busy=False
probe running: button 'Measuring…' busy=True controller busy=True pending=probe
Detect pressed: status 'Replaced baseline_probe with detect.schmitt. The previous run's result will not be applied.'
EVERYTHING FINISHED: button 'Measuring…' enabled=True busy=True | controller busy=False pending=None
second click on the button: controller busy=False (a working button starts a probe) | status 'Replaced baseline_probe with detect.schmitt. The previous run's result will not be applied.'
after opening another recording: button 'Measure this recording' busy=False
RESULT: REPRODUCED - the probe button is stuck on 'Measuring...' with nothing running
```


### S8-NUM2 - Step readout cuts off notch ringing and reports a fraction of the real overshoot

> the Response tab's Step readout ("Applied: 10-90 % rise ... overshoot ... pre-shoot ...") is measured over a window whose width is set by the FASTEST stage, so for the most ordinary chain (a mains notch plus a low-pass) it cuts the notch's 50/60 Hz ring-down off and reports a fraction of the real overshoot: "overshoot 0.0 % . pre-shoot 0.2 %" for a chain whose real step response overshoots by 1.54 % (and 0.8 % / 0.9 % for notch 60 Hz x3 + low-pass, whose real value is 3.42 %)

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). the Response tab's Step readout ("Applied: 10-90 % rise ... overshoot ... pre-shoot ...") is measured over a window whose width is set by the FASTEST stage, so for the most ordinary chain (a mains notch plus a low-pass) it cuts the notch's 50/60 Hz ring-down off and reports a fraction of the real overshoot: "overshoot 0.0 % . pre-shoot 0.2 %" for a chain whose real step response overshoots by 1.54 % (and 0.8 % / 0.9 % for notch 60 Hz x3 + low-pass, whose real value is 3.42 %). The same notch on its own (stage selected) reads 3.4 %, so adding a low-pass appears to remove ringing that a low-pass cannot remove. What it does: a synthetic 1 s CSV recording (white noise, 50 kHz, so no real recording is needed) is opened in the real MainWindow; Analysis > Signal conditioning... > Response > Step is read for (a) the notch alone (stage selected), (b) notch 50 Hz x1 + 5 kHz Bessel low-pass with "Chain response" on, (c) notch 60 Hz x3 + 5 kHz Bessel low-pass with "Chain response" on, and each readout is compared with the overshoot of the app's own conditioned_trace applied to a real 100 pA step (analysis.conditioning.conditioned_trace, the function detection and every export use), measured over +-1 s.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM2_step_readout_truncates_notch_ringing.py`

**Re-run.** `S8-NUM2_step_readout_truncates_notch_ringing.py` → reproduced (exit 7, 16.8 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
(a) notch 60 Hz x3 alone
    window shows : Applied: 10–90 % rise 16.0 µs · overshoot 3.4 % · pre-shoot 3.4 %. Single pass: rise 16.0 µs · overshoot 7.0 %. Solid: a
    printed overshoot 3.4 %, pre-shoot 3.4 %   |   conditioned_trace on a 100 pA step: overshoot 3.43 %, pre-shoot 3.43 %
(b) notch 50 Hz x1 + 5 kHz Bessel
    window shows : Applied: 10–90 % rise 96.8 µs · overshoot 0.0 % · pre-shoot 0.2 %. Single pass: rise 75.9 µs · overshoot 3.8 %. Solid: a
    printed overshoot 0.0 %, pre-shoot 0.2 %   |   conditioned_trace on a 100 pA step: overshoot 1.54 %, pre-shoot 1.54 %
(c) notch 60 Hz x3 + 5 kHz Bessel
    window shows : Applied: 10–90 % rise 96.3 µs · overshoot 0.8 % · pre-shoot 1.0 %. Single pass: rise 76.2 µs · overshoot 3.4 %. Solid: a
    printed overshoot 0.8 %, pre-shoot 1.0 %   |   conditioned_trace on a 100 pA step: overshoot 3.42 %, pre-shoot 3.42 %
RESULT: REPRODUCED
```


### S8-NUM3 - 'Send to notch' snaps to the nominal 60 Hz and leaves a line 0.44 Hz away 15 dB down

> Signal Conditioning > Spectrum > "Send to notch" puts the notch on the NOMINAL mains harmonic (60.0 Hz) for any line the finder measures within 0.5 Hz of it, but a Q = 30 notch only takes 20 dB off a line within 0.33 Hz of its centre (the app's own notch_covers rule)

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Signal Conditioning > Spectrum > "Send to notch" puts the notch on the NOMINAL mains harmonic (60.0 Hz) for any line the finder measures within 0.5 Hz of it, but a Q = 30 notch only takes 20 dB off a line within 0.33 Hz of its centre (the app's own notch_covers rule). A line measured 0.44 Hz away (59.557 Hz) is snapped to 60.0, sent, and left 15.4 dB down instead of removed; the list correctly does not mark it "notched" -- and pressing Send to notch again answers "Notch: 60 Hz is already removed - nothing was added." in the status bar and under the list, over a line that the same window still shows at +4.5 dB. The refusal is checked against the SNAPPED frequency (60.0, which the notch trivially covers), not against the measured line (59.557), so the one control that places the notch cannot place it where the line is. What it does: a synthetic 20 s recording (white noise sigma 4.5 pA + a 59.55 Hz, 2.0 pA tone; CSV, 50 kHz) is opened in the real MainWindow, Signal Conditioning > Spectrum, 2 kHz low-pass (so the band is 20 Hz-2 kHz), the line is ticked and sent; the processed PSD at the line is compared with the raw one, and Send is pressed a second time.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM3_send_to_notch_snaps_past_what_the_notch_covers.py`

**Re-run.** `S8-NUM3_send_to_notch_snaps_past_what_the_notch_covers.py` → reproduced (exit 7, 15.2 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
spectrum: Δf 0.76 Hz (nperseg 65,536) on a 20.00 s window | 1 line in 20 Hz – 2 kHz.
finder: '60.0 Hz · +33.0 dB · mains 60×1'  measured 59.552 Hz, sends 60.0 Hz, snapped=True
first Send: [60.0] -> Notch: added 60 Hz — the explicit line list is now 60 Hz.
notch list now [60.0]; list row: ['60.0 Hz · +33.0 dB · mains 60×1']
line power removed by the notch: 15.4 dB   (notch_covers(measured 59.552, 60.0, Q=30) = False; its radius f0/(6Q) = 0.333 Hz, the line is 0.448 Hz off)
second Send: Notch: 60 Hz is already removed — nothing was added. | under the list: Notch: 60 Hz is already removed — nothing was added.
RESULT: REPRODUCED
```


### S8-NUM4 - Spectrum lines list is silently capped at 24 and the note reads as a count

> Signal Conditioning > Spectrum prints "24 lines in 20 Hz - 25 kHz." for a spectrum in which the line finder found 60: 24 is the list's own cap (conditioning_preview.find_spectral_lines max_lines=24, _MAX_LINE_ROWS = 24), not a count, and nothing on screen says more lines exist or which were left out

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Signal Conditioning > Spectrum prints "24 lines in 20 Hz - 25 kHz." for a spectrum in which the line finder found 60: 24 is the list's own cap (conditioning_preview.find_spectral_lines max_lines=24, _MAX_LINE_ROWS = 24), not a count, and nothing on screen says more lines exist or which were left out. On the owner's own -600 mV apo-ferritin recording, 10 s window, no low-pass, the amplifier's switching comb at 11-22 kHz fills the list (the weakest listed line is +13.7 dB, the strongest unlisted +13.5 dB), so a real 60 Hz line of +14 dB -- exactly what the tab exists to find and "Send to notch" -- is simply not offered, and the note under the list reads as a complete census. What it does: (1) the recording (a COPY), IN15, main trace on 30-40 s, Spectrum tab, no low-pass: the note, the number of rows, and the same spectrum run through find_spectral_lines with no cap; (2) a CSV built from the same 10 s of noise plus a 60 Hz tone (measured +14 dB) opened the same way: the tone is in the uncapped list and absent from the window's.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM4_spectrum_lines_list_is_capped_silently.py`

**Re-run.** `S8-NUM4_spectrum_lines_list_is_capped_silently.py` → reproduced (exit 7, 15.3 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
(1) recording, 10 s window: Δf 1.53 Hz (nperseg 32,768) on a 10.00 s window — resolving 
    note under the list: '24 lines in 20 Hz – 25 kHz.'
    rows listed: 24   lines the finder sees in that spectrum: 60
    weakest listed 13.7 dB, strongest unlisted 13.5 dB
(2) noise + 60 Hz tone: '24 lines in 20 Hz – 25 kHz.'
    uncapped finder: ['60.0 Hz · +13.6 dB · near mains 60×1'] (rank by prominence 25 of 61)
    rows in the window near 60 Hz: []
RESULT: REPRODUCED
```


### S8-NUM5 - A segment drops its parent's question about an undeclared time unit

> a segment cut from a recording whose time column has no declared unit drops the question the app asks about that time base

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). a segment cut from a recording whose time column has no declared unit drops the question the app asks about that time base. The parent's File Info says "this file's time column steps by 0.02 per sample. Read as seconds that is 50 Hz ... If this time base is in milliseconds the true rate is 50,000 Hz and every duration, dwell time and event rate here is 1,000x wrong"; the segment cut from it (File > Create segment, from a Measure selection) carries the same 50 Hz, "measured", a 1000 s Duration and Issues "(none)", because build_segment_document sets issues=() ("the parent's load issues are not the segment's"). The 1,000x question is exactly the load issue that IS the segment's too -- it travels with the samples -- and the segment is the object the user then detects on. What it does: a 2 s slice of the -600 mV apo-ferritin recording (a COPY; IN15 in pA) is written to a temp CSV whose bare "time" column is in MILLISECONDS (the case the new reader asks about), opened in the real MainWindow, a 1 s Measure region is turned into a segment through the real Create-segment action, and File Info is read for the parent and for the segment.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM5_segment_drops_the_undeclared_time_unit_question.py`

**Re-run.** `S8-NUM5_segment_drops_the_undeclared_time_unit_question.py` → reproduced (exit 7, 13.9 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
PARENT  rate: 50 Hz  ·  measured | duration: 2000 s
        issues: this file's time column steps by 0.02 per sample. Read as seconds that is 50 Hz — below the 1,000 Hz any recording this app is written for is sampled 
SEGMENT rate: 50 Hz  ·  measured | duration: 1000 s
        issues: '(none)' | notes: Derived segment of slice_time_in_ms.csv (samples 10,000–60,000  ·  t₀=10200s).
RESULT: REPRODUCED
```


### S8-NUM6 - The chain sentence prints '(order 8)' for a Gaussian and a window the masked median ignores

> the chain sentence the app calls "the recipe NanoScope will actually run" prints two numbers the chain does not have

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). the chain sentence the app calls "the recipe NanoScope will actually run" prints two numbers the chain does not have. The Preprocess card's "Active execution order: ..." line (and the "Chain:" line in the header of every Copy-as-Python script) is analysis.conditioning.chain_description(); for a Gaussian low-pass it appends "(order 8)" -- a Gaussian has no order (the form disables the box, effective_chain drops it) -- and for the event-masked median baseline it prints detrend_window_s, which that method never reads (it uses rough/base). The Signal Conditioning stage list beside it prints the real parameters, so the two surfaces describe one chain differently: Preprocess card : detrend masked_median 2 s -> low-pass gaussian 2000 Hz (order 8) stage list : event-masked median . rough 0.5 s / base 0.02 s | Gaussian . 2 kHz What it does: a synthetic 1 s CSV recording is opened in the real MainWindow, the chain is set to a Gaussian 2 kHz low-pass and an event-masked-median baseline (rough 0.5 s, base 0.02 s, and an inert window of 2 s), and the three sentences are read: the card's order line, the Signal Conditioning stage rows, and the Copy-as-Python header.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM6_execution_order_sentence_prints_numbers_the_chain_ignores.py`

**Re-run.** `S8-NUM6_execution_order_sentence_prints_numbers_the_chain_ignores.py` → reproduced (exit 7, 13.3 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
Preprocess card : Active execution order: detrend masked_median 2 s → low-pass gaussian 2000 Hz (order 8)
Copy as Python  : Chain: detrend masked_median 2 s → low-pass gaussian 2000 Hz (order 8)
stage list      : 4  Baseline drift — event-masked median · rough 0.5 s / base 0.02 s (estimator)  ||  8  Low-pass — Gaussian · 2 kHz (applied inside detection)
effective_chain : {'method': 'masked_median', 'rough_s': 0.5, 'base_s': 0.02, 'mask_sigma': 3.5, 'dilate_ms': 1.0, 'decim': None, 'polarity': -1} {'cutoff_hz': 2000.0, 'kind': 'gaussian'}
RESULT: REPRODUCED
nanoscope/analysis/conditioning.py:185: AmplifiedWanderWarning: the detrend AMPLIFIED the baseline wander it exists to remove: peak wander 0.1 pA in -> 0.1 pA out (+49%), RMS 0.0 -> 0.1 pA, measured on 0.25 s block medians. Wander with a period between about h […]
  processed = preprocess_for_detection(raw, fs, params)
```


### S8-NUM7 - Notch note 'deepest N dB' is a plot-grid artifact, not a property of the filter

> the Response tab's notch note prints "deepest N dB" for the designed notch, but N is the depth of the plot's nearest grid point to the notch centre, not a property of the filter: it falls as Q rises (the opposite of what a narrower notch does) and differs by 19 dB between a 60 Hz and a 50 Hz notch of the same Q. At Q = 200 the note says "deepest -28 dB"; the app's own conditioning kernel removes a 100 pA sine exactly at the centre by more than 100 dB at the same setting, and at a 0.05 Hz offset by ~26-47 dB depending on Q. What it does: a synthetic 1 s CSV recording is opened in the real MainWindow; Signal Conditioning > Response > Magnitude with the Notch stage selected is read for 60 Hz x3 at Q 5 / 30 / 100 / 200 and 50 Hz x3 at Q 30; for each, analysis.conditioning.conditioned_trace (what detection and every export run) is applied to a 100 pA sine at the notch's own centre frequency and the attenuation is measured on the output

*Lens: numeric honesty* · **tier B** — found by the numeric-honesty lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). the Response tab's notch note prints "deepest N dB" for the designed notch, but N is the depth of the plot's nearest grid point to the notch centre, not a property of the filter: it falls as Q rises (the opposite of what a narrower notch does) and differs by 19 dB between a 60 Hz and a 50 Hz notch of the same Q. At Q = 200 the note says "deepest -28 dB"; the app's own conditioning kernel removes a 100 pA sine exactly at the centre by more than 100 dB at the same setting, and at a 0.05 Hz offset by ~26-47 dB depending on Q. What it does: a synthetic 1 s CSV recording is opened in the real MainWindow; Signal Conditioning > Response > Magnitude with the Notch stage selected is read for 60 Hz x3 at Q 5 / 30 / 100 / 200 and 50 Hz x3 at Q 30; for each, analysis.conditioning.conditioned_trace (what detection and every export run) is applied to a 100 pA sine at the notch's own centre frequency and the attenuation is measured on the output.

**See also.** S8-NUM2 (the same Response tab's notch readouts) and S8-NUM3.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-NUM7_notch_note_deepest_is_a_plot_grid_artifact.py`

**Re-run.** `S8-NUM7_notch_note_deepest_is_a_plot_grid_artifact.py` → reproduced (exit 7, 15.7 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
notch 60 Hz x3, Q 5: note says 'deepest -92 dB'; conditioned_trace on a sine at the centre: -152 dB
notch 60 Hz x3, Q 30: note says 'deepest -61 dB'; conditioned_trace on a sine at the centre: -152 dB
notch 60 Hz x3, Q 100: note says 'deepest -40 dB'; conditioned_trace on a sine at the centre: -152 dB
notch 60 Hz x3, Q 200: note says 'deepest -28 dB'; conditioned_trace on a sine at the centre: -117 dB
notch 50 Hz x3, Q 30: note says 'deepest -80 dB'; conditioned_trace on a sine at the centre: -152 dB
RESULT: REPRODUCED
```


### S8-KEYS3 - Tab cannot leave the Results dock's tables

> Tab cannot leave the Results dock's tables: QTableView.tabKeyNavigation is on, so Tab walks the cells (7,291 of them in the Events table of the -600 mV detection) and wraps at the end

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Tab cannot leave the Results dock's tables: QTableView.tabKeyNavigation is on, so Tab walks the cells (7,291 of them in the Events table of the -600 mV detection) and wraps at the end. WHAT IT SHOWS. The Results dock's event table is a plain QTableView, whose tabKeyNavigation defaults to True: Tab / Shift+Tab move the current CELL, and even pressed on the very last (first) cell the key wraps round inside the table instead of leaving it. Shift+Cmd+R "hands the keyboard back to the table", and the Inspector's controls sit before the table in the Tab chain, so a keyboard user arrives in the table and is stuck: 400 Tab presses later the focus is still the table (the cell cursor is at row 17), and the 6 workflow-stage chips, 'Analysis home' and the 7 analysis launcher buttons - the next stops in the chain - cannot be reached by Tab from the table. Shift+Tab is the same. The Summary tab's table (11 rows x 1) does the same. (A keyboard user gets out only by Shift+Cmd+R, which hides the dock and parks the focus on the trace, or by a shortcut that opens another window; Tab and Shift+Tab never leave.)

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS3_tab_trapped_in_results_table.py`

**Re-run.** `S8-KEYS3_tab_trapped_in_results_table.py` → reproduced (exit 7, 35.8 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
Results table: 317 rows x 23 visible columns = 7291 cells; QTableView.tabKeyNavigation() = True
focus before: QTabBar#qt_tabwidget_tabbar
one Tab later: focus = QTableView -> the Results table now holds the keyboard
400 Tab presses from cell (0, 18): focus = QTableView, current cell (17, 0)  (never left the table)
Shift+Tab: focus = QTableView | current cell (17, 17)
Tab pressed ON the last cell (316, 24): focus = QTableView, current cell now (0, 18)
Summary tab table (11 rows x 1): tabKeyNavigation=True; 60 Tab presses later the focus is still in it
Presses needed to Tab through the Events table from its first cell: 7290.
The next stops in the Tab chain, which Tab cannot reach from the table: the 6 workflow-stage chips, 'Analysis home' and the 7 analysis launcher buttons.
RESULT: REPRODUCED
```


### S8-KEYS4 - Cmd+I in the gallery inverts the tile selection instead of opening the inspector

> Cmd+I, documented as "Open the Event inspector", INVERTS THE TILE SELECTION in the Event gallery - and the next R rejects the complement

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Cmd+I, documented as "Open the Event inspector", INVERTS THE TILE SELECTION in the Event gallery - and the next R rejects the complement. WHAT IT SHOWS. USER_GUIDE section 13 and the examples README give Ctrl/Cmd+I as "Open the Event inspector" (unscoped), and main_window.py says "Ctrl+I / Ctrl+G are both free ... in every child window". The gallery window binds Ctrl+I to its own Select > Inverse (event_gallery.py, act_select_inverse, WidgetWithChildren scope). So in the Event gallery the documented chord opens nothing; it swaps the selected tiles for the unselected ones on the page, and the batch keys (A / R / Space / 1-9) then act on the other tiles. (In the Event inspector's own window Cmd+I and Cmd+G are simply dead; the inspector has a Gallery... button.)

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS4_gallery_cmd_i_inverts_selection.py`

**Re-run.** `S8-KEYS4_gallery_cmd_i_inverts_selection.py` → reproduced (exit 7, 16.2 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
USER_GUIDE section 13: | **Ctrl+I** | **Open the Event inspector** |
USER_GUIDE section 13: | **Ctrl+G** | **Open the Event gallery** |
main window, Cmd+I -> Event inspector visible: True
gallery: tiles on the page = 15, selected = [2, 3, 4]
Cmd+I in the gallery -> Event inspector visible: False; selection now 12 tiles [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
then R: accepted 316 -> 305  (11 events rejected - the COMPLEMENT of what was selected)
actions bound to Ctrl+I in the gallery window: ['Inverse']
RESULT: REPRODUCED
```


### S8-KEYS5 - The first Cmd+G of a session replaces the selected event with event 0

> The first Cmd+G of a session replaces the event the user has selected with event 0 and moves the trace and the Results table there

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). The first Cmd+G of a session replaces the event the user has selected with event 0 and moves the trace and the Results table there. WHAT IT SHOWS. MainWindow._open_event_gallery builds the gallery on the first call; its set_events selects tile 0 and announces it (eventSelected -> AnalysisPanel.select_event -> the table row, the trace zoom and the marker), so a curator who has event 150 on screen and presses Cmd+G finds the trace and the Results selection on event 0 instead, the status bar saying "Event 0 selected - the trace, the table and the marker moved". The Event inspector, opened the same way, opens ON the selected event (the CHANGELOG: "opens on the event selected in the table ... instead of event 0"); the gallery was not given that. Only the first open per session does it - a later Cmd+G keeps the selection.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS5_first_cmd_g_jumps_to_event_0.py`

**Re-run.** `S8-KEYS5_first_cmd_g_jumps_to_event_0.py` → reproduced (exit 7, 18.1 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
user clicked Results row 150: selected event 150 (starts 33.000 s), trace shows (94.295, 94.33)
after Cmd+G (first time this session): selected event 0 (event 0 starts 0.000 s), trace shows (61.268, 61.41)
status bar: 'Event 0 selected — the trace, the table and the marker moved. Double-click the tile to bring the trace forward.'
second Cmd+G: selected event 200 -> 200 (kept) - only the FIRST open of a session replaces the selection
Cmd+I with event 250 selected -> the inspector shows event 250
RESULT: REPRODUCED
```


### S8-KEYS6 - Signal Conditioning's toolbar cannot be reached from the keyboard

> Signal Conditioning's toolbar (Follow main trace, Show main trace, Copy as Python, ML chain..., Import protocols..., Export protocols...) cannot be reached from the keyboard at all

*Lens: keyboard, focus, menus* · **tier B** — found by the keyboard-and-focus lens finder, reproduced by the coordinator's re-run, not skeptic-attacked · severity: the coordinator's call (minor)

**What it shows** (the finder's own account, from its repro). Signal Conditioning's toolbar (Follow main trace, Show main trace, Copy as Python, ML chain..., Import protocols..., Export protocols...) cannot be reached from the keyboard at all. WHAT IT SHOWS. SignalConditioningWindow.toolbar is a QToolBar of text actions; Qt gives every toolbar button Qt.NoFocus, none of the six actions has a shortcut, and none exists in a main-window menu (Analysis > Signal conditioning... only opens the window). The same window deliberately makes its other buttons TabFocus (Revert, Run detection..., Compute exact, Send to notch) so a Tab walk visits them - the toolbar is the gap: a real Tab walk around the whole window (back to its first stop) never stops on it. "Follow main trace" decides what is previewed, "Copy as Python" is the reproducibility export, and Import/Export protocols are the library's only doors.

**Repro.** `QT_QPA_PLATFORM=offscreen S8_RECORDINGS=<folder holding the .abf files> /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-KEYS6_signal_conditioning_toolbar_mouse_only.py`

**Re-run.** `S8-KEYS6_signal_conditioning_toolbar_mouse_only.py` → reproduced (exit 7, 15.3 s) — by the coordinator on a53bf94, under a write guard.

**Measured** (what the repro printed on that re-run, Qt and pandas noise removed):

```text
Signal Conditioning open; focus = QListWidget#signalConditioningStages
Toolbar actions (the ONLY place these exist):
   'Follow main trace'      focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
   'Show main trace'        focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
   'Copy as Python'         focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
   'ML chain…'              focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
   'Import protocols…'      focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
   'Export protocols…'      focusPolicy=NoFocus  shortcut=''   also in a main-window menu: False
Tab walked the whole window (11 stops, back to the stage list): stops inside the toolbar = 0
stops: ['PlotWidget', 'QComboBox', 'QDoubleSpinBox', 'QListWidget#signalConditioningStages', 'QPushButton', 'QPushButton#signalConditioningRun', 'QSpinBox', 'QTabBar#qt_tabwidget_tabbar', 'QToolButton', '_ElidingCheckBox']
RESULT: REPRODUCED
```


---

## What the skeptics found that the finders missed

These are NOT independently verified: each was found by a skeptic on the way to refuting something else, with the evidence it gave. Treat them as leads with a repro to write.

- **S8-SK1** *(found while testing: RD: The Results dock's Fits tab shows one stored fit per (recording, channel) under […])* Plots window writes its own fits into the same fit.dwell / fit.depth slot on close (code-read only, NOT measured) — A user fits a dwell or depth column in Analysis > Plots (possibly with different components or on a different accepted subset) and closes the window; PlotsWindow._persist_results files that fit under 'fit.dwell'/'fit.depth' for the source's (doc_id, channel), i.e. the exact key the Results dock's Fits tab reads back, so the Fits tab would then print the Plots fit beneath the event table with no sentence saying it came from the Plots window. Same shared-slot family as S8-RD1; unverified through the widgets. *Evidence:* nanoscope/gui/plots_panel.py:11925-11960 (_FIT_MODULE_IDS maps dwell/depth columns to 'fit.dwell'/'fit.depth'; _persist_results pushes one AnalysisResult per fitted source under those ids on closeEvent); nanoscope/gui/analysis_panel.py:4138-4139 re-reads the slot on resultsChanged for those module ids. Not run; recorded as skeptic-found, unverified.

- **S8-SK2** *(found while testing: RD: After a sample-rate declaration, the Reconciled table's amber line keeps saying […])* A project saved with the CUSUM or Reconciled table selected reopens with the SCHMITT rows in the Results table under the other table's label (and without that table's stale line) — Open the -600 mV recording, run Schmitt (317) and CUSUM (370), select the CUSUM table, save the project, reopen it. The selector reads 'CUSUM — 370 candidates', panel._active_detection and current_detection() (what exports and the inspector read) are the CUSUM result, but the table on screen holds 317 rows: Schmitt's. Same with the Reconciled table selected at save: 'Reconciled — 443 candidates' over 317 Schmitt rows (M1), and on the -700 mV rate-declared project 'Reconciled — 1,970' over 1,130 Schmitt rows with NO sample-rate stale line although the reconciled snapshot is stamped 50 kHz against a live 20 kHz (R0/P0). […] *Evidence:* Repro: QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -u docs/sweep8/repros/S8-RD2_skeptic_reopened_table_shows_schmitt_rows.py -> exit 7: [CUSUM selected, before save] selector='CUSUM — 370 candidates' labelled_rows=370 table_rows=370; [REOPENED] selector='CUSUM — 370 candidates' active='detect.cusum' labelled_rows=370 table_rows=317; [after switching away and back] table_rows=370. Three-table run (scratch roundtrip_norate.py, log roundtrip_norate.log): […]

- **S8-SK3** *(found while testing: EI: In the Event inspector's own window narrower than ~800 px the status bar's widgets […])* Project label in the inspector's status bar elides its own text by one character with the bar nearly empty ("Project: No project f…" at 1100 px, "Project: Saved — apo_600.nsp…" at 1400 px) — Open the Event inspector in its own window at its default 1100 px (no detection yet): the status bar reads 'Project: No project f…' with ~640 px of empty progress line beside it. After a detection and a save (set_project_save_state('/tmp/x/apo_600.nsproj', False)) at 1400 px it reads 'Project: Saved — apo_600.nsp…'; 'Project: Unsaved changes — apo_600.nsp…' likewise. The text 'Project: Unsaved changes — no project file yet' happens to show in full. The user sees a cut file name although nothing is long; the guide says the name gives way only 'when both are long'. *Evidence:* Cause: _ElidedLabel.sizeHint = horizontalAdvance(text)+4 (event_inspector.py:766-769) while _fit elides at contentsRect().width()-4 (:786-788), i.e. at exactly horizontalAdvance; QFontMetrics.elidedText with .AppleSystemUIFont 13 (fractional advances) elides at that width for some strings. _share_status_bar pins the label's maximumWidth to that sizeHint when there is room (:2114-2117, width = want_l), so the label never gets the extra pixel. Pure-widget measurement (<tmp>): 'Project: […]

- **S8-SK4** *(found while testing: EI: Window ▸ Event gallery, Window ▸ Figure Library and Window ▸ Signal Conditioning (and […])* Window menu check-mark semantics differ between the main entry and the child entries when a window is minimized — Minimize the main window and the Event gallery, open the Window menu: 'NanoScope' is unchecked (checked = isVisible and not isMinimized, main_window.py:6965-6966) while 'Event gallery', 'Figure Library' and 'Signal Conditioning' are checked (_on_screen = isVisible, :539-551, true while minimized). The same menu answers 'is it on screen?' two different ways; the guide says the mark means 'on screen'. A facet of the confirmed finding, not a separate repro. *Evidence:* var.py output: 'Window menu check mark for minimized main: checked=False'; 'minimized event_gallery: checked=True'; 'minimized figure_library: checked=True'; 'minimized signal_conditioning: checked=True'.

- **S8-SK5** *(found while testing: DOC: The Signal Conditioning Spectrum header tells a user whose window is already […])* CSV time column written with 8 decimals at 30 kHz is read as 30003 Hz [MEASURED] with no issue noted (median of alternating steps) — A CSV whose time column is in seconds with 8 decimal places at a rate whose period is not representable (1/30000 = 0.0000333…) has steps alternating 0.00003333 / 0.00003334. io/text.py takes the MEDIAN step (line 456-457), which lands on 0.00003333, so File Info / the document report 30003 Hz rather than 30000 Hz, and the jitter guard (std/dt ≈ 1.5e-4 < 1e-3, line 497-500) adds no issue, so nothing on the surface says the rate was rounded. 0.01% on every duration and on the Spectrum axis; low impact, and it is partly a property of the exporter's precision, so it may fall below the bar — recorded as skeptic-found, unverified beyond the measurement below. *Evidence:* Fixture <tmp> (header 'time,Untitled', rows 0.00000000 / 0.00003333 / 0.00006667 …, written with np.savetxt fmt %.8f at exactly 30000 Hz). Opened with win.open_paths in variations.py: doc.meta.sample_rate_hz printed as 30003 (output line '== synth_30k.csv: fs=30003 Hz'); the mean step would give 30000.0. Site: nanoscope/io/text.py:456-457 (dt = float(np.median(diffs)); sample_rate_hz = 1.0 / dt), 497-500 (jitter issue only when std/dt > 1e-3). Not checked: […]

- **S8-SK6** *(found while testing: DOC: USER_GUIDE §9 tells the user a new Event inspector 'can start at the first event […])* Re-opened inspector shows an event outside its kept filter subset with a stale spinner position ("Event 1 of 4" while Event 500 is displayed) — Open the Event inspector, set its Filter to "Rejected only" (4 members after rejecting events 10, 11, 12 on the -600 mV CUSUM table), close the inspector (docked tab closed), click an ACCEPTED row in the Results table (event 500), press Ctrl+I. The inspector re-opens on event 500 (title "Event 500 · dwell 2.048e+04 µs · median depth 202.4 pA · dI/I0 2.68", correct) but the filter combo still reads "Rejected only" and the Event spinner reads "1" of "of 4", a leftover value: 500 is not in the subset and has no position in it. Next then snaps to event 12 (spin 4). […] *Evidence:* variations.py V3 output: "V3 filter set: Rejected only | subset size: 4 | shows: 0"; "V3 table selection (accepted, outside the Rejected subset): 500"; "V3 re-opened on: 500 | filter now: Rejected only | spin: 1 | count label: of 4 | progress: Reviewed 3 / 650 · accepted 646 | title: Event 500 · dwell 2.048e+04 µs · median depth 202.4 pA · dI/I0 2.68"; "V3 after Next: 12 | spin: 4 | table: 12". Sites: […]

---

## Leads from the lens round — observed, not established

None of these is counted in the totals above. Each says what was seen, by whom, and what is still missing before it could be called a defect.

- **S8-LEAD1** — **Intermittent native crashes under open / close / detect churn, seen only once the Signal Conditioning window has been created** *(found by the concurrency finder, re-run and attributed by the coordinator)* The lifecycle-storm probe (`docs/sweep8/repros/storm/s2_lifecycle_storm.py`: 120 random operations over three recordings, one every few milliseconds) ended in a segmentation fault in 3 of 14 runs today (exit 139) and once raised an exception from inside pyqtgraph (`'QWidgetItem' object has no attribute 'children'`, thrown while the trace view built a plot, which is what a wrapper handed back for a recycled address looks like); the other 10 runs finished with every invariant intact. The finder saw the same in 7 of the 20 storms it ran, and the time-unit finder met one more, inside the Plots window's constructor, which it could not repeat and treated as the known intermittent abort. It is timing-dependent: seed 14 crashed in 2 of its 4 runs. Across the 10 crashes of today's S2 and S4 runs the faulting thread is always the GUI thread constructing a Qt object: 6 in the constructor of the probe or load worker (`ProbeWorker.__init__`, workers.py:29, or `LoadWorker.__init__`, workers.py:51) as a recording is opened, in the statement right after `LoadController.abort()`; 3 in `QThread()` inside `start_worker_thread` (worker_thread.py:347), reached from the same open path or from starting a detection; 1 in the trace view while it rebuilds after a close (an `InfiniteLine`, trace_view.py:1919). The finder's own crashes were in the same family: `TraceView.__init__` and `_show_document`. **Attribution.** Further storms with a smaller set of operations: select, close, reopen, detect, stage toggles and pan, 120 per run (`s3_churn_storm.py` and `s4_churn_sc.py` in the same folder). Without the Signal Conditioning window the churn never crashed: 0 of 60 runs, on this tree (36 runs, collector off and on) and on the tree at the end of sweep 7, `1aeb651` (24 runs). With the window open and only reacting to the churn it crashed 5 of 24 runs (Fisher's exact test against the window-less runs, p = 0.0014). With the storm also closing, reopening and driving the window it crashed 0 of 24. With the window opened and closed at once and then left alone it crashed 2 of 24, so having created it is enough. The repository already documents an intermittent PySide6 crash of this family (CHANGELOG, 2026-09-13: a Qt wrapper whose C++ side is gone becomes invisible to the garbage collector; about 734 `QObject`s leak per main-window construct/destroy cycle). What is not established: whether anything here is reachable at human speed, whether the fault is NanoScope's handling of worker lifetimes or PySide6 itself, and why the window matters. No single run is a reliable repro; the storm scripts, the seeds used and how to run them on copies of the recordings are in `docs/sweep8/repros/storm/`. Where to start: the open path's worker creation (`LoadController.probe` and `start`, workers.py), which is where nine of the ten crashes fall, and a run under macOS's malloc guard (`MallocScribble`, `MallocGuardEdges`), which separates a use-after-free from a race.

- **S8-LEAD2** — **A stale Signal Conditioning result under random edits** *(the concurrency finder saw it once; the coordinator could not reproduce it)* The edit-storm probe (`docs/sweep8/repros/storm/s1_edit_storm.py`: 120 to 200 random stage toggles, parameter edits and tab changes on the 3-million-sample recording) printed once "last result is for another chain/source" with a difference of 184. Ten re-runs today (seeds 1–10) never reproduced it: every run ended on a result whose signature, source and sample range matched the current chain, and whose window equalled a recompute on the same padded window with a maximum difference of 0. The one line the probe still prints on most runs, "drawn curve has 2,746 points, result 2,000,000", is the probe comparing pyqtgraph's display-decimated curve with the full array. It is not a defect. Left open because the original sighting was real output and the storm is timing-dependent.

- **S8-LEAD3** — **Dwell text prints "1e+03 µs" and "1e+03 ms" just below a unit boundary** *(the numeric-honesty finder's probe, confirmed by the coordinator at function level)* The Event inspector's dwell formatter (`event_inspector.py`, `_fmt_dwell`, whose docstring promises "16.4 ms", not "1.642e+04 µs") formats with three significant digits, so a dwell from 999.5 to just under 1000 µs prints as "1e+03 µs" and one from 999.5 to just under 1000 ms prints as "1e+03 ms" instead of rolling over to "1 ms" and "1 s". Measured by calling the function: 999.4 → "999 µs", 999.5 → "1e+03 µs", 999.99998 → "1e+03 µs", 1000.0 → "1 ms", 999499 → "999 ms", 999500 → "1e+03 ms". No event in the three of the owner's tables the finder checked (2,088 events) lands in the band, so nothing the user has seen is wrong; it needs a rounding fix when the function is next touched. Cosmetic.

- **S8-LEAD4** — **Delete forgets a saved protocol at one click** *(the data-safety finder's probe, a design question rather than a defect)* In the Preprocess protocol library, Delete removes the selected protocol with no confirmation and no undo, and the protocol is gone after a restart. Its tooltip says "Forget the selected protocol.", so this may be intended. It is listed because a protocol is hand-tuned, user-authored content (the finder's library held 14) and Export protocols… is the only backup. Whether a confirmation is wanted is the owner's call.

---

## Raised and not upheld

- **S8-GEO1** — A saved window geometry cut short (4–45 of its 66 bytes) is silently used. Votes: confirmed (Reproduced through the real MainWindow exactly as the finder describes, and the variation sweep shows the hole is wider than claimed: on the 1512-wide screen `_restore_saved_geometry` returns True for EVERY prefix of 4..65 bytes of a 66-byte blob (only 0..3 bytes are refused), so the header check `_saved_normal_rect` written to name cut-short bytes never runs and nothing reaches the info bar or the status bar; […]); not-a-defect (The finder's measurements are accurate and reproduce: a 4–65-byte prefix of a sound 66-byte geometry blob is accepted by Qt's restoreGeometry() on the 1512-wide laptop screen, so `_restore_saved_geometry` returns at the first line and the header check never runs; the main window opens at 792x521 and the Event gallery at 770x281 with nothing on the info bar or the status bar. […])

---

## Housekeeping, not a product defect — the sweep wrote a sidecar beside a real recording

A sweep agent wrote a calibration sidecar next to the owner's real `-600 mV` recording, `500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf.npcal.json`. The Event-inspector finder's harness (`EI/p03_details_status.py:85` and `EI/p14_reset_sizes_stale.py:39`) called `Session.set_calibration(did, 2, gain * 1.5)` on the real path; `set_calibration` writes the sidecar beside the file by design (`write_sidecar=True`), so the file was created at 15:17:23 (`gain 1.5`) and rewritten at 15:40:25 (`gain 2.25`, i.e. 1.5 × 1.5, the script multiplying the gain it had just loaded). NanoScope would have applied it silently on the next open, scaling every current, depth and threshold on that recording by 2.25. It was reported unprompted by the documentation finder (as a coordinator note) and by the time-unit skeptics (as an additional finding).

**What it held before.** The script multiplies the gain currently in force and preserves every other field, and the first write produced exactly 1.5, so the calibration beforehand was gain 1.0 with blank units and note — what a recording with no sidecar loads as (the app deletes a sidecar that would hold only defaults). No backup exists to confirm the prior bytes (Time Machine is not configured, there are no local APFS snapshots).

**What was done.** The file was MOVED (not deleted) to the sweep's scratch folder under `sweep8/quarantine/` (inside the system temp directory, so the system may clear it), with a README giving the restore command, and the app's own loader was then run on the recording under a write guard: IN15 loads as `current`, gain 1.0, no calibration issue. The owner's two genuine sidecars (the `-1000 mV` 385–400 s trace, Sep 1; the nisin HDF5, Aug 24) were not touched. A final check found no other file the sweep could have written beside the recordings: the only files changed directly under the home directory since the sweep began are Claude Code's own configuration and Finder's folder-view file, and the repository is as it was apart from this document and its repro folder (its two other uncommitted entries, the icon and `examples/`, predate the sweep).

**Why it happened twice.** Sweep 7 recorded the same incident (a sidecar beside the `-500 mV` recording) and ended with "A future sweep must copy the data, or redirect the sidecar path, before driving calibration." That lesson sits at the foot of its document; this sweep's brief was written from its head and said the recordings were read-only without saying to work on copies. The resumed lens round therefore ran on copy-on-write clones of every recording (`sweep8/recordings/`) behind a `sandbox-exec` guard that denies writes to files directly under the home directory, and its repros copy their input into a temporary directory. A structural fix is cheaper than a lesson that depends on being read: an environment variable that redirects or disables sidecar writes, as `NANOSCOPE_RECOVERY_ROOT` already does for recovery files, would make this impossible by construction.

---

## Territory covered — what held up

What each finder tried that did NOT turn into a defect, as it reported it. Absence from this list is not evidence of correctness.

**Signal Conditioning workspace**

- Opening: Analysis ▸ Signal conditioning… (menu action) and ⇧⌘F (QTest Ctrl+Shift+F on the main window) both show the one window, built once; closing hides it (object kept, worker thread kept and idle); […]
- Without a detection the status row reads 'No Schmitt/CUSUM detection is shown, so the chain has nothing to be compared with.' with Revert disabled and Run detection… enabled; with a Schmitt detection it reads 'The form's chain is the one the Schmitt detection on screen ran with.'
- Time tab, three previews vs analysis/conditioning.conditioned_trace on the same params, sample for sample on the -600 mV ABF (10 s window, 500,000 samples): low-pass 10 kHz Bessel 0.0; Gaussian 1 kHz 0.0; Butterworth 10-pole 200 Hz 3.8e-6; hp 2 Hz butter + low-pass 7.6e-6; […]
- Stage toggles through the stage-list tick (notch row → notch_enabled True on the form, row text '1 Notch — 60/120/180 Hz · Q 30') and through the form (median, hp, rmean, savgol, detrend, normalize) all re-preview to the conditioned_trace of the new params; […]
- Window estimate vs exact: Baseline drift (block median 0.5 s) → 'Window estimate — Baseline drift uses whole-recording statistics · press Compute exact for the true trace', legend 'processed — window estimate', preview differs from the whole-recording trace by a constant 1.17 […]
- Pad-limited and short-recording cases are disclosed, not silent: notch 5 Hz Q 500 → 'Preview unavailable: notch settling requires 2,993,797 padding samples per side … above the safe 2,000,000-sample limit…' and 'Exact pass failed: …' with the button disabled; […]
- Debounce/mailbox: 10 rapid cutoff edits → exactly 1 conditioned_trace call on the worker thread ('Dummy-1'), final preview == conditioned_trace(8100 Hz); […]
- Spectrum / Send to notch: on a 20 s fixture (real -600 mV samples + 60/180/300 Hz hum + 1234.5/4570.8/7777.7 Hz lines, Δf 0.76 Hz) the finder lists the three mains lines snapped ('mains 60×1/3/5') and the others to 0.1 Hz; […]
- Response tab: the magnitude view equals an independent |H|² (sosfreqz squared for Bessel/Butterworth/high-pass/notch, Gaussian/running-mean/Savitzky-Golay once) to ≤ 1e-9 dB for every stage and for the 5-stage chain on the same grid (−5.97 dB at the 10 kHz Bessel cutoff, i.e. […]
- Protocols: Save as… stores exactly the CHAIN_KEYS of the form; an edit drops the combo to '— manual —' and recalling restores cutoff/notch list; saving onto an existing name asks 'Overwrite protocol?' (No → unchanged); […]
- Copy as Python on the -600 mV ABF: clipboard == returned script; run standalone it rebuilds the previewed window to 2.3e-4 pA of the preview (float32 rounding; […]
- 'Chain changed' chip and Revert: editing the cutoff after a Schmitt run shows '⚠ Chain changed' on the Results toolbar (actions revert/run/open), '⚠ Differs from the Schmitt run: low-pass 10000 → 8000 Hz' in the window with Revert enabled; […]
- Geometry/state: geometry, splitter, tab, time mode, response view and Follow are kept across close/reopen in the session and written at quit (signalConditioning/* keys), restored on relaunch (clamped to the 800×800 offscreen screen); […]
- Not in the territory but noted: the -200 mV '30-35' ABF is a 'CPYA'-variant the reader refuses by name (UnsupportedFormatError), so it could not serve as the short recording; the 2 s sweep-7 CSV was used instead.

**Results dock**

- Boot: real create_app, QSettings redirected to a scratch .ini, recovery root redirected, MainWindow(recovery_enabled=False), offscreen; the owner's -600/-1000 mV ABFs opened through open_paths; […]
- ⇧⌘R round trip (the QAction and the real Ctrl+Shift+R key on the window and on the table): hiding parks the focus on the TraceView, 'r' on the trace changes nothing, re-showing returns the focus to the QTableView, selectedRows and currentIndex survive, r/a/f curate the same row […]
- Dock height rule at (1512,900) (1440,900) (1280,800): dock == _results_default_height() == a third of the dock area (280/280/247), 4.36 rows at the offscreen 28 px row; […]
- Dragging the dock short: no height floor beyond the two header rows (minimumSizeHint 65 px), the Events page scrolls (max 66 at 120 px) instead of squeezing the table; Merge… open at the default height also scrolls (strip 130 px, scroll max 40) and moves no header rect; […]
- The stale line, every kind, measured painted text / full line / message / refusal / badges / popup: calibration ('⚠ Depths stale — calibration changed', Re-run; REMEASURED_COLUMNS badged), baseline knob ('⚠ Depths stale — estimator changed', Re-measure depths; […]
- The gated ('action unavailable') variant: voltage override -> '⚠ Depths stale · Re-run unavailable: Detection runs on CURRENT ch…' painted at a 532 px label; at 201 px '⚠ unavailable: Detection…' (reason first); […]
- Hidden-dock hints: 'The shown event table is now stale; the Results panel says why — View ▸ Panels ▸ Results (⇧⌘R).' once per stale transition (a second re-judgement says nothing), 'New events are in the Results panel, which is closed — …' on a detection landing behind the […]
- Summary / Fits / Quality tabs after Fit, Merge all (317→284 and 317→247, hand rejection carried, provenance n_events restamped 284 with merged{n_before,n_after}), Undo, second Undo, Redo, re-run and Undo of the re-run (summary '317 events · 314 accepted' and Quality fp=2 […]
- Merge strip: Merge… toggles the strip above the table and switches to Events, reads ON from any tab, Merge all / in-selection gated with a reason on the Reconciled table, stays open across a recording switch with the buttons disabled; […]
- Table selector among Schmitt / CUSUM / Reconciled: switching shows the stored table, 'CUSUM — no stored result' empties honestly, Reconciled refuses merge/geometry edits with the reason in the tooltip, R on a reconciled row curates it and Undo names it; […]
- Saved layouts v1: dock widths/height round-trip across a quit (file 250, inspector 700, results 350); a version-2 windowState is refused and the info bar names it ('1 saved preference(s) … windowState'), the layout opens at its defaults and the next quit writes a healthy version […]
- Docked Event inspector together with the Results dock: EventInspectorDock tabified with InspectorDock in the right column, the Results dock keeps 720x280 between the columns, table selection drives the docked inspector, R/A in the table curate, ⇧⌘R round trip unchanged, Details […]
- The Inspector keeps only the forms: AnalysisPanel's layout holds the title label and the DetectionPanel; the Detection page's main InspectorScrollArea viewport is 540 px (96 dpi) / 580 px (13 px font, in-window menu bar) at 1512x900 with no scrollbar — consistent with the […]
- Sample-rate territory as seen from the Results dock: a declaration on the parent moves a segment's clock (both 25000.0) and marks the segment's table stale; […]
- Tabified behind the File manager, the dock's toggle still hides/shows it and the focus round trip works; the stale hint is not raised there (Qt reports the dock visible).

**Event inspector docking, Window menu, curation speed**

- Harness: the real app (create_app, QSettings .ini redirected and cleared, NANOSCOPE_RECOVERY_ROOT / MPLCONFIGDIR / XDG_CACHE_HOME / incubation library under the scratch dir, MainWindow(recovery_enabled=False)) on a 1512x900 offscreen screen […]
- HOST MOVES (P01): Dock beside trace / Own window from the toolbar button and from Window ▸ Dock Event inspector beside the trace (both directions) move the SAME EventInspectorWindow (same id) and keep: […]
- REMEMBERED CHOICE (P01, P09): the dock's ✕ then Ctrl+I reopens docked with the state above; unticking the menu item while the inspector is closed opens nothing and writes 'window', and the next Ctrl+I is an own window with the same event/filter. […]
- MINIMUM SIZES AND RE-FLOW (P02, P03): own window minimumSizeHint 452x548 at the two-row width (Details shown) and 452x439 (Details hidden); at 452 px wide the five-row window's minimum is 452x681 (Details shown) / 452x572 (hidden) — the CHANGELOG's 'about 452x550' describes the […]
- DETAILS (P03, P14): starts hidden docked and shown in a window, remembered per host across moves and across a quit; the button reads 'Details ▸ / ▾'; […]
- KEYS DOCKED (P04, P14): A / R / Space / 1 / 2 and the custom hotkey x (then q after the palette changed while docked; x then dead) reach the docked inspector exactly once per press with the waveform, the Filter combo, the Context combo or the ▶ button focused, and write the […]
- UNDO / ⌘U PATHS (P05): ⌘Z and ⇧⌘Z in the own window and docked, Edit ▸ Undo with the keyboard in the docked inspector (via _undo_from_menu → request_undo), the gallery's ⌘Z on its view and its Undo button each reach MainWindow._undo_events and history.undo exactly once; […]
- BATCH AND LABELS (P06): inspector Reject all / Accept all in both hosts emit one curationBatchChanged, zero resultsChanged, set dirty and label the step 'Undo Reject all events' / 'Undo Accept all events'; the refusal note '1 event kept rejected: […]
- GALLERY PAGE / CURSOR (P07, P07b, P16): on page 3 with tile 6 selected, the page, cursor and selection survive the table's checkbox on another row, Edit ▸ Undo and Redo of it, an inspector label on the current tile and its undo, an inspector R on the current tile under Accepted […]
- WINDOW MENU (P08, P12, P17): menubar File/Edit/View/Analysis/Window/Help; one action ('Bring All to Front') before the first open; every entry opens or raises its window (Event inspector, Event gallery, Analysis Explorer and its page entries once pages exist, Figure Library, […]
- RESET WINDOW LAYOUT (P10): asks first (seam); the docked, Details-shown inspector returns to its own window at 1100x860 and stays open with its filter and event, Details defaults restored, the dock hidden, windows/* and signalConditioning/* keys removed; […]
- DOCKED THROUGH A SESSION (P13, P16): a second recording selected shows 'No detection — run detection to browse events.'; detection on it feeds the docked inspector; switching back restores the -600 table with its per-lineage refusal note; […]
- SUITE: the territory's own modules pass at HEAD — 177 tests in tests/gui/test_c_inspector_hosts.py, test_c_inspector_keys.py, test_c_inspector_layout.py, test_c_window_geometry.py, test_c_window_menu.py, test_sc_window_persistence.py, test_b_inspector_throughput.py, […]
- NOT TESTED / OUT OF REACH OFFSCREEN: window stacking (the gallery lowered under a docked inspector, 'Show in trace' lowering nothing), auto-repeat suppression through the platform shortcut map (only the keyPressEvent fallback is testable with synthetic events), the real […]

**Geometry, layout reset, quit, hand-add**

- Geometry adoption across displays (61cdea3), measured with real saveGeometry blobs written by the app on an offscreen 3840x2160 screen and read on 1512x982, and the reverse: […]
- A maximized main window saved on 3840 and booted on 1512 is maximized before show and after (1508x978, WindowMaximized), nothing named. (A full-screen main window saved on 3840 comes back as a normal 1512x982 window on the laptop: […]
- Genuinely damaged geometry bytes — b'garbage', b'', a wrong magic number, major version 9 — are named at every door: the info bar ('1 saved preference(s) held a value this app did not write and could not use … geometry'), the gallery's status line ('The remembered size and place […]
- Window ▸ Reset Window Layout at 1512x900 after perturbing everything (file dock 200, Inspector 700, Results 400 / floating at (200,300) / hidden via ⇧⌘R, Event inspector docked, gallery and SC moved): […]
- The close guard with unsaved work (recovery enabled, redirected): Cancel keeps the main window, the gallery and Signal Conditioning visible, the SC preview thread the same live thread, the 317-row detection intact; […]
- Quit with a detection running (`_analysis_controller.is_busy()` True) and Signal Conditioning open after a Compute exact: window closed in 0.35 s, `_teardown()` returned 0 in 0.04 s, the SC worker thread stopped, no detached threads, no hang, no zombie window in Qt's accounting […]
- Quit with the Event inspector docked: `windows/event_inspector/host`=dock written; the relaunch has an empty info bar, builds the inspector lazily and opens it docked at RightDockWidgetArea 460 wide, visible.
- Quit with Signal Conditioning open at (150,120,900,700), splitter 450/430, Spectrum tab, a preview pending: the relaunch reopens it at (150,120,900,700), 450/430, Spectrum, with no 'Ignored' line.
- File ▸ Quit (self.close()) with a floating Results dock and the gallery open: the dock comes back floating and visible on relaunch (3cc3f6b's LC-1 holds through the app's own Quit and the close box; only the native closeAllWindows path fails, S8-GEO4).
- Window menu: 'Signal Conditioning' listed (enabled, unchecked), its entry opens the window, checked while on screen, unchecked once hidden; 'Dock Event inspector beside the trace', 'Bring All to Front', 'Reset Window Layout…' present.
- Hand-added missed events on the real -600 mV recording (317 Schmitt events, IN15, t0 = 61.3 s on the instrument clock): a bracket in the 24,916-sample gap after event 0 adds event 317 (added=True, accepted=True, dwell 20.0 ms, baseline -43.40 pA, depth 3.80 pA), provenance […]
- 3cc3f6b's gallery claims on the real recording: with the Event gallery open a hand-add and its undo announce no tile (eventSelected never fires) and leave the trace x-range [61.834, 62.010] unchanged; Merge all at 10 ms (319 -> 249) with the gallery open and its undo ('Undone: […]
- Hand-add with the Results dock hidden: the status bar says 'New events are in the Results panel, which is closed — View ▸ Panels ▸ Results (⇧⌘R).' before the 'Added missed event' line, and the table holds the row (319) when the panel is shown again.
- Analysis ▸ Export events after two hand-adds writes 318 accepted of 319 with the 'added' column and both hand ids, plus the .meta.json sidecar; […]

**The time-unit declaration through the real app**

- Fixtures: a 2 s / 100,000-sample slice (t = 10 s on) of the owner's -600 mV ABF (IN15, pA) written as 'time_ms,current' (ms), 'time,current' in ms, 'time,current' in µs, 'time,current' in s, and HDF5 /time float64 in ms, in s and in µs — all under the scratch dir; […]
- File ▸ Open (open_paths, the one door) on each fixture, no Large-file box for 2 s: 'time_ms' header → 50,000 Hz · measured, t₀ 10 s, Duration 2 s, Note 'the time column 'time_ms' declares milliseconds in its header and was read as such: 50,000 Hz. […]
- The real SampleRateDialog on every fixture: header text, default mode 'file', unit defaulting to the header's unit (ms for time_ms.csv), previews for s/ms/µs/ns and for a rate ('Would read as 50,000 Hz, t₀ = 10 s — …'; […]
- After declaring ms on the bare-ms CSV: trace re-framed to [10.0, 11.99998] with the old zoom (10,500–10,520 's') and the old Zoom Back trail forgotten (trail [(10, 12)], ptr 0), pan scrollbar extent (10, 11.99998); […]
- Staleness: a detection made at 50 Hz (5,050 events) is judged stale on the declaration: amber line '⚠ Times stale — sample rate changed', Re-run offered with no refusal, full sentence 'The sample rate changed since this detection ran (50 Hz → 50,000 Hz): […]
- Edit ▸ Undo after the Re-run: 'Undo Merge events (5050 → 9)' restored the 50 Hz table (fs 49.99, 5,050 rows) and the rate-stale banner came back with the right columns; Redo returned the 50 kHz table and hid it. […]
- Detection under the declared rate vs the same slice in seconds: HDF5-in-ms declared ms → 10 events identical to the seconds CSV (start/end_idx exact; t_start_s, dwell_s, ecd_pA_s, rise_s, fall_s within 3.78e-11 relative, depth/baseline exact); […]
- Segments: a segment cut BEFORE the declaration (samples 30,000–40,000, t₀_abs 10,600 'at 50 Hz') followed the parent live (fs 49,999.99 · user, t₀_abs 10.6 s, note 'Derived segment of … (samples 30,000–40,000 · t₀=10.6s)'), File Info '50,000 Hz · user (follows the recording it […]
- Manual measurement rows: a row added at 50 Hz (start 30,000, end 40,000, dwell 200 s) is marked rate_stale after the declaration with the 'time base that has since been re-declared' tooltip, never rescaled; cleared on return to the file's own time base; […]
- Project round trip (real _write_project / _open_project): the declaration (mode time_unit, unit ms, note) saved under the workspace key, the archive holding only results/0.parquet, measurements.parquet, manifest.json; after close-all and reopen: parent 49,999.99 Hz · user (file: […]
- Two recordings open with one declared: the seconds CSV stays '50,000 Hz · measured', no declaration, its File Info and status bar unaffected while the other reads 'user'; Batch detect under the declared rate: […]
- Measure tool and declarations on a refused recording: the region placed on the sample axis (20,000–20,500) survives declaring µs with bounds unchanged and the region re-placed at 10.4–10.41 s. […]
- Not reproduced: one segmentation fault inside PlotsWindow.__init__ (plots_panel.py:2888) while opening Distributions with the Event inspector open on a 5,050-event 50 Hz table (scratch explore6); […]

**Cross-surface interactions**

- Which chain runs (s02_chains.py, -600 mV ABF): after editing the chain, Signal Conditioning's Run, the Detect button, the chip's own Run and Batch detect (Re-run all) each ran the FORM's current chain (provenance filter_cutoff_hz 5000 / 10000 / 2000 / 1000 Hz, 266 / 265 / 251 / […]
- Batch detect with CUSUM selected on the form and an edited chain: runs Schmitt with that chain (its tooltip and the 'already have a Schmitt detection' dialog say so), the selector stays on 'CUSUM — no stored result' and no false chip appears — disclosed, not reported.
- Merge all with the docked inspector (s03, s12): 317 -> 247 rows, provenance n_events restamped to 247 (S7-DET4 fixed) and merged {n_before 317, n_after 247}; the inspector is re-fed and resets to event 0 together with the table selection (both cleared, so no surface disagrees); […]
- Reconcile with the docked inspector (s03): sources Schmitt+CUSUM, the selector switches to 'Reconciled — 375 candidates', the inspector follows (reconciled, 375), the stale line's Re-run is gated with the reconciled refusal text, a chain edit while the reconciled table is shown […]
- Window menu and the study doors (s04): the launcher ('Incubation study'), the legacy 'Manual study' shortcut and the 'Original Zn reference recipe' are listed, checked while on screen, registered under their keys, re-placed by Reset Window Layout and written to […]
- Reset Window Layout with everything open (s04): Signal Conditioning re-placed to its default rect (1200x700 at 52,52 -> 796x700 at 2,98), the docked inspector returned to its own window keeping event 5 and the Unreviewed filter, the inspector dock removed, the Results dock […]
- Project round-trip with every new state (s06, same window and a fresh process): the ms declaration ('time column declared in milliseconds', rate label '50,000 Hz · user'), the ABF curation (248/251 accepted), the detection's chain (5 kHz) and the edited form (10 kHz -> '⚠ Chain […]
- Crash recovery (s07): flush wrote generation revision 10 under the redirected root; the relaunch offered it once (startup_recovery_checked True); […]
- Edit > Undo across the additions (s08, s10b): R in the docked inspector -> 'Undo Curate event'; the hand-add (Mark selection as missed event through the real Measure region) -> 'Undo Add event' (8 rows, added 1); […]
- Focus and shortcuts (s09): ⇧⌘R from the docked inspector hides/shows the Results dock and the keyboard stays with the inspector; R keeps curating with the dock hidden; a Tab walk from the docked inspector never lands in the hidden Results dock; […]
- Hidden Results dock + declaration (s05): the staleness is announced on the status bar ('The shown event table is now stale; the Results panel says why — View ▸ Panels ▸ Results (⇧⌘R)'), the line is there on re-show, Re-run replays the stored params under the new rate (prov fs […]
- Batch detect Re-run all over a sample-rate-stale table (s12, s13): the fresh Schmitt table (prov fs 50000) comes back with the stale line hidden; the untouched CUSUM table keeps '⚠ Times stale — sample rate changed'; the batch dialog names the stake.
- Closing the recording under an open Signal Conditioning window with an edited chain (s12): the window empties ('No recording selected — select a channel…'), Run is disabled, the chip and the Detection card sentence clear, the form keeps its edit (7000 Hz).
- Quit with the inspector docked, SC open and the Results dock hidden (s12): windowState, geometry, windows/event_inspector/{host,details_dock,details_window} and signalConditioning/{geometry,splitter,followMain,tab,timeMode,responseView} are written.
- The study windows the launcher creates do not crash the Window menu, Bring All to Front or Reset; only their absence from them is the defect (S8-XS4).

**Documentation against the live widgets**

- examples README 'Setup' numbers on the real −600 mV recording through the real widgets (open_paths → measure → CUSUM k 0.75 / h 28 / 1 ms / 0.5 ms / 5 kHz Bessel 8): […]
- Results dock (CHANGELOG/§1/inventory): View ▸ Panels = File Manager / Inspector / Results with Ctrl+Shift+R; first-launch File dock 320 px, Inspector 460 px; […]
- Window menu (CHANGELOG/inventory/README §7): titles File/Edit/View/Analysis/Window/Help; entries Bring All to Front (no shortcut), the main window by title (checked), Event inspector / Event gallery / Analysis Explorer / Figure Library / Signal Conditioning (checked while on […]
- Event inspector docking (CHANGELOG/§9/inventory/README §6): opens 1100x860 titled 'NanoScope — Event inspector'; minimumSizeHint 452x548 in a window and 452x440 docked; toolbar two rows wide / five rows docked; waveform floor 180 px; […]
- Signal Conditioning (CHANGELOG/§5/inventory/manual/README §2-4): Ctrl+Shift+F, the card's Open button and the Window menu open the one window (closing hides it, settings stay); title 'Signal Conditioning — <file> · IN15'; […]
- Time base (132b42b: CHANGELOG/§2/§11a/§14) on fixtures cut from the −600 mV IN15 channel: time_ms / 'Time (ms)' / t_us / time_s headers read in their unit (50,000 Hz measured, t₀ 10 s, a Note '… declares milliseconds in its header and was read as such: 50,000 Hz…', no Issue); […]
- §13 keyboard table (and §9's key table, README §5): every main-window shortcut binds as listed — Ctrl+O/S/Shift+S/Z/Shift+Z/D/I/G/M/Shift+M/Shift+F/0/E/Shift+R, Ctrl+←/→ and Alt+←/→ (Close document binds QKeySequence.Close, which is Ctrl+F4 under the offscreen theme and Ctrl+W […]
- Example project and inventory quotes: outputs/apo-ferritin-new-features.nanoscope opens on the recording at its original path with the 650-row CUSUM table, the palette shape=single-level(1)/shape=multi-level(2)/quality=artifact(3), A on 217 (dI/I0 2.24), 1 on 218, 2 on 219, 225 […]
- Not measurable offscreen and not claimed: multi-display placement in a live desktop session (README §7 'Not shown here' already says so), 'Show main trace' window stacking, the 24 px Mac row height (the formula holds at 96 dpi: […]

**Lens round — reconstructed from the finders' transcripts** (the four lens finders were cut off before reporting, so this is the coordinator's reading of what their probes printed, not their own account):

- **Data safety and persistence.** A time-unit declaration, a hand-added event and a batch of labels came back identical after a project save and reopen, and after a simulated crash followed by recovery: the reopened 319-row table (2 hand-added, 18 reviewed, 7 labelled) and the recovered 320-row table matched their pre-save states cell for cell.
- **Data safety and persistence.** Older builds are safe readers of a project that carries the declaration. The build at the end of sweep 7 and the first build with the extensible workspace both opened it, read the recording at its file rate, kept the table with its stale marker, and wrote the declaration back unchanged on their own save. The current build read it back intact.
- **Data safety and persistence.** A segment follows its parent's declaration (50 Hz to 50 kHz) and its detection is marked stale. Both came back after a project round trip. An HDF5 recording with a microsecond clock kept its declaration through save and reopen, and its detection was not stale afterwards.
- **Data safety and persistence.** Re-running a detection over curated work asks first and names what is at stake (2 rejected, 1 hand-added, 3 labelled, 6 reviewed). Undo restored the curated table exactly and Redo brought the re-run back.
- **Data safety and persistence.** Fifteen different curation gestures across the table, the inspector, the gallery and hand-add each undid as one step and was named in Edit ▸ Undo. After a project reopen, hand-add, label, accept-all, Merge all and re-run walked back to the restored table and forward to the final one, with no column changing type.
- **Data safety and persistence.** Undo and redo from any of the three curation surfaces kept the user's place (inspector event, gallery selection, filter and page); only a hand-add moves it, to the new event.
- **Data safety and persistence.** Closing a recording emptied the inspector and the gallery and a decision aimed at its table was refused. Switching projects left no label palette or manual measurements of the previous project behind. Manual measurements went stale on a declaration, stayed stale through save and reopen, and recovered when the declaration was removed.
- **Data safety and persistence.** The protocol library round-tripped 14 protocols through the settings store and the export file, including names with a leading space, `@Variant(x)`, brackets, a backslash, a newline, 300 characters and non-ASCII. CSV time headers in ms, µs (micro sign and Greek mu), ns and with spaces gave the right unit, and the open-time probe and the load gave the same rate, start time and unit sentence (the probe only adds its sample-count estimate). The example project under `examples/` (650 events) restored with no notes, re-saved with exactly one new manifest key and identical results data. The window layout did not drift over four relaunches. After every scenario the folder holding its input was unchanged.
- **Concurrency and lifetime.** Quitting during four kinds of running job (a Signal Conditioning exact pass, a pending preview, a closed window, a detection) drained to zero with no live thread left and no uncaught exception. Rejecting events while a re-detect ran was not lost silently: when the new table landed the status said "Detection replaced the previous table (1081 events, 6 rejected by hand). Your curation is not carried over — Edit ▸ Undo puts it back", and Undo walked back through the re-run and all six rejects. A detection replaced by the baseline probe left the table untouched and no phantom undo step.
- **Concurrency and lifetime.** Across the storms that finished (see the crash lead for those that did not), the Signal Conditioning source equalled the panel's source, the last result carried the current chain and source, no job was left behind, and the quit drain was clean. Over ten edit storms of 120 to 200 random edits each, the displayed result always matched the current chain, source and range and equalled a recompute (maximum difference 0).
- **Numeric honesty.** The Time tab's exact window equalled the whole-recording conditioning for 15 chains (largest difference 3.4e-4 pA, on a 20-line notch), at both ends of the recording and for windows down to 16 samples. Stages that use whole-recording statistics are labelled "Window estimate", and they do differ from a script by up to 12 pA, as labelled. Copy as Python reproduced the window exactly for exact stage sets, calibration included.
- **Numeric honesty.** The low-pass response curves matched the response measured on the real kernel to 0.03 dB and the high-pass curves to 0.55 dB. Group delay, overshoot and pre-shoot matched an independent SciPy computation for the Bessel and Butterworth chains, and rise time agreed to within 3 µs (a sample is 20 µs). The notch curves differ from the measurement at isolated frequencies by up to 7 dB and were not pursued beyond S8-NUM2, S8-NUM3 and S8-NUM7. The "true −3 dB corner" the form prints was within 0.05 % of a root search on 168 designs.
- **Numeric honesty.** The Summary tab's figures matched an independent computation from the results table after a detection (266 events, 265 accepted, 4.426 Hz, median dwell 18.56 ms, median depth 87.58 pA). Auto polarity from 10 s windows agreed with the whole-recording answer in all 93 windows of three recordings. A ranged load of a 50-million-sample ABF reported the right start time and duration. HDF5 time-base declarations in ms, µs and s gave the right rate, start and duration, windowed loads included. Three of the owner's events tables (2,088 events) never printed a dwell in scientific notation.
- **Keyboard and focus.** Trace-focused navigation behaved consistently (Shift+arrows pan and zoom, Home fits, ⌘←/→ walk the zoom history, ⌘0 fits). ⇧⌘R hid the Results dock to the trace and brought the keyboard back into the table, and left a spin box that had the keyboard where it was. A docked inspector took the keyboard when clicked anywhere, and with the inspector behind its tab, R changed nothing (316 accepted before and after).
- **Keyboard and focus.** ⌘D pressed repeatedly during a running detection replaced the run each time with a one-line status and no dialog, and one detection finished (317 events). ⇧⌘M over an existing event was refused with a sentence naming it; in a gap it added the event without moving the trace (the selection is cleared), and Undo removed it. The Window menu's entries enabled, ticked and cleared with the windows they name (S8-XS4 is the exception) and with the Analysis Explorer's pages.

