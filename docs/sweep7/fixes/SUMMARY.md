# Sweep 7 — the fix batch, for Abraham (2026-09-01 → 09-02)

## What you asked for, and what happened

You said "apply the fixes from sweep 7". All 24 confirmed findings are fixed, plus the 8 the
skeptics had found and the register never assigned to anyone. Then the fixes themselves were
attacked: 11 skeptics re-drove every one through the real widgets on your own recordings, found
53 new defects inside the repairs, and 50 of those were fixed and independently re-verified.

**30 source files changed. 327 new tests in 54 files. Suite: 7174 passed, zero failures**
(3636 non-GUI + 3535 GUI + 3 lifetime; it was 6793 before).

**No number moved.** The 20 detected columns hash identical on the −600 mV file at Detect, after
Merge all, and after a probe-confirm-re-measure. The four exported figure CSV sidecars are
byte-identical. A dedicated skeptic did nothing but hunt for a moved byte and found none.

## The five that matter most to your papers

1. **A rejected fit no longer leaves the app dressed as a measurement.** At the app's own defaults,
   on its own shipped dwell template, the exported figure printed `τ = 20411.6 [17554.2, 25035.2] µs`
   for a curve the app's own KS test rejects at p = 1.6e-39 — an interval is a stronger claim than a
   point estimate. The verdict now appears in the plot title, the legend and on the exported PDF at
   every switch setting, and the confidence interval is withheld for a rejected model.
2. **The exported scatter no longer calls your enhancement events "blockade depth".** 236 of 275
   events on the −600 mV recording have dI/I0 above 1, meaning the current magnitude GREW. The depth
   histogram already said so; the scatter, the figure most likely to be a paper's Figure 1, did not.
3. **"Detector score" stopped reporting the detector's own rejections as your false positives.**
   With a 5 ms max-dwell it read "precision 0.243, F1 0.391" after zero curation gestures. It now
   counts only real user gestures and shows the amber caption when there are none.
4. **Batch detect stopped claiming you measured recordings you never opened.** It stamped one
   recording's confirmed open-pore windows onto every other as "proposed" and switched off the
   "never measured" advisory. There is now a distinct `batch` source that says the windows were
   copied, and the advisory fires for it.
5. **Merge and Re-measure stopped lying about counts.** After a merge the report said "71 of the
   detector's 317 regions were curated OUT" when you had curated one; a re-measure silently deleted
   the filter-limited and boundary-clipped advisories, which then reached the voltage-dependence
   report. Both now re-derive every row-derived count and rebuild the warnings from the rows.

## Everything else, briefly

- **Calibration override**: "Treat this channel as: current" now means the same thing everywhere.
  The Detect button, Batch detect, PSD, I-V, the workflow strip, the calibration chip and the
  Calibration panel's own conductance boxes all resolve the channel kind through one function in
  the core. Previously the app both refused the work and did the work.
- **Curation surfaces**: a multi-tile gallery selection survives curation made anywhere else;
  selecting an event the gallery's filter excludes no longer leaves a different tile armed for the
  R key; both windows carry the events table's stale-column badge; the subpopulation assignment
  survives navigating to another recording and back, and the report's buttons give the true reason
  when it was retired.
- **Notch lines**: pasting a spreadsheet column no longer truncates at 32,767 characters mid-number
  and invents a 54.0 Hz notch. Capped at 32 lines like the ML box, with typos refused by name.
- **Calibration panel**: min/max/span are now exact over the whole channel (on your −500 mV file the
  max had the wrong sign), a gain that would overflow float32 is refused before Apply, and Apply on
  a recording whose file has moved no longer claims to have saved beside it.

## What is still open

- **The manual and the Control reference are stale** — you said to skip them. `manual.md` still
  describes "Treat this channel as" as a layout-only override and still quotes five surfaces this
  batch rewrote. USER_GUIDE.md IS current.
- **Two documentation-only items** (CW-4, X-SURF-3/8) are carried in the register for that reason.
- The pre-existing carry-over list from the fixers is at `docs/sweep7/fixes/CARRYOVER.md`.

## Where everything is

| what | where |
|---|---|
| The register of findings | `docs/sweep7/FINDINGS.md` |
| What each agent changed | `docs/sweep7/fixes/reports/*.md` |
| Every verdict, defect and re-verification | `docs/sweep7/fixes/VERIFICATION.md` |
| Runnable repros | `docs/sweep7/repros/` (exit 7 = reproduced, 0 = fixed) |
| The pre-fix tree, for comparison | `~/Desktop/nanoscope-reports/pre-sweep7-fixes-2026-09-01/` |
| The Aug 31 build, preserved | `~/Desktop/nanoscope-oracle-2026-07-28/NanoScope-aug31-build` |

## Housekeeping

The junk calibration sidecar a sweep agent left beside your −500 mV recording is gone, and the
identity sidecar for the −1000 mV excerpt was restored to `gain 1.0 / offset 0 / pA`. Your home
directory holds exactly two sidecars: that one and your own Aug-24 nisin file. Every agent in this
batch verified that before finishing.

**Still not in git.** This tree has no version control, and two Claude sessions have edited it.

## The rebuilt app (2026-09-02, 04:08)

`dist/NanoScope.app` (1.1 GB) + `dist/NanoScope-0.1.0-arm64.dmg` (347 MB), ad-hoc signed,
`LSMinimumSystemVersion` 15.0 matching what the binaries actually need.

- **Self-test: 18/18 passed** on your real −1000 mV ABF and `student_unet_teacher_best.keras` —
  including 115 events end to end, 60 ML events on 500 k real samples surviving teardown, and both
  export paths (figure PDF/PNG/CSV, events CSV 73 rows × 22 cols).
- **Bundle vs the Aug 31 build: 96 modules both, 30 changed, 66 byte-identical, none added or
  removed** — exactly the 30 source files this batch touched, so nothing shipped from stale source.

`/Applications/NanoScope.app` **is now this build** (sha `2b845f32…`, was `e665b121…`), quarantine
cleared, signature valid, `LSMinimumSystemVersion` 15.0, and its installed copy passes the same
18/18 self-test. The Aug 31 app it replaced is preserved whole at
`~/Desktop/nanoscope-oracle-2026-07-28/NanoScope-aug31.app`, so the swap is reversible:

```bash
rm -rf /Applications/NanoScope.app && ditto ~/Desktop/nanoscope-oracle-2026-07-28/NanoScope-aug31.app /Applications/NanoScope.app
```

Double-clicked `.abf` files now open in the new build, since LaunchServices points at that path.
