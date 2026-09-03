<!-- GENERATED from nanoscope.analysis.remeasure.release_note(); do not edit by
     hand. `tests/analysis/test_remeasure.py` asserts the two are identical, so the
     wording and the behaviour can only change together. -->

# Release note — open-pore baseline correction

Attach this to CSVs exported before the change. The app cannot repair them: it does
not own them, and the estimator that produced them is not recorded in the file. From
this version on, every `Analysis > Export events...` writes a `<name>.meta.json`
sidecar naming the estimator, so the question is answerable from the files.

```
NanoScope v0.1.0 -- open-pore baseline correction

CSVs exported before this version carry a BLOCK-MEDIAN baseline measured over a
fixed 10 ms window. On recordings whose events are long compared with half that
window -- which includes bursty solid-state nanopore data, where more than half
the samples in a 10 ms block can be inside an event -- that median lands INSIDE
the blockade rather than on the open pore, so the amplitude columns are
understated.

Affected columns: baseline_pA, depth_pA, frac_depth, ecd_pA_s, mean_dev_pA,
max_dev_pA, snr, rise_s, fall_s.

Unaffected: event_id, recording, pore_id, start_idx, end_idx, t_start_s, dwell_s,
n_levels, filter_limited, accepted, flag. Detection geometry and curation are
unchanged BY A RE-MEASURE -- but read the next section before comparing a
re-measured table with a fresh one.

Measured on a 22 nm pore, 1x PBS, -600 mV apo-ferritin recording (2,993,798
samples at 50 kHz), 1641 events detected at min_dwell_s = 1e-4 and re-measured
against the same spans:

    baseline_pA   -120.87  ->   -35.21 pA   (true open pore -34.8 pA)
    depth_pA        23.70  ->   110.17 pA   4.65x  (per-event median 4.73x)
    frac_depth      0.190  ->    3.116      16.4x
    ecd_pA_s        0.0106 ->    0.0501     4.72x
    mean_dev_pA     24.55  ->   112.24      4.57x
    max_dev_pA      41.18  ->   130.92      3.18x
    snr              5.76  ->    26.76      4.65x

1636 of 1641 events (99.7%) had their stamped open-pore reference move by more
than 10 pA.

rise_s and fall_s are measured at 10% and 90% OF EACH EVENT'S OWN DEPTH, so a
depth that moves 4.7x moves them too: 1243 and 1236 of the 1641 rows change, and
about 1000 of each become NaN. That is correct, not a loss -- see the next
section for why those rows had no measurable transition in the first place.

On recordings without this pathology the same columns move by ~1.0x. The app
cannot tell the two apart from an exported table alone, which is why this note
exists: re-measure the recording in NanoScope v0.1.0 or later and compare, or
use Analysis > Compare baseline estimators to see the factor for your own file
before deciding.

RE-MEASURING IS NOT THE SAME AS RE-DETECTING, AND ON THIS FILE THE DIFFERENCE IS
LARGER THAN THE DEPTH CORRECTION
-------------------------------------------------------------------------------
A re-measure keeps every event boundary exactly where it was, because that is
what protects your curation. But the old baseline did not only mis-stamp the
depths -- it also fed the detector's threshold, and a reference that dives into
each blockade rises and falls WITH the event, so a single long blockade was cut
into a string of short fragments. On the same apo recording, same detector
settings, detected fresh under each estimator:

    block median, 10 ms (before)   1641 events   median dwell   0.44 ms
    block median, 200 ms            327 events   median dwell  17.48 ms
    masked median, 0.5 / 0.1 s      315 events   median dwell  17.56 ms
    masked median, 2.0 / 0.5 s      275 events   median dwell  18.18 ms
        (2.0 / 0.5 s is what the Preprocess tab's probe proposes for this file)

Total blocked time goes 1.80 s -> 4.60 s. The event COUNT falls 5x and the median
dwell rises 40x, and neither of those is a mask or a merge artefact: the 200 ms
block median, which uses no event mask and no dilation at all, already gives 327
events at 17.48 ms.

So: a re-measured table has correct depths against dwell times that were measured
by the estimator being replaced. For a depth histogram that is fine. For a
dwell-time fit, a dwell-vs-depth scatter, an event rate, or ecd_pA_s -- all of
which integrate or bin over the span -- RE-DETECT the recording rather than
re-measure it, and expect a different number of events.

Exports from v0.1.0 onward write a <name>.meta.json sidecar naming the
estimator (baseline_method, its windows, and a baseline_fingerprint), so this
question is answerable from the files themselves from now on.
```

## What to do with an old export

1. Reopen the project (or re-open the recording and re-run the detection) in
   NanoScope. Old projects load unchanged; the results list marks them stale and the
   nine affected columns are badged in the event table.
2. `Analysis > Compare baseline estimators...` measures, on that recording, exactly
   what a re-measure would change: old vs new `depth_pA` per event, the ratio
   distribution, and how many events have a stamped baseline that moves more than
   10 pA. Nothing is modified by looking.
3. If you accept it, the banner's **Re-measure depths** button rewrites the amplitude
   columns only. Events are not re-detected, so every span, accept/reject and
   hand-written flag survives.
4. Export again. The new CSV's sidecar names the estimator and, for a re-measured
   table, records which estimator it came from.

## What does not change, and why that is the catch

`start_idx`, `end_idx`, `t_start_s` and `dwell_s` are bit-identical before and after
a re-measure. That is what protects your curation, and it is also the limitation:
on the apo recording the old estimator got the dwell times more wrong than the
depths. Detected fresh with identical detector settings, the same trace gives

| estimator | events | median dwell |
|---|---|---|
| block median, 10 ms (before) | 1641 | 0.44 ms |
| block median, 200 ms | 327 | 17.48 ms |
| masked median, 0.5 / 0.1 s (new default) | 315 | 17.56 ms |
| masked median, 2.0 / 0.5 s (probed for this file) | 275 | 18.18 ms |

A baseline that dives into each blockade rises and falls with the event, so the
Schmitt trigger re-crosses inside it and one 17 ms blockade is emitted as a string
of 0.4 ms fragments. Total blocked time goes 1.80 s → 4.60 s. This is not a mask or
a merge artefact: the 200 ms block median uses no event mask and no dilation at all
and already gives 327 events.

**So a re-measured table is right for a depth histogram and wrong for anything that
uses the span.** Dwell-time fits, event rates, `ecd_pA_s`, and the depth-vs-dwell
scatter all need a fresh detection, not a re-measure — and a fresh detection will
return a different number of events.

## Re-detecting no longer costs you your curation

The paragraph above tells you to **re-detect rather than re-measure** for any dwell
fit, event rate or ECD. Until this build, that was also the single most destructive
click in the application: re-running detection replaced the whole event table, and the
measured result on a six-event curated table was *two hand rejections gone, zero
dialogs, nothing in the status bar*. There was no undo anywhere in the program.

There is now. **Edit ▸ Undo (⌘Z)** walks the event table on screen back through every
change made to it in this session — a re-detection, a merge, an accept/reject, Accept
all / Reject all, an Event-Inspector mask resize, a hand-added missed event, a
re-measure — and the menu names the step it will undo ("Undo Re-run detection").
Redo is ⇧⌘Z. Verified on the apo `_0006_` recording: detect → curate → merge at 5 ms →
resize a mask, then three Ctrl+Z, and all 21 columns come back bit-identical at each
stage, including the hand-written `flag` token and the hand-added marker.

Two things worth knowing:

* **A re-detection over curated events now says so.** The status bar reports what the
  replaced table held ("6 events, 2 rejected and 1 hand-added") and points at Undo. It
  is deliberately not a modal — re-detecting is the recommended move, and it is now
  recoverable.
* **Closing a recording is the one thing undo cannot reach**, because a closed document
  takes its results and its history with it. So closing a recording that carries
  detected events now asks first, and names them. For a **segment** it says the thing
  that matters: a segment is held in memory, not on disk, so there is nothing to reopen
  and nothing to detect again. Save the project first.

The history holds up to 30 steps PER EVENT TABLE, within a 128 MiB budget for the whole
session, whichever binds first, and is dropped when its recording closes or a project is
opened. On the apo recording a step costs 126 KiB;
twenty operations measured +1.9 MB of RSS.
