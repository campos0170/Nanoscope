# Manual revision — 2026-11-11

This edition describes installed NanoScope **46.31.52**, build ID
`20261111T163237642239Z-36aa140fa2b2-dirty-8c6f5e37d549`, built from commit `36aa140`.
Source SHA-256: `8c6f5e37d54985a5869391c104f783eb55b29b1ec3b29bf91492254c2894209c`.
Executable SHA-256: `85e81d18dc66b6caf128273931335b83f342f4da32a63526716aedbf2ecb38e1`.
The build id's `dirty` marks the application icon that every build regenerates; no source
file differed from the commit.

The installed build passed its in-bundle self-test (22 of 22 checks) from `dist/` and again
from `/Applications`. The source behind it passed the whole test suite, one process per file,
with no failing test that the previous commit did not also fail. This revision updates prose;
it does not re-run the scientific chapters or re-capture illustrations.

## Changed instructions

- **CUSUM noise firings:** a CUSUM run counts its own detections in the opposite direction
  and adds a short advisory when that count is material; with High-pass on the advisory says
  the count is not a noise estimate. Raising *h* or *k* is the remedy.
- **Validation studies follow the detector:** Null control runs the selected detector and its
  readout names it; Sweep threshold and Recovery map are Schmitt-only and are greyed out with
  CUSUM selected; a recovery map never corrects a CUSUM table's dwell fit.
- **Batch detect:** runs whichever of Schmitt or CUSUM the detector box shows (ML stays out);
  each recording then shows the table the batch made; a second batch is refused while one runs.
- **Noise page:** a Signal choice, Raw (the default) or Processed (the Detection panel's
  Preprocess chain, Normalize left out); the result names its basis, and a later chain edit
  marks the figure for an update.
- **Run buttons:** Detect, Apply and the Preprocess workspace's Run detection… show ● while
  they have edits they have not run; Ctrl/⌘+Return presses the run button of the panel you are
  in. The guide opens with the two rules for changes.

## Illustrations and retained evidence

No screenshot was added or replaced. Every illustration keeps the build and date in its
caption (most read 45.37.98, the 2026-11-05 edition); controls added since then, such as the
Noise page's Signal row and the ● on run buttons, do not yet appear in a picture. The
2026-11-05, 2026-11-03, 2026-09-29 and 2026-09-11 revision notes and capture manifests are
unchanged.
