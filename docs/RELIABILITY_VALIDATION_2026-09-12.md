# Reliability first-pass validation — 2026-09-12

Base source revision: `812ae96d4c25a5e86255917baf6cf613251857a2`.
Implementation remains uncommitted source; no app/DMG/PDF rebuild, installation or push
was performed. Earlier manual changes in the working tree were preserved.

## Delivered scope

- Additive baseline/polarity/intrusion advisories for Schmitt and sequential CUSUM,
  including zero-event and explicit-reference cases. Quality is separate from acceptance.
- Actual quantitative-domain quality reassessment in the high-pass adapters, with
  automatic stale marking when event measurements/reference metadata subsequently change.
- Per-channel ABF acquisition declarations, explicit user overrides, immutable original
  header evidence, and detection-time snapshots. ML receives bandwidth provenance but
  does not gain the new classical-detector quality assessment in this pass.
- File Info metadata editor and project persistence, including channel identity remapping,
  segment inheritance/reset, unreadable metadata preservation/refusal, and exact editable
  cutoff/order round-trip. No signal filtering or new hardware resolution rule.
- Explicit live/snapshot heatmap and spectrogram image-axis mapping. Counts, numerical
  grids, edges, marginals, event numbers and existing fit-selection defaults are unchanged.
- All new advisory messages remain outside plot artwork, in Summary/File Info/provenance.

## Test evidence

Tests used `/opt/anaconda3/bin/python`, `QT_QPA_PLATFORM=offscreen`,
`PYTHONDONTWRITEBYTECODE=1`, isolated Matplotlib/cache directories, and
`pytest -q -p no:cacheprovider`.

Final combined targeted gate: **154 passed in 22.36 seconds**:

```text
tests/analysis/test_measurement_quality_advisory.py
tests/core/test_acquisition_bandwidth.py
tests/core/test_acquisition_state.py
tests/io/test_abf_acquisition_bandwidth.py
tests/gui/test_acquisition_workflow.py
tests/gui/test_event_heatmap_coordinates.py
tests/gui/test_spectrogram_coordinates.py
tests/gui/test_measurement_advisory_details.py
tests/gui/test_warnings_surface.py
```

Broader scientific regression gate: **212 passed in 136.41 seconds**:

```text
tests/analysis/test_open_pore_reference_integration.py
tests/analysis/test_baseline_occupancy_reference.py
tests/analysis/test_event_amplitude_lifecycle.py
tests/analysis/test_remeasure.py
tests/analysis/test_remeasure_install_seam.py
tests/analysis/test_detect_cusum.py
tests/analysis/test_cusum_wiring.py
tests/analysis/test_cusum_refit_inputs.py
```

Independent overlapping agent suites also passed: 200 detector/QC/reference checks;
144 acquisition/core/reader checks; 42 event-plot/export checks; 57 PSD/spectrogram/export
checks; and 18 Summary/advisory checks. These counts overlap and must not be added as a
unique total. Offscreen visual inspection confirmed readable dialogs, File Info and
current/stale Summary messages without clipping. `git diff --check` passed.

## Existing failures reproduced before the change

A broader project/flag run produced 99 passes and three failures in
`tests/gui/test_project_io.py`:

```text
test_a_cusum_level_table_survives_save_and_reopen
test_a_restored_level_table_is_REACHABLE_from_the_menu
test_a_level_table_retired_by_a_re_detection_is_not_persisted_at_all
```

All three reproduced independently on a clean `git archive HEAD` checkout at the base
revision (3 failures in 21.39 seconds), before this implementation. Their asserted level
fit was absent at the check. They were not suppressed, rewritten or claimed fixed here.
This is not a claim that the whole application test suite is green.

## Scientific interpretation limits

The changes disclose uncertainty; they do not automatically improve detection counts,
repair biased amplitudes, establish the true open-pore level, or validate all difficult
recordings. Numerical preservation is covered by regression checks, including exact
event-frame comparisons with and without the advisory path. The external audit's full
seeded benchmark was not reproduced in this pass.

A reported acquisition cutoff is metadata, not a transfer-function measurement. Hardware
temporal resolution remains explicitly unassessed, and `filter_limited` keeps its previous
software-conditioning meaning. ABF1 hardware cutoff usually remains unknown with the
currently installed Neo parser. See [the usage addendum](MEASUREMENT_RELIABILITY.md).
