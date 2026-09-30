# Manual edition: 29 September 2026

This edition documents the unified incubation workflow in NanoScope build
**40.15.59**. The main manual, User Guide, incubation companion, control inventory,
HTML and PDF are maintained together. Recording-specific Zn/Cu guides retain
their original numerical recipes and dated scientific examples.

The update covers one **Analysis > Incubation study…** entry for General, Zn, Cu
and Acid; new-study profile defaults and saved-method routing; editable study
metadata for General/Zn/Cu; ordinary main-workspace curation; and the explicit
**Save curation to study > Return to study > selection confirmation > dependent
recalculation** sequence. It distinguishes the Acid calibrated method from the
general workflow, and the advanced original Zn reference recipe from reviewed
staged analysis. Profiles do not infer chemistry from filenames or overwrite
saved scientific settings.

Older main-analysis chapters retain their 11 September 2026 source review at
revision `703a96a`, unless another date is stated. That audit's warnings are not
assertions that every unrelated behavior was retested in build 40.15.59. Earlier
screenshots and historical apo-ferritin, Zn and Cu observations remain dated;
they have not all been recaptured or rerun. No new biological validation,
physical calibration or detector-performance claim is made by this edition.

New incubation interface illustrations, when included, are identified in their
captions and capture manifest. They demonstrate navigation and curation controls;
their displayed sample data do not constitute experimental validation.

The main PDF is rendered offline from the verified self-contained HTML with
ReportLab. The build checks image embedding, internal anchors and contents links,
and records source/output hashes in `manual-build-manifest.json`. A separate
rendered-page review checks the updated title, incubation chapter and control
reference. Updating documents does not rebuild an application executable.
