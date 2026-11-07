# Manual revision — 2026-11-03

## 2026-11-04 integration correction source update

The manual and control inventory now describe Snapshots/compositions inside the Explorer's single
Figures list, selected-entry actions and frozen exports, figure-owned sources and pending states,
the Distributions **Import plot settings** route, shared Format editing in Study/Report, computer
styles and the All live scope's open retained editors. An embedded saved-figure capture replaces
the old separate-Library illustration. New integration captures are labelled with their synthetic
fixture scope; the composition example retains its frozen apo-ferritin provenance.

The final integration capture report passed all ten recorded cases with unchanged inputs. Its
approved source-tree fingerprint is recorded in `manual-design-connections-captures.json`; the
seven-slot capture plan is complete with nine installed images. The HTML/PDF/ZIP build includes
these assets, the final Follow selection instructions and the 4 November PDF title/footer.
Generated-output hashes belong to `manual-build-manifest.json`; visual QA and source acceptance
remain distinct from scientific validation. Historical Phase 4 provenance below is unchanged.

## Original Phase 4 edition record

This edition describes NanoScope’s Phases 1–4 development source and the implemented reference
workflow. It is not a new released-build claim. Final Phase 4 assets were approved from validated development source `403311a6cb44a7d250f4487331acb4a77c84ed0a`.
The eight required capture slots and their source/input hashes are recorded in
`manual-phase4-captures.json`. Captures use the Qt offscreen platform; the user screen
was not used. Generated HTML/PDF/ZIP and their source hashes are recorded separately in the build manifest.

The updated chapters cover independent project Figures, v13 conversion/backups, the common
Data & analysis / Format interface, New/Duplicate and recipe wording opt-in, pinned project/local
styles and their three application scopes, per-series appearance and project source defaults,
frame/palette/physical size and synthetic preview, millisecond display with canonical stored units,
Snapshots/compositions and panel-letter styles, and serial selected-figure export without analysis.
The established scientific quantity-change reset rules, calibration caveats and source identities
remain. Raw seconds, sample indices, fit parameters and stable CSV keys are not renamed to imply ms.

The final static cross-check also corrects older companion-guide paragraphs that still described
tool tabs, selectable time units, noneditable Comparison axes and outlined Matplotlib exports.
Historical screenshots, old scientific readouts and the composer's actual Save to Figure Library
button retain their dates or exact labels; current Explorer capture uses Snapshot to Figures.

The incubation chapter retains its 2026-09-29 review against build 40.15.59; other earlier chapters
retain their 2026-09-11 review (703a96a) unless dated otherwise. Historical apo-ferritin counts and
figures keep their original recipes and do not establish detector performance or verified gain.
Older illustrations are explicitly dated or marked as earlier layouts. Current field-unit labels
and navigation take precedence over an older screenshot's control location or time unit.

## Source mapping

| Source | Revised content | Implementation reference |
|---|---|---|
| `manual-markdown/manual.md` | Main navigation, independent figures/recipes, ms contract, styles/series, Snapshots/composition, single/batch export; retained historical examples qualified | `gui/analysis_explorer.py`, `figure_list.py`, `figure_workspace.py`, `figure_recipes.py`, `format_inspector.py`, `figure_composition.py` |
| `USER_GUIDE.md` | Companion lifecycle, v13, units, Format/style/recipe routes and export behavior | Same common interface plus `core/quantity_display.py`, `figure_quantity_basis.py`, `gui/time_controls.py` |
| `inventory.json` | New project-figures area; fixed-ms controls, actual style scopes, source defaults, explicit dimensions/export and corrected legacy entries | `plot_styles.py`, `project_style_editor.py`, `project_sources.py`, `figure_batch_export_dialog.py`, `plot_export_options.py` |
| `gen_manual.py` | New reference-area title; no new screenshot assumed | Inventory renderer only |
| `gen_page.py` | Phase 1–4 footer/build provenance; final capture manifest plus completed asset-plan packaging/hashes | No rendering performed by this source change |
| `BUILDING-THE-MANUAL.md` | Source/capture freeze, rendering and visual-QA handoff | Existing `gen_page.py --pdf --zip` pipeline |
| `manual-phase4-capture-plan.json` | Completed eight-slot asset checklist with installed paths and hashes | Actual captures/provenance are recorded in `manual-phase4-captures.json` |
| `CHANGELOG.md` | User-facing Phases 3–4/full reference behavior | Development changes, not a new tested release tag |

## Capture and build handoff

The completed asset plan and `manual-phase4-captures.json` identify
the validated source revision, actual file paths/hashes, background capture method and unchanged
input-project/source hashes. Selected PNG bytes and captions are installed in `manual-markdown/`. Do not silently promote Phase 1 or older
screenshots into Phase 4 provenance. No placeholder image link should reach the generated manual.

Runtime test/capture results belong to the integration report. This static source review is not
a passing runtime or visual rendering check. After final validation, follow
`BUILDING-THE-MANUAL.md`, inspect rendered pages and verify matching HTML/PDF/ZIP hashes. Installed
application Help changes only with a separate application build/install.

The integration lead verified `tests/gui/test_event_heatmap_coordinates.py` on development
source `9ddafefa`: **8 passed**, covering asymmetric peaks, both global image orders, live/frozen
rectangular restoration and PNG/CSV/SVG/PDF coordinates. Log family:
`phase4-final-owner-and-pixels`. The September 11 orientation warning is retained as historical
evidence, not asserted as a remaining defect. No old screenshot is retroactively validated by this gate.
