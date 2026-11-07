# Phase 1 change fragment: F (the Foundation)

Written by the Foundation (design section 15). G6 merges this into `CHANGELOG.md`, the guide and
the inventory; nothing else edits those files for F.

## CHANGELOG-ready text

- The Analysis Explorer has one toolbar on every page: Undo, Redo, **Update plot** (with a
  count of pending changes in brackets and a drop-down item, **Refresh from recordings**),
  **Style ▾**, **Reset figure**, **Export ▾** and an **Inspector** switch.
- A right-hand **inspector** with two tabs, **Data & analysis** and **Format**, holds each
  page's controls (the Section drop-down on the page is no longer shown). Click the tab, or
  press ⌘1 (Data & analysis) and ⌘2 (Format); ⌘Return is Update plot on every page.
  The inspector is 360 px wide by default (320 to 560 when dragged), hides on Analysis home
  and with the toolbar switch, and remembers its tab, width and which groups are open.
- Export ▾ lists the page's own exports (the same list as the right-click menu) and, below
  them, **Save to Figure Library…**, **Save figure bundle…** and **Open Figure Library**. The
  old buttons stay for now and are enabled by the same rule.
- Four journal looks join the appearance presets: **7 pt Arial · Nature, Elsevier**,
  **8 pt Arial · PLOS, Springer, AIP**, **9 pt Times New Roman · IEEE** and
  **Slides · 18 pt Arial** (sizes by the rule "N for title and axis labels, N-1 for ticks and
  legend"; no fit label, no word and no axis range is carried).
- I-V: Update plot is greyed while the open-pore gather runs (before, only a hidden button was).
- The window-template group and the Source names and colours editor were re-flowed to fit the
  inspector column (two rows of buttons; a two-column form).

## Labels added

Toolbar: `↶ Undo`, `↷ Redo`, `Update plot`, `Update plot (n)`, `Refresh from recordings`,
`Style ▾` (`This page`, `Built in`, the four looks above and the two shipped ones),
`Reset figure`, `Export ▾`, `Save to Figure Library…`, `Save figure bundle…`,
`Open Figure Library`, `Inspector`. Inspector: tabs `Data & analysis` and `Format`; groups
`Estimator` (I-V), and for pages not yet ported `Data`, `Analysis`, `Export & recipes` (Distributions:
`Sources`, `Plot`, `Groups`, `Recipes`). Accessible names: `Plot inspector`, `<Title> group`,
`Analysis toolbar`.

## Labels removed (while a page is hosted in the Explorer)

The page's own toolbar row: `Section` and its drop-down (Data, Analysis, Appearance, Export),
the bar's `Reset figure`, `Export ▾` and verb buttons (`Analyze / refresh`, `Generate`,
`Plot / Apply`, `Apply fit` of Voltage dependence; the objects stay, hidden, and keep their
text) and Ctrl+1 to 4 as Section keys (⌘3 and ⌘4 are unbound). Nothing is removed from a page
opened on its own. Distributions' own `Plot` and `Apply fit` buttons stay beside their controls
until the Distributions group hides them; Subpopulations keeps its bar on screen (its pointer
readout lives there) until the Pages B group moves the readout.
The Explorer's old header buttons (`Refresh current`, `Refresh availability`) and its Figure
Library row are still there: the shell group removes them.

## Doc passages this makes stale (anchors drift; re-find them)

- `docs/USER_GUIDE.md`: the "Plot pages: Ctrl/⌘+1 / 2 / 3 / 4" shortcut row (about :3680; becomes
  "⌘1 / ⌘2 and ⌘Return"); every "choose the Section …" step in section 10 "Population plots" and
  "A consistent plotting workflow" (about :2542-2775); the Reset figure placement sentences
  ("just before Export ▾"); the Figure fonts passage (about :2776); the Figure Library passages
  that call the three buttons by name (about :259, :2852, :2904).
- `docs/inventory.json`: the groups "Analysis Explorer and Distributions workflow", "Figure
  Library", the Event waveforms toolbar item and the preset list (four new looks).
- `docs/manual-markdown/manual.md`: the "Choose Section: …" steps and the verbs table.
- Docstrings: `analysis_explorer.py:1-12`, `plot_workflow.py:1, :36-40`, `plot_appearance.py:667-699`
  (G6, after the owners' patches).

## For the integrator

- Widget-count ceilings (measured, offscreen, per page: what adoption adds / what the first
  show of Format adds): distributions 28/1, summaries 24/1, comparison 26/1, events 24/1,
  noise 24/1, I-V 20/1, voltage dependence 24/1, subpopulations 23/1; the toolbar is 12 widgets
  and the inspector shell 5. They are frozen (plus 10%) in `tests/gui/test_p1_explorer_hosting.py`.
- `tests/gui/test_p1_page_contract.py` keeps `FALLBACK`: each page group removes its page id.
  The fallback adapter (`analysis_inspector.fallback_adapter`) is deleted once it is empty.
- Crash-rate file lists for section 14 (Tier 1, the new Phase 1 files and the M1 list) are in
  `redesign/phase1/crash_files_*.txt`; `pages_loop.py` gained `NS_ADOPT=1` and `loop_rate.sh`
  the two trailing arguments `adoptA adoptB`.
- Widths with each page hosted at 840x560 (offscreen, 1x; page minimum width / widest Data-tab
  group / sideways scroll range of the Data tab): Distributions 420/295/0, Summaries 106/481/175,
  Comparison 576/386/80, Event waveforms 580/665/357, Noise 460/327/1, **I-V 496/102/0**,
  Voltage dependence 496/526/218, Subpopulations 426/311/5. Only I-V meets the inspector budget
  (page at most 500, groups at most 290); the others are the fallback's wide strips and are the
  page groups' to re-flow. The Format tab is a stub (the 4-column appearance widget, 483 px, in
  a column that scrolls sideways) until the Format group lands.
- Page lifetime (review round): `AnalysisExplorer.remove_page` returns the page with its
  controls intact. The page's pane stays a hidden child of the inspector until the page itself is
  destroyed (`page.destroyed`), because `QStackedWidget.removeWidget` gives a widget to Python
  and a pane dropped there takes every control moved into it along. A removed page can be added
  again, and `add_page` under an id that holds another widget closes the old one without raising.
- Known suite state after F (by failing id; each alone, offscreen): red and expected, with
  their owners: `test_p0_distributions_switches_and_reset_figure::test_in_the_explorer_the_page_in_front_has_the_button_and_it_resets_that_page`
  and `test_plot_ux_quick_wins::test_ctrl_2_on_a_real_page_shows_analysis_without_computing`
  (the shell group rewrites both: Reset figure and ⌘2 moved); red on the Phase 0
  tree too: `test_p0_g6a_review_fixes::test_the_documents_say_what_the_fixes_do`.
  `test_p0_g6a_review_fixes::test_the_numbers_boxes_keep_their_natural_spacing_in_a_wide_strip`
  was red in the first F; it now builds a bare `PlotAppearanceWidget` at 1000 px (no Explorer),
  so the Phase 0 fix it pins (spare width goes to an empty fourth column) keeps a running test.
- Test helpers edited (not weakened): `test_p0_reset_on_quantity_change._on_the_toolbar` now also
  refuses a verb that Update plot would not run (it runs the "●" verbs, else the first), so the
  swap cannot click the wrong thing.
- Tests that open Export ▾ do so with `showMenu()` and click the entry for real: a press on an
  `InstantPopup` button runs a nested event loop and does not return while the menu is open.
