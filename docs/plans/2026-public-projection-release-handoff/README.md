# 2026 Public Projection Release Handoff

This folder is the working handoff for the 2026 North Dakota population
projection public release. Marketing receives the report structure, copy
direction, and 2024 State Data Center PDF reference now. Data-driven draft
assets were regenerated on 2026-05-27 after the ADR-065/066 baseline rerun.
The public release is baseline-only: the public path is Baseline
(CBO-Adjusted), while the older unadjusted baseline and the restricted/high
sensitivities remain internal.

The 2026 report should feel similar in scale and usefulness to the 2024 SDC
PDF: a compact public report that can stand on its own, with a downloadable
workbook available for users who need exact values.

## The Draft Package

All draft artifacts live in [`marketing-ready/`](marketing-ready/) and the
[`marketing-ready/drafts/`](marketing-ready/drafts/) subfolder. The current
draft data/PNG artifacts use the CBO-adjusted baseline and Census PEP Vintage
2025 county totals. Marketing can use the storyboard and structure now, but
data-driven visuals and numeric callouts should still be refreshed after final
number lock before they are used in final layout.

**Expected refreshed data package:**

- `marketing-ready/drafts/PUB-2026 Draft Public Workbook.xlsx` — consolidated
  workbook matching the public download schema. Includes the tidy 1,922-row
  dataset (1 scenario × 62 geographies × 31 years), key-year county table,
  age-group breakouts, baseline chart-ready cuts, baseline pyramid-ready
  cuts, and a data dictionary.
- `marketing-ready/drafts/PUB-2026 Draft Public Dataset.csv` — tidy
  consolidated CSV with the same 1,922 public rows.

**Expected refreshed reference visuals:**

- `marketing-ready/drafts/pyramid_state_2025.png` and
  `pyramid_state_2055_baseline.png` — statewide pyramids for current vs.
  the public baseline.
- `marketing-ready/drafts/pyramid_cass_*.png` and `pyramid_ward_*.png` —
  pyramids for the leading growth and decline counties (2025 vs 2055
  baseline).
- `marketing-ready/drafts/chart_state_baseline_line.png` — the storyboard
  page 5 statewide line chart.
- `marketing-ready/drafts/chart_region_baseline_bars.png` — page 7 regional
  change bars.
- `marketing-ready/drafts/chart_county_top_bottom.png` — page 8 top-6 /
  bottom-6 county bars.
- `marketing-ready/drafts/chart_age_group_trend.png` — page 9 age-group trend
  lines.

**Starting-point reference:**

- `marketing-ready/PUB-2026 Reference - 2024 SDC PDF.pdf` — the 2024 ND
  Population Projections report. Use as a design and pacing reference, not as
  a language template.

**Narrative and storyboard (Word):**

- `marketing-ready/PUB-2026 Marketing Handoff Packet.docx` — combined
  narrative packet.
- Plus separate `.docx` files for the draft PDF copy, storyboard, 2024 PDF
  reference notes, and rounded draft numbers.

## What Is Still Provisional

The numbers in the current workbook and visuals are draft layout values from
the March 2026 exports. Those figures were built against the older
three-scenario setup and are now stale under ADR-065. They can inform layout
shape only; they are not approved public figures and must not be reused as
final copy.

Final public numbers depend on the CF-001 college-fix model decision and the
ADR-065 baseline adjustment. If CF-001 is promoted, production projections
need to be rerun before the final PDF, workbook, and CSV are built. Rerun the
generator script [`scripts/exports/build_public_draft_package.py`](../../../scripts/exports/build_public_draft_package.py)
to refresh the entire draft package from the new baseline-only projection
outputs.

## What Comes After Final Production

After CF-001 is resolved and final production outputs pass QA, the data team
will deliver:

- Final numbers in the public workbook (same sheet structure as the draft,
  baseline-only).
- Final public CSV (tidy, 1,922 rows, same schema as the workbook's annual
  sheets).
- Final pyramid and chart PNGs regenerated from the final run.
- Final release language and source/run metadata.

The public release covers the state, 8 economic planning regions, and 53
counties. City and place projections are not part of this public release.

## How To Use The 2024 SDC PDF

The bundled `PUB-2026 Reference - 2024 SDC PDF.pdf` has the right
public-report scale: cover, table of contents, executive summary, statewide
chart, method page, regional section, demographic section, state/region
summary, county appendix, and contact information.

The 2026 report needs updated framing. It should lead with Baseline
(CBO-Adjusted), which includes the CBO additive migration adjustment and the
-5% fertility adjustment. The unadjusted trend-continuation path is now an
internal sensitivity, and the restricted/high sensitivities are not part of the
public marketing package.

## Working Docs In This Folder

| File | Audience | Use |
|------|----------|-----|
| [`draft-public-pdf-copy.md`](draft-public-pdf-copy.md) | Marketing/content review | Draft public report text |
| [`pdf-content-outline.md`](pdf-content-outline.md) | Marketing/design | Page storyboard and exhibit plan |
| [`2024-sdc-pdf-review.md`](2024-sdc-pdf-review.md) | Marketing/design | Practical guidance for using the 2024 PDF |
| [`provisional-number-snapshot.md`](provisional-number-snapshot.md) | Marketing/design | Rounded draft numbers for layout |
| [`marketing-intake-brief.md`](marketing-intake-brief.md) | SDC + marketing | Intake meeting agenda and review questions |

Internal repo records (decision log, QA control checklist, technical download
specification) live in this folder but are not part of the marketing packet.
