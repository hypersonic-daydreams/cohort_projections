# 2026 Public Projection Release Handoff

This folder is the working handoff for the 2026 North Dakota population
projection public release. **Status: numbers are final and locked.** The data
artifacts were generated from the ADR-068-corrected locked production run
(`m2026r1` / `cfg-20260611-production-lock`, config sha256(16)
`a6e0bfbc2d70be85`; full-horizon corrected run **2026-06-16** — see
`final-run-metadata.md` for the authoritative provenance block). CF-001 was
disposed 2026-06-11 (ADR-061 Accepted as modified; ADR-067 applied); the
public release is baseline-only per ADR-065 — the public path is Baseline
(CBO-Adjusted); the unadjusted trend-continuation and restricted/high
sensitivities are internal.

Headline set (locked): **799,358 (2025) → shallow trough 797,298 (2027,
−0.26%) → 898,907 (2055, +12.45%)**; 36 of 53 counties decline; growth
concentrates in Cass, Williams, and Burleigh (more than three-quarters of
gross gains).

The 2026 report reads as the next edition of the State Data Center series: it
follows the 2024 report's scale and structure (with a components-of-change
exhibit and a "How these projections compare with the 2024 edition" box), and
adds what the 2024 release did not have — public data downloads and a
plain-language methods companion.

## The Marketing Packet

Everything marketing needs lives in [`marketing-ready/`](marketing-ready/)
(see its README for the "What To Send" table): seven generated `PUB-2026
*.docx` narrative documents, the bundled 2024 SDC reference PDF, and the
generated data/visual artifacts in [`marketing-ready/drafts/`](marketing-ready/drafts/)
— the consolidated public workbook, the tidy 1,922-row public CSV
(1 scenario × 62 geographies × 31 years), five reference chart PNGs
(statewide line, regional bars, county top/bottom, age trend, components of
change), and six pyramid PNGs.

Two rules keep the packet from drifting:

- The `.docx` files are **generated, never hand-edited** — edit the markdown
  sources and rerun
  [`scripts/exports/build_marketing_docx.py`](../../../scripts/exports/build_marketing_docx.py)
  (use `--strict` so hand-authored prose that drifts from the locked CSV fails
  the build).
- The data/visual artifacts regenerate via
  [`scripts/exports/build_public_draft_package.py`](../../../scripts/exports/build_public_draft_package.py).
  Rerun only if the locked production outputs change; any rerun re-triggers
  the prose-sync checklist in `draft-public-pdf-copy.md`'s banner.

## Errata history

- **2026-06-13** — first locked run (superseded: carried a ~3× international
  migration input error and two 90+ survival defects).
- **2026-06-15/16 (ADR-068 + amendment)** — corrected full-horizon rerun;
  all artifacts and prose re-synced (PRs #25–#27).
- **2026-07-06/07 (PUB-2026-ERRATA)** — pre-handoff assessment vs the 2024
  report and release-machine verification (see
  [`docs/reviews/audit/`](../../reviews/audit/)); fixed the stale
  declining-county count (37 → **36**), replaced the unsourced "0.7%"
  track-record claim with figures attributed to the 2024 edition, fixed
  pyramid/workbook age-bracket labels, added the 2024-edition comparison box
  and series-convention cover, added the components-of-change exhibit, added
  the public methods companion to the packet, and extended the strict
  prose-sync guard.

## Working Docs In This Folder

| File | Audience | Use |
|------|----------|-----|
| [`draft-public-pdf-copy.md`](draft-public-pdf-copy.md) | Marketing/content review | Draft public report text (source of the PDF-copy docx) |
| [`how-these-projections-work.md`](how-these-projections-work.md) | Public / marketing | Plain-language methods companion (source of the companion docx) |
| [`figure-drafts.md`](figure-drafts.md) + [`figures/`](figures/) | SDC + marketing | Candidate explainer figures (selection menu; rendered mockups) |
| [`pdf-content-outline.md`](pdf-content-outline.md) | Marketing/design | Page storyboard and exhibit plan |
| [`2024-sdc-pdf-review.md`](2024-sdc-pdf-review.md) | Marketing/design | Practical guidance for using the 2024 PDF |
| [`final-run-metadata.md`](final-run-metadata.md) | SDC / QA | Locked-run provenance (config sha, run date, supersessions) |
| [`marketing-intake-brief.md`](marketing-intake-brief.md) | Historical | 2026-05-27 intake meeting record (superseded) |
| [`provisional-number-snapshot.md`](provisional-number-snapshot.md) | Historical | Pre-lock layout-numbers note (superseded — use the generated "Draft Numbers For Layout" docx) |

Internal repo records (decision log, QA checklists, release-readiness
checklist, download spec, finality remediation plan) live in this folder but
are not part of the marketing packet.

## Remaining Work

Repo-side content work is complete. What remains:

1. **Marketing:** layout/branding/accessibility (QA Gate 5), fill the
   contact/download placeholders in the PDF copy, re-verify every number at
   rendered-PDF layout, delivery items (Gate 6).
2. **Owner:** manual IT-sanctioned backup of `data/projections/baseline/**`
   and `marketing-ready/**` — this machine holds the only copy of the
   corrected production outputs, and rclone/bisync is disabled per IT policy
   (2026-07-06). See `release-readiness-checklist.md` §3.
3. City and place projections stay excluded from the public release.
   Geographies: 1 state, 8 planning regions (R1–R8), 53 counties.
