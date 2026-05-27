# PUB-2026 Release QA Checklist

Internal repo record. Do not include this file in the marketing handoff packet.
It is a final publication QA control for the data and content team.

Status: draft checklist for final-number and marketing handoff.

## Gate 1: Method And Data Lock

- [ ] CF-001 decision is resolved and recorded.
- [ ] If CF-001 is promoted, production projections are rerun after promotion.
- [ ] Final production run metadata is recorded.
- [ ] Final data source notes point to the exact run used for public outputs.
- [ ] Current March 2026 draft exports are not treated as final numbers.

## Gate 2: Public Download QA

- [ ] Public Excel workbook is generated from a clean staging directory.
- [ ] Public CSV is generated from a clean staging directory.
- [ ] CSV contains exactly `5,766` rows: 3 scenarios x 62 geographies x 31 years.
- [ ] CSV contains only `state`, `region`, and `county` geography levels.
- [ ] No city/place rows are present.
- [ ] No `2025_2045` stale-horizon files are included in the public staging area.
- [ ] State totals match county sums within tolerance.
- [ ] Region totals match member-county sums within tolerance.
- [ ] State scenario ordering holds by year: `restricted_growth <= baseline <= high_growth`.
- [ ] Data dictionary matches the final CSV columns.
- [ ] Excel sheet names match `public-download-spec.md`.

## Gate 3: PDF Content QA

- [ ] PDF includes state, region, and county coverage only.
- [ ] Baseline-led headlines include restricted-growth context nearby.
- [ ] High growth is described as secondary planning context.
- [ ] Required ADR-042 caveats appear near the first statewide scenario exhibit.
- [ ] Methodology summary identifies base data, fertility, mortality, migration, and scenario construction.
- [ ] County appendix uses final public numbers.
- [ ] Any rounded PDF numbers reconcile to the exact downloadable values.

## Gate 4: Language QA

- [ ] Run an ADR-042 banned-language search against public-facing copy.
- [ ] Replace any disallowed point-estimate wording before handoff.
- [ ] Confirm `projection`, `scenario`, and `trend continuation` terminology is used consistently.
- [ ] Confirm public text does not imply a single guaranteed path.

## Gate 5: Accessibility And Design QA

- [ ] All charts can be interpreted without color alone.
- [ ] Tables remain readable at PDF scale.
- [ ] Map labels and legends are legible.
- [ ] PDF has logical reading order.
- [ ] Contact block and download links are current.
- [ ] Figure titles and captions are plain-language and scenario-aware.

## Gate 6: Final Handoff

- [ ] Final public Excel workbook delivered to marketing.
- [ ] Final public CSV delivered to marketing.
- [ ] Chart-ready tables delivered to marketing.
- [ ] Final PDF copy reviewed against this folder.
- [ ] Decision log updated with any changes made after marketing intake.
- [ ] `DEVELOPMENT_TRACKER.md` updated with publication-handoff status.
