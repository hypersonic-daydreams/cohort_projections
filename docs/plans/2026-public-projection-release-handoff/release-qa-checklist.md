# PUB-2026 Release QA Checklist

Internal repo record. Do not include this file in the marketing handoff packet.
It is a final publication QA control for the data and content team.

Status: draft checklist for final-number and marketing handoff.

## Gate 1: Method And Data Lock

- [ ] ADR-065 baseline assumptions are resolved and recorded.
- [ ] CF-001 decision is resolved and recorded.
- [ ] If CF-001 is promoted, production projections are rerun after promotion.
- [ ] Input-coverage verification (ADR-067 F2): the mortality-improvement file (`data/processed/mortality/nd_adjusted_survival_projections.parquet`) spans the full projection horizon (2025-2055, 31 years); the residual-migration metadata (`residual_migration_metadata.json`) records adjustments matching the locked production config (11 college counties, GQ fraction 0.75); convergence rates are present for all 53 counties. This guards against the 2026-06-01 silent bisync stale-replacement failure mode.
- [ ] Final production run metadata is recorded.
- [ ] Final data source notes point to the exact run used for public outputs.
- [ ] Current March 2026 draft exports are not treated as final numbers.

## Gate 1b: Demographic Plausibility

- [ ] A dated sanity-check review of the final production run is completed and linked.
- [ ] The 2025-2030 projected trajectory is reconciled against Census PEP Vintage 2025 observed components, including an explicit explanation of the 2025-2028 dip as the intended CBO front-loaded migration adjustment.
- [ ] Projected 2025-2030 components of change (births, deaths, net migration) are cross-checked against PEP observed components.
- [ ] State and large-county age structures and sex ratios at 2035/2045/2055 are reviewed for plausibility.
- [ ] The largest county-level divergences from the 2024 SDC series (Ward, Grand Forks) have a written disposition (corrective ADR or accepted-divergence rationale).
- [ ] No county exhibits implausible terminal dynamics (e.g., population collapse to near zero, runaway growth, or sex-ratio drift) without documented explanation.

## Gate 2: Public Download QA

- [ ] Public Excel workbook is generated from a clean staging directory.
- [ ] Public CSV is generated from a clean staging directory.
- [ ] CSV contains exactly `1,922` rows: 1 scenario x 62 geographies x 31 years.
- [ ] CSV contains only `state`, `region`, and `county` geography levels.
- [ ] CSV contains only Baseline (CBO-Adjusted) rows; no `recent_trend_continuation`, `restricted_growth`, or `high_growth` rows are present.
- [ ] No city/place rows are present.
- [ ] No `2025_2045` stale-horizon files are included in the public staging area.
- [ ] State totals match county sums within tolerance.
- [ ] Region totals match member-county sums within tolerance.
- [ ] Public workbook uses `State Key Years` instead of `Scenario Summary`.
- [ ] Data dictionary matches the final CSV columns.
- [ ] Excel sheet names match `public-download-spec.md`.

## Gate 3: PDF Content QA

- [ ] PDF includes state, region, and county coverage only.
- [ ] Public copy uses Baseline (CBO-Adjusted) as the only public path.
- [ ] Public copy states that the baseline is a projection, not a guaranteed outcome.
- [ ] Required ADR-042 caveats appear near the first statewide exhibit.
- [ ] Methodology summary identifies base data, fertility, mortality, migration, and baseline construction.
- [ ] County appendix uses final public numbers.
- [ ] Any rounded PDF numbers reconcile to the exact downloadable values.

## Gate 4: Language QA

- [ ] Run an ADR-042 banned-language search against public-facing copy.
- [ ] Replace any disallowed point-estimate wording before handoff.
- [ ] Confirm `projection`, `scenario`, and `Baseline (CBO-Adjusted)` terminology is used consistently.
- [ ] Confirm `recent_trend_continuation` appears only in internal notes, not public copy.
- [ ] Confirm public text does not imply a single guaranteed path.

## Gate 5: Accessibility And Design QA

- [ ] All charts can be interpreted without color alone.
- [ ] Tables remain readable at PDF scale.
- [ ] Map labels and legends are legible.
- [ ] PDF has logical reading order.
- [ ] Contact block and download links are current.
- [ ] Figure titles and captions are plain-language and baseline-aware.

## Gate 6: Final Handoff

- [ ] Final public Excel workbook delivered to marketing.
- [ ] Final public CSV delivered to marketing.
- [ ] Baseline chart-ready tables delivered to marketing.
- [ ] Final PDF copy reviewed against this folder.
- [ ] Decision log updated with any changes made after marketing intake.
- [ ] `DEVELOPMENT_TRACKER.md` updated with publication-handoff status.
