# PUB-2026 Release QA Checklist

Internal repo record. Do not include this file in the marketing handoff packet.
It is a final publication QA control for the data and content team.

Status: draft checklist for final-number and marketing handoff.

## Gate 1: Method And Data Lock

- [x] ADR-065 baseline assumptions are resolved and recorded. → [defensibility memo](../../reviews/2026-06-12-adr-065-defensibility-memo.md) (both CBO adjustments affirmed, 2026-06-12).
- [x] CF-001 decision is resolved and recorded. → ADR-061 Accepted-as-modified; decision record `../../reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md` (Approved).
- [x] If CF-001 is promoted, production projections are rerun after promotion. → locked rerun 2026-06-13.
- [x] Input-coverage verification (ADR-067 F2): mortality file spans 2025-2055 (31 yrs) ✓; residual-migration metadata = 11 college counties + GQ 0.75 ✓; convergence rates present for all 53 counties ✓ (verified 2026-06-13). Guards against the 2026-06-01 silent bisync stale-replacement failure mode.
- [x] Final production run metadata is recorded. → [final-run-metadata.md](final-run-metadata.md).
- [ ] Final data source notes point to the exact run used for public outputs. (Stage 5 — at packaging.)
- [x] Current March 2026 draft exports are not treated as final numbers. → superseded by the 2026-06-13 locked run.

## Gate 1b: Demographic Plausibility

- [x] A dated sanity-check review of the final production run is completed and linked. → [2026-06-13-locked-run-sanity-check.md](../../reviews/2026-06-13-locked-run-sanity-check.md).
- [x] The 2025-2030 projected trajectory is reconciled against Census PEP Vintage 2025 observed components, including an explicit explanation of the 2025-2028 dip as the intended CBO front-loaded migration adjustment. → sanity check §2; cbo_off run shows no dip.
- [x] Projected 2025-2030 components of change (births, deaths, net migration) are cross-checked against PEP observed components. → sanity check §3 (note: deaths are household-basis; ~2,000/yr GQ gap explained).
- [x] State and large-county age structures and sex ratios at 2035/2045/2055 are reviewed for plausibility. → sanity check §4 (state + Cass/Burleigh/Grand Forks/Ward/Williams).
- [x] The largest county-level divergences from the 2024 SDC series (Ward, Grand Forks) have a written disposition (corrective ADR or accepted-divergence rationale). → ADR-067 (corrective investigation) + [divergent-counties framing](../../reviews/2026-06-13-divergent-counties-methods-and-framing.md) (incl. Williams); user-signed-off 2026-06-13.
- [x] No county exhibits implausible terminal dynamics (e.g., population collapse to near zero, runaway growth, or sex-ratio drift) without documented explanation. → sanity check §5 (53-county scan; oil-county growth is conservative-migration + young-age natural increase).

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
