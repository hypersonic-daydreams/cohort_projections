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
- [x] Final data source notes point to the exact run used for public outputs. → workbook README sheet embeds method/config/commit/sha256 provenance; [final-run-metadata.md](final-run-metadata.md) (2026-06-13).
- [x] Current March 2026 draft exports are not treated as final numbers. → superseded by the 2026-06-13 locked run.

## Gate 1b: Demographic Plausibility

- [x] A dated sanity-check review of the final production run is completed and linked. → [2026-06-13-locked-run-sanity-check.md](../../reviews/2026-06-13-locked-run-sanity-check.md).
- [x] The 2025-2030 projected trajectory is reconciled against Census PEP Vintage 2025 observed components, including an explicit explanation of the 2025-2028 dip as the intended CBO front-loaded migration adjustment. → sanity check §2; cbo_off run shows no dip.
- [x] Projected 2025-2030 components of change (births, deaths, net migration) are cross-checked against PEP observed components. → sanity check §3 (note: deaths are household-basis; ~2,000/yr GQ gap explained).
- [x] State and large-county age structures and sex ratios at 2035/2045/2055 are reviewed for plausibility. → sanity check §4 (state + Cass/Burleigh/Grand Forks/Ward/Williams).
- [x] The largest county-level divergences from the 2024 SDC series (Ward, Grand Forks) have a written disposition (corrective ADR or accepted-divergence rationale). → ADR-067 (corrective investigation) + [divergent-counties framing](../../reviews/2026-06-13-divergent-counties-methods-and-framing.md) (incl. Williams); user-signed-off 2026-06-13.
- [x] No county exhibits implausible terminal dynamics (e.g., population collapse to near zero, runaway growth, or sex-ratio drift) without documented explanation. → sanity check §5 (53-county scan; oil-county growth is conservative-migration + young-age natural increase).

## Gate 2: Public Download QA

All items verified and adversarially re-confirmed 2026-06-13 → [release QA sign-off](../../reviews/2026-06-13-release-qa-signoff.md).

- [x] Public Excel workbook is generated from a clean staging directory. → rebuilt 2026-06-13 from locked outputs; 15 sheets.
- [x] Public CSV is generated from a clean staging directory.
- [x] CSV contains exactly `1,922` rows: 1 scenario x 62 geographies x 31 years.
- [x] CSV contains only `state`, `region`, and `county` geography levels.
- [x] CSV contains only Baseline (CBO-Adjusted) rows; no `recent_trend_continuation`, `restricted_growth`, or `high_growth` rows are present.
- [x] No city/place rows are present.
- [x] No `2025_2045` stale-horizon files are included in the public staging area.
- [x] State totals match county sums within tolerance. → max abs diff ≈ 2.3e-10 (floating-point zero).
- [x] Region totals match member-county sums within tolerance. → max abs diff 0.0 across all 8 regions.
- [x] Public workbook uses `State Key Years` instead of `Scenario Summary`.
- [x] Data dictionary matches the final CSV columns. → 18 columns, exact order match.
- [x] Excel sheet names match `public-download-spec.md`. → all 9 required sheets present (incl. new State Age-Sex Detail) + 6 Chart sheets.

## Gate 3: PDF Content QA

Verified against the PDF copy source `draft-public-pdf-copy.md` (2026-06-13). Final rendered-PDF
layout is produced by marketing; re-verify these at layout. → [sign-off](../../reviews/2026-06-13-release-qa-signoff.md).

- [x] PDF includes state, region, and county coverage only. → copy covers state/region/county; "City and place projections are not included."
- [x] Public copy uses Baseline (CBO-Adjusted) as the only public path.
- [x] Public copy states that the baseline is a projection, not a guaranteed outcome.
- [x] Required ADR-042 caveats appear near the first statewide exhibit. → four caveats placed at the Executive Summary / first statewide exhibit with refreshed values.
- [x] Methodology summary identifies base data, fertility, mortality, migration, and baseline construction.
- [x] County appendix uses final public numbers. → copy callouts (Williams +52%/~63k, Ward −13%/~59k, Cass +32%/~266k) reconcile to the locked CSV; full appendix populated from downloads at layout.
- [x] Any rounded PDF numbers reconcile to the exact downloadable values. → spot-checked statewide callouts (799k/787k@2028/889k) against CSV.

## Gate 4: Language QA

ADR-042 banned-language pass executed and adversarially re-confirmed 2026-06-13 → [sign-off](../../reviews/2026-06-13-release-qa-signoff.md).

- [x] Run an ADR-042 banned-language search against public-facing copy. → every hit is a required negation/caveat; only bare uses are in the non-rendering editorial header.
- [x] Replace any disallowed point-estimate wording before handoff. → none found in the public-facing body.
- [x] Confirm `projection`, `scenario`, and `Baseline (CBO-Adjusted)` terminology is used consistently.
- [x] Confirm `recent_trend_continuation` appears only in internal notes, not public copy. → absent from the copy.
- [x] Confirm public text does not imply a single guaranteed path.

## Gate 5: Accessibility And Design QA

These concern the **rendered** PDF and final figures, which marketing produces. Deferred to
marketing layout; re-verify at layout. → [sign-off, Open items](../../reviews/2026-06-13-release-qa-signoff.md).
The chart-ready PNGs are reference cuts; the stale "refresh after final production rerun" chart
watermark was corrected 2026-06-13, and a Contact & Downloads section (with PLACEHOLDERs) was added
to the PDF copy.

- [ ] All charts can be interpreted without color alone. *(layout — bar charts also encode sign by axis position; line/age charts use markers)*
- [ ] Tables remain readable at PDF scale. *(layout)*
- [ ] Map labels and legends are legible. *(layout)*
- [ ] PDF has logical reading order. *(layout)*
- [~] Contact block and download links are current. *(section added to PDF copy with PLACEHOLDERs; marketing inserts live URLs/contact before publication)*
- [ ] Figure titles and captions are plain-language and baseline-aware. *(layout; chart subtitles now pre-publication-marked; 2025 pyramids are base-year, scenario-independent)*

## Gate 6: Final Handoff

Deliverables are build-complete and correct (2026-06-13); the delivery and publication actions
remain. → [sign-off](../../reviews/2026-06-13-release-qa-signoff.md).

- [ ] Final public Excel workbook delivered to marketing. *(built & QA'd; delivery pending)*
- [ ] Final public CSV delivered to marketing. *(built & QA'd; delivery pending)*
- [ ] Baseline chart-ready tables delivered to marketing. *(workbook Chart-* sheets + PNGs built; delivery pending)*
- [ ] Final PDF copy reviewed against this folder. *(copy reviewed; final rendered PDF reviewed at layout)*
- [x] Decision log updated with any changes made after marketing intake. → CF-001 disposition recorded in [benchmark decision record](../../reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md), [ADR-067](../../governance/adrs/067-ward-grand-forks-divergence-investigation.md), and DEVELOPMENT_TRACKER.
- [x] `DEVELOPMENT_TRACKER.md` updated with publication-handoff status. → PUB-2026 row updated 2026-06-13 (locked run + Stage 4/5 complete).
