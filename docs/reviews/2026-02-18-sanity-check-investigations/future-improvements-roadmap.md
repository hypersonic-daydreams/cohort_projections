# Future Improvements Roadmap

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Author** | Claude Code (Opus 4.6), directed by N. Haarstad |
| **Parent** | [Sanity Check Investigation Synthesis](README.md) |
| **Scope** | Non-blocking improvements identified during sanity check investigations |
| **Status** | Active backlog |

---

## Purpose

This document captures longer-term improvement suggestions surfaced during the 2026-02-18 sanity check investigations. None of these items block the current publication cycle, but each would strengthen the projection system's accuracy, transparency, or defensibility over time.

Items are organized into two priority tiers based on impact and feasibility.

---

## Tier 1: Medium-Term Improvements

These items address known methodological weaknesses that affect projection quality for specific populations or geographies. They should be considered for the next major methodology revision cycle.

### 1. Race-Specific Survival Rates for AIAN Populations

**Source finding**: [Finding 3 -- Reservation County Declines](finding-3-reservation-county-declines.md)

**Description**: The residual migration method currently applies statewide survival rates uniformly across all races. AIAN populations have substantially lower life expectancy (AIAN male e0 ~66.1 years vs. statewide ~75.5 years). When statewide survival rates are applied to AIAN-majority counties, excess mortality is misattributed to out-migration in the residual computation, inflating out-migration estimates by 2-3x relative to PEP components.

**Rationale**: This is the largest single contributor to the residual method amplification documented in Finding 3. Correcting it would directly reduce the bias for reservation counties (Benson, Sioux, Rolette) and improve the accuracy of the residual method statewide.

**Data requirements**:
- CDC WONDER or NVSS race-specific life tables (AIAN, at minimum)
- State-level or national AIAN life tables as a fallback if ND-specific tables are suppressed due to small counts

**Estimated complexity**: Medium. Requires extending the survival rate loader to accept race-stratified tables and modifying the residual migration computation to look up survival rates by race. The projection engine already handles race-differentiated survival (via CDC life tables in `data/processed/survival_rates.parquet`); the gap is only in the residual migration preprocessing step.

**Key files**: `cohort_projections/data/process/residual_migration.py`, `cohort_projections/data/load/survival_rate_loader.py`

**Dependencies**: None. Can be implemented independently.

---

### 2. IPF (Iterative Proportional Fitting) for Race Allocation

**Source findings**: [Findings 4-5 -- Race-Specific Trends](finding-4-5-race-specific-trends.md)

**Description**: The current Census+PUMS hybrid approach (ADR-041) uses Census age-sex marginals multiplied by PUMS within-cell race conditionals. IPF would instead jointly fit three sets of constraints:
1. Census age-sex marginals (authoritative)
2. Census state-level race marginals (authoritative)
3. PUMS race-within-cell conditionals (used as a prior/seed matrix)

IPF iteratively adjusts the seed matrix until all marginals match their Census targets simultaneously, producing a distribution that is consistent with all known totals while preserving the PUMS-derived within-cell structure where it is not contradicted by marginals.

**Rationale**: The current multiplicative approach does not guarantee that race marginals match Census totals. For small race groups (Black: ~5,000; AIAN: ~36,000), the PUMS-derived conditionals are noisy enough that the resulting race totals can diverge meaningfully from Census counts. IPF would eliminate this divergence while being only modestly more complex to implement.

**Data requirements**:
- Census cc-est2024-alldata (already available) for state-level race marginals
- Census cc-est2024-agesex-all (already used) for age-sex marginals
- Existing PUMS race conditional file as seed matrix

**Estimated complexity**: Low-medium. IPF is a well-understood algorithm with off-the-shelf implementations (e.g., `ipfn` Python package). The main work is setting up the constraint matrices and integrating the output into the existing distribution CSV pipeline.

**Key files**: `data/raw/population/nd_age_sex_race_distribution.csv`, `cohort_projections/data/load/base_population_loader.py`

**Dependencies**: None. Can be implemented independently. ADR-041 already mentions IPF as a considered alternative.

**Related ADR**: ADR-041 (Census+PUMS Hybrid Base Population)

---

### 3. County-Specific Medium Window Narrowing

**Source finding**: [Finding 2 -- Oil County Growth](finding-2-oil-county-growth.md), specifically section 2-D

**Description**: The convergence interpolation currently uses a single medium window `[2014, 2025]` for all 53 counties. For oil-patch counties where the 2010-2015 boom produced extreme migration rates, this window includes boom-era data that inflates the medium rate. Allowing county-specific medium windows (e.g., `[2019, 2025]` or `[2020, 2025]` for oil counties) would exclude boom years from their medium average, producing more representative rates without affecting other counties.

**Rationale**: ADR-040 dampening reduced McKenzie County growth from +114% to +84%, but the investigation concluded that dampening alone is insufficient because boom-era rates still contaminate the medium window. Narrowing the window for affected counties is a complementary approach that addresses the root cause rather than applying a correction factor after the fact.

**Data requirements**: None beyond existing data. This is a configuration and logic change.

**Estimated complexity**: Medium. Requires adding county-level window overrides to `projection_config.yaml` and modifying `convergence_interpolation.py` to accept per-county window parameters. The conditional logic must be clearly documented to avoid configuration drift as the county list evolves.

**Key files**: `config/projection_config.yaml`, `cohort_projections/data/process/convergence_interpolation.py`

**Dependencies**: Should be designed alongside any broader rate-cap or dampening redesign to avoid overlapping correction mechanisms.

---

### 4. Year-by-Year Component Reporting

**Source finding**: [Finding 6 -- Baseline 1M Plausibility](finding-6-baseline-1m-plausibility.md)

**Description**: The projection output parquet files currently contain only total population by year for each county-age-sex-race cohort. Adding births, deaths, and net migration as separate columns would enable transparent decomposition of population change and make validation significantly easier.

**Rationale**: During the sanity check investigations, decomposing projected growth into components required reverse-engineering the engine's internal state. If component columns were included in the output, investigators could immediately see whether growth is driven by natural increase vs. migration, and at what point components shift. This would also support the presentation caveat recommended in Finding 6 (showing stakeholders that baseline growth depends heavily on sustained international migration).

**Data requirements**: None. The projection engine already computes births, deaths, and net migration internally; they just need to be retained in the output DataFrame.

**Estimated complexity**: Low. The cohort component engine (`cohort_projections/core/cohort_component.py`) computes these values each projection year. The change involves collecting them into the output structure and updating the export pipeline to include the additional columns.

**Key files**: `cohort_projections/core/cohort_component.py`, `scripts/pipeline/02_run_projections.py`, export scripts in `scripts/exports/`

**Dependencies**: Export format changes (ADR-038) should be coordinated. The multi-workbook export would need updated column mappings.

---

## Tier 2: Longer-Term Improvements

These items require more substantial design work, external engagement, or architectural changes. They are important for system maturity but are not expected in the near term.

### 5. Sensitivity Analysis on `intl_share`

**Source finding**: [Finding 6 -- Baseline 1M Plausibility](finding-6-baseline-1m-plausibility.md)

**Description**: The `intl_share` parameter (currently 0.91, per ADR-039) controls what fraction of net migration is classified as international and therefore subject to the CBO time-varying factor in the restricted growth scenario. Running the baseline with `intl_share` values of 0.50, 0.75, and 0.91 would reveal how sensitive the state growth trajectory is to the international migration composition assumption.

**Rationale**: Finding 6 documented that 91% of projected net migration is international, and 89% of growth is concentrated in 3 counties. The `intl_share` parameter is derived from 2023-2025 PEP data, a short window that may not represent the long-run composition. A sensitivity analysis would quantify the trajectory range and inform whether the parameter needs a time-decay or alternative estimation approach.

**Estimated complexity**: Low. This is a scripting exercise -- run existing projections with alternative parameter values and compare outputs. No code changes required; only configuration variations.

**Dependencies**: Should follow any redesign of the high-growth scenario (currently deactivated per Finding 7), since both involve migration scaling.

---

### 6. Small-Population Safeguards

**Source findings**: Cross-cutting across [Finding 2](finding-2-oil-county-growth.md), [Finding 3](finding-3-reservation-county-declines.md), [Findings 4-5](finding-4-5-race-specific-trends.md)

**Description**: A systematic framework for handling projections when populations are too small for standard demographic methods to produce stable results. This would include:
- **Rate caps**: Maximum absolute migration rate per age-sex-race cell (partially being designed separately)
- **Minimum cell counts**: Suppress or pool cells with fewer than N persons before applying rates
- **Bayesian shrinkage**: Blend county-level rates toward state or regional averages as population decreases, weighted by reliability
- **Suppression thresholds**: Flag or suppress race-specific projections below a minimum total population (e.g., do not publish race-specific results for groups below 1,000)

**Rationale**: Three distinct findings trace to the same underlying issue -- small populations producing volatile or unreliable rates:
- Billings County (1,071 total pop): 16 in-migrants create a +16.5% rate in one cell (Finding 2)
- Black non-Hispanic (5,027 statewide): PUMS yields only 44 observations, 7 populated cells (Finding 4-5)
- Sioux County (3,667 total pop): Residual method misattributes Census undercount as out-migration (Finding 3)

A unified framework would address all three cases under a consistent set of rules rather than ad-hoc per-county fixes.

**Estimated complexity**: High. Requires architectural design decisions (where in the pipeline to apply safeguards, what thresholds to use, how to communicate suppressed results to users). Bayesian shrinkage in particular requires defining a hierarchical model for borrowing strength across geographies.

**Key files**: Would touch `cohort_projections/data/process/residual_migration.py`, `convergence_interpolation.py`, and potentially the core engine modules.

**Dependencies**: Rate caps (being designed separately) are a prerequisite or at least a co-design item. The suppression threshold decision should be coordinated with the export format (ADR-038).

---

### 7. Tribal Consultation Protocol

**Source finding**: [Finding 3 -- Reservation County Declines](finding-3-reservation-county-declines.md), Section 9

**Description**: Before publishing projections for reservation-area counties (Benson/Spirit Lake, Sioux/Standing Rock, Rolette/Turtle Mountain), seek input from tribal data offices regarding:
- Plausibility of migration assumptions given planned economic development or housing initiatives
- Sensitivity of publication framing (avoiding narratives of abandonment or decline)
- Tribal enrollment trends that may not align with Census-based residence counts
- Methodological transparency (sharing the projection approach for tribal review)

**Rationale**: Finding 3 documented that publishing projections showing reservation populations declining by nearly half carries risks of misinterpretation and policy harm. Tribal sovereignty means these populations have their own data governance considerations. The investigation explicitly recommended against publishing current reservation projections without either methodological correction or tribal engagement.

**Estimated complexity**: Low (technically) but high (organizationally). This is a process and relationship-building effort, not a code change. It requires identifying appropriate tribal contacts, establishing a review protocol, and potentially adjusting publication timelines.

**Dependencies**: Should follow the methodological corrections for reservation counties (recalibration using PEP components, migration deceleration) so that the projections presented for tribal review reflect the improved methodology.

---

## Dependency Map

```
Independent items (can proceed in any order):
  [1] Race-specific survival rates
  [2] IPF for race allocation
  [4] Year-by-year component reporting
  [5] Sensitivity analysis on intl_share

Items with dependencies:
  [3] County-specific medium window  -->  coordinate with rate-cap design
  [6] Small-population safeguards    -->  depends on rate-cap design (in progress)
  [7] Tribal consultation            -->  follows reservation county method fixes
```

---

## Relationship to Current Publication Blockers

The [Recommended Action Sequence](README.md#recommended-action-sequence) in the synthesis document defines P0-P3 priorities. The items in this roadmap correspond to P3 actions (longer-term improvements) and are explicitly **not required for the current publication cycle**. The P0-P2 actions (deactivate high_growth, replace PUMS race allocation, recalibrate reservation migration, strengthen oil county dampening) should be completed first.

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-18 |
| **Version** | 1.0 |
