# Executive Summary: SDC 2024 vs. 2026 Methodology Comparison

**North Dakota State Data Center Population Projections**
**Date**: 2026-02-20
**Series**: Methodology Review -- SDC 2024 vs. Current 2026 Cohort-Component Projections

---

## 1. Overview

This document summarizes the methodological differences between the North Dakota State Data Center's 2024 population projections (published February 2024) and the current 2026 cohort-component projection system developed as their successor. Both systems use the cohort-component method -- the standard demographic technique that projects population by aging cohorts forward, adding births, subtracting deaths, and applying net migration. The two implementations differ substantially in resolution, data recency, scenario treatment, and the handling of migration assumptions. Understanding these differences is essential for interpreting why the two projections produce divergent population trajectories and for evaluating the fitness of each approach for planning purposes.

---

## 2. At-a-Glance Comparison

| Dimension | SDC 2024 | Current 2026 |
|-----------|----------|--------------|
| **Base Year** | 2020 (April 1, Decennial Census) | 2025 (July 1, PEP Vintage 2025) |
| **Projection Horizon** | 2020--2050 (30 years) | 2025--2055 (30 years) |
| **Time Step** | 5-year intervals (6 steps) | Annual / 1-year intervals (30 steps) |
| **Age Resolution** | 18 five-year groups (0--4 through 85+) | 91 single-year ages (0 through 90+) |
| **Sex Categories** | Male, Female | Male, Female |
| **Race/Ethnicity Categories** | None | 6 (White NH, Black NH, AIAN NH, Asian/PI NH, Two+ Races NH, Hispanic) |
| **Total Cohort Cells per County** | 36 (18 ages x 2 sexes) | 1,092 (91 ages x 2 sexes x 6 races) |
| **Fertility Data Source** | ND DHHS Vital Statistics (2016--2022), blended with CDC national | CDC NCHS national ASFR (2024), race-specific |
| **Fertility Assumption** | Constant (TFR ~2.33 state avg.) | Constant baseline (TFR ~1.62 national); scenario-adjustable |
| **Mortality Data Source** | CDC ND 2020 life tables | CDC 2023 national life tables, ND-adjusted |
| **Mortality Assumption** | Constant (no improvement) | 0.5% annual improvement in death rates |
| **Migration Data Source** | Census residual, 2000--2020 (4 periods) | Census PEP residual, 2000--2024 (5 periods) |
| **Migration Periods** | 4 equal 5-year periods | 5 periods (four 5-year + one 4-year) |
| **Migration Forward-Projection** | Flat base rate x period multiplier (0.2--0.7) | Convergence interpolation: recent to medium to long-term (5-10-5 schedule) |
| **Special County Adjustments** | ~32,000 manual person-adjustments per period | Algorithmic: oil-county dampening, male dampening, PEP recalibration (reservation), college-age smoothing, age-aware rate caps |
| **Scenarios Published** | 1 (single trajectory) | 3 active (Baseline, Restricted Growth, High Growth) |
| **Implementation Platform** | Microsoft Excel (45-sheet workbook) | Python 3.12 / pandas / NumPy |
| **Reproducibility** | Requires specific workbook; adjustments undocumented | Fully reproducible from code + config; 1,165+ automated tests |

---

## 3. Key Methodological Departures (Ranked by Impact)

The following changes are ranked by their estimated effect on projected population outcomes, from largest to smallest.

**1. Migration convergence replaces period multipliers**
The SDC used six ad-hoc period multipliers (0.2 to 0.7) applied uniformly to all counties. The 2026 system uses Census Bureau-style convergence interpolation, where each county-age-sex cell independently transitions from its recent historical rate toward its long-term mean over a structured 5-10-5 year schedule (ADR-036). This is the single largest driver of divergence between the two projections because migration is the dominant component of North Dakota's population change.

**2. Fertility rates updated from state-specific to current national levels**
The SDC used ND-specific blended rates from 2016--2022 (state-average TFR ~2.33). The 2026 system uses 2024 CDC national rates (TFR ~1.62), reflecting the historic decline in U.S. fertility since the SDC's reference period (ADR-001). This 44% difference in TFR directly reduces projected births by a comparable magnitude.

**3. Three CBO-grounded scenarios replace single trajectory**
The SDC published one "most likely" projection. The 2026 system publishes three scenarios grounded in Congressional Budget Office data: a baseline (trend continuation), a restricted growth scenario modeling reduced immigration (ADR-037, ADR-050), and a high growth scenario using BEBR-optimistic migration rates (ADR-046). This provides stakeholders with a plausible range of outcomes rather than a single point estimate.

**4. Mortality improvement introduced**
The SDC held survival rates constant over 30 years. The 2026 system applies a 0.5% annual decline in death rates (ADR-002), consistent with long-term U.S. mortality trends. Over 30 years, this reduces cumulative deaths by approximately 14%, adding to projected population -- particularly among the elderly.

**5. Race/ethnicity dimension added**
The SDC did not project by race. The 2026 system carries 6 race/ethnicity categories with race-specific fertility, mortality, and migration rates (ADR-007, ADR-044). This enables planning for reservation communities, immigrant populations, and health disparities, and corrects compositional errors that arise from applying a single blended rate to demographically diverse counties.

**6. Manual adjustments replaced by algorithmic corrections**
The SDC applied approximately 32,000 undocumented manual person-adjustments per 5-year period. The 2026 system replaces these with documented, testable algorithmic corrections: oil-county dampening (ADR-040, ADR-051), reservation county PEP recalibration (ADR-045), college-age smoothing (ADR-049), and age-aware rate caps (ADR-043).

**7. Resolution increased from 5-year to annual / single-year-of-age**
The 2026 system operates at 30x the resolution of the SDC (1,092 vs. 36 cohort cells per county, annual vs. 5-year time steps). This eliminates staircase artifacts from uniform age-group splitting, captures within-period interactions between components, and produces output aligned with annual planning cycles.

---

## 4. What Stayed the Same

Despite the substantial methodological upgrades, the following elements are fundamentally unchanged:

- **Core method**: Both systems use the cohort-component method -- the standard demographic projection technique used worldwide.
- **County geography**: Both project 53 North Dakota counties independently, with state totals computed by summation.
- **Sex categories**: Both project males and females separately.
- **Bottom-up aggregation**: Both build state totals from county projections (not top-down allocation).
- **Migration estimation**: Both derive migration rates from Census residual calculations (observed population minus expected population after survival).
- **Projection horizon**: Both project 30 years forward.
- **Bakken dampening concept**: Both recognize the need to reduce the influence of the 2010--2015 oil boom on long-term migration assumptions, though the mechanisms differ.

---

## 5. Root Cause of SDC vs. 2026 Divergence

The SDC projects North Dakota reaching approximately **957,000** by 2050. The current baseline projects approximately **815,000** by 2055. This gap of roughly **140,000--170,000 persons** is not the result of any single methodological change; it emerges from the interaction of several factors, with migration assumptions as the primary driver.

**Migration is the dominant factor.** North Dakota's natural increase (births minus deaths) is near zero and turning negative. Virtually all projected growth comes from net migration. The SDC's period multipliers allow migration to accelerate in the final decades (rising from 50% to 70% of the historical base rate), producing +37,600 net migrants in 2045--2050 alone. The 2026 system's convergence approach moves rates toward the 25-year long-term mean and holds there, producing stable but lower annual migration volumes.

**Fertility compounds the gap.** The SDC's higher TFR (~2.33 vs. ~1.62) generates substantially more births each year, which compounds over 30 years as those additional births grow up and themselves enter the reproductive and labor-force ages.

**Mortality works in the opposite direction.** The 2026 system's mortality improvement keeps more people alive in the later decades, partially offsetting the lower fertility and migration. The SDC's constant mortality slightly overstates deaths, particularly among the elderly.

The key takeaway: projections for a state where 91% of recent net migration is international are acutely sensitive to migration assumptions. The two systems embody fundamentally different views of how migration evolves over time -- the SDC assumes an accelerating return toward historical patterns, while the 2026 system assumes convergence to a long-term mean. Neither is inherently correct; the 2026 system addresses this uncertainty by publishing three scenarios rather than one.

---

## 6. Detailed Component Documents

For readers who need depth on any specific component, the following detailed comparison documents are available:

| Document | File | Coverage |
|----------|------|----------|
| 01 -- Base Population | `01_base_population.md` | Base year selection, age resolution (5-year vs. single-year), race/ethnicity categories, county-specific distributions, small-county blending |
| 02 -- Fertility | `02_fertility.md` | ASFR data sources, TFR levels, county-specific vs. national rates, race-specific fertility, sex ratio at birth, scenario adjustments |
| 03 -- Mortality | `03_mortality.md` | Life table sources, 5-year vs. single-year survival, mortality improvement (0.5%/year Lee-Carter), Census NP2023 time-varying rates, race-specific mortality |
| 04 -- Migration Computation | `04_migration_computation.md` | Residual migration method, historical periods, annualization, oil-county dampening, male dampening, reservation PEP recalibration, college-age smoothing |
| 05 -- Migration Convergence | `05_migration_convergence.md` | SDC period multipliers vs. 5-10-5 convergence interpolation, age-aware rate caps, BEBR high-growth increment, migration floor |
| 06 -- Scenarios | `06_scenarios.md` | Single trajectory vs. three-scenario framework, CBO-grounded restricted growth, BEBR high growth, additive vs. multiplicative adjustments, presentation requirements |
| 07 -- Structural / Engine | `07_structural_engine.md` | Time step (annual vs. 5-year), age resolution, race dimension, Excel vs. Python implementation, reproducibility, automated testing |

All documents are located in `docs/reviews/methodology_comparison/`.
