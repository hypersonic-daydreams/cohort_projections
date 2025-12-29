# Data Update Manifest

**Generated:** 2025-12-28 20:48:53

This document records the data sources used to create updated SDC methodology inputs.

---

## Data Sources

### Base Population

- **Source:** Census Vintage 2024 (co-est2024-alldata.csv)
- **Description:** 2020 Census age/sex distribution scaled to 2024 totals. Base year: July 1, 2024. Counties with PEP data: 53.

### Survival Rates

- **Source:** CDC National Life Tables 2023 (cdc_lifetables_2023_combined.csv)
- **Description:** 5-year survival probabilities calculated from 2023 national life tables. More recent than ND 2020 life tables used by SDC.

### Fertility Rates

- **Source:** SDC 2024 Processed + Original County Distribution
- **Description:** Using original SDC county-level fertility patterns. Updated rates available but county adjustment requires additional processing.

### Migration Rates

- **Source:** SDC 2024 Original (2000-2020 Census Residual)
- **Description:** INTENTIONALLY UNCHANGED from SDC methodology. Uses 2000-2020 average with 60% Bakken dampening. This preserves SDC's migration assumptions while updating other components.

### Adjustment Factors

- **Source:** SDC 2024 Original
- **Description:** Bakken dampening (60%) and period-specific multipliers preserved from SDC methodology.

---

## Notes

- Using national 2023 life tables instead of ND-specific 2020 tables. National tables reflect post-COVID mortality improvement.
- Fertility rates maintain original SDC county-level distribution. Future enhancement: apply 2023 NCHS rates with county adjustment factors.
- Migration rates are the most uncertain component. Keeping SDC's original rates allows isolation of the impact from updating base population and vital rates.

---

## Methodology Preservation

The following SDC 2024 methodology elements are preserved:

- 5-year age groups (0-4 through 85+)
- 5-year projection intervals (2025, 2030, 2035, 2040, 2045)
- Sex-specific rates (male/female)
- County-level projections for all 53 ND counties
- SDC's 2000-2020 migration rates with 60% Bakken dampening

The following are updated:

- Base population scaled to 2024 using Census Vintage 2024 estimates
- Survival rates from 2023 CDC national life tables
- Fertility rates (where updated data available)
