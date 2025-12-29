# Immigration Policy Scenario Data Manifest

**Generated:** 2025-12-29 03:39:54 UTC
**Scenario:** Full Cbo

---

## Data Sources

### Base Population
- **Source:** Census Vintage 2024 (same as "updated" variant)
- **Description:** 2020 Census age/sex distribution scaled to 2024 totals

### Survival Rates
- **Source:** CDC National Life Tables 2023 (same as "updated" variant)
- **Description:** 5-year survival probabilities from 2023 national life tables

### Fertility Rates
- **Source:** SDC 2024 Original (same as "updated" variant)
- **Description:** Original SDC county-level fertility patterns

### Migration Rates
- **Source:** SDC 2024 Original × Policy Adjustment Factor
- **Description:** SDC migration rates adjusted for 2025 immigration policy changes

---

## Policy Adjustment Methodology

### Empirical Basis
- **Data:** Census Bureau Components of Population Change (2010-2024)
- **Analysis:** Regression of ND international migration on US international migration
- **Transfer Coefficient:** ND receives 30.98% of its migration from international sources

### Adjustment Calculation
1. SDC migration rates combine domestic + international migration
2. International share of ND migration: 30.98%
3. Policy multiplier for "full_cbo": -0.1284
4. Rate adjustment factor: 0.6504

**Formula:** `new_rate = sdc_rate × 0.6504`

### Interpretation
Multiply SDC migration rates by 0.6504

---

## Scenario Details

### Full Cbo Scenario

Based on CBO September 2025 demographic outlook revision:
- Original 2025 projection: +1.1 million net international migration
- Revised 2025 projection: -290,000 net international migration
- Policy multiplier applied: -12.84% of baseline

---

## Files in This Directory

1. `base_population_by_county.csv` - Same as updated variant
2. `fertility_rates_by_county.csv` - Same as updated variant
3. `survival_rates_by_county.csv` - Same as updated variant
4. `adjustment_factors_by_county.csv` - Same as updated variant
5. `migration_rates_by_county.csv` - **ADJUSTED** for policy scenario
6. `period_multipliers.json` - Period-specific migration adjustments

---

## References

- [ADR-018: Immigration Policy Scenario Methodology](../../docs/adr/018-immigration-policy-scenario-methodology.md)
- [CBO Demographic Outlook Update 2025-2055](https://www.cbo.gov/publication/61735)
- [Immigration Policy Research Report](../../docs/research/2025_immigration_policy_demographic_impact.md)
