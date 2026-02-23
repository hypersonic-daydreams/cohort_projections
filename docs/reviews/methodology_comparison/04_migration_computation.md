# 04: Migration Rate Computation -- Methodology Comparison

**Scope**: How raw migration rates are derived from historical data. This document covers the computation of base migration rates from source data through the averaging step. It does **not** cover how rates are projected forward (convergence interpolation, scenario multipliers, or the projection engine itself).

**Last Updated**: 2026-02-20

---

## Table of Contents

1. [SDC 2024 Approach](#1-sdc-2024-approach)
2. [Current 2026 Approach](#2-current-2026-approach)
3. [Key Differences](#3-key-differences)
4. [Formulas and Calculations](#4-formulas-and-calculations)
5. [Rationale for Changes](#5-rationale-for-changes)

---

## 1. SDC 2024 Approach

### 1.1 Data Source

**Census Residual Method (Component Method II)**, using decennial Census and intercensal Population Estimates Program (PEP) data for the period 2000-2020.

Source files:
- `Mig Rate 2000-2020_final.xlsx` (migration rate derivation workbook)
- `Projections_Base_2023.xlsx` (primary calculation engine)

The residual method infers migration as the difference between observed population and expected population (after aging and applying survival). No direct migration survey or administrative records are used.

### 1.2 Periods Used

Four five-year periods, all of equal length:

| Period | Start Pop Source | End Pop Source | Length |
|--------|:---------------:|:--------------:|:------:|
| 2000-2005 | Census 2000 | PEP 2005 | 5 years |
| 2005-2010 | PEP 2005 | Census 2010 | 5 years |
| 2010-2015 | Census 2010 | PEP 2015 | 5 years |
| 2015-2020 | PEP 2015 | Census 2020 | 5 years |

All four periods are given **equal weight** in the simple average that produces the final base migration rate.

### 1.3 Residual Calculation Formula

For each county, sex, and five-year age group:

```
Expected_Pop[age+5, t+5] = Pop[age, t] * Survival_Rate[age, sex]
Migration[age+5]         = Actual_Pop[age+5, t+5] - Expected_Pop[age+5, t+5]
```

For the open-ended 85+ group:
```
Expected_85plus = (Pop[80-84, t] * Surv[80-84]) + (Pop[85+, t] * Surv[85+])
Migration_85plus = Actual_Pop[85+, t+5] - Expected_85plus
```

The 0-4 birth cohort at the end of a period cannot be computed via the residual method (no starting cohort to age forward). Children born to in-migrants are subsumed into the migration component.

**Cell dimensions**: 18 age groups x 2 sexes x 53 counties = **1,908 cells per period**, x 4 periods = 7,632 total cell-period observations.

### 1.4 Rate Expression

Rates are expressed as a proportion of the expected (naturally surviving) population:

```
Migration_Rate[age, sex, county, period] = Migration / Expected_Pop
```

The four period-specific rates are then averaged into a single base rate:

```
Base_Rate[age, sex, county] = (1/4) * SUM over 4 periods [ Migration_Rate[period] ]
```

There is **no annualization**. The rate is a 5-year-period rate applied directly in the 5-year-step projection engine.

### 1.5 Bakken Dampening

SDC applied a general dampening to reduce the influence of the Bakken oil boom (2010-2020) on projected migration. From the SDC report:

> "The rate of migration was typically reduced to about 60% of what was found."

However, the dampening is not applied to the base rate directly. Instead, SDC uses **period-specific multipliers** on the migration amount during projection:

| Projection Period | Multiplier | Rationale |
|:-----------------:|:----------:|-----------|
| 2020-2025 | 0.2 | COVID-19 adjustment, post-boom transition |
| 2025-2030 | 0.6 | Bakken dampening |
| 2030-2035 | 0.6 | Bakken dampening |
| 2035-2040 | 0.5 | Further reduction (conservative estimate) |
| 2040-2045 | 0.7 | Gradual return toward historical patterns |
| 2045-2050 | 0.7 | Gradual return toward historical patterns |

These multipliers create a U-shaped migration trajectory: very low immediately after the base year, increasing through mid-projection, then stabilizing at 70% of the computed base rate.

**Application formula** (during projection, not rate computation):
```
Effective_Migration = Natural_Growth * Base_Rate * Period_Multiplier
```

### 1.6 Manual Adjustments

SDC applies manual person-count adjustments beyond the formula-based calculations, totaling approximately **~32,000 person-adjustments per 5-year projection period**. These are added as a separate column in the projection workbook:

```
Population[age, county, t+5] = Natural_Growth + Migration + Adjustment[age, sex, county, period]
```

Both positive and negative adjustments are applied by county, age group, and sex. The net effect varies by period.

### 1.7 College-Age Manual Corrections

From the SDC report:

> "Counties with significant college age populations typically required additional adjustments as the algorithm tends to not capture the in- and out-migration of college age residents as well as it should."

Affected counties include Grand Forks (UND), Cass (NDSU), Ward (Minot State), and others. The adjustments are manual and embedded in the workbook; no algorithmic formula is documented.

### 1.8 Sex-Ratio Adjustments

From the SDC report:

> "The rate of male migration was further reduced compared to female migration as the pattern found from 2000 to 2020 when in-migration was dominated by males is unlikely to continue into the future and would have resulted in unrealistic sex ratio in future years."

This is a manual reduction applied to male migration rates in oil-impacted counties. The exact factors are not published; the adjustment is embedded in the workbook formulas and manual adjustment columns.

---

## 2. Current 2026 Approach

### 2.1 Data Source

**Census Residual Method**, using Census PEP population estimates for the period 2000-2024 (Vintage 2025). The switch from IRS county-to-county flows to Census PEP was decided in **ADR-035**.

Source files and loaders:
- Census 2000 county age-sex: `Census 2000 County Age and Sex.xlsx`
- PEP 2010-2019 intercensal: `cc-est2019-agesex-38.xlsx`
- PEP 2020-2024: `cc-est2024-agesex-all.parquet`
- SDC 2024 base population (for 2020 snapshot): `base_population_by_county.csv`
- Survival rates: `survival_rates_sdc_2024_by_age_group.csv` (CDC ND 2020 life tables)
- PEP county components (for recalibration): `pep_county_components_2000_2025.parquet`

Implementation: `cohort_projections/data/process/residual_migration.py`

### 2.2 Periods Used

Five periods, with the final period being 4 years instead of 5:

| Period | Start Pop Source | End Pop Source | Length |
|--------|:---------------:|:--------------:|:------:|
| 2000-2005 | Census 2000 | PEP 2005 | 5 years |
| 2005-2010 | PEP 2005 | PEP 2010 | 5 years |
| 2010-2015 | PEP 2010 | PEP 2015 | 5 years |
| 2015-2020 | PEP 2015 | Census 2020 | 5 years |
| 2020-2024 | Census 2020 | PEP 2024 | **4 years** |

All five periods receive **equal weight** in the simple average. The additional period (2020-2024) incorporates post-pandemic recovery data not available to SDC in 2024.

Configuration reference (`config/projection_config.yaml`):
```yaml
residual:
  periods:
    - [2000, 2005]
    - [2005, 2010]
    - [2010, 2015]
    - [2015, 2020]
    - [2020, 2024]
  survival_source: "CDC_ND_2020"
  averaging: "simple_average"
```

### 2.3 Residual Calculation Formula

Identical to SDC in structure, but with **annualization** of the period rate:

```
Expected_Pop[age+5, t+5] = Pop[age, t] * Survival_Rate_adj[age, sex]
Migration[age+5]         = Actual_Pop[age+5, t+5] - Expected_Pop[age+5, t+5]
Migration_Rate_period    = Migration / Expected_Pop     (if Expected_Pop > 0)
Migration_Rate_annual    = annualize(Migration_Rate_period, period_length)
```

**Survival rate adjustment for non-5-year periods**:
```
Surv_adj = Surv_5yr ^ (period_length / 5)
```

For the 2020-2024 period (4 years): `Surv_4yr = Surv_5yr ^ (4/5) = Surv_5yr ^ 0.8`

**Cell dimensions**: 18 age groups x 2 sexes x 53 counties = 1,908 cells per period, x 5 periods = 9,540 total cell-period observations.

### 2.4 Annualization

Because the projection engine operates on **annual** time steps (1-year intervals) rather than SDC's 5-year steps, period rates must be converted to annual equivalents. The annualization formula uses compound-rate decomposition:

```
annual_rate = (1 + period_rate) ^ (1 / period_length) - 1
```

For the edge case where `period_rate <= -1.0` (complete cohort loss), the annual rate is clamped to `-1.0`.

### 2.5 Oil County Boom Dampening

Dampening is applied to the **per-period residual rates** (not during projection) for designated Bakken oil counties during boom periods. This modifies the base rates before averaging.

**Target counties** (FIPS codes):
| FIPS | County | Notes |
|------|--------|-------|
| 38105 | Williams | Williston; Bakken hub |
| 38053 | McKenzie | Watford City; peak boom growth |
| 38061 | Mountrail | Stanley; fringe Bakken |
| 38025 | Dunn | Killdeer; fringe Bakken |
| 38089 | Stark | Dickinson; regional center |
| 38007 | Billings | Added ADR-051; +88% undampened 2010-2015 |

**Boom periods and dampening factors** (ADR-040 + ADR-051):

| Boom Period | Dampening Factor | ADR |
|:-----------:|:----------------:|:---:|
| 2005-2010 | 0.50 | ADR-051 (was 0.60 in ADR-040) |
| 2010-2015 | 0.40 | ADR-051 (was 0.60 in ADR-040) |
| 2015-2020 | 0.50 | ADR-040 extended, ADR-051 tuned |

Non-boom periods (2000-2005, 2020-2024) receive **no dampening** (factor = 1.0).

Configuration reference:
```yaml
dampening:
  enabled: true
  factor:
    "2005-2010": 0.50
    "2010-2015": 0.40
    "2015-2020": 0.50
    default: 0.60
  counties:
    - "38105"  # Williams
    - "38053"  # McKenzie
    - "38061"  # Mountrail
    - "38025"  # Dunn
    - "38089"  # Stark
    - "38007"  # Billings
  boom_periods:
    - [2005, 2010]
    - [2010, 2015]
    - [2015, 2020]
```

**Application** (in `apply_period_dampening()`):
```
dampened_rate = undampened_rate * factor
dampened_migration = undampened_migration * factor
```

Dampening is applied only to rows where `county_fips` is in the target list **and** the period matches a boom period.

### 2.6 Male Migration Dampening

During boom periods (2005-2010 and 2010-2015), male migration rates are additionally reduced by a factor of 0.80 to correct for the disproportionately male in-migration during the oil boom, which would otherwise produce unrealistic sex ratios in projections.

**Configuration**:
```yaml
male_dampening:
  enabled: true
  factor: 0.80
  boom_periods:
    - [2005, 2010]
    - [2010, 2015]
```

**Application** (in `apply_male_migration_dampening()`): For boom periods, all male rows (all counties, all age groups) have their migration rate and net migration multiplied by 0.80.

**Combined effective dampening for oil county males during 2010-2015**:
```
effective_factor = oil_dampening * male_dampening = 0.40 * 0.80 = 0.32
```

### 2.7 Reservation County PEP Recalibration (ADR-045)

Three AIAN reservation counties have migration rates recalibrated using PEP component totals because the residual method systematically overestimates out-migration due to Census undercounts on tribal lands and application of statewide survival rates to populations with lower life expectancy.

**Target counties**:
| FIPS | County | Reservation |
|------|--------|-------------|
| 38005 | Benson | Fort Berthold (partial) / Spirit Lake |
| 38085 | Sioux | Standing Rock |
| 38079 | Rolette | Turtle Mountain |

**Two-tier method** (applied per period, after dampening and male dampening, before averaging):

**Tier 1 -- Hybrid scaling** (when PEP and residual have the same sign and |residual| >= threshold):
```
k = PEP_total / Residual_total
scaled_rate = original_rate * k
scaled_migration = original_migration * k
```

This preserves the county-specific age-sex shape while anchoring the magnitude to PEP.

**Tier 2 -- Rogers-Castro fallback** (when sign reversal occurs or |residual| < near_zero_threshold):
The PEP total is distributed across age groups and sexes using the Rogers-Castro standard migration age pattern with a 50/50 sex split, then converted to annualized rates by dividing by expected population.

**Configuration**:
```yaml
pep_recalibration:
  enabled: true
  counties: ["38005", "38085", "38079"]
  pep_data_path: "data/processed/pep_county_components_2000_2025.parquet"
  fallback_method: "rogers_castro"
  near_zero_threshold: 10
```

**PEP period convention**: For period `(start_year, end_year)`, sum PEP annual `netmig` values for years `(start_year + 1)` through `end_year` inclusive. Example: period (2000, 2005) sums PEP years 2001, 2002, 2003, 2004, 2005.

### 2.8 College-Age Smoothing (ADR-049)

College counties exhibit extreme in/out migration for ages 15-24 due to student enrollment cycles that do not represent permanent migration. The smoothing blends county-specific rates with the statewide average at the **period level** (before averaging).

**Target counties**:
| FIPS | County | Institution |
|------|--------|-------------|
| 38035 | Grand Forks | University of North Dakota |
| 38017 | Cass | North Dakota State University |
| 38101 | Ward | Minot State University |
| 38015 | Burleigh | University of Mary, Bismarck State |

**Target age groups**: 15-19, 20-24

**Method**: Blend with statewide average (method = "smooth"):
```
smoothed_rate = blend_factor * county_rate + (1 - blend_factor) * statewide_average_rate
```

Where `blend_factor = 0.5` (50% county-specific + 50% statewide average).

The statewide average is computed across all 53 counties for the same age-group x sex combination within the same period.

**Configuration**:
```yaml
college_age:
  enabled: true
  method: "smooth"
  counties: ["38035", "38017", "38101", "38015"]
  age_groups: ["15-19", "20-24"]
  blend_factor: 0.5
```

**Pipeline placement** (ADR-049 fix): College-age smoothing is applied to each period's rates individually (before the period-level file is saved and before period averaging). This ensures both the averaged rates and the period-level rates consumed by the convergence pipeline inherit the smoothing. Prior to ADR-049, smoothing was applied only after averaging, causing the convergence pipeline to read unsmoothed period-level rates.

### 2.9 Period Averaging

After all adjustments (dampening, male dampening, PEP recalibration, college-age smoothing), rates are averaged across all periods using a simple mean:

```
Averaged_Rate[age, sex, county] = (1/N) * SUM over N periods [ adjusted_rate[period] ]
```

Where N = 5 (the number of periods). Each period receives equal weight regardless of length.

### 2.10 Pipeline Step Order

The complete pipeline in `run_residual_migration_pipeline()` executes in this order:

```
Step 1: Load population snapshots (2000, 2005, 2010, 2015, 2020, 2024)
Step 2: Load survival rates (CDC ND 2020)
Step 3: Compute residual migration for each period
Step 4: Apply oil-county boom dampening (per period)
Step 5: Apply male migration dampening (per period, boom periods only)
Step 6: Apply PEP recalibration for reservation counties (per period)
Step 7: Apply college-age smoothing (per period)
Step 8: Combine all period rates; save period-level file
Step 9: Average rates across periods; save averaged file
Step 10: Save metadata JSON
```

---

## 3. Key Differences

### 3.1 Summary Comparison Table

| Dimension | SDC 2024 | Current 2026 | Impact |
|-----------|----------|--------------|--------|
| **Data source** | Census residual (2000-2020) | Census residual (2000-2024) via PEP | Additional 4 years including post-pandemic recovery |
| **Number of periods** | 4 (all 5-year) | 5 (four 5-year + one 4-year) | More data points; includes 2020-2024 |
| **Period lengths** | Uniform 5-year | 5,5,5,5,4 years | 4-year period requires survival rate and annualization adjustments |
| **Time step** | 5-year projection intervals | Annual (1-year) projection intervals | Rates must be annualized for annual engine |
| **Age groups** | 18 five-year groups (0-4 to 85+) | 18 five-year groups (same for residual computation) | Identical at computation stage; engine uses single-year ages via distribution |
| **Rate annualization** | None (5-year rate used directly) | Compound annualization: `(1+r)^(1/n) - 1` | Required for annual engine compatibility |
| **Boom dampening scope** | General 60% reduction at projection time | Period-specific factors (0.40-0.50) applied to base rates | More granular; applied earlier in pipeline |
| **Boom dampening application** | Period multiplier during projection | Applied to residual rates before averaging | Fundamentally different pipeline placement |
| **Boom periods defined** | Not explicitly period-specific in rate computation | 2005-2010, 2010-2015, 2015-2020 | 2015-2020 added per ADR-040 |
| **Oil county list** | Not explicitly listed in published methodology | 6 counties: Williams, McKenzie, Mountrail, Dunn, Stark, Billings | Billings added per ADR-051 |
| **Period multipliers** | 0.2, 0.6, 0.6, 0.5, 0.7, 0.7 | N/A (dampening applied to base rates) | Different mechanism entirely |
| **Male dampening** | Manual sex-ratio adjustment (undocumented factors) | Algorithmic: 0.80 factor on all male rates during 2005-2015 | Reproducible; affects all counties, not just oil counties |
| **College-age correction** | Manual adjustments (~32K person-adjustments/period) | Algorithmic: 50% blend with statewide average for 4 counties, ages 15-24 | Systematic and reproducible |
| **Reservation correction** | None documented | PEP-anchored recalibration (hybrid scaling + Rogers-Castro fallback) for 3 counties | Addresses ~2x overestimation of out-migration |
| **Manual adjustments** | ~32,000 person-adjustments per period | None | Fully algorithmic pipeline |
| **Reproducibility** | Requires specific workbook; adjustments not fully documented | Fully reproducible from config + code | Major governance improvement |

### 3.2 Dampening Factor Comparison

**SDC 2024 -- projection-time multipliers** (applied to migration amount during each projection step):

| Projection Period | SDC Multiplier |
|:-----------------:|:--------------:|
| 2020-2025 | 0.20 |
| 2025-2030 | 0.60 |
| 2030-2035 | 0.60 |
| 2035-2040 | 0.50 |
| 2040-2045 | 0.70 |
| 2045-2050 | 0.70 |

**Current 2026 -- base rate dampening** (applied to historical residual rates before averaging):

| Historical Period | Oil County Factor | Male Factor (boom only) | Combined (oil county males) |
|:-----------------:|:-----------------:|:-----------------------:|:---------------------------:|
| 2000-2005 | 1.00 (no dampening) | 1.00 | 1.00 |
| 2005-2010 | 0.50 | 0.80 | 0.40 |
| 2010-2015 | 0.40 | 0.80 | 0.32 |
| 2015-2020 | 0.50 | 1.00 | 0.50 |
| 2020-2024 | 1.00 (no dampening) | 1.00 | 1.00 |

These two approaches are structurally different. SDC dampens at projection time (reducing projected migration), while the current system dampens at rate computation time (reducing the historical inputs to the average). The current approach produces a single "already-dampened" average rate that feeds into the convergence interpolation schedule (see the convergence methodology document for forward projection details).

---

## 4. Formulas and Calculations

### 4.1 Core Residual Migration (Both Systems)

Both SDC and the current system use the same fundamental residual formula:

**Standard age groups (0-4 through 80-84):**
```
expected[age+5, t+5] = pop[age, t] * surv[age, sex]
migration[age+5]     = pop_actual[age+5, t+5] - expected[age+5, t+5]
rate_period          = migration / expected            (if expected > 0, else 0)
```

**Open-ended 85+ group:**
```
expected_85plus = (pop[80-84, t] * surv[80-84, sex]) + (pop[85+, t] * surv[85+, sex])
migration_85plus = pop_actual[85+, t+5] - expected_85plus
rate_85plus_period = migration_85plus / expected_85plus    (if expected_85plus > 0)
```

**0-4 birth cohort:**
Cannot be computed via residual (no starting cohort). Both systems assign `migration_rate = 0.0` for this group; births are handled separately by the fertility component.

### 4.2 Survival Rate Adjustment (Current System Only)

For periods not exactly 5 years long (i.e., the 2020-2024 period at 4 years):

```
surv_adjusted[age, sex] = surv_5yr[age, sex] ^ (period_length / 5)
```

Example for the 2020-2024 period:
```
surv_4yr = surv_5yr ^ (4/5) = surv_5yr ^ 0.80
```

SDC does not need this adjustment because all periods are exactly 5 years.

### 4.3 Annualization (Current System Only)

Converts the multi-year period rate to an annual equivalent:

```
annual_rate = (1 + period_rate) ^ (1 / period_length) - 1
```

**Boundary condition**: If `period_rate <= -1.0`, return `-1.0` (complete cohort loss cannot be meaningfully annualized).

**Numerical example** (McKenzie County, 20-24 Males, 2010-2015):
```
period_rate = +0.60 (60% net in-migration over 5 years)
annual_rate = (1 + 0.60)^(1/5) - 1 = 1.60^0.2 - 1 = 0.0986 (~9.86%/year)
```

Compare to simple division: `0.60 / 5 = 0.12` (12%/year). The compound annualization is more conservative for large rates and more accurate for compounding effects.

### 4.4 Oil County Boom Dampening

**SDC formula** (at projection time):
```
migration_amount = natural_growth * base_rate * period_multiplier
```

Where `period_multiplier` is from the table in Section 1.5 (0.2 to 0.7 depending on projection period).

**Current formula** (at rate computation time):
```
dampened_rate     = undampened_rate * factor[period]
dampened_migration = undampened_migration * factor[period]
```

Where `factor[period]` is the period-specific dampening factor from the config (0.40 to 0.50 depending on historical period), applied only to rows matching both the county FIPS list and the boom period list.

### 4.5 Male Migration Dampening (Current System Only)

Applied after oil county dampening, during boom periods only:

```
dampened_male_rate      = undampened_male_rate * male_factor
dampened_male_migration = undampened_male_migration * male_factor
```

Where `male_factor = 0.80` and boom periods are `[2005-2010, 2010-2015]`.

This applies to **all counties** (not just oil counties) during boom periods, for male rows only.

**Combined effective factor for oil county males, 2010-2015**:
```
effective = oil_factor * male_factor = 0.40 * 0.80 = 0.32
```

In other words, only 32% of the observed 2010-2015 boom-era male migration in oil counties is retained in the base rate.

### 4.6 PEP Recalibration Scaling Factor (Current System Only)

For reservation counties (Benson 38005, Sioux 38085, Rolette 38079):

**Step 1 -- Compute PEP total for the period:**
```
PEP_total = SUM(pep_data.netmig)
            where geoid = county
            and year > period_start
            and year <= period_end
```

Example: For Sioux County, period (2010, 2015):
```
PEP_total = netmig[2011] + netmig[2012] + netmig[2013] + netmig[2014] + netmig[2015]
```

**Step 2 -- Compute residual total for the same county-period:**
```
Residual_total = SUM(net_migration)
                 for all age-sex rows where county_fips = county
                 in this period's residual results
```

**Step 3 -- Choose method:**

| Condition | Method |
|-----------|--------|
| Same sign AND \|Residual_total\| >= 10 | Hybrid scaling |
| Different signs (sign reversal) | Rogers-Castro fallback |
| \|Residual_total\| < 10 (near zero) | Rogers-Castro fallback |

**Tier 1 -- Hybrid scaling:**
```
k = PEP_total / Residual_total
recalibrated_rate      = original_rate * k
recalibrated_migration = original_migration * k
```

This preserves the age-sex shape from the residual method while matching the PEP total.

**Tier 2 -- Rogers-Castro fallback:**

When scaling cannot be applied (sign reversal or near-zero residual), the PEP total is redistributed using the Rogers-Castro standard migration schedule.

### 4.7 Rogers-Castro Fallback Distribution (Current System Only)

**Step 1 -- Compute Rogers-Castro propensity weights** for 18 five-year age groups:

The system calls `get_standard_age_migration_pattern(peak_age=25, method="rogers_castro")` from `migration_rates.py`, which implements the standard Rogers-Castro model:

```
M(x) = a1 * exp(-alpha1 * x) + a2 * exp(-alpha2 * (x - mu2) - exp(-lambda2 * (x - mu2))) + c
```

With parameters:
- `a1 = 0.02` (childhood migration with parents)
- `alpha1 = 0.08` (rate of decrease in childhood component)
- `a2 = 0.06` (young adult peak magnitude)
- `mu2 = 25` (peak migration age)
- `alpha2 = 0.5` (rate of decrease from peak)
- `lambda2 = 0.4` (shape of peak)
- `c = 0.001` (baseline constant)

Single-year-of-age propensities are aggregated to five-year groups and normalized so they sum to 1.0.

**Step 2 -- Split by sex:**
```
male_total   = PEP_total * sex_ratio       (sex_ratio = 0.5)
female_total = PEP_total * (1 - sex_ratio)
```

**Step 3 -- Distribute to age groups:**
```
migration_count[age, sex] = sex_total * rc_weight[age]
```

**Step 4 -- Convert counts to rates:**
```
rate_period[age, sex] = migration_count[age, sex] / expected_pop[age, sex]
rate_annual[age, sex] = (1 + rate_period)^(1/period_length) - 1
```

Where `expected_pop[age, sex]` is the expected population for that county-period cell from the residual computation.

### 4.8 College-Age Smoothing Blend (Current System Only)

For each college county, for age groups 15-19 and 20-24, within each period:

**Step 1 -- Compute statewide average rate:**
```
statewide_avg[age_group, sex] = MEAN(migration_rate)
                                across all 53 counties
                                for the same age_group and sex
                                within the same period
```

**Step 2 -- Blend:**
```
smoothed_rate = blend_factor * county_rate + (1 - blend_factor) * statewide_avg
```

Where `blend_factor = 0.5`.

**Numerical example** (Cass County, 20-24 Males, one period):
```
county_rate    = 0.124  (12.4% annual -- reflects NDSU enrollment)
statewide_avg  = 0.038  (3.8% annual average across all 53 counties)
smoothed_rate  = 0.5 * 0.124 + 0.5 * 0.038 = 0.081  (8.1% annual)
```

### 4.9 Period Averaging

**SDC**:
```
Base_Rate = (1/4) * [rate_2000_05 + rate_2005_10 + rate_2010_15 + rate_2015_20]
```

Equal-weighted average of 4 periods. No annualization; rates are 5-year period rates.

**Current**:
```
Averaged_Rate = (1/5) * [rate_2000_05 + rate_2005_10 + rate_2010_15 + rate_2015_20 + rate_2020_24]
```

Equal-weighted average of 5 periods. All rates are already annualized before averaging. Dampening, male dampening, PEP recalibration, and college-age smoothing have already been applied to each period's rates before this step.

---

## 5. Rationale for Changes

### 5.1 Why PEP Over IRS

**ADR-035** documents this decision in detail. Key reasons:

1. **Comprehensiveness**: PEP captures both domestic and international migration. IRS county-to-county flows capture only domestic address changes, missing approximately 1,100-1,200 international migrants per year for North Dakota.

2. **Temporal coverage**: PEP provides 24 years (2000-2024) vs. IRS's 4 years (2019-2022). The IRS window coincides with the worst recent migration conditions (COVID-19 + Bakken bust aftermath), producing an average of -987 people/year vs. PEP's full-period average of +1,624 people/year.

3. **Methodological alignment with SDC**: The SDC 2024 projections use the Census residual method (2000-2020). Switching to PEP residual aligns our data source and allows direct comparison.

4. **Quantified impact**: The data source choice alone accounts for approximately 74,000-80,000 people by 2045 (~10% of North Dakota's population).

### 5.2 Why Period-Specific Dampening

SDC uses a single general dampening description ("typically reduced to about 60%") applied via projection-time multipliers. The current system uses period-specific factors applied to the base rates because:

1. **The evidence varies by period**: 2020-2025 data (available to us but not to SDC in early 2024) shows near-zero or negative growth for McKenzie and Williams counties, confirming that the 2010-2015 boom was largely transient. This justifies a more aggressive factor (0.40) for 2010-2015 than for 2005-2010 (0.50) or 2015-2020 (0.50).

2. **Earlier application is cleaner**: Applying dampening to the historical rates before averaging means the averaged rate already reflects the dampened reality. The projection engine and convergence pipeline then operate on "clean" rates without needing to know about boom dynamics.

3. **Billings County addition (ADR-051)**: Billings County (38007) was not in the original oil county list but showed +88% undampened growth in 2010-2015. The per-period factor approach made it straightforward to add this county to the dampening list.

### 5.3 Why Algorithmic Over Manual Adjustments

SDC applies approximately 32,000 manual person-adjustments per 5-year projection period. The current system replaces all manual adjustments with algorithmic corrections because:

1. **Reproducibility**: Manual adjustments embedded in an Excel workbook cannot be independently verified or systematically varied for scenario analysis. Every adjustment in the current system is derived from documented formulas and configuration parameters.

2. **Transparency**: Each algorithmic correction (dampening, male dampening, PEP recalibration, college-age smoothing) has a documented rationale (ADR) and a specific configuration block in `projection_config.yaml`.

3. **Auditability**: The metadata JSON output records exactly which corrections were applied, to which counties and periods, with what parameters. SDC's manual adjustments are embedded in workbook cells without systematic documentation.

4. **Scalability**: When new data becomes available (e.g., PEP Vintage 2026) or new evidence warrants adjustment (e.g., the dampening factor needs tuning), the change is a configuration update rather than a manual workbook revision.

### 5.4 Why Reservation Recalibration (ADR-045)

Three reservation counties were projected to decline 45-47% over 30 years, which is 2.7-11x their historical 20-year decline rates of -11% to -15%. Investigation revealed:

1. **Census undercounts on tribal lands**: Differential census coverage between decennial censuses inflates apparent population loss, which the residual method attributes to out-migration.

2. **Survival rate mismatch**: Statewide survival rates applied to AIAN populations with lower life expectancy cause excess deaths to be misattributed as out-migration (expected surviving population is too high).

3. **PEP provides a more reliable total**: PEP net migration totals for these counties show out-migration roughly half the residual estimate (median residual/PEP ratio of 1.42-2.00x). In some periods, PEP shows net in-migration while the residual shows net out-migration (sign reversal), confirming the residual method is qualitatively wrong.

The hybrid approach (scale when possible, Rogers-Castro fallback when not) preserves the most information from both sources.

### 5.5 Why College-Age Smoothing

University counties show extreme migration rates for ages 15-24 that reflect enrollment cycles rather than permanent migration. Example: Cass County's raw 20-24 annual migration rate of 12.4% implies the equivalent of 12% of Fargo's young adult population arrives as new permanent residents every year, which is implausible.

The 50% blend with the statewide average reduces the signal while preserving the directional pattern (net in-migration for university counties). SDC addresses the same issue with manual adjustments; the current system uses the algorithmic blend.

ADR-049 corrected a bug where smoothing was applied only to the averaged rates, not to the period-level rates consumed by the convergence pipeline. This caused Cass County to be projected at +63% instead of approximately +48%.

### 5.6 Why Annualization

SDC operates with 5-year time steps and directly applies 5-year rates. The current system uses annual time steps (1-year intervals), which provides:

1. **Finer temporal resolution**: Annual projections support more granular output and allow mid-decade interpolation.
2. **More precise compounding**: Annual rates compound correctly over arbitrary horizons without assumptions about within-period timing.
3. **Compatibility with mortality improvement**: The annual mortality improvement factor (0.5%/year per `config/projection_config.yaml`) requires annual time steps to apply correctly.

The compound annualization formula `(1+r)^(1/n) - 1` is mathematically exact for converting a multi-year rate to its annual equivalent, assuming continuous compounding within the period.

---

## References

### Architecture Decision Records (ADRs)

| ADR | Title | Relevance |
|-----|-------|-----------|
| [ADR-003](../../governance/adrs/003-migration-rate-processing.md) | Migration Rate Processing Methodology | Original migration processing framework (IRS-based) |
| [ADR-035](../../governance/adrs/035-migration-data-source-census-pep.md) | Census PEP Components for Migration | Decision to switch from IRS to PEP |
| [ADR-040](../../governance/adrs/040-extend-boom-dampening-2015-2020.md) | Extend Boom Dampening to 2015-2020 | Added 2015-2020 to boom periods |
| [ADR-045](../../governance/adrs/045-reservation-county-pep-recalibration.md) | Reservation County PEP Recalibration | PEP-anchored migration for reservation counties |
| [ADR-049](../../governance/adrs/049-college-age-smoothing-convergence-pipeline.md) | College-Age Smoothing Propagation | Fixed smoothing to apply at period level |
| [ADR-051](../../governance/adrs/051-oil-county-dampening-recalibration.md) | Oil County Dampening Recalibration | Period-specific dampening factors |

### Source Code

| File | Purpose |
|------|---------|
| [`cohort_projections/data/process/residual_migration.py`](../../../cohort_projections/data/process/residual_migration.py) | Full residual migration pipeline |
| [`config/projection_config.yaml`](../../../config/projection_config.yaml) | Migration configuration (dampening, periods, counties) |
| [`sdc_2024_replication/METHODOLOGY_SPEC.md`](../../../sdc_2024_replication/METHODOLOGY_SPEC.md) | SDC 2024 technical specification |

### SDC Source Files

| File | Purpose |
|------|---------|
| `Mig Rate 2000-2020_final.xlsx` | SDC migration rate derivation workbook |
| `Projections_Base_2023.xlsx` | SDC primary projection calculation engine |
| `sdc_2024_replication/data/migration_rates_by_county.csv` | SDC migration rates (53 counties x 18 age groups x 2 sexes) |
