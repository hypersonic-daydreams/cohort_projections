---
title: "Implementation Plan: Census Bureau Method Upgrade"
created: 2026-02-13T18:00:00-06:00
status: implementation-plan
author: Claude Code (Opus 4.6)
parent_document: docs/plans/census-method-assessment-and-path-forward.md
purpose: >
  Detailed implementation plan for upgrading the cohort projection engine
  from Rogers-Castro model-based migration to empirical age-specific residual
  rates with Census Bureau convergence interpolation and mortality improvement.
  Follows Path B from the assessment document, with all 6 design decisions resolved.
decisions_resolved:
  1_convergence_schedule: "5-10-5 (configurable)"
  2_mortality: "Option B — ND-adjusted Census Bureau survival ratios"
  3_age_groups: "5-year age groups (18 groups × 2 sexes = 36 rate cells)"
  4_fertility: "Hold constant"
  5_sdc_adjustments: "Replicate both college-age AND male migration adjustments"
  6_dampening: "Boom-era periods only (2005-2010, 2010-2015)"
---

# Implementation Plan: Census Bureau Method Upgrade

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Data Inventory & Gaps](#3-data-inventory--gaps)
4. [Phase 1: Empirical Age-Specific Migration Rates](#4-phase-1-empirical-age-specific-migration-rates)
5. [Phase 2: Age-Specific Convergence Interpolation](#5-phase-2-age-specific-convergence-interpolation)
6. [Phase 3: Census Bureau Mortality Improvement](#6-phase-3-census-bureau-mortality-improvement)
7. [Phase 4: Engine Wiring for Time-Varying Rates](#7-phase-4-engine-wiring-for-time-varying-rates)
8. [Phase 5: Validation & Comparison](#8-phase-5-validation--comparison)
9. [New Research Findings (from SDC 2024 Deep Dive)](#9-new-research-findings)
10. [Risk Register](#10-risk-register)

---

## 1. Overview

### Goal

Replace the current Rogers-Castro model-based migration pipeline with empirical
age-specific residual rates, add Census Bureau convergence interpolation for
time-varying migration, and wire Census Bureau mortality improvement into the
projection engine.

### Current State → Target State

| Component | Current | Target |
|-----------|---------|--------|
| Migration age profile | Rogers-Castro bell curve (generic) | Empirical residual rates per county/age/sex |
| Migration over time | Constant rates (same every year) | 3-phase convergence (recent → medium → long-term) |
| Dampening | Uniform 60% all periods (oil counties) | 60% boom-era only (2005-2010, 2010-2015) |
| Mortality | Constant CDC rates (no improvement) | Census Bureau projected survival ratios (ND-adjusted) |
| Age resolution | Single-year ages 0-90 via model | 18 five-year groups × 2 sexes = 36 cells/county |

### Phase Dependencies

```
Phase 1: Empirical age-specific rates ──────┐
Phase 2: Age-specific convergence ──────────┤── Phase 4: Engine wiring ── Phase 5: Validation
Phase 3: Mortality improvement (independent)┘
```

Phase 3 can proceed in parallel with Phases 1-2 since it has no migration
dependencies.

---

## 2. Architecture Summary

### Current Data Flow

```
02_run_projections.py
  └─ load_demographic_rates()
       ├─ fertility_rates.parquet          → DataFrame (shared across counties)
       ├─ survival_rates.parquet           → DataFrame (shared across counties)
       └─ migration_rates_pep_*.parquet    → dict[fips, DataFrame] (per-county)
            └─ Each DataFrame: 1,092 rows (91 ages × 2 sexes × 6 races)
               Uses Rogers-Castro distribution of aggregate PEP net migration

  └─ run_multi_geography_projections()
       └─ For each county FIPS:
            └─ CohortComponentProjection(
                 base_population,
                 fertility_rates,      ← shared
                 survival_rates,       ← shared
                 migration_rates,      ← per-county, CONSTANT across years
               )
               └─ project_single_year(year)  ← uses same rates every year
```

### Target Data Flow

```
02_run_projections.py
  └─ load_demographic_rates()
       ├─ fertility_rates.parquet          → DataFrame (shared, constant)
       ├─ survival_rates_by_year           → dict[year, DataFrame] (time-varying)
       │    └─ Census Bureau NP2023 survival ratios, ND-adjusted
       └─ migration_rates_by_year          → dict[year, dict[fips, DataFrame]]
            └─ Each DataFrame: 36 rows (18 age groups × 2 sexes)
               Empirical residual rates with convergence interpolation

  └─ run_multi_geography_projections()
       └─ For each county FIPS:
            └─ CohortComponentProjection(
                 base_population,
                 fertility_rates,                ← shared, constant
                 survival_rates_by_year,         ← TIME-VARYING
                 migration_rates_by_year[fips],  ← TIME-VARYING, per-county
               )
               └─ project_single_year(year)
                    ├─ survival = survival_rates_by_year.get(year, default)
                    └─ migration = migration_rates_by_year.get(year, default)
```

### Key Interface Changes

| Module | Current Signature | New Signature |
|--------|-------------------|---------------|
| `CohortComponentProjection.__init__` | `migration_rates: DataFrame` | + `migration_rates_by_year: dict[int, DataFrame] \| None` |
| `CohortComponentProjection.__init__` | `survival_rates: DataFrame` | + `survival_rates_by_year: dict[int, DataFrame] \| None` |
| `project_single_year` | Uses `self.migration_rates` | Looks up `self.migration_rates_by_year[year]` if available |
| `load_demographic_rates` | Returns `(fert, surv, mig)` | Returns `(fert, surv, mig, surv_by_year, mig_by_year)` |

All changes are backward-compatible: if `*_by_year` is None, falls back to
constant rates (existing behavior).

---

## 3. Data Inventory & Gaps

### Available Data

| File | Content | Age Detail | Years | Status |
|------|---------|------------|-------|--------|
| `cc-est2019-agesex-38.xlsx` | PEP county pop by age/sex | 18 five-year groups | 2010-2019 | Ready |
| `Census 2000 County Age and Sex.xlsx` | County pop by age/sex | 19 AGEGRP codes (5yr) | 2000-2010 | Ready |
| `sdc_2024_replication/data/base_population_by_county.csv` | 2020 Census base pop | 18 five-year groups × 2 sexes | 2020 | Ready |
| `np2023_a4_survival_ratios.csv` | Census Bureau projections | Single-year 0-100 | 2023-2100 | Ready |
| `data/processed/sdc_2024/survival_rates_sdc_2024_by_age_group.csv` | CDC ND life tables | 18 five-year groups | 2020 | Ready |
| `cc-est2020int-alldata.parquet` (shared-data) | PEP intercensal age/sex/race | 18 AGEGRP codes (long format) | 2010-2020 | Ready |
| `cc-est2024-agesex-all.parquet` (shared-data) | PEP postcensal age/sex | Wide format (96 cols) | 2020-2024 | Ready |
| `pep_county_components_2000_2024.parquet` | PEP aggregate components | Total only (no age) | 2000-2024 | Ready |
| `migration_rates_pep_baseline.csv` | Current Rogers-Castro rates | Single-year 0-90 | Constant | Will be replaced |

### Data Gaps

**No remaining data gaps.** All time points (2000-2024) have actual Census
Bureau data available. See matrix below.

**Note:** 2005 and 2015 are NOT gaps. Both are directly available as actual
PEP intercensal estimates (see matrix below). No interpolation needed.

**2020-2024 data**: `cc-est2024-agesex-all.csv` was published by Census Bureau
on June 26, 2025 and downloaded on 2026-02-13. Available as parquet at
`shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`.
**Format note**: This file uses wide format (age columns like `AGE04_TOT`,
`AGE85PLUS_FEM`) unlike the 2010-2020 file which uses long format (`AGEGRP`
column). The data loader must pivot wide-to-long.

### Assembling the Historical Population Matrix

For the residual migration calculation, we need county × age group × sex
populations at the following time points:

```
Time Point    Source                                            Status
────────────────────────────────────────────────────────────────────────
2000          Census 2000 County Age and Sex.xlsx                ✓ Available
              (ESTIMATESBASE2000 or POPESTIMATE2000 column)
2005          Census 2000 County Age and Sex.xlsx                ✓ Available
              (POPESTIMATE2005 column — actual PEP estimate)
2010          cc-est2019-agesex-38.xlsx (YEAR=1, Census base)   ✓ Available
              OR Census 2000 file (CENSUS2010POP column)
2015          cc-est2019-agesex-38.xlsx (YEAR=6)                ✓ Available
              (actual PEP intercensal estimate)
2020          sdc_2024_replication/data/base_population_by_county.csv
              OR cc-est2020int-alldata.parquet (YEAR=11)         ✓ Available
2024          cc-est2024-agesex-all.parquet (shared-data)        ✓ Available
              (wide format — must pivot age columns to long)
```

All 6 time points use actual Census Bureau population estimates.
This enables 5 complete periods for residual migration calculation.

---

## 4. Phase 1: Empirical Age-Specific Migration Rates

### 4.1 Objective

Replace Rogers-Castro model-based migration with empirical residual rates
computed from historical Census population data, matching SDC 2024 methodology.

### 4.2 New Module: `cohort_projections/data/process/residual_migration.py`

This is the primary new code artifact. It replaces the Rogers-Castro pipeline
for computing age-specific migration rates.

#### 4.2.1 Core Function: `compute_residual_migration_rates()`

```python
def compute_residual_migration_rates(
    pop_start: pd.DataFrame,       # Population at start of period
    pop_end: pd.DataFrame,         # Population at end of period
    survival_rates: pd.DataFrame,  # 5-year survival rates by age/sex
    period: tuple[int, int],       # (start_year, end_year)
) -> pd.DataFrame:
    """Compute age-sex-specific net migration rates via the residual method.

    Formula per age group and sex:
        expected_pop = pop_start[age] * survival_rate[age, sex]
        migration = pop_end[age+5] - expected_pop
        migration_rate = migration / expected_pop

    For the 85+ open-ended group:
        expected_pop = (pop_start[80-84] * survival[80-84]) + (pop_start[85+] * survival[85+])
        migration = pop_end[85+] - expected_pop

    Args:
        pop_start: DataFrame with [county_fips, age_group, sex, population]
        pop_end: DataFrame with [county_fips, age_group, sex, population]
        survival_rates: DataFrame with [age_group, sex, survival_rate_5yr]
        period: Tuple of (start_year, end_year) for metadata

    Returns:
        DataFrame with [county_fips, age_group, sex, migration_rate, period_start,
                        period_end, expected_pop, actual_pop, net_migration]
    """
```

**Age group shifting logic:**
- Input age group 0-4 at time t produces survivors in age group 5-9 at t+5
- Input age group 5-9 at time t produces survivors in age group 10-14 at t+5
- ...
- Input age group 75-79 at time t produces survivors in age group 80-84 at t+5
- Input age group 80-84 at time t contributes to 85+ pool at t+5
- Input age group 85+ at time t contributes to 85+ pool at t+5

**Birth cohort (age 0-4 at t+5):**
- Cannot compute residual migration for the 0-4 group at t+5 using the shift
  method because these children were born during the period
- SDC 2024 approach: compute births from fertility rates, apply infant survival,
  difference is infant/child migration
- Simpler approach: use aggregate PEP net migration for ages 0-4, distributed
  proportionally

#### 4.2.2 Period Assembly Function: `assemble_period_populations()`

```python
def assemble_period_populations(
    data_sources: dict[int, pd.DataFrame | Path],
) -> dict[int, pd.DataFrame]:
    """Assemble population-by-age-sex-county for all required time points.

    Loads Census/PEP data for all years (2000, 2005, 2010, 2015, 2020, 2024).
    All time points use actual Census Bureau intercensal/postcensal estimates
    — no interpolation needed.

    Data sources by year:
    - 2000: Census 2000 file, ESTIMATESBASE2000 column
    - 2005: Census 2000 file, POPESTIMATE2005 column (actual PEP estimate)
    - 2010: cc-est2019 file, YEAR=1 (Census 2010 base)
    - 2015: cc-est2019 file, YEAR=6 (actual PEP estimate)
    - 2020: sdc_2024_replication/data/base_population_by_county.csv
            OR cc-est2020int-alldata.parquet (YEAR=11)
    - 2024: cc-est2024-agesex-all.parquet (shared-data, wide format — pivot to long)

    Args:
        data_sources: Mapping from year to DataFrame or file path

    Returns:
        Dict mapping year to standardized DataFrame with columns:
        [county_fips, age_group, sex, population]
    """
```

#### 4.2.3 Dampening Function: `apply_period_dampening()`

```python
def apply_period_dampening(
    period_rates: pd.DataFrame,
    period: tuple[int, int],
    dampening_config: dict,
) -> pd.DataFrame:
    """Apply 60% dampening to oil county rates for boom-era periods only.

    Per Decision 6: dampening applies ONLY to 2005-2010 and 2010-2015.
    Pre-boom (2000-2005), post-boom (2015-2020), and recent (2020-2024)
    periods are NOT dampened.

    Args:
        period_rates: DataFrame with [county_fips, age_group, sex, migration_rate]
        period: Tuple (start_year, end_year)
        dampening_config: Config with 'factor', 'counties', 'boom_periods'

    Returns:
        DataFrame with dampened rates for qualifying counties/periods
    """
```

**Configuration (projection_config.yaml update):**
```yaml
rates:
  migration:
    domestic:
      dampening:
        enabled: true
        factor: 0.60
        boom_periods:
          - [2005, 2010]
          - [2010, 2015]
        counties:
          - "38105"  # Williams
          - "38053"  # McKenzie
          - "38061"  # Mountrail
          - "38025"  # Dunn
          - "38089"  # Stark
```

#### 4.2.4 SDC Adjustments: College-Age and Male Migration

**Decision 5 — What we know from SDC 2024 research:**
- SDC applied ~32,000 manual person-adjustments per 5-year projection period
- College-age adjustments for Grand Forks, Cass, Ward (+ possibly Burleigh,
  Richland, Stutsman, Rolette)
- Male migration rates dampened more than female in boom-era periods
- These were NOT formulaic — they were expert judgment

**Our implementation approach:**

```python
def apply_college_age_adjustment(
    rates: pd.DataFrame,
    college_counties: dict[str, dict],
    method: str = "smooth",
) -> pd.DataFrame:
    """Adjust migration rates for college counties where the residual method
    produces artifacts from student population churn.

    Method "smooth": For ages 15-24 in college counties, compute the ratio
    of the county's 15-24 rate to the state average 15-24 rate. If the
    ratio exceeds a threshold (e.g., 2.0), blend toward the state average:
        adjusted_rate = county_rate * (1 - blend_factor) + state_rate * blend_factor

    Method "cap": Cap the absolute value of 15-24 rates at a multiple of
    the county's 25-34 rate (which is less affected by student churn).

    Args:
        rates: DataFrame with all counties' rates
        college_counties: Dict mapping FIPS to county metadata (enrollment, etc.)
        method: "smooth" or "cap"

    Returns:
        DataFrame with adjusted rates for college counties
    """
```

```python
def apply_male_migration_dampening(
    rates: pd.DataFrame,
    period: tuple[int, int],
    boom_periods: list[tuple[int, int]],
    male_dampening_factor: float = 0.80,
) -> pd.DataFrame:
    """Apply additional dampening to male migration in boom-era periods.

    Per SDC 2024: male in-migration during 2005-2015 was driven by oil field
    employment and was higher than female in-migration. This pattern is
    unlikely to continue. Apply additional reduction to male rates in boom
    periods to prevent unrealistic sex ratios in projected population.

    Only applies to boom-era periods. Non-boom periods are not adjusted.

    Args:
        rates: DataFrame with [county_fips, age_group, sex, migration_rate]
        period: Current period being processed
        boom_periods: List of (start, end) tuples for boom-era
        male_dampening_factor: Multiplier for male rates (e.g., 0.80 = 20% reduction)

    Returns:
        DataFrame with adjusted male rates in qualifying periods
    """
```

**Flag for review:** The exact `male_dampening_factor` and `blend_factor` values
need calibration. Plan: run the pipeline with no adjustments first, examine the
resulting sex ratios in projected populations, then calibrate the factors to
produce realistic trajectories. Document the calibration in an ADR.

#### 4.2.5 Multi-Period Averaging: `average_period_rates()`

```python
def average_period_rates(
    period_rates: dict[tuple[int, int], pd.DataFrame],
    method: str = "simple_average",
) -> pd.DataFrame:
    """Average migration rates across multiple 5-year periods.

    Computes the arithmetic mean of migration rates across periods,
    per county × age group × sex. This produces the baseline constant
    rate that feeds into convergence interpolation (Phase 2).

    SDC 2024 uses 4 periods: 2000-2005, 2005-2010, 2010-2015, 2015-2020.
    We extend to include 2020-2024 (as a partial 5th period).

    Args:
        period_rates: Dict mapping (start, end) to rate DataFrames
        method: "simple_average" or "trimmed_average" (drop min/max per cell)

    Returns:
        DataFrame with averaged rates: [county_fips, age_group, sex, migration_rate]
    """
```

### 4.3 Data Loading Functions

**New module: `cohort_projections/data/load/census_age_sex_population.py`**

```python
def load_census_2000_county_age_sex(
    file_path: Path,
    state_fips: str = "38",
) -> pd.DataFrame:
    """Load Census 2000 county population by 5-year age group and sex.

    Source: Census 2000 County Age and Sex.xlsx
    Sheet: co-est00int-agesex-5yr (1)
    Filters to state_fips, maps AGEGRP codes to age group labels,
    filters to SEX=1 (Male) and SEX=2 (Female).

    Returns:
        DataFrame [county_fips, age_group, sex, population] for Census 2000
    """


def load_pep_2010_2019_county_age_sex(
    file_path: Path,
    state_fips: str = "38",
    year: int = 2010,
) -> pd.DataFrame:
    """Load PEP intercensal county population by 5-year age group and sex.

    Source: cc-est2019-agesex-38.xlsx
    Year codes in this file: 1=Census 2010 base, 2=2010 estimate, ..., 12=2019.
    Extracts the requested year, standardizes to common format.

    Returns:
        DataFrame [county_fips, age_group, sex, population] for requested year
    """


def load_census_2020_county_age_sex(
    file_path: Path | None = None,
    state_fips: str = "38",
) -> pd.DataFrame:
    """Load Census 2020 county population by 5-year age group and sex.

    Source: base_population_by_county.csv (already processed)
    OR: Census Bureau API (DP table P12)

    Returns:
        DataFrame [county_fips, age_group, sex, population] for Census 2020
    """


def load_pep_2020_2024_county_age_sex(
    file_path: Path,
    state_fips: str = "38",
    year: int = 2024,
) -> pd.DataFrame:
    """Load PEP Vintage 2024 county population by age group and sex.

    Source: cc-est2024-agesex-all.parquet (shared-data/census/popest/)
    Published by Census Bureau June 26, 2025.

    IMPORTANT: This file uses WIDE FORMAT (age groups as columns like
    AGE04_TOT, AGE513_MALE, AGE85PLUS_FEM) unlike cc-est2020int which
    uses long format with AGEGRP rows. This loader must pivot the wide
    age columns into the standard long format.

    YEAR codes in this file: 1=Census 2020 base, 2=2020 estimate, ...,
    6=2024 estimate.

    Args:
        file_path: Path to cc-est2024-agesex-all.parquet
        state_fips: State FIPS code (default "38" for ND)
        year: Year to extract (default 2024, maps to YEAR code 6)

    Returns:
        DataFrame [county_fips, age_group, sex, population] for requested year
    """
```

### 4.4 Config Changes

Add to `projection_config.yaml`:

```yaml
rates:
  migration:
    domestic:
      method: "residual_age_specific"  # NEW — replaces PEP_components
      age_groups: "5_year"  # 18 groups: 0-4 through 85+
      residual:
        periods:
          - [2000, 2005]
          - [2005, 2010]
          - [2010, 2015]
          - [2015, 2020]
          - [2020, 2024]  # Uses cc-est2024-agesex-all.parquet (downloaded 2026-02-13)
        survival_source: "CDC_ND_2020"  # same as SDC 2024
        averaging: "simple_average"
      dampening:
        enabled: true
        factor: 0.60
        boom_periods:
          - [2005, 2010]
          - [2010, 2015]
        counties:
          - "38105"  # Williams
          - "38053"  # McKenzie
          - "38061"  # Mountrail
          - "38025"  # Dunn
          - "38089"  # Stark
      adjustments:
        college_age:
          enabled: true
          method: "smooth"  # "smooth" or "cap"
          counties:
            - "38035"  # Grand Forks (UND)
            - "38017"  # Cass (NDSU)
            - "38101"  # Ward (Minot State)
            - "38015"  # Burleigh (U of Mary, Bismarck State)
          age_groups: ["15-19", "20-24"]
          blend_factor: 0.5  # NEEDS CALIBRATION
        male_dampening:
          enabled: true
          factor: 0.80  # NEEDS CALIBRATION
          boom_periods:
            - [2005, 2010]
            - [2010, 2015]
```

### 4.5 Output Format

Phase 1 produces constant rates (one table per county) — the same for every
projection year. Time-varying rates come in Phase 2.

```
data/processed/migration/
  residual_migration_rates.parquet     # All counties, all periods
  residual_migration_rates_averaged.parquet  # Multi-period average (baseline)
  residual_migration_metadata.json     # Processing metadata
```

Schema for averaged rates:
```
county_fips  | age_group | sex    | migration_rate | n_periods | std_dev
38017        | 0-4       | Male   | 0.0234         | 5         | 0.0089
38017        | 0-4       | Female | 0.0198         | 5         | 0.0076
38017        | 5-9       | Male   | 0.0156         | 5         | 0.0052
...
```

### 4.6 Tests

**New test file: `tests/test_data/test_residual_migration.py`**

```
TestComputeResidualMigrationRates
├── test_basic_residual_calculation
│   └── Simple 2-period case with known populations and survival rates
├── test_age_group_shifting
│   └── Verify age 0-4 at t maps to 5-9 at t+5
├── test_85plus_open_ended_group
│   └── Both 80-84 and 85+ survivors feed into 85+ pool
├── test_negative_migration_rate
│   └── Out-migration counties produce negative rates
├── test_zero_population_handling
│   └── Counties with 0 pop in an age group don't divide by zero

TestPeriodDampening
├── test_boom_period_dampened
│   └── 2005-2010 and 2010-2015 get 60% factor for oil counties
├── test_non_boom_period_undampened
│   └── 2000-2005, 2015-2020, 2020-2024 unchanged
├── test_non_oil_county_undampened
│   └── Counties not in list are never dampened
├── test_dampening_preserves_sign
│   └── Negative rates stay negative after dampening

TestCollegeAgeAdjustment
├── test_college_county_rates_smoothed
├── test_non_college_county_unchanged
├── test_only_affects_target_age_groups

TestMaleMigrationDampening
├── test_male_rates_reduced_in_boom_periods
├── test_female_rates_unchanged
├── test_non_boom_periods_unchanged

TestMultiPeriodAveraging
├── test_simple_average_across_periods
├── test_all_counties_present
├── test_output_shape_36_rows_per_county

TestEndToEnd
├── test_full_pipeline_produces_valid_rates
├── test_comparison_to_sdc_2024_state_totals
└── test_output_files_created
```

### 4.7 Implementation Steps (ordered)

1. **Create data loaders** (`census_age_sex_population.py`)
   - Load Census 2000, PEP 2010-2019, Census 2020
   - Standardize to common schema: `[county_fips, age_group, sex, population]`
   - Write unit tests for each loader

2. **Fetch missing data** (Census 2020 by age, PEP 2020-2024 by age)
   - Add to `scripts/fetch_data.py` or create `scripts/fetch_census_age_data.py`
   - Cache locally in `data/raw/population/`

3. **Implement residual calculation** (`residual_migration.py`)
   - `assemble_period_populations()` — standardize + interpolate
   - `compute_residual_migration_rates()` — core formula
   - Write tests against known values

4. **Implement dampening** (period-specific, oil counties only)
   - `apply_period_dampening()` — boom-era check
   - Test: boom vs non-boom periods, oil vs non-oil counties

5. **Implement SDC adjustments**
   - `apply_college_age_adjustment()` — smooth or cap
   - `apply_male_migration_dampening()` — sex-specific boom-era
   - Flag: calibration values are initial estimates

6. **Implement averaging**
   - `average_period_rates()` — simple or trimmed average
   - Test: correct aggregation, all counties present

7. **Integration test**
   - Full pipeline from raw data to averaged rates
   - Compare state-level totals to SDC 2024 published rates

---

## 5. Phase 2: Age-Specific Convergence Interpolation

### 5.1 Objective

Implement time-varying migration where each age-sex group's rate converges
independently from its recent value toward its long-term mean.

### 5.2 Modifications to Existing Code

**Refactor `calculate_interpolated_rates()` in `migration_rates.py`:**

The existing function (lines 1150-1237) operates on aggregate `mean_netmig` per
county. It needs to be generalized to operate on age-sex-specific rates.

```python
def calculate_age_specific_convergence(
    recent_rates: pd.DataFrame,     # Rates from most recent period(s)
    medium_rates: pd.DataFrame,     # Rates averaged over medium window
    longterm_rates: pd.DataFrame,   # Rates averaged over full history
    projection_years: int = 20,
    convergence_schedule: dict[str, int] | None = None,
) -> dict[int, pd.DataFrame]:
    """Calculate year-varying migration rates with age-specific convergence.

    Each age-sex group converges independently from its recent rate toward
    its long-term mean. This replaces the uniform scaling factor approach.

    Default 3-phase schedule (5-10-5):
    - Years 1-5: linear interpolation from recent to medium
    - Years 6-15: hold at medium-term average
    - Years 16-20: linear interpolation from medium to long-term

    Args:
        recent_rates: DataFrame [county_fips, age_group, sex, migration_rate]
                      from most recent period (e.g., 2020-2024)
        medium_rates: DataFrame with rates averaged over medium window
                      (e.g., 2014-2024)
        longterm_rates: DataFrame with rates averaged over full history
                        (e.g., 2000-2024)
        projection_years: Total horizon (default 20)
        convergence_schedule: Phase timing parameters

    Returns:
        Dict mapping year_offset (1-20) to DataFrame with
        [county_fips, age_group, sex, migration_rate]
        Each year has 36 rows per county (18 age groups × 2 sexes).
    """
```

### 5.3 Computing Period Windows for Convergence

Using Phase 1 period rates, compute three averaging windows:

| Window | Periods Included | Interpretation |
|--------|-----------------|----------------|
| Recent | 2020-2024 | Most recent observed behavior |
| Medium | 2010-2015, 2015-2020, 2020-2024 | ~10-year window |
| Long-term | All 5 periods (2000-2024) | Full 24-year history |

Each window's rate is the simple average of the included periods' rates,
per county × age group × sex.

### 5.4 Output Format

```
data/processed/migration/
  convergence_rates_by_year.parquet   # All years × counties × age groups
```

Schema:
```
year_offset | county_fips | age_group | sex    | migration_rate
1           | 38017       | 0-4       | Male   | 0.0289
1           | 38017       | 0-4       | Female | 0.0245
...
20          | 38017       | 85+       | Female | -0.0134
```

Total rows: 20 years × 53 counties × 36 cells = 38,160 rows.

### 5.5 Tests

```
TestAgeSpecificConvergence
├── test_year1_equals_recent_rates
├── test_year5_equals_medium_rates
├── test_year6_through_15_hold_at_medium
├── test_year20_equals_longterm_rates
├── test_each_age_group_converges_independently
├── test_positive_and_negative_rates_converge_correctly
├── test_volatile_vs_stable_age_groups_differ
├── test_output_shape_per_year
└── test_custom_schedule_parameters
```

---

## 6. Phase 3: Census Bureau Mortality Improvement

### 6.1 Objective

Replace unimplemented 0.5%/year flat mortality improvement with Census Bureau
projected survival ratios, adjusted for North Dakota baseline.

### 6.2 Implementation

**New module: `cohort_projections/data/process/mortality_improvement.py`**

```python
def load_census_survival_projections(
    file_path: Path,
    years: tuple[int, int] = (2025, 2045),
    sex_filter: list[int] | None = None,
    group_filter: int = 0,
) -> pd.DataFrame:
    """Load Census Bureau NP2023-A4 survival ratio projections.

    Extracts year-specific survival ratios from the published CSV.
    Filters to our projection window and demographic groups.

    Source: np2023_a4_survival_ratios.csv
    Format: Wide (SRAT_0 through SRAT_100 columns)

    Args:
        file_path: Path to np2023_a4_survival_ratios.csv
        years: (start_year, end_year) inclusive
        sex_filter: List of SEX codes (1=Male, 2=Female). Default: [1, 2]
        group_filter: GROUP code for race (0=All races). Default: 0

    Returns:
        Long-format DataFrame [year, age, sex, survival_ratio]
    """


def compute_nd_adjustment_factors(
    nd_survival_rates: pd.DataFrame,
    census_2023_national: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ND-to-national adjustment factors (Decision 2, Option B).

    Ratio = ND_CDC_survival[age, sex] / Census_2023_national[age, sex]

    These ratios capture how ND mortality differs from the national average.
    They are applied multiplicatively to the Census Bureau projected rates.

    Args:
        nd_survival_rates: ND-specific CDC rates [age, sex, survival_rate]
        census_2023_national: Census Bureau 2023 national rates [age, sex, survival_ratio]

    Returns:
        DataFrame [age, sex, adjustment_factor]
    """


def build_nd_adjusted_survival_projections(
    census_projections: pd.DataFrame,
    adjustment_factors: pd.DataFrame,
    years: tuple[int, int] = (2025, 2045),
) -> dict[int, pd.DataFrame]:
    """Build year-indexed ND-adjusted survival rate tables.

    For each projection year:
        ND_survival[age, sex, year] = Census_projected[age, sex, year] * ND_adjustment[age, sex]

    Caps survival rates at 1.0 and floors at 0.0.

    Returns:
        Dict mapping year to DataFrame [age, sex, survival_rate]
    """
```

### 6.3 Aggregation to 5-Year Age Groups

The Census Bureau data is single-year ages (0-100). Our engine currently uses
single-year ages (0-90). Two options:

**Option A:** Keep single-year survival rates in the engine (current resolution).
Convert Census Bureau SRAT values from single-year to annual rates.

**Option B:** Aggregate to 5-year groups to match migration rate resolution.
Use geometric mean of single-year rates within each group.

**Recommendation:** Option A — keep single-year survival in the engine. The
migration rates are 5-year groups, but survival rates should stay at the
finest available resolution. The engine already handles single-year ages for
survival. Only migration is shifting to 5-year groups.

### 6.4 Output Format

```
data/processed/mortality/
  nd_adjusted_survival_projections.parquet
```

Schema:
```
year | age | sex    | survival_rate | source
2025 | 0   | Male   | 0.99372       | Census_NP2023_ND_adjusted
2025 | 0   | Female | 0.99567       | Census_NP2023_ND_adjusted
2025 | 1   | Male   | 0.99981       | Census_NP2023_ND_adjusted
...
2045 | 90  | Female | 0.87234       | Census_NP2023_ND_adjusted
```

### 6.5 Tests

```
TestLoadCensusSurvivalProjections
├── test_loads_correct_year_range
├── test_filters_sex_correctly
├── test_long_format_output
├── test_survival_ratios_in_valid_range

TestNDAdjustmentFactors
├── test_adjustment_factor_computation
├── test_nd_higher_mortality_produces_factor_below_1
├── test_nd_lower_mortality_produces_factor_above_1

TestBuildNDAdjustedProjections
├── test_year_indexed_output
├── test_survival_capped_at_1
├── test_survival_floored_at_0
├── test_improvement_over_time (rates increase each year)
├── test_male_female_differential_preserved
```

---

## 7. Phase 4: Engine Wiring for Time-Varying Rates

### 7.1 Objective

Modify the projection engine and pipeline to accept and use time-varying
migration and survival rates, while maintaining backward compatibility with
constant rates.

### 7.2 Changes to `cohort_component.py`

```python
class CohortComponentProjection:

    def __init__(
        self,
        base_population: pd.DataFrame,
        fertility_rates: pd.DataFrame,
        survival_rates: pd.DataFrame,
        migration_rates: pd.DataFrame,
        config: dict[str, Any] | None = None,
        # NEW parameters (optional, backward compatible)
        migration_rates_by_year: dict[int, pd.DataFrame] | None = None,
        survival_rates_by_year: dict[int, pd.DataFrame] | None = None,
    ):
        # ... existing init code ...
        self.migration_rates_by_year = migration_rates_by_year
        self.survival_rates_by_year = survival_rates_by_year

    def _get_migration_rates(self, year: int) -> pd.DataFrame:
        """Get migration rates for a specific year.

        If time-varying rates are available, look up the year.
        Falls back to constant rates if year not found or time-varying
        rates not provided.
        """
        if self.migration_rates_by_year is not None:
            year_offset = year - self.base_year + 1
            if year_offset in self.migration_rates_by_year:
                return self.migration_rates_by_year[year_offset]
        return self.migration_rates

    def _get_survival_rates(self, year: int) -> pd.DataFrame:
        """Get survival rates for a specific year.

        If time-varying rates are available, look up the year.
        Falls back to constant rates if year not found.
        """
        if self.survival_rates_by_year is not None:
            if year in self.survival_rates_by_year:
                return self.survival_rates_by_year[year]
        return self.survival_rates

    def project_single_year(self, population, year, scenario=None):
        # CHANGE: Replace self.survival_rates.copy() with:
        survival_rates = self._get_survival_rates(year)

        # CHANGE: Replace self.migration_rates.copy() with:
        migration_rates = self._get_migration_rates(year)

        # ... rest of method unchanged ...
```

### 7.3 Changes to `multi_geography.py`

```python
def run_single_geography_projection(
    fips: str,
    level: str,
    base_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    config: dict | None = None,
    output_dir: Path | None = None,
    save_results: bool = True,
    scenario: str = "baseline",
    # NEW parameters
    migration_rates_by_year: dict[int, pd.DataFrame] | None = None,
    survival_rates_by_year: dict[int, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    # ... pass new params to CohortComponentProjection ...
```

### 7.4 Changes to `02_run_projections.py`

```python
def load_demographic_rates(config, scenario="baseline"):
    # ... existing fertility + survival loading ...

    # NEW: Load convergence rates (Phase 2 output)
    migration_rates_by_year = None
    convergence_path = processed_dir / "migration" / "convergence_rates_by_year.parquet"
    if convergence_path.exists():
        convergence_df = pd.read_parquet(convergence_path)
        # Split into per-year, per-county structure
        migration_rates_by_year = _build_year_county_rate_dicts(convergence_df)

    # NEW: Load ND-adjusted survival projections (Phase 3 output)
    survival_rates_by_year = None
    survival_proj_path = processed_dir / "mortality" / "nd_adjusted_survival_projections.parquet"
    if survival_proj_path.exists():
        survival_df = pd.read_parquet(survival_proj_path)
        survival_rates_by_year = {
            year: group.drop(columns=["year"]).reset_index(drop=True)
            for year, group in survival_df.groupby("year")
        }

    return (fertility_rates, survival_rates, migration_rates,
            migration_rates_by_year, survival_rates_by_year)
```

### 7.5 Migration Rate Format Bridge

The engine currently expects single-year ages (0-90) × 2 sexes × 6 races for
migration. Phase 1 produces 5-year age groups × 2 sexes (no race breakdown).

**Bridge function needed:**

```python
def expand_5yr_migration_to_engine_format(
    rates_5yr: pd.DataFrame,
    population: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Expand 5-year age group migration rates to single-year engine format.

    Steps:
    1. Expand each 5-year rate to the 5 constituent single-year ages
       (same rate for ages 20, 21, 22, 23, 24 within the 20-24 group)
    2. Distribute across race/ethnicity proportional to population composition
    3. Convert from rate to net_migration counts if engine expects absolute

    This bridges the gap between:
    - Phase 1 output: 36 rows/county (18 age groups × 2 sexes)
    - Engine input: 1,092 rows/county (91 ages × 2 sexes × 6 races)

    Args:
        rates_5yr: DataFrame [county_fips, age_group, sex, migration_rate]
        population: Base population for race distribution
        config: Projection config

    Returns:
        DataFrame [age, sex, race, net_migration OR migration_rate]
    """
```

### 7.6 Tests

```
TestTimeVaryingEngine
├── test_constant_rates_backward_compatible
│   └── No *_by_year params → identical to current behavior
├── test_migration_varies_by_year
│   └── Different rates in year 1 vs year 10 vs year 20
├── test_survival_varies_by_year
│   └── Improving survival over projection horizon
├── test_missing_year_falls_back_to_constant
│   └── Year not in dict → uses base rates
├── test_combined_time_varying_migration_and_survival

TestFormatBridge
├── test_5yr_to_single_year_expansion
├── test_race_distribution_proportional
├── test_total_migration_preserved
```

---

## 8. Phase 5: Validation & Comparison

### 8.1 Validation Tests

1. **Backward compatibility**: Run with constant rates, verify identical results
2. **Conservation check**: Time-varying rates don't create/destroy population
3. **Convergence correctness**: Year 20 rates equal long-term average
4. **Mortality improvement**: Life expectancy increases over projection horizon
5. **No negative populations**: After migration + mortality, all cohorts >= 0

### 8.2 Comparison Outputs

```python
def compare_to_sdc_2024(
    our_projections: pd.DataFrame,
    sdc_projections_path: Path,
) -> pd.DataFrame:
    """Compare our projections to SDC 2024 published projections.

    Key comparisons:
    - Total state population by year (2025, 2030, 2035, 2040, 2045)
    - County-level population (top 10 counties)
    - Age structure (dependency ratio, median age)
    - Migration direction (net in vs net out for key counties)

    Returns:
        Comparison DataFrame with columns:
        [geography, year, our_projection, sdc_projection, difference, pct_difference]
    """
```

### 8.3 Sensitivity Analysis

Run convergence with alternative schedules and document impact:

| Schedule | Description | Expected Impact |
|----------|-------------|-----------------|
| 5-10-5 (default) | Our chosen schedule | Baseline |
| 3-7-10 | Faster convergence to medium, longer to long-term | Higher near-term, lower long-term |
| 7-8-5 | Slower near-term convergence | More weight on recent rates |
| 0-0-20 | Pure linear convergence (no phases) | Simplest comparison |
| SDC (constant) | No convergence — constant rates | Match SDC 2024 approach |

### 8.4 Documentation Deliverables

1. **ADR update**: Revise ADR-036 to reflect new methodology
2. **Methodology report**: Document the full residual + convergence pipeline
3. **Comparison report**: Side-by-side with SDC 2024

---

## 9. New Research Findings

### 9.1 SDC Dampening is Variable, Not Uniform

**Discovery:** The SDC 2024 used **variable period multipliers**, not a uniform
60% across all projection periods:

| Period | Multiplier | Rationale |
|--------|------------|-----------|
| 2020-2025 | 0.2 | COVID + post-boom adjustment |
| 2025-2030 | 0.6 | Bakken dampening |
| 2030-2035 | 0.6 | Continued dampening |
| 2035-2040 | 0.5 | Further conservative reduction |
| 2040-2045 | 0.7 | Gradual return |
| 2045-2050 | 0.7 | Gradual return |

**Our approach:** Per Decision 6, we apply 60% dampening to boom-era INPUT
periods (2005-2010, 2010-2015) rather than varying the multiplier across
projection periods. The convergence interpolation (Phase 2) handles the
time-varying aspect of migration rates going forward. The SDC's variable
multipliers were their substitute for convergence interpolation.

### 9.2 SDC Adjustments Are Manual Expert Judgment

**Discovery:** The SDC applied ~32,000 person-adjustments per 5-year period.
College-age and male migration adjustments were NOT formulaic — they were
manual edits in the Excel workbook after the initial rate calculation.

**Implication:** We cannot precisely replicate SDC adjustments. Our
implementation should be algorithmic (reproducible, configurable) rather than
manual. The `college_age_adjustment` and `male_migration_dampening` functions
defined in Phase 1 are our algorithmic approximation. Parameters need
calibration against observed patterns.

### 9.3 Census 2000 File is Multi-State

**Discovery:** The `Census 2000 County Age and Sex.xlsx` file contains data for
324 counties across 51 states/territories, not just ND. We need to filter to
`STATE == 38` (North Dakota) when loading.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Census API data fetch fails or format changes | Medium | HIGH | Cache all fetched data locally; document exact API endpoints; have manual download fallback |
| College-age adjustment calibration doesn't converge | Medium | Medium | Start with no adjustment, run pipeline, examine results, calibrate iteratively |
| Male dampening factor produces unrealistic sex ratios | Low | Medium | Validate projected sex ratios against historical; bound factor to [0.7, 1.0] range |
| 5-year to single-year age expansion introduces artifacts | Low | Low | Uniform expansion within group; validate total migration preserved |
| Backward compatibility breaks with existing tests | Low | HIGH | All new params are optional with None defaults; run full test suite after every change |
| Data size becomes unwieldy (20 years × 53 counties × 36 cells) | Very Low | Low | 38,160 rows total — trivially small for pandas |

---

## Appendix A: File Creation Summary

### New Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `cohort_projections/data/process/residual_migration.py` | Core residual rate computation | 1 |
| `cohort_projections/data/load/census_age_sex_population.py` | Data loaders for Census/PEP age-sex data | 1 |
| `cohort_projections/data/process/mortality_improvement.py` | Census Bureau mortality improvement | 3 |
| `tests/test_data/test_residual_migration.py` | Tests for residual migration | 1 |
| `tests/test_data/test_mortality_improvement.py` | Tests for mortality improvement | 3 |
| `scripts/fetch_census_age_data.py` | Fetch missing Census API data | 1 |

### Existing Files to Modify

| File | Changes | Phase |
|------|---------|-------|
| `config/projection_config.yaml` | Add residual migration config, boom_periods | 1 |
| `cohort_projections/data/process/migration_rates.py` | Refactor `calculate_interpolated_rates()` for age-specific convergence | 2 |
| `cohort_projections/core/cohort_component.py` | Add `*_by_year` params, `_get_*_rates()` methods | 4 |
| `cohort_projections/geographic/multi_geography.py` | Thread `*_by_year` params | 4 |
| `scripts/pipeline/02_run_projections.py` | Load time-varying rates, pass to engine | 4 |

### No Changes Needed

| File | Reason |
|------|--------|
| `cohort_projections/core/fertility.py` | Fertility held constant (Decision 4) |
| `cohort_projections/core/mortality.py` | Interface unchanged; new data flows through existing `apply_survival_rates()` |
| `cohort_projections/core/migration.py` | Interface unchanged; new data flows through existing `apply_migration()` |
