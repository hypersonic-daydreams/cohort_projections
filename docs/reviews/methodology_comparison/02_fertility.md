# Methodology Comparison: Fertility Rates and Births

**Document**: 02 -- Fertility Rates and Births
**Date**: 2026-02-20
**Scope**: Detailed comparison of fertility/birth methodology between the SDC 2024 population projections and the current (2026) cohort-component projection system for North Dakota.

---

## Table of Contents

1. [SDC 2024 Approach](#1-sdc-2024-approach)
2. [Current 2026 Approach](#2-current-2026-approach)
3. [Key Differences](#3-key-differences)
4. [Formulas and Calculations](#4-formulas-and-calculations)
5. [Rationale for Changes](#5-rationale-for-changes)

---

## 1. SDC 2024 Approach

### 1.1 Data Sources

The SDC 2024 projections use a blended fertility rate derived from three tiers of data:

| Tier | Source | Role |
|------|--------|------|
| Primary | North Dakota DHHS Vital Statistics (2016-2022 birth data by county) | County-level age-specific fertility rates |
| Secondary | State-level rates derived from ND vital statistics | Smoothing for small-county instability |
| Tertiary | CDC National Vital Statistics Reports (2021 national ASFR) | Additional stability blending |

From the SDC report:

> "Rates are smoothed to reduce anomalies in the number of births during this timeframe."

The blending methodology is not fully documented. County-level rates are blended with state-level rates and then further blended with national rates. The exact blending weights are not specified in the published report or the `Projections_Base_2023.xlsx` workbook.

**Reference**: `../sdc_2024_replication/METHODOLOGY_SPEC.md`, Section 3.1.

### 1.2 Age Groups

Fertility rates are defined for **eight 5-year age groups**:

| Age Group | Index in SDC Workbook | Used in Birth Calculation? |
|-----------|-----------------------|---------------------------|
| 10-14 | 2 | No (present in data file but excluded from engine) |
| 15-19 | 3 | Yes |
| 20-24 | 4 | Yes |
| 25-29 | 5 | Yes |
| 30-34 | 6 | Yes |
| 35-39 | 7 | Yes |
| 40-44 | 8 | Yes |
| 45-49 | 9 | Yes |

The SDC replication engine (`../sdc_2024_replication/scripts/projection_engine.py`) defines:

```python
CHILDBEARING_AGE_INDICES: list[int] = [3, 4, 5, 6, 7, 8, 9]
```

This corresponds to ages 15-19 through 45-49. The 10-14 age group is present in the data file (`../sdc_2024_replication/data/fertility_rates_by_county.csv`) but is **not used** in the birth calculation. Most counties carry a near-zero placeholder rate of 0.000178 for age 10-14; only 7 of 53 counties have rates above 0.001 (predominantly reservation counties: Benson, Ramsey, Rolette, Sioux).

### 1.3 Rate Format and Units

The SDC fertility rates in `fertility_rates_by_county.csv` are stored as **births per woman per 5-year projection period** (i.e., the cumulative rate across one 5-year time step). This is equivalent to the annual age-specific fertility rate multiplied by 5.

Evidence for this interpretation:

| County | Age 25-29 Rate | Implied Annual ASFR | Implied TFR (sum) |
|--------|---------------|--------------------|--------------------|
| Cass | 0.596 | 0.119 (119/1,000) | 1.773 |
| Burleigh | 0.646 | 0.129 (129/1,000) | 1.940 |
| Grand Forks | 0.636 | 0.127 (127/1,000) | 1.801 |
| Williams | 0.784 | 0.157 (157/1,000) | 2.630 |

State average TFR (mean across 53 counties, ages 15-49): **2.332**

If these rates were annual per-woman rates, the TFR would be sum * 5 = 11.66, which is biologically implausible. The correct interpretation is that TFR = sum of the stored rates across all age groups.

**Important note on the SDC replication engine**: The replication code at line 204 computes `age_births = female_pop * fertility_rate * PERIOD_LENGTH` (where `PERIOD_LENGTH = 5`). If the stored rates are already 5-year cumulative rates, this multiplication by 5 would be erroneous (double-counting the period width). The METHODOLOGY_SPEC.md formula `Births = SUM[Female_Pop * Rate] * 5` appears to assume the stored rates are **annual** rates. This ambiguity in the SDC methodology is unresolved; the published SDC report does not specify the rate units explicitly. Our best interpretation, based on plausible TFR values, is that the rates in the CSV represent 5-year cumulative rates and the correct formula is `Births = SUM[Female_Pop * Rate]` without the additional multiplication by 5.

### 1.4 County-Level vs. Statewide Rates

The SDC uses **county-specific** fertility rates for all 53 counties. Each county has its own ASFR by age group, producing county-specific TFRs that range from approximately 1.6 (urban counties like Cass) to 3.4 (reservation counties like Benson).

County-level rate variation (TFR, sum of 5-year rates across ages 15-49):

| TFR Range | County Count | Examples |
|-----------|-------------|----------|
| < 1.80 | 5 | Cass (1.77), Grand Forks (1.80), Cavalier (1.59) |
| 1.80 - 2.20 | 19 | Burleigh (1.94), Stark (2.08), Ward (2.11) |
| 2.20 - 2.60 | 15 | Williams (2.63), Morton (2.22), Richland (2.39) |
| 2.60 - 3.00 | 8 | McKenzie (2.91), Mountrail (2.67) |
| > 3.00 | 6 | Benson (3.41), Sioux (3.36), Rolette (3.16) |

### 1.5 Temporal Treatment

Fertility rates are **held constant** across all projection periods (2020-2050). No fertility decline, improvement, or trend scenarios are applied. The same county-level ASFR used for the 2020-2025 period is applied identically to 2045-2050.

### 1.6 Race/Ethnicity

The SDC 2024 projections **do not include race/ethnicity** as a dimension. Fertility rates are applied to the total female population without disaggregation by race. This means that counties with large AIAN populations (which tend to have higher fertility rates) carry that higher rate implicitly through the county-level rate, but race-specific fertility differentials within a county are not modeled.

### 1.7 Sex Ratio at Birth

The SDC replication engine uses a male birth proportion of **0.512** (51.2% male, 48.8% female):

```python
MALE_BIRTH_RATIO: float = 0.512
```

This is slightly above the commonly cited U.S. ratio of 1.05:1 (which implies 51.22% male), but effectively equivalent.

### 1.8 Infant Survival

After computing total births over the 5-year period, the SDC applies an **infant survival rate** to convert births into the 0-4 population at the end of the period:

```
Population[0-4, t+5] = Births * Survival_Rate[0-4, sex]
```

The survival rate for the 0-4 age group is sex-specific and derived from CDC life tables for North Dakota (2020). This rate accounts for infant and early childhood mortality across the entire 5-year period.

---

## 2. Current 2026 Approach

### 2.1 Data Sources

The current system uses national age-specific fertility rates from **CDC NCHS** sources:

| Data Source | Content | Year |
|-------------|---------|------|
| CDC NCHS Vital Statistics Rapid Release (Natality Dashboard) | Quarterly ASFR by race for White NH, Black NH, Hispanic | 2024 |
| NVSS Births Final Data (nvsr73-02, nvsr74-01) | ASFR by race including AIAN and Asian/PI | 2022-2023 |
| CDC WONDER Natality | Supplemental race-specific rates | 2020-2023 |

The processed input file is `data/raw/fertility/asfr_processed.csv`. Rates for White NH, Black NH, and Hispanic are from 2024 data. AIAN and Asian/PI rates are from 2022 NVSS final data (the most recent year with race-specific detail for these groups).

**Reference**: `data/raw/fertility/DATA_SOURCE_NOTES.md`, `docs/governance/adrs/001-fertility-rate-processing.md` (ADR-001).

### 2.2 Age Groups

Input data uses **seven 5-year age groups** (15-19 through 45-49). During the data transformation pipeline (`scripts/pipeline/02_run_projections.py`, function `_expand_age_groups_to_single_years`), these are expanded to **single-year ages** (15, 16, 17, ..., 49) by applying the same rate uniformly across all single-year ages within each 5-year group.

| Age Group | Input ASFR (per 1,000) | Single-Year Rate (per woman) |
|-----------|----------------------|----------------------------|
| 15-19 | 12.8 | 0.0128 |
| 20-24 | 56.8 | 0.0568 |
| 25-29 | 90.9 | 0.0909 |
| 30-34 | 94.7 | 0.0947 |
| 35-39 | 54.6 | 0.0546 |
| 40-44 | 12.7 | 0.0127 |
| 45-49 | 1.1 | 0.0011 |

The 10-14 age group is **not included** in the input data or the projection.

### 2.3 Rate Format and Units

Raw rates in `asfr_processed.csv` are stored as **births per 1,000 women per year** (the standard NCHS reporting convention). During transformation, rates are converted to **births per woman per year** by dividing by 1,000:

```python
# From scripts/pipeline/02_run_projections.py, _transform_fertility_rates()
if df["fertility_rate"].max() > 1:
    df["fertility_rate"] = df["fertility_rate"] / 1000
```

After transformation, the rate for a 25-year-old woman of White NH race is 0.0892 (births per woman per year).

### 2.4 Race/Ethnicity Disaggregation

The current system uses **race-specific** age-specific fertility rates for 6 race/ethnicity categories:

| Race/Ethnicity Category | Source Year | TFR |
|------------------------|-------------|-----|
| White alone, Non-Hispanic | 2024 | 1.533 |
| Black alone, Non-Hispanic | 2024 | 1.535 |
| AIAN alone, Non-Hispanic | 2022 | 1.422 |
| Asian/PI alone, Non-Hispanic | 2022 | 1.485 |
| Two or more races, Non-Hispanic | Averaged* | ~1.495 |
| Hispanic (any race) | 2024 | 1.956 |

*The "Two or more races" category is not available directly from CDC sources. The system computes it as the average of all other race-specific rates by age (`scripts/pipeline/02_run_projections.py`, lines 216-220).

Race code mapping from raw data codes to projection categories is defined in `RACE_CODE_TO_NAME`:

```python
RACE_CODE_TO_NAME = {
    "white_nh": "White alone, Non-Hispanic",
    "black_nh": "Black alone, Non-Hispanic",
    "aian_nh": "AIAN alone, Non-Hispanic",
    "asian_nh": "Asian/PI alone, Non-Hispanic",
    "hispanic": "Hispanic (any race)",
    "two_or_more_nh": "Two or more races, Non-Hispanic",
}
```

The "total" race code from the raw data is dropped during transformation (it is not mapped).

**Reference**: ADR-001 (Decision 3: SEER Race Code Mapping), `cohort_projections/data/process/fertility_rates.py`.

### 2.5 Scope: National Rates Applied Statewide

Unlike the SDC approach, the current system uses **national** ASFR applied uniformly to all counties. There are no county-specific fertility rates. The county-level birth count variation arises entirely from differences in:

1. The age structure of the female population (more women of childbearing age = more births)
2. The racial composition of the female population (Hispanic women have higher ASFR than White NH women)

This is a deliberate simplification. ADR-001, Alternative 3 ("Hierarchical Averaging: State -> County -> Place") was considered and rejected:

> "State-level fertility rates are the primary use case. County/place rates often unavailable or unreliable. Can implement later if needed."

### 2.6 Temporal Treatment: Baseline and Scenarios

The baseline scenario uses **constant** fertility rates throughout the projection horizon (2025-2055). However, the system supports scenario-based fertility adjustments:

| Scenario | Fertility Setting | Effect |
|----------|------------------|--------|
| Baseline | `constant` | No change to ASFR across years |
| Restricted Growth | `-5_percent` | All ASFRs reduced by 5% (grounded in CBO TFR revision) |
| High Growth | `+5_percent` | All ASFRs increased by 5% |
| (Available) | `trending` | 0.5% annual compound decline |

The scenario adjustments are applied in two places:

1. **Pre-projection** (static): `apply_scenario_rate_adjustments()` in `scripts/pipeline/02_run_projections.py` applies a uniform multiplier to all age-race-specific rates before the projection begins.
2. **Per-year** (dynamic): `apply_fertility_scenario()` in `cohort_projections/core/fertility.py` can apply year-varying adjustments (e.g., compounding decline for the `trending` scenario).

For the restricted growth and high growth scenarios, the CBO-grounded methodology (ADR-037) provides the empirical basis:

- **-5% fertility** in the restricted growth scenario is grounded in CBO's -4.3% TFR revision between their January 2025 and January 2026 Demographic Outlook publications, reflecting the compositional effect of reduced immigration on aggregate fertility.
- **+5% fertility** in the high growth scenario reflects the counterfactual of higher aggregate fertility associated with elevated immigration (CBO projects foreign-born TFR of ~1.79 vs. native-born TFR of ~1.53).

**Reference**: ADR-037, `config/projection_config.yaml` scenarios section.

### 2.7 Sex Ratio at Birth

The default sex ratio at birth is **0.51** (51% male, 49% female), configurable via `rates.fertility.sex_ratio_male` in `projection_config.yaml`:

```python
# cohort_projections/core/fertility.py, line 57
sex_ratio_at_birth = birth_config.get("sex_ratio_male", 0.51)
```

This is slightly below the commonly cited biological ratio of 1.05:1 (51.22% male). The difference is minor but produces marginally fewer male births per cycle.

### 2.8 Birth Calculation: Annual Cycle

Births are calculated **annually** (not over 5-year periods). In each projection year:

1. The female population at reproductive ages (15-49) is extracted.
2. Each female age-race cohort is multiplied by its corresponding ASFR.
3. Births are summed across all ages within each race.
4. Total births per race are split into male and female newborns using the sex ratio.
5. Newborns are assigned `age = 0` and the mother's race.

There is **no explicit infant survival** step applied to births. Newborns enter the population at age 0, and survival is applied when the cohort ages forward from age 0 to age 1 in the next projection year. This means first-year infant mortality is captured through the age-0 survival rate in the next annual cycle, rather than being applied as a birth-to-0-4-group survival factor.

**Reference**: `cohort_projections/core/fertility.py`, `calculate_births()` function; `cohort_projections/core/cohort_component.py`, `project_single_year()` method.

### 2.9 TFR Validation

The data processing pipeline (`cohort_projections/data/process/fertility_rates.py`) includes TFR validation:

- TFR < 1.0: warning (unusually low for developed countries)
- TFR > 3.0: warning (unusually high)
- ASFR > 0.15 (150/1,000): error (biologically implausible)
- ASFR > 0.13 (130/1,000): warning (above typical maximum)

Current TFR values (from `asfr_processed.csv`, 2024 national rates) range from 1.42 (AIAN) to 1.96 (Hispanic), all within the plausible range for U.S. populations.

---

## 3. Key Differences

| Dimension | SDC 2024 | Current 2026 | Impact |
|-----------|----------|--------------|--------|
| **Data source** | ND DHHS Vital Statistics (2016-2022) blended with CDC national | CDC NCHS national ASFR (2024 with 2022 supplemental) | SDC uses state-specific data; 2026 uses more recent national data |
| **Geographic specificity** | County-specific rates (53 separate rate schedules) | National rates applied uniformly to all counties | SDC captures county-level variation; 2026 relies on demographic composition |
| **Race/ethnicity** | Not included (total population only) | 6 race/ethnicity categories with race-specific ASFR | 2026 captures fertility differentials by race |
| **Age resolution** | 5-year groups (15-19 through 45-49) | Single-year ages (15 through 49), expanded from 5-year groups | Finer resolution in 2026; more precise interaction with single-year age structure |
| **Temporal resolution** | 5-year projection steps | Annual projection steps | Annual calculation avoids aggregation assumptions |
| **Projection period** | 2020-2050 (6 five-year steps) | 2025-2055 (30 annual steps) | Different horizons; 2026 is more granular |
| **Fertility trend** | Constant (no change 2020-2050) | Constant baseline; scenario-adjustable (+/-5%, trending) | 2026 has richer scenario capability |
| **TFR (statewide avg)** | ~2.33 (county avg, sum of 5yr rates) | ~1.62 (total, national 2024) | SDC uses older, higher rates; 2026 reflects current national decline |
| **Sex ratio at birth** | 0.512 (51.2% male) | 0.51 (51% male) | Negligible difference (0.2 percentage points) |
| **Infant survival** | Explicit: `Births * Survival_Rate[0-4]` applied in birth step | Implicit: age-0 survival applied in next annual aging step | Mechanically different but functionally equivalent |
| **Rate units (stored)** | Births per woman per 5-year period | Births per 1,000 women per year (converted to per-woman in pipeline) | Different storage conventions; same conceptual quantity after transformation |
| **Blending/smoothing** | County + state + national blending (weights undocumented) | None (direct CDC rates, no smoothing) | SDC reduces small-sample noise; 2026 relies on national rates being stable |
| **Scenario adjustments** | None | CBO-grounded (-5%, +5%, trending) per ADR-037 | 2026 supports policy-relevant scenario analysis |
| **10-14 age group** | Present in data, excluded from calculation | Not included | No practical difference |

### 3.1 Impact on Birth Counts

The most consequential differences affecting birth counts are:

1. **TFR level**: The SDC state-average TFR (~2.33) is substantially higher than the current national TFR (~1.62). This 44% difference in TFR translates directly to approximately 44% more births per year in the SDC projection, all else being equal. The SDC rates reflect 2016-2022 ND data (which includes the Bakken boom period with elevated fertility), while the 2026 rates reflect 2024 national data during a period of historically low U.S. fertility.

2. **County-specific vs. national rates**: The SDC's county-specific rates cause birth counts to vary substantially by county above and beyond what demographic composition would produce. For example, Williams County (Bakken core) has a TFR of 2.63 vs. Cass County (Fargo) at 1.77 -- a 49% difference. In the current 2026 system, these counties would produce different birth counts only because of differences in female age and race distributions, not because of different fertility rate schedules.

3. **Race-specific fertility**: The 2026 system's race-specific rates cause the racial composition of each county to influence its birth rate. Counties with large Hispanic populations will produce more births (Hispanic TFR 1.96) than similar-sized counties with predominantly White NH populations (TFR 1.53). The SDC captures this implicitly through its county-level rate blending, but cannot distinguish the contribution of racial composition from other county-level factors.

---

## 4. Formulas and Calculations

### 4.1 SDC 2024 Birth Formula

#### Step 1: Calculate Total Births Over the 5-Year Period

```
Births[county, t to t+5] = SUM over a in {15-19, 20-24, ..., 45-49} [
    Female_Pop[a, county, t] * ASFR[a, county]
]
```

Where:
- `Female_Pop[a, county, t]` = female population in 5-year age group `a`, in county, at time `t`
- `ASFR[a, county]` = age-specific fertility rate for age group `a` in county (births per woman per 5-year period)

**Note on multiplication by 5**: The METHODOLOGY_SPEC.md formula shows `* 5 years` after the summation, implying the stored rates are annual. However, as discussed in Section 1.3, the stored rates appear to already incorporate the 5-year period width (TFR = sum of rates = 2.33, not sum * 5 = 11.66). This ambiguity exists in the SDC documentation. The replication engine (`projection_engine.py`, line 204) includes `* PERIOD_LENGTH`, which would be correct only if rates are annual. We present the formula as it appears in the METHODOLOGY_SPEC.md:

```
Births[county, t to t+5] = SUM_a [ Female_Pop[a, county, t] * ASFR_annual[a, county] ] * 5
```

Where `ASFR_annual` is the annual age-specific fertility rate per woman.

#### Step 2: Split by Sex

```
Male_Births   = Births * 0.512
Female_Births = Births * 0.488
```

#### Step 3: Apply Infant Survival

```
Pop[0-4, Male, county, t+5]   = Male_Births   * Survival_Rate[0-4, Male]
Pop[0-4, Female, county, t+5] = Female_Births * Survival_Rate[0-4, Female]
```

Where:
- `Survival_Rate[0-4, sex]` = probability of surviving from birth through the 0-4 age group (sex-specific, from CDC ND 2020 life tables)

#### Complete Chain

```
Pop[0-4, sex, county, t+5] = (
    SUM_a [ Female_Pop[a, county, t] * ASFR[a, county] ]
) * SexRatio[sex] * Survival_Rate[0-4, sex]
```

#### Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `Female_Pop[a, county, t]` | Female population in 5-year age group `a` at time `t` | Previous projection step or Census 2020 base |
| `ASFR[a, county]` | Age-specific fertility rate (births per woman per 5-year period) | ND DHHS / CDC blended, constant across periods |
| `SexRatio[Male]` | Proportion of births that are male | 0.512 |
| `SexRatio[Female]` | Proportion of births that are female | 0.488 |
| `Survival_Rate[0-4, sex]` | Probability of surviving from birth to the 0-4 cohort | CDC ND 2020 life table |

**File references**:
- Fertility rates: `../sdc_2024_replication/data/fertility_rates_by_county.csv`
- Projection engine: `../sdc_2024_replication/scripts/projection_engine.py`, `calculate_births()` (line 155), `apply_infant_survival()` (line 214)
- Methodology: `../sdc_2024_replication/METHODOLOGY_SPEC.md`, Sections 2.4, 3.1, 7.3

---

### 4.2 Current 2026 Birth Formula

#### Step 1: Calculate Births by Mother's Age, Race (Annual)

For each projection year `y`:

```
Births_by_mother[a, r, y] = Female_Pop[a, r, y] * ASFR[a, r]
```

Where:
- `a` = single-year age (15, 16, ..., 49)
- `r` = race/ethnicity category (one of 6)
- `Female_Pop[a, r, y]` = female population at age `a`, race `r`, in year `y`
- `ASFR[a, r]` = annual age-specific fertility rate (births per woman per year) for age `a`, race `r`

#### Step 2: Sum Births by Race

```
Total_Births[r, y] = SUM over a in {15, 16, ..., 49} [
    Births_by_mother[a, r, y]
]
```

#### Step 3: Split by Sex

```
Male_Births[r, y]   = Total_Births[r, y] * 0.51
Female_Births[r, y] = Total_Births[r, y] * 0.49
```

#### Step 4: Assign to Age 0

Newborns are placed into the population at `age = 0` with the mother's race:

```
Pop[age=0, Male, r, y] = Male_Births[r, y]
Pop[age=0, Female, r, y] = Female_Births[r, y]
```

#### Step 5: Survival Applied in Next Year

There is no explicit infant survival step in the birth calculation. Instead, when the population is aged forward from year `y` to `y+1`, the age-0 cohort has the age-0 survival rate applied:

```
Pop[age=1, sex, r, y+1] = Pop[age=0, sex, r, y] * Survival_Rate[age=0, sex, r]
```

This is functionally equivalent to the SDC's infant survival step but distributed across the annual aging cycle rather than applied as a one-time correction.

#### Complete Chain (Per Year)

```
Pop[age=0, sex, r, y+1] = (
    SUM over a in {15..49} [
        Female_Pop[a, r, y] * ASFR[a, r]
    ]
) * SexRatio[sex]
```

With scenario adjustment applied to ASFR:

```
ASFR_adjusted[a, r] = ASFR[a, r] * ScenarioFactor
```

Where `ScenarioFactor` is:
- 1.00 for baseline (`constant`)
- 0.95 for restricted growth (`-5_percent`)
- 1.05 for high growth (`+5_percent`)
- `(1 - 0.005)^(y - base_year)` for `trending`

#### Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `Female_Pop[a, r, y]` | Female population at single-year age `a`, race `r`, year `y` | Previous projection step or 2025 base population |
| `ASFR[a, r]` | Annual age-specific fertility rate (births per woman per year) | CDC NCHS 2024/2022, stored in `data/raw/fertility/asfr_processed.csv` |
| `SexRatio[Male]` | Proportion of births that are male | 0.51 (configurable: `rates.fertility.sex_ratio_male`) |
| `SexRatio[Female]` | Proportion of births that are female | 0.49 |
| `ScenarioFactor` | Multiplicative adjustment for scenario | ADR-037: 0.95 (restricted), 1.05 (high), 1.00 (baseline) |
| `Survival_Rate[age=0, sex, r]` | Probability of surviving from age 0 to age 1 | CDC life tables with optional annual improvement |

**File references**:
- ASFR data: `data/raw/fertility/asfr_processed.csv`
- Rate transformation: `scripts/pipeline/02_run_projections.py`, `_transform_fertility_rates()` (line 189)
- Birth calculation: `cohort_projections/core/fertility.py`, `calculate_births()` (line 16)
- Scenario adjustment: `cohort_projections/core/fertility.py`, `apply_fertility_scenario()` (line 208)
- Engine integration: `cohort_projections/core/cohort_component.py`, `project_single_year()` (lines 287-289)
- Configuration: `config/projection_config.yaml`, `rates.fertility` section

---

### 4.3 Numerical Example

#### SDC 2024: Cass County, 2020-2025

Assuming Cass County has 5,000 females aged 25-29 in 2020:

```
Births_25_29 = 5,000 * 0.596142 = 2,981 (over 5 years)
```

Summing across all age groups (hypothetical total of 15,000 reproductive-age females):

```
Total_Births = 15,000 * weighted_average_rate = ~9,500 (over 5 years)
Male_Births   = 9,500 * 0.512 = 4,864
Female_Births = 9,500 * 0.488 = 4,636

Pop[0-4, Male, Cass, 2025]   = 4,864 * ~0.993 = 4,830
Pop[0-4, Female, Cass, 2025] = 4,636 * ~0.995 = 4,613
```

#### Current 2026: Cass County, 2025-2026 (Single Year)

Same county, but using national rates and annual calculation. A 25-year-old White NH woman:

```
Births_25_whiteNH = Female_Pop[25, WhiteNH, 2025] * 0.0892 = 1 * 0.0892 = 0.0892
```

Summing across all 35 single-year ages (15-49) and 6 races for all ~15,000 reproductive-age females:

```
Total_Births = ~1,600 (for one year)
Male_Births   = 1,600 * 0.51 = 816
Female_Births = 1,600 * 0.49 = 784
```

Pop[age=0, Male, all_races, 2026] = 816 (split by mother's race)
Pop[age=0, Female, all_races, 2026] = 784 (split by mother's race)

Over 5 years: approximately 1,600 * 5 = ~8,000 births (versus SDC's ~9,500). The difference arises primarily from the lower TFR (~1.62 national vs. ~1.77 Cass county-specific in SDC).

---

## 5. Rationale for Changes

### 5.1 National vs. County-Specific Rates

**Change**: From SDC's county-specific blended rates to national rates applied uniformly.

**Rationale**:
- County-level fertility rate estimation for small counties (many ND counties have < 5,000 population) is inherently noisy. The SDC addresses this with a blending methodology, but the blending weights are undocumented and thus not reproducible.
- The current system achieves county-level birth variation through demographic composition (age + race structure), which is a more principled approach when county-specific vital statistics are unavailable or unreliable.
- North Dakota does not publish machine-readable county-level ASFR by race/ethnicity. Attempting to construct such rates would require small-area estimation techniques that add complexity without clear benefit (see ADR-001, Alternative 4: Bayesian Small Area Estimation -- rejected as "overkill for state-level projections").

**When to reconsider**: If ND DHHS publishes county-level race-specific ASFR, or if small-area estimation methods are validated for ND counties.

### 5.2 Race-Specific Fertility

**Change**: From no race dimension to 6-category race-specific ASFR.

**Rationale**:
- The current system projects population by race/ethnicity, which requires race-specific fertility rates to correctly distribute births.
- Fertility differentials by race are substantial in the U.S.: Hispanic TFR (1.96) is 28% higher than White NH TFR (1.53). Ignoring these differentials would misallocate births by race and produce incorrect racial composition projections.
- The 2020 Census shows ND's AIAN population is ~5% of total but has distinct fertility patterns. Without race-specific rates, AIAN population projections would be inaccurate.

### 5.3 Annual vs. 5-Year Calculation

**Change**: From 5-year batch birth calculation to annual birth calculation.

**Rationale**:
- Annual calculation matches the single-year age resolution of the current system. A 5-year calculation requires averaging or aggregating across 5 single-year ages, which introduces approximation.
- Annual calculation also allows year-by-year scenario adjustments (e.g., the `trending` fertility scenario applies a compounding annual decline) and integrates cleanly with the annual survival/aging cycle.
- The SDC's 5-year step necessarily introduces timing approximations: all births in a 5-year period use the population at the start of the period, not accounting for population changes within the period. Annual steps reduce this approximation.

### 5.4 Infant Survival Approach

**Change**: From explicit infant survival in the birth step to implicit survival in the aging step.

**Rationale**:
- In the SDC system, births are calculated for a 5-year period and then a survival factor is applied to convert total births into the 0-4 population at period end. This is mathematically necessary because the births span 5 years and some will die before the end of the period.
- In the annual system, births enter the population at age 0 for the given year. Survival from age 0 to age 1 is applied in the next annual step. This is more precise (applying age-specific infant survival rather than a 0-4 group survival rate) and consistent with the annual aging mechanism.
- The functional effect is equivalent: both systems account for infant mortality. The annual approach is slightly more granular because it uses single-year age-0 survival rates rather than a 5-year group rate.

### 5.5 TFR Level

**Change**: From SDC state-average TFR ~2.33 (ND, 2016-2022 blended) to national TFR ~1.62 (CDC, 2024).

**Rationale**:
- The SDC rates reflect a period (2016-2022) that includes the tail of the Bakken boom and pre-pandemic fertility levels. U.S. fertility has declined substantially since then; the 2024 national TFR of ~1.62 is at a historic low.
- ND-specific fertility may be slightly higher than the national rate, but the SDC's 44% premium (2.33 vs. 1.62) likely overstates the difference, as the SDC rates are also influenced by the blending methodology and the older reference period.
- Using national rates is a defensible default for a projection system that disaggregates by race (since national race-specific rates are well-estimated) rather than by county (where rates are noisy).

**When to reconsider**: If ND-specific race-age-specific ASFR data becomes available, it would be appropriate to use state-specific rates rather than national rates.

### 5.6 Scenario Adjustments

**Change**: From no fertility scenarios to CBO-grounded +/-5% adjustments.

**Rationale**:
- ADR-037 establishes that all scenario parameters should be empirically grounded, not arbitrary.
- The -5% fertility adjustment in the restricted growth scenario is derived from CBO's own TFR revision between their January 2025 and January 2026 Demographic Outlook publications (-4.3%, rounded to -5%).
- The +5% fertility adjustment in the high growth scenario is symmetric and grounded in the differential between foreign-born TFR (~1.79) and native-born TFR (~1.53) -- a larger immigrant population pushes aggregate fertility upward.
- The `trending` option (0.5% annual decline) is available but not used in any active scenario. It is retained as infrastructure for future fertility-decline scenarios.

### 5.7 Sex Ratio at Birth

**Change**: From 0.512 to 0.51.

**Rationale**:
- The difference is negligible (0.2 percentage points). Both values are within the biological range for human populations.
- The commonly cited ratio of 1.05 male births per female birth implies 51.22% male. The SDC's 0.512 is slightly closer to this value; the current system's 0.51 is a round-number approximation.
- Over a 30-year projection for a state with ~10,000 births per year, the difference produces approximately 60 fewer male births total -- well within projection uncertainty.

---

## References

### Architecture Decision Records
- **ADR-001**: Fertility Rate Processing Methodology (`docs/governance/adrs/001-fertility-rate-processing.md`)
- **ADR-037**: CBO-Grounded Scenario Methodology (`docs/governance/adrs/037-cbo-grounded-scenario-methodology.md`)

### Source Code
- **Birth calculation engine**: `cohort_projections/core/fertility.py` -- `calculate_births()`, `apply_fertility_scenario()`, `validate_fertility_rates()`
- **Fertility data processing**: `cohort_projections/data/process/fertility_rates.py` -- `process_fertility_rates()`, `create_fertility_rate_table()`
- **Rate transformation pipeline**: `scripts/pipeline/02_run_projections.py` -- `_transform_fertility_rates()`, `_expand_age_groups_to_single_years()`
- **Projection engine (current)**: `cohort_projections/core/cohort_component.py` -- `project_single_year()`
- **SDC replication engine**: `../sdc_2024_replication/scripts/projection_engine.py` -- `calculate_births()`, `apply_infant_survival()`
- **Configuration**: `config/projection_config.yaml` -- `rates.fertility`, `scenarios`

### Data Files
- **SDC fertility rates**: `../sdc_2024_replication/data/fertility_rates_by_county.csv` (53 counties x 8 age groups = 424 records)
- **Current ASFR input**: `data/raw/fertility/asfr_processed.csv` (7 age groups x 6 races = 42 records, 2024/2022 vintages)
- **Data source documentation**: `data/raw/fertility/DATA_SOURCE_NOTES.md`

### SDC Documentation
- **Methodology specification**: `../sdc_2024_replication/METHODOLOGY_SPEC.md`, Sections 2.4, 3.1, 7.3
- **SDC 2024 Report**: "Population Projections for the State of North Dakota and its Counties" (February 6, 2024)
- **SDC source workbook**: `Projections_Base_2023.xlsx`

### External Sources
- CDC NCHS Vital Statistics Rapid Release, Natality Dashboard (2024)
- National Vital Statistics Reports, Vol. 73 No. 2 (Births: Final Data for 2022)
- National Vital Statistics Reports, Vol. 74 No. 1 (Births: Final Data for 2023)
- CBO Publication 60875, Demographic Outlook (January 2025)
- CBO Publication 61879, Demographic Outlook (January 2026)
