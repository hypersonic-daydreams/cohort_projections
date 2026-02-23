# Methodology Comparison: Base Population Setup

**Series**: SDC 2024 vs. Current 2026 Cohort-Component Projections
**Section**: 01 -- Base Population Construction
**Date**: 2026-02-20
**Author**: Methodology review (automated)

---

## 1. SDC 2024 Approach

### 1.1 Overview

The North Dakota State Data Center (SDC) 2024 projections use the **2020 Decennial Census** (April 1, 2020) as their base population. The base year population is organized by county, five-year age group, and sex. Race/ethnicity is **not a dimension** of the SDC projection; the base population carries no race disaggregation.

### 1.2 Base Year and Vintage

| Attribute | Value |
|-----------|-------|
| Base year | 2020 (April 1, Census Day) |
| Source | 2020 Decennial Census |
| State total | 779,094 |

The SDC anchors to the decennial census directly rather than to a postcensal Population Estimates Program (PEP) vintage. This is the simplest possible choice: the census is the highest-quality population count, and using it avoids any dependence on the PEP estimation methodology.

### 1.3 Age Grouping

The SDC uses **18 five-year age groups**:

| Index | Group | Index | Group |
|-------|-------|-------|-------|
| 1 | 0-4 | 10 | 45-49 |
| 2 | 5-9 | 11 | 50-54 |
| 3 | 10-14 | 12 | 55-59 |
| 4 | 15-19 | 13 | 60-64 |
| 5 | 20-24 | 14 | 65-69 |
| 6 | 25-29 | 15 | 70-74 |
| 7 | 30-34 | 16 | 75-79 |
| 8 | 35-39 | 17 | 80-84 |
| 9 | 40-44 | 18 | 85+ (open-ended) |

The 85+ group is an open-ended terminal category. Because the projection operates in five-year steps (2020, 2025, 2030, ..., 2050), the five-year age grouping is naturally aligned with the projection interval -- each cohort advances exactly one age group per period. There is no need for interpolation or disaggregation within groups.

### 1.4 Sex Detail

The SDC projects males and females separately. Every cell in the base population is identified by `(county, age_group, sex)`. This yields:

```
53 counties x 18 age groups x 2 sexes = 1,908 cells
```

### 1.5 Race/Ethnicity Treatment

**Race is not included.** The SDC report contains no race-specific projections, no race-specific rates (fertility, mortality, migration), and no race dimension in the base population matrix. All demographic rates are applied uniformly regardless of race composition.

### 1.6 Geographic Resolution

The primary projection unit is the **county** (53 counties). State totals are computed by summing county projections. The SDC also reports results for 8 economic planning regions, which are aggregations of counties.

### 1.7 Data Sources

| Input | Source |
|-------|--------|
| Base population | 2020 Decennial Census, by county x age group x sex |
| Workbook | `Projections_Base_2023.xlsx` |

### 1.8 Interpolation and Disaggregation

None. The five-year age groups from the Census are used directly without interpolation. The five-year projection step matches the five-year age grouping, so no within-group splitting is required.

### 1.9 Known Issues

The SDC workbook (`Projections_Base_2023.xlsx`) produces a 2020 base population of 749,134 -- a gap of 29,960 from the published total of 779,094. This discrepancy remains unresolved and suggests the workbook may represent an earlier version or exclude certain population groups. See `sdc_2024_replication/METHODOLOGY_SPEC.md`, Section 6.1.

---

## 2. Current 2026 Approach

### 2.1 Overview

The 2026 projections use **Census Bureau Population Estimates Program (PEP) Vintage 2025** county totals as the base population, disaggregated into a detailed cohort matrix of `(county, single_year_age, sex, race_ethnicity)` using a multi-source distribution pipeline. The projection engine operates at single-year-of-age resolution with annual time steps.

### 2.2 Base Year and Vintage

| Attribute | Value |
|-----------|-------|
| Base year | 2025 (July 1) |
| Population total source | Census PEP Vintage 2025 (`data/raw/population/nd_county_population.csv`, column `population_2025`) |
| State total | 799,358 |
| Config reference | `config/projection_config.yaml`, `project.base_year: 2025` |

The choice to use PEP V2025 rather than the 2020 Census provides a more current starting point (5 years closer to the present), incorporates post-census demographic changes, and reduces the horizon over which extrapolation occurs.

### 2.3 Age Grouping

The engine operates at **single-year-of-age** resolution, ages 0 through 90, where age 90 is the open-ended terminal group (90+). This yields **91 age categories**.

```
Config: demographics.age_groups.min_age = 0
Config: demographics.age_groups.max_age = 90
```

The base population distribution is loaded at single-year resolution. Two resolution modes are supported, controlled by `base_population.age_resolution` in config:

| Mode | Resolution | Source | Distribution Rows | Status |
|------|-----------|--------|-------------------|--------|
| `"single_year"` (default) | Ages 0-90 | SC-EST2024-ALLDATA6 | 1,092 | Current production |
| `"five_year_uniform"` (legacy) | 18 five-year groups, uniformly split | cc-est2024-alldata | 216 (expanded to 1,092) | Retained for backward compatibility |

The single-year mode (ADR-048) uses Census Bureau **SC-EST2024-ALLDATA6**, which provides state-level single-year-of-age x sex x race/ethnicity estimates directly from the Census Bureau's demographic analysis program. This eliminates the staircase artifacts produced by the legacy uniform splitting of five-year groups.

### 2.4 Sex Detail

Males and females are projected separately, consistent with SDC. Every cell is identified by `(county, age, sex, race)`.

### 2.5 Race/Ethnicity Categories

The 2026 projections carry **6 race/ethnicity categories** through the entire projection:

| Category | Label in Engine | ADR Reference |
|----------|-----------------|---------------|
| White alone, Non-Hispanic | `White alone, Non-Hispanic` | ADR-007 |
| Black alone, Non-Hispanic | `Black alone, Non-Hispanic` | ADR-007 |
| AIAN alone, Non-Hispanic | `AIAN alone, Non-Hispanic` | ADR-007 |
| Asian/PI alone, Non-Hispanic | `Asian/PI alone, Non-Hispanic` | ADR-007 |
| Two or more races, Non-Hispanic | `Two or more races, Non-Hispanic` | ADR-007 |
| Hispanic (any race) | `Hispanic (any race)` | ADR-007 |

Asian and Native Hawaiian/Pacific Islander are combined into a single category per ADR-007, driven by data availability constraints.

### 2.6 Distribution Data Sources and Evolution

The methodology for constructing the age-sex-race distribution evolved through three stages:

| Stage | ADR | Data Source | Rows | Key Defect |
|-------|-----|-------------|------|------------|
| Original | -- | ACS PUMS only (~12,277 ND records) | 115 | Sex ratio = 119.1 (actual: 105.5) |
| Hybrid | ADR-041 | Census cc-est2024-agesex-all (age-sex) + PUMS (race within cell) | 115 | Zero Black females at reproductive ages; 45% of Hispanics in one cell |
| Full-count | ADR-044 | Census cc-est2024-alldata-38 (full joint distribution) | 216 | State-level distribution applied to all counties identically |

**Current production** uses the ADR-044 full-count approach as the statewide anchor, with ADR-047 county-specific distributions and ADR-048 single-year-of-age resolution layered on top.

### 2.7 State-Level Distribution: Single-Year-of-Age (ADR-048)

**Source file**: `data/raw/population/sc-est2024-alldata6.csv` (Census Bureau State Characteristics, Vintage 2024)

**Output file**: `data/raw/population/nd_age_sex_race_distribution_single_year.csv` (1,092 rows + header = 1,093 lines on disk)

**Processing pipeline** (implemented in `scripts/data/build_race_distribution_from_census.py`, function `build_single_year_statewide_distribution()`):

1. Read `sc-est2024-alldata6.csv`
2. Filter to `STATE=38` (North Dakota)
3. Exclude totals: `SEX in {1, 2}` and `ORIGIN in {1, 2}` (drop SEX=0 and ORIGIN=0 aggregate rows)
4. Map each `(ORIGIN, RACE)` pair to project race category using `SCEST_RACE_MAP`:
   - `(1,1)` -> `white_nonhispanic`, `(1,2)` -> `black_nonhispanic`, `(1,3)` -> `aian_nonhispanic`
   - `(1,4)` and `(1,5)` -> `asian_nonhispanic` (Asian + NHPI combined)
   - `(1,6)` -> `multiracial_nonhispanic`
   - `(2, any)` -> `hispanic` (all races under Hispanic origin)
5. Map `SEX`: `1` -> `male`, `2` -> `female`
6. Use `POPESTIMATE2024` column
7. Aggregate by `(AGE, sex, race_ethnicity)`, summing population (merges NHPI into Asian, merges all Hispanic races)
8. For ages 0-84: use directly as single years
9. For age 85+: expand to ages 85-90 using exponential-decay terminal age weights
10. Compute proportions: `proportion = cell_count / total_population`
11. Write to CSV with columns `[age, sex, race_ethnicity, estimated_count, proportion]`

**Schema**: Each row represents one `(age, sex, race)` cell with its estimated count and its share of the total state population.

Sample rows from the output file:

```
age,sex,race_ethnicity,estimated_count,proportion
0,female,aian_nonhispanic,245.0,0.000307569...
0,female,asian_nonhispanic,140.0,0.000175753...
0,female,black_nonhispanic,358.0,0.000449428...
0,female,hispanic,549.0,0.000689206...
0,female,multiracial_nonhispanic,343.0,0.000430597...
0,female,white_nonhispanic,3013.0,0.003782476...
```

### 2.8 Terminal Age Expansion: 85+ to Ages 85-90

The SC-EST data provides ages 0 through 85, where 85 represents the open-ended 85+ group. The projection engine requires ages 0-90 (where 90 is the engine's open-ended terminal group). The 85+ population is distributed across ages 85-90 using **exponential decay** with a survival factor of 0.7 per year.

**Formula** (implemented in `_terminal_age_weights()`, `build_race_distribution_from_census.py`, lines 357-377):

For ages 85-89:
```
weight[i] = s^(i - 85)    for i = 85, 86, 87, 88, 89
```

For age 90 (open-ended 90+, absorbing the geometric tail):
```
weight[90] = s^5 / (1 - s)
```

Where `s = 0.7` is the `TERMINAL_SURVIVAL_FACTOR`.

**Normalization**: All weights are divided by their sum so they total 1.0.

**Computed weights** (s = 0.7):

| Age | Raw Weight | Normalized Weight |
|-----|-----------|------------------|
| 85 | 1.0000 | 0.2647 |
| 86 | 0.7000 | 0.1853 |
| 87 | 0.4900 | 0.1297 |
| 88 | 0.3430 | 0.0908 |
| 89 | 0.2401 | 0.0636 |
| 90 | 0.5604 | 0.1483 |
| **Sum** | **3.7765** | **1.0000** |

Note: The 90+ weight is larger than ages 87-89 because it absorbs the entire geometric tail (ages 90, 91, 92, ...). This is standard demographic practice for terminal open-ended age groups.

### 2.9 County-Specific Distributions (ADR-047)

Rather than applying a single statewide distribution to all 53 counties (which would give Fargo the same youth share as Slope County, and Sioux County the statewide 4.8% AIAN share instead of its actual 78%), county-specific distributions are built from the same `cc-est2024-alldata-38.csv` source.

**Output file**: `data/processed/county_age_sex_race_distributions.parquet`

**Dimensions**: 53 counties x 216 cells = **11,448 rows**

**Schema**: `[fips, age_group, sex, race, proportion]`

Note that the county distributions are stored at the 5-year age group level (from cc-est2024-alldata), not at single-year resolution. The expansion from 5-year groups to single years happens at load time in the base population loader.

**Processing pipeline** (function `build_county_distributions()` in `build_race_distribution_from_census.py`):

1. Read `cc-est2024-alldata-38.csv`, filter to `YEAR=6` (July 1, 2024), `AGEGRP > 0`, `COUNTY > 0`
2. Map `AGEGRP` codes (1-18) to age group strings ("0-4" through "85+")
3. Construct 5-digit FIPS: `"38" + zero-padded 3-digit county code`
4. For each county:
   a. Extract race-sex columns using `RACE_COLUMN_MAP` (same mapping as statewide)
   b. Compute per-county proportion: `cell_count / county_total`
   c. Apply blending for small counties (see below)
5. Write to Parquet

### 2.10 Small-County Blending (ADR-047)

For counties with total population below a configurable threshold (default: 5,000), the county-specific distribution is blended with the statewide distribution to prevent instability from zero-cell artifacts.

**Formula**:

```
alpha = min(county_population / threshold, 1.0)

blended_proportion = alpha * county_proportion + (1 - alpha) * state_proportion
```

Where:
- `threshold = 5,000` (config: `base_population.county_distributions.blend_threshold`)
- `alpha = 1.0` for counties with population >= 5,000 (no blending; pure county distribution)
- `alpha < 1.0` for counties below threshold, proportional to their population

**After blending**, proportions are renormalized so they sum to 1.0 for each county.

**Rationale for the 5,000 threshold**: Counties below 5,000 have >30% zero cells in the 14 race-sex columns x 18 age groups. Counties above 5,000 have <15% zero cells. The threshold marks the transition from sparse to dense cell coverage.

**Example**:
- Slope County (population ~660): `alpha = 660 / 5000 = 0.132`, so 86.8% statewide + 13.2% county-specific
- Benson County (population ~5,756): `alpha = min(5756 / 5000, 1.0) = 1.0`, so 100% county-specific

### 2.11 Loading and Expansion at Runtime

The base population loader (`cohort_projections/data/load/base_population_loader.py`) performs the following at runtime when loading a county's base population:

**Step 1**: Load statewide single-year distribution (1,092 rows from `nd_age_sex_race_distribution_single_year.csv`)

**Step 2**: Attempt to load county-specific distribution from `county_age_sex_race_distributions.parquet`. If county distributions are disabled or the file is missing, fall back to statewide.

**Step 3**: If using county-specific distribution (which has 5-year groups), expand to single years. The expansion method depends on `age_resolution`:

- **`"single_year"` mode** (current): Use statewide single-year proportions as **within-group interpolation weights**. For each 5-year group in the county distribution, the statewide single-year pattern within that group determines how the county's group proportion is split across the 5 (or 6, for 85+) single years.

- **`"five_year_uniform"` mode** (legacy): Divide each 5-year group proportion equally across constituent single years.

**Within-group weight computation** (function `_build_statewide_single_year_weights()`, lines 295-356):

For each `(age_group, sex, race)` combination:

1. Collect statewide single-year proportions for all ages within the group
2. Normalize so they sum to 1.0 within the group
3. These normalized weights are used to distribute the county's 5-year proportion

```
For group "25-29", Male, White alone Non-Hispanic:
  State proportions: age_25=0.00523, age_26=0.00510, age_27=0.00498, age_28=0.00487, age_29=0.00475
  Within-group sum: 0.02493
  Weights: age_25=0.2098, age_26=0.2045, age_27=0.1997, age_28=0.1953, age_29=0.1905

  County group proportion: 0.0150 (hypothetical)
  -> age_25 = 0.0150 * 0.2098 = 0.003147
  -> age_26 = 0.0150 * 0.2045 = 0.003068
  ... etc.
```

This preserves the county's overall group-level race composition while imposing a smooth within-group age profile derived from the higher-quality statewide data.

**Step 4**: Look up the county's total population from `nd_county_population.csv` (column `population_2025`).

**Step 5**: Multiply proportions by total population:

```
population[age, sex, race] = proportion[age, sex, race] * county_total_population
```

**Step 6**: Add `year = base_year` column; output DataFrame with schema `[year, age, sex, race, population]`.

**Final cell count per county**: 91 ages x 2 sexes x 6 races = **1,092 cells** per county.

**Total cells for all 53 counties**: 53 x 1,092 = **57,876 cells**.

### 2.12 Normalization and Completeness Guarantees

The loader enforces two guarantees:

1. **Completeness**: After loading any distribution, the loader merges with a complete index of all expected `(age, sex, race)` combinations (91 x 2 x 6 = 1,092). Missing cells are filled with `proportion = 0.0`. This ensures every county has a full cohort matrix even if the source distribution has gaps.

2. **Normalization**: After completeness filling, proportions are renormalized to sum to 1.0. This step runs twice -- once after loading the raw distribution and once after the completeness merge -- to ensure the population adds up exactly to the county total.

### 2.13 Data Vintage Mismatch

There is a deliberate vintage mismatch in the current pipeline:

| Component | Vintage | Date |
|-----------|---------|------|
| County total population | PEP Vintage 2025 | July 1, 2025 |
| State single-year distribution | SC-EST2024-ALLDATA6 (V2024) | July 1, 2024 |
| County race distributions | cc-est2024-alldata (V2024) | July 1, 2024 |

The V2024 proportions are applied to V2025 population totals. ADR-044 and ADR-048 both acknowledge this gap as negligible: racial composition and age structure change slowly (<1% shift in any group's share annually). When V2025 county characteristics files are released (expected mid-2026), the proportions can be updated.

---

## 3. Key Differences

| Dimension | SDC 2024 | Current 2026 |
|-----------|----------|--------------|
| **Base year** | 2020 (April 1, Census Day) | 2025 (July 1, PEP estimate) |
| **Population source** | 2020 Decennial Census | Census PEP Vintage 2025 |
| **State total** | 779,094 | 799,358 |
| **Age resolution** | 18 five-year groups (0-4 through 85+) | 91 single-year ages (0 through 90+) |
| **Age grouping source** | Direct from Census (no transformation) | SC-EST2024-ALLDATA6 single-year estimates |
| **Terminal age group** | 85+ (open-ended) | 85+ expanded to 85-90 using exponential decay (s=0.7) |
| **Sex** | Male, Female | Male, Female |
| **Race/ethnicity** | **Not included** | 6 categories (White NH, Black NH, AIAN NH, Asian/PI NH, Multiracial NH, Hispanic) |
| **Race data source** | N/A | Census cc-est2024-alldata (5yr groups) + SC-EST2024-ALLDATA6 (single year) |
| **Geographic resolution** | 53 counties | 53 counties (+ places) |
| **County-specific detail** | Each county has its own Census age-sex counts | Each county has its own distribution from cc-est2024-alldata, with blending for small counties |
| **Small-county treatment** | Direct Census counts (no blending needed) | Population-weighted blending with statewide distribution for counties < 5,000 |
| **Interpolation** | None (5-yr groups match 5-yr step) | Statewide single-year pattern used as within-group weights for county 5-yr data |
| **Cells per county** | 18 x 2 = 36 | 91 x 2 x 6 = 1,092 |
| **Total cells (53 counties)** | 1,908 | 57,876 |
| **Projection interval** | 5-year | 1-year (annual) |

---

## 4. Formulas and Calculations

### 4.1 SDC: Base Population Assignment

The SDC approach requires no formula for base population setup. Census counts are read directly from tabulations:

```
Pop[county, age_group, sex, t=2020] = Census_2020_Count[county, age_group, sex]
```

### 4.2 Current: Statewide Single-Year Proportion

From SC-EST data, after race mapping and aggregation:

```
proportion[age, sex, race] = estimated_count[age, sex, race] / SUM(estimated_count)
```

Where the sum runs over all `(age, sex, race)` cells.

### 4.3 Current: Terminal Age Expansion (85+ -> 85-90)

```
s = 0.7    (TERMINAL_SURVIVAL_FACTOR)

For ages 85-89:
  raw_weight[i] = s^(i - 85)

For age 90 (open-ended):
  raw_weight[90] = s^5 / (1 - s)

normalized_weight[i] = raw_weight[i] / SUM(raw_weight)

expanded_count[i, sex, race] = count_85plus[sex, race] * normalized_weight[i]
```

### 4.4 Current: County-Specific Proportion

From cc-est2024-alldata, for each county:

```
county_proportion[age_grp, sex, race] = county_count[age_grp, sex, race] / county_total
```

Where `county_total = SUM(county_count)` across all cells for that county.

### 4.5 Current: Small-County Blending

```
threshold = 5000    (base_population.county_distributions.blend_threshold)

alpha = min(county_population / threshold, 1.0)

blended[age_grp, sex, race] = alpha * county_proportion[age_grp, sex, race]
                              + (1 - alpha) * state_proportion[age_grp, sex, race]

# Renormalize:
blended[age_grp, sex, race] = blended[age_grp, sex, race] / SUM(blended)
```

### 4.6 Current: Within-Group Single-Year Weight Computation

For expanding county 5-year groups to single years using statewide pattern:

```
For a given (age_group, sex, race):
  Let A = set of single-year ages in the group (e.g., {25, 26, 27, 28, 29})

  state_prop[a] = statewide single-year proportion for age a, sex, race
  group_sum = SUM over a in A of state_prop[a]

  within_group_weight[a] = state_prop[a] / group_sum    (normalized to sum to 1.0)
```

If `group_sum = 0` (statewide has zero for this sex-race in this age group), fall back to uniform: `within_group_weight[a] = 1 / |A|`.

### 4.7 Current: County Single-Year Proportion Expansion

```
For each (age_group, sex, race) in the county distribution:
  For each age a in age_group:
    single_year_proportion[a, sex, race] = county_5yr_proportion[age_group, sex, race]
                                            * within_group_weight[a]
```

This guarantees that the sum of single-year proportions within each 5-year group equals the original 5-year group proportion.

### 4.8 Current: Final Population Assignment

```
population[county, age, sex, race] = proportion[age, sex, race] * county_total_population

Where county_total_population is read from nd_county_population.csv, column population_2025.
```

---

## 5. Rationale for Changes

### 5.1 Base Year: 2020 -> 2025

**Why changed**: The SDC's 2020 base year was dictated by the timing of the 2020 Census. Our projections begin from 2025, using PEP Vintage 2025 county totals. This provides a more current starting point, reduces the projection horizon (30 years from 2025 vs. 30 years from 2020), and incorporates five years of observed demographic change (including COVID-19 impacts, post-census migration shifts, and updated population estimates).

**Trade-off**: PEP estimates are postcensal and carry estimation uncertainty, whereas the decennial census is a full count (though it too has coverage error). The PEP methodology is well-established and regularly validated against subsequent censuses.

### 5.2 Age Resolution: Five-Year Groups -> Single-Year Ages

**Why changed (ADR-048)**: The projection engine operates at single-year-of-age resolution with annual time steps. When the base population was provided only in 5-year groups, the loader uniformly split each group across 5 single years (`proportion_per_year = group_proportion / 5`). This created staircase artifacts -- flat plateaus within each group with abrupt jumps at boundaries (e.g., a ~4.4% jump between ages 29 and 30). These artifacts persisted through the entire 30-year projection, never smoothing out, and caused a spurious -38% drop in annual growth rate at 2046-2047 as the artificially shaped base-year cohorts aged through key demographic thresholds.

**How SDC avoids this**: The SDC operates entirely in 5-year groups with 5-year time steps, so the step functions never arise. Their approach avoids the artifact but at the cost of coarser resolution.

**Our solution**: Use Census SC-EST2024-ALLDATA6, which provides the Census Bureau's own single-year-of-age estimates. This is real data, not interpolation, and eliminates the need for any within-group splitting at the state level. For county-level data (where only 5-year groups are available), the statewide single-year pattern serves as interpolation weights, producing smooth within-group profiles.

**References**: ADR-048, `base_population_loader.py` lines 67-103 (`_load_single_year_distribution`), `build_race_distribution_from_census.py` lines 380-491 (`build_single_year_statewide_distribution`).

### 5.3 Race/Ethnicity: Not Included -> 6 Categories

**Why changed**: The SDC does not project population by race. For a state agency producing projections for planning purposes, race-specific projections are increasingly important for:
- Facilities planning (health, education) in reservation communities with predominantly AIAN populations
- Understanding differential growth trajectories by racial group
- Producing projections that are useful for federal reporting requirements

**Data evolution** (see ADR-041, ADR-044):
- **Stage 1**: PUMS-only distribution -- 115 of 216 cells populated, sex ratio of 119.1, zero Black females at reproductive ages. Catastrophically insufficient for a small state.
- **Stage 2 (ADR-041)**: Census+PUMS hybrid -- fixed sex ratio to 105.5 by using Census for age-sex structure, but PUMS for race within cells retained the 115-cell limit and the missing-group defects.
- **Stage 3 (ADR-044)**: Census full-count cc-est2024-alldata -- all 216 cells populated, all critical defects fixed, single authoritative source. This is the current approach.

**Validation results** (ADR-044):

| Metric | PUMS-based | Census Full-Count |
|--------|-----------|-------------------|
| Populated cells | 115 / 216 | 216 / 216 |
| Sex ratio | 105.5 (after ADR-041) | 105.5 |
| Black females 15-49 | 0 | 7,600 |
| Hispanic largest cell share | 45% (F 10-14) | 7.0% |
| Race categories | 6 | 6 |

**References**: ADR-007, ADR-041 (superseded), ADR-044.

### 5.4 County-Specific Distributions: Uniform Statewide -> Per-County

**Why changed (ADR-047)**: Applying a single statewide distribution to all 53 counties misallocates population for every county that deviates from the state average. The magnitude is substantial:

| County | Population | % Misallocated | Worst Error |
|--------|-----------|:--------------:|-------------|
| Sioux | 3,713 | 76.2% | AIAN: 2,903 actual -> 177 modeled |
| Rolette | 11,692 | 72.3% | AIAN: 8,867 actual -> 557 modeled |
| Benson | 5,756 | 46.6% | AIAN: 2,868 actual -> 274 modeled |
| Median (all 53) | -- | 17.6% | -- |

The reservation counties are the most severely affected: 46-76% of population is placed in the wrong race category. Since AIAN populations have distinct age structure (35-37% under 18 vs. 26.5% statewide) and different vital rates, the error compounds through the projection.

**The SDC avoids this problem differently**: Because the SDC does not include race, every county gets its own age-sex structure directly from the Census. The SDC's county age-sex distributions are inherently county-specific.

**Our county distributions** are built from the same `cc-est2024-alldata-38.csv` file used for the statewide distribution (ADR-044). The data was already available in the project; the prior approach simply aggregated across counties and discarded the county dimension.

**References**: ADR-047, `build_race_distribution_from_census.py` lines 603-726 (`build_county_distributions`), `base_population_loader.py` lines 359-568 (`load_county_age_sex_race_distribution`).

### 5.5 Small-County Blending: Not Applicable -> Alpha-Weighted

**Why introduced (ADR-047)**: For the 10 smallest counties (population 660-2,141), 37.1% of race-sex cells are zero. While many zeros are genuine (truly no residents of that group), the projection engine may behave unexpectedly when zero-population cells later receive migration inflows. Blending provides a small "floor" in otherwise-empty cells, analogous to a Bayesian prior.

**Why the SDC does not need this**: The SDC works with direct Census counts (no distributional splitting), uses no race dimension, and operates at a coarser resolution where even small counties have adequate cell populations.

**References**: ADR-047, `build_race_distribution_from_census.py` lines 703-716 (blending implementation).

### 5.6 Terminal Age Expansion: Not Applicable -> Exponential Decay

**Why introduced**: The engine's age range is 0-90, with 90 as the open-ended terminal group. The SC-EST data provides ages 0-85+. The 85+ count must be distributed across ages 85, 86, 87, 88, 89, and 90+. A uniform split would overweight very old ages where survival is low. Exponential decay with s=0.7 approximates the age-specific survival gradient at very old ages, consistent with standard demographic practice.

**Why the SDC does not need this**: The SDC keeps 85+ as a single terminal group, never disaggregating it.

**References**: ADR-048, `build_race_distribution_from_census.py` lines 357-377 (`_terminal_age_weights`).

### 5.7 What Was Not Changed

| Aspect | SDC 2024 | Current 2026 | Why Unchanged |
|--------|----------|--------------|---------------|
| Sex categories | Male, Female | Male, Female | Binary sex is the standard Census reporting dimension and sufficient for fertility/mortality rate application |
| County as primary unit | 53 counties | 53 counties | County is the fundamental administrative geography for ND planning |
| Bottom-up aggregation | Counties sum to state | Counties sum to state | Ensures county-level accountability in state totals |

---

## 6. Source File Reference

| File | Purpose | Section |
|------|---------|---------|
| `sdc_2024_replication/METHODOLOGY_SPEC.md` | SDC 2024 methodology specification | Sections 1-2 |
| `cohort_projections/data/load/base_population_loader.py` | Runtime loader: distribution loading, expansion, county assembly | Sections 2.7-2.12 |
| `scripts/data/build_race_distribution_from_census.py` | Build script: statewide + county distributions + single-year | Sections 2.7-2.10 |
| `config/projection_config.yaml` | Configuration (base_population, demographics) | Sections 2.2-2.3 |
| `data/raw/population/nd_age_sex_race_distribution_single_year.csv` | Single-year statewide distribution (1,092 rows) | Section 2.7 |
| `data/raw/population/nd_age_sex_race_distribution.csv` | Five-year statewide distribution (216 rows) | Section 2.6 |
| `data/processed/county_age_sex_race_distributions.parquet` | County-specific distributions (11,448 rows) | Section 2.9 |
| `data/raw/population/nd_county_population.csv` | County total populations (V2025) | Section 2.2 |
| `data/raw/population/cc-est2024-alldata-38.csv` | Census county x age x sex x race (source) | Sections 2.6, 2.9 |
| `data/raw/population/sc-est2024-alldata6.csv` | Census state x single-year age x sex x race (source) | Section 2.7 |
| `docs/governance/adrs/041-census-pums-hybrid-base-population.md` | ADR-041: Hybrid approach (superseded) | Section 2.6 |
| `docs/governance/adrs/044-census-full-count-race-distribution.md` | ADR-044: Full-count race distribution | Sections 2.6, 5.3 |
| `docs/governance/adrs/047-county-specific-age-sex-race-distributions.md` | ADR-047: County-specific distributions | Sections 2.9-2.10, 5.4-5.5 |
| `docs/governance/adrs/048-single-year-of-age-base-population.md` | ADR-048: Single-year-of-age resolution | Sections 2.7-2.8, 5.2, 5.6 |
