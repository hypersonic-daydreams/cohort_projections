# Methodology Comparison: Mortality / Survival Rates

**SDC 2024 vs. Current 2026 Cohort-Component Projections**

| Attribute | Value |
|-----------|-------|
| **Document** | `docs/reviews/methodology_comparison/03_mortality.md` |
| **Date** | 2026-02-20 |
| **Component** | Mortality / Survival Rates |
| **Status** | Draft |

---

## 1. SDC 2024 Approach

### 1.1 Data Source

The SDC 2024 projections use **CDC NCHS Life Tables for North Dakota, published 2022 (for reference year 2020)**. The source publication is NVSR 71-02 (`nvsr71-02.pdf`). Processed life table files are stored as `ND1.xlsx`, `ND2.xlsx`, `ND3.xlsx`, `ND4.xlsx`.

- **Reference:** `../sdc_2024_replication/METHODOLOGY_SPEC.md`, Section 3.2
- **Extracted data:** `../sdc_2024_replication/data/survival_rates_by_county.csv`

### 1.2 Rate Structure

| Dimension | SDC 2024 Specification |
|-----------|----------------------|
| **Age groups** | 18 five-year groups: 0-4, 5-9, 10-14, ..., 80-84, 85+ |
| **Sex** | Male, Female (projected separately) |
| **Race/ethnicity** | Not differentiated; single set of rates for all races |
| **Geographic level** | **Statewide** -- same survival rates applied to all 53 counties |
| **Temporal variation** | **Constant** -- rates held fixed across all 6 projection periods (2020-2050) |

### 1.3 Survival Rate Computation

The SDC computes **5-year survival probabilities** -- the probability that an individual in a given 5-year age group at time *t* survives to the next 5-year age group at time *t+5*. These are derived from the CDC life table columns `qx` (probability of dying between exact ages *x* and *x+1*) and `lx` (number surviving to exact age *x* out of a radix of 100,000).

For regular age groups (0-4 through 80-84), the 5-year survival probability is read directly from the life table as the ratio of persons surviving across the 5-year span. Concretely, the SDC workbook (sheet "5-Year Survival Rate By Sex") stores pre-computed 5-year survival probabilities.

For the **85+ open-ended age group**, the SDC uses a formula based on the life table's `Tx` and `Lx` columns:

```
S(85+) = T(90) / (T(85) + L(85)/2)
```

Where:
- `T(90)` = total person-years lived above age 90
- `T(85)` = total person-years lived above age 85
- `L(85)` = person-years lived in the age interval 85-86
- The denominator approximates the effective person-years at risk in the 85+ group

This formula yields the probability that someone currently in the 85+ group will still be alive 5 years later (remaining within the 85+ group).

### 1.4 Extracted SDC Survival Rates

From `../sdc_2024_replication/data/survival_rates_by_county.csv`:

| Age Group | Male S(5yr) | Female S(5yr) |
|-----------|-------------|---------------|
| 0-4 | 0.991501 | 0.994584 |
| 5-9 | 0.999419 | 0.999408 |
| 10-14 | 0.998685 | 0.999420 |
| 15-19 | 0.998177 | 0.998345 |
| 20-24 | 0.994937 | 0.998019 |
| 25-29 | 0.992659 | 0.997046 |
| 30-34 | 0.989728 | 0.995355 |
| 35-39 | 0.987551 | 0.994393 |
| 40-44 | 0.985968 | 0.994304 |
| 45-49 | 0.980802 | 0.988215 |
| 50-54 | 0.977639 | 0.985892 |
| 55-59 | 0.965157 | 0.977838 |
| 60-64 | 0.952132 | 0.971673 |
| 65-69 | 0.933496 | 0.959411 |
| 70-74 | 0.897161 | 0.935792 |
| 75-79 | 0.835485 | 0.892027 |
| 80-84 | 0.735279 | 0.811291 |
| 85+ | 0.542841 | 0.638956 |

### 1.5 Application in Projection Engine

The SDC projection engine applies survival rates within the 5-year cohort-component step as follows:

**Regular age groups (5-9 through 80-84):**
```
Natural_Growth[age+5, t+5] = Population[age, t] * Survival_Rate[age, sex]
```

**Open-ended 85+ group** (absorbing state):
```
Natural_Growth[85+, t+5] = Pop[80-84, t] * Survival[80-84, sex]
                         + Pop[85+, t]   * Survival[85+, sex]
```

The 85+ group receives both (a) survivors from the 80-84 cohort who age into it and (b) survivors from within the 85+ group who do not die during the 5-year interval.

**Births (0-4 age group):**
```
Population[0-4, t+5] = Births[t to t+5] * Survival_Rate[0-4, sex]
```

The 0-4 survival rate serves as an infant/early childhood survival factor applied to total births.

- **Source:** `../sdc_2024_replication/scripts/projection_engine.py`, lines 253-318

### 1.6 No Mortality Improvement

The SDC holds survival rates constant across all projection periods. The same survival rates derived from the 2020 CDC life tables are applied in every 5-year step from 2020-2025 through 2045-2050. No adjustment is made for anticipated improvements in medical technology, public health, or age-specific mortality trends over the 30-year projection horizon.

---

## 2. Current 2026 Approach

### 2.1 Data Sources

The current system uses two data sources for survival rates, depending on the pipeline path:

1. **Static baseline:** CDC NCHS Life Tables, year **2023** (configured in `config/projection_config.yaml` as `rates.mortality.life_table_year: 2023`). Source files are SEER/CDC life tables providing `qx`, `lx`, `Lx`, `Tx`, and `ex` columns by single year of age, sex, and race/ethnicity.

2. **Time-varying mortality improvement (Phase 3):** Census Bureau **NP2023-A4** national survival ratio projections, adjusted to North Dakota using an ND-to-national scaling factor derived from CDC ND 2020 baseline rates. This produces year-specific survival rates for every calendar year in the projection.

- **ADR:** `docs/governance/adrs/002-survival-rate-processing.md`
- **Static processor:** `cohort_projections/data/process/survival_rates.py`
- **Time-varying processor:** `cohort_projections/data/process/mortality_improvement.py`
- **Engine application:** `cohort_projections/core/mortality.py`
- **Engine orchestrator:** `cohort_projections/core/cohort_component.py`

### 2.2 Rate Structure

| Dimension | 2026 Specification |
|-----------|-------------------|
| **Age groups** | Single-year ages: 0, 1, 2, ..., 89, 90+ (91 groups) |
| **Sex** | Male, Female |
| **Race/ethnicity** | 6 categories: White NH, Black NH, AIAN NH, Asian/PI NH, Two+ Races NH, Hispanic |
| **Geographic level** | **Statewide** -- same rates for all counties (but ND-adjusted from national) |
| **Temporal variation** | **Improving** -- 0.5% annual mortality improvement (configurable), or year-specific via Census NP2023 |

### 2.3 Survival Rate Computation (Static Baseline)

The static processor (`survival_rates.py`) supports three methods with automatic selection based on available columns.

**Method 1 -- lx method (preferred):**
```
S(x) = l(x+1) / l(x)
```
Where `lx` is the number surviving to exact age *x* from the life table radix (100,000). This is the most direct method and is used whenever the `lx` column is present.

**Method 2 -- qx method (fallback):**
```
S(x) = 1 - q(x)
```
Where `qx` is the probability of dying between exact ages *x* and *x+1*. This is simpler but does not account for within-interval timing effects.

**Method 3 -- Lx method:**
```
S(x) = L(x+1) / L(x)
```
Where `Lx` is person-years lived in the age interval *x* to *x+1*. This accounts for the timing of deaths within the interval.

The method is selected automatically:
```python
if 'lx' in df.columns:
    method = 'lx'   # Preferred
elif 'qx' in df.columns:
    method = 'qx'   # Fallback
else:
    raise ValueError("Need either lx or qx column")
```

- **Source:** `cohort_projections/data/process/survival_rates.py`, function `calculate_survival_rates_from_life_table()`

### 2.4 Terminal Age Group Handling (90+)

The 90+ open-ended group uses a special Tx-based formula when `Tx` and `Lx` are available:

```
S(90+) = T(91) / (T(90) + L(90)/2)

Where:
    T(91) = T(90) - L(90)     # person-years lived above age 91
    T(90) = total person-years lived above age 90
    L(90) = person-years lived in age interval 90-91
```

**Fallback strategy:**
1. If `Tx` and `Lx` available: use formula above
2. If only `qx` available: `S(90+) = 1 - q(90+)`
3. If neither available: use default `S(90+) = 0.65`

Typical values from this formula are in the range 0.60--0.70, consistent with empirical data.

- **Source:** `cohort_projections/data/process/survival_rates.py`, lines 306-320; `docs/governance/adrs/002-survival-rate-processing.md`, Decision 2

### 2.5 Mortality Improvement (Lee-Carter Style)

The baseline scenario applies a **0.5% annual mortality improvement** to death rates, compounded each year from the life table base year.

**Formula (applied to death rates, not survival rates):**
```
q(x, t) = q(x, base_year) * (1 - alpha)^(t - base_year)
S(x, t) = 1 - q(x, t)
```

Where:
- `alpha` = 0.005 (annual improvement factor, configurable)
- `t` = projection year
- `base_year` = life table reference year (2023 in current config)
- `q(x, base_year)` = base death probability from life table

The improvement is applied at runtime within the projection engine. For each projection year, the `apply_mortality_improvement()` function in `cohort_projections/core/mortality.py`:

```python
# Convert survival rate to death rate
death_rate = 1 - survival_rate

# Apply improvement
improvement_multiplier = (1 - improvement_factor) ** years_elapsed
death_rate = death_rate * improvement_multiplier

# Convert back to survival rate, capped at 1.0
survival_rate = min(1 - death_rate, 1.0)
```

**Configurable improvement factors:**

| Scenario | Factor | Description |
|----------|--------|-------------|
| Optimistic | 0.010 (1.0%) | Aggressive mortality decline |
| **Baseline** | **0.005 (0.5%)** | Historical U.S. average |
| Conservative | 0.0025 (0.25%) | Cautious improvement |
| Constant | 0.0 (0%) | No improvement (matches SDC) |

**Numerical example (Female age 70, base year 2023):**
- Base survival rate: S(70, 2023) = 0.9868
- Base death rate: q(70, 2023) = 0.0132
- At year 2045 (22 years elapsed):
  - `improvement_multiplier = (1 - 0.005)^22 = 0.8958`
  - `q(70, 2045) = 0.0132 * 0.8958 = 0.01182`
  - `S(70, 2045) = 1 - 0.01182 = 0.98818`
  - Net improvement: +0.00138 (from 0.9868 to 0.9882)

- **Source:** `cohort_projections/core/mortality.py`, function `apply_mortality_improvement()`; `config/projection_config.yaml`, `rates.mortality.improvement_factor`

### 2.6 Time-Varying Mortality Improvement (Phase 3 Pipeline)

The more sophisticated mortality improvement pipeline (`mortality_improvement.py`) produces **year-specific, ND-adjusted survival rates** using Census Bureau NP2023 national projections.

**Core formula:**
```
ND_survival[age, sex, year] = Census_NP2023[age, sex, year] * ND_adjustment[age, sex]

Where:
    ND_adjustment[age, sex] = ND_CDC_baseline[age, sex] / Census_national_2025[age, sex]
```

This approach has two advantages over the static Lee-Carter approximation:
1. The Census Bureau's own projected mortality improvement schedule (which is age-specific and nonlinear) is embedded in the NP2023 survival ratios.
2. The ND adjustment factor preserves North Dakota's mortality differential relative to the national average (ND may have higher or lower mortality at specific ages).

**Pipeline steps:**
1. Load Census Bureau NP2023-A4 survival ratio projections (2025-2055)
2. Load ND CDC baseline rates (expanded from 5-year groups to single-year ages)
3. Extract Census 2025 national baseline for computing adjustment factors
4. Compute ND adjustment factors: `ND_CDC[age,sex] / Census_national_2025[age,sex]`
5. Apply adjustments to all projection years: `Census_projected[age,sex,year] * adjustment[age,sex]`
6. Cap at 1.0, floor at 0.0

When time-varying survival rates are provided to the engine, the Lee-Carter improvement factor is **automatically disabled** to avoid double-counting:

```python
# In CohortComponentProjection.project_single_year():
if self.survival_rates_by_year is not None and year in self.survival_rates_by_year:
    survival_config = copy.deepcopy(self.config)
    survival_config["rates"]["mortality"]["improvement_factor"] = 0.0
```

- **Source:** `cohort_projections/data/process/mortality_improvement.py`; `cohort_projections/core/cohort_component.py`, lines 271-277

### 2.7 Application in Projection Engine

The 2026 engine operates on **single-year steps** (annual projection intervals).

**Regular ages (0 to 89):**
```
Survived_Population[age+1, t+1] = Population[age, t] * S(age, sex, race, t)
```

Each single-year cohort is advanced by exactly one year. Survival rates are age-, sex-, and race-specific.

**Open-ended 90+ group (absorbing state):**
```
Population[90+, t+1] = Pop[89, t] * S(89, sex, race, t)     # new entrants from age 89
                      + Pop[90+, t] * S(90+, sex, race, t)   # within-group survivors
```

Missing survival rates for the 90+ group default to 0.5 (rather than 0.0) to avoid completely eliminating the oldest-old population.

- **Source:** `cohort_projections/core/mortality.py`, function `apply_survival_rates()`

### 2.8 Validation and Quality Assurance

The system includes age-specific plausibility thresholds (ADR-002, Decision 4):

| Age Group | Expected S(x) Range | Error Threshold | Warning Threshold |
|-----------|---------------------|-----------------|-------------------|
| 0 (infant) | 0.993--0.995 | < 0.990 or > 0.998 | < 0.993 or > 0.995 |
| 1-14 (children) | > 0.9995 | < 0.999 | < 0.9995 |
| 15-44 (young adults) | > 0.999 | < 0.995 | < 0.999 |
| 45-64 (middle age) | 0.985--0.998 | < 0.98 or > 0.999 | -- |
| 65-84 (elderly) | 0.93--0.98 | < 0.90 or > 0.99 | < 0.93 or > 0.98 |
| 90+ (oldest-old) | 0.60--0.70 | < 0.50 or > 0.80 | < 0.60 or > 0.70 |

A simplified life expectancy at birth (e0) is computed for each sex-race combination as a QA metric:
```python
lx = cumulative_product(survival_rates)  # Survival to each age
e0 = sum(lx)                             # Approximation of life expectancy
```

Expected e0 ranges (U.S. 2020-2023): 75-87 years depending on sex and race.

- **Source:** `cohort_projections/data/process/survival_rates.py`, functions `validate_survival_rates()` and `calculate_life_expectancy()`

---

## 3. Key Differences

| Dimension | SDC 2024 | Current 2026 | Impact |
|-----------|----------|--------------|--------|
| **Life table vintage** | CDC ND 2020 (NVSR 71-02) | CDC 2023 (national, ND-adjusted) | 2023 tables reflect post-COVID mortality recovery; 3 more years of data |
| **Geographic specificity of life table** | North Dakota-specific | National life tables adjusted to ND via ratio method | ND-specific tables have higher sampling variance; national tables are more stable |
| **Age resolution** | 5-year groups (18 groups) | Single-year ages (91 groups: 0-90+) | Single-year avoids artificial within-group uniformity; finer-grained aging |
| **Terminal age group** | 85+ (open-ended) | 90+ (open-ended) | 90+ allows ages 85-89 to be modeled individually rather than pooled |
| **Terminal group formula** | `T(90) / (T(85) + L(85)/2)` | `T(91) / (T(90) + L(90)/2)` | Both use same Tx-based approach; differ only in where the open-ended cutoff falls |
| **Projection interval** | 5-year steps | Annual (1-year) steps | Annual steps compound survival more precisely; avoids lumping 5 years of mortality |
| **Sex-specific** | Yes | Yes | Same |
| **Race-specific** | No (single set of rates) | Yes (6 race/ethnicity categories) | Race-specific rates capture differential mortality (e.g., AIAN vs. Asian/PI) |
| **Mortality improvement** | None (constant rates) | 0.5% annual decline in death rates (Lee-Carter style) | Over 30 years, constant rates overstate deaths; improvement adds ~2-3 years to e0 |
| **Time-varying rates** | No | Yes (Census NP2023 year-specific rates available) | Census Bureau age-specific improvement trajectories, ND-adjusted |
| **Improvement mechanism** | N/A | `q(x,t) = q(x,base) * (1-0.005)^(t-base)` | Applied to death rates, converted back to survival; capped at S=1.0 |
| **Validation** | Manual inspection | Automated age-specific plausibility checks + life expectancy QA | Systematic validation catches data errors before they propagate |
| **Default for missing data** | N/A (complete for 2 sexes) | Age-appropriate defaults: infant=0.994, children=0.9995, adults=0.997, elderly=0.95, 90+=0.65 | Ensures projection engine runs for all race-age-sex combinations even with sparse data |

---

## 4. Formulas and Calculations

### 4.1 SDC 2024: 5-Year Survival Probability from Life Table

The SDC reads pre-computed 5-year survival probabilities directly from the CDC ND 2020 life table workbook. Conceptually, for a regular 5-year age group starting at age *a*:

```
S_5yr(a) = l(a+5) / l(a)
```

This is equivalent to the product of five consecutive single-year survival rates:

```
S_5yr(a) = product_{x=a}^{a+4} S(x) = product_{x=a}^{a+4} [1 - q(x)]
```

For the **85+ open-ended group**:
```
S_5yr(85+) = T(90) / (T(85) + L(85)/2)

Where:
    T(x)  = total person-years lived above age x = sum_{y=x}^{omega} L(y)
    L(x)  = person-years lived in age interval [x, x+1)
    omega = terminal age in life table
```

**Numerical examples from extracted SDC data:**

| Calculation | Male | Female |
|-------------|------|--------|
| S_5yr(0-4) | 0.991501 | 0.994584 |
| S_5yr(65-69) | 0.933496 | 0.959411 |
| S_5yr(85+) | 0.542841 | 0.638956 |

### 4.2 Current 2026: Single-Year Survival Rate from Life Table

**lx method (preferred):**
```
S(x) = l(x+1) / l(x)

For ages 0 through 89 (where l(x+1) exists in the table).
```

**qx method (fallback):**
```
S(x) = 1 - q(x)
```

**90+ open-ended group:**
```
S(90+) = T(91) / (T(90) + L(90)/2)

Where:
    T(91) = T(90) - L(90)
```

### 4.3 Conversion: SDC 5-Year Rate to Equivalent Annual Rate

For interoperability with the 2026 system, the SDC 5-year rates are converted to annual equivalents using the 5th root:

```
S_1yr(a) = S_5yr(a)^(1/5)
```

This assumes uniform mortality within the 5-year age group (a simplification).

**Numerical examples:**

| Age Group | Male S_5yr | Male S_1yr | Female S_5yr | Female S_1yr |
|-----------|-----------|-----------|-------------|-------------|
| 0-4 | 0.991501 | 0.998294 | 0.994584 | 0.998914 |
| 65-69 | 0.933496 | 0.986331 | 0.959411 | 0.991747 |
| 80-84 | 0.735279 | 0.940352 | 0.811291 | 0.959037 |
| 85+ | 0.542841 | 0.884982 | 0.638956 | 0.914312 |

These converted values are stored in `data/processed/sdc_2024/survival_rates_sdc_2024_by_age_group.csv` (column `survival_rate_1yr`) and in the expanded single-year file `data/processed/sdc_2024/survival_rates_sdc_2024.csv` where each 5-year rate is assigned uniformly to its constituent single-year ages.

### 4.4 Lee-Carter Mortality Improvement

**Death rate improvement (compounded annually):**
```
q(x, t) = q(x, base_year) * (1 - alpha)^(t - base_year)
```

**Equivalent survival rate:**
```
S(x, t) = 1 - q(x, t)
       = 1 - [q(x, base_year) * (1 - alpha)^(t - base_year)]
       = 1 - [(1 - S(x, base_year)) * (1 - alpha)^(t - base_year)]
```

**Capped at:**
```
S(x, t) = min(S(x, t), 1.0)
```

**Implementation parameters:**
- `alpha` (improvement_factor) = 0.005 (from `config/projection_config.yaml`)
- `base_year` = 2025 (from `config/projection_config.yaml`)
- `cap_survival_at` = 1.0

**Cumulative improvement multiplier over time:**

| Years Elapsed | Multiplier (1 - 0.005)^t | Death Rate Reduction |
|---------------|--------------------------|---------------------|
| 5 | 0.9752 | -2.5% |
| 10 | 0.9511 | -4.9% |
| 15 | 0.9277 | -7.2% |
| 20 | 0.9048 | -9.5% |
| 25 | 0.8826 | -11.7% |
| 30 | 0.8610 | -13.9% |

### 4.5 Census NP2023 ND-Adjusted Improvement

**Adjustment factor computation:**
```
ND_adjustment[age, sex] = ND_CDC_2020_baseline[age, sex] / Census_national_2025[age, sex]
```

**Year-specific projection:**
```
ND_survival[age, sex, year] = Census_NP2023[age, sex, year] * ND_adjustment[age, sex]
ND_survival[age, sex, year] = clip(ND_survival[age, sex, year], 0.0, 1.0)
```

This produces a full matrix of survival rates for each (year, age, sex) triple. The Census Bureau's own improvement trajectory (which is nonlinear and age-specific) is embedded in the NP2023 survival ratios, meaning the improvement is not a simple constant percentage but reflects actuarial modeling.

### 4.6 Application in SDC 5-Year Engine vs. 2026 Annual Engine

**SDC 5-year step (for regular ages):**
```
Pop[a+5, t+5] = Pop[a, t] * S_5yr(a, sex)
```

**2026 annual step (for regular ages):**
```
Pop[a+1, t+1] = Pop[a, t] * S(a, sex, race, t)
```

Over a 5-year span, the 2026 engine applies 5 separate annual survival steps, each potentially using a different (improved) survival rate for that year:

```
Pop[a+5, t+5] = Pop[a, t] * S(a, sex, race, t) * S(a+1, sex, race, t+1) * ... * S(a+4, sex, race, t+4)
```

This is not equivalent to the SDC's single multiplication by `S_5yr` because:
1. Each year's survival rate may differ (due to mortality improvement)
2. The cohort ages through single-year ages, each with its own rate (rather than a uniform 5-year group rate)

---

## 5. Rationale for Changes

### 5.1 Mortality Improvement vs. Constant Rates

The SDC holds survival rates constant over the entire 30-year projection horizon (2020-2050). This is the simplest assumption but is inconsistent with long-term mortality trends in the United States.

**Historical evidence:** U.S. age-adjusted mortality has declined at roughly 0.5-1.0% per year over the past several decades (pre-COVID), corresponding to gains of approximately 1.5-2.5 years of life expectancy per decade. The COVID-19 pandemic temporarily reversed this trend, but mortality rates have since returned to near pre-pandemic trajectories.

**Impact of constant-rate assumption:** Over a 30-year projection, holding mortality constant progressively overstates deaths. By 2050, a constant-rate projection may overcount deaths by 10-15% relative to a projection that incorporates a 0.5% annual improvement. This translates to:
- An undercount of the 65+ population (where mortality is concentrated)
- Underestimation of the old-age dependency ratio
- Underestimation of healthcare and social service demand

The 0.5% default was chosen as a moderate assumption consistent with Census Bureau, SSA, and state demographer practices. It can be set to 0% for sensitivity analysis or comparison with SDC.

### 5.2 Updated Life Table Vintage (2023 vs. 2020)

The SDC used ND-specific 2020 life tables. The 2020 reference year is problematic because:

1. **COVID-19 distortion:** 2020 mortality was elevated due to the pandemic, making these life tables unrepresentative of longer-term mortality conditions.
2. **Small-state variance:** North Dakota-specific life tables are based on a relatively small population (~780,000), introducing more sampling variability than national life tables.

The 2026 system uses 2023 CDC national life tables, adjusted to ND using a ratio method. This captures:
- Post-COVID mortality recovery (2023 rates are closer to "normal" than 2020)
- Greater statistical stability from national data
- ND-specific deviations preserved through the adjustment factor

### 5.3 Single-Year Ages vs. 5-Year Groups

The 2026 system uses single-year ages (0 through 90+) rather than 5-year groups. This provides several advantages:

1. **No within-group averaging:** In a 5-year group, all ages share the same survival rate, which underrepresents mortality variation (e.g., age 80 and age 84 have very different mortality).
2. **Annual projection steps:** Single-year ages naturally pair with annual projection intervals, avoiding the approximation of converting 5-year survival to annual equivalents.
3. **Cohort tracking:** Individual birth cohorts can be tracked year by year rather than in 5-year blocks, enabling more precise analysis of age structure evolution.

### 5.4 Race-Specific Mortality

The SDC uses a single set of survival rates for all race/ethnicity groups. The 2026 system differentiates survival by 6 categories. This is important because:

- Life expectancy at birth varies substantially by race: e.g., AIAN males ~70-75 years vs. Asian/PI females ~83-87 years
- North Dakota has a significant AIAN population (~5.3%) with notably different mortality patterns
- Race-specific rates improve projection accuracy for diverse populations

### 5.5 Terminal Age Group (90+ vs. 85+)

The 2026 system extends the terminal age group from 85+ to 90+. This means ages 85, 86, 87, 88, and 89 each have their own age-specific survival rates rather than being pooled. Given that the 85+ population is the fastest-growing demographic segment in many projections, this additional detail is valuable for:

- Healthcare planning (care needs differ substantially between ages 85 and 95)
- Estimating nursing home and assisted living demand
- Social Security and pension modeling

---

## 6. Source File Reference

| File | Role |
|------|------|
| `../sdc_2024_replication/METHODOLOGY_SPEC.md` | SDC technical specification (Section 3.2: Survival Rates) |
| `../sdc_2024_replication/data/survival_rates_by_county.csv` | Extracted SDC 5-year survival rates (state-level, by sex) |
| `../sdc_2024_replication/scripts/extract_sdc_county_data.py` | Script that extracted SDC survival data from Excel workbook |
| `../sdc_2024_replication/scripts/projection_engine.py` | SDC replication engine showing 85+ handling (lines 253-318) |
| `../sdc_2024_replication/scripts/prepare_updated_data.py` | Updated data preparation showing 5yr-to-1yr conversion |
| `data/processed/sdc_2024/survival_rates_sdc_2024_by_age_group.csv` | SDC rates with both 5yr and 1yr columns |
| `data/processed/sdc_2024/survival_rates_sdc_2024.csv` | SDC rates expanded to single-year ages |
| `docs/governance/adrs/002-survival-rate-processing.md` | ADR for survival rate methodology |
| `cohort_projections/data/process/survival_rates.py` | Static survival rate processor (lx/qx/Lx methods) |
| `cohort_projections/data/process/mortality_improvement.py` | Time-varying mortality improvement pipeline (Census NP2023) |
| `cohort_projections/core/mortality.py` | Runtime survival application and Lee-Carter improvement |
| `cohort_projections/core/cohort_component.py` | Projection engine orchestrator |
| `config/projection_config.yaml` | Configuration: `rates.mortality` section |
