# North Dakota Cohort-Component Population Projection Model: Methodology

**Version:** 1.0
**Date:** February 2026
**Base Year:** 2025 (Census PEP Vintage 2025)
**Projection Horizon:** 2025--2055 (30 years)

---

## Table of Contents

- [1. Overview and Introduction](#1-overview-and-introduction)
  - [1.1 The Cohort-Component Method](#11-the-cohort-component-method)
  - [1.2 Geographic Scope](#12-geographic-scope)
  - [1.3 Projection Horizon and Time Steps](#13-projection-horizon-and-time-steps)
  - [1.4 Demographic Dimensions](#14-demographic-dimensions)
  - [1.5 Scenarios](#15-scenarios)
- [2. Base Population](#2-base-population)
  - [2.1 Data Source](#21-data-source)
  - [2.2 Statewide Age-Sex-Race Distribution](#22-statewide-age-sex-race-distribution-adr-048)
  - [2.3 Sprague Osculatory Interpolation](#23-sprague-osculatory-interpolation-adr-048)
  - [2.4 County-Specific Race Distributions](#24-county-specific-race-distributions-adr-047)
  - [2.5 Group Quarters Separation](#25-group-quarters-separation-adr-055)
  - [2.6 Summary: Base Population Assembly Pipeline](#26-summary-base-population-assembly-pipeline)
- [3. Fertility Component](#3-fertility-component)
  - [3.1 Data Source and Processing](#31-data-source-and-processing)
  - [3.2 North Dakota Adjustment](#32-north-dakota-adjustment-adr-053)
  - [3.3 Birth Calculation](#33-birth-calculation)
  - [3.4 Total Fertility Rate](#34-total-fertility-rate)
  - [3.5 Scenario Adjustments](#35-scenario-adjustments)
- [4. Mortality Component](#4-mortality-component)
  - [4.1 Survival Rate Derivation from Life Tables](#41-survival-rate-derivation-from-life-tables)
  - [4.2 North Dakota Adjustment](#42-north-dakota-adjustment-adr-053)
  - [4.3 The 90+ Open-Ended Age Group](#43-the-90-open-ended-age-group)
  - [4.4 Applying Survival Rates](#44-applying-survival-rates)
  - [4.5 Mortality Improvement](#45-mortality-improvement)
- [5. Migration Component](#5-migration-component)
  - [5a. Residual Migration Method](#5a-residual-migration-method)
  - [5b. GQ-Corrected Migration Rates](#5b-gq-corrected-migration-rates-adr-055-phase-2)
  - [5c. Oil-Boom Dampening](#5c-oil-boom-dampening-adr-040-adr-051)
  - [5d. Male Migration Dampening](#5d-male-migration-dampening)
  - [5e. PEP Recalibration for Reservation Counties](#5e-pep-recalibration-for-reservation-counties-adr-045)
  - [5f. College-Age Smoothing](#5f-college-age-smoothing-adr-049)
  - [5g. Multi-Period Averaging](#5g-multi-period-averaging)
  - [5h. Convergence Interpolation](#5h-convergence-interpolation-5-10-5-schedule)
  - [5i. Age-Aware Rate Cap](#5i-age-aware-rate-cap-adr-043)
  - [5j. BEBR High-Growth Increment](#5j-bebr-high-growth-increment-adr-046)
  - [5k. Ward County Migration Floor](#5k-ward-county-migration-floor-adr-052)
- [6. Scenario Methodology](#6-scenario-methodology)
  - [6.1 Baseline Scenario](#61-baseline-scenario)
  - [6.2 Restricted Growth Scenario](#62-restricted-growth-scenario)
  - [6.3 High Growth Scenario](#63-high-growth-scenario)
  - [6.4 Scenario Comparison Summary](#64-scenario-comparison-summary)
- [7. Projection Engine](#7-projection-engine)
  - [7.1 Annual Projection Step](#71-annual-projection-step)
  - [7.2 Multi-Year Execution](#72-multi-year-execution)
  - [7.3 Group Quarters Re-Addition](#73-group-quarters-re-addition-adr-055)
  - [7.4 State Aggregation](#74-state-aggregation-adr-054)
  - [7.5 Place Projection Orchestration](#75-place-projection-orchestration-pp-003-phase-2)
- [8. Data Sources and References](#8-data-sources-and-references)
  - [8.1 Primary Data Sources](#81-primary-data-sources)
  - [8.2 Methodological References](#82-methodological-references)
  - [8.3 Architecture Decision Records](#83-architecture-decision-records)

---

## 1. Overview and Introduction

### 1.1 The Cohort-Component Method

This document describes the methodology underlying the North Dakota Cohort-Component Population Projection Model, which produces county-level population projections for the state of North Dakota from 2025 through 2055. The model follows the **cohort-component method**, the standard framework used by the U.S. Census Bureau, state demography offices, and the United Nations Population Division for subnational population projections.

The cohort-component method tracks population by disaggregating it into demographic cohorts defined by age, sex, and race/ethnicity. At each annual time step, three demographic processes are applied to every cohort:

1. **Survival** -- members of each cohort are aged forward one year and subjected to age-, sex-, and race-specific mortality.
2. **Fertility** -- women of reproductive age produce births, which enter the population as the age-0 cohort.
3. **Migration** -- net migration (the balance of in-migration and out-migration) is applied to each cohort.

The population at the end of each projection year becomes the starting population for the next year. Formally, for a cohort defined by age $a$, sex $s$, and race $r$ in county $c$:

$$P_{c}(a+1, s, r, t+1) = P_{c}(a, s, r, t) \times S(a, s, r) + M_{c}(a, s, r, t)$$

where $P$ is population, $S$ is the survival rate, and $M$ is net migration. Births enter as:

$$P_{c}(0, s, r, t+1) = \sum_{a=15}^{49} F(a, r) \times P_{c,\text{female}}(a, r, t)$$

where $F(a, r)$ is the age-specific fertility rate for race $r$, and the newborns are split between male and female sexes using a biological sex ratio.

### 1.2 Geographic Scope

The model covers all **53 counties** in North Dakota. The state-level projection is derived as the arithmetic sum of county-level projections rather than as an independent projection (ADR-054). This bottom-up aggregation eliminates the divergence that arises when state and county projections are run independently with different migration rate compositions. Independent state and county projections can diverge by more than 10% over a 30-year horizon due to Jensen's inequality -- the compounding effect of heterogeneous migration rates applied to independently evolving county populations. By defining the state total as:

$$P_{\text{state}}(t) = \sum_{c=1}^{53} P_{c}(t)$$

the model guarantees exact consistency between county and state totals by construction.

The geographic unit for all rate computations and projections is the county, identified by 5-digit FIPS code (state FIPS 38 concatenated with the 3-digit county code). Geographic definitions follow the Census Bureau's 2020 vintage boundaries.

### 1.3 Projection Horizon and Time Steps

| Parameter | Value |
|-----------|-------|
| Base year | 2025 |
| Projection horizon | 30 years |
| Terminal year | 2055 |
| Time step | 1 year (annual) |
| Base population | Census PEP Vintage 2025 |

The model advances in annual increments. This annual resolution, combined with single-year-of-age tracking, allows the model to capture cohort dynamics with greater precision than the quinquennial (5-year) approach used by some state demography offices.

### 1.4 Demographic Dimensions

Each county's population is disaggregated along three dimensions:

**Age.** The model tracks 91 single-year age groups: ages 0 through 89 as closed intervals, and an open-ended terminal group $90+$. This yields age indices $a \in \{0, 1, 2, \ldots, 90\}$ where age 90 represents all persons aged 90 and older.

**Sex.** Two categories: Male and Female.

**Race/ethnicity.** Six mutually exclusive and exhaustive categories, following Census Bureau and SEER conventions:

1. White alone, Non-Hispanic
2. Black alone, Non-Hispanic
3. American Indian/Alaska Native (AIAN) alone, Non-Hispanic
4. Asian/Pacific Islander (Asian/PI) alone, Non-Hispanic
5. Two or more races, Non-Hispanic
6. Hispanic (any race)

The total number of cohorts per county is $91 \times 2 \times 6 = 1{,}092$.

### 1.5 Scenarios

The model produces three active projection scenarios that share the same base population and projection engine but differ in their assumptions about future demographic rates:

**Baseline (Trend Continuation).** Historical PEP-derived migration trends are continued forward using convergence interpolation. Fertility rates are held constant at observed levels. Mortality improves at 0.5% per year. This scenario represents the most likely trajectory if recent demographic patterns persist.

**Restricted Growth (CBO Policy-Adjusted).** Models the demographic impact of reduced immigration using Congressional Budget Office (CBO) estimates of federal immigration enforcement effects (ADR-037, ADR-050). Migration is reduced via an additive per-capita decrement derived from the state's international migration volume, phased in over 2025--2029. Fertility is reduced by 5%.

**High Growth (Elevated Immigration).** A counterfactual scenario representing sustained elevated immigration, using BEBR-optimistic convergence rates (ADR-046). Fertility is increased by 5%. A migration floor (ADR-052) ensures no county's mean convergence rate is negative, reflecting the premise that institutional anchors (military bases, universities, regional medical centers) prevent population decline in a genuinely high-growth environment.

Two additional scenarios (Zero Net Migration and SDC 2024 Methodology Replication) are defined in the configuration but are not active in production runs.


## 2. Base Population

### 2.1 Data Source

The base population for all projections is anchored to **Census Bureau Population Estimates Program (PEP) Vintage 2025** data, specifically the SC-EST2024-ALLDATA6 file, which provides state-level population by single year of age, sex, and race/ethnicity. County total populations are drawn from the PEP Vintage 2025 estimates (column `population_2025` in the county population reference file).

The base year is July 1, 2025, with a total North Dakota population of **799,358**.

The base population construction proceeds in four stages:

1. Build the statewide age $\times$ sex $\times$ race distribution from SC-EST single-year data.
2. Build county-specific distributions using Census cc-est2024-alldata, with Sprague interpolation to single-year ages.
3. Apply each county's distribution to its PEP total population to produce the 1,092-cell cohort matrix.
4. Separate group quarters (GQ) population from total population, retaining only household population for projection.

### 2.2 Statewide Age-Sex-Race Distribution (ADR-048)

The statewide distribution is derived directly from the SC-EST2024-ALLDATA6 file, which provides Census Bureau estimates of population by single year of age (0 through 85+), sex, and race/ethnicity at the state level. This file is filtered to North Dakota (STATE = 38) and the most recent estimate year.

The SC-EST data provides single-year-of-age detail through age 85, after which a terminal $85+$ group is reported. The model requires ages 0 through 90 (with 90 representing $90+$). The terminal group is expanded to single years 85--90 using an exponential survival tail:

$$w_{90+} = \frac{s^5}{1 - s}, \qquad \text{where } s = 0.7$$

That is, the fraction of the $85+$ group assigned to the $90+$ age is:

$$f_{90+} = \frac{w_{90+}}{w_{85} + w_{86} + w_{87} + w_{88} + w_{89} + w_{90+}} = \frac{s^5 / (1-s)}{\sum_{i=0}^{4} s^i + s^5/(1-s)}$$

where the weight for each single year within 85--89 follows a geometric survival pattern $s^i$ for $i = 0, 1, \ldots, 4$. The ages 85--89 values from the Sprague output (see Section 2.3) are rescaled to sum to the $85+$ total minus the $90+$ allocation.

Raw Census race codes (including separate NHPI and Asian categories) are mapped to the model's six race/ethnicity categories. NHPI (Native Hawaiian/Pacific Islander) is combined with Asian into the "Asian/PI alone, Non-Hispanic" category. Proportions are normalized to sum to 1.0 across all 1,092 cells.

### 2.3 Sprague Osculatory Interpolation (ADR-048)

County-level population data from the Census Bureau (cc-est2024-alldata) is available only in quinquennial (5-year) age groups: 0--4, 5--9, ..., 80--84, 85+. To produce the single-year-of-age resolution required by the projection engine, the model applies **Sprague osculatory interpolation**, the standard method used by the UN Population Division and the U.S. Census Bureau for graduating grouped data to single years.

#### 2.3.1 The Sprague Coefficient Matrix

Sprague interpolation uses a fixed $5 \times 5$ matrix of multipliers, $\mathbf{M}$, that operates on a sliding window of five consecutive age-group totals. For a target group $i$ with group total $Y_i$, the five single-year values within that group are obtained from:

$$\mathbf{v}_i = \mathbf{M} \cdot \begin{pmatrix} Y_{i-2} \\ Y_{i-1} \\ Y_i \\ Y_{i+1} \\ Y_{i+2} \end{pmatrix}$$

where $\mathbf{v}_i$ is the vector of five single-year values and the Sprague multiplier matrix is:

$$\mathbf{M} = \begin{pmatrix} -0.0128 & 0.0848 & 0.1504 & -0.0240 & 0.0016 \\ -0.0016 & 0.0144 & 0.2224 & -0.0416 & 0.0064 \\ 0.0064 & -0.0336 & 0.2544 & -0.0336 & 0.0064 \\ 0.0064 & -0.0416 & 0.2224 & 0.0144 & -0.0016 \\ 0.0016 & -0.0240 & 0.1504 & 0.0848 & -0.0128 \end{pmatrix}$$

A critical property of this matrix is that its column sums are $(0, 0, 1, 0, 0)$, which guarantees that the five interpolated single-year values sum exactly to the center group's total $Y_i$. This group-total preservation property ensures that the graduated population is consistent with the original grouped data.

In plain language: each single-year value within a 5-year age group is computed as a weighted combination of that group's total and the totals of the two groups on either side. The weights (the rows of $\mathbf{M}$) are derived from osculatory polynomial fitting, which ensures smooth transitions across group boundaries while preserving totals.

#### 2.3.2 Boundary Padding

The Sprague method requires two neighboring groups on each side of the target group. For the first two and last two age groups in the schedule, these neighbors do not exist in the data. The model addresses this by **linearly extrapolating** two virtual groups on each end:

**Left boundary:**

$$\tilde{Y}_{-1} = 2 Y_0 - Y_1, \qquad \tilde{Y}_{-2} = 2 \tilde{Y}_{-1} - Y_0$$

**Right boundary (for $n$ groups):**

$$\tilde{Y}_{n} = 2 Y_{n-1} - Y_{n-2}, \qquad \tilde{Y}_{n+1} = 2 \tilde{Y}_{n} - Y_{n-1}$$

This padding strategy follows the approach used by the DemoTools R package and the UN Population Division. The padded array has length $n + 4$, and every original group can then sit at the center of a 5-group window.

#### 2.3.3 Negative Clamping and Renormalization

For very small populations (such as minority race groups in rural counties), the Sprague interpolation can produce small negative values at extreme ages. These are artifacts of the polynomial fitting and are not demographically meaningful. The model clamps all negative values to zero and renormalizes within each 5-year group:

$$v_j^* = \max(v_j, 0), \qquad v_j^{**} = v_j^* \times \frac{Y_i}{\sum_{k=1}^{5} v_k^*}$$

If all five values within a group are negative or zero (possible only for groups with near-zero population), the group total is distributed uniformly: $v_j = Y_i / 5$.

This ensures that (a) no cohort has a negative population, and (b) the sum of single-year values still equals the original group total.

#### 2.3.4 Application to County Distributions

For each county, the Sprague graduation is applied independently to each sex $\times$ race combination. The 18 five-year group proportions (0--4 through 85+) for a given sex-race pair are assembled into a vector, padded, graduated to 90 single-year values (ages 0--89), and then the terminal group is handled as described in Section 2.2 to produce ages 85--90. The result is a 91-element vector of proportions for that sex-race pair, which, when assembled across all 12 sex-race combinations and applied to the county's total population, produces the complete 1,092-cell cohort matrix.

### 2.4 County-Specific Race Distributions (ADR-047)

Prior to ADR-047, all 53 counties shared an identical statewide age-sex-race distribution, meaning a county like Sioux County (78% AIAN) was modeled with the statewide AIAN share of approximately 4.8%. This produced misallocation errors exceeding 76% of the total population in reservation counties.

The model now constructs **county-specific distributions** from the Census cc-est2024-alldata file, which provides county-level population by 5-year age group, sex, and race/ethnicity. Each county's distribution is expressed as a set of proportions across all age-group $\times$ sex $\times$ race cells, normalized to sum to 1.0.

#### 2.4.1 Population-Weighted Blending for Small Counties

Counties with very small populations have sparse distributions with many zero cells, which can introduce instability in the projection. To mitigate this, counties below a population threshold of 5,000 use a **blended distribution** that combines the county-specific pattern with the statewide distribution:

$$d_{\text{blended}} = \alpha \cdot d_{\text{county}} + (1 - \alpha) \cdot d_{\text{state}}$$

where the blending weight $\alpha$ is:

$$\alpha = \min\!\left(\frac{P_{\text{county}}}{5{,}000},\; 1.0\right)$$

Counties at or above the 5,000 threshold use their own distribution exclusively ($\alpha = 1$). Below the threshold, the weight decreases linearly with population, so a county with 2,500 people uses a 50/50 blend of its own distribution and the statewide distribution. The threshold of 5,000 was chosen because counties below this size have more than 30% zero cells in their race-age distributions, while counties above have fewer than 15%.

This blending preserves the distinctive demographic character of larger counties (including the high AIAN shares in reservation counties like Sioux, Rolette, and Benson) while stabilizing the distributions of the smallest rural counties where individual cell values may be based on fewer than five persons.

### 2.5 Group Quarters Separation (ADR-055)

#### 2.5.1 Rationale

Group quarters (GQ) populations -- military barracks, college dormitories, nursing facilities, and correctional institutions -- undergo institutional rotation cycles that are driven by enrollment schedules, military assignment orders, and facility capacity rather than by the county-level demographic forces that the cohort-component model is designed to project. When GQ residents are included in the total population, their turnover inflates apparent migration rates and, over a 30-year projection, leads to the gradual erosion of institutional populations that are in reality structurally stable.

For example, when a UND freshman replaces a graduating senior in the same dormitory bed, the residual migration method counts an in-migrant at age 18 and an out-migrant at age 25. Both are real population movements in Census accounting, but neither represents a change in the county's settled population. Similar patterns arise from military Permanent Change of Station (PCS) transfers.

The standard practice in state demography offices (North Carolina OSBM, New Hampshire OPD, New York City DCP, among others) is to separate GQ from household population before computing rates and projecting.

#### 2.5.2 GQ Data Source and Age-Sex Disaggregation

GQ population counts are drawn from the **Census PEP Vintage 2025 stcoreview** file, which reports GQ population (the `GQpop` variable) by county and three broad age groups: 0--17, 18--64, and 65+.

Because stcoreview does not provide age detail finer than these broad groups, the model disaggregates to 5-year age groups using **allocation profiles** that reflect the institutional composition of each county's GQ population:

**Ages 0--17:** Distributed uniformly across 0--4, 5--9, 10--14, and 15--17 (GQ population under 18 consists mainly of juvenile facilities and is small and evenly spread).

**Ages 18--64 (college counties -- Grand Forks, Cass, Ward, Burleigh):** Concentrated in 18--19, 20--24, and 25--29, reflecting the dominance of university dormitories in these counties' GQ populations. The relative weights are 3.0, 5.0, and 2.0 for ages 18--19, 20--24, and 25--29 respectively, tapering to 0.2 for ages 55--64.

**Ages 18--64 (other counties):** More evenly spread across 18--19 through 40--44, reflecting the mix of correctional facilities, group homes, and smaller institutional settings. Relative weights range from 2.0 for ages 20--34 down to 0.4 for ages 60--64.

**Ages 65+:** Concentrated in older ages, reflecting nursing facility populations. Relative weights increase from 0.5 at ages 65--69 to 4.0 at ages 85+.

Within each 5-year age group, GQ is split evenly between Male and Female. While individual institution types skew by sex (military barracks skew male; nursing facilities skew female), these effects partially offset at the county level. The 50/50 split is the least-biased assumption in the absence of county-level sex-specific GQ data.

#### 2.5.3 Race Distribution of GQ

County-level GQ data by race is not available. The model distributes GQ across race categories in proportion to each county's overall population race shares:

$$\text{GQ}(a, s, r, c) = \text{GQ}(a, s, c) \times \frac{P_c(r)}{\sum_{r'} P_c(r')}$$

where $P_c(r)$ is the total population of race $r$ in county $c$. This assumes that the racial composition of GQ residents mirrors the county's overall composition -- a simplification, but one that avoids introducing artificial racial detail that the source data cannot support.

#### 2.5.4 Separation and Hold-Constant Procedure

At the base year, GQ population is subtracted from total population to yield household-only population:

$$P_{\text{HH}}(a, s, r, c, t_0) = P_{\text{total}}(a, s, r, c, t_0) - \text{GQ}(a, s, r, c)$$

GQ is capped at the total population in each cell to prevent negative household populations. The projection engine then operates exclusively on $P_{\text{HH}}$. At each projection year $t$, the GQ population is added back as a constant:

$$P_{\text{total}}(a, s, r, c, t) = P_{\text{HH,projected}}(a, s, r, c, t) + \text{GQ}(a, s, r, c)$$

The hold-constant assumption means that GQ is fixed at 2025 levels throughout the projection horizon. Changes in institutional capacity (base closures, dormitory construction, nursing facility expansion) are policy decisions that fall outside the scope of a demographic projection model. If specific institutional changes are anticipated, they can be incorporated as scenario adjustments.

The GQ populations for the four largest institutional counties at the base year (2025 PEP) are:

| County | GQ Population | GQ as % of Total | Primary Institutions |
|--------|:------------:|:-----------------:|---------------------|
| Cass (38017) | 8,684 | 4.3% | NDSU dormitories, nursing facilities |
| Burleigh (38015) | 5,274 | 5.1% | University of Mary/BSC, correctional, nursing |
| Grand Forks (38035) | 4,512 | 6.1% | UND dormitories, GFAFB barracks, nursing |
| Ward (38101) | 2,161 | 3.2% | MAFB barracks, Minot State dormitories, nursing |

#### 2.5.5 GQ-Corrected Migration Rates (Phase 2)

In addition to separating GQ from the base population, GQ is also subtracted from the historical population snapshots used to compute residual migration rates (see Section 5). This ensures that migration rates are computed on household-only population, removing institutional rotation signals from the rates themselves. Historical GQ is estimated at six time points (2000, 2005, 2010, 2015, 2020, 2024): stcoreview provides GQ data for 2020 and 2024 directly, while years 2000 through 2015 use 2020 GQ levels as a backward constant. The backward-constant assumption is defensible because institutional capacity (barracks, dormitories, nursing beds) changes slowly over 5- to 10-year windows. The detailed methodology for GQ-corrected migration rates is described in Section 5.

### 2.6 Summary: Base Population Assembly Pipeline

The complete base population assembly for a single county proceeds as follows:

1. **Load county total population** from PEP Vintage 2025 ($P_{\text{total},c}$).
2. **Load county-specific distribution** from the pre-built county distributions file, or fall back to the statewide distribution if county-specific data is unavailable.
3. **Apply Sprague interpolation** to graduate the county's 5-year age-group proportions to single-year ages (18 groups $\rightarrow$ 91 single-year ages).
4. **Multiply proportions by total population** to produce population counts: $P_c(a, s, r) = d_c(a, s, r) \times P_{\text{total},c}$.
5. **Separate GQ**: subtract GQ from total to yield household population; store GQ for later re-addition.
6. **Output**: a DataFrame with columns `[year, age, sex, race, population]` containing 1,092 rows of household-only population, ready for the projection engine.

This pipeline is repeated for all 53 counties. The state total is the sum of county totals (ADR-054).


## 3. Fertility Component

The fertility component generates the youngest cohort in each projection year by applying age-specific fertility rates to the female population of reproductive age, producing births that enter the population as age-0 individuals.

### 3.1 Data Source and Processing

Age-specific fertility rates (ASFR) are derived from CDC WONDER Natality data for North Dakota, pooled over 2020--2023 (four years) to smooth annual volatility in small race-by-age cells (ADR-053, Part A). The raw birth counts are cross-classified by mother's five-year age group (15--19, 20--24, ..., 45--49) and by mother's single race in six categories following the OMB standard: White non-Hispanic, Black non-Hispanic, American Indian/Alaska Native non-Hispanic, Asian/Pacific Islander non-Hispanic, Two or More Races non-Hispanic, and Hispanic of any race. Hispanic ethnicity takes priority over race in the classification scheme --- a Hispanic White mother is categorized as Hispanic, not White non-Hispanic --- consistent with the project's six-category race/ethnicity schema.

The denominator for each ASFR cell is the corresponding female population from the Census Bureau's County Characteristics file (cc-est2024-alldata), summed across all 53 North Dakota counties and matched to the same four-year window (July 2020 through July 2023). The resulting ASFR is expressed as births per 1,000 women per year:

$$
\text{ASFR}(a, r) = \frac{\text{ND births}(a, r, 2020\text{--}2023)}{\sum_{t=2020}^{2023} \text{ND females}(a, r, t)} \times 1{,}000
$$

where $a$ indexes the five-year age group and $r$ indexes the race/ethnicity category.

For cells with fewer than 10 births across the four-year pooling window --- the CDC suppression threshold --- national ASFR values serve as a fallback. In practice, five of 49 cells required this fallback, primarily at the tails of the reproductive age range (Asian non-Hispanic ages 15--19, and several race groups at ages 45--49).

The resulting rate table is a complete $7 \times 7$ grid (seven age groups by seven race/ethnicity categories, including a "total" row). The downstream pipeline expands five-year age groups to single-year-of-age rates using within-group weight interpolation (consistent with ADR-048).

### 3.2 North Dakota Adjustment (ADR-053)

National fertility rates substantially understate North Dakota's fertility. The national Total Fertility Rate (TFR) in 2023 was approximately 1.621, while North Dakota's TFR was 1.863 --- a premium of roughly 15 percent. Using national rates without adjustment would systematically undercount approximately 1,400--1,600 births per year, an error that compounds over the 30-year projection horizon.

ADR-053 adopted ND-specific ASFR computed directly from CDC WONDER birth counts rather than applying a uniform scalar to national rates. A uniform scalar would assume the ND fertility premium is equal across all ages and races, whereas in reality the premium varies: it is larger for ages 25--34 and smaller for teenagers; it is substantial for AIAN (ND TFR 2.298 vs. national 1.422, ratio 1.62) and modest for White non-Hispanic (ND 1.748 vs. national 1.550, ratio 1.13). The cell-specific approach preserves these differentials.

Validation against independently reported ND vital statistics confirms the calibration:

- Computed ND TFR: 1.863, within 2 percent of the target range of 1.85--1.90.
- Average annual births implied by the rates: 9,804, within 1.6 percent of the 9,647 births actually recorded in 2023.

### 3.3 Birth Calculation

In each projection year $t$, births are calculated by applying the ASFR to the female population in reproductive ages (15--49). For a given race/ethnicity group $r$, total births are:

$$
B(r, t) = \sum_{a=15}^{49} F(a, r) \times P_f(a, r, t)
$$

where $F(a, r)$ is the age-specific fertility rate (expressed as births per woman, not per 1,000) for single year of age $a$ and race $r$, and $P_f(a, r, t)$ is the female population at age $a$, race $r$, at time $t$. Rates are merged to the female population by age and race; any age-race cell lacking a fertility rate is assigned a rate of zero.

Births are then split by sex using the standard sex ratio at birth. The proportion male at birth is configured as 0.51, corresponding to the well-established biological sex ratio of approximately 105 males per 100 females:

$$
B_{\text{male}}(r, t) = B(r, t) \times 0.51
$$

$$
B_{\text{female}}(r, t) = B(r, t) \times 0.49
$$

All newborns are assigned age 0. The mother's race/ethnicity is assigned to the child, a simplifying assumption that follows standard cohort-component practice.

### 3.4 Total Fertility Rate

The Total Fertility Rate (TFR) is a summary measure representing the average number of children a woman would bear over her lifetime if she experienced the current age-specific rates at each age. For single-year-of-age rates, TFR is simply the sum of the ASFR across the reproductive span:

$$
\text{TFR}(r) = \sum_{a=15}^{49} F(a, r)
$$

For five-year age group rates (as in the input data), TFR is computed by multiplying the sum of the group-specific rates by the group width:

$$
\text{TFR}(r) = 5 \times \sum_{g} \text{ASFR}_g(r) / 1{,}000
$$

where $g$ indexes the five-year age groups and the division by 1,000 converts from per-thousand to per-woman units. TFR is calculated for each race/ethnicity group and serves as the primary validation metric for the fertility component.

### 3.5 Scenario Adjustments

The projection system supports multiple fertility scenarios, each implemented as a multiplicative adjustment to the base ASFR. The scenario is applied uniformly to all age-race cells. Available scenarios are:

| Scenario | Adjustment | Application |
|----------|-----------|-------------|
| `constant` (baseline) | No change | Rates held at base-year levels throughout the projection |
| `+5_percent` (high growth) | $F'(a,r) = F(a,r) \times 1.05$ | 5 percent increase in all rates |
| `-5_percent` (restricted) | $F'(a,r) = F(a,r) \times 0.95$ | 5 percent decrease in all rates |
| `trending` | $F'(a,r,t) = F(a,r) \times (1 - 0.005)^{t - t_0}$ | 0.5 percent annual decline, compounding from base year $t_0$ |

The trending scenario reflects the long-run fertility decline observed across developed countries. After $n$ years, the adjustment factor is $(0.995)^n$; after 30 years, fertility rates are approximately 86 percent of their base-year values. In all scenarios, rates are clipped to be non-negative after adjustment.


## 4. Mortality Component

The mortality component ages the population forward by one year and accounts for deaths by applying age-, sex-, and race-specific survival rates derived from life tables. Survival rates represent the probability that an individual alive at age $a$ will survive to age $a+1$.

### 4.1 Survival Rate Derivation from Life Tables

Survival rates are computed from period life tables published by the CDC (NVSR 74-06 for national race-specific tables, NVSR 74-12 for North Dakota state tables). The primary calculation method uses the $l_x$ (survivors) column of the life table:

$$
S(a, s, r) = \frac{l_{x+1}(s, r)}{l_x(s, r)}
$$

where $l_x$ is the number of survivors to exact age $x$ out of a radix of 100,000, $s$ indexes sex, and $r$ indexes race/ethnicity. This ratio gives the proportion of the population at age $a$ that survives to age $a+1$.

An alternative method uses the $q_x$ (probability of dying) column directly:

$$
S(a, s, r) = 1 - q_x(a, s, r)
$$

where $q_x$ is the probability of dying between ages $x$ and $x+1$. The two methods are algebraically equivalent when applied to the same life table. The system selects the $l_x$ method when the $l_x$ column is available; it falls back to the $q_x$ method otherwise.

In either case, survival rates are bounded to the interval $[0, 1]$.

### 4.2 North Dakota Adjustment (ADR-053)

National life tables provide survival rates by age, sex, and race but not by state. North Dakota state life tables (NVSR 74-12, 2022 data year) provide survival by age and sex but not by race. A hybrid approach combines the two sources to produce ND-calibrated, race-specific survival rates.

The adjustment proceeds in two steps. First, an ND/national mortality ratio is computed for each age and sex using the probability-of-dying ($q_x$) values:

$$
R(a, s) = \frac{q_x^{\text{ND}}(a, s)}{q_x^{\text{US}}(a, s)}
$$

This ratio captures how North Dakota's mortality at each age deviates from the national level. The ratio is capped to the interval $[0.5, 2.0]$ to prevent extreme adjustments at ages where small ND death counts produce volatile $q_x$ estimates.

Second, the ratio is applied to the national race-specific life tables:

$$
q_x^{\text{ND}}(a, s, r) = q_x^{\text{US}}(a, s, r) \times R(a, s)
$$

$$
S^{\text{ND}}(a, s, r) = 1 - q_x^{\text{ND}}(a, s, r)
$$

This method preserves the race-specific mortality differentials from the national data (e.g., the higher mortality among Black and AIAN populations relative to White and Asian populations) while shifting all rates to match North Dakota's overall mortality level.

Empirically, North Dakota has slightly lower mortality than the national average. The 2022 NVSR 74-12 state tables report ND life expectancy at birth of 77.93 years versus 77.46 nationally, a difference of +0.47 years. The mean ND/national $q_x$ ratio is approximately 0.937 for both sexes, indicating that ND death rates are roughly 6 percent below national at most ages. The ratio is dominated by the White majority (approximately 82 percent of the ND population), so the aggregate adjustment primarily captures White mortality patterns. A future enhancement would compute an AIAN-specific ratio from CDC WONDER death counts to better reflect the substantial AIAN mortality disadvantage in North Dakota.

### 4.3 The 90+ Open-Ended Age Group

The projection uses age 90 as the terminal open-ended age group. Because individuals in this group cannot "age out," a special survival rate formulation is required. When the life table columns $T_x$ (total person-years lived above age $x$) and $L_x$ (person-years lived in the interval) are available, the open-ended survival rate is:

$$
S(90+, s, r) = \frac{T_{91}(s, r)}{T_{90}(s, r) + L_{90}(s, r) / 2}
$$

where $T_{91} = T_{90} - L_{90}$ represents person-years lived above age 91. The denominator adjusts for the mid-year exposure convention. When these columns are not available, the system falls back to $S(90+) = 1 - q_x(90)$ or a default value of 0.65, which reflects typical survival in this age range.

In plain language, the 90+ survival rate captures the probability that the average member of the 90-and-older population survives another year. Typical values range from 0.60 to 0.70, reflecting the high mortality at advanced ages.

### 4.4 Applying Survival Rates

Each projection step advances the population by one year. The survival operation differs for regular ages and the open-ended group.

**Regular ages (0 through 89).** Each cohort at age $a$ is multiplied by its survival rate and advanced to age $a+1$:

$$
P(a+1, s, r, t+1) = P(a, s, r, t) \times S(a, s, r)
$$

where $P(a, s, r, t)$ is the population at age $a$, sex $s$, race $r$, and time $t$.

**Open-ended group (90+).** The 90+ population at time $t+1$ consists of two components: (1) survivors from the previous 90+ population, and (2) new entrants --- individuals who were 89 at time $t$ and survived to 90:

$$
P(90+, s, r, t+1) = P(89, s, r, t) \times S(89, s, r) + P(90+, s, r, t) \times S(90+, s, r)
$$

The first term ages the 89-year-old cohort into the open-ended group. The second term applies within-group survival to those already in the 90+ pool. This formulation ensures that the oldest population is neither lost nor artificially inflated.

Missing survival rates are handled defensively: for regular ages, a missing rate defaults to zero (equivalent to assuming all individuals die, triggering a warning); for the 90+ group, a missing rate defaults to 0.5.

### 4.5 Mortality Improvement

Over long projection horizons, assuming constant mortality understates life expectancy gains. The system applies a Lee-Carter-style exponential decline in age-specific death rates, calibrated at 0.5 percent per year (the `rates.mortality.improvement_factor` configuration parameter, $\delta = 0.005$).

The improved death rate at age $a$ in year $t$ is:

$$
q_x(a, t) = q_x(a, t_0) \times (1 - \delta)^{t - t_0}
$$

where $t_0$ is the base year of the life table and $\delta$ is the annual improvement factor. The corresponding survival rate is:

$$
S(a, t) = 1 - q_x(a, t)
$$

Survival rates are capped at 1.0 to prevent biologically implausible values. No improvement is applied when $t \leq t_0$ or when the improvement factor is configured to zero.

In plain terms, this means that for each year beyond the base year, the probability of dying at every age is reduced by 0.5 percent of its current value. The effect is compounding: after 10 years, death rates are approximately 95.1 percent of their base-year values; after 30 years, approximately 86.1 percent. This produces a gradual increase in life expectancy over the projection horizon, consistent with long-run trends in developed countries.

The 0.5 percent annual improvement rate is a moderate assumption. Historical U.S. experience has seen periods of both faster improvement (1.0--1.5 percent per year in the mid-20th century) and stagnation (near-zero improvement in certain years of the 2010s and during the COVID-19 pandemic). The chosen rate represents a consensus long-term average and is consistent with assumptions used by the Social Security Administration's intermediate projections.

The mortality improvement mechanism operates independently of the ND adjustment described in Section 4.2. The ND ratio calibrates the *level* of mortality to match North Dakota's current experience, while the improvement factor models *future trends* in mortality decline. The two adjustments are applied sequentially: ND-adjusted base rates are stored in the processed survival rate file, and the improvement factor is applied dynamically during each projection year.


## 5. Migration Component

Migration is the most volatile and methodologically challenging component of the cohort-component model. Unlike fertility and mortality, which are governed by relatively stable biological and behavioral patterns, migration responds to labor market shocks, institutional changes, and policy decisions that can reverse direction within a single intercensal period. North Dakota's migration history -- punctuated by the Bakken oil boom, military base realignments, and university enrollment cycles -- demands a rate-estimation pipeline with multiple layers of calibration and dampening.

This section describes the full migration rate pipeline, from the initial residual computation through convergence interpolation and scenario-specific adjustments. The pipeline processes rates at the county-by-age-group-by-sex level (53 counties, 18 five-year age groups, 2 sexes = 1,908 cells per period), applying a series of transformations that progressively refine raw residual estimates into projection-ready annual rates.

### 5a. Residual Migration Method

Net migration rates are estimated using the Census Bureau residual method, which infers migration as the difference between the observed population at the end of a period and the population that would have been expected from aging and survival alone. This approach captures all forms of migration -- domestic and international, voluntary and involuntary -- in a single net flow.

**Core formula.** For each county, sex, and five-year age group $a$, the residual migration computation proceeds in four steps:

$$\text{Expected}(a+5, t_1) = P(a, t_0) \times S_{5\text{yr}}(a, s)$$

$$\text{Migration}(a+5, t_1) = P(a+5, t_1) - \text{Expected}(a+5, t_1)$$

$$r_{\text{period}} = \frac{\text{Migration}(a+5, t_1)}{\text{Expected}(a+5, t_1)}$$

$$r_{\text{annual}} = (1 + r_{\text{period}})^{1/n} - 1$$

where $P(a, t_0)$ is the population in age group $a$ at the start of the period, $P(a+5, t_1)$ is the observed population in the next older age group at the end of the period, $S_{5\text{yr}}(a, s)$ is the five-year survival rate for age group $a$ and sex $s$, $r_{\text{period}}$ is the net migration rate over the full period, $n$ is the period length in years, and $r_{\text{annual}}$ is the compound-annualized rate.

In plain language: the model ages each cohort forward by one five-year step, applies the appropriate survival rate, and attributes the difference between this expected population and the observed population to net migration. Compound annualization converts the multi-year rate into the annual rate required by the projection engine.

**Time points and periods.** The pipeline uses six population snapshots drawn from Census Bureau data:

| Time Point | Source |
|:----------:|--------|
| 2000 | Census 2000 (ESTIMATESBASE2000) |
| 2005 | Census 2000 file (POPESTIMATE2005) |
| 2010 | PEP 2010-2019 (census base) |
| 2015 | PEP 2010-2019 (YEAR=6) |
| 2020 | SDC 2024 base population |
| 2024 | PEP Vintage 2025 (YEAR=6) |

These six time points define five intercensal periods: 2000-2005, 2005-2010, 2010-2015, 2015-2020, and 2020-2024. Each period produces a complete set of 1,908 migration rate cells.

**Period length adjustment.** The final period (2020-2024) spans four years rather than five. To maintain comparability, the survival rate is adjusted:

$$S_{4\text{yr}}(a, s) = S_{5\text{yr}}(a, s)^{4/5}$$

This exponent-based adjustment preserves the exponential decay structure of the survival function. The annualization step accounts for the shorter period by setting $n = 4$ in the compound annualization formula.

**Open-ended 85+ age group.** The terminal age group (85+) cannot be computed by a simple one-step age shift because it receives survivors from two source groups: those aged 80-84 who survive and age into the 85+ group, and those already in the 85+ group who survive within it. The expected population is:

$$\text{Expected}_{85+} = P(80\text{-}84) \times S_{5\text{yr}}(80\text{-}84, s) + P(85+) \times S_{5\text{yr}}(85+, s)$$

The migration residual is then computed as:

$$\text{Migration}_{85+} = P_{\text{end}}(85+) - \text{Expected}_{85+}$$

**Birth cohort (0-4).** The youngest age group at the end of each period (ages 0-4) represents children born during the period. Because there is no starting cohort to age forward, the residual method cannot attribute any migration to this group. These cells are included in the output with a migration rate of zero.

### 5b. GQ-Corrected Migration Rates (ADR-055 Phase 2)

The residual method as described in Section 5a uses total resident population (Respop), which includes both household population and group quarters (GQ) population. Group quarters encompass military barracks, college dormitories, nursing facilities, and correctional institutions. For counties with significant institutional populations, this conflation creates systematic distortions: when a university freshman replaces a graduating senior in the same dormitory bed, the residual method counts an in-migrant at age 18 and an out-migrant at age 25, even though the institutional capacity is unchanged.

To address this, the pipeline subtracts historical GQ populations from the six population snapshots before computing residual migration, producing rates that reflect household-only migration patterns (ADR-055 Phase 2).

**GQ subtraction formula.** For each county, age group, sex, and time point:

$$P_{\text{hh}}(a, t) = \max\bigl(P_{\text{total}}(a, t) - \text{GQ}(a, t),\; 0\bigr)$$

where $P_{\text{hh}}$ is the household-only population used in the residual computation. The floor at zero prevents negative household populations in cells where the GQ estimate exceeds the total population (which can occur due to the race-distribution approximation).

**Historical GQ estimation.** GQ by county, five-year age group, and sex is available from the Census Bureau's PEP stcoreview product for years 2020 and 2024. For earlier time points (2000, 2005, 2010, 2015), the 2020 GQ levels are applied as a backward constant. This assumption is defensible because institutional capacity -- barracks, dormitories, nursing beds -- changes slowly over five- to ten-year windows. The primary goal is removing institutional rotation signals from the rates, not precisely tracking historical GQ changes.

**Effect on rates.** The dominant effect of GQ correction is on the most recent period (2020-2024), where state GQ grew from 26,223 to 30,884 (+17.8%), driven primarily by NDSU dormitory construction in Cass County (+2,929). By subtracting this GQ growth from the 2024 population snapshot, the pipeline correctly identifies dorm construction as institutional expansion rather than demographic in-migration. This prevents dormitory construction from inflating future household migration projections through the convergence pipeline.

For periods with backward-constant GQ (2000-2005 through 2015-2020), the same GQ is subtracted from both start and end populations. The migration numerator changes by approximately $-\text{GQ} \times (1 - S)$ (GQ mortality removed from apparent migration), and the denominator shrinks. The net effect is a modest amplification of existing rate magnitudes.

### 5c. Oil-Boom Dampening (ADR-040, ADR-051)

The Bakken oil boom (2006-2015) and its aftermath produced migration flows into western North Dakota that were historically unprecedented and are unlikely to recur at the same scale. Left undampened, these flows dominate the multi-period average and inflate long-term projections for oil-impacted counties.

**Target counties.** Six counties in the Bakken formation region receive dampening:

| County | FIPS |
|--------|:----:|
| Williams (Williston) | 38105 |
| McKenzie (Watford City) | 38053 |
| Mountrail (Stanley) | 38061 |
| Dunn (Killdeer) | 38025 |
| Stark (Dickinson) | 38089 |
| Billings (Medora) | 38007 |

**Period-specific dampening factors.** Dampening is applied only to boom and boom-adjacent periods, with period-specific factors that reflect the varying intensity of boom-era migration (ADR-051):

| Period | Factor | Rationale |
|--------|:------:|-----------|
| 2005-2010 | 0.50 | Early boom; rapid in-migration |
| 2010-2015 | 0.40 | Peak boom; most extreme migration flows |
| 2015-2020 | 0.50 | Boom-adjacent; infrastructure momentum and family reunification |

The periods 2000-2005 and 2020-2024 are not dampened: 2000-2005 predates the boom, and 2020-2024 shows the post-boom migration reversal that the model should capture.

**Dampening formula.** For each affected county and boom period, all age-sex migration rate cells are scaled:

$$r_{\text{dampened}}(a, s) = r_{\text{residual}}(a, s) \times f_{\text{period}}$$

where $f_{\text{period}}$ is the period-specific factor from the table above. The dampening is applied uniformly across all age-sex cells within a county-period, meaning it preserves the county's age-sex migration shape while reducing its magnitude.

**Calibration.** The dampening factors were calibrated so that 20-year baseline projections for oil counties align within 2 percentage points of the SDC 2024 reference projections. ADR-051 evaluated whether further reduction (from the original 0.60 to 0.40) was warranted, but fresh projections on the consistent 20-year horizon showed adequate calibration with the current factors. The ADR was rejected -- calibration is already adequate.

### 5d. Male Migration Dampening

During the peak oil boom periods (2005-2010 and 2010-2015), male in-migration was disproportionately high relative to female in-migration, reflecting the male-dominated workforce in oil extraction and related industries. An additional dampening factor is applied to all male migration rates statewide during these periods to prevent the sex-skewed boom-era migration from distorting the multi-period average.

**Formula.** For boom periods only:

$$r_{\text{male,dampened}}(a) = r_{\text{male}}(a) \times 0.80$$

This 20% reduction applies to all male age-sex cells across all counties (not just oil counties), because the boom's labor market effects extended beyond the immediate Bakken region through secondary employment, housing construction, and supply chain activity.

The dampening applies only to periods 2005-2010 and 2010-2015. All other periods retain their unmodified male rates. Female rates are not dampened in any period.

**Pipeline ordering.** Male dampening is applied after oil-boom dampening. For male cells in oil counties during boom periods, both dampening factors compound. For example, a male cell in McKenzie County during 2010-2015 receives an effective factor of $0.40 \times 0.80 = 0.32$.

### 5e. PEP Recalibration for Reservation Counties (ADR-045)

Three counties with significant American Indian/Alaska Native (AIAN) reservation populations -- Benson (38005, Fort Berthold), Sioux (38085, Standing Rock), and Rolette (38079, Turtle Mountain) -- exhibit a systematic divergence between residual-method migration estimates and Census PEP component estimates. The residual method overestimates out-migration by a factor of 1.4x to 2.0x, producing 30-year projected declines of 45-47% that are implausible given historical trajectories of -11% to -15% per 20 years.

The divergence arises from two sources: differential census undercounts on tribal lands that inflate apparent population loss between censuses, and the application of statewide survival rates to populations with lower life expectancy, which causes excess deaths to be misattributed as out-migration.

**Recalibration approach.** For each target county and period, the pipeline compares the residual total with the PEP total and applies one of two recalibration methods:

**Case 1: Same sign and non-trivial residual.** When the residual and PEP totals agree in direction and the absolute residual exceeds a near-zero threshold (10 persons), the age-sex rates are uniformly scaled to match the PEP total while preserving the county-specific age-sex shape:

$$k = \frac{\text{PEP}_{\text{total}}}{\text{Residual}_{\text{total}}}$$

$$r_{\text{recalibrated}}(a, s) = r_{\text{residual}}(a, s) \times k$$

**Case 2: Sign reversal or near-zero residual.** When the PEP and residual disagree in sign (e.g., PEP shows net in-migration while the residual shows net out-migration), or when the absolute residual is below the near-zero threshold, the county-specific age-sex shape is unreliable. In these cases, the PEP total is redistributed across age-sex cells using a Rogers-Castro standard migration age profile with a 50/50 sex split:

$$r_{\text{RC}}(a, s) = \frac{\text{PEP}_{\text{total}} \times w_s \times w_{\text{RC}}(a)}{\text{Expected}(a, s)}$$

where $w_s$ is the sex share (0.5 for male, 0.5 for female), $w_{\text{RC}}(a)$ is the normalized Rogers-Castro age-specific migration propensity weight, and $\text{Expected}(a, s)$ is the expected population for the cell (used as the rate denominator). The resulting rate is then annualized using the same compound formula as in Section 5a.

The Rogers-Castro model -- a well-established parametric model of age-specific migration propensity (Rogers, 1988) -- produces a plausible age profile when the county-specific residual data is unreliable. Its peak migration propensity at age 25 reflects the labor-force entry pattern that dominates county-level migration for most populations.

**Pipeline position.** PEP recalibration is applied after oil-boom dampening and male dampening, but before college-age smoothing and multi-period averaging. This ordering ensures that dampening adjustments (which affect all counties) are applied first, and the PEP recalibration corrects the reservation-specific totals before rates enter the averaging step.

### 5f. College-Age Smoothing (ADR-049)

Counties with major universities exhibit extreme migration rates at college ages (15-19 and 20-24) that reflect transient student enrollment cycles rather than permanent settlement patterns. For example, Cass County (NDSU) showed a raw 20-24 annual migration rate of +12.4% -- implying that Fargo receives the equivalent of 12% of its young adult population as new permanent residents every year, which is implausible.

To prevent enrollment-driven spikes from dominating the convergence rates, the pipeline blends college-county rates with the statewide average for the affected age groups.

**Target counties and age groups:**

| County | FIPS | Institution |
|--------|:----:|-------------|
| Grand Forks | 38035 | University of North Dakota |
| Cass | 38017 | North Dakota State University |
| Ward | 38101 | Minot State University |
| Burleigh | 38015 | University of Mary, Bismarck State College |

The smoothing applies to age groups 15-19 and 20-24.

**Smoothing formula.** For each affected county, age group, sex, and period:

$$r_{\text{smoothed}} = 0.5 \times r_{\text{county}} + 0.5 \times \bar{r}_{\text{state}}$$

where $r_{\text{county}}$ is the county-specific rate and $\bar{r}_{\text{state}}$ is the arithmetic mean of that age-sex cell's rate across all 53 counties for the same period. The 50/50 blend factor preserves half of the county-specific enrollment signal while pulling extreme values toward the state average.

**Application timing.** A critical design decision (ADR-049) is that smoothing is applied to period-level rates before multi-period averaging, not after. The convergence pipeline reads its input from the period-level rates file (`residual_migration_rates.parquet`). If smoothing were applied only to the averaged rates, the convergence pipeline would operate on unsmoothed period-level rates, bypassing the correction entirely. By smoothing at the period level, both the averaged rates and the convergence pipeline inputs inherit the correction.

**No double-smoothing.** Because the averaged rates are computed from already-smoothed period rates, a second smoothing pass on the averaged rates would over-correct. The pipeline explicitly skips re-smoothing at the averaging stage.

### 5g. Multi-Period Averaging

After all period-level adjustments (dampening, recalibration, smoothing), the five periods of migration rates are combined into a single set of averaged rates per county-by-age-group-by-sex cell.

**Method.** The pipeline uses a simple arithmetic mean (equal weight to each period):

$$\bar{r}(c, a, s) = \frac{1}{5} \sum_{p=1}^{5} r_p(c, a, s)$$

where $r_p(c, a, s)$ is the annualized, adjusted migration rate for county $c$, age group $a$, sex $s$, and period $p$.

Equal weighting treats all five periods as equally informative about the future. This is a deliberate choice: while the most recent period might seem most relevant, it captures only four years of data and may reflect temporary shocks (e.g., COVID-19 effects on 2020-2024). Equal weighting allows the 25-year historical record to moderate period-specific anomalies.

The averaged rates are saved as `residual_migration_rates_averaged.parquet` and serve as a summary diagnostic. The convergence pipeline (Section 5h) reads the period-level rates directly and computes its own window averages, so the averaged file is not an input to the projection engine.

### 5h. Convergence Interpolation (5-10-5 Schedule)

Rather than applying a single static migration rate for all 20 projection years, the model uses a convergence interpolation schedule that transitions rates from their recent observed values toward long-term historical means. This reflects the demographic expectation that recent migration patterns -- which may reflect transient conditions -- will gradually revert to longer-term norms.

**Window definitions.** The pipeline defines three temporal windows, each of which produces an average rate by computing the arithmetic mean of all period-level rates whose time span overlaps the window:

| Window | Config Range | Periods Included | Interpretation |
|--------|:------------:|:----------------:|----------------|
| Recent | [2022, 2024] | (2020, 2024) | Current conditions |
| Medium | [2014, 2024] | (2010, 2015), (2015, 2020), (2020, 2024) | Medium-term trend |
| Long-term | [2000, 2024] | All 5 periods | Full historical record |

A period is included in a window if it overlaps with the window's year range. The window averages are computed independently for each county-by-age-group-by-sex cell.

**Convergence schedule.** The 5-10-5 schedule divides the 20-year projection horizon into three phases:

**Phase 1 (Years 1-5): Linear interpolation from recent to medium.** Each year blends the recent and medium window averages with a linearly shifting weight:

$$r(y) = r_{\text{recent}} \times \left(1 - \frac{y}{5}\right) + r_{\text{medium}} \times \frac{y}{5}$$

At year 1, the rate is 80% recent and 20% medium. By year 5, the rate has fully converged to the medium-term average.

**Phase 2 (Years 6-15): Hold at medium.** The migration rate is held constant at the medium window average:

$$r(y) = r_{\text{medium}}$$

This ten-year hold reflects the assumption that medium-term migration trends are the best available predictor for the middle horizon of the projection.

**Phase 3 (Years 16-20): Linear interpolation from medium to long-term.** The rate transitions from the medium window average toward the long-term historical average:

$$r(y) = r_{\text{medium}} \times \left(1 - \frac{t}{5}\right) + r_{\text{longterm}} \times \frac{t}{5}$$

where $t = y - 15$. By year 20, the rate equals the long-term average, reflecting the assumption that over a full generation, migration patterns revert to their 25-year historical norm.

**Rationale for 5-10-5.** The schedule balances responsiveness to recent trends (Phase 1) with stability during the core projection period (Phase 2) and mean reversion over the long run (Phase 3). The ten-year medium hold ensures that the projection is not overly sensitive to either the most recent period (which may be anomalous) or the oldest periods (which may be outdated).

### 5i. Age-Aware Rate Cap (ADR-043)

After convergence interpolation, an age-aware rate cap is applied to clip statistically implausible extreme values. Small counties with tiny cell populations (often 5-30 people in an age-sex cell) can produce double-digit percentage migration rates from a single family moving in or out. These rates are statistical noise, not demographic signal.

**Cap thresholds:**

| Age Groups | Cap | Rationale |
|:----------:|:---:|-----------|
| 15-19, 20-24 (college ages) | $\pm 15\%$ | Preserves legitimate university enrollment dynamics (peak rates of 13-14%) |
| All other ages | $\pm 8\%$ | Clips small-county noise; sits at the 99th percentile of the medium-term distribution |

**Application formula.** For each cell:

$$r_{\text{capped}} = \begin{cases} \text{clip}(r, -0.15, +0.15) & \text{if } a \in \{15\text{-}19, 20\text{-}24\} \\ \text{clip}(r, -0.08, +0.08) & \text{otherwise} \end{cases}$$

The cap is applied after the convergence interpolation for each year offset but before storing the result, so it catches all three convergence phases without modifying the underlying window averages. This preserves data lineage while guarding against outlier propagation.

**Scope of impact.** The cap clips approximately 2.4% of all cells across all year offsets (1,372 of 57,240 cells). It is complementary to -- not a substitute for -- oil-boom dampening: the cap addresses individual cell-level statistical noise, while dampening addresses the broad pattern of elevated working-age rates across oil counties.

### 5j. BEBR High-Growth Increment (ADR-046)

The high-growth scenario models a counterfactual future with sustained elevated in-migration, grounded in BEBR (Bureau of Economic and Business Research) multi-period averaging methodology. Rather than applying a multiplicative factor to net migration rates -- which would amplify out-migration in counties with negative net rates, producing the opposite of the intended effect -- the high scenario uses an additive rate increment applied to the convergence window averages before interpolation.

**Increment computation.** For each county, the increment is derived from the difference between BEBR high and BEBR baseline migration files:

$$\Delta_{\text{county}} = \text{BEBR}_{\text{high,net}} - \text{BEBR}_{\text{baseline,net}}$$

$$\delta_{\text{cell}} = \frac{\Delta_{\text{county}}}{P_{\text{county}} \times 36}$$

where $\Delta_{\text{county}}$ is the absolute difference in total net migration (persons per year) between the high and baseline BEBR scenarios, $P_{\text{county}}$ is the county's reference population (from the most recent period's starting population), and 36 is the number of age-sex cells per county (18 age groups times 2 sexes). The per-cell increment $\delta_{\text{cell}}$ is expressed as an annual migration rate.

**Application.** The increment is added uniformly to all three window averages (recent, medium, and long-term) for every cell in each county:

$$r_{\text{high,window}}(c, a, s) = r_{\text{baseline,window}}(c, a, s) + \delta_{\text{cell}}(c)$$

After lifting, the modified window averages pass through the standard 5-10-5 convergence interpolation and rate cap, producing `convergence_rates_by_year_high.parquet`. The projection engine loads these scenario-specific convergence rates instead of the baseline rates when running the high-growth scenario.

**Why additive.** The increment is added rather than multiplied because an additive increment guarantees $r_{\text{high}} \geq r_{\text{baseline}}$ regardless of the sign of the base rate. Since 43 of 53 North Dakota counties have negative net migration at the medium convergence hold, a multiplicative approach would amplify out-migration and produce lower population than baseline -- the opposite of the scenario's intent.

**Grounding.** The statewide BEBR high-vs-baseline increment of approximately +1,302 migrants per year is corroborated by three independent estimates: the CBO January 2025 elevated-vs-long-term immigration trajectory at North Dakota's national share (~1,163/year), the PEP international migration surge (50% of excess, ~1,243/year), and the BEBR rate file difference itself (+1,302/year).

### 5k. Ward County Migration Floor (ADR-052)

Ward County (Minot, population ~68,000) presents a particular challenge: despite being North Dakota's fourth-largest county with significant institutional anchors -- Minot Air Force Base, Minot State University, and a regional medical/retail center -- it projects population decline under all three scenarios when using the standard pipeline. The high-growth scenario, intended to represent an optimistic future, showed a -1.2% decline before correction.

The root cause is that the 2020-2024 period -- which dominates the "recent" convergence window -- captures an anomalously negative interval of COVID effects, reduced oil activity, and possible military force adjustments. The convergence schedule carries this negative starting point through the projection.

**Migration floor.** For the high-growth scenario only, after the BEBR increment has been added to the window averages, the pipeline checks each county's mean convergence rate across all age-sex cells. If the mean is negative, all cells are lifted uniformly so that the county mean equals zero (neutral migration):

$$\text{lift} = \max(0,\; f_{\text{floor}} - \bar{r}_{\text{county}})$$

$$r_{\text{floored}}(a, s) = r(a, s) + \text{lift}$$

where $f_{\text{floor}} = 0.0$ (neutral migration) and $\bar{r}_{\text{county}}$ is the arithmetic mean of the county's migration rates across all 36 cells. The lift is applied to all three window averages (recent, medium, long-term) before convergence interpolation.

In plain language: if a county's high-growth migration rates still average to net out-migration after the BEBR increment, the pipeline shifts the entire rate distribution upward until the county is at least replacing its out-migrants. This prevents the high-growth scenario from showing population decline for counties with significant institutional anchors whose long-term viability is not in question.

**Scope.** The floor applies only to the high-growth scenario. Baseline and restricted scenarios are unaffected. In practice, the floor affects Ward County and a small number of other declining counties whose BEBR increment is insufficient to offset persistent negative migration. Counties with positive BEBR-boosted rates are untouched.

**Interaction with other adjustments.** The floor is applied after the BEBR increment lift (Section 5j) but before convergence interpolation and rate capping. This means the floored rates still pass through the standard 5-10-5 schedule and age-aware rate cap, maintaining consistency with the rest of the pipeline.


## 6. Scenario Methodology

The projection system produces three scenarios that bracket a plausible range of future population outcomes for North Dakota. Each scenario shares the same cohort-component engine but differs in the assumptions applied to fertility, mortality, and migration rates. The scenario definitions are codified in the project configuration (`config/projection_config.yaml`) and governed by a series of Architecture Decision Records.

### 6.1 Baseline Scenario

The baseline scenario represents a trend-continuation projection in which historical demographic patterns are carried forward without explicit policy adjustments. It serves as the central estimate against which the other two scenarios are compared.

**Fertility.** Age-specific fertility rates are held constant at their base-year (2025) values throughout the projection horizon. No secular trend in fertility is assumed. The TFR remains fixed at its initial level for each race/ethnicity group.

**Mortality.** Survival rates improve at a compound annual rate of 0.5%, following a Lee-Carter-style mortality decline. For each projection year $t$, the age-specific death rate is:

$$q(a, s, r, t) = q(a, s, r, t_0) \times (1 - \delta)^{t - t_0}$$

where $\delta = 0.005$ and $t_0 = 2025$ is the base year. The corresponding survival rate is $S(a, s, r, t) = 1 - q(a, s, r, t)$, capped at 1.0.

**Migration.** The baseline uses convergence-interpolated migration rates derived from the Census Bureau residual method (see Section 5). These are time-varying, year-specific rates that converge from recent historical patterns toward long-term averages over the 30-year projection horizon using a 5-10-5 interpolation schedule. No additional scenario adjustment is applied to migration rates; the `recent_average` label in the engine means the convergence rates are used as-is.

### 6.2 Restricted Growth Scenario

The restricted growth scenario models the demographic impact of reduced international immigration due to federal enforcement policy changes. The methodology is grounded in Congressional Budget Office (CBO) immigration projections (ADR-037) and uses an additive migration adjustment to avoid a mathematical artifact that plagued earlier multiplicative approaches (ADR-050).

#### 6.2.1 CBO-Derived Immigration Enforcement Factors

The time-varying enforcement factors represent the fraction of normal international immigration that is realized in each year, based on the difference between CBO's January 2025 and January 2026 immigration outlooks. The schedule ramps up from severe restriction to full recovery:

| Year | Factor $f(t)$ | Interpretation |
|------|--------------|----------------|
| 2025 | 0.20 | 80% reduction from baseline international migration |
| 2026 | 0.37 | 63% reduction |
| 2027 | 0.55 | 45% reduction |
| 2028 | 0.78 | 22% reduction |
| 2029 | 0.91 | 9% reduction |
| 2030+ | 1.00 | Full recovery; no further restriction |

After 2030, the factor equals 1.0 and the restricted scenario migration rates are identical to the baseline. The restricted scenario therefore converges to the baseline trajectory over time, with the cumulative population deficit persisting as a permanent level shift.

#### 6.2.2 Additive Migration Reduction (ADR-050)

Earlier implementations used a multiplicative approach, scaling migration rates by the enforcement factor. This produced a sign-interaction defect: for counties with net-negative migration rates (i.e., net out-migration), multiplying by a factor less than 1.0 made the rate *less negative*, resulting in restricted-scenario populations that exceeded the baseline---a logical impossibility for a scenario modeling reduced immigration.

The corrected approach uses an **additive reduction**. Two state-level reference values anchor the computation:

- $M_{\text{intl}} = 10{,}051$: average annual international net migration to North Dakota (PEP 2023--2025 average)
- $P_{\text{ref}} = 799{,}358$: state population at the base year (2025)

For each projection year $t$, the per-capita rate decrement is:

$$\Delta r(t) = \frac{M_{\text{intl}} \times \bigl(1 - f(t)\bigr)}{P_{\text{ref}}}$$

This represents the per-capita share of international migrants who do not arrive due to enforcement. The adjusted migration rate for every age-sex-race cell is then:

$$m_{\text{restricted}}(a, s, r, t) = m_{\text{baseline}}(a, s, r, t) - \Delta r(t)$$

Because $\Delta r(t) \geq 0$ for all $t$ (the factor $f(t) \leq 1$), and the same non-negative decrement is subtracted from every cell, the restricted rate is always less than or equal to the baseline rate. This **guarantees** that $P_{\text{restricted}} \leq P_{\text{baseline}}$ for every county, regardless of the sign of the underlying migration rate.

The total person-reduction for a given county is proportional to its population: a county with population $P_c$ loses approximately $\Delta r(t) \times P_c$ persons per year relative to the baseline, which is the county's proportional share of the state-level immigration reduction.

**Example.** In 2025, $f(t) = 0.20$, so $\Delta r = 10{,}051 \times 0.80 / 799{,}358 \approx 0.01006$ per capita. A county with 50,000 residents would see approximately 503 fewer net migrants relative to the baseline in that year.

#### 6.2.3 Fertility Adjustment

The restricted growth scenario also applies a 5% reduction to all age-specific fertility rates:

$$F_{\text{restricted}}(a, r) = F_{\text{baseline}}(a, r) \times 0.95$$

This reflects the assumption that reduced immigration, which disproportionately involves younger adults with higher-than-average fertility, slightly depresses the aggregate fertility rate.

#### 6.2.4 Mortality

Mortality improvement is identical to the baseline (0.5% annual improvement in death rates).

### 6.3 High Growth Scenario

The high growth scenario represents a counterfactual in which migration remains elevated relative to recent trends and fertility modestly increases. It is designed as an upper bound for planning purposes, particularly relevant for infrastructure and service-capacity analysis (ADR-046).

#### 6.3.1 BEBR-Boosted Migration Rates

Rather than applying an in-engine multiplier to migration rates, the high growth scenario uses an entirely separate set of convergence-interpolated rates stored in `convergence_rates_by_year_high.parquet`. These rates are produced by the convergence pipeline (see Section 5) with an additive rate increment derived from the Bureau of Economic and Business Research (BEBR) at the University of Florida.

The increment is computed as the difference between the BEBR high and baseline county-level net migration estimates, converted to a per-cell rate:

$$\Delta r_{\text{high}}(c) = \frac{M_{\text{BEBR,high}}(c) - M_{\text{BEBR,baseline}}(c)}{P(c) \times N_{\text{cells}}}$$

where $P(c)$ is the county population and $N_{\text{cells}} = 36$ (18 five-year age groups $\times$ 2 sexes). This increment is added uniformly to all three convergence window averages (recent, medium, and long-term) before the 5-10-5 interpolation is applied, ensuring the high-growth signal persists across the entire projection horizon.

#### 6.3.2 Migration Floor (ADR-052)

Even after the BEBR increment, some counties with strong structural out-migration may retain negative mean convergence rates. For a high growth scenario, projecting population decline in counties with strong institutional anchors (military bases, universities, regional service centers) produces results that are not useful for planning purposes.

ADR-052 therefore imposes a migration floor: if any county's mean convergence rate across all age-sex cells remains negative after the BEBR lift, all cells in that county are uniformly shifted upward so that the mean rate equals zero:

$$\text{If } \bar{m}(c) < 0: \quad m_{\text{floored}}(a, s, c) = m(a, s, c) + (0 - \bar{m}(c))$$

This floor is applied to all three window averages before convergence interpolation, and only in the high growth scenario.

#### 6.3.3 Fertility Adjustment

The high growth scenario applies a 5% increase to all age-specific fertility rates:

$$F_{\text{high}}(a, r) = F_{\text{baseline}}(a, r) \times 1.05$$

#### 6.3.4 Mortality

Mortality improvement is identical to the baseline (0.5% annual improvement in death rates).

### 6.4 Scenario Comparison Summary

| Parameter | Baseline | Restricted Growth | High Growth |
|-----------|----------|-------------------|-------------|
| Fertility | Constant | $-5\%$ | $+5\%$ |
| Mortality improvement | 0.5%/yr | 0.5%/yr | 0.5%/yr |
| Migration rates | Convergence (baseline) | Convergence $-\ \Delta r(t)$ | Convergence (BEBR-boosted) |
| Migration adjustment | None | Additive reduction (ADR-050) | BEBR increment + floor (ADR-046, ADR-052) |
| CBO enforcement factors | N/A | 0.20 to 1.00 (2025--2030+) | N/A |
| Governing ADRs | --- | ADR-037, ADR-039, ADR-050 | ADR-046, ADR-052 |


## 7. Projection Engine

The cohort-component projection engine advances the population of each county forward one year at a time, applying survival, fertility, and migration sequentially. This section describes the mechanics of a single annual step, the treatment of time-varying rates, group quarters re-addition, and the aggregation strategy that produces state-level totals.

### 7.1 Annual Projection Step

For each county and each projection year $t \to t+1$, the engine performs the following operations in sequence.

#### Step 1: Survival (Aging and Mortality)

Every existing cohort is aged by one year and reduced by the age-, sex-, race-, and year-specific survival rate:

$$P_{\text{survived}}(a+1, s, r, t+1) = P(a, s, r, t) \times S(a, s, r, t)$$

where $S(a, s, r, t)$ is the probability of surviving from age $a$ to age $a+1$ during year $t$.

**Open-ended age group (90+).** Persons aged 89 who survive advance into the 90+ group, while persons already in the 90+ group experience within-group mortality but remain at age 90+:

$$P(90{+}, s, r, t{+}1) = P(89, s, r, t) \times S(89, s, r, t) + P(90{+}, s, r, t) \times S(90{+}, s, r, t)$$

The survival rate for the 90+ group represents the probability of surviving at least one more year within this open-ended interval.

**Time-varying survival.** When year-specific survival tables are provided (via the mortality improvement mechanism), the engine retrieves the table for calendar year $t$. The improvement formula (see Section 4) is applied to the base-year death rates before conversion back to survival rates. When pre-computed year-specific tables are loaded directly, the improvement factor is set to zero within the engine to avoid double-application.

#### Step 2: Births

Total births by race are calculated by applying age-specific fertility rates to the female population of reproductive age (15--49):

$$B(r, t) = \sum_{a=15}^{49} F(a, r) \times P_f(a, r, t)$$

where $F(a, r)$ is the age-specific fertility rate for age $a$ and race $r$, and $P_f$ denotes female population. Births are then split by sex using a fixed sex ratio at birth:

$$B_{\text{male}}(r, t) = B(r, t) \times 0.51$$
$$B_{\text{female}}(r, t) = B(r, t) \times 0.49$$

The sex ratio of 0.51 male (equivalently, 105 males per 100 females) follows the standard demographic convention for developed-country populations. All births are assigned age 0 and inherit the mother's race/ethnicity classification.

Note that births are calculated from the pre-migration population (i.e., the population at the start of the year), which is the standard cohort-component convention. Newborns do not experience migration in their birth year.

#### Step 3: Migration

Net migration is applied to the survived population using rate-mode migration. For each age-sex-race cell:

$$P_{\text{final}}(a, s, r, t{+}1) = P_{\text{survived}}(a, s, r, t{+}1) + P_{\text{survived}}(a, s, r, t{+}1) \times m(a, s, r, t)$$

which simplifies to:

$$P_{\text{final}}(a, s, r, t{+}1) = P_{\text{survived}}(a, s, r, t{+}1) \times \bigl(1 + m(a, s, r, t)\bigr)$$

where $m(a, s, r, t)$ is the net migration rate for the cell in year $t$. A positive rate represents net in-migration; a negative rate represents net out-migration.

**Time-varying migration.** The engine retrieves year-specific migration rates from the convergence interpolation output (see Section 5). The lookup is by year offset ($t - t_0 + 1$), where $t_0 = 2025$ is the base year. If a year-specific rate table is not available for a given offset, the engine falls back to the constant (base) migration rates.

**Scenario adjustments.** Before migration is applied, scenario-specific adjustments modify the rates. For the restricted growth scenario, the additive reduction (Section 6.2.2) is subtracted from each cell. For the high growth scenario, the BEBR-boosted convergence rates are loaded directly, so no in-engine adjustment is needed beyond the standard convergence lookup.

#### Step 4: Combine Survived Population and Births

The survived-and-migrated population (ages 1--90+) is concatenated with the newborn cohort (age 0). Any duplicate age-sex-race cells arising from this combination are aggregated by summation.

#### Step 5: Non-Negativity Enforcement

After all demographic operations, any cohort with a negative population count is clamped to zero:

$$P(a, s, r, t{+}1) = \max\bigl(P(a, s, r, t{+}1),\ 0\bigr)$$

Negative populations can arise when large net out-migration exceeds the surviving population in a small cohort. The engine logs a warning when clamping occurs but does not halt execution, since small-cell negativity in race-detailed projections is a known artifact of rate-based migration.

### 7.2 Multi-Year Execution

The full projection iterates the single-year step from the base year through the end of the projection horizon:

```
for t in [2025, 2026, ..., 2054]:
    population(t+1) = project_single_year(population(t), t, scenario)
```

The engine stores the population state at each year, producing a complete time series of 31 annual snapshots (2025 through 2055, inclusive). Each snapshot contains the full age $\times$ sex $\times$ race population matrix for the county.

### 7.3 Group Quarters Re-Addition (ADR-055)

The projection engine operates on household-only population. Before projection begins, group quarters (GQ) population---comprising persons in college dormitories, military barracks, nursing facilities, correctional institutions, and other institutional or non-institutional group living arrangements---is separated from the base population (see Section 2).

After the projection is complete for each county, the GQ population is added back as a constant at every projection year:

$$P_{\text{total}}(a, s, r, t) = P_{\text{HH}}(a, s, r, t) + P_{\text{GQ}}(a, s, r)$$

where $P_{\text{HH}}$ is the projected household population and $P_{\text{GQ}}$ is the base-year (2025) group quarters population, which is held constant throughout the projection horizon. This hold-constant assumption reflects the institutional nature of GQ housing: its capacity is determined by physical infrastructure (dormitory beds, barracks capacity, nursing home beds) rather than by demographic trends.

The GQ data is stored at the county level by age, sex, and race. Because Census GQ tabulations do not provide detailed race breakdowns at the county level, the GQ population is distributed across race categories proportionally to each county's overall race composition.

### 7.4 State Aggregation (ADR-054)

The system does not produce an independent state-level projection. Instead, the state total is derived strictly as the sum of all 53 county projections:

$$P_{\text{state}}(a, s, r, t) = \sum_{c=1}^{53} P_c(a, s, r, t)$$

This bottom-up aggregation strategy (ADR-054) ensures perfect internal consistency: the state population at every age, sex, race, and year is exactly equal to the sum of the corresponding county values. No reconciliation or pro-rata adjustment is needed, and there is no "remainder" or residual term.

The rationale for this design is that county-level migration patterns are inherently heterogeneous---oil-patch counties, university counties, reservation counties, and metropolitan counties experience fundamentally different demographic dynamics. An independent state projection with top-down allocation to counties would either suppress this heterogeneity or require an additional allocation step that introduces its own assumptions and artifacts.

### 7.5 Place Projection Orchestration (PP-003 Phase 2)

Place-level outputs are generated from county projections using a county-constrained share-trending workflow (ADR-033 implementation, PP-003 Phase 2). The orchestration pipeline:

1. Loads historical place shares and a place-to-county crosswalk for the projection universe (HIGH, MODERATE, LOWER tiers; EXCLUDED places omitted from projection output).
2. Fits and projects place shares within each county using the selected trend variant (from the PP-003 backtesting winner), then reconciles place plus balance-of-county shares to sum to 1.0 for every county-year.
3. Converts projected shares to place totals by multiplying by county totals:

$$P_{\text{place}}(t) = s_{\text{place}}(t) \times P_{\text{county}}(t)$$

4. Allocates county demographic structure to place totals by confidence tier:
   - **HIGH**: 18 five-year age groups by sex (36 rows per year),
   - **MODERATE**: 6 broad age groups by sex (12 rows per year),
   - **LOWER**: total population only.
5. Writes per-place parquet/CSV/JSON outputs plus run-level `places_summary.csv` and `places_metadata.json`, including balance-of-county rows for county accounting transparency.

This process preserves county consistency by construction: for each county-year, projected place totals plus balance-of-county equal the projected county total (subject only to floating-point tolerance).


## 8. Data Sources and References

### 8.1 Primary Data Sources

**Census Bureau Population Estimates Program (PEP), Vintage 2025.** County-level population estimates by age, sex, and race/ethnicity for North Dakota, 2020--2025. Accessed via the Census Bureau's stcoreview data interface. These estimates serve as the basis for base population construction, residual migration computation, and PEP-anchored recalibration of reservation counties.

- U.S. Census Bureau. (2025). *County Population Estimates by Characteristics: Annual County Resident Population Estimates by Age, Sex, Race, and Hispanic Origin.* Vintage 2025. Retrieved from https://www.census.gov/programs-surveys/popest.html

**Census Bureau Sub-County Estimates (SC-EST).** Single-year-of-age population estimates by sex and race used for constructing the base population with single-year age detail. These provide the state-level single-year-of-age distribution that is combined with county-level five-year age group totals via Sprague osculatory interpolation.

- U.S. Census Bureau. (2025). *State Population by Characteristics: SC-EST2024.* Retrieved from https://www.census.gov/data/datasets/time-series/demo/popest/2020s-state-detail.html

**CDC National Center for Health Statistics (NCHS) / National Vital Statistics System (NVSS).** Age-specific fertility rates by race and Hispanic origin. North Dakota-specific rates are computed from NVSS natality microdata and adjusted to match the ND total fertility rate (ADR-053).

- Centers for Disease Control and Prevention. (2024). *National Vital Statistics Reports: Births.* National Center for Health Statistics. Retrieved from https://www.cdc.gov/nchs/nvss/births.htm

**CDC / Surveillance, Epidemiology, and End Results (SEER) Program Life Tables.** U.S. and state-level life tables by age, sex, and race/ethnicity used to derive survival rates. North Dakota-specific adjustments are applied as a ratio of ND to national age-specific mortality (ADR-053).

- National Cancer Institute. (2024). *U.S. Population Data, 1969--2023.* SEER Program. Retrieved from https://seer.cancer.gov/popdata/
- Arias, E., & Xu, J. (2024). *United States Life Tables, 2023.* National Vital Statistics Reports, 73(1). National Center for Health Statistics.

**Congressional Budget Office (CBO) Immigration Projections.** The January 2025 and January 2026 CBO budget and economic outlooks provide baseline and revised net immigration projections. The year-by-year difference between these two outlooks is used to derive the enforcement factor schedule for the restricted growth scenario.

- Congressional Budget Office. (2025, January). *The Budget and Economic Outlook: 2025 to 2035.* Washington, DC.
- Congressional Budget Office. (2026, January). *The Budget and Economic Outlook: 2026 to 2036.* Washington, DC.

**Census Bureau Group Quarters Population.** County-level group quarters population by age group and sex, extracted from the Census Bureau stcoreview data interface for 2020 and 2024. Used for GQ separation (ADR-055 Phase 1) and GQ-corrected migration rates (ADR-055 Phase 2).

- U.S. Census Bureau. (2025). *Population in Group Quarters by County: stcoreview HHpop/GQpop Tabulations.* Vintage 2025.

### 8.2 Methodological References

**Cohort-Component Method.** The foundational demographic projection method used throughout this system.

- Smith, S. K., Tayman, J., & Swanson, D. A. (2013). *A Practitioner's Guide to State and Local Population Projections.* Springer.
- Preston, S. H., Heuveline, P., & Guillot, M. (2001). *Demography: Measuring and Modeling Population Processes.* Blackwell.

**BEBR Migration Methodology.** The Bureau of Economic and Business Research (University of Florida) multi-period migration averaging method, adapted for use in the convergence interpolation pipeline.

- Smith, S. K., & Rayer, S. (2014). Projections of Florida Population by County, 2015--2040. *Florida Population Studies, Bulletin 168.* Bureau of Economic and Business Research, University of Florida.

**Census Bureau Residual Migration Method.** The residual method for estimating county-level net migration from successive population estimates and survival rates.

- Voss, P. R., McNiven, S., Hammer, R. B., Johnson, K. M., & Fuguitt, G. V. (2004). County-Specific Net Migration by Five-Year Age Groups, Hispanic Origin, Race, and Sex, 1990--2000. *CDE Working Paper No. 2004-24.* University of Wisconsin-Madison.

**Sprague Osculatory Interpolation.** The fifth-difference method used to graduate five-year age groups into single-year-of-age populations while preserving group totals and ensuring smooth age distributions.

- Sprague, T. B. (1880). Explanation of a new formula for interpolation. *Journal of the Institute of Actuaries,* 22(4), 270--285.
- Siegel, J. S., & Swanson, D. A. (2004). *The Methods and Materials of Demography* (2nd ed.). Elsevier Academic Press.

**Rogers-Castro Migration Age Schedule.** The model migration schedule used for age-profile redistribution in PEP-anchored recalibration of reservation counties (ADR-045), providing a theoretically grounded fallback when residual-method age profiles are unreliable.

- Rogers, A., & Castro, L. J. (1981). *Model Migration Schedules.* IIASA Research Report RR-81-30. International Institute for Applied Systems Analysis, Laxenburg, Austria.

**Lee-Carter Mortality Model.** The stochastic mortality forecasting framework that provides the theoretical basis for the 0.5% annual mortality improvement assumption used across all scenarios.

- Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association,* 87(419), 659--671.

### 8.3 Architecture Decision Records

The following Architecture Decision Records (ADRs) govern the methodological choices documented in this report. Each ADR is maintained in the project repository at `docs/governance/adrs/`.

| ADR | Title | Status | Section(s) |
|-----|-------|--------|------------|
| ADR-033 | City-Level Projection Methodology | Deferred | 7.4 |
| ADR-036 | Migration Averaging Methodology -- Multi-Period and Interpolation Approaches | Accepted | 5 |
| ADR-037 | CBO-Grounded Scenario Methodology | Accepted | 6.2 |
| ADR-039 | International-Only Migration Factor for Restricted Growth Scenario | Accepted | 6.2 |
| ADR-040 | Extend Bakken Oil Boom Migration Dampening to 2015--2020 Period | Accepted | 5 |
| ADR-043 | Age-Aware Migration Rate Cap for Convergence Rates | Accepted | 5 |
| ADR-045 | Reservation County PEP-Anchored Migration Recalibration | Accepted | 5 |
| ADR-046 | High Growth Scenario via BEBR Convergence Rates | Accepted | 6.3 |
| ADR-047 | County-Specific Age-Sex-Race Distributions | Accepted | 2 |
| ADR-048 | Single-Year-of-Age Base Population from Census SC-EST Data | Accepted | 2 |
| ADR-049 | College-Age Smoothing Propagation to Convergence Pipeline | Accepted | 5 |
| ADR-050 | Restricted Growth Additive Migration Adjustment | Accepted | 6.2 |
| ADR-051 | Oil County Dampening Recalibration | Rejected | 5 |
| ADR-052 | Ward County (Minot) Projection Review and High-Growth Scenario Floor | Accepted | 6.3 |
| ADR-053 | North Dakota-Specific Fertility and Mortality Rates | Accepted | 3, 4 |
| ADR-054 | State-County Aggregation Reconciliation | Accepted | 7.4 |
| ADR-055 | Group Quarters Population Separation | Accepted | 2, 5, 7.3 |
