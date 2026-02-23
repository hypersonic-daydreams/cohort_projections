# 07 Structural and Engine Differences

**Date:** 2026-02-20
**Series:** SDC 2024 vs. 2026 Cohort-Component Methodology Comparison
**Focus:** Projection architecture, time stepping, age resolution, demographic dimensions, and engine mechanics

---

## 1. SDC 2024 Approach

The North Dakota State Data Center's 2024 projections are built on a **modified cohort survival component method** implemented in a single Excel workbook (`Projections_Base_2023.xlsx`). The workbook contains 45 sheets organized around a six-period, five-year-step projection chain running from Census 2020 through 2050.

### 1.1 Structural Parameters

| Parameter | Value |
|-----------|-------|
| **Base year** | 2020 (Census April 1 count) |
| **Projection horizon** | 2020--2050 (30 years) |
| **Time step** | 5-year intervals |
| **Projection points** | 2025, 2030, 2035, 2040, 2045, 2050 |
| **Age structure** | 18 five-year groups: 0--4, 5--9, ... 80--84, 85+ |
| **Sex categories** | Male, Female (projected separately) |
| **Race/ethnicity** | Not included |
| **Geographic units** | 53 counties (summed to 8 planning regions and state) |
| **Implementation** | Microsoft Excel workbook with cell formulas |
| **Adjustment mechanism** | Manual per-period "Adjustments" sheets (~32,000 person-adjustments per period) |
| **Cohort cell count** | 18 age groups x 2 sexes x 53 counties = **1,908** cells per projection point |

### 1.2 Workbook Architecture

The Excel workbook organizes each 5-year period into a pipeline of sheets:

```
Census 2020 base population
  --> Fer 2020-2025          (fertility/births calculation)
  --> Nat_Grow 2020-2025     (survival applied to age cohorts)
  --> Adjustments 2020-2025  (manual corrections)
  --> 2020-2025 Migration    (migration = natural_growth x rate x multiplier)
  --> 2025 Pro               (final projected population)
      --> feeds into Fer 2025-2030 ...
```

Each sheet references the previous one through cell formulas, creating a rigid sequential pipeline. Modifying any upstream assumption (e.g., a survival rate) cascades automatically through downstream sheets for all subsequent periods.

### 1.3 Rate Handling

All component rates are held **constant** across projection periods:

- **Survival rates**: Derived from CDC 2020 ND life tables; 5-year survival probabilities by age group and sex. No mortality improvement applied.
- **Fertility rates**: Blended from ND DHHS 2016--2022 vital statistics, state averages, and national CDC rates. Held constant across all periods.
- **Migration rates**: Averaged from four 5-year census residual periods (2000--2020), then modulated by **period-varying multipliers** (0.2 for 2020--2025, 0.6 for 2025--2035, 0.5 for 2035--2040, 0.7 for 2040--2050). This is the only time-varying rate element in the SDC system.

---

## 2. Current 2026 Approach

Our 2026 projections use a Python-based cohort-component engine (`cohort_projections/core/cohort_component.py`) orchestrating three modular component subpackages for fertility, mortality, and migration. The engine is configured through `config/projection_config.yaml` and operates on pandas DataFrames with vectorized operations.

### 2.1 Structural Parameters

| Parameter | Value |
|-----------|-------|
| **Base year** | 2025 (Census PEP Vintage 2025 estimate) |
| **Projection horizon** | 2025--2055 (30 years) |
| **Time step** | Annual (1-year intervals) |
| **Projection points** | Every year: 2026, 2027, ... 2055 |
| **Age structure** | Single-year ages: 0, 1, 2, ... 89, 90+ |
| **Sex categories** | Male, Female |
| **Race/ethnicity** | 6 categories: White NH, Black NH, AIAN NH, Asian/PI NH, Two+ NH, Hispanic |
| **Geographic units** | 53 counties (summable to regions, state; place-level optional at 500+ threshold) |
| **Implementation** | Python 3.12 / pandas / NumPy |
| **Adjustment mechanism** | Algorithmic: convergence interpolation, scenario multipliers, additive reductions |
| **Cohort cell count** | 91 ages x 2 sexes x 6 races x 53 counties = **57,876** cells per projection year |

### 2.2 Engine Architecture

The engine is decomposed into four modules with clean interfaces (ADR-004):

```
CohortComponentProjection (orchestrator)
  |
  |-- mortality.py:  apply_survival_rates(population, rates, year, config)
  |                    --> ages cohorts by 1 year, applies survival probabilities
  |                    --> handles 90+ open-ended group in place
  |                    --> applies compounding mortality improvement
  |
  |-- fertility.py:  calculate_births(female_pop, rates, year, config)
  |                    --> filters to reproductive ages (15--49)
  |                    --> merges age x race fertility rates
  |                    --> splits births by sex ratio (51% male / 49% female)
  |                    --> assigns newborns to age 0
  |
  |-- migration.py:  apply_migration(survived_pop, rates, year, config)
  |                    --> supports both absolute (net_migration) and rate-based
  |                    --> clips negative populations to zero with warning
  |
  +-- convergence_interpolation.py:  time-varying migration rate pipeline
                      --> 5-10-5 schedule from recent -> medium -> long-term rates
                      --> age-aware rate caps (+/-15% college ages, +/-8% general)
```

### 2.3 Rate Handling

Component rates are time-varying in multiple dimensions:

- **Survival rates**: Base rates from CDC/SEER life tables (2023 vintage); **0.5% annual mortality improvement** applied as compounding reduction in death rates: `DR_t = DR_base x (1 - 0.005)^t`. Optionally overridden by pre-computed year-specific survival tables.
- **Fertility rates**: Age x race specific from CDC NCHS; held constant in baseline, adjustable by scenario (+/-5%, +/-10%, trending decline at 0.5%/year).
- **Migration rates**: Time-varying via **convergence interpolation** (Census Bureau method). Each county x age-group x sex cell independently converges from its recent historical rate to its long-term mean over a 5-10-5 year schedule. An age-aware cap clips extreme rates. Multiple scenarios apply additional adjustments (CBO additive reduction, BEBR-optimistic rates, zero migration).

---

## 3. Key Differences

| Dimension | SDC 2024 | Current 2026 | Ratio / Contrast |
|-----------|----------|--------------|------------------|
| **Time step** | 5-year | 1-year | 5x finer temporal resolution |
| **Age groups** | 18 five-year groups (0--4 ... 85+) | 91 single-year ages (0 ... 90+) | 5x finer age resolution |
| **Open-ended age** | 85+ | 90+ | 5 additional years of age detail |
| **Sex categories** | 2 | 2 | Same |
| **Race/ethnicity** | None | 6 categories | 6x dimensional expansion |
| **Cohorts per geography** | 36 (18 x 2) | 1,092 (91 x 2 x 6) | 30x more cohorts per county |
| **Total cells (state)** | 1,908 per point | 57,876 per year | 30x more cells |
| **Projection points** | 6 (every 5 years) | 30 (every year) | 5x more output points |
| **Total data points** | ~11,448 (1,908 x 6 points) | ~1,736,280 (57,876 x 30 years) | ~152x more data |
| **Base year** | 2020 Census | 2025 PEP | 5 years more recent |
| **Mortality trend** | Constant (no improvement) | 0.5% annual improvement | Declining death rates over time |
| **Fertility trend** | Constant | Constant (baseline); scenario-adjustable | Comparable in baseline |
| **Migration structure** | Constant base rates x period multiplier | Convergence interpolation (time-varying per cell) | Cell-level time variation vs. global multiplier |
| **Adjustments** | Manual (spreadsheet edits) | Algorithmic (config-driven) | Reproducibility difference |
| **Scenario support** | Single projection | 4+ scenarios (baseline, restricted, high, zero) | Built-in uncertainty range |
| **Implementation** | Excel | Python/pandas | Reproducibility, version control, testing |
| **Validation** | Expert judgment | Automated multi-level checks | Systematic vs. ad hoc |

---

## 4. Core Algorithm Comparison

### 4.1 SDC 5-Year Cohort-Component Step

The SDC engine advances the entire population by one 5-year interval. The following pseudocode is derived directly from the Excel workbook formulas documented in `METHODOLOGY_SPEC.md`:

```
FUNCTION sdc_project_one_period(Pop[age, sex, county], period_end_year):
    multiplier = PERIOD_MULTIPLIERS[period_end_year]

    FOR EACH county IN 53 counties:
        FOR EACH sex IN {Male, Female}:

            # --- BIRTHS (age group 0-4) ---
            IF sex == Female:
                total_births = 0
                FOR age_group IN {15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49}:
                    total_births += Pop[age_group, Female, county]
                                    * FertilityRate[age_group, county]
                total_births *= 5                    # 5 years of births
                male_births   = total_births * 0.512
                female_births = total_births * 0.488

            births_this_sex = male_births IF sex==Male ELSE female_births
            NewPop[0-4] = births_this_sex * SurvivalRate[0-4, sex]

            # --- AGING + SURVIVAL (age groups 5-9 through 80-84) ---
            FOR i IN 1..16:        # age groups 5-9 through 80-84
                source_group = AGE_GROUPS[i-1]     # e.g., 0-4 for target 5-9
                target_group = AGE_GROUPS[i]       # e.g., 5-9
                NatGrow[target_group] = Pop[source_group, sex, county]
                                        * SurvivalRate[source_group, sex]

            # --- 85+ OPEN-ENDED GROUP ---
            NatGrow[85+] = Pop[80-84, sex, county] * SurvivalRate[80-84, sex]
                         + Pop[85+,   sex, county] * SurvivalRate[85+, sex]

            # --- MIGRATION ---
            FOR EACH age_group IN all 18 groups:
                Migration[age_group] = NatGrow[age_group]
                                       * MigRate[age_group, sex, county]
                                       * multiplier

            # --- MANUAL ADJUSTMENTS ---
            FOR EACH age_group IN all 18 groups:
                Adj[age_group] = Adjustments[age_group, sex, county, period]

            # --- FINAL POPULATION ---
            FOR EACH age_group:
                NewPop[age_group] = NatGrow[age_group]
                                  + Migration[age_group]
                                  + Adj[age_group]
                NewPop[age_group] = MAX(0, NewPop[age_group])

    RETURN NewPop
```

**Key formula (from SDC workbook cell references):**

```
2030_Pro[age+5, sex, county] =
    (2025_Pro[age, sex, county] * Survival[age, sex])          -- natural growth
  + (2025_Pro[age, sex, county] * Survival[age, sex])          -- natural growth
    * Mig_Rate[age, sex, county] * 0.6                         -- migration
  + Adjustments[age, sex, county, 2025-2030]                   -- manual
```

Simplified:

```
Pop_new = NatGrow * (1 + MigRate * multiplier) + Adjustment
```

### 4.2 Current 2026 Annual Cohort-Component Step

Our engine advances every single-year-of-age cohort by exactly one year. The following pseudocode reflects the actual implementation in `cohort_component.py` and its component modules:

```
FUNCTION project_single_year(Pop[year, age, sex, race], year, scenario):

    # --- RATE PREPARATION ---
    surv_rates  = get_survival_rates(year)     # may be time-varying
    fert_rates  = fertility_rates.copy()
    mig_rates   = get_migration_rates(year)    # convergence-interpolated

    IF scenario is not None:
        fert_rates = apply_fertility_scenario(fert_rates, scenario, year)
        mig_rates  = apply_migration_scenario(mig_rates, scenario, year)

    # --- STEP 1: SURVIVAL (aging + mortality) ---
    # Regular ages (0 through 89):
    FOR EACH cohort (age, sex, race) WHERE age < 90:
        Survived[age+1, sex, race] = Pop[age, sex, race]
                                     * surv_rates[age, sex, race]

    # Mortality improvement (applied to rates before use):
    #   death_rate_t = death_rate_base * (1 - 0.005)^(year - base_year)
    #   survival_rate_t = 1 - death_rate_t
    #   Capped at 1.0

    # Open-ended age group (90+):
    Survived[90, sex, race] += Pop[90, sex, race]
                               * surv_rates[90, sex, race]
    # (survivors from age 89 already placed at age 90 by the loop above)

    # --- STEP 2: BIRTHS ---
    FOR EACH race:
        fertile_females = Pop[15..49, Female, race]
        births_race = SUM( fertile_females[age] * fert_rates[age, race] )

        Births[0, Male, race]   = births_race * 0.51
        Births[0, Female, race] = births_race * 0.49

    # --- STEP 3: MIGRATION ---
    FOR EACH cohort (age, sex, race):
        IF mig_rates has 'migration_rate' column:
            net_mig = Survived[age, sex, race] * mig_rates[age, sex, race]
        ELSE:  # absolute net_migration
            net_mig = mig_rates[age, sex, race]

        Migrated[age, sex, race] = Survived[age, sex, race] + net_mig
        Migrated[age, sex, race] = MAX(0, Migrated[age, sex, race])

    # --- STEP 4: COMBINE ---
    NewPop = CONCAT(Migrated, Births)
    NewPop = GROUP_SUM(NewPop, by=[year, age, sex, race])

    # --- STEP 5: VALIDATION ---
    IF total(NewPop) <= 0: RAISE ERROR
    IF any(NewPop < 0):    SET to 0, LOG WARNING

    NewPop['year'] = year + 1
    RETURN NewPop
```

**Key formula (mathematical notation from ADR-004):**

```
P(a+1, s, r, t+1) = P(a, s, r, t) * S(a, s, r, t) + M(a+1, s, r, t)
B(0, s, r, t+1)   = [ SUM_a P(a, Female, r, t) * F(a, r, t) ] * SRB(s)
```

Where:
- `P` = population, `S` = survival rate, `M` = net migration, `F` = fertility rate
- `SRB` = sex ratio at birth (0.51 male, 0.49 female)
- `a` = single-year age, `s` = sex, `r` = race/ethnicity, `t` = calendar year

### 4.3 Side-by-Side Comparison of the Core Step

| Aspect | SDC 5-Year Step | 2026 Annual Step |
|--------|-----------------|------------------|
| **Aging** | Entire age group shifts up by one group (0--4 becomes 5--9) | Each single-year cohort advances by exactly 1 year (age 17 becomes 18) |
| **Survival** | `Pop[a] * S_5yr[a, sex]` -- 5-year probability | `Pop[a] * S_1yr[a, sex, race]` -- annual probability |
| **Births** | `SUM(FemPop[ag] * ASFR[ag]) * 5` -- 5-year birth accumulation, then infant survival | `SUM(FemPop[a] * ASFR[a, race])` -- annual births, age 0 cohort created directly |
| **Migration timing** | Applied to natural growth: `NatGrow * rate * multiplier` | Applied to post-survival population: `Survived + net_migration` |
| **Migration modulation** | Global period multiplier (one scalar per 5-year period) | Per-cell time-varying rates via convergence interpolation + scenario adjustments |
| **Manual adjustments** | Explicit additive term per cell per period | None; all adjustments are algorithmic |
| **Open-ended group** | 85+: two-source accumulation (80--84 survivors + 85+ survivors) | 90+: same logic at age 90 (89 survivors flow in + 90+ survivors remain) |
| **Negative pop guard** | Not explicitly documented | `MAX(0, population)` with logged warning at both migration and final steps |

---

## 5. Implications of Structural Differences

### 5.1 Annual vs. 5-Year Time Stepping

**Numerical precision.** A 5-year step applies a single compound survival/migration transformation over 5 years, while annual stepping applies 5 successive transformations. These are mathematically equivalent only if rates are perfectly constant and interactions between components are negligible. In practice:

- **Compound interaction effects**: When migration adds (or removes) people during a 5-year window, those migrants themselves experience fertility and mortality within that window. Annual stepping captures this interaction each year. The SDC 5-year step ignores it -- births are computed from the population at the *start* of the period, and migration is applied to natural growth that also uses start-of-period population. This creates a minor systematic error whose magnitude depends on the migration rate and the age structure of migrants.

- **Sensitivity to volatility**: If migration rates change significantly within a 5-year window (e.g., COVID-19 causing near-zero migration in 2020--2021 followed by recovery), a 5-year step cannot represent this. The SDC handles this crudely with period-specific multipliers (the 2020--2025 multiplier of 0.2 is an attempt to average over the COVID disruption). Annual stepping naturally incorporates year-to-year rate variation when time-varying rates are provided.

- **Practical impact estimate**: For North Dakota's population scale (~800,000), the compound interaction error from 5-year stepping is on the order of hundreds of persons per period -- small relative to the uncertainty in migration assumptions, but not negligible for county-level projections of small counties.

### 5.2 Single-Year Age vs. 5-Year Age Groups

**Age-specific rate application.** Demographic rates -- fertility, mortality, and especially migration -- vary sharply within 5-year age bands. Key examples:

- **College-age migration**: The SDC's 15--19 and 20--24 age groups are the most volatile for migration. In reality, migration patterns differ dramatically between age 18 (college entry, strong in-migration to university counties) and age 22 (post-graduation, strong out-migration). The SDC lumps ages 15--19 together, which averages high-school students (low mobility) with college freshmen (high mobility). Our single-year engine can assign distinct migration rates to ages 15, 16, 17, 18, and 19.

- **Peak fertility**: The SDC's ASFR for the 25--29 age group applies a single rate across all five years. In reality, fertility peaks sharply around ages 27--28 and declines by age 29. Single-year rates capture this peak precisely.

- **Mortality gradient at old ages**: Between ages 80 and 84, annual mortality roughly doubles. A single 5-year survival rate for the 80--84 group averages over this steep gradient, slightly underestimating deaths among the oldest members and overestimating deaths among the youngest.

**Age-distribution artifacts.** Five-year groups force a uniform distribution assumption within each group. When the SDC computes births from the female 20--24 population, it implicitly assumes that women are uniformly distributed across ages 20--24. If a county has a university (e.g., Grand Forks), the 20--24 population is concentrated at ages 18--22, not uniformly distributed. Single-year ages eliminate this distortion entirely.

**Cohort tracking.** Annual stepping allows tracking the exact trajectory of a birth cohort as it ages through the projection. A researcher can follow the 2025 birth cohort at ages 0, 1, 2, ... 30 through the projection. With 5-year stepping, the finest resolution is observing cohorts at 5-year intervals, and within-interval dynamics are invisible.

### 5.3 Race/Ethnicity Detail

The SDC projects total population without racial or ethnic disaggregation. Our engine projects 6 race/ethnicity categories independently, with race-specific fertility rates, survival rates, and migration patterns.

**What race detail adds:**

- **Differential fertility**: Total fertility rates differ markedly by race in North Dakota. AIAN women have higher fertility than White NH women; Hispanic fertility is also above the state average. Projecting a single blended rate misses compositional changes as the racial mix shifts.

- **Differential mortality**: Life expectancy varies by race. AIAN populations have significantly lower life expectancy than White populations. A single survival rate for the total population overestimates survival for AIAN cohorts and underestimates it for White cohorts.

- **Differential migration**: Migration patterns are race-specific. The AIAN population on reservation counties (Benson, Sioux, Rolette) has distinct migration dynamics from the White population in those same counties. Our engine models these separately; the SDC cannot.

- **Policy relevance**: State agencies, tribal governments, health departments, and school districts need projections by race for service planning, resource allocation, and equity analysis. The SDC projection cannot serve these needs.

**Dimensional cost:** The race dimension multiplies the total cell count by 6, from ~1,900 to ~58,000 per year. This is computationally trivial in Python/pandas but would be unwieldy in a spreadsheet.

### 5.4 Reproducibility: Excel vs. Python

| Criterion | SDC Excel Workbook | Python Engine |
|-----------|-------------------|---------------|
| **Version control** | Binary `.xlsx` files; diffs are opaque | Plain-text `.py` files in git; line-level diffs |
| **Reproducibility** | Depends on the exact file; cell references can be broken by structural edits | Deterministic given the same config and input data |
| **Testing** | Manual inspection; no automated test suite | 1,165+ automated tests with pytest |
| **Auditability** | Requires opening the workbook and tracing cell references across 45 sheets | Read the source code; every formula is explicit |
| **Collaboration** | Single-user editing; merge conflicts intractable | Standard git workflow; multiple contributors |
| **Error detection** | Circular references and broken links are common Excel failure modes | Type checking (mypy), linting (ruff), and runtime validation |
| **Scalability** | Adding race would require multiplying sheets by 6; adding counties requires structural edits | Adding dimensions requires only data changes; code handles arbitrary shapes |
| **Documentation** | Embedded in cell comments (if present) | ADRs, docstrings, inline comments, and AGENTS.md |

A critical illustration of the reproducibility gap: the SDC's own workbook produces totals that differ from their published projections by 30,000--49,000 persons per year. This gap remains unexplained and cannot be audited because the reconciliation step is not captured in the workbook. In our Python system, every intermediate result is logged, testable, and traceable from input data through published output.

### 5.5 Open-Ended Age Group: 85+ vs. 90+

The SDC's 85+ terminal group collapses ages 85 through 100+ into a single bin. Our 90+ terminal group provides five additional years of age resolution (85, 86, 87, 88, 89 as individual cohorts).

This matters because:

- **Mortality acceleration**: Death rates roughly double every 7--8 years at advanced ages (Gompertz law). The survival rate for an 85-year-old is markedly different from that of a 95-year-old. The SDC's single 85+ survival rate (0.5428 for males, 0.6998 for females) is a weighted average that becomes less representative as the composition of the 85+ group shifts over time.

- **Health care planning**: Long-term care needs differ substantially between ages 85 and 95. Projecting them as a single group limits the utility of the projections for health care planners.

- **Growing population**: The 85+ population is the fastest-growing segment in many states. More granular age detail within this group improves projection accuracy for service planning.

---

## 6. Rationale for Changes

### 6.1 Annual Time Stepping

**Why we chose it:** Annual stepping is the standard used by the U.S. Census Bureau, the UN Population Division, and essentially all modern state demographic offices. It eliminates the compound interaction error inherent in multi-year steps, allows direct integration with annual data sources (IRS migration flows, annual vital statistics), and produces year-by-year output that aligns with planning cycles (annual budgets, enrollment forecasts).

**What we gave up:** Simplicity. A 5-year model has only 6 iterations for a 30-year projection; ours has 30. The computational cost is trivial (under 30 seconds for 53 counties), but the conceptual complexity of time-varying rates, year-specific survival tables, and convergence schedules is substantially greater than a static-rate 5-year model.

### 6.2 Single-Year Ages

**Why we chose it:** Single-year-of-age data is available from the Census Bureau's Population Estimates Program (PEP) and the decennial census. Using it eliminates the within-group uniform distribution assumption and allows precise application of age-specific rates. It is essential for modeling college-age migration (the single most consequential age-specific migration dynamic in North Dakota) and for producing output useful to school enrollment forecasters (who need age-specific cohorts).

**What we gave up:** Data smoothness. Single-year populations for small counties can be noisy (e.g., 3 people at age 47 in Slope County). We mitigate this through rate smoothing, convergence interpolation, and the age-aware rate cap in the migration pipeline (ADR-043).

### 6.3 Race/Ethnicity Detail

**Why we chose it:** North Dakota's demographic composition is changing. The AIAN population (concentrated on reservations), the Hispanic population (growing from immigration and higher fertility), and the Asian population (refugees, university enrollment) each have distinct demographic profiles. Projecting total population only -- as the SDC does -- implicitly assumes the current racial composition will persist. It cannot reveal compositional shifts that are policy-relevant.

Our 6 categories follow the Census Bureau / SEER standard taxonomy, ensuring compatibility with federal data sources.

**What we gave up:** Statistical power per cell. With 6 race categories, each county x age x sex x race cell has a smaller population, increasing the variance of rate estimates. We address this through:

- County-specific age-sex-race distributions built from Census full-count data (ADR-044), providing 216 populated cells per county (vs. 115 from the earlier PUMS approach).
- Blending small-county distributions with statewide patterns below a population threshold (5,000).
- Rate caps that clip extreme small-cell rates (ADR-043).

### 6.4 Python Implementation

**Why we chose it:** Excel is not an appropriate platform for a production demographic projection system. The SDC workbook's 45-sheet, 1,908-cell-per-point structure is at the limit of what can be maintained manually. Adding race would require 270,000+ cells per point, making the workbook intractable. Python provides:

- **Version control**: Every change is tracked in git.
- **Automated testing**: 1,165+ tests verify correctness after every change.
- **Modularity**: Components can be developed, tested, and modified independently.
- **Scalability**: The same engine handles state, county, and place projections without structural changes.
- **Reproducibility**: Given identical inputs and configuration, the engine produces identical outputs. No cell-reference fragility, no hidden manual edits.

**What we gave up:** Accessibility to non-programmers. The SDC's Excel workbook can be opened and inspected by anyone with Microsoft Office. Our Python system requires programming literacy. We mitigate this by producing Excel output workbooks, comprehensive documentation (ADRs, methodology notes), and visualization pipelines that present results in accessible formats.

### 6.5 Algorithmic Adjustments (Replacing Manual Adjustments)

**Why we chose it:** The SDC applies approximately 32,000 manual person-adjustments per 5-year period. These adjustments are documented only in the workbook's "Adjustments" sheets, without published rationale for individual cell values. This creates several problems:

- **Non-reproducibility**: Another demographer cannot replicate the adjustments without access to the exact workbook and the judgment of the original analyst.
- **Opacity**: Users of the projections cannot assess how much of the result comes from the model versus manual overrides.
- **Maintenance burden**: Every projection update requires re-evaluating thousands of manual adjustments.

Our system replaces manual adjustments with algorithmic mechanisms:

| SDC Manual Adjustment | Our Algorithmic Equivalent |
|-----------------------|---------------------------|
| College-age population corrections | College-age smoothing in convergence pipeline (ADR-049); age-aware rate caps with 15% threshold for ages 15--24 (ADR-043) |
| Bakken region sex-ratio balancing | Male dampening factor (0.80) applied algorithmically to boom-period migration rates |
| Regional economic adjustments | Oil-county dampening with period-specific factors (ADR-051); convergence interpolation naturally pulls extreme rates toward long-term means |
| Reservation county corrections | PEP-anchored recalibration with Rogers-Castro fallback for near-zero cells (ADR-045) |

Every algorithmic adjustment is documented in an ADR, configured in YAML, and tested in the automated test suite.

---

## Appendix A: Numerical Example -- What Annual Stepping Captures That 5-Year Stepping Misses

Consider a 20-year-old male migrating to Cass County (NDSU) and a 24-year-old male migrating out.

**SDC 5-year model**: Both are in the 20--24 age group. The net migration rate for 20--24 males in Cass County is a single number. If in-migration of 20-year-olds and out-migration of 24-year-olds partially cancel, the net rate understates both flows. The composition of the 20--24 group at the end of the period is unknown.

**Our annual model**: Age 20 has a high positive migration rate (college enrollment). Age 24 has a negative migration rate (post-graduation departure). These distinct rates apply independently. After 5 annual steps, ages 20--24 reflect the actual enrollment-and-departure dynamic rather than a blended net rate.

This matters because the SDC model's single net rate for 20--24 cannot distinguish between:
- A county where 500 people move in at age 20 and 400 leave at age 24 (net +100, high turnover)
- A county where 100 people move in across all ages 20--24 (net +100, low turnover)

These two scenarios have identical 5-year net migration but produce different population structures, different birth projections (women at age 20 vs. 24 have different fertility), and different downstream migration patterns.

## Appendix B: Cohort Cell Count Comparison

| Dimension | SDC 2024 | Current 2026 | Expansion Factor |
|-----------|----------|--------------|------------------|
| Ages | 18 groups | 91 single years | 5.1x |
| Sexes | 2 | 2 | 1x |
| Races | 1 (total only) | 6 | 6x |
| **Cohorts per county** | **36** | **1,092** | **30.3x** |
| Counties | 53 | 53 | 1x |
| **Total cells per point** | **1,908** | **57,876** | **30.3x** |
| Projection points | 6 | 30 | 5x |
| **Total data volume** | **11,448** | **1,736,280** | **151.7x** |

Despite this ~152x expansion in data volume, the Python engine completes a full 30-year, 53-county projection in under 30 seconds on commodity hardware. The SDC workbook's Excel recalculation time is comparable for its smaller dataset.

## Appendix C: SDC Workbook Survival Rates (Reference)

Extracted from `Projections_Base_2023.xlsx`, "5-Year Survival Rate By Sex" sheet:

| Age Group | Male S(5yr) | Female S(5yr) |
|-----------|-------------|---------------|
| 0--4 | 0.9915 | 0.9946 |
| 5--9 | 0.9994 | 0.9994 |
| 10--14 | 0.9987 | 0.9994 |
| 15--19 | 0.9982 | 0.9983 |
| 20--24 | 0.9949 | 0.9980 |
| 25--29 | 0.9927 | 0.9972 |
| 30--34 | 0.9897 | 0.9961 |
| 35--39 | 0.9876 | 0.9950 |
| 40--44 | 0.9860 | 0.9936 |
| 45--49 | 0.9808 | 0.9914 |
| 50--54 | 0.9776 | 0.9878 |
| 55--59 | 0.9652 | 0.9817 |
| 60--64 | 0.9521 | 0.9725 |
| 65--69 | 0.9335 | 0.9593 |
| 70--74 | 0.8972 | 0.9405 |
| 75--79 | 0.8355 | 0.9092 |
| 80--84 | 0.7353 | 0.8525 |
| 85+ | 0.5428 | 0.6998 |

Our engine uses single-year survival rates derived from the same CDC life table tradition. The approximate annual-equivalent for a 5-year survival rate is `S_1yr = S_5yr^(1/5)`. For example, the SDC's male 80--84 rate of 0.7353 corresponds to an approximate annual rate of `0.7353^0.2 = 0.9401`, but our engine applies age-specific annual rates rather than this group average.

## Appendix D: Migration Rate Time-Variation Comparison

**SDC approach** -- one scalar multiplier per 5-year period, applied globally to all cells:

```
Effective_Rate[age, sex, county, period] =
    Base_Rate[age, sex, county] * Period_Multiplier[period]

Period_Multipliers: {2025: 0.2, 2030: 0.6, 2035: 0.6, 2040: 0.5, 2045: 0.7, 2050: 0.7}
```

Every cell in the state gets the same percentage of its base rate in any given period. A county with strong in-migration at age 20 and a county with out-migration at age 70 both have their rates scaled by the same factor.

**Our approach** -- per-cell convergence interpolation over 30 annual steps:

```
FOR EACH cell (county, age_group, sex):
    recent_rate  = average(migration_rate for 2023-2025)
    medium_rate  = average(migration_rate for 2014-2025)
    longterm_rate = average(migration_rate for 2000-2025)

    FOR year_offset IN 1..30:
        IF year_offset <= 5:     # years 1-5
            rate = LERP(recent_rate, medium_rate, year_offset/5)
        ELIF year_offset <= 15:  # years 6-15
            rate = medium_rate
        ELSE:                    # years 16-20 (and hold beyond)
            progress = (year_offset - 15) / 5
            rate = LERP(medium_rate, longterm_rate, MIN(progress, 1.0))

        rate = CLIP(rate, -cap, +cap)   # age-aware: 15% for ages 15-24, 8% otherwise
        convergence_rates[cell, year_offset] = rate
```

Each cell has its own trajectory. A county experiencing recent in-migration boom will see its rate converge toward its long-term mean independently of other counties. This eliminates the need for a blunt global multiplier like the SDC's Bakken dampening.
