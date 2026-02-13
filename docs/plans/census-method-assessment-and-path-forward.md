---
title: "Census Bureau Methodology Assessment & Path Forward"
created: 2026-02-13T14:00:00-06:00
status: findings-document
author: Claude Code (Opus 4.6)
purpose: >
  Comprehensive assessment of Census Bureau cohort-component methodology
  vs. our current implementation, with identified gaps, simplification flags,
  and recommended path forward. Produced at user request before implementing
  convergence interpolation, to ensure methodological fidelity.
context: >
  User directed: focus on Census Bureau methods as primary (not BEBR),
  retain 60% oil county dampening, avoid over-simplification, flag every
  departure from Census Bureau or SDC 2024 practices for review.
related_adrs:
  - ADR-035 (Census PEP Data Source)
  - ADR-036 (Migration Averaging Methodology)
related_plans:
  - docs/plans/convergence-interpolation-scaling-factors.md (superseded by this assessment)
census_documents_reviewed:
  - methodstatement23.pdf (2023 National Population Projections methodology)
  - sptoolkitusersguide.pdf (Subnational Projections Toolkit v2.0, Nov 2014)
  - methods-statement-v2023.pdf (Population Estimates methodology, Dec 2023)
  - np2023_a4_survival_ratios.csv (age-specific survival ratios, 2023-2100)
  - np2023_a3_life_expectancy.csv (life expectancy projections, 2023-2100)
  - np2023_a1_fertility_rates.csv (age-specific fertility rates, 2023-2100)
---

# Census Bureau Methodology Assessment & Path Forward

## 1. Executive Summary

After detailed review of three Census Bureau methodology documents, the Census Bureau's
published projection data files, the SDC 2024 source files, and our current codebase,
this document identifies six specific areas where our implementation departs from
Census Bureau and/or SDC 2024 practices. It recommends a phased implementation path
("Path B") that addresses these departures in order of impact, starting with the most
significant: switching from model-based to empirical age-specific migration rates.

**Key finding:** There is no single "Census Bureau migration convergence algorithm" to
replicate. The Census Bureau treats migration differently across its products. Our
convergence interpolation is a defensible design informed by Census Bureau State Interim
Projections and UK ONS practices — but it should be documented as our own methodological
choice, not presented as a Census Bureau prescription.

---

## 2. What the Census Bureau Actually Does

### 2.1 Migration — No Universal Convergence Method

| Census Bureau Product | Migration Treatment |
|---|---|
| **National Projections 2023** (methodstatement23.pdf) | Foreign-born emigration rates: **held constant** for all projection years. Net native migration: **held constant**. Immigration: power function producing smoothly changing levels. **No convergence model for migration.** |
| **Subnational Projections Toolkit** (sptoolkitusersguide.pdf) | Migration is entirely **user-provided**. The toolkit implements convergence for fertility (logistic curve to TFR target) and mortality (logistic curve to e0 target), but **not for migration**. The MIGSUB tool takes user-specified migration volume assumptions and distributes them by age/sex using reference-period patterns. |
| **State Interim Projections (2004-2030)** | The **only Census product** with migration convergence: ARIMA models for first 5 years, then linear interpolation toward long-term series mean, then held constant. Based on 25 years of IRS data at the **state** level. This product is from 2004 and has not been updated. |
| **Population Estimates Program** (methods-statement-v2023.pdf) | Estimates (not projections). Computes county-level NDM rates from IRS (ages 0-64) and Medicare (ages 65+) data. Rates are age-specific via characteristic-specific out-rates and in-proportions. |

**Implication:** Our 3-phase convergence schedule (recent → medium → long-term) is best
described as *inspired by the Census Bureau State Interim Projections and UK ONS practices*.
It is a defensible methodological choice, but not a direct replication of any current Census
Bureau procedure.

### 2.2 Fertility — Linear Convergence to Common Target

The Census Bureau (methodstatement23.pdf, p. 5) linearly converges all six nativity/race
fertility groups toward a common TFR of **1.52** (native White 2020 level) by **2123**.
Age-specific fertility rates are linearly interpolated from 2021 values toward the 2123
target. Over our 20-year horizon (2025-2045), this produces minimal change: total TFR
moves from ~1.64 to ~1.60.

### 2.3 Mortality — Log-Linear Convergence to UN Model Life Tables

The Census Bureau (methodstatement23.pdf, p. 7) uses a three-step process:
1. Project e0 to 2123 by extrapolating log(100 - e0). Targets: male 87, female 91.
2. Select UN Model Life Tables matching those target e0 values.
3. Interpolate between 2019 observed age-specific death rates and target rates **on the
   natural log scale** (producing non-linear convergence: faster improvement initially,
   decelerating over time).

**Published data available:** `np2023_a4_survival_ratios.csv` contains Census Bureau
projected survival ratios by single year of age (0-100), sex, race/ethnicity group, and
year (2023-2100). These can be used directly for our 2025-2045 projection window rather
than inventing our own improvement schedule.

### 2.4 SDC 2024 Approach (Comparison Baseline)

From SDC source file analysis (Projections_Base_2023.xlsx):
- **Age-specific rates**: Empirical residual rates by 18 five-year age groups × 2 sexes
  per county, computed from Census 2000/2010/2020 populations and CDC survival rates.
- **Time periods**: Four 5-year periods averaged (2000-2005, 2005-2010, 2010-2015, 2015-2020).
- **Dampening**: Rates reduced to ~60% for Bakken oil counties; additional manual adjustments
  for college-age populations and male migration imbalance.
- **Application**: Constant rates — same table every 5-year projection period. No convergence.
- **Mortality**: Constant survival rates (no improvement visible in source files).

---

## 3. Simplification Audit: Departures from Census Bureau & SDC 2024

Each flag is graded by impact on projection outcomes and assigned a recommended action.

### FLAG 1 — Rogers-Castro Model vs. Empirical Age-Specific Rates
**Severity: HIGH | Action: FIX (Path B, Phase 1)**

| | SDC 2024 | Census Bureau PEP | Our Current Approach |
|---|---|---|---|
| Method | Residual: Pop(t+5) - Survived(t) by age group | Out-rates and in-proportions from IRS/Medicare micro data by age/sex/race | Total county migration distributed by Rogers-Castro model curve |
| Age detail | 18 five-year age groups × 2 sexes, per county | 3 broad age groups (totals); full age/sex/race (characteristics) | Single generic bell curve peaked at age 25, applied uniformly to all counties |

**Why it matters:** The SDC 2024 data shows fundamentally different age patterns across
counties. For example, at the state level: ages 20-24 males show +0.328 in-migration rate,
but ages 25-29 females show -0.243 out-migration. Rogers-Castro cannot capture these real
county-specific and sex-specific patterns — it produces a symmetric, generic curve.

**Data requirements for fix:** We need Census population by 5-year age groups at county
level for at least two time points per period (e.g., 2010, 2015, 2020, 2024). The SDC
used Census 2000, 2010, 2020, and interpolated 2005/2015. We have partial data:
- `cc-est2019-agesex-38.xlsx` — 5-year age groups by county, 2010-2019
- Census 2020 data (available via Census API or from existing downloads)
- Census 2000 county age data (partially available in SDC reference files)

### FLAG 2 — Linear Interpolation vs. ARIMA for Near-Term Migration
**Severity: LOW | Action: DOCUMENT (accept simplification)**

The Census Bureau State Interim Projections used ARIMA for the first 5 projection years.
We use linear interpolation.

**Why we accept this simplification:**
- ARIMA requires sufficient data for parameter estimation. With 25 annual data points per
  county and extreme volatility (e.g., Williams County losing 2,665 people in a single year),
  ARIMA models would likely produce unstable and overfitted near-term forecasts.
- The Census Bureau used ARIMA at the **state** level with 25 years of data, where migration
  series are much smoother. County-level volatility is qualitatively different.
- The UK ONS explicitly uses linear interpolation for this same step in their subnational
  projections, providing published precedent for the simpler approach.
- For our 20-year horizon, the near-term transition covers years 1-5 only. Linear
  interpolation and ARIMA would produce similar results for the remaining 15 years.

**Sensitivity test recommended:** Compare linear interpolation vs. exponential smoothing
(a lightweight alternative to ARIMA) for the near-term phase to quantify the impact.

### FLAG 3 — Convergence Schedule is Our Design Choice
**Severity: MEDIUM | Action: DOCUMENT + SENSITIVITY ANALYSIS**

Our 3-phase schedule (5 years recent→medium, 10 years hold at medium, 5 years medium→
long-term) is not prescribed by any Census Bureau document. It is our own design, informed
by:
- Census Bureau State Interim Projections: ARIMA (5 years) → convergence → held constant
- UK ONS: linear interpolation from recent to 10-year average for near-term
- Statistics Canada: linear interpolation from recent rates toward long-term average over
  first 10 years, held constant thereafter

The specific timing parameters (5-10-5) are assumptions. Alternative schedules (3-10-7,
7-8-5, 3-7-10, etc.) would produce different results.

**Action:** Run sensitivity analysis varying the convergence schedule. Document the
schedule choice and rationale in an ADR. Present schedule parameters as configurable
(which they already are in `projection_config.yaml`).

### FLAG 4 — Uniform Scaling Factor vs. Age-Specific Convergence
**Severity: MEDIUM (becomes HIGH if Flag 1 is fixed) | Action: FIX (Path B, Phase 2)**

The current plan applies a single scalar factor to all age/sex/race groups per county
per year. This is mathematically equivalent to age-specific convergence ONLY when the
age distribution is model-based (Rogers-Castro) — because the same model generates
both the baseline and the convergence target.

**Once we switch to empirical age-specific rates (Flag 1)**, each age group's migration
rate should converge independently toward its own long-term mean. A uniform scalar would
no longer be appropriate because:
- Age 20-24 migration may be volatile (college/military) while age 65+ migration is stable
- The convergence rate should reflect each age group's actual historical variance
- Different age groups may converge to long-term means of different signs

**Action:** Implement age-specific convergence alongside Flag 1.

### FLAG 5 — Mortality: Constant Rates vs. Census Bureau Improvement
**Severity: MEDIUM | Action: FIX (Path B, Phase 3)**

| | Census Bureau 2023 | SDC 2024 | Our Current |
|---|---|---|---|
| Method | Log-linear convergence to UN Model Life Table targets | Constant (no improvement) | Config says 0.5%/year, **not implemented** |
| Published data | `np2023_a4_survival_ratios.csv`: survival ratios by single year of age, sex, race, 2023-2100 | CDC 2020 life tables, held constant | SEER/CDC life tables, held constant |

**Recommended approach:** Use the Census Bureau's published survival ratio projections
directly. For our 2025-2045 window, extract the year-specific survival ratios from
`np2023_a4_survival_ratios.csv` and feed them into the engine as time-varying rates.
This is more faithful than inventing a flat 0.5%/year improvement factor.

**Data available:**
- 1,728 rows: 3 nativity groups × 3 sex categories × 7 race groups × ~78 years
- Single-year ages 0-100
- Years 2023-2100

For our purposes, we would use SEX=1 (Male) and SEX=2 (Female), GROUP=0 (All races)
or race-specific groups, for years 2025-2045. The survival ratios already incorporate
the Census Bureau's log-linear mortality improvement model.

**Life expectancy trajectory (from np2023_a3_life_expectancy.csv):**

| Year | Male e0 (All) | Female e0 (All) | Male e0 (NH White) | Female e0 (NH White) |
|------|-------------|---------------|-------------------|---------------------|
| 2025 | 77.7 | 82.5 | 78.1 | 82.6 |
| 2030 | 78.5 | 83.0 | 78.8 | 83.2 |
| 2035 | 79.2 | 83.6 | 79.5 | 83.7 |
| 2040 | 79.9 | 84.1 | 80.2 | 84.3 |
| 2045 | 80.6 | 84.7 | 80.8 | 84.8 |

Over our 20-year horizon, this represents ~3 years of life expectancy gain for males
and ~2.2 years for females — meaningful for a cohort projection.

**Note:** These are national rates. North Dakota life expectancy is close to the national
average (ND male ~76.5, female ~81.8 per our Census methodology reference), so national
projected improvement rates are a reasonable proxy. An adjustment factor could be applied
if ND-specific life tables diverge significantly. Flag for review.

### FLAG 6 — Fertility: Constant vs. Census Bureau Convergence
**Severity: LOW | Action: DOCUMENT (accept for now)**

The Census Bureau linearly converges all fertility groups toward TFR 1.52 by 2123. Over
our 20-year window, this moves total TFR from ~1.64 to ~1.60 — a change of ~2.4%. Given
that our ND-specific TFR (~1.73) already differs from the national TFR by more than this,
and that ND fertility trends may not follow the national convergence pattern, holding
fertility constant is a defensible simplification for a 20-year horizon.

**Census Bureau projected ASFRs are available** in `np2023_a1_fertility_rates.csv`
(single-year ages 14-54, by group, 2023-2100) if we decide to incorporate fertility
convergence in the future.

**Consistent with SDC 2024**, which also uses approximately constant fertility rates
(averaged over 2018-2022 period, applied to all projection periods).

---

## 4. Path B: Recommended Implementation

### Phase 1: Empirical Age-Specific Migration Rates (Fixes Flag 1)

**Objective:** Replace Rogers-Castro model-based age distribution with empirical
age-specific migration rates computed via the residual method, consistent with SDC 2024
methodology.

**Method:**
```
Migration_Rate(age, sex, county, period) =
  [Pop(age+5, sex, county, t+5) - Pop(age, sex, county, t) × Survival(age, sex, 5yr)] / Pop(age, sex, county, t)
```
Where the 5-year survival rate comes from CDC life tables (same source as SDC 2024).

**Data needed:**
- County population by 5-year age group and sex for: 2000, 2005, 2010, 2015, 2020, 2024
- Sources: Census 2000 (decennial), PEP cc-est for 2005-2019, Census 2020 (decennial),
  PEP for 2020-2024
- We have partial data; need to fetch remaining from Census API or archives

**Periods to compute:**
- 2000-2005, 2005-2010, 2010-2015, 2015-2020 (matching SDC 2024)
- Optionally 2020-2024 (using PEP estimates for recent period)

**Dampening:** Apply 60% factor to the 5 Bakken counties' rates AFTER computing residual
rates, BEFORE using them in convergence. This matches SDC 2024 practice.

**Output:** Age-specific migration rate tables per county per period:
`rate_table[county][period] → DataFrame[age_group, sex, migration_rate]`

### Phase 2: Age-Specific Convergence Interpolation (Fixes Flag 4)

**Objective:** Implement time-varying migration where each age-sex group's rate converges
independently from its recent value toward its long-term mean.

**Method:** For each county, age group, and sex:
```
rate(age, sex, year) = interpolate(
  recent = mean_rate(age, sex, 2020-2024),
  medium = mean_rate(age, sex, 2014-2024),
  longterm = mean_rate(age, sex, 2000-2024),
  schedule = {recent_to_medium: 5yrs, medium_hold: 10yrs, medium_to_longterm: 5yrs}
)
```

This replaces the uniform scaling factor approach. Each age-sex group converges at its
own trajectory. A group with volatile recent rates will show larger changes than a group
with stable rates.

**Engine changes:** Instead of `convergence_factors: dict[int, float]` (one scalar per
year), the engine receives year-specific rate tables:
`migration_rates: dict[int, pd.DataFrame]` (year_offset → full rate table).

This is more data than the scaling factor approach (20 rate tables per county vs. 20
floats), but each table is only ~36 rows (18 five-year age groups × 2 sexes) per county,
which is very manageable. Total: 20 years × 36 rows × 53 counties = ~38,160 rows.

### Phase 3: Census Bureau Mortality Improvement (Fixes Flag 5)

**Objective:** Replace the unimplemented 0.5%/year flat improvement with Census Bureau
projected survival ratios.

**Method:**
1. Load `np2023_a4_survival_ratios.csv` for years 2025-2045
2. Filter to SEX=1 (Male) and SEX=2 (Female), select appropriate race GROUP
3. Extract year-specific survival ratios by single year of age
4. Pass time-varying survival rates to the engine alongside time-varying migration rates

**Approach (Decision 2 — resolved):** Option B (ND-adjusted). Compute the ratio between
our current ND-specific CDC survival rates and the Census Bureau 2023 national rates.
Apply that ratio as an adjustment factor to the Census Bureau's projected rates for
2025-2045. This preserves ND's baseline mortality level while adopting the national
improvement trajectory from the Census Bureau's log-linear convergence model.

**Engine changes:** `self.survival_rates` becomes year-varying, similar to migration rates.
The `project_single_year()` method already copies survival rates before use (line 155),
so the change is: look up the year-specific table instead of always using the same one.

### Phase 4: Wire Everything Into the Engine

**Objective:** Modify the projection engine and pipeline to accept time-varying migration
and survival rates.

**Engine (`cohort_component.py`):**
- `__init__()` accepts `migration_rates_by_year: dict[int, pd.DataFrame] | None` and
  `survival_rates_by_year: dict[int, pd.DataFrame] | None`
- If time-varying rates are provided, `project_single_year()` selects the appropriate
  year's table. If not provided, falls back to constant rates (backward compatible).

**Multi-geography (`multi_geography.py`):**
- Thread year-indexed rate dicts through to each county's projection

**Pipeline (`02_run_projections.py`):**
- Load precomputed convergence rate tables and Census Bureau survival ratios
- Pass to projection engine

### Phase 5: Validation and Comparison

**Tests:**
- Unit tests for residual migration rate computation
- Unit tests for age-specific convergence interpolation
- Integration tests for time-varying engine behavior
- Backward compatibility: constant rates still produce identical results
- Comparison: our projections vs. SDC 2024 projections (should be closer now that we
  use empirical rates and dampening)

**Comparison outputs:**
- Side-by-side: constant rates (current) vs. convergence rates (new)
- Side-by-side: our projections vs. SDC 2024 for key counties
- Age profile comparison: Rogers-Castro vs. empirical residual rates for select counties

---

## 5. Implementation Order and Dependencies

```
Phase 1: Empirical age-specific rates
├── Fetch historical county population data (Census 2000, 2010, 2020, PEP 2005-2024)
├── Compute 5-year residual migration rates by age/sex/county/period
├── Apply 60% dampening to oil counties
├── Multi-period averaging (match SDC 2024: 4 periods averaged)
└── Tests: validate against SDC 2024 published rates

Phase 2: Age-specific convergence (depends on Phase 1)
├── Compute recent/medium/long-term averages per age-sex group per county
├── Implement age-specific interpolation (3-phase schedule)
├── Output: year-indexed rate tables per county
└── Tests: convergence schedule correctness, edge cases

Phase 3: Mortality improvement (independent of Phases 1-2)
├── Load Census Bureau survival ratio projections
├── Adjust for ND-specific baseline (Option B if chosen)
├── Output: year-indexed survival rate tables
└── Tests: survival ratio trends, ND adjustment factor

Phase 4: Engine wiring (depends on Phases 1-3)
├── Modify CohortComponentProjection for time-varying rates
├── Thread through multi_geography.py
├── Wire into 02_run_projections.py pipeline
└── Tests: engine produces correct time-varying behavior

Phase 5: Validation (depends on Phase 4)
├── Full pipeline run
├── Comparison to SDC 2024
├── Sensitivity analysis on convergence schedule
└── Documentation updates (ADRs, methodology reports)
```

Phase 3 (mortality) can proceed in parallel with Phases 1-2 since it has no dependencies
on migration rate changes.

---

## 6. Data Inventory

### What We Have

| Data | Location | Detail | Status |
|---|---|---|---|
| PEP county net migration 2000-2024 | `data/processed/pep_county_components_2000_2024.parquet` | Aggregate total per county/year | Ready |
| CDC/SEER survival rates | `data/raw/mortality/survival_rates_processed.csv` | Single-year age, sex, race | Ready |
| Census Bureau survival projections | `data/raw/census_bureau_methodology/np2023_a4_survival_ratios.csv` | Single-year age 0-100, sex, race, 2023-2100 | Ready |
| Census Bureau life expectancy | `data/raw/census_bureau_methodology/np2023_a3_life_expectancy.csv` | By sex, race, 2023-2100 | Ready |
| Census Bureau fertility projections | `data/raw/census_bureau_methodology/np2023_a1_fertility_rates.csv` | Single-year age 14-54, by group, 2023-2100 | Available if needed |
| PEP 2010-2019 by 5yr age groups | `data/raw/nd_sdc_2024_projections/source_files/reference/cc-est2019-agesex-38.xlsx` | 5-year age groups, county, sex | Ready |
| Census 2000 county age/sex | `data/raw/nd_sdc_2024_projections/source_files/reference/Census 2000 County Age and Sex.xlsx` | To verify | Needs check |
| 2025 base population | `data/processed/base_population.parquet` | Single-year age, sex, race, county | Ready |

### What We Need to Fetch

| Data | Source | Detail Needed | Purpose |
|---|---|---|---|
| Census 2020 county population by age/sex | Census API or data.census.gov | 5-year age groups minimum | Residual migration for 2015-2020 period |
| PEP 2020-2024 county population by age/sex | Census API (Vintage 2024 estimates) | 5-year age groups minimum | Residual migration for 2020-2024 period |

The SDC 2024 also interpolated population for 2005 and 2015 (they had Census 2000, 2010,
2020 and interpolated between). We may need to do the same, or use PEP intercensal
estimates if available.

---

## 7. Decisions — Resolved (2026-02-13)

### Decision 1: Convergence schedule — PROCEED WITH 5-10-5
Accept the 5-10-5 schedule as the initial implementation. Run sensitivity analysis after
implementation to quantify how timing affects results. Parameters are already configurable
in `projection_config.yaml`. Not blocking.

### Decision 2: Mortality — OPTION B (ND-ADJUSTED)
Compute the ratio between current ND-specific CDC survival rates and Census Bureau 2023
national rates. Apply that ratio as an adjustment factor to the Census Bureau's projected
rates for 2025-2045. This preserves North Dakota's baseline mortality level while adopting
the national improvement trajectory from the Census Bureau's log-linear convergence model.

### Decision 3: Age group resolution — 5-YEAR AGE GROUPS
Matches SDC 2024 methodology exactly. More statistically stable at county level. 18 groups
× 2 sexes = 36 rate cells per county. Directly comparable to SDC 2024 published rates.

### Decision 4: Fertility — HOLD CONSTANT
Over 20 years the Census Bureau convergence moves TFR from 1.64 to ~1.60 (2.4% change).
Our ND-specific TFR (~1.73) already differs from national by more than this. Consistent
with SDC 2024. Census Bureau fertility data (`np2023_a1_fertility_rates.csv`) is available
if we revisit.

### Decision 5: SDC adjustments — REPLICATE BOTH
Replicate both college-age and male migration adjustments from SDC 2024:

**College-age adjustment:** The residual method distorts migration for college counties
because students are counted as migrants when they arrive/leave. Counties to adjust:
- Cass County (NDSU, Concordia) — FIPS 38017
- Grand Forks County (UND) — FIPS 38035
- Ward County (Minot State) — FIPS 38101
- Burleigh County (U of Mary, Bismarck State) — FIPS 38015
- Possibly Richland (Wahpeton NDSCS), Stutsman (Jamestown), Rolette (Turtle Mountain CC)

The SDC did not fully document their specific college adjustments. Implementation approach:
identify anomalous age 15-24 in/out patterns in college counties and smooth or dampen them.
This requires further research into what the SDC actually did — flag for design phase.

**Male migration adjustment:** Reduce male migration rates more than female in boom-era
periods. The 2000-2020 average shows male net rate of +0.034 vs female -0.019 — a pattern
driven by oil field employment. The SDC reduced male rates further because this gender
imbalance was unlikely to continue. Implementation: apply a sex-specific dampening ratio
to boom-era periods (2005-2010, 2010-2015) in addition to the 60% county dampening.
Specific ratio needs research — flag for design phase.

### Decision 6: Dampening — BOOM-ERA PERIODS ONLY
Apply 60% dampening factor only to the periods that captured the oil boom: 2005-2010 and
2010-2015. The pre-boom (2000-2005), post-boom (2015-2020), and recent (2020-2024)
periods reflect non-boom conditions and should not be dampened.

**Rationale:** The purpose of dampening is to reduce the influence of the Bakken boom,
which was a specific historical event in a specific time window. Dampening all periods
equally (as SDC 2024 did) over-corrects by dampening periods that don't contain boom
effects. Period-specific dampening is more precise.

**Implementation:** In the residual rate computation, after computing rates for each
5-year period:
```
if period in [(2005, 2010), (2010, 2015)] and county in oil_counties:
    rate *= 0.60
```

This interacts with the male migration adjustment (Decision 5): the sex-specific
dampening also applies only to boom-era periods.

---

## 8. New Research Findings (2026-02-13, Post-Decision)

Deep dive into SDC 2024 source files and workbooks revealed several important
details not captured in the initial assessment:

### 8.1 SDC Dampening is Variable, Not Uniform 60%

Analysis of the SDC 2024 projection workbook (`Projections_Base_2023.xlsx`)
revealed **variable period multipliers** on projected migration, despite
the SDC public methodology report describing only "typically reduced to
about 60%." The specific multipliers, documented in our replication analysis
(`sdc_2024_replication/README.md`, `METHODOLOGY_SPEC.md`, ADR-017):

| Projection Period | Multiplier | Rationale |
|-------------------|------------|-----------|
| 2020-2025 | 0.2 | COVID + post-boom transition |
| 2025-2030 | 0.6 | Bakken dampening |
| 2030-2035 | 0.6 | Continued dampening |
| 2035-2040 | 0.5 | Further conservative reduction |
| 2040-2045 | 0.7 | Gradual return |

*Source: Extracted from SDC workbook in prior analysis session. The SDC public
report does not document these specific per-period values.*

This differs from our approach (Decision 6): we dampen the historical INPUT
periods at 60%, then use convergence interpolation for the time-varying
aspect. The SDC varied the multiplier on projection output. Our rationale for
dampening inputs rather than outputs: boom-era effects are a property of the
historical data, not a forward-looking projection assumption, so they should
be corrected at the source rather than adjusted ad-hoc across projection
periods.

### 8.2 SDC Adjustments Are Manual Expert Judgment

The college-age and male migration adjustments were NOT algorithmic. The SDC
methodology report states that college-age populations "typically required
additional adjustments" and male migration rates "were further reduced," but
does not quantify the scope of these edits.

Our replication analysis (`sdc_2024_replication/METHODOLOGY_SPEC.md`, ADR-017)
estimated approximately 32,000 total person-adjustments per 5-year projection
period based on examining the workbook's adjustment columns. *Note: this
figure lacks independent verification and should be treated as an order-of-
magnitude estimate, not a precise count.*

**Implication for our implementation:** We cannot precisely replicate SDC
adjustments. Our Phase 1 implementation includes algorithmic approximations
(`apply_college_age_adjustment()` and `apply_male_migration_dampening()`) that
need calibration against observed patterns. Parameters are flagged for review.

### 8.3 Data Inventory Confirmation

Confirmed available data for residual migration computation:
- **Census 2000**: Available in `Census 2000 County Age and Sex.xlsx` (all US
  counties, AGEGRP codes 0-18, SEX 0/1/2) — filter to STATE=38
- **PEP 2010-2019**: Available in `cc-est2019-agesex-38.xlsx` (53 ND counties,
  18 five-year age groups, male/female)
- **Census 2020 base**: Available in `base_population_by_county.csv` (53 counties,
  18 five-year groups, male/female)
- **SDC 2024 survival rates**: Available in `survival_rates_sdc_2024_by_age_group.csv`
  (36 rows: 18 age groups × 2 sexes, CDC ND 2020)
- **Census survival projections**: Available in `np2023_a4_survival_ratios.csv`
  (single-year 0-100, 2023-2100)

**PEP Vintage 2024 age-sex data**: ~~Initially reported as unavailable.~~
**CORRECTION (2026-02-13)**: `cc-est2024-agesex-all.csv` was published by
Census Bureau on June 26, 2025 at
`datasets/2020-2024/counties/asrh/cc-est2024-agesex-all.csv`. Downloaded and
archived as parquet at
`shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`
(18,864 rows, 96 columns, wide format). All 5 periods (2000-2024) now use
actual Census data. See `docs/plans/audit-implementation-plan-and-findings.md`
Section 2.3 and Section 7 for full correction history.

### 8.4 Implementation Plan Created

Detailed implementation plan with function signatures, test specifications,
and architecture diagrams created at:
`docs/plans/implementation-plan-census-method-upgrade.md`

---

## 10. Relationship to Prior Plan

The earlier plan document (`convergence-interpolation-scaling-factors.md`) proposed a
scaling-factor approach to wire convergence into the engine on top of BEBR trimmed-average
rates. That plan is **superseded** by this assessment. Key changes:

| Aspect | Prior Plan | This Assessment |
|---|---|---|
| Primary method | BEBR trimmed average with convergence modifier | Census Bureau convergence interpolation as standalone primary method |
| Age distribution | Rogers-Castro model (uniform scaling factor) | Empirical residual rates per age/sex/county (age-specific convergence) |
| Mortality | Deferred (flat 0.5%/year not implemented) | Census Bureau projected survival ratios (time-varying) |
| Dampening | Applied to BEBR scenarios | Applied to period averages before convergence |
| BEBR role | Primary baseline | Available as comparison/validation |

---

## 11. References

### Census Bureau Methodology Documents (Reviewed)
1. "Methodology, Assumptions, and Inputs for the 2023 National Population Projections" (methodstatement23.pdf)
2. "Subnational Population Projections Toolkit User's Guide, Version 2.0" (sptoolkitusersguide.pdf, November 2014)
3. "Methodology for the United States Population Estimates: Vintage 2023" (methods-statement-v2023.pdf, December 2023)

### Census Bureau Data Files (Available)
4. np2023_a4_survival_ratios.csv — Single-year age survival ratios, 2023-2100
5. np2023_a3_life_expectancy.csv — Life expectancy projections, 2023-2100
6. np2023_a1_fertility_rates.csv — Age-specific fertility rates, 2023-2100

### Best Practices Literature
7. Smith & Tayman (2003), "The relationship between the length of the base period and population forecast errors," *International Journal of Forecasting*
8. Migration Averaging Best Practices Review (docs/reports/migration_averaging_best_practices.md)

### SDC 2024 Comparison
9. Methodology Comparison: SDC 2024 (docs/methodology_comparison_sdc_2024.md)
10. SDC 2024 source files (data/raw/nd_sdc_2024_projections/source_files/)

### Internal Documentation
11. ADR-035: Census PEP Components of Change for Migration Inputs
12. ADR-036: Migration Averaging Methodology (Proposed)
13. Prior convergence plan: docs/plans/convergence-interpolation-scaling-factors.md
