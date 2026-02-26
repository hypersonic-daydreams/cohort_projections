# Methodology Document Writing Plan

**Purpose:** Direct sub-agents to write `docs/methodology.md` — a comprehensive methodology document with LaTeX formulas and plain-language explanations.

**Output:** Single markdown file at `docs/methodology.md`

---

## Document Structure (8 Sections)

### Section 1: Overview & Introduction
- Cohort-component method overview (age × sex × race × county)
- Geography: 53 ND counties, state = sum of counties (ADR-054)
- Base year 2025, horizon 30 years (2025-2055), annual steps
- 91 ages (0-90+) × 2 sexes × 6 race/ethnicity categories
- Three scenarios: Baseline, Restricted Growth, High Growth
- Key source file: `config/projection_config.yaml`

### Section 2: Base Population
**Source files to read:** `cohort_projections/data/load/base_population_loader.py`, `cohort_projections/utils/demographic_utils.py`

Topics:
- **Data source:** Census PEP Vintage 2025, SC-EST2024 single-year-of-age
- **Sprague osculatory interpolation (ADR-048):** Graduates 5-year age groups to single years
  - Coefficient matrix at `demographic_utils.py:243-249` (SPRAGUE_MULTIPLIERS, 5×5 matrix)
  - Boundary padding via linear extrapolation (`_pad_groups`)
  - Negative clamping + within-group renormalization
  - Formula: `single_year_values = SPRAGUE_MULTIPLIERS @ window` where window = 5 consecutive group totals
- **County-specific race distribution (ADR-047):** Population-weighted blend with statewide for small counties (threshold: 5,000)
- **Group quarters separation (ADR-055 Phase 1):**
  - GQ subtracted from base population before projection
  - GQ data from stcoreview (broad age → 5-year via allocation profiles, 50/50 sex split)
  - GQ held constant and re-added at each projection year
  - Source: `scripts/data/fetch_census_gq_data.py`

### Section 3: Fertility Component
**Source files to read:** `cohort_projections/core/fertility.py`, `cohort_projections/data/process/fertility_rates.py`

Topics:
- **ASFR source:** SEER/NVSS data, race-harmonized to 6 categories, multi-year averaged
- **ND adjustment (ADR-053):** ND TFR 1.863 vs national 1.621
- **Birth calculation formula:**
  - `B(r) = Σ_{a=15}^{49} F(a,r) × P_f(a,r)` where F = ASFR, P_f = female population
  - Sex split: `B_male = B × 0.51`, `B_female = B × 0.49`
  - Births assigned age 0, mother's race → child's race
- **TFR:** `TFR(r) = Σ_{a=15}^{49} ASFR(a,r)` (sum of single-year rates)
- **Scenario adjustments:** constant (baseline), +5% (high), -5% (restricted), trending (-0.5%/yr)

### Section 4: Mortality Component
**Source files to read:** `cohort_projections/core/mortality.py`, `cohort_projections/data/process/survival_rates.py`

Topics:
- **Survival rates from life tables:** `S(a,s,r) = l(a+1) / l(a)` (lx method primary)
  - Also supports qx method: `S(a) = 1 - q(a)`
- **ND adjustment:** ND/national ratio applied to national SEER life tables
- **90+ open-ended group:** `S(90+) = T(91) / (T(90) + L(90)/2)`, stays at age 90
- **Survival application:**
  - Regular ages: `P(a+1, t+1) = P(a, t) × S(a, s, r)`
  - 90+ group: `P(90+, t+1) = P(89, t) × S(89) + P(90+, t) × S(90+)`
- **Mortality improvement (Lee-Carter style):**
  - `q(a, t) = q(a, base) × (1 - δ)^{t - base}` where δ = 0.005 (0.5%/yr)
  - `S(a, t) = 1 - q(a, t)`, capped at 1.0
  - Config: `rates.mortality.improvement_factor`

### Section 5: Migration Component (largest section)
**Source files to read:** `cohort_projections/data/process/residual_migration.py`, `cohort_projections/data/process/convergence_interpolation.py`, `cohort_projections/core/migration.py`

#### 5a: Residual Migration Method
- **Core formula (Census Bureau residual):**
  - `expected(a+5) = P_start(a) × S_5yr(a, s)`
  - `migration(a+5) = P_end(a+5) - expected(a+5)`
  - `rate_period = migration / expected`
  - `rate_annual = (1 + rate_period)^{1/n} - 1` (compound annualization)
- **6 time points:** 2000, 2005, 2010, 2015, 2020, 2024
- **5 periods:** 2000-2005, 2005-2010, 2010-2015, 2015-2020, 2020-2024
- **Period length adjustment:** For 2020-2024 (4yr), `S_4yr = S_5yr^{4/5}`
- **85+ open-ended:** `expected_85+ = P(80-84) × S(80-84) + P(85+) × S(85+)`
- **0-4 birth cohort:** Set to rate = 0 (no starting cohort to age forward)

#### 5b: GQ-Corrected Migration Rates (ADR-055 Phase 2)
- Subtract historical GQ from population snapshots before residual computation
- `P_hh(a, t) = max(P_total(a, t) - GQ(a, t), 0)`
- Historical GQ: stcoreview 2020+2024, backward constant for 2000-2015
- Effect: removes dorm turnover / military PCS / nursing rotation from rates

#### 5c: Oil-Boom Dampening (ADR-040, ADR-051)
- 6 Bakken counties: Williams, McKenzie, Mountrail, Dunn, Stark, Billings
- Period-specific factors: 2005-2010 → 0.50, 2010-2015 → 0.40, 2015-2020 → 0.50
- `rate_dampened = rate × factor` (only during boom periods)

#### 5d: Male Migration Dampening
- Boom periods only (2005-2010, 2010-2015)
- `rate_male_dampened = rate_male × 0.80`

#### 5e: PEP Recalibration (ADR-045)
- Reservation counties: Benson (38005), Sioux (38085), Rolette (38079)
- Same-sign + non-trivial residual: scale by `k = PEP_total / residual_total`
- Sign reversal or near-zero: Rogers-Castro age-profile redistribution of PEP total

#### 5f: College-Age Smoothing (ADR-049)
- Counties: Grand Forks, Cass, Ward, Burleigh
- Ages 15-19, 20-24
- `rate_smoothed = 0.5 × rate_county + 0.5 × rate_statewide`
- Applied per-period BEFORE averaging (no double-smoothing)

#### 5g: Multi-Period Averaging
- Simple arithmetic mean across all 5 periods per county × age × sex cell

#### 5h: Convergence Interpolation (5-10-5 Schedule)
- Three windows: recent (2023-2025), medium (2014-2025), long-term (2000-2025)
- Window averages: arithmetic mean of periods overlapping each window
- **Convergence formula:**
  - Years 1-5: `r(y) = r_recent × (1 - y/5) + r_medium × (y/5)`
  - Years 6-15: `r(y) = r_medium`
  - Years 16-20: `r(y) = r_medium × (1 - t/5) + r_longterm × (t/5)` where `t = y - 15`
- Source: `convergence_interpolation.py:250-264`

#### 5i: Age-Aware Rate Cap (ADR-043)
- College ages (15-24): ±15%
- All other ages: ±8%
- Applied after convergence interpolation

#### 5j: BEBR High-Growth Increment (ADR-046)
- High scenario lifts all three window averages by per-cell rate increment
- `increment = (BEBR_high_net - BEBR_baseline_net) / county_pop / 36`
- Distributed uniformly across 36 cells (18 age groups × 2 sexes)

#### 5k: Ward County Migration Floor (ADR-052)
- High scenario only: if county mean rate < 0, lift all cells so mean = 0
- Prevents high-growth scenario from showing decline for institutional anchor counties

### Section 6: Scenario Methodology
**Source files to read:** `config/projection_config.yaml` (scenarios section), `cohort_projections/core/migration.py`

Topics:
- **Baseline:** Constant fertility, improving mortality (0.5%/yr), convergence migration rates
- **Restricted Growth (ADR-037, ADR-050):**
  - CBO-derived time-varying immigration enforcement factors: 2025→0.20, 2026→0.37, ..., 2029→0.91, 2030+→1.00
  - **Additive reduction** (not multiplicative): `reduction_rate = ref_intl × (1 - factor) / ref_pop`
  - `adjusted_rate = base_rate - reduction_rate`
  - Guarantees restricted ≤ baseline for ALL counties regardless of sign
  - ref_intl = 10,051 (PEP 2023-2025 avg), ref_pop = 799,358
  - Fertility: -5%
- **High Growth (ADR-046):**
  - Uses `convergence_rates_by_year_high.parquet` (BEBR-boosted convergence rates)
  - Fertility: +5%
  - Migration floor (ADR-052): no county mean rate below zero

### Section 7: Projection Engine
**Source files to read:** `cohort_projections/core/cohort_component.py`

Topics:
- **Annual step (for each county, each year):**
  1. Apply survival rates (age + mortality): `P_survived(a+1) = P(a) × S(a,s,r,t)`
  2. Calculate births: `B = Σ ASFR × P_female`, split by sex
  3. Apply migration: `P_final(a) = P_survived(a) + P_survived(a) × m(a,s,r,y)` (rate mode)
  4. Combine survived + births
  5. Clamp negative cohorts to 0
- **Time-varying rates:** Year-specific migration from convergence, year-specific survival from mortality improvement
- **GQ re-addition:** After projection, GQ (constant) added back at each year
- **State aggregation (ADR-054):** State = Σ counties (bottom-up, no independent state projection)

### Section 8: Data Sources & References
- Census PEP Vintage 2025 (stcoreview)
- SEER/NVSS fertility rates (CDC NCHS)
- CDC/SEER life tables (2023)
- CBO January 2025 vs January 2026 immigration projections
- BEBR (University of Florida) migration methodology
- Census Bureau residual migration method
- Sprague (1880) osculatory interpolation
- Rogers-Castro migration age schedule
- Lee-Carter mortality improvement
- List of all ADRs referenced (033, 036, 037, 040, 043, 045, 046, 047, 048, 049, 050, 051, 052, 053, 054, 055)

---

## Sub-Agent Dispatch Strategy

Use **4 parallel writing agents**, each producing one chunk of the document:

| Agent | Sections | Key Source Files |
|-------|----------|-----------------|
| A | 1 (Overview) + 2 (Base Pop) | `config/projection_config.yaml`, `base_population_loader.py`, `demographic_utils.py`, `fetch_census_gq_data.py` |
| B | 3 (Fertility) + 4 (Mortality) | `fertility.py`, `fertility_rates.py`, `mortality.py`, `survival_rates.py` |
| C | 5 (Migration — all subsections) | `residual_migration.py`, `convergence_interpolation.py`, `migration.py` |
| D | 6 (Scenarios) + 7 (Engine) + 8 (Sources) | `cohort_component.py`, `migration.py`, `projection_config.yaml` |

After all 4 agents return, concatenate their output into `docs/methodology.md` with a unified table of contents.

### Prompt Template for Each Agent

> You are writing sections [X] and [Y] of a comprehensive methodology document for a North Dakota cohort-component population projection model. The document is aimed at demographers and policy analysts.
>
> **Requirements:**
> - Use LaTeX notation (wrapped in `$...$` for inline, `$$...$$` for display) for all formulas
> - Pair each formula with a plain-language explanation
> - Reference specific ADRs where applicable
> - Use section headers (##, ###) consistently
> - Do not include a table of contents (the orchestrator will add one)
> - Write in a formal but accessible technical style
>
> Read these source files for exact formulas and parameters: [list files]
> Use this plan section for structure: [paste relevant section from plan]
>
> Output only the markdown content for your assigned sections.

### Assembly

After all agents complete:
1. Add document title and auto-generated TOC
2. Concatenate sections 1-8 in order
3. Write to `docs/methodology.md`
4. Verify no duplicate headers or broken LaTeX
