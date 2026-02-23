# 06. Scenario Definitions and Implementation

**Methodology Comparison: SDC 2024 vs. 2026 Cohort-Component Projections**

This section compares how the SDC 2024 projections and the current 2026 system handle uncertainty, alternative futures, and scenario construction. The SDC published a single projection trajectory with implicit uncertainty managed through period multipliers. The 2026 system implements three explicit, empirically grounded scenarios that bracket a range of plausible outcomes.

---

## 1. SDC 2024 Approach

### 1.1 Single-Trajectory Projection

The SDC 2024 projections published **one trajectory only**. There were no named scenarios, no high/low variants, no confidence intervals, and no alternative growth paths presented in the report or workbook. The published tables show a single population number for each county at each 5-year benchmark (2025, 2030, ..., 2050).

From the SDC report:

> "These projections are an attempt to provide the most likely outcome of population change expected in the state."

This framing positions the projection as a point estimate of the "most likely" future, though the report also acknowledges uncertainty:

> "Past long-term projections have often been proven to be inaccurate."

### 1.2 Fertility and Mortality: Constant Rates

Both fertility and mortality rates were held **constant** across all projection periods:

| Component | Treatment | Implication |
|-----------|-----------|-------------|
| **Fertility** | Constant (2016-2022 blended rates) | No fertility decline modeled, despite the report noting "a gradual rate of decline over time" nationally |
| **Mortality** | Constant (CDC 2020 life tables) | No mortality improvement modeled; survival probabilities identical in 2025 and 2050 |

The decision to hold both rates constant means that all projected population change in the SDC methodology comes from the combination of (a) cohort aging effects and (b) migration, which is the only component that varies across projection periods.

### 1.3 Period Multipliers as Implicit Uncertainty Management

The SDC's only mechanism for expressing uncertainty about the future was the **period multiplier** applied to migration rates. These multipliers scaled the base migration rates (averaged from four 5-year periods, 2000-2020) differently for each projection interval:

| Projection Period | Multiplier | Effect on Base Migration | Rationale |
|-------------------|:----------:|:------------------------:|-----------|
| 2020-2025 | 0.20 | 80% reduction | COVID-19, post-Bakken transition |
| 2025-2030 | 0.60 | 40% reduction | Bakken dampening |
| 2030-2035 | 0.60 | 40% reduction | Continued dampening |
| 2035-2040 | 0.50 | 50% reduction | Further reduction |
| 2040-2045 | 0.70 | 30% reduction | Gradual return toward historical |
| 2045-2050 | 0.70 | 30% reduction | Gradual return toward historical |

The formula for each cohort in each period was:

```
Effective_Migration = Natural_Growth * Base_Migration_Rate * Period_Multiplier
```

These multipliers create a U-shaped migration trajectory: very low in 2020-2025, increasing through 2030, dipping slightly at 2035-2040, then stabilizing at 70% of historical rates. The non-monotonic pattern (0.60, 0.60, 0.50, 0.70) is notable -- the 2035-2040 multiplier is lower than the surrounding periods, though the SDC report does not explain this anomaly.

### 1.4 Manual Adjustments as Additional Uncertainty Resolution

Beyond the period multipliers, the SDC applied approximately **32,000 person-adjustments per 5-year period** through manual corrections. These adjustments addressed:

1. **College-age population corrections** for counties with universities (Grand Forks, Cass, Ward)
2. **Bakken sex-ratio balancing** to reduce unrealistic male-to-female ratios caused by oil-boom migration patterns
3. **Regional economic adjustments** for western North Dakota energy counties

These manual adjustments were applied at the county-age-sex level and represent a form of expert judgment layered on top of the formula-based projection. The adjustments were not documented with sufficient specificity to replicate exactly; they appear in a separate column/sheet in the workbook but the criteria for their derivation are not formalized.

### 1.5 What SDC Did Not Publish

- No alternative scenarios (high growth, low growth, restricted)
- No confidence intervals or uncertainty bands
- No sensitivity analysis showing the impact of different multiplier choices
- No separation of migration into domestic and international components
- No CBO or other external anchoring for the multiplier values

---

## 2. Current 2026 Approach

### 2.1 Three-Scenario Framework

The 2026 system implements three named scenarios, each with distinct demographic assumptions for fertility, mortality, and migration. All three are run through the same cohort-component engine at the same resolution (annual time steps, single-year-of-age, 53 counties).

| Scenario | Purpose | Active |
|----------|---------|:------:|
| **Baseline** | Trend continuation: what happens if recent patterns persist | Yes |
| **Restricted Growth** | Policy-adjusted: CBO-modeled immigration enforcement impact | Yes |
| **High Growth** | Elevated immigration: BEBR-optimistic migration continuation | Yes |
| Zero Migration | Natural change only (analytical reference) | No |
| SDC 2024 | Replication of SDC methodology (research reference) | No |

The three active scenarios are designed to bracket a plausible range of outcomes. Per ADR-042, the baseline must **never** be published in isolation; it must always appear alongside restricted growth to prevent misinterpretation as a forecast.

### 2.2 Scenario 1: Baseline (Trend Continuation)

**ADR reference**: ADR-037

The baseline scenario extends observed historical demographic patterns into the future with no policy adjustments. It represents "what if recent trends continue unchanged."

| Component | Setting | Detail |
|-----------|---------|--------|
| **Fertility** | `constant` | Current observed age-specific fertility rates held constant (2024 NCHS rates) |
| **Mortality** | `improving` | Annual mortality improvement factor of 0.5% applied to survival rates each year |
| **Migration** | `recent_average` | Historical PEP-derived rates (2000-2025) processed through convergence interpolation |

**Migration rates**: The baseline uses the standard convergence interpolation pipeline, which produces year-varying migration rates for each county-age-sex cell using a 5-10-5 schedule:

- **Years 1-5**: Linear interpolation from recent-period average (2023-2025) toward medium-period average (2014-2025)
- **Years 6-15**: Hold at medium-period average
- **Years 16-20+**: Linear interpolation from medium toward long-term average (2000-2025)

No in-engine migration adjustments are applied. The rates come directly from the convergence pipeline output file (`convergence_rates_by_year.parquet`).

**Configuration** (from `config/projection_config.yaml`):

```yaml
baseline:
  name: "Baseline (Trend Continuation)"
  description: "Historical PEP trends with convergence interpolation; no policy adjustment"
  fertility: "constant"
  mortality: "improving"
  migration: "recent_average"
  active: true
```

### 2.3 Scenario 2: Restricted Growth (CBO Policy-Adjusted)

**ADR references**: ADR-037 (framework), ADR-039 (international-only, superseded), ADR-050 (additive reduction)

The restricted growth scenario models the demographic impact of federal immigration enforcement policy as quantified by the Congressional Budget Office. CBO published two Demographic Outlook reports one year apart (January 2025 and January 2026), and the ratio of the two provides year-specific adjustment factors that quantify the expected near-term reduction in international migration.

| Component | Setting | Detail |
|-----------|---------|--------|
| **Fertility** | `-5_percent` | 5% reduction in all age-specific fertility rates (grounded in CBO's -4.3% TFR revision between Jan 2025 and Jan 2026 publications) |
| **Mortality** | `improving` | Same as baseline (0.5%/year improvement) |
| **Migration** | `additive_reduction` | CBO-derived per-capita rate decrement applied to all cells, converging to zero adjustment by 2030 |

#### 2.3.1 CBO Factor Schedule

The time-varying factors are derived from the ratio of CBO January 2026 net immigration projections to CBO January 2025 projections:

| Projection Year | CBO Jan 2025 (thousands) | CBO Jan 2026 (thousands) | Factor (Jan 2026 / Jan 2025) | Reduction from Baseline |
|:---------------:|:------------------------:|:------------------------:|:----------------------------:|:-----------------------:|
| 2025 | 2,010 | 408 | **0.20** | 80% |
| 2026 | 1,563 | 574 | **0.37** | 63% |
| 2027 | 1,265 | 694 | **0.55** | 45% |
| 2028 | 1,068 | 833 | **0.78** | 22% |
| 2029 | 1,070 | 969 | **0.91** | 9% |
| 2030+ | ~1,073 | ~1,086 | **1.00** | 0% (converged) |

CBO's data actually shows factors of 1.01-1.11 for 2030+, meaning immigration slightly exceeds the pre-policy baseline in the long run. The implementation conservatively sets post-convergence factors to 1.00 rather than introducing a permanent upward adjustment that would require additional subnational justification.

**Source**: CBO Publication 60875 (January 2025) and CBO Publication 61879 (January 2026).

#### 2.3.2 Additive Reduction Formula (ADR-050)

The restricted growth scenario uses an **additive** per-capita rate decrement rather than a multiplicative factor. This design was adopted to fix a structural bug: when migration rates are negative (net out-migration, which applies to 39-45 of 53 North Dakota counties), a multiplicative factor less than 1.0 makes the rate *less* negative, producing *higher* population than baseline -- the opposite of the scenario's intent.

**Reference values** (from PEP 2023-2025):

| Parameter | Value | Source |
|-----------|------:|--------|
| `reference_intl_migration` | 10,051 | Annual international migration average, PEP 2023-2025 |
| `reference_population` | 799,358 | North Dakota state population at base year (2025) |

**Formulas**:

For each projection year, given the CBO factor for that year:

```
annual_reduction = reference_intl_migration * (1 - factor)     [persons/year]
reduction_rate   = annual_reduction / reference_population       [per-capita rate]
adjusted_rate    = base_rate - reduction_rate                    [always subtracts]
```

**Worked example for 2025** (factor = 0.20):

```
annual_reduction = 10,051 * (1 - 0.20) = 8,041 persons
reduction_rate   = 8,041 / 799,358     = 0.01006 per capita
adjusted_rate    = base_rate - 0.01006
```

The reduction rate is applied uniformly to every age-sex cell in every county. Because the adjustment is subtractive, it always reduces the migration rate regardless of sign:

- County with base rate of **+0.03**: adjusted to +0.0199 (less positive -- fewer people)
- County with base rate of **-0.05**: adjusted to -0.0601 (more negative -- fewer people)

This guarantees `restricted <= baseline` for all 53 counties, all 30 years, and all age-sex cells.

#### 2.3.3 Why Additive, Not Multiplicative

The original implementation (ADR-039) used a multiplicative decomposition:

```
effective_factor = 1 - intl_share * (1 - factor)
adjusted_rate    = base_rate * effective_factor
```

With `intl_share = 0.91` and `factor = 0.20`:

```
effective_factor = 1 - 0.91 * (1 - 0.20) = 0.272
```

For a county with base rate of -0.05 (net out-migration):

```
adjusted = -0.05 * 0.272 = -0.0136    (less negative = MORE people, wrong direction)
```

This produced ordering violations in 39 of 53 counties. ADR-050 replaced the multiplicative approach with the additive formula. The same class of bug (multiplicative scaling of signed rates) was fixed simultaneously for the high growth scenario in ADR-046.

#### 2.3.4 Convergence to Baseline

When `factor = 1.00` (2030 onward), `annual_reduction = 0` and the adjusted rate equals the base rate exactly. The restricted growth scenario converges to the baseline trajectory after 2030, though the population level remains permanently lower due to the cumulative deficit from 2025-2029.

**Configuration** (from `config/projection_config.yaml`):

```yaml
restricted_growth:
  name: "Restricted Growth (CBO Policy-Adjusted)"
  description: "CBO-derived immigration enforcement impact via additive migration reduction"
  fertility: "-5_percent"
  mortality: "improving"
  migration:
    type: "additive_reduction"
    schedule:
      2025: 0.20
      2026: 0.37
      2027: 0.55
      2028: 0.78
      2029: 0.91
    default_factor: 1.00
    reference_intl_migration: 10051
    reference_population: 799358
  active: true
```

### 2.4 Scenario 3: High Growth (Elevated Immigration)

**ADR references**: ADR-046 (BEBR convergence), ADR-052 (migration floor)

The high growth scenario models a counterfactual where elevated post-2020 immigration trends continue without policy intervention. Unlike the restricted growth scenario, which applies an in-engine adjustment, the high growth scenario bakes its migration advantage into the convergence rates at the preprocessing stage.

| Component | Setting | Detail |
|-----------|---------|--------|
| **Fertility** | `+5_percent` | 5% increase in all age-specific fertility rates (reflects higher aggregate fertility associated with larger immigrant population) |
| **Mortality** | `improving` | Same as baseline (0.5%/year improvement) |
| **Migration** | `recent_average` + `convergence_variant: "high"` | BEBR-optimistic convergence rates with migration floor applied |

#### 2.4.1 BEBR High Scenario Rates

The BEBR (Bureau of Economic and Business Research) multi-period averaging methodology produces two migration rate files:

- `migration_rates_pep_baseline.parquet`: Uses the trimmed average across historical periods
- `migration_rates_pep_high.parquet`: Uses the **maximum period mean** for each county (the most optimistic historical period)

The high file has the property of always producing higher migration than baseline for all 53 counties:

| Metric | Baseline | High | Difference |
|--------|:--------:|:----:|:----------:|
| State total annual net migration | +1,485 | +2,787 | **+1,302** |
| Counties higher than baseline | -- | 53/53 | All |

The +1,302/year state-level increment is corroborated by three independent estimates:

| Derivation Method | Annual Increment | Source |
|-------------------|:----------------:|--------|
| CBO Jan 2025 elevated vs. long-term, at ND share | ~1,163 | CBO 60875; PEP 2023-2025 |
| ND PEP international surge (50% of excess) | ~1,243 | PEP components |
| BEBR high-vs-baseline file difference | 1,302 | migration_rates_pep_high.parquet |

#### 2.4.2 Rate Increment Calculation and Convergence Routing

Rather than applying a multiplicative adjustment in the engine (which fails for signed rates, as documented in ADR-046), the high scenario computes an additive increment from the BEBR files and adds it to the convergence pipeline inputs.

**Per-county, per-cell rate increment formula**:

```
county_diff    = high_net_migration - baseline_net_migration    [absolute persons]
pop_ref        = median(pop_start) across residual periods       [persons]
n_cells        = 36                                              [18 age groups x 2 sexes]
rate_increment = county_diff / pop_ref / n_cells                 [annual rate per cell]
```

This increment is added uniformly to all 36 age-sex cells for each county, lifting all three convergence window averages (recent, medium, long-term) before interpolation. The convergence schedule (5-10-5) then processes the lifted rates exactly as it does for the baseline.

The output is saved as `convergence_rates_by_year_high.parquet`. The projection engine loads this file instead of the baseline convergence file when it sees `convergence_variant: "high"` in the scenario configuration.

#### 2.4.3 Migration Floor (ADR-052)

For counties where the BEBR-boosted rates are still net-negative at the medium hold point, a migration floor lifts all cells so the county mean reaches zero. This prevents the high growth scenario from showing decline for counties with institutional anchors (military bases, universities, regional centers) where net decline is not a useful planning scenario.

```yaml
migration_floor:
  enabled: true
  floor_value: 0.0   # Minimum average convergence rate at medium hold
```

#### 2.4.4 Rate Cap Interaction

Both baseline and high convergence rates are subject to the same age-aware rate cap (ADR-043):

| Age Range | Cap (positive and negative) |
|-----------|:-----:|
| Ages 15-24 | +/-15% |
| All other ages | +/-8% |

In 1,371 of 57,240 cells, the high rate equals the baseline rate because both hit the cap ceiling. The rate cap preserves demographic plausibility by preventing extreme rates in small counties from dominating the projection.

**Configuration** (from `config/projection_config.yaml`):

```yaml
high_growth:
  name: "High Growth (Elevated Immigration)"
  description: "Counterfactual: BEBR-optimistic migration rates representing sustained elevated immigration"
  fertility: "+5_percent"
  mortality: "improving"
  migration: "recent_average"
  convergence_variant: "high"
  migration_floor:
    enabled: true
    floor_value: 0.0
  active: true
```

### 2.5 Engine Implementation

The projection engine (`cohort_projections/core/cohort_component.py`) handles scenarios through a uniform interface. For each projection year, the `project_single_year()` method:

1. Looks up the scenario configuration from `config.scenarios.<scenario_name>`
2. Applies the fertility scenario adjustment (e.g., `+5_percent` multiplies all ASFR by 1.05)
3. Applies the migration scenario adjustment by passing the migration config dict to `apply_migration_scenario()` in `cohort_projections/core/migration.py`
4. Applies survival rates (with mortality improvement if configured)

The `apply_migration_scenario()` function in `migration.py` dispatches on the `type` field of the migration configuration:

| Type | Dispatch | Used By |
|------|----------|---------|
| `"additive_reduction"` | Subtracts per-capita rate decrement | Restricted growth |
| `"time_varying"` | Multiplies by effective factor (legacy, retained for backward compatibility) | -- |
| String (e.g., `"recent_average"`) | Static multiplier or no-op | Baseline, high growth |

For the high growth scenario, no in-engine migration adjustment occurs (`migration: "recent_average"` is a no-op). The scenario difference is entirely encoded in the convergence rate file loaded by the projection runner before the engine is initialized.

---

## 3. Key Differences

### 3.1 Summary Comparison Table

| Dimension | SDC 2024 | 2026 System |
|-----------|----------|-------------|
| **Number of scenarios** | 1 (single trajectory) | 3 active (baseline, restricted, high) |
| **Scenario labeling** | "Most likely outcome" | "Trend continuation", "CBO Policy-Adjusted", "Elevated Immigration" |
| **Framing** | Implicit forecast | Explicit conditional projections (never labeled as "forecast") |
| **Fertility** | Constant across all periods | Scenario-dependent: constant (baseline), -5% (restricted), +5% (high) |
| **Mortality** | Constant (no improvement) | Improving at 0.5%/year in all scenarios |
| **Migration uncertainty** | Period multipliers (0.2-0.7) on total rates | Additive CBO-derived reduction (restricted); BEBR convergence rates (high) |
| **Migration adjustment basis** | Expert judgment ("typically reduced to about 60%") | CBO Demographic Outlook publications; BEBR multi-period methodology |
| **Temporal structure of adjustment** | 6 discrete multipliers (non-monotonic) | 5-year CBO schedule converging to 1.0 (restricted); baked into convergence file (high) |
| **Manual adjustments** | ~32,000 person-adjustments per period | None (automated convergence interpolation + rate cap) |
| **Domestic/international separation** | Not differentiated | Restricted growth targets international only (via additive decrement) |
| **Publication requirement** | Single number per county per period | Baseline must always be paired with restricted growth (ADR-042) |
| **Mandatory caveats** | General caution note | Four specific caveats required: international migration dependency, geographic concentration, historical context, scenario framing |

### 3.2 Migration Adjustment Mechanism Comparison

The SDC period multipliers and the 2026 restricted growth factors both model reduced migration, but they differ fundamentally in structure:

| Feature | SDC Period Multipliers | 2026 CBO Additive Reduction |
|---------|:----------------------:|:---------------------------:|
| Applied to | Total net migration (combined) | International component only |
| Method | Multiplicative (rate * multiplier) | Additive (rate - decrement) |
| Temporal pattern | Non-monotonic: 0.2, 0.6, 0.6, 0.5, 0.7, 0.7 | Monotonically converging: 0.20, 0.37, 0.55, 0.78, 0.91, 1.00 |
| Convergence | Never returns to 1.0 (peaks at 0.7) | Returns to 1.0 by 2030 |
| Permanent effect | Yes (30% permanent reduction at horizon) | No (rates identical to baseline after convergence) |
| Derivation | Expert judgment | CBO publication ratio |
| Sign interaction | Dampens both in-migration and out-migration | Always reduces population (subtractive) |
| County-level correctness | Uniform multiplier to all counties | Uniform per-capita decrement to all counties |

### 3.3 Scenario Coverage of Component Space

| Component | SDC 2024 (single) | Baseline | Restricted Growth | High Growth |
|-----------|:------------------:|:--------:|:-----------------:|:-----------:|
| Fertility trend | Constant | Constant | -5% | +5% |
| Mortality trend | Constant | Improving | Improving | Improving |
| Migration level | Dampened (0.2-0.7x) | Convergence rates | Convergence rates minus CBO decrement | BEBR-boosted convergence rates |
| Migration adjustment | Multiplicative, permanent | None | Additive, temporary (2025-2029) | Pre-baked into rate file |

---

## 4. Formulas and Calculations

### 4.1 Restricted Growth: Complete Calculation

**Input parameters** (from `config/projection_config.yaml`):

```
reference_intl_migration = 10,051    (PEP 2023-2025 annual average)
reference_population     = 799,358   (state population at 2025)
```

**Step-by-step for each projection year**:

1. Look up CBO factor for the year:

| Year | Factor |
|:----:|:------:|
| 2025 | 0.20 |
| 2026 | 0.37 |
| 2027 | 0.55 |
| 2028 | 0.78 |
| 2029 | 0.91 |
| 2030+ | 1.00 |

2. Compute annual reduction (persons not arriving):

```
annual_reduction = 10,051 * (1 - factor)
```

| Year | Factor | 1 - factor | Annual Reduction (persons) |
|:----:|:------:|:----------:|:--------------------------:|
| 2025 | 0.20 | 0.80 | 8,041 |
| 2026 | 0.37 | 0.63 | 6,332 |
| 2027 | 0.55 | 0.45 | 4,523 |
| 2028 | 0.78 | 0.22 | 2,211 |
| 2029 | 0.91 | 0.09 | 905 |
| 2030+ | 1.00 | 0.00 | 0 |

3. Convert to per-capita rate decrement:

```
reduction_rate = annual_reduction / 799,358
```

| Year | Reduction Rate (per capita) |
|:----:|:---------------------------:|
| 2025 | 0.01006 |
| 2026 | 0.00792 |
| 2027 | 0.00566 |
| 2028 | 0.00277 |
| 2029 | 0.00113 |
| 2030+ | 0.00000 |

4. Apply to each cell's migration rate:

```
adjusted_rate[age, sex, county] = base_rate[age, sex, county] - reduction_rate
```

The same `reduction_rate` is subtracted from every cell. The total person-reduction for a county is proportional to its population since the rates are per-capita.

**Implementation** (from `cohort_projections/core/migration.py`, lines 215-235):

```python
if isinstance(scenario, dict) and scenario.get("type") == "additive_reduction":
    schedule = scenario.get("schedule", {})
    default_factor = scenario.get("default_factor", 1.0)
    factor = schedule.get(year, default_factor)
    if factor < 1.0:
        ref_intl = scenario.get("reference_intl_migration", 0)
        ref_pop = scenario.get("reference_population", 1)
        annual_reduction = ref_intl * (1.0 - factor)
        reduction_rate = annual_reduction / ref_pop
        adjusted_rates[migration_col] = (
            adjusted_rates[migration_col] - reduction_rate
        )
```

### 4.2 High Growth: BEBR Increment Calculation

**Step-by-step for each county**:

1. Load BEBR baseline and high migration rate files
2. Compute county-level total net migration for each variant:

```
bl_county_net = SUM(baseline_pep.net_migration) for county
hi_county_net = SUM(high_pep.net_migration)     for county
```

3. Compute absolute difference:

```
county_diff = hi_county_net - bl_county_net     [persons/year]
```

4. Get population reference from residual rates:

```
pop_ref = median(pop_start) across historical periods for county
```

5. Distribute uniformly across 36 cells:

```
n_cells = 36                                    [18 age groups x 2 sexes]
rate_increment = county_diff / pop_ref / n_cells [per-cell annual rate]
```

6. Add increment to all three window averages before convergence interpolation:

```
recent_rate[cell]   += rate_increment[county]
medium_rate[cell]   += rate_increment[county]
longterm_rate[cell] += rate_increment[county]
```

7. Apply standard 5-10-5 convergence interpolation to the lifted rates
8. Apply rate cap (same caps as baseline)
9. Apply migration floor (lift county mean to 0.0 if still negative)

**Implementation** (from `cohort_projections/data/process/convergence_interpolation.py`, `_compute_high_scenario_rate_increment()` and `_lift_window_averages()`):

The convergence pipeline's `run_convergence_pipeline(variant="high")` orchestrates all steps and produces `convergence_rates_by_year_high.parquet`.

### 4.3 SDC Period Multiplier Formula (for reference)

```
Migration[age, sex, county, period] =
    Natural_Growth[age, sex, county, period] *
    Base_Migration_Rate[age, sex, county] *
    Period_Multiplier[period]

Where:
    Base_Migration_Rate = AVG over 4 periods (2000-2020) [
        (Actual_Pop - Expected_Pop) / Expected_Pop
    ]
```

---

## 5. Rationale for Changes

### 5.1 Why Three Scenarios Instead of One

The SDC's single-trajectory approach was a defensible simplification for a state agency producing a reference projection. However, it has three limitations that the 2026 system addresses:

1. **No uncertainty communication**: A single number at each time point provides no information about the range of plausible outcomes. Stakeholders may treat it as a forecast rather than a conditional projection. The 2026 system's three scenarios visually communicate a range without requiring probabilistic interpretation.

2. **No policy sensitivity**: The SDC projection cannot be used to evaluate "what if" questions about policy changes. The 2026 restricted growth scenario directly models the demographic impact of 2025 federal immigration enforcement, making the projections timely and policy-relevant.

3. **Implicit assumptions become explicit**: The SDC's period multipliers embed assumptions about future migration levels (60% of historical, tapering to 70%), but these are presented as part of the methodology rather than as scenario parameters. The 2026 system makes every assumption a named, configurable parameter with documented provenance.

### 5.2 Why CBO-Grounded Restricted Growth

The restricted growth scenario replaced arbitrary multipliers (the prior system used +/-25% static factors) with factors derived from CBO's Demographic Outlook:

- **Empirical basis**: The CBO factors are the ratio of two published, nonpartisan projections -- not expert judgment or round numbers. Any analyst can independently verify the derivation by dividing CBO Publication 61879 values by CBO Publication 60875 values.

- **Time-varying structure**: CBO's data reveals a shock-and-recovery pattern (80% reduction in 2025, converging to baseline by 2030). A static multiplier cannot represent this temporal structure. The 2026 system's year-indexed schedule preserves the shape of CBO's projection, which is its most analytically important feature.

- **International-only targeting**: Immigration enforcement affects international migration, not domestic migration. The additive reduction formula subtracts a decrement derived from international migration volumes, leaving domestic patterns undisturbed. This is more accurate than the SDC's uniform multiplier applied to total migration.

### 5.3 Why BEBR for High Growth

The high growth scenario uses BEBR (Bureau of Economic and Business Research) optimistic migration rates rather than a simple percentage uplift:

- **Sign-correctness**: A multiplicative `+15%` factor (the original high growth design per ADR-037) amplifies negative migration rates, producing *lower* population for 45 of 53 counties -- the opposite of the scenario's intent. The BEBR additive increment always adds to the base rate regardless of sign.

- **County-specific grounding**: Each county's increment reflects its own historical best-period migration, not a uniform statewide percentage. Counties with highly variable migration histories (e.g., oil patch) get appropriately sized increments.

- **Consistency with convergence framework**: Both baseline and high use the same convergence interpolation schedule and rate cap. The only difference is the level of the window averages before interpolation. This means the scenarios share the same temporal shape and differ only in magnitude.

### 5.4 Why Additive Over Multiplicative

Both the restricted growth and high growth scenarios evolved from multiplicative to additive adjustment formulas. The fundamental problem with multiplicative scaling of signed rates is:

| Scenario | Intended Direction | Base Rate Sign | Multiplicative Result | Correct? |
|----------|:------------------:|:--------------:|:---------------------:|:--------:|
| Restricted (x0.27) | Lower population | Positive (+0.03) | +0.008 (less positive) | Yes |
| Restricted (x0.27) | Lower population | Negative (-0.05) | -0.014 (less negative) | **No** |
| High (x1.15) | Higher population | Positive (+0.03) | +0.035 (more positive) | Yes |
| High (x1.15) | Higher population | Negative (-0.05) | -0.058 (more negative) | **No** |

Since 39-45 of 53 North Dakota counties have net-negative migration rates, multiplicative scaling produces the wrong directional effect for the majority of counties. Additive adjustments (subtracting for restricted, adding for high) guarantee correct directionality regardless of the base rate sign.

### 5.5 Presentation Requirements (ADR-042)

The 2026 system imposes four mandatory caveats on any publication that includes baseline projections:

1. **International migration dependency**: "91% of recent net migration to North Dakota is international. The baseline assumes continuation of recent immigration levels; actual migration will depend on federal policy, global conditions, and economic factors."

2. **Geographic concentration**: "89% of projected growth is concentrated in Cass, Burleigh, and Williams counties. Statewide totals do not reflect uniform growth across all regions."

3. **Historical context**: "The baseline growth rate of 0.77%/yr exceeds all historical decades except the 2010-2015 oil boom. Sustained growth at this rate for 30 years would be unprecedented."

4. **Scenario framing**: "This is a trend-continuation scenario, not a forecast. It shows what would happen if recent demographic patterns persist unchanged. See the restricted growth scenario for an alternative trajectory."

The SDC report included a general caution note ("past long-term projections have often been proven to be inaccurate") but did not mandate specific caveats or require companion scenarios.

---

## 6. References

### ADRs

| ADR | Title | Relevance |
|-----|-------|-----------|
| [ADR-037](../../governance/adrs/037-cbo-grounded-scenario-methodology.md) | CBO-Grounded Scenario Methodology | Defines the three-scenario framework and CBO factor derivation |
| [ADR-039](../../governance/adrs/039-international-only-migration-factor.md) | International-Only Migration Factor | Original multiplicative decomposition (superseded by ADR-050) |
| [ADR-042](../../governance/adrs/042-baseline-projection-presentation-requirements.md) | Baseline Projection Presentation Requirements | Mandatory caveats and pairing requirements |
| [ADR-046](../../governance/adrs/046-high-growth-bebr-convergence.md) | High Growth via BEBR Convergence Rates | Replaces broken +15% multiplicative with BEBR additive |
| [ADR-050](../../governance/adrs/050-restricted-growth-additive-migration-adjustment.md) | Restricted Growth Additive Migration Adjustment | Replaces multiplicative effective_factor with additive decrement |
| [ADR-052](../../governance/adrs/052-ward-county-high-growth-floor.md) | Ward County High-Growth Scenario Floor | Migration floor preventing decline in high growth for major counties |

### Source Code

| File | Role |
|------|------|
| `cohort_projections/core/migration.py` | `apply_migration_scenario()`: dispatches additive_reduction, time_varying, and static scenarios |
| `cohort_projections/core/cohort_component.py` | `project_single_year()`: orchestrates fertility, mortality, and migration scenario application |
| `cohort_projections/data/process/convergence_interpolation.py` | `run_convergence_pipeline(variant="high")`: generates high-scenario convergence rates |
| `config/projection_config.yaml` | All scenario definitions, CBO schedule, reference values |

### External Sources

| Source | Use |
|--------|-----|
| CBO Publication 60875 (January 2025 Demographic Outlook) | Pre-policy immigration baseline |
| CBO Publication 61879 (January 2026 Demographic Outlook) | Post-enforcement immigration revision |
| SDC 2024 Population Projections Report (February 6, 2024) | Single-trajectory methodology and published results |
| `Projections_Base_2023.xlsx` (SDC workbook) | Period multiplier values and manual adjustment magnitudes |

---

*Last updated: 2026-02-20*
