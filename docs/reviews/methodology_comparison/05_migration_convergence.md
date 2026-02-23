# 05 -- Migration Convergence and Forward Projection

**Scope**: How computed historical migration rates are projected into the future -- the convergence/dampening schedule applied to rates before they enter the projection engine. This document does _not_ cover the upstream computation of raw residual rates (covered separately); it begins where those rates end and follows them through the forward-projection pipeline.

**Sources**:
- SDC 2024: `sdc_2024_replication/METHODOLOGY_SPEC.md` (Sections 3.3, Appendix C)
- ADR-036: `docs/governance/adrs/036-migration-averaging-methodology.md` (BEBR multi-period + convergence design)
- ADR-043: `docs/governance/adrs/043-migration-rate-cap.md` (age-aware rate cap)
- ADR-046: `docs/governance/adrs/046-high-growth-bebr-convergence.md` (high-growth BEBR convergence)
- ADR-052: `docs/governance/adrs/052-ward-county-high-growth-floor.md` (migration floor for declining counties)
- Implementation: `cohort_projections/data/process/convergence_interpolation.py`
- Configuration: `config/projection_config.yaml` (sections `rates.migration.interpolation`, `scenarios`)

---

## 1. SDC 2024 Approach -- Period Multipliers on a Flat Rate

### 1.1 Rate Construction

SDC computes a single static migration rate for each of the 1,908 county x age-group x sex cells (53 counties x 18 five-year age groups x 2 sexes). The rate is the simple arithmetic mean of four residual migration observations drawn from intercensal periods:

| Period | Observation |
|--------|------------|
| 2000--2005 | Residual migration / natural growth |
| 2005--2010 | Residual migration / natural growth |
| 2010--2015 | Residual migration / natural growth |
| 2015--2020 | Residual migration / natural growth |

All four periods receive equal weight. The resulting rate for each cell is expressed as a **proportion of natural growth** (survived population), not as a proportion of the starting population.

### 1.2 Forward-Projection Mechanism: Period Multipliers

SDC does not implement convergence. Instead, the single averaged rate is scaled by a **period multiplier** that varies by 5-year projection interval but is identical across all counties, ages, and sexes:

| Period | Years | Multiplier | Effective % of Base Rate |
|--------|-------|:----------:|:------------------------:|
| 1 | 2020--2025 | 0.20 | 20% |
| 2 | 2025--2030 | 0.60 | 60% |
| 3 | 2030--2035 | 0.60 | 60% |
| 4 | 2035--2040 | 0.50 | 50% |
| 5 | 2040--2045 | 0.70 | 70% |
| 6 | 2045--2050 | 0.70 | 70% |

The formula for each cell in each period is:

```
Effective_Migration = Natural_Growth * Base_Rate * Period_Multiplier
```

### 1.3 Properties of the SDC Approach

1. **No convergence concept**: The base rate never changes. The multiplier scales it uniformly.
2. **Spatially uniform dampening**: Every county, age group, and sex receives the same multiplier in a given period. A 20-24 Male in Cass County and a 75-79 Female in Slope County are both scaled by 0.60 in 2025--2030.
3. **U-shaped trajectory**: Migration starts very low (0.20, reflecting COVID + post-boom adjustment), rises to 0.60 by 2025, dips to 0.50 in 2035--2040, and finishes at 0.70 in 2040--2050. This creates an acceleration of net migration toward the end of the horizon.
4. **Never reaches the full base rate**: Even at the maximum multiplier (0.70), the projected migration rate is only 70% of the historical average. The implicit assumption is that 2000--2020 conditions (including the Bakken boom) will never fully repeat.
5. **Manual adjustments supplement**: Approximately 32,000 person-adjustments per 5-year period are applied on top of the formula. These are not documented at the cell level in published materials.

### 1.4 Implications for 2040--2050

In the final two periods (2040--2050), the multiplier is held at 0.70. Because the underlying base rate is static, migration volumes in these decades depend entirely on the size of the natural-growth denominator (survived population from prior periods). As cohorts age and natural increase declines (turning negative by 2045--2050), net migration grows in relative importance -- accounting for roughly 75% of total population change by the end of the horizon. The increasing multiplier (from 0.50 to 0.70) amplifies this: SDC projects net migration of +37,624 in 2045--2050, the highest of any period, despite natural change turning negative (-5,526).

---

## 2. Current 2026 Approach -- Multi-Period BEBR Convergence

### 2.1 Conceptual Framework

The 2026 method combines two established demographic traditions (ADR-036):

- **BEBR multi-period averaging** (Florida Bureau of Economic and Business Research): compute window averages from overlapping historical periods to smooth volatility
- **Census Bureau convergence interpolation**: transition rates from short-term conditions toward long-term means over the projection horizon

The result is a **time-varying** migration rate for each of the 57,240 county x age-group x sex x year cells (53 counties x 18 five-year age groups x 2 sexes x 30 projection years), where the rate changes every year as it moves through the convergence schedule.

### 2.2 Three Averaging Windows

Phase 1 (the residual migration pipeline) produces annualized migration rates for five historical periods. These are grouped into three overlapping windows and averaged within each window:

| Window | Config Key | Year Range | Periods Included | Rationale |
|--------|-----------|:----------:|-----------------|-----------|
| **Recent** | `recent_period` | 2023--2025 | Most recent period only | Captures current conditions |
| **Medium** | `medium_period` | 2014--2025 | ~2--3 periods | Approximates Smith & Tayman optimal 10-year base |
| **Long-term** | `longterm_period` | 2000--2025 | All 5 periods | Full available history (25 years) |

Within each window, the average is the **simple arithmetic mean** of `migration_rate` across all periods that overlap the window range, computed independently for each county x age-group x sex cell.

### 2.3 Convergence Schedule (5-10-5)

The 30-year projection horizon is divided into three phases. The default schedule is 5-10-5 with a 30-year horizon, which means:

| Phase | Projection Years | Duration | Rate Source |
|-------|:----------------:|:--------:|-------------|
| Phase 1 | 1--5 (2026--2030) | 5 years | Linear interpolation: Recent --> Medium |
| Phase 2 | 6--15 (2031--2040) | 10 years | **Hold** at Medium rate |
| Phase 3 | 16--20 (2041--2045) | 5 years | Linear interpolation: Medium --> Long-term |
| Extended | 21--30 (2046--2055) | 10 years | **Hold** at Long-term rate |

(Note: The convergence schedule parameters define a 20-year cycle -- 5+10+5. With `projection_horizon: 30`, years 21--30 hold at the long-term rate because the interpolation parameter `t` is clamped at 1.0 in the Phase 3 formula.)

#### Year-by-Year Convergence Schedule

For a cell with Recent rate `R`, Medium rate `M`, and Long-term rate `L`:

| Year Offset | Formula | Interpolation Parameter `t` |
|:-----------:|---------|:---------------------------:|
| 1 | `R * (1 - 1/5) + M * (1/5)` = `0.8R + 0.2M` | 0.20 |
| 2 | `R * (1 - 2/5) + M * (2/5)` = `0.6R + 0.4M` | 0.40 |
| 3 | `R * (1 - 3/5) + M * (3/5)` = `0.4R + 0.6M` | 0.60 |
| 4 | `R * (1 - 4/5) + M * (4/5)` = `0.2R + 0.8M` | 0.80 |
| 5 | `R * (1 - 5/5) + M * (5/5)` = `M` | 1.00 |
| 6--15 | `M` (held constant) | -- |
| 16 | `M * (1 - 1/5) + L * (1/5)` = `0.8M + 0.2L` | 0.20 |
| 17 | `M * (1 - 2/5) + L * (2/5)` = `0.6M + 0.4L` | 0.40 |
| 18 | `M * (1 - 3/5) + L * (3/5)` = `0.4M + 0.6L` | 0.60 |
| 19 | `M * (1 - 4/5) + L * (4/5)` = `0.2M + 0.8L` | 0.80 |
| 20 | `M * (1 - 5/5) + L * (5/5)` = `L` | 1.00 |
| 21--30 | `L` (held constant, `t` clamped at 1.0) | 1.00 |

### 2.4 Age-Aware Rate Cap (ADR-043)

After interpolation at each year offset, an age-aware rate cap is applied to clip extreme values caused by small-population statistical noise. The cap is applied symmetrically (same threshold for positive and negative rates):

| Age Group | Cap Threshold | Rationale |
|-----------|:------------:|-----------|
| 15--19, 20--24 (college ages) | +/-15% | Preserves legitimate university enrollment dynamics (Cass 20-24F at 13.2%, Grand Forks 20-24M at 14.0%) |
| All other ages | +/-8% | P99 of medium-term distribution for non-college ages is 8.13%; clips only the statistical tail |

The cap is applied **after** the interpolation formula and **before** storing the result, so it catches all three convergence phases without modifying the underlying window averages. Across all 57,240 cells (2,862 cells x 20 year offsets), approximately 2.4% of cells are clipped (1,372 of 57,240).

**Implementation** (from `_apply_rate_cap()` in `convergence_interpolation.py`):
```python
college_mask = age_groups.isin(college_ages)
capped = rate.clip(lower=-general_cap, upper=general_cap)
capped[college_mask] = rate[college_mask].clip(lower=-college_cap, upper=college_cap)
```

### 2.5 High-Growth Scenario: BEBR Convergence Rates (ADR-046)

The high-growth scenario does not use a multiplicative scaling factor. Instead, it generates a separate set of convergence rates by computing an additive rate increment from the BEBR high-vs-baseline migration difference and adding it to all three window averages before convergence interpolation.

#### Increment Computation

1. Load BEBR baseline (`migration_rates_pep_baseline.parquet`) and high (`migration_rates_pep_high.parquet`) migration rate files. The BEBR "high" uses the **maximum period mean** for each county (the most optimistic historical period).
2. For each county, compute the absolute migration difference:
   ```
   county_diff = high_net_migration - baseline_net_migration   (persons/year)
   ```
3. Convert to a per-cell rate increment:
   ```
   pop_ref = county population from most recent residual period  (persons)
   n_cells = 36  (18 age groups x 2 sexes)
   rate_increment = county_diff / pop_ref / n_cells              (annual rate)
   ```
4. Add the increment uniformly to all 36 cells for each county, in all three window averages (recent, medium, long-term).
5. Run the standard convergence interpolation (5-10-5) on the lifted window averages.

This produces `convergence_rates_by_year_high.parquet` with the same schema as the baseline file.

**Validated properties**: The increment guarantees `high >= baseline` for all 53 counties, all 30 years, all age-sex cells. Of 57,240 total cells, 55,869 are strictly higher and 1,371 are equal (hitting the rate cap ceiling).

**State-level magnitude**: The increment adds approximately +1,302 net migrants/year at the state level, consistent with three independent estimates:

| Derivation | Annual Increment |
|-----------|:----------------:|
| CBO Jan 2025 elevated vs. long-term, at ND share | ~1,163 |
| ND PEP international surge (50% of excess) | ~1,243 |
| BEBR high-vs-baseline rate file difference | 1,302 |

### 2.6 Migration Floor for High-Growth Scenario (ADR-052)

For counties where the BEBR-boosted rates are still net-negative at the medium hold (e.g., Ward County, whose best historical period was only marginally better than average), a migration floor prevents the high-growth scenario from showing decline.

**Mechanism**: For each county, compute the mean convergence rate across all 36 cells. If this mean is below the floor value (default: 0.0), lift all cells by `|county_mean|` so the county average reaches zero (neutral migration).

```python
if variant == "high":
    for fips in window_rates["county_fips"].unique():
        county_mean = window_rates.loc[mask, "migration_rate"].mean()
        if county_mean < floor_value:
            lift = floor_value - county_mean
            window_rates.loc[mask, "migration_rate"] += lift
```

The floor is applied to all three window averages (recent, medium, long-term) before convergence interpolation, so the 5-10-5 schedule operates on the floored rates.

**Impact**: Only Ward County and a small number of other declining counties are affected. Counties with positive BEBR-boosted rates are untouched.

**Configuration**:
```yaml
high_growth:
  migration_floor:
    enabled: true
    floor_value: 0.0  # Minimum average convergence rate at medium hold
```

---

## 3. Key Differences

| Dimension | SDC 2024 | Current 2026 |
|-----------|----------|-------------|
| **Forward-projection mechanism** | Period multipliers on a static rate | Convergence interpolation across three time-varying windows |
| **Rate varies over time?** | No -- base rate is constant; only the multiplier changes | Yes -- rate changes every year through the 5-10-5 schedule |
| **Cell-level granularity of dampening** | None -- uniform multiplier for all cells in a period | Full -- each county x age x sex cell converges independently |
| **Temporal resolution** | 5-year intervals (6 periods) | Annual (30 years) |
| **Number of historical windows** | 1 (simple average of 4 intercensal periods) | 3 (recent, medium, long-term) |
| **Historical periods used** | 2000--2020 (pre-Vintage 2025) | 2000--2025 (Vintage 2025 PEP) |
| **Rate expression** | Proportion of natural growth (survived population) | Proportion of starting population |
| **End-state rate** | 70% of 2000--2020 average, held indefinitely | Full 25-year (2000--2025) average, held from year 20 onward |
| **Rate cap** | None | Age-aware: +/-15% (college ages), +/-8% (all others) |
| **High-growth mechanism** | Not applicable (single scenario) | Additive BEBR increment + migration floor |
| **Manual adjustments** | ~32,000 person-adjustments/period | None |
| **Boom-era handling** | Uniform 0.20--0.70 multipliers across all counties | County-specific boom dampening (0.40--0.60) for 6 oil counties only (ADR-040/051), applied upstream in residual computation |
| **Migration direction (net)** | Net in-migration (positive sign convention) | Net rate per cell (can be positive or negative per cell) |
| **Late-horizon migration volume** | Accelerating (multiplier rises from 0.50 to 0.70) | Stabilizing (converges to long-term mean and holds) |

---

## 4. Formulas and Calculations

### 4.1 Window Averaging (2026 Method)

**Input**: Phase 1 residual migration rates, a DataFrame with columns `[county_fips, age_group, sex, period_start, period_end, migration_rate]`.

**Period-to-window mapping**: A historical period `(period_start, period_end)` is included in a window `[win_start, win_end]` if the two intervals overlap:

```
period included if: period_end >= win_start AND period_start <= win_end
```

**Window average**: For each window, the simple arithmetic mean of `migration_rate` across all included periods, computed independently per cell:

```
window_avg(county, age, sex) = (1/N) * SUM over included periods [migration_rate(county, age, sex, period)]
```

where `N` is the number of periods that overlap the window range.

**Current configuration** (`config/projection_config.yaml`):
```yaml
recent_period: [2023, 2025]     # Matches most recent period only
medium_period: [2014, 2025]     # Matches ~2-3 periods
longterm_period: [2000, 2025]   # Matches all 5 periods
```

### 4.2 Linear Interpolation Between Windows

For Phase 1 (years 1 through `P1`, default P1 = 5):

```
rate(year) = R * (1 - year/P1) + M * (year/P1)
```

where `R` = recent window average, `M` = medium window average.

For Phase 2 (years `P1+1` through `P1+P2`, default P1+P2 = 15):

```
rate(year) = M
```

For Phase 3 (years `P1+P2+1` through `P1+P2+P3`, default P1+P2+P3 = 20):

```
years_into_phase3 = year - P1 - P2
t = min(years_into_phase3 / P3, 1.0)
rate(year) = M * (1 - t) + L * t
```

where `L` = long-term window average.

For years beyond `P1+P2+P3` (years 21--30, when `projection_horizon > 20`):

```
rate(year) = L    (t is clamped at 1.0)
```

**Code reference** (`convergence_interpolation.py`, lines 250--263):
```python
for year in range(1, projection_years + 1):
    if year <= phase1:
        t = year / phase1
        rate = merged["recent"] * (1 - t) + merged["medium"] * t
    elif year <= phase1 + phase2:
        rate = merged["medium"]
    else:
        years_into_phase3 = year - phase1 - phase2
        t = years_into_phase3 / phase3
        t = min(t, 1.0)
        rate = merged["medium"] * (1 - t) + merged["longterm"] * t
```

### 4.3 Rate Cap Application

Applied after interpolation at each year offset:

```
For each cell (county, age_group, sex):
    if age_group in {"15-19", "20-24"}:
        capped_rate = clip(rate, -0.15, +0.15)
    else:
        capped_rate = clip(rate, -0.08, +0.08)
```

**Interaction with convergence**: The cap is applied to the interpolated rate, not to the window averages. This means:
- A cell with Recent = +0.12 and Medium = +0.06 (non-college age) will be capped at +0.08 during early years when the interpolated rate exceeds +0.08, but will be uncapped once convergence brings it below +0.08.
- The cap does not alter the window averages themselves, preserving data lineage.

### 4.4 High-Variant BEBR Rate Increment Computation (ADR-046)

**Step 1**: Load BEBR baseline and high PEP migration files:
```
baseline_net(county) = SUM over cells [net_migration] from migration_rates_pep_baseline.parquet
high_net(county) = SUM over cells [net_migration] from migration_rates_pep_high.parquet
```

**Step 2**: Compute county-level absolute difference:
```
county_diff = high_net(county) - baseline_net(county)   [persons/year]
```

**Step 3**: Get population reference from residual rates (most recent period):
```
county_pop = SUM over cells [pop_start] from residual_migration_rates.parquet
             where period_start = max(period_start)
```

**Step 4**: Compute per-cell rate increment:
```
n_cells = count of (age_group, sex) combinations per county  [= 36]
rate_increment = county_diff / max(county_pop, 1.0) / n_cells
```

**Step 5**: Add the increment to all three window averages before interpolation:
```
recent_rate_high(c, a, s)   = recent_rate_baseline(c, a, s)   + rate_increment(c)
medium_rate_high(c, a, s)   = medium_rate_baseline(c, a, s)   + rate_increment(c)
longterm_rate_high(c, a, s) = longterm_rate_baseline(c, a, s) + rate_increment(c)
```

The increment is **uniform across all age-sex cells** within a county but **county-specific** across counties.

**Code reference** (`convergence_interpolation.py`, `_compute_high_scenario_rate_increment()`, lines 296--378).

### 4.5 Migration Floor Lift Calculation (ADR-052)

Applied to all three window averages for the high variant only:

```
For each county c and each window W in {recent, medium, longterm}:
    county_mean(c, W) = mean over all 36 cells [migration_rate(c, a, s, W)]
    if county_mean(c, W) < floor_value:
        lift = floor_value - county_mean(c, W)    [always positive]
        For each cell (c, a, s) in W:
            migration_rate(c, a, s, W) += lift
```

The default `floor_value` is 0.0 (neutral migration). This means any county whose mean convergence rate is negative will have all its cell rates lifted by the same absolute amount so the county mean reaches exactly zero. Individual cells can still be negative; the constraint applies to the county **mean**.

**Code reference** (`convergence_interpolation.py`, lines 501--525).

### 4.6 SDC 2024 Period Multiplier Formula (for comparison)

SDC's formula is much simpler. For each 5-year period `p`:

```
Effective_Migration(age, sex, county, p) =
    Natural_Growth(age, sex, county, p) * Base_Rate(age, sex, county) * Multiplier(p)
```

where:

```
Base_Rate(age, sex, county) = (1/4) * SUM over k in {1..4} [
    Observed_Migration(age, sex, county, period_k) / Natural_Growth(age, sex, county, period_k)
]

Multiplier(p) = {0.20, 0.60, 0.60, 0.50, 0.70, 0.70} for p in {1..6}
```

There is no convergence interpolation, no rate cap, and no cell-level dampening variation.

---

## 5. Rationale for Changes

### 5.1 Why Convergence Instead of Constant Multipliers

SDC's period multipliers create a rate trajectory that is determined entirely by six subjective numbers chosen to reflect the analyst's expectation about migration intensity over 30 years. These six numbers have no published methodological basis; they are ad-hoc calibration parameters. Moreover, because the multipliers are spatially uniform, they cannot capture the fundamentally different migration dynamics of Cass County (university + metro growth), McKenzie County (oil extraction), and Sioux County (reservation decline).

The convergence approach replaces the six subjective multipliers with a **structured framework** drawn from two established traditions:

1. **Multi-period window averaging** (BEBR): by computing separate averages for recent, medium, and long-term windows, the method captures different information from different segments of history without requiring the analyst to choose a single set of weights.

2. **Time-varying interpolation** (Census Bureau / UK ONS): the 5-10-5 schedule embodies the well-established demographic principle that recent conditions carry more information for the near term but should not dominate the long-term outlook. The rate converges naturally toward the long-run mean, which is the most defensible neutral assumption (Smith & Tayman 2003).

Because convergence operates at the cell level -- each county x age x sex combination converges independently -- it automatically produces different effective dampening for different demographic profiles. A county whose recent period is very different from its long-term mean will see large convergence changes; a stable county will see almost none.

### 5.2 Why Rate Caps

Small-county cells with populations of 5--30 people produce migration rates that are statistically meaningless. A single family moving into a cell with 10 people produces a +10% migration rate. Over a 30-year projection, these extreme rates compound into implausible outcomes (e.g., Billings County at +55% growth from a population of ~950).

The age-aware cap addresses this with minimal disruption:

- The **8% general cap** sits at P99 of the medium-term distribution for non-college ages. It clips only 2.4% of cells, targeting the statistical tail rather than meaningful demographic signals.
- The **15% college-age cap** provides headroom above the highest observed legitimate university enrollment rates (Grand Forks 20-24M at 14.0%), preventing the general cap from flattening real institutional dynamics.
- The cap is applied **after interpolation**, so it does not distort the window averages or the convergence schedule itself. It functions as a guardrail on the output, not a modification of the method.

Without the cap, the convergence method would propagate noise from tiny cells over 30 years, producing unrealistic county-level results even though the state-level aggregation would smooth out.

### 5.3 Why BEBR Convergence for the High-Growth Scenario

The original high-growth scenario used a multiplicative `+15_percent` factor applied to all net migration rates. This is fundamentally broken for North Dakota because 45 of 53 counties have negative net migration (net out-migration). Multiplying a negative rate by 1.15 makes out-migration **more negative**, producing **lower** population than baseline -- the exact opposite of the scenario's intent.

| County Migration Sign | Effect of x1.15 | Desired Effect |
|:---------------------:|:----------------:|:--------------:|
| Positive (+) | More in-migration | Correct |
| Negative (-) | More out-migration | **Wrong** |
| Near zero | Negligible | Insufficient |

The BEBR convergence approach solves this by computing an **additive** increment: the per-county difference between the BEBR high scenario (most optimistic historical period) and baseline scenario, expressed as a rate and added uniformly to all cells. Because the increment is additive:

- Counties with negative baseline rates become **less negative** (or positive)
- Counties with positive baseline rates become **more positive**
- The high scenario is guaranteed to produce `high >= baseline` for all counties, all years, all cells

The magnitude of the increment (~1,302 persons/year at the state level) is independently validated against CBO immigration estimates and PEP international migration data, providing empirical grounding that a 15% multiplicative factor lacks.

### 5.4 The Problem with Multiplicative High-Growth Factors

The multiplicative sign problem is not unique to North Dakota. Any subnational projection where a majority of units have negative net migration will exhibit this inversion. The fundamental issue is that "high growth" is an **additive** concept -- more people arrive -- while multiplicative scaling is a **proportional** concept that preserves the sign and amplifies the magnitude of whatever rate exists. For net migration, which can be positive or negative, these two concepts diverge when the base rate is negative.

This is documented in detail in ADR-046, which shows that the problem affects 45 of 53 counties at year offset 1, and 47 of 53 by year offset 20.

The BEBR additive approach also avoids a subtler problem: multiplicative factors have no effect on counties with near-zero migration, even though those counties might benefit most from an optimistic immigration scenario. The additive increment provides a non-zero boost to every county.

---

## Appendix A: Complete Convergence Schedule -- Worked Example

Consider a hypothetical cell with:
- Recent window average `R = +0.04` (4% annual in-migration rate)
- Medium window average `M = +0.02` (2%)
- Long-term window average `L = +0.01` (1%)
- Non-college age group (8% rate cap applies)

| Year | Phase | Formula | Rate | Capped? |
|:----:|:-----:|---------|:----:|:-------:|
| 1 | 1 | 0.8(0.04) + 0.2(0.02) | 0.036 | No |
| 2 | 1 | 0.6(0.04) + 0.4(0.02) | 0.032 | No |
| 3 | 1 | 0.4(0.04) + 0.6(0.02) | 0.028 | No |
| 4 | 1 | 0.2(0.04) + 0.8(0.02) | 0.024 | No |
| 5 | 1 | 0.0(0.04) + 1.0(0.02) | 0.020 | No |
| 6 | 2 | hold at M | 0.020 | No |
| 7 | 2 | hold at M | 0.020 | No |
| ... | 2 | hold at M | 0.020 | No |
| 15 | 2 | hold at M | 0.020 | No |
| 16 | 3 | 0.8(0.02) + 0.2(0.01) | 0.018 | No |
| 17 | 3 | 0.6(0.02) + 0.4(0.01) | 0.016 | No |
| 18 | 3 | 0.4(0.02) + 0.6(0.01) | 0.014 | No |
| 19 | 3 | 0.2(0.02) + 0.8(0.01) | 0.012 | No |
| 20 | 3 | 0.0(0.02) + 1.0(0.01) | 0.010 | No |
| 21--30 | ext | hold at L | 0.010 | No |

The rate declines smoothly from 3.6% to 1.0% over 20 years, then holds. No cell in this example exceeds the 8% cap.

Now consider a small-county cell with `R = +0.12`, `M = +0.09`, `L = +0.05` (non-college age):

| Year | Phase | Uncapped Rate | Capped Rate | Clipped? |
|:----:|:-----:|:-------------:|:-----------:|:--------:|
| 1 | 1 | 0.114 | **0.080** | Yes |
| 2 | 1 | 0.108 | **0.080** | Yes |
| 3 | 1 | 0.102 | **0.080** | Yes |
| 4 | 1 | 0.096 | **0.080** | Yes |
| 5 | 1 | 0.090 | **0.080** | Yes |
| 6--15 | 2 | 0.090 | **0.080** | Yes |
| 16 | 3 | 0.082 | **0.080** | Yes |
| 17 | 3 | 0.074 | 0.074 | No |
| 18--20 | 3 | 0.066--0.050 | 0.066--0.050 | No |
| 21--30 | ext | 0.050 | 0.050 | No |

The cap compresses years 1--16 to 0.080, then the natural convergence brings the rate below the cap by year 17.

---

## Appendix B: SDC Period Multiplier Trajectory vs. 2026 Convergence

To aid visualization, the table below contrasts the effective rate scaling for a hypothetical cell at each time point. For SDC, the "effective rate" is `Base_Rate * Multiplier`. For 2026, it is the interpolated rate from the convergence schedule. Assume the SDC base rate = the 2026 medium window average for comparability.

| Calendar Year | SDC Effective Rate (% of base) | 2026 Effective Rate (conceptual) |
|:------------:|:------------------------------:|:--------------------------------:|
| 2025 | 20% of base | Recent (may be above or below base) |
| 2026 | 20% | ~80% Recent + 20% Medium |
| 2027 | 60% | ~60% Recent + 40% Medium |
| 2028 | 60% | ~40% Recent + 60% Medium |
| 2029 | 60% | ~20% Recent + 80% Medium |
| 2030 | 60% | Medium (held) |
| 2031--2035 | 60% (2030--35) then 50% (2035--40) | Medium (held) |
| 2036--2040 | 50% | Medium (held) |
| 2041 | 70% | 80% Medium + 20% Long-term |
| 2042 | 70% | 60% Medium + 40% Long-term |
| 2043 | 70% | 40% Medium + 60% Long-term |
| 2044 | 70% | 20% Medium + 80% Long-term |
| 2045 | 70% | Long-term (held) |
| 2046--2050 | 70% | Long-term (held) |
| 2051--2055 | -- (horizon ends at 2050) | Long-term (held) |

The key structural difference: SDC's effective rate never exceeds 70% of the base rate and follows an arbitrary trajectory. The 2026 method's effective rate is governed by the relationship between recent, medium, and long-term averages -- it can start above or below the medium rate depending on recent conditions, and it converges toward the full historical mean rather than an arbitrary fraction of it.

---

## References

1. **SDC 2024 Methodology**: `sdc_2024_replication/METHODOLOGY_SPEC.md`
2. **ADR-036**: Migration Averaging Methodology -- `docs/governance/adrs/036-migration-averaging-methodology.md`
3. **ADR-043**: Age-Aware Migration Rate Cap -- `docs/governance/adrs/043-migration-rate-cap.md`
4. **ADR-046**: High Growth BEBR Convergence -- `docs/governance/adrs/046-high-growth-bebr-convergence.md`
5. **ADR-052**: Ward County High-Growth Floor -- `docs/governance/adrs/052-ward-county-high-growth-floor.md`
6. **Implementation**: `cohort_projections/data/process/convergence_interpolation.py`
7. **Configuration**: `config/projection_config.yaml`
8. Smith, S.K. & Tayman, J. (2003). "The relationship between the length of the base period and population forecast errors." _International Journal of Forecasting_.
9. Florida BEBR (2024). Population projections methodology. https://bebr.ufl.edu/
10. U.S. Census Bureau State Interim Projections methodology. https://wonder.cdc.gov/wonder/help/populations/population-projections/methodology.html
