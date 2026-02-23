# Design: Reservation County Migration Recalibration

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Author** | Claude Code (Opus 4.6) |
| **Parent Finding** | [Finding 3: AIAN Reservation County Declines](finding-3-reservation-county-declines.md) |
| **Status** | Draft -- design proposal for review |
| **Scope** | Benson (38005), Sioux (38085), Rolette (38079) |

---

## 1. Problem Summary

Three AIAN reservation counties are projected to decline 45-47% over 30 years under the baseline scenario. The root cause is a systematic bias in the residual migration method: it produces out-migration estimates 2-3x larger than PEP component estimates for these counties. This bias arises from (a) Census undercounts on tribal lands inflating the apparent population loss between censuses, and (b) statewide survival rates being applied to populations with lower life expectancy, causing excess mortality to be misattributed as out-migration.

The fix is to anchor migration totals to PEP component data (which directly measures net migration) while preserving the age-sex detail needed for cohort-component projection. A secondary mechanism -- migration deceleration -- addresses the model's lack of feedback between population decline and continued out-migration.

---

## 2. Data Structure Analysis

### 2.1 PEP Components (`data/processed/pep_county_components_2000_2025.parquet`)

- **Shape**: 1,378 rows x 15 columns
- **Grain**: One row per county per year (annual)
- **Years**: 2000-2025 (26 years)
- **Key columns**: `county_fips` (3-digit), `year`, `netmig`, `intl_mig`, `domestic_mig`, `residual`
- **Dataset sources**: `co-est2009-alldata` (2000-2009), `co-est2019-alldata` (2010-2019), `stcoreview-v2025` (2020-2025)
- **Preferred estimates**: Filtered by `is_preferred_estimate == True`
- **Migration decomposition**: `netmig = domestic_mig + intl_mig` (verified)
- **Units**: Person counts (integer)

### 2.2 Residual Migration Rates (`data/processed/migration/residual_migration_rates.parquet`)

- **Shape**: 9,540 rows x 10 columns
- **Grain**: One row per county x age_group x sex x period
- **Periods**: 5 periods: (2000,2005), (2005,2010), (2010,2015), (2015,2020), (2020,2024)
- **Key columns**: `county_fips` (5-digit), `age_group` (18 groups: 0-4 through 85+), `sex` (Male/Female), `period_start`, `period_end`, `pop_start`, `expected_pop`, `pop_end`, `net_migration`, `migration_rate`
- **Rate units**: Annual rates (already annualized from period rates in the pipeline)
- **Counties**: 53 North Dakota counties

### 2.3 Key Dimensional Mismatch

| Dimension | PEP Components | Residual Rates |
|-----------|:-------------:|:--------------:|
| Time | Annual (26 years) | 5 periods (5-year intervals) |
| Geography | County (3-digit FIPS) | County (5-digit FIPS) |
| Age | None (total only) | 18 age groups |
| Sex | None (total only) | Male/Female |
| Race | None (total only) | None |
| Migration type | Domestic + International | Combined (net) |

The bridge problem: PEP has reliable totals but no demographic detail; the residual method has demographic detail but unreliable totals. The recalibration must merge PEP totals with residual age-sex shapes.

---

## 3. Quantified Divergence Between PEP and Residual

### 3.1 Period-by-Period Comparison

**Benson County (38005)**

| Period | PEP Net Mig | Residual Net Mig | Ratio (Res/PEP) |
|--------|----------:|---------------:|:--------------:|
| 2000-2005 | -334 | -668 | 2.00 |
| 2005-2010 | -492 | -499 | 1.01 |
| 2010-2015 | -83 | -261 | 3.15 |
| 2015-2020 | -473 | -1,318 | 2.79 |
| 2020-2024 | -225 | -586 | 2.61 |

**Sioux County (38085)**

| Period | PEP Net Mig | Residual Net Mig | Ratio (Res/PEP) |
|--------|----------:|---------------:|:--------------:|
| 2000-2005 | -218 | -309 | 1.42 |
| 2005-2010 | -273 | -298 | 1.09 |
| 2010-2015 | +79 | -81 | -1.02 (sign reversal) |
| 2015-2020 | -422 | -817 | 1.93 |
| 2020-2024 | -228 | -393 | 1.73 |

**Rolette County (38079)**

| Period | PEP Net Mig | Residual Net Mig | Ratio (Res/PEP) |
|--------|----------:|---------------:|:--------------:|
| 2000-2005 | -727 | -1,056 | 1.45 |
| 2005-2010 | -765 | -774 | 1.01 |
| 2010-2015 | +65 | -394 | -6.06 (sign reversal) |
| 2015-2020 | -1,088 | -3,199 | 2.94 |
| 2020-2024 | -438 | -1,118 | 2.55 |

### 3.2 Key Observations

1. **The residual method consistently overestimates out-migration**, with the largest divergence in the 2015-2020 period (2.8-2.9x for Benson and Rolette). This period spans the Census 2020, which had known undercount issues on reservations.

2. **Sign reversals in 2010-2015**: PEP shows net in-migration for Sioux (+79) and Rolette (+65), while the residual shows net out-migration (-81 and -394). This is the most extreme form of divergence and represents a period where the residual method is qualitatively wrong.

3. **The 2005-2010 period shows near-agreement** (ratios ~1.0), suggesting the residual method works acceptably when census coverage is consistent (both endpoints from the same census vintage).

4. **For comparison**, a non-reservation declining county (Pembina, 38067) shows a median Res/PEP ratio of ~0.87, i.e., the residual and PEP are roughly in agreement (within 13%). The reservation-specific amplification factor is a distinct phenomenon.

### 3.3 Residual Age-Shape Stability

Cross-period correlations of the age-sex migration shape for Benson County:

| Period Pair | Correlation |
|------------|:----------:|
| 2000-2005 vs 2005-2010 | 0.800 |
| 2000-2005 vs 2010-2015 | 0.483 |
| 2000-2005 vs 2015-2020 | 0.681 |
| 2005-2010 vs 2020-2024 | 0.608 |
| 2010-2015 vs 2020-2024 | 0.615 |
| 2015-2020 vs 2020-2024 | 0.262 |

The age-sex shape is moderately stable across periods (median r = 0.60), with the 2015-2020 period being an outlier (r = 0.26 vs 2020-2024). This suggests the residual method's age-sex allocation is *usable* information even when the total is biased -- the shape captures real age-differential migration patterns (young adult departure, elderly apparent "out-migration"), just at the wrong magnitude.

---

## 4. Bridge Approach Comparison

### 4.1 Option A: Replace Residual Totals with PEP Totals, Then Redistribute Using Residual Age-Sex Shape

**Method**: For each period, compute the residual age-sex share profile (what fraction of total migration each age-sex cell accounts for), then multiply by the PEP total.

```
pep_total = sum(PEP annual netmig for the period)
residual_shares[age, sex] = residual_migration[age, sex] / sum(residual_migration)
recalibrated[age, sex] = pep_total * residual_shares[age, sex]
recalibrated_rate[age, sex] = recalibrated[age, sex] / expected_pop[age, sex]
```

**Pros**:
- Preserves the relative age-sex migration pattern from the residual method
- Anchors the total to PEP (more reliable)
- Simple to implement and explain

**Cons**:
- **Fails on sign reversals**: When PEP is positive and residual is negative (2010-2015 for Sioux and Rolette), the redistributed age-sex values flip sign incoherently. An age group that showed net out-migration would flip to net in-migration, which may not reflect reality.
- Residual shares can be noisy for small populations (high variance in individual cells)
- Propagates any age-specific biases in the residual shape (e.g., elderly "out-migration" from survival rate mismatch)

### 4.2 Option B: Apply a County-Period-Specific Scaling Factor

**Method**: For each period, compute a single scalar `k = PEP_total / Residual_total` and multiply all age-sex rates for that county-period by `k`.

```
k = pep_total / residual_total
recalibrated_rate[age, sex] = residual_rate[age, sex] * k
```

**Pros**:
- Simplest implementation (one scalar per county-period)
- Naturally preserves the age-sex rate profile shape
- Easy to audit and explain
- Produces sensible rates when both totals have the same sign

**Cons**:
- **Fails on sign reversals**: When `k < 0`, all rates flip sign, which is incoherent
- **Fails when residual total is near zero**: `k` becomes very large or undefined
- Does not address the underlying survival rate bias (just masks it)
- The scaling factor `k` varies substantially across periods (0.32 to 0.99), so the averaged result depends heavily on which periods happen to have large/small `k` values

### 4.3 Option C: Use PEP Totals with Rogers-Castro Age Pattern Allocation

**Method**: Discard the residual age-sex rates entirely. Use the PEP net migration total for each period and distribute it to age-sex cells using the Rogers-Castro standard migration age pattern (already implemented in `get_standard_age_migration_pattern()`).

```
pep_total = sum(PEP annual netmig for the period)
rogers_castro_pattern = get_standard_age_migration_pattern(method='rogers_castro')
age_migration = distribute_migration_by_age(pep_total, rogers_castro_pattern)
age_sex_migration = distribute_migration_by_sex(age_migration, sex_ratio=0.5)
recalibrated_rate[age, sex] = age_sex_migration / expected_pop[age, sex]
```

**Pros**:
- Handles sign reversals naturally (Rogers-Castro distributes both positive and negative totals)
- No dependency on residual method output
- Uses the same infrastructure already built for `process_pep_migration_rates()`
- Consistent treatment across all periods

**Cons**:
- Discards county-specific age-sex information from the residual method
- Rogers-Castro assumes a generic migration age pattern; reservation counties may have different mobility patterns (e.g., higher young-adult departure rates than the standard model)
- 50/50 sex ratio may not match observed patterns
- The Rogers-Castro implementation uses single-year ages (0-90) while the residual method uses 5-year age groups; aggregation needed

### 4.4 Option D (Recommended): Hybrid -- PEP-Anchored Residual Shape with Rogers-Castro Fallback

**Method**: A two-tier approach that uses the best available information for each county-period:

1. **When PEP and residual have the same sign**: Apply Option B scaling (`k = PEP_total / Residual_total`) to preserve the county-specific age-sex shape from the residual method.

2. **When they have different signs (sign reversal)**: Fall back to Option C (Rogers-Castro allocation of the PEP total), since the residual age-sex shape is unreliable when the residual method gets the direction of migration wrong.

3. **When the residual total is near zero** (|residual| < threshold, e.g., 10 persons): Fall back to Rogers-Castro allocation, since the shape is dominated by noise.

```python
def recalibrate_period(residual_rates, pep_total, residual_total,
                       rc_pattern, threshold=10):
    if abs(residual_total) < threshold:
        # Near-zero residual: use Rogers-Castro
        return rogers_castro_allocate(pep_total, rc_pattern)

    k = pep_total / residual_total

    if k < 0:
        # Sign reversal: use Rogers-Castro for PEP total
        return rogers_castro_allocate(pep_total, rc_pattern)
    else:
        # Same sign: scale residual rates
        return residual_rates * k
```

**Pros**:
- Uses the best information available in each scenario
- Handles all edge cases (sign reversals, near-zero residuals)
- Preserves county-specific age-sex patterns when the residual method is directionally correct
- Falls back gracefully to the standard demographic model when the residual method fails
- Can be implemented as a post-processing step on existing residual output
- Easy to configure: the list of counties to recalibrate and the threshold are config parameters

**Cons**:
- More complex than a single approach
- Introduces a discontinuity between scaled periods and Rogers-Castro periods in the averaging
- Requires documentation of which periods used which method for audit trail
- The Rogers-Castro pattern may produce slightly different age-sex shapes than the residual method, creating inconsistency across periods

---

## 5. Recommended Approach: Option D (Hybrid)

### 5.1 Rationale

Option D is recommended because it maximizes the use of available information while handling edge cases robustly. The key insight is that the residual method's age-sex shape is *partially* useful -- it captures real age-differential patterns like young-adult departure and elderly mortality-as-out-migration -- but its totals are unreliable for reservation counties. Option D retains the shape when it is trustworthy (same-sign periods) and falls back to a standard demographic model when it is not.

The sign-reversal problem (which affects 2 of 5 periods for Sioux and Rolette) rules out Options A and B as standalone solutions. Option C works for all cases but discards county-specific information unnecessarily. Only Option D handles all cases while preserving the most information.

### 5.2 Implementation Steps

#### Step 1: Add configuration parameter for recalibration counties

In `config/projection_config.yaml`, under `rates.migration.domestic.residual`:

```yaml
residual:
  periods: [[2000, 2005], [2005, 2010], [2010, 2015], [2015, 2020], [2020, 2024]]
  averaging: simple_average
  # NEW: PEP-anchored recalibration for reservation counties
  pep_recalibration:
    enabled: true
    counties: ["38005", "38085", "38079"]  # Benson, Sioux, Rolette
    pep_data_path: "data/processed/pep_county_components_2000_2025.parquet"
    fallback_method: "rogers_castro"  # Used when sign reversal or near-zero residual
    near_zero_threshold: 10           # Persons; below this, use fallback
    rogers_castro_peak_age: 25
    sex_ratio: 0.5
```

#### Step 2: Implement recalibration function

Add a new function to `cohort_projections/data/process/residual_migration.py`:

```python
def apply_pep_recalibration(
    period_rates: pd.DataFrame,
    period: tuple[int, int],
    pep_data: pd.DataFrame,
    counties: list[str],
    rc_pattern: pd.DataFrame,
    near_zero_threshold: float = 10.0,
    sex_ratio: float = 0.5,
) -> pd.DataFrame:
    """Recalibrate residual rates for specified counties using PEP totals.

    For each target county and period:
    1. Compute PEP total net migration for the period
    2. Compute residual total net migration for the period
    3. If same sign and |residual| > threshold: scale all rates by k = PEP/residual
    4. If sign reversal or near-zero: redistribute PEP total using Rogers-Castro

    Non-target counties are passed through unchanged.

    Returns:
        DataFrame with recalibrated rates for target counties.
    """
```

#### Step 3: Integrate into pipeline

In `run_residual_migration_pipeline()`, insert the recalibration step after period dampening and male dampening but before period averaging:

```python
# After dampening, before averaging:
if pep_recal_config.get("enabled", False):
    for period_key, rates in period_results.items():
        period_results[period_key] = apply_pep_recalibration(
            rates, period_key, pep_data, recal_counties,
            rc_pattern, near_zero_threshold, sex_ratio
        )
```

#### Step 4: Rogers-Castro fallback helper

Implement a helper that takes a total net migration value and returns age-group-level rates (not single-year ages) compatible with the residual method's 18 age groups:

```python
def _rogers_castro_to_age_group_rates(
    total_migration: float,
    rc_pattern: pd.DataFrame,       # Single-year age pattern
    expected_pop: pd.DataFrame,      # Population by age_group x sex
    sex_ratio: float = 0.5,
) -> pd.DataFrame:
    """Distribute total migration to age groups using Rogers-Castro.

    Steps:
    1. Distribute total to single-year ages using rc_pattern
    2. Split by sex using sex_ratio
    3. Aggregate to 5-year age groups
    4. Convert to rates using expected_pop
    """
```

#### Step 5: Add metadata tracking

Record which method was used for each county-period combination in the metadata JSON:

```json
{
  "pep_recalibration": {
    "38005": {
      "2000-2005": {"method": "scaled", "k": 0.500},
      "2005-2010": {"method": "scaled", "k": 0.986},
      "2010-2015": {"method": "scaled", "k": 0.318},
      "2015-2020": {"method": "scaled", "k": 0.359},
      "2020-2024": {"method": "scaled", "k": 0.384}
    },
    "38085": {
      "2010-2015": {"method": "rogers_castro", "reason": "sign_reversal"}
    }
  }
}
```

#### Step 6: Tests

- Unit test: scaling factor computation (same sign, different sign, near-zero)
- Unit test: Rogers-Castro fallback produces correct totals after aggregation
- Integration test: full pipeline with recalibration enabled produces rates within expected range
- Regression test: non-target counties are not affected by recalibration

---

## 6. Migration Deceleration Mechanism

### 6.1 Rationale

Even with PEP-anchored rates, the projection model applies a constant migration rate to a shrinking population, producing a linear-looking decline that can be unrealistically steep over long horizons. In reality, out-migration naturally decelerates as:

- The "at-risk" population of young adults (the most mobile cohort) shrinks, reducing absolute out-migration
- The remaining population becomes increasingly composed of people with stronger ties to the community (selection effect)
- Economic and infrastructure changes respond to population decline (feedback loops)

No currently implemented mechanism captures this. The convergence interpolation schedule modestly reduces rates over 20 years (about 0.6-1.0 ppt reduction), but it converges between historical averages -- when all historical windows show substantial out-migration, convergence provides little relief.

### 6.2 Options Considered

#### Option 1: Population Floor

A hard floor below which population cannot fall. Migration is capped to prevent the population from dropping below a threshold (e.g., 50% of base year population).

```
if projected_pop < floor:
    net_migration = floor - pre_migration_pop  # Clamp
```

**Pros**: Simple, guarantees a minimum population.
**Cons**: Creates an abrupt discontinuity; the population sits exactly at the floor indefinitely, which is unrealistic. Not used by any state demography office we are aware of.

#### Option 2: Rate Dampening Proportional to Cumulative Decline

Reduce the migration rate proportional to how much population has already declined relative to the base year.

```
dampening_factor = current_pop / base_year_pop
effective_rate = base_rate * dampening_factor
```

When population has declined to 70% of base year, rates are reduced by 30%.

**Pros**: Smooth, continuous dampening. Produces an asymptotic curve (population approaches a floor but never reaches it). Captures the intuition that smaller populations have fewer people "at risk" of migrating.
**Cons**: The dampening function shape (linear in population ratio) may be too aggressive or too mild. Requires storing base-year population in the projection state.

#### Option 3: Asymptotic Convergence Toward Zero Migration

Model out-migration rate as converging asymptotically toward zero over the projection horizon.

```
effective_rate = base_rate * exp(-lambda * year_offset)
```

Where `lambda` controls the rate of convergence. At `lambda = 0.03`, rates decay by ~60% over 30 years.

**Pros**: Smooth exponential decay; easy to calibrate with a single parameter.
**Cons**: The decay is time-dependent rather than population-dependent, so it does not respond to the actual trajectory. Two counties with the same base rate but different starting populations get the same dampening, which may not be appropriate.

#### Option 4: Hybrid Population-Proportional Dampening with Floor (Recommended)

Combine population-proportional dampening (Option 2) with a soft floor to prevent unrealistic declines:

```python
def apply_migration_deceleration(
    migration_rate: float,
    current_pop: float,
    base_pop: float,
    min_ratio: float = 0.50,      # Soft floor: 50% of base
    dampening_power: float = 1.0,  # Controls aggressiveness
) -> float:
    """Apply population-proportional migration deceleration.

    Effective rate = base_rate * max(pop_ratio, min_ratio) ** power

    When pop_ratio = 1.0 (no decline yet): no dampening
    When pop_ratio = 0.7 (30% decline): rate reduced to 70%
    When pop_ratio = 0.5 (soft floor): rate reduced to 50%, stays there
    """
    pop_ratio = current_pop / base_pop
    dampening = max(pop_ratio, min_ratio) ** dampening_power
    return migration_rate * dampening
```

**Pros**:
- Population-responsive: dampening increases as decline deepens
- Soft floor prevents collapse below a configurable threshold
- `dampening_power` parameter allows calibration (power > 1 = more aggressive)
- Can be applied per-county in the projection loop with minimal code change
- Matches the intuitive dynamic: fewer people remaining means fewer people leaving

**Cons**:
- Adds state dependency to the migration step (needs access to base-year population)
- Requires careful calibration of `min_ratio` and `dampening_power`
- Not standard across state demography offices (though the concept is used in various forms)

### 6.3 Practice at Other State Demography Offices

Based on published methodology documents:

- **Florida (BEBR)**: Uses multiple scenarios (low/medium/high) rather than a deceleration mechanism. The low scenario implicitly dampens by using the most favorable historical period.

- **Texas (TDC)**: Uses an economic-demographic model where migration responds to employment conditions, providing implicit deceleration as declining communities lose economic pull factors.

- **Washington (OFM)**: Uses a "shares-of-growth" method where county migration is allocated proportional to the county's share of recent state growth. This naturally dampens declining counties because their share approaches zero.

- **Census Bureau (national projections)**: Uses constant net international migration rates that do not decelerate, but national-level projections do not face the small-population instability issue.

None of these offices publishes an explicit deceleration formula. The most common approach is to use scenario brackets (low/medium/high) where the "low out-migration" scenario serves as the deceleration alternative. However, for sub-county reservation areas with known data quality issues, explicit dampening is defensible.

### 6.4 Recommended Deceleration Approach

Implement Option 4 (population-proportional dampening with soft floor) as a configurable option in the projection engine, applied per-county during the migration step.

**Configuration**:

```yaml
rates:
  migration:
    deceleration:
      enabled: true
      counties: ["38005", "38085", "38079"]  # Or "all" for universal application
      min_ratio: 0.50           # Soft floor at 50% of base population
      dampening_power: 1.0      # Linear dampening
      base_year: 2025           # Reference year for pop_ratio
```

**Integration point**: In `cohort_projections/core/migration.py`, within `apply_migration()`, after computing migration amounts but before applying them:

```python
if deceleration_config.get("enabled", False):
    county_fips = ...  # From projection context
    if county_fips in deceleration_counties or deceleration_counties == "all":
        base_pop = deceleration_config["base_populations"][county_fips]
        current_pop = population["population"].sum()
        dampening = apply_migration_deceleration(
            current_pop, base_pop,
            min_ratio=deceleration_config["min_ratio"],
            dampening_power=deceleration_config["dampening_power"],
        )
        migration_amount *= dampening
```

---

## 7. Expected Impact on Reservation County Projections

### 7.1 Rate Impact

| County | Current Mean Rate | Option D Rate (est.) | PEP Direct Rate | Reduction |
|--------|------------------:|--------------------:|----------------:|----------:|
| Benson (38005) | -2.52%/yr | -1.20%/yr | -1.00%/yr | ~52% |
| Sioux (38085) | -2.62%/yr | -1.53%/yr | -1.08%/yr | ~42% |
| Rolette (38079) | -2.55%/yr | -1.17%/yr | -0.89%/yr | ~54% |

Option D rates fall between the pure PEP rate and the current residual rate, closer to the PEP rate. The remaining gap vs pure PEP reflects the residual shape's tendency to concentrate migration in certain age-sex cells at higher rates.

### 7.2 Projected 30-Year Population Trajectories (Simplified Model)

These estimates use a compound-rate approximation that ignores natural increase and age-structure effects. Actual cohort-component results will differ somewhat due to fertility, mortality, and age-structure dynamics.

| County | Scenario | Pop 2025 | Pop 2055 (est.) | 30-Yr Change |
|--------|----------|--------:|----------------:|------------:|
| Benson | Current projection | 5,759 | 2,678 | -53.5% |
| Benson | Option D (recalibration only) | 5,759 | 4,012 | -30.3% |
| Benson | Option D + deceleration | 5,759 | ~4,400 | ~-23% |
| Benson | Historical 2000-2020 trend | 5,759 | ~4,883 | -15.2% |
| | | | | |
| Sioux | Current projection | 3,667 | 1,656 | -54.9% |
| Sioux | Option D (recalibration only) | 3,667 | 2,310 | -37.0% |
| Sioux | Option D + deceleration | 3,667 | ~2,750 | ~-25% |
| Sioux | Historical 2000-2020 trend | 3,667 | ~3,261 | -11.1% |
| | | | | |
| Rolette | Current projection | 11,688 | 5,382 | -54.0% |
| Rolette | Option D (recalibration only) | 11,688 | 8,214 | -29.7% |
| Rolette | Option D + deceleration | 11,688 | ~9,200 | ~-21% |
| Rolette | Historical 2000-2020 trend | 11,688 | ~10,327 | -11.6% |

### 7.3 Plausibility Assessment

With both recalibration and deceleration:

- **Benson** moves from -47% to approximately -23%, compared to its historical 2000-2020 trajectory of -15%. A somewhat steeper decline than historical is plausible given the accelerated departure patterns observed in recent years.

- **Sioux** moves from -47% to approximately -25%, compared to a historical trajectory of -11%. This is steeper than historical, reflecting the county's very small population base and high migration volatility.

- **Rolette** moves from -46% to approximately -21%, compared to a historical trajectory of -12%. This represents a modest acceleration from historical trends, which is plausible given the 2015-2019 out-migration spike.

All three counties move from "implausibly steep" (2.7-11x historical) to "moderately steeper than historical" (1.5-2.3x), which is defensible as a projection that accounts for recent trends while not mechanically extrapolating biased estimates.

---

## 8. Implementation Priority and Sequencing

### Phase 1: PEP Recalibration (Immediate)

1. Add `pep_recalibration` config section
2. Implement `apply_pep_recalibration()` in `residual_migration.py`
3. Implement `_rogers_castro_to_age_group_rates()` helper
4. Wire into `run_residual_migration_pipeline()` between dampening and averaging
5. Add metadata tracking for which method was used per county-period
6. Write unit tests and integration test
7. Re-run projections and verify reservation county results
8. Document decision in an ADR

**Estimated effort**: 4-6 hours of implementation + testing

### Phase 2: Migration Deceleration (Near-Term)

1. Add `deceleration` config section
2. Implement `apply_migration_deceleration()` in `cohort_projections/core/migration.py`
3. Wire into `apply_migration()` with county-level base population lookup
4. Add base population storage to projection state
5. Write unit tests (verify dampening factor at various population ratios)
6. Re-run projections and compare with Phase 1 results
7. Calibrate `min_ratio` and `dampening_power` against historical trajectories

**Estimated effort**: 3-4 hours of implementation + testing + calibration

### Phase 3: Validation and Documentation

1. Compare recalibrated projections against historical trajectories for all 53 counties
2. Verify non-reservation counties are not affected
3. Produce comparison charts for the three reservation counties
4. Write review document summarizing the impact
5. Update methodology text in `scripts/exports/_methodology.py`

**Estimated effort**: 2-3 hours

---

## 9. Open Questions

1. **Should recalibration apply to all counties or only reservation counties?** The PEP-residual divergence exists for many counties (Pembina shows similar patterns), but the magnitude is largest for reservation counties. A universal application would be more methodologically consistent but would change more projections.

2. **Should deceleration apply universally?** Population-proportional dampening is conceptually valid for any declining county. Limiting it to three counties creates an ad-hoc exception. A universal but configurable threshold (e.g., only activate when cumulative decline exceeds 20%) would be more defensible.

3. **How should the recalibration interact with convergence interpolation?** Currently, convergence operates on the per-period residual rates. If those rates are recalibrated, the convergence windows (recent, medium, long-term) automatically incorporate the adjustment. No additional change to the convergence logic is needed.

4. **Should the 2015-2020 period be downweighted or excluded?** This period shows the largest PEP-residual divergence and is likely affected by Census 2020 coverage issues. An alternative to recalibration would be to simply exclude this period from the averaging. However, recalibration is a more principled fix because it addresses the root cause rather than discarding data.

5. **What is the appropriate ADR number?** Based on project conventions, the next available ADR is ADR-040 or later. This should be confirmed against `docs/governance/adrs/`.

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-18 |
| **Version** | 1.0 |
| **Related ADR** | TBD (reservation county migration methodology) |
