# ADR-050: Restricted Growth Additive Migration Adjustment

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-23

## Scope
Replace multiplicative CBO migration factor with additive adjustment for the restricted growth scenario to fix scenario ordering violations

**Amends**: [ADR-039](039-international-only-migration-factor.md) (fixes the sign-interaction problem identified but insufficiently mitigated)

## Context

### Problem: Multiplicative Factor on Net-Negative Migration Produces Ordering Violations

ADR-039 introduced the formula:

```
effective_factor = 1 - intl_share * (1 - factor)
adjusted_rate = base_rate * effective_factor
```

This formula applies a multiplicative factor to **total** net migration rates, using `intl_share = 0.91` to isolate the international component. In 2025, with `factor = 0.20`:

```
effective_factor = 1 - 0.91 * (1 - 0.20) = 0.272
```

For a county with net migration rate of **-0.05** (net out-migration):
```
adjusted = -0.05 * 0.272 = -0.0136
```

The rate becomes **less negative**, meaning the county **retains more people** under restricted growth than baseline. This is the opposite of the scenario's intent.

### Scale of the Violation

| Metric | Count |
|--------|:-----:|
| Counties with restricted > baseline (any year) | **39 of 53** |
| Total year-county violations | 248 |
| Concentration | 2026-2040 (before CBO factor reaches 1.0) |
| Max violation at 2055 | Slope County: +4.8% |
| Counties with persistent 2055 violation | 1 (Slope) |

### This Is the Same Bug Class as the High Growth Inversion (ADR-046)

ADR-046 documented the identical structural problem for the `high_growth` scenario: a multiplicative `+15_percent` factor on net-negative migration rates produced lower populations than baseline for 45 of 53 counties. That was classified as a **design bug** and fixed by switching to an additive approach using BEBR convergence rates.

The restricted growth scenario has the same class of bug with opposite polarity:
- High growth: multiplying negative rates by >1.0 makes them more negative (lower population — wrong direction)
- Restricted growth: multiplying negative rates by <1.0 makes them less negative (higher population — wrong direction)

### Why ADR-039's Mitigation Was Insufficient

ADR-039 acknowledged the risk (line 153-154): *"For counties where domestic and international migration have opposite signs, the effective factor could overshoot or undershoot."* The stated mitigation was: *"The state-level share averages over all counties, smoothing out county-level sign conflicts."*

This mitigation is insufficient because:
1. 39 of 53 individual counties have the wrong ordering in early projection years
2. The statewide `intl_share` of 0.91 produces an extreme `effective_factor` of 0.272 in 2025
3. The "averaging" happens at the state level, but projections run at the county level where the sign conflict manifests

### The Conceptual Flaw

The decomposition `domestic_part = base_rate * (1 - intl_share)` is only valid when the total rate and each component share the same sign. For a declining rural county:
- Actual domestic migration: approximately -3% to -6% (people leaving for cities)
- Actual international migration: approximately +0.5% to +2% (refugees, immigrant workers)
- Actual net migration: approximately -3% to -4%

When immigration enforcement reduces international in-migration, the county's net migration should become *more negative* (losing the international arrivals who partially offset domestic departures). The multiplicative formula makes it *less negative* by treating 91% of the net out-migration as "international" and reducing its magnitude.

## Decision

### Switch to Additive Migration Reduction (Option B)

**Decision**: Replace the multiplicative `effective_factor` approach with an additive per-capita rate decrement, computed from the CBO schedule and state-level international migration volume.

**Formula**:

```
state_intl_migration = 10,051  # From PEP 2023-2025 annual average
annual_reduction = state_intl_migration * (1 - factor)  # Persons/year not arriving
reduction_rate = annual_reduction / state_population     # Per-capita rate decrement
adjusted_rate = base_rate - reduction_rate               # Always subtracts
```

For 2025 (`factor = 0.20`):
```
annual_reduction = 10,051 * (1 - 0.20) = 8,041 persons
reduction_rate = 8,041 / 799,358 = 0.01006 per capita
adjusted_rate = base_rate - 0.01006
```

The reduction is distributed proportionally to each county based on population share, then distributed uniformly across age-sex cells within each county.

**Why this guarantees correct ordering**: The additive adjustment always **subtracts** from the base rate, regardless of its sign. A county with baseline rate of -0.05 gets adjusted to -0.0601, which is more negative (fewer people). A county with baseline rate of +0.03 gets adjusted to +0.0199, which is less positive (also fewer people). The ordering `restricted ≤ baseline` holds universally.

### Configuration

```yaml
restricted_growth:
  fertility: "-5_percent"
  migration:
    type: "additive_reduction"
    schedule:
      2025: 0.20
      2026: 0.37
      2027: 0.55
      2028: 0.78
      2029: 0.91
    default_factor: 1.00
    reference_intl_migration: 10051  # Annual international migration (PEP 2023-2025)
    reference_population: 799358     # State population at base year
```

### Implementation

In `cohort_projections/core/migration.py`, add handling for `type: "additive_reduction"`:

```python
if scenario.get("type") == "additive_reduction":
    factor = schedule.get(year, default_factor)
    if factor < 1.0:
        ref_intl = scenario.get("reference_intl_migration", 0)
        ref_pop = scenario.get("reference_population", 1)
        annual_reduction = ref_intl * (1.0 - factor)      # persons/year not arriving
        reduction_rate = annual_reduction / ref_pop         # per-capita rate decrement
        adjusted_rates[migration_col] -= reduction_rate     # subtract from all cells
```

Since migration rates are already per-capita, the same `reduction_rate` applies uniformly to every cell. The total person-reduction for a county is proportional to its population (`reduction_rate * county_pop`), which correctly distributes the statewide reduction by population share without any explicit per-county scaling.

### Why Additive, Not "Only Apply to Positive Rates" (Option D)

A simpler fix would be to apply the multiplicative factor only when the base rate is positive:

```python
positive_mask = adjusted_rates[migration_col] > 0
adjusted_rates.loc[positive_mask, migration_col] *= effective_factor
```

This was considered as a minimal fix but rejected because:
1. It treats negative-migration counties as if they receive zero international migration, which is incorrect — most do receive some international in-migration
2. It creates a discontinuity at rate = 0 (rates just above zero are heavily reduced, rates just below are untouched)
3. The additive approach is more principled and consistent with how ADR-046 resolved the analogous high_growth problem

### Alternatives Considered

| Option | Description | Verdict |
|--------|-------------|---------|
| A: Apply factor only to positive rates | Simple mask on positive rates | Rejected — incorrect for counties with offsetting domestic/international flows |
| **B: Additive reduction (chosen)** | Subtract per-capita rate from CBO schedule | **Selected** — guarantees ordering, consistent with ADR-046 approach |
| C: Separate component modeling | Track domestic/international through engine | Rejected — requires significant refactoring (noted in ADR-039) |
| D: Directional factor | Multiplicative for positive, unchanged for negative | Rejected — discontinuity at zero, ignores international component of negative-migration counties |

## Consequences

### Positive

1. **Eliminates ordering violations**: `restricted ≤ baseline` is guaranteed for all 53 counties, all 30 years, all age-sex cells
2. **Correct directional effect**: Counties losing international migrants see more negative net migration (correct)
3. **Consistent with ADR-046**: Same class of fix (additive vs multiplicative) for the same class of problem
4. **CBO convergence preserved**: When `factor = 1.0` (2030+), `reduction = 0` and rates are identical to baseline
5. **Empirically grounded**: The reduction magnitude derives from PEP international migration data, not arbitrary parameters

### Negative

1. **Uniform distribution across cells**: The per-capita reduction is distributed uniformly across age-sex cells. In reality, international migration has a distinct age pattern (concentrated in 20-40 year-olds). A future enhancement could use an age-weighted distribution.
2. **Static reference values**: `reference_intl_migration` and `reference_population` are fixed in config. They should be updated when new PEP data is available.
3. **Breaks backward compatibility**: The `intl_share` parameter and multiplicative formula in `migration.py` become unused for restricted_growth. The old code path should be retained for backward compatibility (e.g., legacy configs) but is no longer the default.

### Expected Impact

| Metric | Before (multiplicative) | After (additive) |
|--------|:-----------------------:|:----------------:|
| Counties with restricted > baseline (2030) | 39 | **0** |
| Slope County 2055 ordering | restricted > baseline by +4.8% | restricted < baseline |
| State total 2055 restricted_growth | 926,376 | ~920,000-925,000 (slightly lower) |
| Scenario spread (restricted to high) | ~94,000 | ~95,000-100,000 |

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `cohort_projections/core/migration.py` | Add `additive_reduction` handler in `apply_migration_scenario()` |
| `config/projection_config.yaml` | Change restricted_growth `migration.type` to `additive_reduction`; add `reference_intl_migration` and `reference_population` |
| `scripts/pipeline/02_run_projections.py` | Pass county population to `apply_migration_scenario()` for per-county scaling |

### Testing Strategy

1. **Unit test**: Verify `apply_migration_scenario()` with `additive_reduction` always produces rates ≤ baseline for both positive and negative base rates
2. **Ordering test**: Run restricted_growth for all 53 counties; verify `restricted ≤ baseline` at every year for every county
3. **Magnitude test**: State-level 2055 restricted_growth should be 3-8% below baseline (consistent with CBO scenario intent)
4. **Convergence test**: After 2030 (factor = 1.0), restricted rates exactly equal baseline rates
5. **Regression**: Baseline and high_growth scenarios are unaffected

### Pipeline Rerun Required

1. **Step 02**: Projections (new migration scenario type)
2. **Step 03**: Exports

### Interaction with ADR-046 (High Growth)

The restricted growth fix is independent of the high growth fix. High growth uses BEBR convergence rates (baked into the convergence file), while restricted growth uses an in-engine additive reduction. Both fixes ensure correct scenario ordering from different directions:
- Restricted ≤ Baseline: guaranteed by additive reduction (this ADR)
- Baseline ≤ High: guaranteed by BEBR additive increment (ADR-046)

## References

1. **ADR-039**: International-Only Migration Factor — the current (flawed) implementation
2. **ADR-046**: High Growth BEBR Convergence — analogous fix for the high_growth scenario
3. **ADR-037**: CBO-Grounded Scenario Methodology — defines the three-scenario framework
4. **Census PEP Components**: `data/processed/pep_county_components_2000_2025.parquet` — source for `reference_intl_migration`
5. **CBO January 2026 Demographic Outlook** (Publication 61879): Source of time-varying migration factors
6. **Sanity Check Finding**: 39 of 53 counties have restricted > baseline in early years

## Implementation Results (2026-02-23)

### Changes Made

| File | Change |
|------|--------|
| `cohort_projections/core/migration.py` | Added `additive_reduction` handler in `apply_migration_scenario()` (lines 215-235). Computes `reduction_rate = ref_intl * (1 - factor) / ref_pop` and subtracts uniformly from all cells. Comprehensive docstring added explaining formula and ordering guarantee. |
| `config/projection_config.yaml` | Changed restricted_growth `migration.type` from `time_varying` to `additive_reduction`. Added `reference_intl_migration: 10051` and `reference_population: 799358`. Removed `intl_share` parameter. |
| `tests/test_core/test_migration.py` | Added `TestAdditiveReductionScenario` class with 11 tests covering: positive rates, negative rates (the critical bug case), mixed-sign rates, correct magnitude, factor=1.0 convergence, schedule variation by year, net_migration column support, column preservation, zero-reference edge case, and uniform reduction across cells. Also added `TestTimeVaryingMigrationScenario` class (2 tests) to ensure backward compatibility of the old multiplicative path. |

### Verification

- **Ordering guarantee**: The additive decrement always subtracts from the base rate, so `restricted <= baseline` holds for all signs:
  - Positive rate 0.05 -> 0.05 - 0.01006 = 0.0399 (less positive, correct)
  - Negative rate -0.05 -> -0.05 - 0.01006 = -0.0601 (more negative, correct)
  - Zero rate 0.00 -> 0.00 - 0.01006 = -0.01006 (becomes negative, correct)
- **Convergence**: When `factor = 1.0` (year 2030+), `reduction_rate = 0`, so restricted rates exactly equal baseline rates.
- **Old code preserved**: The `time_varying` (multiplicative) code path in `apply_migration_scenario()` is retained for backward compatibility but is no longer used by the restricted_growth scenario.
- **Pipeline integration**: `apply_scenario_rate_adjustments()` in `02_run_projections.py` correctly detects dict-based migration configs (including `additive_reduction`) and defers per-year adjustment to the engine.

### Key Design Notes

1. **Per-capita uniformity**: The reduction rate is applied identically to every age-sex-race cell. This is a deliberate simplification; international migration has a distinct age pattern (concentrated in ages 20-40). A future enhancement could use age-weighted distribution.
2. **No per-county scaling needed**: Because migration rates are already per-capita, applying the same `reduction_rate` to every cell automatically distributes the total person-reduction proportionally to each county's population. A county with 10x the population loses 10x the people.
3. **Static reference values**: `reference_intl_migration` (10,051) and `reference_population` (799,358) are derived from PEP 2023-2025 data and should be updated when new PEP vintages are released.

## Revision History

- **2026-02-23**: Accepted and implemented — added `additive_reduction` handler, 13 new unit tests, updated config; corrected pseudocode in Implementation section to match actual per-capita approach
- **2026-02-18**: Initial version (ADR-050) — Additive migration reduction for restricted growth

## Related ADRs

- **ADR-039: International-Only Migration Factor** — Amended by this ADR; the `intl_share` multiplicative approach is replaced
- **ADR-037: CBO-Grounded Scenario Methodology** — Defines the restricted growth scenario intent and validation requirements
- **ADR-046: High Growth BEBR Convergence** — Same class of fix (multiplicative → additive) for the opposite scenario
- **ADR-035: Census PEP Components** — Source data for the reference international migration parameter
