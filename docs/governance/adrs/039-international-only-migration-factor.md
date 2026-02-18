# ADR-039: International-Only Migration Factor for Restricted Growth Scenario

## Status
Accepted

## Date
2026-02-17

## Last Reviewed
2026-02-17

## Scope
Restricted growth scenario migration factor application

**Amends**: [ADR-037](037-cbo-grounded-scenario-methodology.md) (refines how time-varying migration factors are applied)

## Context

### Problem: CBO Migration Factor Applied to Total Migration

ADR-037 introduced CBO-grounded time-varying migration factors for the restricted growth scenario. These factors model the impact of federal immigration enforcement on net migration, with a schedule ranging from 0.20 (80% reduction) in 2025 to 1.00 (no adjustment) by 2030.

The original implementation applied these factors to **total** net migration (domestic + international combined). This is incorrect because:

1. **CBO factors model immigration policy**: The CBO Demographic Outlook quantifies the impact of deportations, border enforcement, and visa policy changes. These policies affect international migration only. Domestic (interstate/intercounty) migration is driven by employment, housing, and lifestyle factors unrelated to federal immigration enforcement.

2. **Empirical evidence from V2025 PEP data**: Census PEP components of change data (2020-2025) for North Dakota shows that international and domestic migration move independently:
   - International migration was consistently positive (+30 to +4,083 per year)
   - Domestic migration swung between -4,716 and +6,374, driven by economic cycles (oil, agriculture, military)
   - In recent years (2023-2025), international migration accounted for approximately 91% of total net migration at the state level

3. **Distortion of county-level projections**: For counties with substantial domestic out-migration offset by international in-migration (e.g., rural counties with refugee resettlement), applying the factor to total migration incorrectly reduces the domestic out-migration component, producing unrealistic results.

### Requirements

- The CBO time-varying factor must apply only to the international component of migration
- Domestic migration rates must be unaffected by the restricted growth scenario's migration factor
- The approach must work with the existing residual migration rate pipeline, which produces total (combined) migration rates without a domestic/international breakdown at the age-sex level
- The solution must be transparent and auditable (documented in config)

### Challenges

- The projection engine operates on total migration rates (a single `migration_rate` column per age-sex-race cell). Domestic and international components are not tracked separately through the engine.
- The residual migration method computes total net migration rates from population changes. There is no clean separation of domestic and international at the age-group level.
- PEP data provides county-level absolute migration counts (domestic and international) but not age-sex distributed rates.
- The ratio `intl_mig / netmig` is unstable for counties and years where net migration is near zero.

## Decision

### Decision 1: Decompose Using State-Level International Share

**Decision**: Apply the CBO time-varying factor to only the international portion of total migration rates, using a state-level international share parameter (`intl_share`) to decompose the rate.

**Formula**:
```
effective_factor = 1 - intl_share * (1 - factor)
adjusted_rate = base_rate * effective_factor
```

Where:
- `factor` is the CBO time-varying factor for the year (e.g., 0.20 in 2025)
- `intl_share` is the proportion of total net migration that is international
- `effective_factor` is the combined multiplier applied to the total rate

This formula is equivalent to:
```
domestic_part = base_rate * (1 - intl_share)     # unchanged
intl_part     = base_rate * intl_share * factor   # scaled by CBO factor
adjusted_rate = domestic_part + intl_part
```

**Rationale**:
- Correctly isolates the policy effect on international migration while preserving domestic migration
- Works with existing total-rate pipeline architecture without requiring separate domestic/international rate columns
- The multiplicative decomposition preserves the sign and direction of the rate for all counties
- Mathematically reduces to the original implementation when `intl_share = 1.0`, providing backward compatibility

### Decision 2: State-Level Share Rather Than Per-County

**Decision**: Use a single state-level `intl_share` value (0.91) for all counties, computed from PEP components of change data for 2023-2025.

**Computation**:
```
PEP state totals, 2023-2025:
  intl_mig  = +10,051
  netmig    = +11,070
  intl_share = 10,051 / 11,070 = 0.91
```

**Rationale**:
- Per-county shares are extremely noisy. Many counties have near-zero net migration, producing undefined or extreme ratios. The share ranges from -12.0 to +4.3 across counties.
- The CBO factors model a national/state-level policy effect. Using a state-level share is the appropriate geographic resolution for a federal policy scenario.
- The 2023-2025 period captures the most recent migration composition and aligns with the convergence interpolation recent window.

### Decision 3: Store Share in Configuration

**Decision**: Store `intl_share` as a parameter in the restricted growth scenario config in `projection_config.yaml`, alongside the existing time-varying schedule.

```yaml
restricted_growth:
  migration:
    type: "time_varying"
    schedule:
      2025: 0.20
      2026: 0.37
      # ...
    default_factor: 1.00
    intl_share: 0.91  # ADR-039: proportion of net migration that is international
```

**Rationale**:
- Transparent and auditable: the value and its source are documented in the config
- Easy to update when new PEP data becomes available
- The engine reads it from the scenario dict at runtime, requiring no additional data loading

**Implementation**:

In `cohort_projections/core/migration.py`, the `apply_migration_scenario()` function reads `intl_share` from the scenario dict when `type == "time_varying"`:

```python
intl_share = scenario.get("intl_share", 1.0)
effective_factor = 1.0 - intl_share * (1.0 - factor)
adjusted_rates[migration_col] = adjusted_rates[migration_col] * effective_factor
```

The default of `intl_share = 1.0` preserves backward compatibility: if the parameter is missing, the entire rate is treated as international (same as original behavior).

**Alternatives Considered**:

- **Per-county shares**: Rejected due to noise in small-county ratios and the state-level nature of the policy being modeled.
- **Additive correction (subtract international rate)**: Would require computing per-capita international migration rates and distributing them to age-sex groups. More complex, and PEP doesn't provide age-sex international migration breakdowns.
- **Separate domestic/international rate columns through the engine**: Would require significant refactoring of the entire migration pipeline. Deferred as a potential future enhancement.

## Consequences

### Positive
1. **Correct policy modeling**: The CBO migration factor now correctly targets only international migration, matching the scope of the policy it models
2. **Preserved domestic migration**: Counties with significant domestic migration patterns (oil counties, college towns, military bases) are no longer incorrectly affected by immigration policy factors
3. **Minimal code changes**: The fix is a single formula change in one function, plus a config parameter addition
4. **Backward compatible**: The `intl_share` parameter defaults to 1.0 if omitted, preserving the original behavior for any configs that don't include it
5. **Transparent**: The share value, its computation, and rationale are documented in config comments and this ADR

### Negative
1. **Approximation**: Using a single state-level share for all counties is an approximation. Counties with very different domestic/international migration compositions will have slightly inaccurate decompositions.
2. **Static share**: The share is computed from a specific recent period and doesn't vary over the projection horizon. In reality, the composition could shift over time.
3. **Assumes proportional age distribution**: The share is applied uniformly to all age-sex cells, assuming international migration has the same age distribution as total migration. This is approximate.

### Risks and Mitigations

**Risk**: The `intl_share` value becomes stale as new PEP data is released.
- **Mitigation**: The value is stored in config (not hard-coded) and includes documentation of how it was computed. When PEP V2026 data is available, recompute using the same methodology.

**Risk**: For counties where domestic and international migration have opposite signs, the effective factor could overshoot or undershoot.
- **Mitigation**: The state-level share averages over all counties, smoothing out county-level sign conflicts. For the state aggregate, the decomposition is accurate.

## Implementation Notes

### Key Functions/Classes
- `apply_migration_scenario()` in `cohort_projections/core/migration.py`: Updated to read `intl_share` from the scenario dict and compute the effective factor
- `config/projection_config.yaml`: `restricted_growth.migration.intl_share` added

### Configuration Integration
The `intl_share` parameter is added to the existing `time_varying` migration dict in the restricted growth scenario config. The YAML config loader passes it through as part of the scenario dict, which flows unchanged to the engine's `apply_migration_scenario()` function.

### Testing Strategy
1. **Unit tests**: Verify `apply_migration_scenario()` computes correct effective factors for various `intl_share` and `factor` combinations
2. **Integration tests**: Run restricted growth scenario for a representative county; verify domestic migration rates are unchanged while international portion is reduced
3. **Regression tests**: Verify baseline and high growth scenarios are unaffected
4. **Validation**: Compare restricted growth results before and after the change; the restricted scenario should now show less total migration reduction (since domestic is no longer penalized)

## References

1. **CBO January 2026 Demographic Outlook** (Publication 61879): Source of time-varying migration factors
2. **Census PEP Components of Change (2000-2025)**: `data/processed/pep_county_components_2000_2025.parquet` -- source for `intl_share` computation
3. **ADR-037: CBO-Grounded Scenario Methodology**: Original implementation of time-varying migration factors

## Revision History

- **2026-02-17**: Initial version (ADR-039) -- Apply CBO migration factors to international migration only

## Related ADRs

- **ADR-037: CBO-Grounded Scenario Methodology** -- Introduced the time-varying migration schedule; amended by this ADR
- **ADR-035: Census PEP Components of Change for Migration Inputs** -- PEP data source used to compute `intl_share`
- **ADR-036: Migration Averaging Methodology** -- Defines baseline migration rate computation
- **ADR-018: Immigration Policy Scenario Methodology** -- Original policy scenario framework

## Related Reviews

- [Vintage 2025 Census Data Analysis](../../reviews/2026-02-17-vintage-2025-census-data-analysis.md): Analysis of V2025 PEP data that informed the `intl_share` computation
