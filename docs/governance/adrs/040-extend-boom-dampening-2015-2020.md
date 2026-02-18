# ADR-040: Extend Bakken Oil Boom Migration Dampening to 2015-2020 Period

## Status
Accepted

## Date
2026-02-17

## Last Reviewed
2026-02-17

## Scope
Boom-period migration dampening configuration for oil-impacted counties

**Extends**: [ADR-036](036-migration-averaging-methodology.md) (adds 2015-2020 to boom dampening periods)

## Context

### Problem: Undampened 2015-2020 Migration Inflates Oil County Projections

The projection system uses a 0.60x dampening factor on residual migration rates for five oil-impacted counties (Williams 38105, McKenzie 38053, Mountrail 38061, Dunn 38025, Stark 38089) during boom periods. This dampening, documented in ADR-036, reflects the judgment that Bakken oil boom migration conditions are unlikely to repeat and should not anchor long-term projections.

Currently, only 2005-2010 and 2010-2015 are configured as boom periods. However, the **2015-2020 period is not dampened** despite still reflecting elevated boom-era migration. McKenzie County recorded +2,612 net migrants in 2015-2020 — higher than its dampened 2010-2015 figure. This creates an inconsistency: the supposedly "post-boom" period carries a stronger migration signal than the dampened boom period itself.

### Impact on Projections

The convergence interpolation schedule (Census Bureau method) uses a medium window [2014, 2025] that maps to periods (2010-2015), (2015-2020), and (2020-2024). Because 2015-2020 is undampened, the medium-window average for oil counties is strongly positive (e.g., +0.016 for McKenzie). Under the convergence schedule, this medium-window rate is held for 10 years (years 6-15) of the 30-year projection.

The result is unrealistic long-term growth:
- **McKenzie**: projects +114% growth over 30 years
- **Williams**: projects +74% growth over 30 years
- **State average**: +28% growth over 30 years

These projections are contradicted by the most recent data. The 2020-2024 period shows **negative** net migration for both counties (McKenzie: -893, Williams: -3,033), indicating the boom-era migration pattern has fully reversed.

### Requirements

- Reduce the influence of boom-adjacent migration on oil county projections
- Maintain consistency with the existing dampening methodology
- Minimize collateral impact on non-oil counties
- Config-only change preferred (no code modifications)

## Decision

### Decision: Add 2015-2020 to Boom Periods List

**Decision**: Add `[2015, 2020]` to the `boom_periods` list in `projection_config.yaml`, so the 0.60x dampening factor also applies to the 2015-2020 residual migration period for the five oil-impacted counties.

```yaml
dampening:
  enabled: true
  factor: 0.60
  counties:
    - "38105"  # Williams
    - "38053"  # McKenzie
    - "38061"  # Mountrail
    - "38025"  # Dunn
    - "38089"  # Stark
  boom_periods:
    - [2005, 2010]
    - [2010, 2015]
    - [2015, 2020]  # ADR-040: boom-adjacent period, undampened rates exceeded dampened boom
```

This is a **config-only change**. No code modifications are needed — the existing dampening logic in the residual migration pipeline already reads `boom_periods` from the config and applies the factor to any listed period.

**Rationale**:

1. **2015-2020 is objectively boom-adjacent**: Infrastructure investment, family reunification, and economic momentum from the 2011-2015 Bakken boom carried through 2019. The period does not represent a return to pre-boom migration conditions.

2. **Internal consistency**: McKenzie's 2015-2020 net migration (+2,612) exceeded its dampened 2010-2015 figure. Leaving 2015-2020 undampened while dampening 2010-2015 creates a logical inconsistency where the "post-boom" period has a stronger migration signal than the dampened boom period.

3. **Surgically targeted**: The dampening affects only 5 of 53 counties, and only for one additional period. All other counties are unaffected.

4. **Methodologically consistent**: This extends an existing, documented dampening approach rather than introducing a new mechanism. The same factor (0.60x) and the same county list are used.

**Alternatives Considered**:

- **Narrowing the medium convergence window** (e.g., [2019, 2025] instead of [2014, 2025]): Rejected because this would affect all counties, not just the oil-impacted ones. It would also discard valuable medium-term trend information for counties where the 2015-2020 period is representative.
- **Rate caps** (capping migration rates at a maximum value): Rejected because cap thresholds are difficult to calibrate objectively. A cap that restrains McKenzie might also clip legitimate growth in other counties.
- **Secondary dampening layer** (a separate dampening factor for "boom-adjacent" periods): Rejected because it adds configuration complexity without meaningful analytical benefit. The existing 0.60x factor is already calibrated to match the SDC 2024 methodology.

## Consequences

### Positive
1. **Realistic oil county projections**: Medium-window averages for oil counties will decrease, producing 30-year growth trajectories consistent with recent trends rather than boom-era migration
2. **McKenzie expected to drop from +114% to approximately +40-60%** over 30 years; Williams from +74% to approximately +30-50%
3. **No code changes**: The fix is entirely in configuration, reducing risk of regressions
4. **Consistent treatment**: All three boom and boom-adjacent periods (2005-2010, 2010-2015, 2015-2020) are now dampened uniformly

### Negative
1. **Judgment call**: Classifying 2015-2020 as "boom-adjacent" is a subjective assessment, even though the data supports it. Future users should document their reasoning if further periods are added.
2. **Dampening is a blunt instrument**: The 0.60x factor is applied uniformly to all age-sex-race cells. In reality, different demographic groups may have experienced different levels of boom-related migration in 2015-2020.

### Pipeline Rerun Required

Adding this period to the config requires rerunning:
1. **Step 01**: Residual migration computation (to apply dampening to 2015-2020 rates)
2. **Step 01b**: Convergence interpolation (to recompute medium-window averages)
3. **Step 02**: Population projections (to generate updated projections)
4. **Step 03**: Export workbooks (to reflect updated results)

## Implementation Notes

### Key Files
- `config/projection_config.yaml`: `migration.dampening.boom_periods` list updated to include `[2015, 2020]`

### Configuration Integration
The residual migration pipeline reads `boom_periods` from the dampening config block and applies the `factor` (0.60) to any period that matches. Adding `[2015, 2020]` to the list causes the existing code path to dampen that period automatically for the listed counties. No new code paths are introduced.

### Testing Strategy
1. **Spot-check dampened rates**: Verify that McKenzie and Williams 2015-2020 residual migration rates are reduced by 0.60x after rerunning step 01
2. **Convergence validation**: Verify that medium-window averages for oil counties decrease after rerunning step 01b
3. **Projection reasonableness**: Confirm McKenzie and Williams 30-year growth projections fall within the +30-60% range
4. **Non-oil county regression**: Verify that projections for non-oil counties (e.g., Cass, Burleigh, Grand Forks) are unchanged

## References

1. **ADR-036: Migration Averaging Methodology** -- Established the multi-period averaging approach and original boom dampening configuration
2. **Census PEP Components of Change (2000-2025)**: `data/processed/pep_county_components_2000_2025.parquet` -- source for 2015-2020 and 2020-2024 net migration figures
3. **SDC 2024 Methodology**: Source of the 0.60x dampening factor calibration

## Revision History

- **2026-02-17**: Initial version (ADR-040) -- Extend boom dampening to 2015-2020 period

## Related ADRs

- **ADR-036: Migration Averaging Methodology** -- Defines the multi-period averaging and boom dampening framework; extended by this ADR
- **ADR-037: CBO-Grounded Scenario Methodology** -- Scenario-level migration adjustments (operates independently of boom dampening)
- **ADR-039: International-Only Migration Factor** -- Refines how CBO migration factors decompose into domestic/international components

## Related Reviews

- [Bakken Migration Dampening Review](../../reviews/2026-02-17-bakken-migration-dampening-review.md): Detailed analysis of oil county growth rates and dampening effectiveness
