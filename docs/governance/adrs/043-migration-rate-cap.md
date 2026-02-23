# ADR-043: Age-Aware Migration Rate Cap for Convergence Rates

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Age-aware asymmetric rate cap applied during convergence interpolation to clip statistical noise in small-county migration rate cells

**Motivated by**: [Finding 2 (Oil County Growth)](../../reviews/2026-02-18-sanity-check-investigations/finding-2-oil-county-growth.md) and [Finding 3 (Reservation Declines)](../../reviews/2026-02-18-sanity-check-investigations/finding-3-reservation-county-declines.md) from the 2026-02-18 projection output sanity check

## Context

### Problem: Extreme Migration Rates from Small-Population Statistical Noise

Individual age-sex cells in the convergence rate data carry extreme migration rates driven by small-population statistical noise and oil-boom residuals. At year offset 10 (medium hold), rates range from -15.6% to +16.0%, with the most extreme values concentrated in small counties (population under 5,000) where a single family moving in or out can produce double-digit percentage swings.

Key findings from the [design investigation](../../reviews/2026-02-18-sanity-check-investigations/design-migration-rate-cap.md):

1. **71% of cells above +5%** are in small counties (population under 5,000), driven by tiny base populations (often 5-30 people in a cell).

2. **College-town rates (ages 15-24) are genuine.** University enrollment produces migration rates of 10-14% in Cass (NDSU), Grand Forks (UND), and Ward (Minot State). These reflect real institutional dynamics stable for decades.

3. **Non-college-age rates above 8% are almost exclusively noise.** Outside ages 15-24, rates above 8% at the medium hold are found only in 7 small counties with tiny cell populations plus McKenzie County with 1 cell marginally above (8.45%).

4. **A rate cap alone is insufficient for oil-county growth.** McKenzie County's +45% 30-year growth is driven by many moderately positive cells (25-29 at 8.5%, 5-9 at 3.9%, 30-34 at 3.7%), not a single extreme outlier.

### Requirements

- Clip statistically implausible outlier rates without distorting legitimate patterns
- Preserve college-town university enrollment dynamics (10-14% rates for ages 15-24)
- Leave Cass/Fargo completely unaffected (+20.5% 30-year growth)
- Complement boom dampening (ADR-040), not replace it
- Be configurable and disableable

## Decision

### Decision 1: Age-Aware Asymmetric Rate Cap

**Decision**: Apply a two-tier symmetric cap during convergence interpolation:
- Ages 15-24 (college ages): cap at +/-15%
- All other ages: cap at +/-8%

The cap is applied after computing the interpolated rate for each year offset but before storing it in the results, so it catches all three convergence phases without modifying underlying window averages.

**Rationale**:
- The 8% general threshold sits at P99 of the medium-term distribution for non-college ages (P99 = 8.13%, P99.5 = 9.97%), targeting only the statistical tail
- The 15% college threshold provides headroom above the highest legitimate university enrollment rates (Cass 20-24 Female at 13.2%, Grand Forks 20-24 Male at 14.0%)
- This clips 2.4% of all cells (1,372 of 57,240 across all year offsets)
- Cass/Fargo is completely unaffected; Billings sees the largest correction (-16.6 percentage points on 30-year growth)

**Implementation**:
```yaml
# config/projection_config.yaml
rates:
  migration:
    interpolation:
      rate_cap:
        enabled: true
        college_ages: ["15-19", "20-24"]
        college_cap: 0.15
        general_cap: 0.08
```

```python
# In _apply_rate_cap():
college_mask = age_groups.isin(college_ages)
capped = rate.clip(lower=-general_cap, upper=general_cap)
capped[college_mask] = rate[college_mask].clip(lower=-college_cap, upper=college_cap)
```

**Alternatives Considered**:
- Symmetric cap (single threshold for all ages): Any cap that clips oil-county noise also clips legitimate college-town dynamics. Rejected.
- Population-tiered cap (different thresholds by county size): Added implementation complexity without clearer demographic justification. Rejected.
- Relative cap (N times state average): Too aggressive at any reasonable multiplier since state average is often near zero. Rejected.
- Asymmetric cap (different positive/negative limits): Cannot distinguish college-town 20-24 from oil-county 25-29 rates. Rejected.

See design document Sections 8.1-8.4 for detailed evaluation of each alternative.

### Decision 2: Insertion Point in Pipeline

**Decision**: Apply the cap inside `calculate_age_specific_convergence()`, after the interpolation computation for each year offset and before storing the result in the output dict.

**Rationale**:
1. Catches all three convergence phases (recent-to-medium, medium hold, medium-to-long-term)
2. Operates on the final rate used by the projection engine
3. Does not modify underlying window averages, preserving data lineage

### Decision 3: Complementary to Boom Dampening

**Decision**: The rate cap and boom dampening (ADR-040) are complementary mechanisms, not substitutes. Both remain active.

| Mechanism | Target | Scope |
|-----------|--------|-------|
| Boom dampening (ADR-040) | Reduce oil-boom period rates before averaging | Specific counties (dampening list) |
| Rate cap (this ADR) | Clip extreme individual cells after averaging | All counties, automatic |

**Rationale**: The rate cap handles statistical noise that dampening cannot address (e.g., Logan County 25-29 Female at 16% has nothing to do with oil). Dampening handles the broad pattern of elevated working-age rates in oil counties that the cap cannot address (McKenzie's many cells at 3-8%).

## Consequences

### Positive
1. **Eliminates implausible small-county outliers**: Billings County 30-year growth drops from +55% to +38%; Logan County extreme 25-29 Female rate capped from 16% to 8%
2. **Preserves legitimate patterns**: Cass/Fargo unchanged at +20.5%; Grand Forks benefits from capping extreme 25-29 exit rates
3. **Minimal footprint**: Only 2.4% of cells are affected across all year offsets
4. **Configurable**: Can be disabled or tuned via `projection_config.yaml` without code changes
5. **Improves reservation county trajectories**: Modest reduction in extreme 85+ exit rates (2-3 percentage points less decline)

### Negative
1. **McKenzie remains at +45%**: The rate cap clips only 1 cell from 8.45% to 8.0%, reducing 30-year growth by only 0.5 percentage points. Oil-county growth requires boom dampening, not rate capping.
2. **Reservation declines remain at -56% to -58%**: The fundamental driver is persistent negative migration across many working-age cells, which is a structural demographic pattern, not statistical noise.
3. **The 85+ cap may clip genuine elderly out-migration**: However, these rates are the most statistically noisy (smallest populations) and the most likely to reflect measurement error.

### Risks and Mitigations

**Risk**: College-age dynamics shift if university enrollment patterns change
- **Mitigation**: The 15% cap provides headroom above current peak rates (14.0%). Review if base data is updated with substantially different enrollment patterns.

**Risk**: The 8% general cap is slightly below McKenzie 25-29 Female (8.45%)
- **Mitigation**: This is intentional. The 8.45% rate is a boom residual that the cap correctly trims. McKenzie's growth is primarily addressed by boom dampening (ADR-040).

## Implementation Notes

### Key Functions
- `_apply_rate_cap()`: New helper function in `convergence_interpolation.py` that applies the two-tier cap
- `calculate_age_specific_convergence()`: Updated to accept `rate_cap_config` parameter and apply cap after interpolation
- `run_convergence_pipeline()`: Updated to read `rate_cap` from config and pass through

### Key Files
- `config/projection_config.yaml`: Rate cap configuration under `rates.migration.interpolation.rate_cap`
- `cohort_projections/data/process/convergence_interpolation.py`: Cap implementation
- `tests/test_data/test_convergence_interpolation.py`: Unit tests for cap logic

### Configuration Integration

The rate cap is configured under `rates.migration.interpolation.rate_cap` in `projection_config.yaml`. Setting `enabled: false` disables the cap without removing the configuration. All cap parameters (college_ages, college_cap, general_cap) have sensible defaults in the code.

### Testing Strategy
1. **Unit tests**: Verify cap clips extreme rates while leaving moderate rates unchanged
2. **Age-aware tests**: Verify college-age cells use the wider cap (15%) and non-college cells use the tighter cap (8%)
3. **Disabled-cap tests**: Verify cap has no effect when `enabled: false`
4. **Regression tests**: Existing convergence schedule tests pass unchanged when cap is not configured

## References

1. **Design Document**: [design-migration-rate-cap.md](../../reviews/2026-02-18-sanity-check-investigations/design-migration-rate-cap.md) -- Full analysis of rate distribution, outlier identification, and cap option evaluation
2. **Finding 2**: [Oil County Growth](../../reviews/2026-02-18-sanity-check-investigations/finding-2-oil-county-growth.md) -- Identifies extreme positive rates in oil-patch counties
3. **Finding 3**: [Reservation Declines](../../reviews/2026-02-18-sanity-check-investigations/finding-3-reservation-county-declines.md) -- Identifies extreme negative rates in reservation counties
4. **Sanity Check**: [Projection Output Sanity Check](../../reviews/2026-02-18-projection-output-sanity-check.md) -- Parent review

## Revision History

- **2026-02-18**: Initial version (ADR-043) -- Implement age-aware migration rate cap for convergence interpolation

## Related ADRs

- **ADR-040: Extend Bakken Boom Dampening to 2015-2020** -- Complementary mechanism for oil-county growth control; dampening reduces boom period rates before averaging, cap clips individual cell outliers after averaging
- **ADR-036: Migration Averaging Methodology** -- Defines the convergence interpolation pipeline where the cap is applied
- **ADR-042: Baseline Projection Presentation Requirements** -- Presentation requirements for projection outputs affected by the cap
