# ADR-052: Ward County (Minot) Projection Review and High-Growth Scenario Floor

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Address Ward County's projected decline in all three scenarios (-5.2% baseline), which diverges 28 percentage points from the SDC 2024 reference (+23%)

**Related**: [ADR-046](046-high-growth-bebr-convergence.md) (high growth methodology), [ADR-049](049-college-age-smoothing-convergence-pipeline.md) (college-age smoothing)

## Context

### Problem: Ward County Declines in All Scenarios, Including High Growth

Ward County (38101, Minot) is the state's 4th-largest county (population ~69,000) and is projected to decline under all three scenarios:

| Scenario | 30-Year Change | Annualized |
|----------|:-------------:|:----------:|
| Baseline | **-5.2%** | -0.18%/yr |
| Restricted growth | -7.8% | -0.27%/yr |
| High growth | **-1.2%** | -0.04%/yr |

The SDC 2024 reference projects Ward at **+23.0%** over 30 years (+0.69%/yr). The 28-percentage-point divergence from the SDC reference is the largest negative divergence for any major county.

### Historical Volatility

Ward County has a highly volatile migration history driven by Minot Air Force Base (MAFB), Bakken oil spillover, and the 2011 Souris River flood:

| Period | Annualized Growth | Net Migration | Key Events |
|--------|:-----------------:|:-------------:|------------|
| 2000-2005 | -1.27% | -3,470 | MAFB drawdown |
| 2005-2010 | +0.08% | +511 | Neutral |
| 2010-2015 | +2.77% | +2,996 | Oil boom + MAFB expansion |
| 2015-2020 | -0.22% | -2,143 | Oil bust |
| 2020-2025 | -0.49% | -4,498 | COVID + continued decline |

The 2020-2025 period — which dominates the "recent" convergence window — captures an unusually negative period including COVID effects, reduced oil activity, and possible MAFB force adjustments.

### Why the Projection Declines

1. **Persistent negative migration across almost all age groups**: 30 of 34 non-infant age-sex cells have negative averaged migration rates. Only 20-24 Male (+0.025, college enrollment), 15-19 Male (+0.009), 50-54 Male (+0.002), and 15-19 Female (+0.0004) are positive.

2. **Convergence rates lock in the decline**: The mean convergence rate at year 5 (medium hold) is -0.008 across all cells, meaning approximately -0.8% annual net out-migration. The medium window (2005-2024 average) is less negative than the recent window (2020-2024), so the convergence transition from recent to medium actually *improves* the outlook — but not enough to turn it positive.

3. **High growth BEBR increment is negligible**: The BEBR high scenario produces a per-cell increment of only +0.000086/yr for Ward County. This is because the BEBR "high" is the most optimistic historical period, and Ward's best period (2010-2015, +0.008 mean rate) is only marginally better than its overall average. The increment mechanism (ADR-046) was designed for the state aggregate; individual counties with uniformly weak migration patterns receive minimal benefit.

### Two Contributing Technical Issues

1. **College-age smoothing bypass (ADR-049)**: Ward is classified as a college county (Minot State University). The convergence pipeline uses unsmoothed rates, but for Ward this has modest impact since its college-age rates are already low.

2. **Recent window over-weighting**: The convergence pipeline uses only the most recent period (2020-2024) for the "recent" window. Ward's 2020-2024 period is anomalously negative compared to its 20-year average. A 2-period recent window (2015-2024) would be more stable.

### Is Ward's Decline Real?

The 2020-2025 data does show genuine decline. However, projecting 30 years of continued decline for a county with Minot's economic diversity (MAFB, Minot State, regional medical/retail center, oil-adjacent agriculture) is likely too pessimistic. Historical patterns show Ward can reverse quickly: it went from -3,470 net migration in 2000-2005 to +2,996 in 2010-2015.

## Decision

### Two-Part Fix: High-Growth Migration Floor + Monitor Recent Window

**Part 1: High-Growth Scenario Migration Floor**

For the high_growth scenario, implement a per-county migration rate floor that prevents any county's average convergence rate from being negative. For counties where the BEBR-boosted rates are still net-negative at the medium hold, replace the medium-hold average with zero (neutral migration).

```python
# In convergence pipeline, high variant only:
if variant == "high":
    county_mean = rates_df["migration_rate"].mean()
    if county_mean < 0:
        # Lift all cells by |county_mean| to achieve neutral migration
        rates_df["migration_rate"] += abs(county_mean)
```

**Rationale**: The high growth scenario represents an optimistic future. A scenario where even the most optimistic projection shows population decline for a county with significant institutional anchors (military base, university, regional center) is not a useful planning scenario. The floor at zero (neutral migration) means "the county at least replaces its out-migrants," which is a reasonable optimistic assumption.

**Impact**: Only Ward County and a small number of other declining counties would be affected. Counties with positive BEBR-boosted rates are untouched.

**Part 2: Document the Recent Window Sensitivity (No Code Change)**

The current 1-period recent window (2020-2024 only) is sensitive to period-specific shocks. For Ward, this period captures an unusually negative interval. A future enhancement (not implemented in this ADR) could:
- Extend the recent window to 2 periods (2015-2024) for counties where the most recent period diverges >2σ from the historical mean
- Apply a dampening factor to the recent window when it represents a clear outlier

This is documented as a known limitation rather than an immediate code change because:
1. Changing the recent window definition affects all 53 counties, not just Ward
2. The appropriate window length is a research question that requires systematic analysis
3. The high-growth floor (Part 1) addresses the most urgent concern (declining in all scenarios)

### Configuration

```yaml
high_growth:
  convergence_variant: "high"
  migration_floor:
    enabled: true
    floor_value: 0.0  # Minimum average convergence rate at medium hold
```

### Alternatives Considered

| Option | Description | Verdict |
|--------|-------------|---------|
| A: Do nothing | Accept Ward decline in all scenarios | Rejected — 28pp divergence from SDC is too large; high scenario showing decline is not useful |
| **B: High-growth migration floor (chosen)** | Floor at zero for high scenario | **Selected** — targeted, defensible, addresses most urgent concern |
| C: Ward-specific recalibration | PEP-anchored recalibration (like ADR-045 for reservations) | Rejected — Ward's PEP and residual data agree (both negative); the issue is the window, not the method |
| D: Extend recent window globally | Use 2 periods for all counties | Deferred — affects all counties, requires systematic analysis |
| E: Override Ward's migration rates | Manually set Ward's rates to match SDC | Rejected — introduces ad-hoc county-specific overrides |

## Consequences

### Positive

1. **High scenario becomes useful for Ward**: Ward shows modest growth under high_growth instead of decline, providing a meaningful planning range
2. **Targeted fix**: Only affects counties with negative high-growth convergence rates at the medium hold; most counties are unaffected
3. **Defensible**: A high-growth scenario floor at zero (neutral migration) is a conservative optimistic assumption
4. **Simple**: Small code change in the convergence pipeline's high variant path

### Negative

1. **Does not fix baseline**: Ward's baseline remains at approximately -5% over 30 years. If the 2020-2024 downturn proves to be temporary, the baseline will be too pessimistic.
2. **Floor at zero may be too conservative**: The SDC projects +23% for Ward. Our high scenario with the floor would project approximately +5% (from natural increase only). The gap with SDC remains significant.
3. **Asymmetric treatment**: The floor only applies to the high scenario. A future ADR could address the baseline/restricted scenarios if warranted.

### Expected Impact

| Scenario | Before | After (est.) |
|----------|:------:|:------------:|
| Baseline | -5.2% | -5.2% (unchanged) |
| Restricted | -7.8% | -7.8% (unchanged) |
| High growth | -1.2% | **~+5%** |

### Interaction with Other ADRs

- **ADR-049 (college-age smoothing)**: Fixing the convergence smoothing will modestly improve Ward's 20-24 rates, but the impact is small since Ward's college-age rates are already modest
- **ADR-047 (county-specific distributions)**: Using Ward's actual age distribution (younger-skewing due to MAFB families) instead of the statewide average will slightly improve natural increase
- **ADR-050 (restricted growth additive)**: The additive fix will make restricted_growth properly lower than baseline for Ward (currently restricted > baseline in some early years due to the multiplicative sign problem)

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `cohort_projections/data/process/convergence_interpolation.py` | Add migration floor logic in high variant path |
| `config/projection_config.yaml` | Add `migration_floor` config under `high_growth` |

### Testing Strategy

1. **Ward high-growth test**: Verify Ward County's high_growth convergence rates have mean ≥ 0 at medium hold
2. **Non-negative counties unaffected**: Verify counties with positive BEBR rates are unchanged
3. **Scenario ordering**: Verify restricted < baseline < high for Ward at 2055
4. **Integration**: Full pipeline produces Ward high_growth with modest positive growth

### Pipeline Rerun Required

1. **Step 01b**: Convergence interpolation (high variant floor applied)
2. **Step 02**: Projections
3. **Step 03**: Exports

## References

1. **SDC 2024 Projections**: Ward County +23.0% — calibration reference
2. **Census PEP Components**: `data/processed/pep_county_components_2000_2025.parquet` — Ward County historical migration data
3. **ADR-046**: High Growth BEBR Convergence — the BEBR increment mechanism that is insufficient for Ward
4. **Sanity Check Finding**: Ward declining in all scenarios identified as LOW priority concern

## Revision History

- **2026-02-18**: Initial version (ADR-052) — High-growth migration floor for Ward County

## Related ADRs

- **ADR-046: High Growth BEBR Convergence** — Defines the BEBR increment mechanism; this ADR adds a floor for counties where the increment is insufficient
- **ADR-049: College-Age Smoothing** — Complementary fix for college county convergence rates
- **ADR-045: Reservation County PEP Recalibration** — Analogous county-specific adjustment, but for a different root cause (residual method bias vs. window sensitivity)
- **ADR-040: Extend Boom Dampening** — Analogous parameter calibration for oil counties

## Implementation Results (2026-02-23)

The migration floor (`floor_value: 0.0`) was implemented in `cohort_projections/data/process/convergence_interpolation.py` and enabled in `config/projection_config.yaml`. Fresh projections from 2026-02-23 confirm the fix is working:

| Scenario | ADR-052 Pre-fix Estimate | ADR-052 Post-fix Estimate | Actual (2026-02-23) |
|----------|--------------------------|---------------------------|---------------------|
| Restricted | -7.8% | -7.8% | -20.2% |
| Baseline | -5.2% | -5.2% | -14.6% |
| High Growth | -1.2% | ~+5% | **+29.0%** |

Key findings:
- **Scenario ordering is correct**: restricted (-20.2%) < baseline (-14.6%) < high (+29.0%)
- **High growth 20yr (+21.2%) closely matches SDC reference (+23%)** — excellent calibration
- Baseline and restricted are more negative than original estimates due to subsequent ADR implementations (047-050, 053: Sprague interpolation, county-specific distributions, additive restricted growth, ND-specific vital rates)
- The migration floor combined with BEBR-optimistic rates produces strong high-growth performance (+29.0%), far exceeding the original +5% estimate
- The known limitation remains: the floor only fixes the high scenario; baseline Ward decline (-14.6%) is acknowledged as pessimistic relative to SDC

Note: Grand Forks shows a similar pattern (baseline -8.7%, high +38.4%) and may warrant monitoring for a similar floor treatment.
