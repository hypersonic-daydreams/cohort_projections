# ADR-036: Migration Averaging Methodology — Multi-Period and Interpolation Approaches

## Status
Proposed

## Date
2026-02-12

## Context

### Problem: Regime-Weighted Averaging Lacks Methodological Precedent

ADR-035 established Census PEP components of change as the primary migration data source, replacing IRS county-to-county flows. The initial implementation used **regime-weighted averaging** to compute baseline migration assumptions from the 2000–2024 PEP time series. This approach defined four economic regimes (Pre-Bakken, Boom, Bust+COVID, Recovery) and assigned weights emphasizing the most recent period:

| Regime | Period | Years | Weight | Per-Year Weight |
|--------|--------|-------|--------|-----------------|
| Recovery | 2022–2024 | 3 | 50% | 16.7% |
| Bust+COVID | 2016–2021 | 6 | 25% | 4.2% |
| Pre-Bakken | 2000–2010 | 11 | 15% | 1.4% |
| Boom | 2011–2015 | 5 | 10% | 2.0% |

A detailed review of published best practices and the methodologies used by major state, federal, and international demography offices (documented in [Migration Averaging Best Practices Review](../../reports/migration_averaging_best_practices.md)) revealed three concerns:

1. **No major demography office assigns 50% weight to a 3-year window** as their primary projection assumption. Even Statistics Canada's "most recent 3 years" scenario is one extreme of six, not the central case.

2. **"Regime-weighted averaging" is not a recognized standard methodology** in the demographic projections literature. While the concept has informal precedents, the specific formulation lacks published support.

3. **Recency bias is especially dangerous for volatile areas.** Smith and Tayman (2003) document that projection errors are positively related to growth rate volatility. North Dakota — with Bakken boom/bust swings exceeding 14,000 people/year — is precisely the kind of area where over-weighting a short recent window produces unreliable results.

### What Practitioners Actually Do

The best practices review surveyed 12+ state/federal offices and 2 international agencies. Two methodological families emerged as the most defensible for volatile subnational areas:

**Family 1: Multi-Period Averaging (BEBR/Indiana approach)**
- Compute equal-weighted averages from multiple overlapping base periods
- Generate separate projections from each period
- Average results, often with trimming (drop highest/lowest)
- Used by: Florida BEBR (6 base periods, 11 projections, trimmed average), Indiana 2025–2060 (13 models, trimmed average)

**Family 2: Interpolation/Convergence (Census Bureau/UK ONS approach)**
- Near-term: use recent migration rates (or interpolate from recent toward long-term)
- Medium-term: converge to a longer-period average
- Long-term: hold at long-period average or decay toward zero
- Used by: U.S. Census Bureau State Interim Projections (ARIMA → long-term mean), UK ONS (10-year average with interpolation), Indiana 2010–2040 (convergence toward zero), Missouri (root-function decay), Cornell PAD (trend toward constant)

### Requirements

- Migration assumptions must be defensible against published best practices
- Must handle North Dakota's extreme oil-driven volatility without over-fitting to any single period
- Must support scenario analysis (low, baseline, high)
- Must produce county-specific rates for all 53 ND counties
- Must remain compatible with existing pipeline architecture (per-county `dict[str, pd.DataFrame]`)

### Challenges

- North Dakota's migration history includes structural breaks (Bakken boom onset ~2011) that make any single averaging window potentially misleading
- The most recent 3 years (2022–2024) may reflect temporary post-COVID rebound rather than a new steady state
- County-level data is noisier than state-level, requiring smoothing or averaging
- The projection pipeline already supports regime-weighted averaging; changes require careful migration of config and code

## Decision

**Replace regime-weighted averaging with two defensible methods: BEBR-style multi-period averaging (primary) and Census Bureau-style interpolation (secondary). Both methods will be implemented, and their results will be compared and published as complementary projection sets.**

### Decision 1: BEBR-Style Multi-Period Averaging as Primary Method

**Decision**: Compute equal-weighted migration averages from four overlapping base periods, generate separate migration rate sets from each, and combine them via trimmed averaging.

**Base Periods**:

| Period | Window | Duration | Rationale |
|--------|--------|----------|-----------|
| Short | 2019–2024 | 6 years | Captures current conditions including COVID recovery |
| Medium | 2014–2024 | 11 years | Approximates one intercensal decade (Smith & Tayman optimal) |
| Long | 2005–2024 | 20 years | Captures full boom/bust cycle |
| Full | 2000–2024 | 25 years | Maximum available history |

Within each period, **all years receive equal weight** (consistent with Statistics Canada and UK ONS standard practice).

**Combination Method**:
1. For each county, compute the mean annual net migration from each of the 4 base periods
2. Generate a full age/sex/race-distributed rate set from each mean (using existing Rogers-Castro allocation)
3. Produce a **trimmed average**: drop the highest and lowest of the 4 period means, average the remaining 2
4. The trimmed average becomes the baseline projection assumption
5. The untrimmed 4-period results are retained for scenario analysis

**Scenario Variants**:
- **Low**: Use the lowest of the 4 period means (typically the period capturing bust+COVID)
- **Baseline**: Trimmed average of the 4 periods (drop high and low, average middle 2)
- **High**: Use the highest of the 4 period means (typically the period capturing boom recovery)
- **Full-range**: Report all 4 period-specific projections to show the full uncertainty envelope

**Rationale**:
- Follows the BEBR methodology, one of the most respected county-level projection approaches in the U.S.
- Equal weighting within periods is the dominant standard (Statistics Canada, UK ONS, Texas)
- Trimmed averaging eliminates the influence of outlier periods — critical for ND where the boom period would otherwise dominate
- Smith and Tayman (2003) finding that 10 years is optimal is captured in the "medium" period; other periods provide robustness
- No single period receives more than ~44% influence in the trimmed average (vs. 50% for 3 years under regime weighting)
- Indiana's updated 2025–2060 methodology uses a nearly identical multi-model trimmed-average approach

**Alternatives Considered**:
- *Weighted multi-period*: Assign higher weight to recent periods. Rejected because equal weighting is the published standard and weighted schemes introduce the same subjective judgment we're trying to avoid.
- *Single 10-year base*: Use only 2014–2024. Rejected because this misses the pre-boom baseline and the boom itself, both of which provide important context for volatile oil counties.

### Decision 2: Census Bureau-Style Interpolation as Secondary Method

**Decision**: Implement a convergence method that interpolates from recent migration rates toward a long-term average, producing a separate projection set for comparison.

**Method Specification**:

| Projection Year Range | Migration Rate Source |
|-----------------------|---------------------|
| Years 1–5 (2025–2029) | Linear interpolation from 3-year recent average (2022–2024) toward 10-year average (2014–2024) |
| Years 6–15 (2030–2039) | 10-year equal-weighted average (2014–2024), held constant |
| Years 16–20 (2040–2045) | Linear interpolation from 10-year average toward 25-year full-period average (2000–2024) |

**Rationale**:
- Follows the Census Bureau State Interim Projections approach (ARIMA → convergence to long-term mean) and the UK ONS approach (interpolation from recent to long-term)
- Acknowledges that recent conditions have informational value for the near term without projecting them indefinitely
- Natural dampening: extreme recent migration rates (whether high or low) are automatically moderated over the forecast horizon
- Produces results that will differ from the BEBR method, providing a useful methodological comparison

**Scenario Variants**:
- **Low**: Faster convergence (reach 10-year average by year 3 instead of year 5; converge to full-period average by year 10)
- **Baseline**: Standard interpolation schedule as specified above
- **High**: Slower convergence (hold recent 3-year average through year 5; reach 10-year average by year 10; hold through remainder)

**Alternatives Considered**:
- *ARIMA-based near-term*: Use time series modeling for the first 5 years. Rejected as over-engineered for 53 counties with 25 data points each; linear interpolation is more transparent and produces similar results for short horizons.
- *Convergence to zero*: Allow rates to decay toward zero over the projection horizon (Indiana, Missouri). Rejected because zero net migration is not a defensible long-run assumption for North Dakota; convergence to the full-period average is more appropriate.

### Decision 3: Retain Regime Classification for Analysis, Not for Rate Calculation

**Decision**: Keep the regime classification system (oil/metro/rural counties; Pre-Bakken/Boom/Bust+COVID/Recovery periods) for analytical reporting and documentation, but do not use regime weights to compute projection assumptions.

**Rationale**:
- The regime framework provides useful analytical insight into why migration patterns differ across counties and periods
- Regime analysis reports remain valuable for stakeholder communication
- Removing regime weights from rate calculation eliminates the methodological concerns identified in the best practices review
- The `pep_regime_analysis.py` module and its tests remain useful; only the weighting in `migration_rates.py` and `projection_config.yaml` changes

### Decision 4: Boom Dampening Replaced by Trimming

**Decision**: Remove the explicit boom dampening factor (0.60) from the rate calculation. The BEBR trimmed-average method naturally handles extreme periods by dropping the highest and lowest period means.

**Rationale**:
- The 0.60 dampening factor was calibrated to match SDC 2024 methodology. With BEBR multi-period averaging, the boom period's extreme values will naturally be trimmed if they produce the highest or lowest result.
- For the interpolation method, boom-era rates only influence the calculation through the long-period (2000–2024 or 2005–2024) averages, where they are diluted by 20+ years of other data.
- Removing a subjective dampening parameter simplifies the methodology and makes it more transparent.
- SDC 2024 comparison remains possible as a separate scenario using the SDC's published methodology.

## Consequences

### Positive

1. **Methodological defensibility**: Both methods have clear precedent in published literature and major demography office practice
2. **Reduced recency bias**: No single 3-year window can dominate the baseline projection
3. **Natural outlier handling**: Trimmed averaging eliminates boom/bust extremes without subjective dampening parameters
4. **Transparency**: Equal-weighted averages are easy to explain and reproduce
5. **Uncertainty communication**: Two methods plus scenario variants provide a rich picture of projection uncertainty
6. **Robustness**: If any single period proves atypical, it is diluted or trimmed rather than amplified

### Negative

1. **Two projection sets to maintain**: Running both BEBR and interpolation methods doubles the output
2. **Divergent results require explanation**: Stakeholders may be confused by two sets of numbers
3. **Loss of SDC 2024 alignment**: The new methods will not exactly replicate SDC 2024's specific dampening approach (though a separate SDC scenario can be maintained for comparison)
4. **Slightly more complex pipeline**: Two methods instead of one, though the BEBR method is structurally simpler than regime weighting

### Risks and Mitigations

**Risk**: BEBR trimmed average with only 4 periods may drop important signals (e.g., trimming both the boom period and the bust period loses information about volatility amplitude)
- **Mitigation**: Retain all 4 untrimmed period-specific projections for analysis. The full-range scenario set communicates the volatility. Consider adding a 5th period to improve trimming robustness if warranted.

**Risk**: Interpolation method's convergence schedule is somewhat arbitrary (why 5 years to 10-year average? why not 3 or 7?)
- **Mitigation**: Sensitivity analysis varying the convergence speed. The schedule is consistent with Census Bureau practice (ARIMA through year 5, convergence thereafter) and can be adjusted based on backtesting results.

**Risk**: Removing boom dampening may produce higher migration assumptions for oil counties in the BEBR "high" scenario
- **Mitigation**: The "high" scenario is explicitly optimistic. The trimmed-average baseline will not include undampened boom values. Document that the high scenario should be interpreted as an upper bound, not a central estimate.

## Implementation Notes

### Key Functions to Modify

- `process_pep_migration_rates()` in [migration_rates.py](../../../cohort_projections/data/process/migration_rates.py): Replace regime-weighted scenario definitions with BEBR multi-period and interpolation method logic
- `calculate_regime_weighted_average()` in [pep_regime_analysis.py](../../../cohort_projections/data/process/pep_regime_analysis.py): Retain for backward compatibility and analytical use; no longer called from the primary rate pipeline
- `load_demographic_rates()` in [02_run_projections.py](../../../scripts/pipeline/02_run_projections.py): Support loading either BEBR or interpolation method rates
- `run_geographic_projections()` in [02_run_projections.py](../../../scripts/pipeline/02_run_projections.py): For interpolation method, support time-varying migration rates across projection years

### New Functions

- `calculate_multiperiod_averages(pep_data, periods) -> dict[str, pd.DataFrame]`: Compute equal-weighted averages for each base period
- `calculate_trimmed_average(period_averages) -> pd.DataFrame`: Drop highest/lowest, average remaining
- `calculate_interpolated_rates(recent_avg, medium_avg, longterm_avg, projection_years) -> dict[int, pd.DataFrame]`: Generate year-specific rates following the convergence schedule
- `generate_methodology_comparison_report(bebr_results, interp_results, output_dir) -> Path`: Compare the two methods

### Configuration Changes

```yaml
rates:
  migration:
    domestic:
      method: "PEP_components"
      source: "Census_PEP"
      averaging_method: "BEBR_multiperiod"  # Changed from "regime_aware"
      base_periods:
        short: [2019, 2024]      # 6 years
        medium: [2014, 2024]     # 11 years (Smith & Tayman optimal)
        long: [2005, 2024]       # 20 years (full boom/bust cycle)
        full: [2000, 2024]       # 25 years (maximum history)
      combination: "trimmed_average"  # Drop high/low, average middle 2
      smooth_extreme_outliers: true
      # dampening_factor removed — handled by trimming
      # regime_weights removed — replaced by equal-weighted periods
    international:
      method: "PEP_included"
      allocation: "proportional"

    # Secondary method for comparison
    interpolation:
      method: "census_bureau_convergence"
      recent_period: [2022, 2024]    # 3-year recent average
      medium_period: [2014, 2024]    # 10-year convergence target
      longterm_period: [2000, 2024]  # 25-year long-run target
      convergence_schedule:
        recent_to_medium_years: 5    # Years 1-5: interpolate recent → medium
        medium_hold_years: 10        # Years 6-15: hold at medium average
        medium_to_longterm_years: 5  # Years 16-20: interpolate medium → long-term
```

### Testing Strategy

1. **Unit tests**: Verify equal-weighted averaging, trimmed-average calculation, interpolation schedule
2. **Integration tests**: Full pipeline with both BEBR and interpolation methods
3. **Backtesting**: Apply methods to 2000–2015 data, project 2015–2024, compare to actual PEP estimates
4. **Comparison tests**: BEBR baseline vs. interpolation baseline vs. regime-weighted baseline vs. SDC 2024
5. **Regression tests**: Ensure existing test suite (982 tests) continues to pass

### Output Structure

```
data/projections/
├── bebr_multiperiod/
│   ├── baseline/          # Trimmed average of 4 periods
│   ├── low/               # Lowest period mean
│   ├── high/              # Highest period mean
│   └── period_specific/   # All 4 individual period projections
├── census_interpolation/
│   ├── baseline/          # Standard convergence schedule
│   ├── low/               # Fast convergence
│   └── high/              # Slow convergence
└── comparison/
    └── methodology_comparison_report.md
```

## Alternatives Considered

### Alternative 1: Rebalance Regime Weights Only

**Description**: Keep the regime-weighted approach but reduce recovery weight from 50% to ~25% and increase pre-Bakken and bust+COVID proportionally.

**Pros**:
- Minimal code change (just update weight constants)
- Preserves existing regime framework

**Cons**:
- Still uses a non-standard methodology that lacks published precedent
- Any choice of regime weights is subjective
- Does not address the fundamental concern that regime-weighted averaging is not recognized in the literature

**Why Rejected**: Adjusting weights within a non-standard framework does not resolve the methodological defensibility concern. Better to adopt a method with clear published precedent.

### Alternative 2: Single 10-Year Base Period

**Description**: Use only the most recent 10 years (2014–2024) with equal weighting, following the Smith & Tayman finding.

**Pros**:
- Simplest possible approach
- Directly supported by Smith & Tayman empirical finding
- Used by Texas, Virginia, UK ONS

**Cons**:
- Misses the pre-boom baseline (2000–2010) entirely
- For oil counties, the 2014–2024 window captures the tail end of the boom, the full bust, and the recovery — potentially unbalanced
- Does not provide the robustness of multiple base periods

**Why Rejected**: For a state with North Dakota's volatility, the additional robustness from multiple base periods (BEBR approach) justifies the modest additional complexity. However, the 10-year period is included as one of the four BEBR base periods.

### Alternative 3: Economic-Demographic Linked Model

**Description**: Follow Colorado's approach — use economic forecasts to drive migration rather than extrapolating historical rates.

**Pros**:
- Theoretically superior: ties migration to causal drivers
- Naturally handles regime changes (economic conditions determine migration)

**Cons**:
- Requires economic forecast model (substantial additional infrastructure)
- Economic forecasts are themselves uncertain, especially for energy-dependent economies
- Far more complex to implement and maintain
- Not warranted for the current project scope

**Why Rejected**: Would require building or obtaining an economic forecasting model for North Dakota. Deferred to potential future enhancement. The multi-period approach provides a practical solution that handles volatility adequately.

## References

### Primary Research
1. **Smith & Tayman (2003)**: "The relationship between the length of the base period and population forecast errors." *International Journal of Forecasting*. Key finding: 10-year base period optimal.
2. **Smith, Tayman & Swanson (2013)**: *A Practitioner's Guide to State and Local Population Projections*. Springer. Standard reference for subnational demography.
3. Full literature review: [Migration Averaging Best Practices Review](../../reports/migration_averaging_best_practices.md)

### Practitioner Methodologies
4. **Florida BEBR**: Multi-period trimmed averaging. https://bebr.ufl.edu/wp-content/uploads/2024/01/projections_2024.pdf
5. **Indiana STATS 2025–2060**: 13-model trimmed averaging. https://www.stats.indiana.edu/about/pop-projections-2025-60.asp
6. **U.S. Census Bureau State Interim Projections**: ARIMA + convergence. https://wonder.cdc.gov/wonder/help/populations/population-projections/methodology.html
7. **UK ONS 2022-based**: 10-year average + interpolation. https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/methodologies/nationalpopulationprojectionsmigrationassumptions2022based
8. **Statistics Canada 2025**: 6-scenario equal-weighted approach. https://www150.statcan.gc.ca/n1/pub/17-20-0003/172000032026002-eng.htm

### Internal Documentation
9. **ADR-035**: Census PEP data source decision (migration data, not averaging method)
10. **ADR-034**: Census PEP data archive infrastructure
11. **Projection Divergence Analysis**: [docs/reports/PROJECTION_DIVERGENCE_ANALYSIS.md](../../reports/PROJECTION_DIVERGENCE_ANALYSIS.md)

## Revision History

- **2026-02-12**: Initial version (ADR-036) — Proposed replacement of regime-weighted averaging with BEBR multi-period and Census Bureau interpolation methods

## Related ADRs

- **ADR-035: Census PEP Components of Change for Migration Inputs** — Established PEP as data source; this ADR refines how the data is averaged
- **ADR-034: Census PEP Data Archive** — Infrastructure for PEP data access
- **ADR-010: Geographic Scope and Granularity** — County-level focus
