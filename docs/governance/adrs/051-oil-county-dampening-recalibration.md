# ADR-051: Oil County Dampening Recalibration

## Status
Proposed

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Increase boom dampening for McKenzie and Williams counties and address the medium-window averaging issue that perpetuates boom-era migration rates

**Refines**: [ADR-040](040-extend-boom-dampening-2015-2020.md) (adjusts dampening factors established in ADR-040)

## Context

### Problem: McKenzie County Projected at +84%, SDC Reference Is +47%

McKenzie County (38053) is projected to grow +83.7% over 30 years under the baseline scenario, while the SDC 2024 reference projection shows +47.1%. This 37-percentage-point divergence is the largest for any major county.

### Historical Growth Pattern

McKenzie County's growth has decelerated dramatically from its Bakken oil boom peak:

| Period | Annualized Growth | Net Migration (annual avg) | Dampening Applied |
|--------|:-----------------:|:--------------------------:|:-----------------:|
| 2005-2010 | +1.99% | +107/yr | 0.60 × 0.80 (male) = 0.48 |
| 2010-2015 | +15.00% | +1,140/yr | 0.60 × 0.80 (male) = 0.48 |
| 2015-2020 | +2.83% | +549/yr | 0.60 |
| 2020-2025 | +0.65% | +84/yr | None |

The most recent 5-year period (2020-2025) shows near-zero growth. PEP components for 2020-2024 show net migration of -893 people (net negative). Yet the projection converges to a medium-term rate that reflects the boom era.

### Root Cause: Medium Window Dominated by Boom-Era Rates

The convergence pipeline computes a "medium" window average across all periods (2005-2024 or similar). Even after 0.60 dampening, the boom-era rates dominate:

| Period | Mean Migration Rate (dampened) | Weight in Medium |
|--------|:-----------------------------:|:----------------:|
| 2005-2010 | +0.008 | Equal |
| 2010-2015 | **+0.042** | Equal |
| 2015-2020 | **+0.017** | Equal |
| 2020-2024 | **-0.019** | Equal |

The medium average comes out strongly positive (~+0.013) despite the most recent period being negative. Working-age cells (25-29, 30-34) have convergence rates of 0.06-0.08 at year 5+, hitting the rate cap.

### Williams County Is Similar but Less Extreme

Williams County (38105, Williston) is projected at +51.8% vs SDC's +33.4%, an 18-percentage-point divergence. The same boom-era medium-window inflation applies.

### Current Dampening Factors (ADR-040)

```yaml
dampening:
  counties: ["38053", "38105", "38061", "38025", "38007", ...]
  periods:
    2005-2010: 0.60
    2010-2015: 0.60
    2015-2020: 0.60
  male_dampening:
    periods: ["2005-2010", "2010-2015"]
    factor: 0.80
```

The 0.60 dampening was chosen in ADR-040 to balance between "boom-era rates are transient" and "the Bakken has permanently changed these counties." However, with 2020-2025 data now showing near-zero growth, the evidence supports more aggressive dampening of the boom periods.

## Decision

### Reduce Boom-Period Dampening Factor from 0.60 to 0.40

**Decision**: Lower the dampening factor for the 2010-2015 period from 0.60 to 0.40, and optionally reduce 2005-2010 and 2015-2020 as well. This is specifically for the core Bakken counties (McKenzie, Williams) where the post-boom deceleration is most pronounced.

**Revised configuration**:

```yaml
dampening:
  counties: ["38053", "38105", "38061", "38025", "38007", ...]
  periods:
    2005-2010: 0.50  # was 0.60
    2010-2015: 0.40  # was 0.60
    2015-2020: 0.50  # was 0.60
  male_dampening:
    periods: ["2005-2010", "2010-2015"]
    factor: 0.80  # unchanged
```

### Rationale for 0.40

The dampening factor represents "what fraction of the observed boom migration should be treated as permanent." With 2020-2025 data showing near-zero or negative migration for McKenzie, the boom-era migration was largely transient:

| Factor | Effective 2010-2015 male rate | McKenzie 30-yr projection (est.) | Alignment with SDC |
|:------:|:----------------------------:|:-------------------------------:|:------------------:|
| 0.60 | 0.028 | +84% | -37pp divergence |
| 0.50 | 0.024 | ~+68% | -21pp divergence |
| **0.40** | **0.019** | **~+55%** | **-8pp divergence** |
| 0.30 | 0.014 | ~+42% | +5pp divergence |

A factor of 0.40 brings McKenzie within ~8 percentage points of the SDC reference while still acknowledging that the Bakken infrastructure and economy have produced some permanent population growth.

### Alternative: Exclude 2010-2015 from Medium Window

An alternative approach would be to exclude the extreme 2010-2015 boom period from the medium window entirely for oil counties. This would use only 2005-2010, 2015-2020, and 2020-2024 for the medium average. However:
1. This introduces county-specific window definitions, adding complexity
2. The dampening approach is already established (ADR-040) and simply needs parameter tuning
3. Excluding data entirely is more aggressive than dampening and harder to justify theoretically

### Alternatives Considered

| Option | Description | Verdict |
|--------|-------------|---------|
| A: Keep 0.60 | Current factors | Rejected — 37pp divergence from SDC is too large |
| **B: Reduce to 0.40 (chosen)** | Lower dampening factor | **Selected** — brings within ~8pp of SDC |
| C: Exclude boom period from medium | Drop 2010-2015 from medium window | Rejected — complex, aggressive |
| D: County-specific dampening | Different factors per county | Rejected — over-engineering for the current issue |
| E: Reduce to 0.30 | More aggressive dampening | Rejected — risks underprojecting if Bakken has a future resurgence |

## Consequences

### Positive

1. **Closer to SDC reference**: McKenzie projected growth drops from ~+84% to ~+55%, within 8pp of SDC's +47%
2. **Williams alignment**: Similar improvement for Williams County
3. **Evidence-based**: The 2020-2025 data confirms the boom was largely transient
4. **Simple change**: Configuration-only; no code modifications required
5. **Reversible**: If Bakken activity resumes, the factor can be increased

### Negative

1. **Subjectivity**: The choice of 0.40 over 0.35 or 0.45 is somewhat arbitrary, calibrated against the SDC reference. This is inherent to dampening approaches.
2. **Retroactive effect**: Changes the meaning of historical migration rates in the processed data. The processed rates file should be regenerated.
3. **Peripheral counties less affected**: Mountrail (+3.7%) and Dunn (-1.2%) have modest projections that don't need further dampening. The reduced factor applies to all dampened counties equally, potentially over-dampening peripheral oil counties.

### Expected Impact

| County | Current Projection | After (est.) | SDC Reference |
|--------|:-----------------:|:------------:|:-------------:|
| McKenzie (38053) | +84% | ~+55% | +47% |
| Williams (38105) | +52% | ~+38% | +33% |
| Billings (38007) | +54% | ~+40% | — |
| Mountrail (38061) | +4% | ~+1% | +5% |
| Dunn (38025) | -1% | ~-4% | — |

Note: Mountrail and Dunn may require monitoring to ensure the reduced factor doesn't push them too negative. If so, tiered dampening by county could be revisited.

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `config/projection_config.yaml` | Update dampening factors under `rates.migration.domestic.residual.dampening.periods` |

### No Code Changes Required

The dampening is applied in `residual_migration.py` using the factors from config. Changing the config values is the only modification needed.

### Testing Strategy

1. **Residual pipeline validation**: Re-run `01_compute_residual_migration.py` and verify McKenzie 2010-2015 rates are reduced relative to current
2. **Convergence rates**: Re-run `01b_compute_convergence.py` and verify McKenzie medium-term rates are lower
3. **Projection comparison**: Run baseline for McKenzie and Williams; verify 30-year growth is in the +50-60% range
4. **Non-oil county regression**: Verify non-dampened counties are completely unaffected

### Pipeline Rerun Required

1. **Step 01**: Residual migration (dampening factors change processed rates)
2. **Step 01b**: Convergence interpolation (reads new processed rates)
3. **Step 02**: Projections
4. **Step 03**: Exports

## References

1. **ADR-040: Extend Bakken Boom Dampening to 2015-2020** — Established the dampening framework; this ADR tunes the parameters
2. **SDC 2024 Projections**: McKenzie +47.1%, Williams +33.4% — used as calibration reference
3. **Census PEP Components**: `data/processed/pep_county_components_2000_2025.parquet` — historical growth data confirming post-boom deceleration
4. **Sanity Check Finding**: McKenzie +84% identified as potentially overestimated

## Revision History

- **2026-02-18**: Initial version (ADR-051) — Reduce oil-county boom dampening factors

## Related ADRs

- **ADR-040: Extend Bakken Boom Dampening to 2015-2020** — Parent ADR establishing the dampening framework
- **ADR-043: Age-Aware Migration Rate Cap** — Complementary mechanism; clips individual cell outliers after dampening
- **ADR-036: Migration Averaging Methodology** — Defines the window averaging where dampened rates are consumed
