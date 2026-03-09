# ADR-062: Widen Aggregation Violation Tolerance from 1.0 to 2.0

## Status
Accepted

## Date
2026-03-09

## Context

The benchmark evaluation pipeline includes a hard gate that checks whether the sum of county-level projections matches the independently computed state-level projection. This gate exists to catch real aggregation bugs (e.g., missing counties, double-counting).

The tolerance was originally set at 1.0 person: any origin/validation year where `|sum(county_projected) - state_projected| > 1.0` counts as a violation.

### The Problem

During the 2026-03-09 experiment sweep, EXP-B (`college_blend_factor: 0.7`) was classified as `failed_hard_gate` due to a single aggregation violation: origin 2010, validation 2021, where the county sum was 662,399.1 vs the state projected 662,398.0 — a discrepancy of **1.1 persons** on a 662K population.

### Root Cause

Two independent rounding sources create unavoidable drift:

1. **County values** are stored at 0.1 precision (one decimal place). Each county has up to +/-0.05 rounding error. With 53 counties, the sum's standard deviation is ~0.21 (3-sigma ~0.63).
2. **State values** are stored as whole numbers, contributing up to +/-0.5 rounding error (std ~0.29).

The combined theoretical 3-sigma is **~1.07**, meaning a tolerance of 1.0 will occasionally produce false positives under normal rounding drift.

### Empirical Evidence

Analysis of 460 (origin_year, method, validation_year) observations across 5 benchmark runs:

| Statistic | Value |
|-----------|-------|
| Mean | 0.30 |
| Median | 0.30 |
| P95 | 0.70 |
| P99 | 0.80 |
| P99.9 | 0.96 |
| Max | 1.1 |
| Pairs exceeding 1.0 | 1 / 460 (0.22%) |
| Pairs exceeding 1.5 | 0 / 460 |
| Pairs exceeding 2.0 | 0 / 460 |

## Decision

**Decision**: Widen the aggregation tolerance from 1.0 to 2.0 persons.

**Rationale**:
- 2.0 is nearly 2x the theoretical 3-sigma (1.07), providing comfortable headroom for rounding drift as more experiment runs accumulate
- A genuine aggregation bug (missing county, double-counting) would produce errors of hundreds or thousands of persons — 2.0 still catches all real issues
- Zero false positives observed at 1.5 or 2.0 across all historical runs
- The alternative of 1.5 works empirically today but leaves less margin for future runs with different config variants

**Alternatives Considered**:
- **1.5**: Zero false positives in current data, but only ~40% above theoretical 3-sigma. Marginal headroom.
- **Keep 1.0, annotate known exceptions**: Adds operational complexity; each false positive would require manual review and exception tracking.
- **Dynamic tolerance based on county count**: Over-engineered for a fixed 53-county system.

## Consequences

### Positive
1. Eliminates false positive aggregation gate failures from rounding drift
2. EXP-B correctly reclassified from `failed_hard_gate` to `passed_all_gates`
3. No loss of sensitivity to real aggregation bugs (error scale is 100x+)

### Negative
1. Slightly less sensitive to hypothetical sub-2-person aggregation drift — but no legitimate scenario produces such drift outside of rounding

## Implementation Notes

### Key Functions/Classes
- `_aggregation_violations()` in `cohort_projections/analysis/benchmarking.py`: tolerance constant changed from 1.0 to 2.0

### Configuration Integration
The `benchmark_evaluation_policy.yaml` description for `aggregation_violations` updated to reference the 2.0 tolerance.

### Testing Strategy
Existing benchmark tests exercise the aggregation check. No new tests needed — the change is a constant.

### Documentation
- [ ] ~~Update `docs/methodology.md`~~ — not applicable (benchmark infrastructure, not projection logic)

## References

1. EXP-B benchmark run: `br-20260309-182528-m2026r1-201f5eb`
2. Empirical analysis: 460 observations across 5 benchmark runs (2 pre-sweep + 3 sweep)

## Revision History

- **2026-03-09**: Initial version (ADR-062) — widen tolerance based on empirical analysis

## Related ADRs

- ADR-061: College Fix Model Revision (EXP-B tests a parameter from this ADR)
