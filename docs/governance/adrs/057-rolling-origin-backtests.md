# ADR-057: Rolling-Origin Backtests for Place Projections

## Status
Proposed

## Date
2026-03-01

## Context

The Phase 1 place projection system (PP-003) uses a static two-window backtest to validate variant selection: a primary window (train 2000-2014, test 2015-2024) and a secondary window (train 2000-2009, test 2010-2024). While this approach passed acceptance gates and selected the B-II winner variant, a rolling-origin cross-validation framework would provide more rigorous evidence of model stability across different training horizons.

### Requirements
- Validate that the B-II winner variant (WLS + cap-and-redistribute) is robust across multiple training horizons
- Provide per-window diagnostic metrics for transparency
- Maintain compatibility with existing backtest infrastructure (`place_backtest.py`)

### Challenges
- Computational cost scales linearly with the number of rolling windows
- Early windows have limited training data (e.g., train on 2000-2004 = 5 years)
- Need to define aggregation criteria across windows that balances rigor with practical interpretation

## Decision

*To be completed during implementation.*

## Consequences

### Positive
1. More rigorous validation evidence for variant selection
2. Identifies training-horizon sensitivity (if any)
3. Strengthens publication-quality methodology documentation

### Negative
1. Increased computation time for full backtest suite
2. Additional output artifacts to manage and interpret

## Implementation Notes

### Key Functions/Classes
- `run_rolling_origin_backtest()`: Orchestrate expanding-window iteration
- Reuses: `run_single_variant()`, `compute_per_place_metrics()`, `compute_tier_aggregates()`, `compute_variant_score()` from `place_backtest.py`

### Configuration Integration
New `rolling_origin_backtest` block in `projection_config.yaml`.

### Testing Strategy
Unit tests for window iteration, metric aggregation, and edge cases. Production validation run on ND data.

### Documentation
- [ ] Update `docs/methodology.md` with rolling-origin evidence

## References

1. ADR-033: City-Level Projection Methodology (parent methodology)
2. PP-003 IMP-08/IMP-09: Original backtest framework and results

## Revision History

- **2026-03-01**: Initial proposal (ADR-057) - Rolling-origin backtests for place variant validation

## Related ADRs

- ADR-033: City-Level Projection Methodology (extends backtest validation)
