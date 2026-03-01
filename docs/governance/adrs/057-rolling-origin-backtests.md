# ADR-057: Rolling-Origin Backtests for Place Variant Selection

## Status
Accepted

## Date
2026-03-01

## Context

The place share-trending backtest (IMP-08, PP-003) selects a winning variant from a 2x2 matrix (OLS/WLS x proportional/cap_and_redistribute) using a static two-window evaluation: a primary window (train 2000-2014, test 2015-2024) and a secondary window (train 2000-2019, test 2020-2024). The winner is chosen based on the primary window score alone.

### Requirements
- More rigorous evidence for variant selection that is less sensitive to the specific choice of evaluation period.
- Evaluate how variant performance varies across different historical regimes (pre-boom, boom, post-boom).
- Maintain backward compatibility with the existing backtest infrastructure.

### Challenges
- Only 25 years of history (2000-2024) limits the number of non-overlapping evaluation windows.
- Expanding windows mean early windows have very little training data, potentially penalizing data-hungry methods like WLS.
- Must reuse existing per-window computation (`run_single_variant`, `compute_per_place_metrics`, `compute_tier_aggregates`, `compute_variant_score`) without duplication.

## Decision

### Decision 1: Expanding-Window Rolling-Origin Cross-Validation

**Decision**: Implement rolling-origin backtesting with expanding (not sliding) training windows and non-overlapping test periods of fixed length.

**Rationale**:
- Expanding windows match production usage: the final model is always trained on all available history starting from 2000.
- Non-overlapping test windows prevent information leakage between evaluation folds.
- With min_train_years=5 and test_horizon=5, this produces 4 windows from 2000-2024 history, each exercising a different historical regime.

**Implementation**:
```python
generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
# => [(2000,2004,2005,2009),  # pre-boom
#     (2000,2009,2010,2014),  # boom period
#     (2000,2014,2015,2019),  # post-boom
#     (2000,2019,2020,2024)]  # recent/COVID
```

**Alternatives Considered**:
- **Sliding windows**: Rejected because the production model always uses the full history from 2000, so an expanding window better reflects real usage.
- **Leave-one-out by year**: Too many folds with minimal training variation; computationally expensive.

### Decision 2: Mean Score as Primary Aggregation

**Decision**: Use arithmetic mean of per-window variant scores as the primary aggregation criterion, with median available as an alternative.

**Rationale**:
- Mean gives equal weight to all windows, including those where a variant may perform poorly (important for risk assessment).
- Median is available for robustness checks against outlier windows (e.g., if one window covers an unusual regime).
- Existing tie-breaking rules (A > B, I > II) apply to the aggregated score.

**Alternatives Considered**:
- **Worst-case (max) score**: Too conservative; a single bad window could override strong performance elsewhere.
- **All-windows-must-pass threshold**: Overly restrictive given that early windows have minimal training data and may naturally perform worse.

### Decision 3: Reuse Existing Backtest Primitives

**Decision**: The rolling-origin module delegates all per-window computation to existing `place_backtest.py` functions. No metric computation is duplicated.

**Rationale**:
- Single source of truth for metric definitions (MAPE, MedAPE, tier aggregation, scoring).
- Changes to scoring methodology automatically propagate to both static and rolling backtests.
- Reduced testing surface: rolling-origin tests focus on orchestration and aggregation.

## Consequences

### Positive
1. Variant selection is evaluated across 4 distinct historical regimes rather than 1-2.
2. Score stability (standard deviation across windows) provides a confidence measure for the winning variant.
3. Backward compatible: existing static backtest continues to work unchanged.
4. Configuration-driven: `min_train_years` and `test_horizon` are adjustable without code changes.

### Negative
1. 4x more backtest executions (4 windows instead of 1 primary), increasing runtime.
2. Early windows (5-year training) may systematically disadvantage WLS, which benefits from more training data.

### Risks and Mitigations

**Risk**: Runtime increase from 4x window evaluations.
- **Mitigation**: Rolling-origin is a validation tool run offline, not in the production pipeline. Runtime is acceptable for periodic evaluation.

**Risk**: Early windows with limited training data produce unreliable scores.
- **Mitigation**: The `min_train_years` parameter can be increased (e.g., to 10) if early windows prove too noisy; median aggregation can reduce their influence.

## Implementation Notes

### Key Functions/Classes
- `generate_rolling_windows()`: Produces expanding window tuples from history bounds.
- `run_rolling_origin_backtest()`: Orchestrates variant execution across all windows.
- `aggregate_rolling_metrics()`: Computes mean/median/std/min/max across windows per variant.
- `select_rolling_winner()`: Picks winner using aggregated scores and existing tie-breaking rules.
- `build_per_window_summary()`: Flat summary table for reporting.

### Configuration Integration
New block in `config/projection_config.yaml` under `place_projections`:
```yaml
rolling_origin_backtest:
  enabled: true
  min_train_years: 5
  test_horizon: 5
  acceptance_criteria: "mean_score"
```

### Testing Strategy
- 15+ unit tests in `tests/test_data/test_rolling_origin_backtest.py`.
- Window generation: boundary conditions, expanding property, non-overlap.
- Aggregation: mean/median correctness, std NaN for single window.
- Winner selection: tie-breaking, invalid criteria, payload keys.
- Integration: mocked `run_single_variant` to verify orchestration.

### Documentation
- [ ] Update `docs/methodology.md` if this ADR changes formulas, rates, data sources, or projection logic

## References

1. **Tashman (2000)**: "Out-of-sample tests of forecasting accuracy: an analysis and review" -- foundational rolling-origin methodology.
2. **IMP-08 (PP-003)**: Original static backtest implementation.
3. **S04/S05**: Place projection specification documents defining metrics and thresholds.

## Revision History

- **2026-03-01**: Initial version (ADR-057) - Rolling-origin cross-validation for place variant selection.

## Related ADRs

- ADR-033: City-Level Projection Methodology (deferred)
- ADR-056: Testing Strategy Maturation (test patterns)
