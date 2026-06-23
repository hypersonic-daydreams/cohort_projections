# 2026-02-28 PP-003 IMP-09 Backtest Execution and Variant Selection Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Scope** | PP-003 IMP-09 execution of the place backtest matrix on ND production artifacts |
| **Status** | Completed 2026-02-28; approved for winner adoption 2026-03-01 |
| **Related ADR** | ADR-033 |
| **Runner** | `scripts/backtesting/run_place_backtest.py` |

## 1. Execution Summary

- Command executed:
  - `source .venv/bin/activate && python scripts/backtesting/run_place_backtest.py`
- Output directory:
  - `data/backtesting/place_backtest_results/`
- Variant matrix executed:
  - `A-I`, `A-II`, `B-I`, `B-II`
- Windows executed:
  - `primary` (train 2000-2014, test 2015-2024)
  - `secondary` (train 2000-2019, test 2020-2024)

## 2. Winner Selection (Primary Window)

Winner selected by population-weighted MedAPE score (S04 Section 5.3):

- **Winner variant**: `B-II`
- **Fitting method**: `wls`
- **Constraint method**: `cap_and_redistribute`
- **Primary score**: `3.0757840903894844`

Primary-window variant scores:

| Variant | Score |
|---------|-------|
| A-I | 3.309167 |
| A-II | 3.292266 |
| B-I | 3.099250 |
| B-II | 3.075784 |

## 3. S05 Threshold Evaluation (Winner Variant)

### 3.1 Primary Window (Binding Acceptance)

All scored tiers (`HIGH`, `MODERATE`, `LOWER`) passed S05 thresholds.

| Tier | Places | Tier MedAPE | Tier P90 MAPE | Tier Mean ME | Evaluation |
|------|--------|-------------|---------------|--------------|------------|
| HIGH | 9 | 3.002588 | 4.314164 | 1.106081 | PASS |
| MODERATE | 9 | 1.835121 | 17.781376 | -4.280532 | PASS |
| LOWER | 72 | 4.248954 | 11.049137 | 0.775915 | PASS |
| EXCLUDED | 267 | 9.308560 | 36.143633 | -1.619601 | INFORMATIONAL |

### 3.2 Secondary Window (Diagnostic)

Secondary window is diagnostic (non-binding). `HIGH` failed due bias threshold.

| Tier | Places | Tier MedAPE | Tier P90 MAPE | Tier Mean ME | Evaluation |
|------|--------|-------------|---------------|--------------|------------|
| HIGH | 9 | 5.766016 | 8.085342 | 5.263532 | FAIL |
| MODERATE | 9 | 3.722808 | 14.646048 | -1.526485 | PASS |
| LOWER | 72 | 2.843838 | 9.979268 | 0.174874 | PASS |
| EXCLUDED | 265 | 6.317718 | 21.515207 | -7.870851 | INFORMATIONAL |

## 4. Prediction Interval Calibration (Winner Variant)

Empirical PI half-widths from place-year APE distributions:

| Window | Tier | 80% PI Half-Width | 90% PI Half-Width | N Place-Years |
|--------|------|-------------------|-------------------|---------------|
| primary | HIGH | 4.377292 | 5.597436 | 90 |
| primary | MODERATE | 11.321926 | 23.437520 | 90 |
| primary | LOWER | 8.337710 | 13.039024 | 720 |
| primary | EXCLUDED | 23.634509 | 40.814307 | 2660 |
| secondary | HIGH | 7.146683 | 10.442887 | 45 |
| secondary | MODERATE | 11.119722 | 13.966589 | 45 |
| secondary | LOWER | 6.661963 | 10.088201 | 360 |
| secondary | EXCLUDED | 13.971328 | 22.453300 | 1325 |

## 5. Per-Place Flag Summary

- Output file: `data/backtesting/place_backtest_results/backtest_per_place_detail.csv`
- Threshold-exceedance flags:
  - `primary`: 0 places
  - `secondary`: 1 place (`Horace city`, `3838900`, tier `MODERATE`)

## 6. Artifact Manifest

- `data/backtesting/place_backtest_results/backtest_summary_primary.csv`
- `data/backtesting/place_backtest_results/backtest_summary_secondary.csv`
- `data/backtesting/place_backtest_results/backtest_per_place_detail.csv`
- `data/backtesting/place_backtest_results/backtest_variant_scores.csv`
- `data/backtesting/place_backtest_results/backtest_prediction_intervals.csv`
- `data/backtesting/place_backtest_results/backtest_winner.json`

## 7. Human Review Checklist

1. Confirm acceptance of `B-II` as production variant for PP-003 Phase 2+. **Completed 2026-03-01.**
2. Review secondary-window HIGH-tier bias signal (`ME=+5.263532%`) and determine whether any follow-up is required. **Completed 2026-03-01 (documented in IMP-10 narrative).**
3. Proceed to IMP-10 outlier narrative and structural-break documentation. **Completed 2026-03-01.**
