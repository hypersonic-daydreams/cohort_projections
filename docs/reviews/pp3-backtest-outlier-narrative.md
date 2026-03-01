# PP-003 Backtest Outlier Narrative and Structural-Break Review (IMP-10)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Scope** | IMP-10 review package for PP-003 backtest outliers and structural-break exclusion decisions |
| **Status** | Approved 2026-03-01 (human review complete) |
| **Related Artifacts** | `data/backtesting/place_backtest_results/*`, `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md` |

## 1. Context

`IMP-09` selected variant `B-II` (`wls` + `cap_and_redistribute`) from the primary backtest window.

- Primary window (2015-2024 test): all scored tiers (`HIGH`, `MODERATE`, `LOWER`) passed.
- Secondary window (2020-2024 test, diagnostic): `HIGH` tier failed the bias threshold (`|Mean ME| <= 5%`) with observed `Mean ME = +5.263532%`.
- Per-place threshold exceedance flags in `backtest_per_place_detail.csv`: 1 flagged place (secondary window only).

## 2. Flagged Place Narrative (S05 Section 5.3)

### 2.1 Flagged place

| Window | Place | Place FIPS | County FIPS | Tier | MAPE | Mean Error | MaxAPE | Flag |
|--------|-------|------------|-------------|------|------|------------|--------|------|
| secondary | Horace city | 3838900 | 38017 | MODERATE | 30.334938% | -30.334938% | 50.177173% | EXCEEDS_TIER_90TH_CEILING |

### 2.2 Evidence summary

Observed Horace population and share pattern indicates a post-2019 acceleration not visible in the secondary training window:

- 2010-2019 population: `2,437 -> 2,952` (gradual growth)
- 2020-2024 population: `3,249 -> 6,286` (rapid growth)
- Share of Cass County:
  - 2015: `0.014966`
  - 2019: `0.016227`
  - 2024: `0.031282`

Secondary-window `B-II` projections for Horace remain near pre-break share levels (~`0.0156`) and underproject 2020-2024:

| Year | Projected | Actual | Percent Error |
|------|-----------|--------|---------------|
| 2020 | 2873.14 | 3249 | -11.57% |
| 2021 | 2950.96 | 3465 | -14.84% |
| 2022 | 3024.69 | 4290 | -29.49% |
| 2023 | 3091.59 | 5683 | -45.60% |
| 2024 | 3131.86 | 6286 | -50.18% |

### 2.3 Likely cause (for reviewer confirmation)

Most likely cause is a structural growth acceleration beginning around `2020-2022` (likely development/annexation-driven expansion in the Fargo metro context). Because this break occurs inside the secondary test window, a model trained through 2019 does not capture it.

## 3. Secondary HIGH-Tier Bias Review

Secondary `HIGH` tier failed only on bias; central tendency and upper-tail MAPE metrics remain below threshold.

| Metric | Threshold | Observed |
|--------|-----------|----------|
| Tier MedAPE | <=10% | 5.766016% |
| Tier 90th-pctl MAPE | <=20% | 8.085342% |
| \|Tier Mean ME\| | <=5% | 5.263532% (FAIL) |

### 3.1 Decomposition by HIGH places (secondary)

Largest positive mean-error contributors are:

- Williston city: `+12.238631%`
- Mandan city: `+7.047020%`
- Grand Forks city: `+6.061484%`
- Dickinson city: `+5.827480%`
- West Fargo city: `+5.766016%`

Williston appears to be the dominant contributor to threshold crossing:

- Secondary HIGH mean ME with all 9 places: `+5.263532%`
- Secondary HIGH mean ME excluding Williston: `+4.391645%` (passes threshold)

### 3.2 Williston structural-break signal

Williston share of Williams County shows a step-down at 2020:

- 2019 share: `0.766554`
- 2020 share: `0.711792`
- 2024 share: `0.707038`

A model trained only through 2019 tends to extrapolate pre-2020 share levels, yielding systematic overprojection in 2020-2024.

## 4. Multi-County Place Check (S05 Section 6.4)

Backtest detail includes `multi_county_primary` flags. Current summary (scored tiers only):

- Primary: `multi_county_primary` mean MAPE `5.092` vs single-county `5.225`
- Secondary: `multi_county_primary` mean MAPE `6.056` vs single-county `4.415`

Interpretation: modest secondary-window degradation for multi-county places is present, but sample size is small (`n=2` scored places). This should be tracked for Phase 2 splitting refinement, not treated as an immediate blocker.

## 5. Structural-Break Exclusion Decision Record

No exclusions have been applied by the agent. Any exclusion requires explicit human approval and rationale (break year + cause), per S05.

Approved decision table for this cycle (recorded 2026-03-01):

| Tier/Window | Candidate Place | Exclude from tier aggregates? | Rationale (if yes) |
|-------------|-----------------|-------------------------------|--------------------|
| secondary / MODERATE | Horace city (3838900) | **No** | Structural break confirmed (annexation/growth acceleration), retained as diagnostic evidence only |
| secondary / HIGH | Williston city (3886220) | **No** | Structural break confirmed (oil economy share shift), retained as diagnostic evidence only |

## 6. Human Decisions Recorded (2026-03-01)

1. `B-II` winner adoption: **Approved** for production integration.
2. Structural-break interpretations: **Approved** (`Horace` annexation-driven acceleration; `Williston` oil-economy fluctuation/shift).
3. Structural-break exclusions: **None applied**; retain both as documented diagnostic context.
4. Pipeline progression to `IMP-11`: **Approved**.

## 7. Human Review Actions Needed

Resolved. No further approvals are required for `IMP-10`.
