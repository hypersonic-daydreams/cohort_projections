---
title: "Population Projection Explosion Bug Investigation"
created: 2026-02-13T21:32:39Z
updated: 2026-02-13T21:42:38Z
status: completed
author: Codex (GPT-5)
type: investigation-and-remediation
commit_investigated: 62d99b0
scope:
  - scripts/pipeline/02_run_projections.py
  - cohort_projections/core/cohort_component.py
  - cohort_projections/core/migration.py
  - cohort_projections/core/mortality.py
  - cohort_projections/data/process/residual_migration.py
  - cohort_projections/data/process/convergence_interpolation.py
---

# Summary

The population explosion was caused primarily by a migration rate unit mismatch introduced in the new PEP residual + convergence workflow. Multi-year residual migration rates were being treated as annual rates in the projection engine.

## Key Findings

1. Residual migration rates were computed as period rates but consumed as annual rates.
2. Time-varying migration tables bypassed dedicated validation checks in engine initialization.
3. Mortality improvement was applied twice when time-varying survival tables were present:
   - once upstream in ND-adjusted survival projections,
   - again in `apply_survival_rates()` via config `improvement_factor`.

## Root Cause Detail

### 1) Migration Unit Mismatch (Primary)

Residual migration rates in Phase 1 were computed as:

- `period_rate = net_migration / expected_pop`

for full 5-year (or 4-year) periods. Those rates were then averaged/interpolated in Phase 2 and passed to the engine as `migration_rate`. The engine interprets `migration_rate` as annual and applies:

- `annual_migration = population * migration_rate`

each projection year.

This turned multi-year rates into annual multipliers, creating explosive compounding.

### 2) Missing Validation on Time-Varying Migration Tables

Engine input validation only checked the static `migration_rates` table. The year-varying migration tables (`migration_rates_by_year`) were not validated at initialization, allowing extreme values to propagate silently.

### 3) Double Mortality Improvement

When `survival_rates_by_year` is provided, those survival rates already include improvement from the mortality pipeline. The projection step still reapplied config-level `improvement_factor`, raising survival rates further than intended.

## Implemented Changes

1. Residual migration rates are now annualized in Phase 1 using geometric conversion:
   - `annual = (1 + period_rate) ** (1 / period_length) - 1`
2. Convergence metadata now declares `rate_unit: annual_rate`.
3. Pipeline loader now detects legacy convergence files without `rate_unit=annual_rate` and annualizes them defensively at load time.
4. Engine initialization now validates all time-varying migration and survival tables.
5. Projection step now disables config-level mortality improvement when using year-specific survival tables to avoid double application.

## Notes

- Legacy convergence outputs generated before this fix can still be used safely; the loader applies backward-compatible annualization when metadata indicates legacy units.
- For best reproducibility, regenerate Phase 1/2 outputs so stored convergence files are natively annual-rate with updated metadata.

## Verification Snapshot

A full 53-county baseline sanity run after implementation produced:

- 2025: 796,568
- 2045: 914,172
- Growth: +14.76%

This is within the expected non-explosive range and confirms the prior runaway growth behavior is resolved.
