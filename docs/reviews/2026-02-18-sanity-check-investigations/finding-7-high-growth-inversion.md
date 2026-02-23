# Finding 7: High Growth Below Baseline (Scenario Inversion)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Related** | [ADR-037](../../governance/adrs/037-cbo-grounded-scenario-methodology.md) |
| **Status** | CONFIRMED BUG -- Design Error |

## Problem Statement

The `high_growth` scenario produces LOWER populations than `baseline` at every projected time point:

| Year | Baseline | High Growth | Difference |
|------|----------|-------------|-----------|
| 2025 | 799,358 | 799,358 | 0 |
| 2030 | 823,967 | 815,534 | -8,433 |
| 2035 | 867,915 | 856,864 | -11,051 |
| 2040 | 914,480 | 904,698 | -9,782 |
| 2045 | 955,679 | 952,386 | -3,293 |

This is a direct violation of ADR-037 Phase 4 validation requirement #4: "Verify that the high scenario produces higher population than the baseline throughout the projection horizon."

## Root Cause

**The `+15_percent` migration multiplier is applied symmetrically to net migration rates that are overwhelmingly negative (net out-migration). Multiplying a negative rate by 1.15 makes it more negative, amplifying population loss instead of increasing growth.**

### Detailed Mechanism

The convergence interpolation rates (used for all time-varying projections) are net-negative across the state:

| Year Offset | Sum of Migration Rates | Mean Rate | Counties with Net Out-Migration |
|-------------|------------------------|-----------|-------------------------------|
| 1 | -17.10 | -0.00896 | 45 of 53 |
| 5 | -10.97 | -0.00575 | 43 of 53 |
| 10 | -10.97 | -0.00575 | 43 of 53 |
| 20 | -14.11 | -0.00740 | 47 of 53 |
| 30 | -14.11 | -0.00740 | 47 of 53 |

When `+15_percent` is applied:

- **Positive rates** (8 growing counties) are scaled up: `+0.01 * 1.15 = +0.0115` (good -- more in-migration)
- **Negative rates** (45 declining counties) are also scaled up in magnitude: `-0.01 * 1.15 = -0.0115` (bad -- more out-migration)

Since the negative rates dominate, the net effect is **more population loss**:

| Component | Baseline Sum | High Growth Sum (+15%) | Difference |
|-----------|-------------|------------------------|-----------|
| Positive rates (year 1) | +22.91 | +26.35 | +3.44 |
| Negative rates (year 1) | -40.01 | -46.02 | -6.00 |
| **Net** | **-17.10** | **-19.67** | **-2.57** |

Cumulative over 30 projection years: high_growth has 57.8 units of additional net out-migration versus baseline.

## Complete Execution Trace

### Configuration (projection_config.yaml, lines 222-228)

```yaml
high_growth:
    name: "High Growth (Pre-Policy Elevated Immigration)"
    description: "Counterfactual continuation of elevated post-2020 immigration trends (ADR-037)"
    fertility: "+5_percent"
    mortality: "improving"
    migration: "+15_percent"
    active: true
```

### Step 1: Rate Loading

`02_run_projections.py` `load_demographic_rates()` (line 384-527) loads:

1. **Constant PEP migration rates** from `data/processed/migration/migration_rates_pep_baseline.parquet` (always baseline, per config at line 448-453)
2. **Convergence rates** from `data/processed/migration/convergence_rates_by_year.parquet` (line 493-507) -- 30 years x 53 counties x 36 cells = 57,240 rows

Note: `run_pep_projections.py` defines a `PEP_SCENARIO_FILE_MAP` (line 47-51) mapping `high_growth` to `migration_rates_pep_high.parquet`, but this mapping is **never used** in the actual rate loading path. The `verify_pep_rate_files()` function only checks file existence; it does not pass file paths to the pipeline. The `migration_rates_pep_high.parquet` file exists and contains higher rates (sum=2,786 vs baseline sum=1,484), but it is dead code.

### Step 2: Scenario Adjustments (Pre-Engine)

`02_run_projections.py` `apply_scenario_rate_adjustments()` (line 680-824):

- Applies `+5%` to fertility rates (line 752-753) -- **correctly applied**
- Applies `+15%` to the constant PEP migration rates (line 798-800) -- **applied but wasted**
- Passes convergence rates through **unchanged** (line 700-706: "Time-varying rate dicts are passed through unchanged")

### Step 3: Engine Initialization

`multi_geography.py` `run_single_geography_projection()` (line 146-154):

```python
projection_engine = CohortComponentProjection(
    base_population=base_pop,
    ...
    migration_rates=migration_rates,            # PEP baseline * 1.15 (NEVER USED)
    migration_rates_by_year=migration_rates_by_year,  # convergence rates (UNCHANGED)
    ...
)
```

### Step 4: Year-by-Year Projection (The Bug)

`cohort_component.py` `project_single_year()` (line 218-325):

```python
# Line 244: Gets convergence rates (NOT the +15% PEP rates)
migration_rates = self._get_migration_rates(year)  # returns convergence rate for this year

# Line 248-261: Applies scenario config
scenario_config = self.config.get("scenarios", {}).get(scenario, {})
migration_scenario = scenario_config.get("migration", "recent_average")
# For high_growth: migration_scenario = "+15_percent"
migration_rates = apply_migration_scenario(
    migration_rates, "+15_percent", year, self.base_year
)
```

`migration.py` `apply_migration_scenario()` (line 200-201):

```python
elif scenario == "+15_percent":
    adjusted_rates[migration_col] = adjusted_rates[migration_col] * 1.15
```

**This multiplies the convergence rates by 1.15. Since they are net-negative, the population decreases MORE than baseline.**

### Step 5: Baseline Comparison

For baseline, the same convergence rates are used but `migration_scenario = "recent_average"`, which is a no-op (line 190-191):

```python
if scenario == "recent_average" or scenario == "constant":
    pass  # No change
```

So baseline uses raw convergence rates, and high_growth uses convergence rates * 1.15 (which is more negative).

## Secondary Issues Identified

### Issue A: Dead Code -- PEP_SCENARIO_FILE_MAP Never Used

`run_pep_projections.py` line 47-51:

```python
PEP_SCENARIO_FILE_MAP = {
    "baseline": "baseline",
    "high_growth": "high",
    "restricted_growth": "baseline",
}
```

This mapping is used only by `verify_pep_rate_files()` to check that the high rate file exists. It is never passed to the pipeline. The `migration_rates_pep_high.parquet` file (which has 88% more migration than baseline) is generated but never consumed.

### Issue B: Wasted Adjustment in apply_scenario_rate_adjustments

The pre-engine `apply_scenario_rate_adjustments()` applies `+15%` to the constant PEP migration rates, but these rates are completely overridden by convergence rates inside the engine's `_get_migration_rates()` method. The adjustment is wasted CPU cycles and creates a misleading log message:

```
Scenario high_growth: Applying '+15_percent' migration adjustment to 53 counties
```

This log message implies the adjustment is effective, but it is not.

### Issue C: Convergence Rates Not Adjusted for Scenarios

`apply_scenario_rate_adjustments()` explicitly passes convergence rates through unchanged (line 700-706). The code comment says "Scenario adjustments to time-varying rates are a future enhancement." This means the scenario system has two conflicting adjustment paths:

1. Pre-engine adjustment of constant rates (wasted when convergence rates exist)
2. In-engine adjustment via `apply_migration_scenario()` (actually effective)

This creates a confusing dual-adjustment architecture where the same scenario setting is applied in two places but only one has effect.

## Classification

**This is a DESIGN BUG, not a configuration error.**

The `+15_percent` multiplier is mathematically correct (it does multiply by 1.15) but semantically wrong for this application. The high_growth scenario is intended to represent "counterfactual continuation of elevated post-2020 immigration trends" (ADR-037), which means **more net in-migration** compared to baseline. A symmetric multiplier on net migration does not achieve this when net migration is negative.

The restricted_growth scenario avoids this problem because it uses a `time_varying` dict with explicit factors that are applied through the `intl_share` mechanism, which correctly models the international-only component.

## Recommended Fix

### Option 1: Additive Migration Boost (Recommended)

Instead of multiplying net migration by 1.15, add a fixed annual migration increment derived from the difference between the CBO January 2025 projection and the historical baseline:

```yaml
high_growth:
    migration:
      type: "additive_boost"
      annual_increment: <computed_value>  # Additional net migrants per year
```

This ensures high_growth always produces higher population than baseline regardless of the sign of the base migration rates.

### Option 2: International-Only Positive Multiplier

Apply the `+15%` only to the international migration component (using the same `intl_share` mechanism as restricted_growth), and only in the positive direction:

```yaml
high_growth:
    migration:
      type: "time_varying"
      schedule:
        2025: 1.15  # All years get +15% on international component
      default_factor: 1.15
      intl_share: 0.91
```

The effective factor would be: `1 + 0.91 * (1.15 - 1) = 1 + 0.91 * 0.15 = 1.1365`

This would scale total migration by ~1.14x rather than 1.15x, but it would still produce the wrong result when total net migration is negative.

### Option 3: Use the Pre-Computed High Rate File

The `migration_rates_pep_high.parquet` file already contains higher migration rates that are generated by a different averaging methodology (presumably shorter-period averaging that captures recent high-immigration trends). Route the engine to use this file for the high_growth scenario and skip the `+15%` multiplier entirely.

This would require:
1. `load_demographic_rates()` to accept a scenario parameter and load the appropriate rate file
2. The convergence pipeline to generate scenario-specific convergence rates
3. Removing the engine-level `+15%` application

### Option 4: Immediate Tactical Fix

Change the `+15_percent` string to a `time_varying` dict that uses the restricted_growth mechanism in reverse -- boosting international migration by 15% using the `intl_share` approach, but with factors > 1.0:

```yaml
high_growth:
    migration:
      type: "time_varying"
      schedule: {}  # No year-specific overrides needed
      default_factor: 1.15
      intl_share: 0.91
```

This would use the `effective_factor = 1 - intl_share * (1 - factor) = 1 - 0.91 * (-0.15) = 1.1365` formula. However, this still multiplies total net migration, so it has the same sign problem when convergence rates are net-negative.

### Recommendation

**Option 1 (additive boost) is the correct solution.** The fundamental issue is that multiplicative scaling of net migration is semantically incorrect when the goal is "more immigration." More immigration is an additive concept (more people arriving), not a multiplicative concept (scale all flows by X%).

As an interim measure, the high_growth scenario should either be:
- Deactivated (`active: false`) until properly fixed
- Reframed with a methodology note explaining the limitation

## Impact Assessment

- **Severity**: HIGH -- The scenario produces results that are the opposite of what its name implies
- **User Impact**: Any stakeholder reviewing the three-scenario comparison would see "High Growth" below baseline, undermining confidence in all projections
- **Data Impact**: No impact on baseline or restricted_growth -- those are unaffected
- **Scope**: Affects all 53 counties and all projection years for the high_growth scenario only

## Files Involved

| File | Role |
|------|------|
| `config/projection_config.yaml` (lines 222-228) | Scenario definition with `+15_percent` migration |
| `cohort_projections/core/migration.py` (lines 200-201) | `apply_migration_scenario()` multiplies by 1.15 |
| `cohort_projections/core/cohort_component.py` (lines 244-261) | Engine applies scenario to convergence rates |
| `scripts/pipeline/02_run_projections.py` (lines 680-824) | Pre-engine adjustment (wasted) |
| `scripts/projections/run_pep_projections.py` (lines 47-51) | Dead code `PEP_SCENARIO_FILE_MAP` |
| `data/processed/migration/convergence_rates_by_year.parquet` | Net-negative convergence rates |
| `data/processed/migration/migration_rates_pep_high.parquet` | Unused high migration rates |

## Verification

The root cause can be verified with this test:

```python
import pandas as pd

convergence = pd.read_parquet("data/processed/migration/convergence_rates_by_year.parquet")
yo1 = convergence[convergence["year_offset"] == 1]

baseline_sum = yo1["migration_rate"].sum()        # -17.10
high_sum = (yo1["migration_rate"] * 1.15).sum()    # -19.67

assert high_sum < baseline_sum  # True: high_growth has MORE out-migration
```
