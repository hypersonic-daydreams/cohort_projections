---
title: "Wire Census Bureau Convergence Interpolation into Projection Engine"
created: 2026-02-13T11:00:00-06:00
status: planned
author: Claude Code (Opus 4.6)
related_adrs:
  - ADR-036 (BEBR Multi-Period & Census Bureau Interpolation)
related_files:
  - cohort_projections/data/process/migration_rates.py
  - cohort_projections/core/cohort_component.py
  - cohort_projections/geographic/multi_geography.py
  - scripts/pipeline/02_run_projections.py
  - tests/test_data/test_bebr_averaging.py
  - tests/test_core/test_cohort_component.py
---

# Wire Census Bureau Convergence Interpolation into Projection Engine

## Context

The BEBR multi-period averaging and oil county dampening are fully wired in and producing baseline projections. However, the engine applies **constant migration rates** every year — the same table for year 1 and year 20. The existing `calculate_interpolated_rates()` function (migration_rates.py:1150) generates year-varying county-level net migration using Census Bureau convergence scheduling, but it's not connected to the projection engine.

**Problem**: Counties like Ward (-21.1%) and Grand Forks (-13.7%) show overly pessimistic projections because 20+ years of mostly-negative migration history equally weights the BEBR trimmed average. Time-varying rates would start near recent trends and gradually converge to long-term averages, producing more realistic projections.

**Goal**: Wire convergence interpolation so migration rates vary by projection year:
- Years 1-5: Transition from recent (2022-2024) average to medium (2014-2024) average
- Years 6-15: Hold at medium-term average
- Years 16-20: Transition from medium to long-term (2000-2024) average

## Current State (as of 2026-02-13)

- **1018 tests passing**, 0 failures
- BEBR multi-period averaging: fully wired
- Oil county dampening (0.60 factor, 5 counties): fully wired
- `calculate_interpolated_rates()`: EXISTS in migration_rates.py but NOT wired into engine
- Projection engine: applies identical migration rates every year
- Config (`projection_config.yaml`): already has `rates.migration.interpolation` section

## Approach: Per-County Scaling Factors

Rather than passing 20 year-indexed DataFrames through the pipeline, compute a **single float scaling factor** per county per year: `interpolated_total / baseline_total`. The engine multiplies the constant baseline migration table by this factor each year.

- Lightweight: 20 floats per county (vs 20 × 1,092-row DataFrames)
- Minimal engine changes: one dict lookup + one multiplication per year
- Backward compatible: `None` convergence_factors = existing constant behavior
- Preserves age/sex/race distribution (uniform scaling)
- Edge case: if baseline total ≈ 0, factor = 1.0 (0 × 1.0 = 0, safe)

**Convergence applies only to baseline scenario** (migration: "recent_average"). Scenarios like high_growth (+25%) and zero_migration manage their own adjustments.

---

## Files to Modify

| File | Change |
|------|--------|
| `cohort_projections/data/process/migration_rates.py` | Add `compute_convergence_scaling_factors()`; save JSON in `process_pep_migration_rates()` |
| `cohort_projections/core/cohort_component.py` | Accept + apply `convergence_factors` in engine |
| `cohort_projections/geographic/multi_geography.py` | Thread `convergence_factors` through both functions |
| `scripts/pipeline/02_run_projections.py` | Load JSON, pass through pipeline |
| `tests/test_data/test_bebr_averaging.py` | Unit tests for scaling factor computation |
| `tests/test_core/test_cohort_component.py` | Engine integration tests |

**Unchanged**: migration.py (apply_migration() is fine), projection_config.yaml (interpolation config already exists), test_pep_pipeline.py

---

## Step 1: Add `compute_convergence_scaling_factors()` to migration_rates.py

Insert after `calculate_interpolated_rates()` (after line 1237), before `validate_migration_data()`.

```python
def compute_convergence_scaling_factors(
    pep_data: pd.DataFrame,
    bebr_baseline: pd.DataFrame,
    interpolation_config: dict[str, Any],
    projection_years: int = 20,
) -> dict[str, dict[int, float]]:
```

**Logic**:
1. Extract `recent_period`, `medium_period`, `longterm_period` from `interpolation_config`
2. Call `calculate_period_average(pep_data, *period)` for each (reuse existing function)
3. Call `calculate_interpolated_rates(recent, medium, longterm, ...)` (reuse existing function)
4. For each county in `bebr_baseline`:
   - If `abs(baseline_total) < 1.0`: all factors = 1.0
   - Else: `factor[year] = interpolated[year][county] / baseline_total`
5. Return `dict[str, dict[int, float]]` (geoid → year_offset → factor)

Add to `__all__`.

---

## Step 2: Modify `process_pep_migration_rates()` to save convergence factors

After line 1782 (dampening), before line 1790 (Rogers-Castro), insert:

```python
# Step 3c: Compute convergence scaling factors (if configured)
interpolation_config = config.get("rates", {}).get("migration", {}).get("interpolation")
if interpolation_config and interpolation_config.get("method") == "census_bureau_convergence":
    projection_years = config.get("project", {}).get("projection_horizon", 20)
    convergence_factors = compute_convergence_scaling_factors(
        pep_data=pep_df,
        bebr_baseline=bebr_scenarios["baseline"],
        interpolation_config=interpolation_config,
        projection_years=projection_years,
    )
    factors_file = output_dir / "convergence_factors.json"
    serializable = {
        geoid: {str(year): factor for year, factor in yf.items()}
        for geoid, yf in convergence_factors.items()
    }
    with open(factors_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved convergence factors for {len(convergence_factors)} counties")
```

Variables `pep_df`, `bebr_scenarios`, `config`, `output_dir` are all in scope at this point.

---

## Step 3: Modify engine — `cohort_component.py`

### `__init__()` (line 38)

Add parameter after config:

```python
def __init__(self, base_population, fertility_rates, survival_rates,
             migration_rates, config=None, convergence_factors=None):
```

Store: `self.convergence_factors = convergence_factors` (after line 82)

### `project_single_year()` (line 131)

After line 174 (apply_migration_scenario), before line 176 (Step 1: Apply survival):

```python
# Apply convergence scaling (time-varying migration)
if self.convergence_factors is not None:
    year_offset = year - self.base_year + 1
    factor = self.convergence_factors.get(year_offset, 1.0)
    if factor != 1.0:
        migration_col = (
            "net_migration" if "net_migration" in migration_rates.columns
            else "migration_rate"
        )
        migration_rates[migration_col] = migration_rates[migration_col] * factor
        logger.debug(f"Year {year}: convergence factor {factor:.4f}")
```

---

## Step 4: Thread through multi_geography.py

### `run_single_geography_projection()` (line 54)

Add parameter: `convergence_factors: dict[int, float] | None = None`

Pass through at line 140:
```python
projection_engine = CohortComponentProjection(
    ..., convergence_factors=convergence_factors, config=config,
)
```

### `run_multi_geography_projections()` (line 234)

Add parameter: `convergence_factors: dict[str, dict[int, float]] | None = None`

In parallel worker (line 321) and serial loop (line 373):
```python
county_convergence = convergence_factors.get(fips) if convergence_factors else None
# Pass county_convergence to run_single_geography_projection()
```

---

## Step 5: Wire in pipeline — `02_run_projections.py`

### Add `_load_convergence_factors()` helper (after line 736)

```python
def _load_convergence_factors(config):
    """Load convergence factors from JSON if available."""
    interp = config.get("rates", {}).get("migration", {}).get("interpolation")
    if not interp or interp.get("method") != "census_bureau_convergence":
        return None
    pep_output = config.get("pipeline", {}).get("data_processing", {}).get(
        "migration", {}).get("pep_output", "...")
    factors_path = project_root / Path(pep_output).parent / "convergence_factors.json"
    if not factors_path.exists():
        return None
    with open(factors_path) as f:
        raw = json.load(f)
    return {geoid: {int(k): v for k, v in yf.items()} for geoid, yf in raw.items()}
```

### Modify `run_geographic_projections()` (line 739)

Add `convergence_factors` parameter. Gate by scenario:
```python
scenario_config = config.get("scenarios", {}).get(scenario, {})
migration_setting = scenario_config.get("migration", "recent_average")
effective_convergence = (
    convergence_factors if migration_setting in ("recent_average", "constant") else None
)
```

Pass `effective_convergence` to `run_multi_geography_projections()` (line 845).

### Modify `run_all_projections()` (~line 999)

After loading demographic rates, load convergence factors:
```python
convergence_factors = _load_convergence_factors(config)
```

Pass to `run_geographic_projections()`.

---

## Step 6: Tests

### Unit tests in test_bebr_averaging.py

`TestComputeConvergenceScalingFactors`:
- `test_factors_match_interpolation_ratio`: known values → verify factor = interpolated / baseline
- `test_year_5_equals_medium_over_baseline`: at year 5, interpolated = medium
- `test_years_6_to_15_hold_constant`: all factors equal medium/baseline
- `test_zero_baseline_gives_factor_one`: edge case
- `test_multiple_counties_independent`: each county gets own factors
- `test_custom_convergence_schedule`: non-default schedule

### Engine tests in test_cohort_component.py

`TestConvergenceFactors`:
- `test_convergence_varies_migration_by_year`: factors {1: 2.0, 2: 0.5} → year 1 pop > year 1 without, year 2 pop < year 2 without
- `test_no_convergence_backward_compatible`: `convergence_factors=None` = identical to omitting param
- `test_all_factors_one_equals_constant`: factors all 1.0 = same as no factors

---

## Step 7: Re-run baseline projections

After wiring everything:
```bash
# Re-process migration rates (generates convergence_factors.json)
python scripts/data_processing/process_pep_rates.py --scenarios baseline low high

# Re-run baseline projections (now uses time-varying migration)
python scripts/pipeline/02_run_projections.py

# Compare results — Ward and Grand Forks should show less extreme decline
```

---

## Verification

```bash
pytest tests/test_data/test_bebr_averaging.py       # New scaling factor tests
pytest tests/test_core/test_cohort_component.py      # Engine convergence tests
pytest tests/test_integration/test_pep_pipeline.py   # Pipeline still works
pytest                                               # All ~1018 tests pass
pre-commit run --all-files                           # Linting clean
```

---

## Implementation Order

1. Step 1 + tests (purely additive — new function, no existing code changes)
2. Step 2 (save JSON — additive side effect, no consumers yet)
3. Step 3 + tests (engine changes — optional param, backward compatible)
4. Step 4 (multi_geography threading — optional params)
5. Step 5 (pipeline wiring — connects everything)
6. Step 7 (re-run projections, verify results)

---

## Key Data Flow Diagram

```
process_pep_migration_rates()              [data processing step]
  ├── BEBR multi-period averages           (already implemented)
  ├── BEBR scenarios (baseline/low/high)   (already implemented)
  ├── Oil county dampening                 (already implemented)
  ├── compute_convergence_scaling_factors() ← NEW
  │     Uses: calculate_period_average()    (existing)
  │     Uses: calculate_interpolated_rates()(existing)
  │     Saves: convergence_factors.json
  └── Rogers-Castro distribution → parquet (already implemented)

02_run_projections.py                      [pipeline execution]
  ├── load_demographic_rates()             (unchanged)
  ├── _load_convergence_factors()          ← NEW (reads JSON)
  └── run_geographic_projections()
        ├── apply_scenario_rate_adjustments() (unchanged)
        ├── Gate: only pass factors for baseline scenario
        └── run_multi_geography_projections()
              └── For each county:
                    └── CohortComponentProjection(convergence_factors={1: 0.95, 2: 0.97, ...})
                          └── project_single_year(year)
                                ├── migration_rates = self.migration_rates.copy()
                                ├── apply_migration_scenario()  (unchanged, no-op for baseline)
                                ├── migration_rates *= convergence_factors[year_offset] ← NEW
                                └── apply_migration(population, migration_rates, year)
```

## Mortality Improvement Note

Separately from this plan, the user asked about mortality improvement. The engine applies constant survival rates each year, but the config specifies `improvement_factor: 0.005` (0.5%/year). This is a separate, smaller enhancement that could use the same pattern: pass a mortality improvement factor to the engine, and in `project_single_year()`, apply `survival_rate *= (1 + improvement_factor) ^ year_offset`. This is deferred to a future session.
