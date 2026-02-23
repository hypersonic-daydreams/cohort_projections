# ADR-049: College-Age Smoothing Propagation to Convergence Pipeline

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-23

## Scope
Ensure college-age migration rate smoothing is applied to convergence pipeline inputs, not only to averaged rates

**Fixes**: A bug where convergence rates for college counties use unsmoothed period-level rates, inflating projected growth by 10-15 percentage points

## Context

### Problem: College-Age Smoothing Only Applied to Averaged Rates

The residual migration pipeline applies college-age smoothing (blending extreme 20-24 in-migration rates with the statewide average) as **Step 7** of the pipeline, after period averaging in Step 5. However, the convergence pipeline reads its input from `residual_migration_rates.parquet`, which is saved in **Step 8** — containing the **period-level** rates that bypass college-age smoothing.

The pipeline order:

```
Step 5: Period averaging  → residual_migration_rates_averaged.parquet (gets smoothing)
Step 6: College county identification
Step 7: College-age smoothing (applied to averaged rates only)
Step 8: Save period-level rates → residual_migration_rates.parquet (NO smoothing)
     ↓
Convergence pipeline reads residual_migration_rates.parquet (unsmoothed)
```

### Impact: Cass County (Fargo) Overshoot

The effect is quantified for Cass County (38017), the state's largest county:

| Rate | Averaged (smoothed) | Convergence (unsmoothed) | Ratio |
|------|:-------------------:|:-----------------------:|:-----:|
| 20-24 Male migration rate | **0.038** | **0.124** | 3.3x |
| 20-24 Female migration rate | ~0.035 | ~0.110 | 3.1x |

The convergence pipeline uses the unsmoothed 0.124 rate, which compounds over 30 years to produce a +63% projected growth for Cass County. The SDC 2024 reference projection shows +48%. The ~15 percentage point overshoot is largely attributable to this bug.

### Why This Is a Bug, Not a Design Choice

1. The college-age smoothing was intentionally implemented (in `residual_migration.py`, lines 1077-1086) because raw 20-24 in-migration rates for university counties reflect transient student enrollment, not permanent migration. A 12.4% annual in-migration rate for 20-24 year-olds implies that Fargo receives the equivalent of 12% of its young adult population every year as new permanent residents — which is implausible.

2. The averaged rates correctly apply this smoothing, producing a 3.8% rate that blends county-specific patterns with statewide averages using a 50/50 blend factor.

3. The convergence rates bypass this smoothing entirely because they read the period-level file, which is saved without smoothing. This is an ordering/data-flow bug, not an intentional design decision.

### Affected Counties

All counties classified as college counties are affected. The current college county list includes:
- Cass (38017) — NDSU
- Grand Forks (38035) — UND
- Ward (38101) — Minot State
- Burleigh (38015) — University of Mary, Bismarck State

The effect is most pronounced for Cass and Grand Forks, which have the largest 20-24 in-migration rates.

## Decision

### Apply College-Age Smoothing to Period-Level Rates Before Saving

**Decision**: Move college-age smoothing to occur on the period-level rates (before Step 8's save), so that `residual_migration_rates.parquet` contains smoothed rates. This ensures the convergence pipeline inherits the smoothing automatically.

**Implementation**: In `residual_migration.py`, apply the college-age smoothing to each period's rates (not just the averaged rates) before saving `residual_migration_rates.parquet`. The smoothing uses the same blend formula:

```python
smoothed_rate = blend_factor * county_rate + (1 - blend_factor) * statewide_rate
```

Where `blend_factor = 0.5` (configured in projection_config.yaml) and the smoothing applies to the 20-24 age group for identified college counties.

**Revised pipeline order**:

```
Step 5: Period averaging → residual_migration_rates_averaged.parquet
Step 6: College county identification
Step 7: College-age smoothing (applied to BOTH period-level AND averaged rates)
Step 8: Save period-level rates → residual_migration_rates.parquet (NOW smoothed)
     ↓
Convergence pipeline reads residual_migration_rates.parquet (smoothed ✓)
```

### Why Not Smooth in the Convergence Pipeline Instead?

The convergence pipeline (`convergence_interpolation.py`) could apply smoothing after loading period rates. However:
1. This would duplicate the smoothing logic in two places
2. The residual pipeline is the authoritative source for rate computation — smoothing belongs there
3. Any other consumer of `residual_migration_rates.parquet` would also get incorrect (unsmoothed) rates

### Alternatives Considered

1. **Smooth in convergence pipeline**: Rejected — duplicates logic, creates maintenance burden
2. **Save a separate smoothed period file**: Rejected — adds complexity with no benefit over fixing the existing file
3. **Only smooth the averaged rates, lower the rate cap for college ages**: This would catch the worst outliers but not the moderate inflation (0.10-0.12 rates that pass the 0.15 cap). The root cause is the missing smoothing, not an insufficient cap.

## Consequences

### Positive

1. **Fixes Cass County overshoot**: Projected growth should decrease from +63% toward ~+48% (SDC reference range), a ~15 percentage point correction
2. **Consistent smoothing**: Both averaged and convergence rate paths receive the same treatment
3. **Simple fix**: Single code change in the residual pipeline; no convergence pipeline changes needed
4. **No new data required**: Uses existing smoothing logic, existing college county list, existing blend factor

### Negative

1. **Rerun required**: Residual migration, convergence, and projection steps must all be re-run
2. **Historical change**: The `residual_migration_rates.parquet` file content changes. Any analysis referencing the old file should be re-validated.
3. **Possible under-smoothing for some counties**: The 50/50 blend factor may not be aggressive enough for Grand Forks (UND), which has even more pronounced student migration patterns. This can be tuned in a future iteration.

### Expected Impact

| County | Before | After (est.) | Change |
|--------|:------:|:------------:|:------:|
| Cass (Fargo) | +63% | ~+48% | -15pp |
| Grand Forks | +24% | ~+18% | -6pp |
| Burleigh | +35% | ~+32% | -3pp |
| Ward (Minot) | -5% | ~-3% | +2pp |

Note: Ward County may benefit slightly because the smoothing also applies to out-migration patterns at college ages, but the primary effect for Ward is the broader negative migration issue addressed in ADR-052.

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `cohort_projections/data/process/residual_migration.py` | Apply smoothing to period-level rates in Step 7, before saving in Step 8 |
| `data/processed/migration/residual_migration_rates.parquet` | Output changes (smoothed rates) |
| `data/processed/migration/residual_migration_rates_averaged.parquet` | Unchanged (already smoothed) |

### Testing Strategy

1. **Unit test**: Verify that period-level rates for college counties have 20-24 rates smoothed (< raw rate)
2. **Convergence input check**: Load convergence rates and verify 20-24 rates for Cass are ~0.04, not ~0.12
3. **Regression**: Non-college counties are unaffected
4. **Integration**: Full pipeline produces Cass growth in the +45-50% range

### Pipeline Rerun Required

1. **Step 01**: Residual migration (applies smoothing to period rates)
2. **Step 01b**: Convergence interpolation (reads smoothed period rates)
3. **Step 02**: Projections
4. **Step 03**: Exports

## Implementation Results (2026-02-23)

### Changes Made

| File | Change |
|------|--------|
| `cohort_projections/data/process/residual_migration.py` | Moved college-age smoothing to Step 5 of the pipeline, applied to each period's rates individually before combining and saving. Step 6b explicitly skips re-smoothing averaged rates to prevent double-smoothing. Updated module docstring and function docstring to document the corrected pipeline order. Fixed step numbering in all log messages and comments for consistency (Steps 1-7). Added ADR-049 explanatory comments at the smoothing step. |
| `tests/test_data/test_residual_migration.py` | Added `TestCollegeAgeSmoothingPropagation` class with 5 tests: (1) period-level smoothing reduces extreme college-age rates, (2) non-college counties are unchanged, (3) averaged rates inherit period-level smoothing without a second pass, (4) double-smoothing over-corrects (verifies ADR-049 avoidance), (5) smoothed period rates propagate correctly to convergence input (the combined DataFrame that gets saved as `residual_migration_rates.parquet`). |

### Verification

- **Pipeline order confirmed**: College-age smoothing now executes at Step 5, before period rates are concatenated (Step 5b), averaged (Step 6), and saved (Step 7). The convergence pipeline reads `residual_migration_rates.parquet` which contains smoothed period-level rates.
- **No double-smoothing**: Step 6b logs a skip message and does NOT re-apply smoothing to averaged rates. The `test_no_double_smoothing` test verifies that applying smoothing twice produces over-corrected rates that are strictly lower than single-application rates.
- **Non-college counties unaffected**: Tests confirm that counties not in the college county list (`38017`, `38035`, `38101`, `38015`) have identical rates before and after smoothing.
- **Convergence pipeline unchanged**: No changes were made to `convergence_interpolation.py`. It continues to read `residual_migration_rates.parquet` and now automatically receives smoothed rates.
- **All 1,195 tests pass** (5 skipped for missing data files).

### Revised Pipeline Order

The `run_residual_migration_pipeline()` function now follows this step sequence:

```
Step 1:  Load population snapshots
Step 2:  Load survival rates
Step 3:  Compute residual migration per period (with oil-boom + male dampening)
Step 4:  PEP recalibration for reservation counties (ADR-045)
Step 5:  College-age smoothing on period-level rates (ADR-049)
Step 5b: Combine all period rates into single DataFrame
Step 6:  Average rates across periods
Step 6b: Skip re-smoothing averaged rates (ADR-049: avoids double-smoothing)
Step 7:  Save output files (period-level + averaged)
```

### Key Design Notes

1. **Smoothing is per-period, per-county**: The `apply_college_age_adjustment()` function is called once per period, computing a fresh statewide average for that period and blending. This means each period's statewide average reflects only that period's data, which is the correct behavior (statewide averages vary across periods).
2. **No config changes needed**: The existing `rates.migration.domestic.adjustments.college_age` config section already specifies the college counties, age groups, method, and blend factor. The fix only changes *when* in the pipeline this config is applied, not *what* is applied.
3. **Averaged rates are implicitly smoothed**: Since averaging operates on already-smoothed period rates, the resulting averaged rates are also smoothed. This matches the pre-ADR-049 behavior for the averaged file, but now also extends to the period-level file.

## References

1. **Sanity Check Finding**: Cass County +63% vs SDC +48%, identified as college-age smoothing gap
2. **Residual Pipeline**: `cohort_projections/data/process/residual_migration.py`, Step 5 (college-age smoothing)
3. **Convergence Pipeline**: `cohort_projections/data/process/convergence_interpolation.py`, line 447 (reads `residual_migration_rates.parquet`)
4. **SDC 2024 Projections**: `data/raw/nd_sdc_2024_projections/` — reference for expected growth ranges

## Revision History

- **2026-02-23**: Accepted and implemented — moved college-age smoothing to period-level rates (Step 5), added 5 unit tests, updated docstrings and step numbering
- **2026-02-18**: Initial version (ADR-049) — Fix college-age smoothing propagation to convergence pipeline

## Related ADRs

- **ADR-043: Age-Aware Migration Rate Cap** — The 15% college-age cap partially compensates for missing smoothing, but is insufficient (0.124 passes the 0.15 cap)
- **ADR-036: Migration Averaging Methodology** — Defines the averaging pipeline where smoothing is applied
- **ADR-040: Extend Boom Dampening** — Analogous rate adjustment that IS correctly applied before both averaged and period-level saves
