# Methodology Text Audit Report

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-17 |
| **Timestamp** | 2026-02-17T18:40:04Z |
| **Reviewer** | Claude Code (Opus 4.6) |
| **Scope** | All factual claims in `scripts/exports/_methodology.py` METHODOLOGY_LINES, verified against production code and data |
| **Status** | 1 text fix required; 6 stale config entries identified |

---

## Summary

| # | Claim (short) | Lines | Verdict | Fix needed? |
|---|---|---|---|---|
| 1 | Cohort-component, single-year age, sex, race/ethnicity | 39 | CONFIRMED | No |
| 2 | Base year {base_year}, PEP 2024 vintage | 40 | CONFIRMED | No |
| 3 | 20-year horizon, annual steps | 41 | CONFIRMED | No |
| 4 | Fertility: CDC/NCHS (2024 major / 2022 AIAN+Asian), held constant | 42-44 | PARTIALLY CORRECT | No text fix; config is stale |
| 5 | Mortality: CDC/NCHS life tables (2023) with 0.5% annual improvement | 46 | PARTIALLY CORRECT | **YES** |
| 6 | Migration: PEP 2000-2024, BEBR, Rogers-Castro, convergence | 47-50 | CONFIRMED | No |
| 7 | Scenarios: Baseline / Restricted Growth / High Growth + CBO refs | 52-56 | CONFIRMED | No |
| 8 | Geography: 53 ND counties, state = county sums | 58 | CONFIRMED | No |

---

## Claim 1: Model Type

**Text:** `"Model: Cohort-component population projection (single-year age, sex, race/ethnicity)."`

**Verdict: CONFIRMED**

- Engine class: `CohortComponentProjection` in `cohort_projections/core/cohort_component.py`
- `project_single_year()` (lines 218-325) executes survival, births, migration, combine in order
- Config (`config/projection_config.yaml` lines 64-67): `type: "single_year"`, `min_age: 0`, `max_age: 90`
- 6 race/ethnicity categories confirmed in config lines 71-79 and base population loader `RACE_CODE_MAP` (lines 33-42)

---

## Claim 2: Base Year and Data Vintage

**Text:** `"Base year: {base_year} (Census Population Estimates Program, 2024 vintage)."`

**Verdict: CONFIRMED**

- Config line 5: `base_year: 2025`
- Base population loader line 251: `pop_col = "population_2024"` (Vintage 2024 PEP July 1, 2024 estimates)
- Source file: `data/raw/population/nd_county_population.csv`

---

## Claim 3: Projection Horizon

**Text:** `"Horizon: 20 years ({base_year}–{final_year}), annual steps."`

**Verdict: CONFIRMED**

- Config line 6: `projection_horizon: 20`
- Config line 7: `projection_interval: 1`
- Engine loop at `cohort_component.py` line 366: `for year in range(start_year, end_year)` → `range(2025, 2045)` = 20 annual steps

---

## Claim 4: Fertility Data Source

**Text:** `"Fertility: CDC/NCHS age-specific fertility rates (2024 for major groups, 2022 national rates for AIAN/Asian), held constant."`

**Verdict: PARTIALLY CORRECT — text is accurate, config is stale**

### What the text says vs. what the code does

The text is correct. The actual production data file is `data/raw/fertility/asfr_processed.csv`:
- Lines 2-29: year=2024 for total, white_nh, black_nh, hispanic (28 records)
- Lines 30-43: year=2022 for aian_nh and asian_nh (14 records)

Source documented in `data/raw/fertility/DATA_SOURCE_NOTES.md` lines 26-30:
- Primary: CDC NCHS Vital Statistics Rapid Release - Natality Dashboard
- AIAN/Asian: National Vital Statistics Reports, Births: Final Data for 2022

"Held constant" confirmed: config line 108 `assumption: "constant"`, baseline scenario line 185 `fertility: "constant"`.

### Config discrepancy (code maintenance issue)

The config does NOT match the actual data:
- Config line 106: `source: "SEER"` — should be `"CDC_NCHS"`
- Config line 107: `averaging_period: 5` — implies 2018-2022 average, not 2024 rates
- Config line 313: `input_file: "data/raw/fertility/seer_asfr_2018_2022.csv"` — file does not exist

The production pipeline bypasses these config entries because `scripts/pipeline/00_prepare_processed_data.py` (line 125) hardcodes the path to `asfr_processed.csv`.

---

## Claim 5: Mortality Data and Improvement — REQUIRES TEXT FIX

**Text:** `"Mortality: CDC/NCHS life tables (2023) with 0.5% annual improvement."`

**Verdict: PARTIALLY CORRECT — "CDC/NCHS life tables (2023)" is correct; "0.5% annual improvement" is incorrect for the production path**

### What actually happens in production

**Two mechanisms exist. Only one runs in production.**

**Mechanism 1 (fallback, NOT used):** Simple 0.5% annual improvement in `cohort_projections/core/mortality.py` lines 177-226. Config line 114: `improvement_factor: 0.005`. This function exists but is disabled at runtime.

**Mechanism 2 (actual production path):** Census Bureau NP2023 time-varying survival projections, ND-adjusted. Implemented in `cohort_projections/data/process/mortality_improvement.py`. The module docstring (lines 1-15) describes the formula:

```
ND_survival[age, sex, year] = Census_projected[age, sex, year] * ND_adjustment[age, sex]
where: ND_adjustment[age, sex] = ND_CDC_baseline[age, sex] / Census_national_2025[age, sex]
```

Output: `data/processed/mortality/nd_adjusted_survival_projections.parquet` (confirmed on disk, metadata dated 2026-02-17).

**How Mechanism 1 is disabled:** The projection runner (`scripts/pipeline/02_run_projections.py` lines 506-513) loads the NP2023 file. The engine at `cohort_component.py` lines 266-271 overrides the improvement factor to 0.0 when time-varying survival rates are present:

```python
if self.survival_rates_by_year is not None and year in self.survival_rates_by_year:
    survival_config = copy.deepcopy(self.config)
    survival_config.setdefault("rates", {}).setdefault("mortality", {})[
        "improvement_factor"
    ] = 0.0
```

### Required text fix

Replace:
```python
"Mortality: CDC/NCHS life tables (2023) with 0.5% annual improvement.",
```

With:
```python
(
    "Mortality: CDC/NCHS life tables (2023) with time-varying improvement "
    "from Census Bureau NP2023 survival projections, adjusted for North Dakota."
),
```

### Config discrepancy (code maintenance issue)

- Config line 112: `source: "SEER"` — should be `"CDC_life_tables"`
- Config line 113: `life_table_year: 2020` — should be `2023`
- Config line 318: `input_file: "data/raw/mortality/seer_lifetables_2020.csv"` — file does not exist

---

## Claim 6: Migration Pipeline

**Text:** `"Migration: Census PEP components of change (2000–2024), regime-weighted multi-period averaging (BEBR method), Rogers-Castro age allocation, convergence interpolation toward long-term rates."`

**Verdict: CONFIRMED — all four sub-claims verified end-to-end**

### 6a: Census PEP components of change (2000-2024)

- Config lines 117-128: `method: "PEP_components"`, `source: "Census_PEP"`, periods spanning `[2000, 2024]`
- Residual periods (lines 142-148): `[2000,2005], [2005,2010], [2010,2015], [2015,2020], [2020,2024]`
- PEP input file (line 324): `pep_county_components_2000_2024.parquet`
- Loaded at `scripts/pipeline/02_run_projections.py` lines 440-449

### 6b: Regime-weighted multi-period averaging (BEBR method)

- Config line 121: `averaging_method: "BEBR_multiperiod"`
- Config line 127: `combination: "trimmed_average"`
- Implementation: `cohort_projections/data/process/migration_rates.py` — `calculate_multiperiod_averages()` and `calculate_bebr_scenarios()`
- Regime weighting: `cohort_projections/data/process/pep_regime_analysis.py` — classifies counties into regimes (pre_bakken, boom, bust_covid, recovery) with dampening for oil-boom counties (config lines 129-141)

### 6c: Rogers-Castro age allocation

- Implementation: `cohort_projections/data/process/migration_rates.py` — `get_standard_age_migration_pattern()` called with `method="rogers_castro"`, `peak_age=25`
- Allocates county-level total migration to single-year age-specific rates

### 6d: Convergence interpolation toward long-term rates

- Config lines 171-179: `method: "census_bureau_convergence"`, 5-10-5 schedule (recent→medium 5yr, hold 10yr, medium→longterm 5yr)
- Implementation: `cohort_projections/data/process/convergence_interpolation.py` — `calculate_age_specific_convergence()`
- Pipeline: `scripts/pipeline/01b_compute_convergence.py` → output at `data/processed/migration/convergence_rates_by_year.parquet`
- Loaded at `scripts/pipeline/02_run_projections.py` lines 487-501

---

## Claim 7: Scenario Definitions

**Text:** `"Baseline: Recent trend continuation. Restricted Growth: CBO time-varying migration, −5% fertility. High Growth: +15% migration, +5% fertility. CBO Demographic Outlook (Pub. 60875, Jan 2025; Pub. 61879, Jan 2026)."`

**Verdict: CONFIRMED**

### Baseline

Config lines 182-188: `fertility: "constant"`, `migration: "recent_average"`. No adjustments applied.

### Restricted Growth

Config lines 190-203: `fertility: "-5_percent"`, migration `type: "time_varying"` with schedule:
- 2025: 0.20, 2026: 0.37, 2027: 0.55, 2028: 0.78, 2029: 0.91, default: 1.00

Engine processes at `cohort_component.py` lines 258-261 → `cohort_projections/core/migration.py` applies year-specific factors.

### High Growth

Config lines 206-212: `fertility: "+5_percent"`, `migration: "+15_percent"`.

### CBO publication references

Pub. 60875 and 61879 are external CBO documents; cannot be verified in code. Config line 192 references ADR-037 where these are documented.

---

## Claim 8: Geography

**Text:** `"Geography: All 53 North Dakota counties; state totals are county sums."`

**Verdict: CONFIRMED**

- Config lines 10-14: `state: "38"`, `counties: mode: "all"`
- `load_base_population_for_state()` (base_population_loader.py lines 443-493) aggregates all counties via `groupby().agg({"population": "sum"})`
- Config lines 36-38: `validate_aggregation: true`, `aggregation_tolerance: 0.01`

---

## Stale Config Entries Requiring Update

| File | Line | Current | Should Be |
|---|---|---|---|
| `config/projection_config.yaml` | 106 | `source: "SEER"` | `source: "CDC_NCHS"` |
| `config/projection_config.yaml` | 107 | `averaging_period: 5` | Remove or update (2024 rates used directly, no multi-year averaging) |
| `config/projection_config.yaml` | 112 | `source: "SEER"` | `source: "CDC_life_tables"` |
| `config/projection_config.yaml` | 113 | `life_table_year: 2020` | `life_table_year: 2023` |
| `config/projection_config.yaml` | 313 | `input_file: "data/raw/fertility/seer_asfr_2018_2022.csv"` | `input_file: "data/raw/fertility/asfr_processed.csv"` |
| `config/projection_config.yaml` | 318 | `input_file: "data/raw/mortality/seer_lifetables_2020.csv"` | `input_file: "data/raw/mortality/survival_rates_processed.csv"` |

These do not affect production output (the pipeline bypasses them via hardcoded paths in `00_prepare_processed_data.py`) but create confusion for anyone reading the config.
