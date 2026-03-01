# PP3-S08 Implementation Kickoff Packet: City/Place Projections

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Claude (AI Agent) -- requires human review before acceptance |
| **Scope** | PP3-S08 execution-ready implementation task list for Phase 1 city/place projections |
| **Status** | Draft |
| **Related ADR** | ADR-033 |
| **Scoping Gate** | PP3-S07 Approved 2026-02-28 |

---

## Executive Summary

This document is the execution-ready task list for the PP-003 Phase 1 city/place projection workstream. It translates the approved scoping artifacts (S01-S07) into concrete implementation tasks organized in six phases:

1. **Data Assembly** -- Build the prerequisite crosswalk, historical time series, and share artifacts (no model code yet).
2. **Model Implementation** -- Implement the logit-linear share-trending engine, constraint mechanisms, and configuration wiring.
3. **Backtesting** -- Execute the S05 backtest protocol, evaluate the 2x2 variant matrix, select the winning model, and document results.
4. **Production Pipeline Integration** -- Wire place projections into the existing pipeline and produce output files per the S06 contract.
5. **Export and Workbook Integration** -- Build the place workbook, update the provisional workbook, and wire export flags.
6. **Validation and Documentation** -- Integration tests, end-to-end validation, ADR update, and methodology documentation.

**Place universe**: 90 projected places (9 HIGH, 9 MODERATE, 72 LOWER) out of 355 active places.
**Projection horizon**: 2025-2055 annual, three scenarios (baseline, restricted_growth, high_growth).
**Method**: Share-of-county logit-linear trending with 2x2 backtest variant selection.

All human decisions have been made during S01-S07 scoping. The remaining work is implementation, validation, and documentation. Most tasks are suitable for AI agent execution; tasks requiring human judgment are flagged.

---

## Phase 1: Data Assembly

No model code is written in this phase. The goal is to produce the three prerequisite data artifacts that the model needs.

### IMP-01: Build Place-County Crosswalk

**Description**: Build the authoritative place-to-county crosswalk file using Census 2020 relationship files or TIGER shapefiles. Apply the S03 assignment rules (single-county vs. multi-county-primary), flag dissolved places as `historical_only`, and produce the supplemental multi-county detail table.

**Files to create**:
- `scripts/data/build_place_county_crosswalk.py` -- assembly script
- Output: `data/processed/geographic/place_county_crosswalk_2020.csv` (primary crosswalk)
- Output: `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv` (audit table)

**Files to modify**:
- None (new directory `data/processed/geographic/` will be created by the script)

**Dependencies**: None (first task).

**Tests required**:
- `tests/test_data/test_place_county_crosswalk.py`:
  - All 355 active places have exactly one row in the primary crosswalk.
  - No null/invalid FIPS in `place_fips` or `county_fips`.
  - `area_share` in (0, 1] for all rows.
  - `assignment_type` is one of `single_county`, `multi_county_primary`.
  - Multi-county places appear in both the primary crosswalk and the detail table.
  - Dissolved places (Bantry 04740, Churchs Ferry 14140) are flagged `historical_only`.
  - Spot-check: Fargo -> Cass (38017), Bismarck -> Burleigh (38015), Grand Forks -> Grand Forks (38035), Minot -> Ward (38101), West Fargo -> Cass (38017), Williston -> Williams (38105).
  - Primary crosswalk is unique on `place_fips`.

**Acceptance criteria**:
1. Crosswalk file has 355 rows (one per active ND place in the 2020 vintage).
2. All S03 QA rules pass (Section 5 of the S03 note).
3. Every place in `data/raw/geographic/nd_places.csv` maps to exactly one county.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. The Census relationship file or TIGER data is available in the shared data directory. The assignment rules are deterministic and fully specified in S03.

**Human input needed**: Review the crosswalk output for plausibility, especially multi-county assignments. Confirm source data path if relationship files are not in the expected location.

---

### IMP-02: Assemble Historical Place Population Time Series

**Description**: Combine the three PEP vintage files (`sub-est00int` for 2000-2009, `sub-est2020int` for 2010-2019, `sub-est2024` for 2020-2024) into a single long-format place population file covering 2000-2024. Apply the S02 handoff-window rules and lock `sub-est2020int` as canonical for 2010-2019. Filter to ND places (state FIPS 38). Join with the IMP-01 crosswalk to attach `county_fips`.

**Files to create**:
- `scripts/data/assemble_place_population_history.py` -- assembly script
- Output: `data/processed/place_population_history_2000_2024.parquet`

**Files to modify**:
- None

**Dependencies**: IMP-01 (crosswalk needed for county_fips join).

**Tests required**:
- `tests/test_data/test_place_population_history.py`:
  - Output has 25 years x 357 unique places = 8,925 rows (or 8,915 per S02 if some place-years are absent). Validate actual row count against S02 expectation.
  - Year range is exactly 2000-2024 with no gaps.
  - No null population values.
  - Dissolved places (Bantry, Churchs Ferry) have records through 2019 but not 2020-2024.
  - All 355 active places have records for all 25 years.
  - The two dissolved places have `historical_only` flag from crosswalk join.
  - Column schema: `place_fips`, `place_name`, `county_fips`, `year`, `population`.
  - Spot-check: Fargo 2024 population matches the `sub-est2024` source file.
  - Vintage source column traces each row to the correct source file.

**Acceptance criteria**:
1. Continuous 2000-2024 annual coverage for all active places.
2. No missing population cells.
3. County assignment from crosswalk is attached to every row.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. Data sources are identified in S02 and exist in `~/workspace/shared-data/census/popest/`. Rules are deterministic.

**Human input needed**: Confirm shared data paths. Review dissolved-place handling if row counts do not match S02 expectations.

---

### IMP-03: Compute Historical Place Shares

**Description**: For each place-year, compute `share = place_population / county_population`. County population totals come from the existing `data/processed/pep_county_components_2000_2025.parquet` (or the relevant county total column). Apply epsilon clamping (epsilon = 0.001) to shares before storing. Also compute balance-of-county shares for each county-year.

**Files to create**:
- `cohort_projections/data/process/place_shares.py` -- module with `compute_historical_shares()` function
- Output: `data/processed/place_shares_2000_2024.parquet`

**Files to modify**:
- None

**Dependencies**: IMP-02 (place population history), IMP-01 (crosswalk for county assignment).

**Tests required**:
- `tests/test_data/test_place_shares.py`:
  - All shares are in (0, 1] before clamping.
  - After epsilon clamping, all shares are in [0.001, 0.999].
  - For each county-year, sum of place shares + balance-of-county share = 1.0 (within floating-point tolerance).
  - Balance-of-county shares are non-negative for all county-years.
  - Counties with no projected places (all places < 500) have no share rows.
  - Spot-check: Fargo's share of Cass County is plausible (roughly 0.6-0.7).

**Acceptance criteria**:
1. Share file covers all 25 years for all 355 active places.
2. County shares sum to 1.0 (within tolerance) for every county-year.
3. Epsilon clamping applied correctly.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. Pure computation from existing artifacts.

---

### IMP-04: Assign Confidence Tiers

**Description**: Using the 2024 PEP population from the IMP-02 time series, assign each place to a confidence tier: HIGH (>10,000), MODERATE (2,500-10,000), LOWER (500-2,500), EXCLUDED (<500). Flag places within 5% of tier boundaries as `tier_boundary`. Store tier assignments in the crosswalk or a separate reference file.

**Files to create or modify**:
- Modify `data/processed/geographic/place_county_crosswalk_2020.csv` to add `confidence_tier` and `tier_boundary` columns (or create a separate `data/processed/place_tier_assignments.parquet`).

**Dependencies**: IMP-02 (2024 population values).

**Tests required**:
- Add to `tests/test_data/test_place_county_crosswalk.py` or create `tests/test_data/test_place_tier_assignments.py`:
  - 9 HIGH places, 9 MODERATE places, 72 LOWER places, 265 EXCLUDED places.
  - Tier boundaries match S01 thresholds exactly.
  - `tier_boundary` flag set for places within 5% of any threshold.
  - Fargo, Bismarck, Grand Forks, Minot, West Fargo, Williston, Dickinson, Mandan, Jamestown are all HIGH.
  - Sum of tier counts = 355 active places.

**Acceptance criteria**:
1. Tier counts match S01 scope envelope (9 / 9 / 72 / 265).
2. Tier boundary flags are correct.
3. Tests pass.

**Agent suitability**: AI agent -- suitable. Deterministic threshold application.

---

## Phase 2: Model Implementation

Core share-trending engine. No forward projections are run yet; backtest execution is Phase 3.

### IMP-05: Core Share-Trending Module

**Description**: Implement the logit-linear share-trending model as specified in S04. This is the central computation module. It must support both OLS and WLS fitting variants, both constraint mechanisms, time-variable centering, and the balance-of-county independent trend with reconciliation.

**Files to create**:
- `cohort_projections/data/process/place_share_trending.py`

**Functions to implement**:

1. `logit_transform(shares, epsilon=0.001)` -- Clamp and apply logit.
2. `inverse_logit(logit_values)` -- Back-transform to share space.
3. `fit_share_trend(logit_shares, years, method="ols", lambda_decay=0.9)` -- Fit logit-linear trend via OLS or WLS. Return intercept, slope. Center time variable at midpoint of fitting window.
4. `project_shares(intercept, slope, projection_years, center_year)` -- Project shares forward via back-transform.
5. `apply_proportional_rescaling(shares_dict)` -- S04 Section 4.1 constraint.
6. `apply_cap_and_redistribute(shares_dict, base_shares_dict)` -- S04 Section 4.2 constraint with iterative clamping edge case.
7. `reconcile_county_shares(place_shares, balance_share, constraint_method, base_shares)` -- S04 Section 6.2 reconciliation. Flag if adjustment > 0.05.
8. `trend_all_places_in_county(place_share_history, county_pop_history, config)` -- Orchestrate: fit trends for all places + balance, project, reconcile, return projected shares.

**Files to modify**:
- `cohort_projections/data/process/__init__.py` -- add import for new module.

**Dependencies**: IMP-03 (historical shares for testing).

**Tests required**:
- `tests/test_data/test_place_share_trending.py`:
  - **Logit transform**: Correct logit values for known inputs; epsilon clamping works at boundaries (0 and 1).
  - **Inverse logit**: Round-trip consistency (`inverse_logit(logit_transform(s)) ~= clamp(s)`).
  - **OLS fit**: Known regression coefficients for a simple synthetic series.
  - **WLS fit**: Weights correctly applied; lambda=0.9 produces expected weight vector.
  - **Time centering**: Projected shares identical regardless of centering (invariance property).
  - **Proportional rescaling**: Shares sum to 1.0 after rescaling; proportions preserved.
  - **Cap-and-redistribute**: Declining places unchanged; growing places absorb excess; iterative clamping edge case handles over-correction.
  - **Cap-and-redistribute fallback**: When all places grew, reduces to proportional rescaling.
  - **Reconciliation**: Balance-of-county treated as additional "place"; adjustment flag raised when |T - 1| > 0.05.
  - **Single-place county**: Trivial reconciliation (near-zero adjustment).
  - **County with no projected places**: Returns empty (no share output).
  - **Near-zero projected share**: If projected share < epsilon, population set to zero.
  - **End-to-end synthetic county**: 3-place synthetic county with known trends; verify projected shares sum to 1.0 in all years.

**Acceptance criteria**:
1. All S04 mathematical specifications implemented exactly.
2. Both fitting variants (OLS, WLS) work correctly.
3. Both constraint mechanisms work correctly, including edge cases.
4. Balance-of-county reconciliation works correctly.
5. All configuration parameters read from config (no hard-coded values).
6. All tests pass.

**Agent suitability**: AI agent -- suitable. Mathematical specification is fully defined in S04 with pseudocode. This is the most complex implementation task but requires no human judgment.

---

### IMP-06: Place Projection Orchestrator

**Description**: Build the orchestrator that applies share trends to county projections to produce place populations. For each scenario, for each county with projected places: load county projection, apply winning share model, multiply shares by county totals, allocate age-sex detail by tier (HIGH: 18 5-year age groups x 2 sex; MODERATE: 6 broad groups x 2 sex; LOWER: total only), and write per-place output files.

**Files to create**:
- `cohort_projections/data/process/place_projection_orchestrator.py`

**Functions to implement**:

1. `run_place_projections(scenario, config, variant_winner)` -- Top-level entry point. Iterate over counties, apply share trends, allocate age-sex detail, write outputs.
2. `allocate_age_sex_detail(place_total, county_cohort_df, tier)` -- Proportionally allocate county cohort structure to place total. HIGH tier gets 18-bin age x sex. MODERATE tier gets 6-bin broad age x sex. LOWER tier gets total only.
3. `write_place_outputs(place_df, metadata, scenario, config)` -- Write parquet, metadata JSON, and summary CSV per the S06 naming convention and schema.
4. `write_run_level_metadata(all_places_metadata, scenario, config)` -- Write `places_metadata.json`.
5. `write_places_summary(all_places_summaries, balance_rows, scenario, config)` -- Write `places_summary.csv` including balance-of-county rows.

**Files to modify**:
- `cohort_projections/data/process/__init__.py` -- add import.

**Dependencies**: IMP-05 (share-trending module), IMP-04 (tier assignments).

**Tests required**:
- `tests/test_data/test_place_projection_orchestrator.py`:
  - Age-sex allocation for HIGH tier produces 18 age groups x 2 sexes per year.
  - Age-sex allocation for MODERATE tier produces 6 broad groups x 2 sexes per year.
  - LOWER tier produces total-only rows.
  - Allocated cohort populations sum to the place total (within tolerance).
  - Output parquet schema matches S06 Section 3 schemas exactly.
  - Metadata JSON has all required fields from S06 Section 4.
  - Summary CSV has all required columns from S06 Section 3.4.
  - Balance-of-county rows appear in summary CSV with `row_type = "balance_of_county"`.
  - Parquet file metadata (footer key-value pairs) contains all required keys from S06 Section 4.3.
  - File naming matches pattern: `nd_place_{place_fips}_projection_{start}_{end}_{scenario}.{ext}`.
  - Output directory structure matches S06 Section 2.1.

**Acceptance criteria**:
1. Place projections are county-constrained: `sum(place_pops) + balance <= county_total`.
2. All three tier-specific output schemas correct.
3. All S06 metadata fields present.
4. Output file naming convention matches S06 Section 2.2.
5. Tests pass.

**Agent suitability**: AI agent -- suitable. Output contract is fully specified in S06. Follows existing county output patterns.

---

### IMP-07: Configuration Additions

**Description**: Add place-projection configuration parameters to `config/projection_config.yaml` under a new `place_projections` section. All model parameters from S04 Section 8.5 must be configurable, not hard-coded.

**Files to modify**:
- `config/projection_config.yaml` -- add `place_projections` section.

**New configuration block**:
```yaml
place_projections:
  enabled: true
  crosswalk_path: "data/processed/geographic/place_county_crosswalk_2020.csv"
  historical_shares_path: "data/processed/place_shares_2000_2024.parquet"
  place_population_history_path: "data/processed/place_population_history_2000_2024.parquet"

  model:
    epsilon: 0.001                       # Logit clamping (S04 Section 2.3)
    lambda_decay: 0.9                    # WLS decay rate (S04 Section 3.2)
    history_start: 2000                  # Fitting window start
    history_end: 2024                    # Fitting window end
    reconciliation_flag_threshold: 0.05  # Balance QA flag threshold (S04 Section 6.3)

  tiers:
    high_threshold: 10000
    moderate_threshold: 2500
    lower_threshold: 500
    tier_boundary_margin: 0.05           # 5% of threshold for boundary flag

  backtest:
    primary_train: [2000, 2014]
    primary_test: [2015, 2024]
    secondary_train: [2000, 2019]
    secondary_test: [2020, 2024]

  output:
    base_year: 2025
    end_year: 2055
    key_years: [2025, 2030, 2035, 2040, 2045, 2050, 2055]
```

**Dependencies**: None (can be done in parallel with IMP-05).

**Tests required**:
- Add to existing config test suite or create `tests/test_config/test_place_projection_config.py`:
  - Config loads without error.
  - All place_projections fields are present.
  - Epsilon, lambda, thresholds have expected default values.
  - Config is consumed correctly by IMP-05 and IMP-06 modules.

**Acceptance criteria**:
1. All S04 Section 8.5 parameters are in config.
2. No model parameters are hard-coded in IMP-05 or IMP-06.
3. Config loads cleanly with `load_projection_config()`.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. Configuration structure follows existing patterns in `projection_config.yaml`.

---

## Phase 3: Backtesting

Execute the S05 backtest protocol to select the winning variant.

### IMP-08: Backtest Runner Script

**Description**: Build a standalone backtest runner that executes all four variants (A-I, A-II, B-I, B-II) of the 2x2 matrix against both the primary and secondary windows. For each variant: fit the model on training data, project shares for test years, apply the variant's constraint mechanism, multiply by actual county population, compute error metrics against actual place populations.

**Files to create**:
- `scripts/backtesting/run_place_backtest.py` -- runner script
- `cohort_projections/data/process/place_backtest.py` -- backtest computation module

**Functions to implement (in `place_backtest.py`)**:

1. `run_single_variant(variant_id, fitting_method, constraint_method, train_years, test_years, share_history, county_pop, config)` -- Execute one backtest variant for one window.
2. `compute_per_place_metrics(projected, actual)` -- Compute APE, PE, MAPE, MedAPE, ME, MaxAPE, AE_terminal per S05 Section 3.1.
3. `compute_tier_aggregates(place_metrics, tier_assignments)` -- Compute tier MedAPE, mean ME, 90th-percentile MAPE, tier MaxAPE per S05 Section 3.2.
4. `compute_variant_score(tier_aggregates)` -- Population-weighted MedAPE score per S04 Section 5.3.
5. `select_winner(variant_scores)` -- Select lowest-scoring variant with S04 Section 5.4 tie-breaking.

**Files to modify**:
- None

**Dependencies**: IMP-05 (share-trending module), IMP-03 (historical shares), IMP-04 (tier assignments).

**Tests required**:
- `tests/test_data/test_place_backtest.py`:
  - Per-place metric computation is correct for a known synthetic case.
  - Tier aggregation uses median (not mean) for MedAPE.
  - Population-weighted score formula matches S04 Section 5.3 exactly.
  - Tie-breaking prefers simpler specification (A over B, I over II).
  - Winner selection returns a single variant ID.
  - EXCLUDED tier places are backtested but not included in pass/fail evaluation.

**Acceptance criteria**:
1. All four variants execute without error on both windows.
2. Error metric definitions match S05 Section 3 exactly.
3. Variant selection follows S04 Section 5.3-5.5 exactly.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. Metric definitions and selection algorithm are fully specified in S04/S05.

---

### IMP-09: Backtest Execution and Variant Selection

**Description**: Execute the backtest runner on the actual ND place data. Produce the S05 reporting artifacts: summary table (Section 5.1), per-place detail table (Section 5.2), prediction interval calibration table (Section 5.4). Record the winning variant and its score.

**Files to create**:
- Output: `data/backtesting/place_backtest_results/` directory containing:
  - `backtest_summary_primary.csv` -- S05 Section 5.1 summary table (primary window)
  - `backtest_summary_secondary.csv` -- S05 Section 5.1 summary table (secondary window)
  - `backtest_per_place_detail.csv` -- S05 Section 5.2 per-place detail
  - `backtest_variant_scores.csv` -- All four variant scores
  - `backtest_prediction_intervals.csv` -- S05 Section 5.4 PI calibration table
  - `backtest_winner.json` -- Winning variant ID and parameters

**Dependencies**: IMP-08 (backtest module), IMP-01 through IMP-04 (all data artifacts).

**Tests required**:
- Validation checks embedded in the runner script:
  - All four variants produce results for all projected places.
  - Tier MedAPE values are computed and compared against S05 Section 4 thresholds.
  - Pass/fail determination matches S05 rules.

**Acceptance criteria**:
1. All four variants complete without error.
2. Winning variant is selected by the population-weighted MedAPE criterion.
3. All three projected tiers (HIGH, MODERATE, LOWER) pass their acceptance thresholds on the primary window.
4. Summary and detail tables match S05 Section 5 formats.
5. Backtest artifacts are saved for review.

**Agent suitability**: AI agent -- suitable for execution. **Human review required** before accepting the variant selection result. Human must review the per-place detail table and tier-level results.

---

### IMP-10: Outlier Narrative and Structural-Break Documentation

**Description**: For any place flagged in the per-place detail table (exceeding its tier's 90th-percentile MAPE ceiling), write a brief narrative identifying the likely cause (oil-boom surge, annexation, institutional population change, etc.) per S05 Section 5.3. Document any structural-break exclusions per S05 Section 6.3.

**Files to create**:
- `docs/reviews/pp3-backtest-outlier-narrative.md`

**Dependencies**: IMP-09 (backtest results with flagged places).

**Tests required**: None (narrative document).

**Acceptance criteria**:
1. Every flagged place has an identified cause.
2. Any structural-break exclusion has documented break year and cause.
3. If exclusions were applied, post-exclusion tier statistics are re-reported.

**Agent suitability**: **Human input required.** An AI agent can draft initial narratives based on known patterns (oil counties, university towns, reservation communities), but human review is required for structural-break exclusion decisions. This is a judgment call with publication implications.

---

## Phase 4: Production Pipeline Integration

### IMP-11: Pipeline Stage for Place Projections

**Description**: Create a new pipeline stage script that runs place projections for all active scenarios. Follow the existing pattern in `scripts/pipeline/02_run_projections.py` (CLI flags, logging, error handling, scenario iteration). The script reads the backtest winner from `backtest_winner.json` and applies that variant to the full 2000-2024 training window for production.

**Files to create**:
- `scripts/pipeline/02a_run_place_projections.py` -- pipeline stage (numbered to run after county projections but before export)

**Files to modify**:
- `scripts/pipeline/run_complete_pipeline.sh` -- add the new stage to the pipeline sequence.

**Dependencies**: IMP-06 (orchestrator), IMP-09 (winning variant), all data artifacts (IMP-01 through IMP-04).

**Tests required**:
- `tests/test_integration/test_place_pipeline_stage.py`:
  - Script runs without error with `--dry-run` flag.
  - Script produces output files in expected directories for each scenario.
  - Output directory structure matches S06 Section 2.1.
  - All three scenarios produce place output.

**Acceptance criteria**:
1. Pipeline stage runs end-to-end for all three scenarios.
2. Output files are written to `data/projections/{scenario}/place/`.
3. Script integrates into the existing pipeline sequence.
4. Tests pass.

**Agent suitability**: AI agent -- suitable. Follows established pipeline patterns.

---

### IMP-12: QA Artifact Generation

**Description**: Generate all QA artifacts specified in S06 Section 5 after each production run. These are written to `data/projections/{scenario}/place/qa/`.

**Files to create or modify**:
- Add QA generation to `cohort_projections/data/process/place_projection_orchestrator.py` (IMP-06), or create a separate `cohort_projections/output/place_qa.py` module.

**QA artifacts**:
1. `qa_tier_summary.csv` -- S06 Section 5.1
2. `qa_share_sum_validation.csv` -- S06 Section 5.2
3. `qa_outlier_flags.csv` -- S06 Section 5.3 (5 flag types: SHARE_REVERSAL, EXTREME_GROWTH, NEAR_ZERO_SHARE, SHARE_RESCALED, POPULATION_DECLINE_TO_NEAR_ZERO)
4. `qa_balance_of_county.csv` -- S06 Section 5.4

**Dependencies**: IMP-06 (orchestrator), IMP-11 (pipeline stage).

**Tests required**:
- `tests/test_data/test_place_qa_artifacts.py`:
  - Each QA file has the correct column schema per S06 Section 5.
  - Tier summary has exactly 3 rows (HIGH, MODERATE, LOWER).
  - Share-sum validation has one row per county-year.
  - Outlier flags use only the 5 defined flag types.
  - Balance-of-county table has one row per county-year.

**Acceptance criteria**:
1. All four QA artifacts are generated for every scenario.
2. Column schemas match S06 Section 5 exactly.
3. Tests pass.

**Agent suitability**: AI agent -- suitable. QA schemas are fully specified in S06.

---

### IMP-13: Consistency Constraint Enforcement

**Description**: Implement the S06 Section 6 hard constraints as pipeline assertions and soft constraints as QA flags. Hard constraints must cause the pipeline to fail loudly on violation.

**Hard constraints** (S06 Section 6.1):
1. Share bound: `0 <= place_share <= 1.0` for every place-year.
2. County share sum: `sum(place_shares) <= 1.0` for every county-year.
3. Place-county consistency: `sum(place_pops) <= county_total` for every county-year.
4. Non-negative population: No negative values.
5. Monotonic FIPS: Output universe matches crosswalk exactly.
6. Scenario ordering at state level: `restricted <= baseline <= high_growth`.

**Soft constraints** (S06 Section 6.2):
1. Balance-of-county non-negative.
2. Share stability (no share changes > 20pp over 30 years).
3. Tier-appropriate growth (growth rates within uncertainty bands).

**Files to create or modify**:
- Add validation to `cohort_projections/data/process/place_projection_orchestrator.py` or create `cohort_projections/output/place_validation.py`.

**Dependencies**: IMP-06 (orchestrator), IMP-12 (QA artifacts for soft constraints).

**Tests required**:
- `tests/test_data/test_place_consistency_constraints.py`:
  - Hard constraint violations raise exceptions (not silent failures).
  - Soft constraint violations produce QA flags without blocking.
  - Synthetic test case with deliberate violations triggers correct behavior.
  - Scenario ordering is preserved at state level.

**Acceptance criteria**:
1. All 6 hard constraints are enforced with loud failures.
2. All 3 soft constraints produce QA flags.
3. Tests pass.

**Agent suitability**: AI agent -- suitable. Constraint specifications are fully defined in S06.

---

## Phase 5: Export and Workbook Integration

### IMP-14: Place Workbook Builder

**Description**: Build the new standalone place workbook per S06 Section 7.2. Structure: Table of Contents, 9 HIGH-tier sheets (5-year age groups x sex), 9 MODERATE-tier sheets (broad age groups x sex), 1 combined LOWER-tier sheet (total population with uncertainty caveat header), and a Methodology sheet.

**Files to create**:
- `scripts/exports/build_place_workbook.py`

**Files to modify**:
- `scripts/exports/_methodology.py` -- add place-specific methodology line per S06 Section 7.4.

**Dependencies**: IMP-11 (production output files exist), IMP-06 (all output schemas).

**Tests required**:
- `tests/test_output/test_place_workbook.py`:
  - Workbook contains expected number of sheets (1 TOC + 9 HIGH + 9 MODERATE + 1 LOWER + 1 Methodology = 21 sheets).
  - HIGH-tier sheets have 18 age group rows x 2 sex columns at each key year.
  - MODERATE-tier sheets have 6 broad age group rows x 2 sex columns at each key year.
  - LOWER-tier sheet has 72 place rows with total population at 7 key years.
  - LOWER-tier sheet has a prominent uncertainty caveat header.
  - TOC has hyperlinks to all place sheets.
  - Methodology sheet contains place-specific methodology text.
  - File naming matches pattern: `nd_projections_{scenario}_places_{datestamp}.xlsx`.

**Acceptance criteria**:
1. Workbook structure matches S06 Section 7.2.
2. All 90 projected places appear in the workbook.
3. Key years are 2025, 2030, 2035, 2040, 2045, 2050, 2055.
4. LOWER-tier caveat header is prominent.
5. Tests pass.

**Agent suitability**: AI agent -- suitable. Follows existing workbook builder patterns in `build_detail_workbooks.py` and `build_provisional_workbook.py`.

---

### IMP-15: Provisional Workbook Update

**Description**: Add a `Places -- {scenario_short}` sheet to the existing provisional workbook. The sheet contains a summary table with place name, county, tier, key-year populations, and growth rate. Mirrors the existing `Counties -- {scenario_short}` sheets. Include balance-of-county rows.

**Files to modify**:
- `scripts/exports/build_provisional_workbook.py` -- add place summary sheet generation.

**Dependencies**: IMP-11 (production output files for `places_summary.csv`).

**Tests required**:
- `tests/test_output/test_provisional_workbook_places.py`:
  - New `Places -- Baseline` (etc.) sheets exist in the workbook.
  - Sheet contains 90 place rows + balance-of-county rows.
  - Columns include place_fips, name, county, tier, key-year populations, growth_rate.
  - No structural changes to existing county/state sheets.

**Acceptance criteria**:
1. New Places sheets appear for each active scenario.
2. Existing sheets are not modified.
3. Tests pass.

**Agent suitability**: AI agent -- suitable. Follows existing provisional workbook sheet patterns.

---

### IMP-16: Export Pipeline `--places` Flag Wiring

**Description**: Wire the `--places` flag in `scripts/pipeline/03_export_results.py` to trigger place-level output generation (summary CSVs, place workbook). The flag already exists in the CLI argument parser; implementation must connect it to the new place output directory and workbook builder.

**Files to modify**:
- `scripts/pipeline/03_export_results.py` -- implement place-level export logic when `--places` is passed.

**Dependencies**: IMP-14 (place workbook builder), IMP-11 (production output files).

**Tests required**:
- `tests/test_integration/test_export_places.py`:
  - `--places` flag triggers place-level export.
  - Place summary CSV is generated.
  - Place workbook is generated.
  - `--all` includes places.
  - `--dry-run` with `--places` does not create files.

**Acceptance criteria**:
1. `--places` flag works end-to-end.
2. Existing county/state export behavior unchanged.
3. Tests pass.

**Agent suitability**: AI agent -- suitable. Follows existing export patterns.

---

### IMP-17: Methodology Text Update

**Description**: Add the place-specific methodology text to `scripts/exports/_methodology.py` per S06 Section 7.4. The text describes the share-of-county trending method, confidence tiers, and county constraint.

**Files to modify**:
- `scripts/exports/_methodology.py` -- add place methodology constant.
- `docs/methodology.md` -- add place projection methodology section.

**Dependencies**: IMP-09 (winning variant must be known to describe the exact method used).

**Tests required**:
- Verify methodology text includes: method name ("share-of-county trending"), ADR-033 reference, tier definitions, county constraint statement, winning variant description.

**Acceptance criteria**:
1. Place methodology text is accurate and complete.
2. References ADR-033 and the winning backtest variant.
3. `docs/methodology.md` updated.

**Agent suitability**: AI agent -- suitable for drafting. **Human review required** for final methodology text accuracy and tone.

---

## Phase 6: Validation and Documentation

### IMP-18: Integration Tests

**Description**: Write integration tests that run the full place projection pipeline from data assembly through output generation on a small subset of counties (e.g., Cass County only or a 3-county subset). Validate end-to-end correctness.

**Files to create**:
- `tests/test_integration/test_place_projection_integration.py`

**Dependencies**: All IMP-01 through IMP-16.

**Tests required**:
- Full pipeline runs without error on test subset.
- Output files exist in expected locations.
- Place-county consistency holds (sum of places <= county).
- All hard constraints pass.
- QA artifacts are generated.
- Place workbook is generated (if openpyxl available).

**Acceptance criteria**:
1. Integration test passes end-to-end.
2. All consistency constraints validated.
3. Test is fast enough for CI (target: <60s).

**Agent suitability**: AI agent -- suitable.

---

### IMP-19: End-to-End Validation (Full Pipeline Run)

**Description**: Run the complete pipeline for all three scenarios on the full ND place universe. Validate all outputs. This is the final acceptance test before publication.

**Files to create**:
- `docs/reviews/pp3-end-to-end-validation.md` -- validation results document.

**Dependencies**: All IMP-01 through IMP-18.

**Validation checks**:
1. All 90 projected places produce output for all three scenarios.
2. All hard constraints pass (S06 Section 6.1).
3. QA artifacts are generated and reviewed.
4. Scenario ordering holds at state level (restricted <= baseline <= high_growth).
5. Place workbooks generated for all three scenarios.
6. `places_summary.csv` contains 90 place rows + balance-of-county rows per scenario.
7. `places_metadata.json` contains correct counts.
8. Compare place-level totals against county totals (sum of places <= county for all county-years).

**Acceptance criteria**:
1. Zero hard constraint violations.
2. Soft constraint flags are reviewed and documented.
3. Validation document is complete.

**Agent suitability**: AI agent -- suitable for execution. **Human sign-off required** on validation results before publication.

---

### IMP-20: ADR-033 Status Update

**Description**: Update ADR-033 status from "Deferred" to "Accepted/Implemented". Add Implementation Results section documenting the winning backtest variant, acceptance metrics, and publication readiness.

**Files to modify**:
- `docs/governance/adrs/033-city-level-projection-methodology.md` -- update status and add results.

**Dependencies**: IMP-19 (validation complete).

**Tests required**: None (documentation).

**Acceptance criteria**:
1. ADR status updated to "Accepted" with implementation date.
2. Implementation Results section includes: winning variant, tier MedAPE results, prediction interval estimates, any structural-break exclusions.
3. Revision history updated.

**Agent suitability**: AI agent -- suitable for drafting. **Human review required** for final ADR status change.

---

### IMP-21: Methodology Documentation Update

**Description**: Update `docs/methodology.md` with the complete place projection methodology section. This is the publication-facing methodology description.

**Files to modify**:
- `docs/methodology.md`

**Content to add**:
- Share-of-county trending method description.
- Logit-linear specification.
- Winning variant (OLS vs. WLS, proportional vs. cap-and-redistribute) and why it was selected.
- Confidence tier definitions and output granularity.
- Balance-of-county treatment.
- Backtest results summary (tier MedAPE, prediction intervals).
- County constraint and reconciliation.
- Caveats and limitations.

**Dependencies**: IMP-09 (backtest results), IMP-19 (validation complete).

**Tests required**: None (documentation).

**Acceptance criteria**:
1. Methodology section is complete, accurate, and consistent with S04/S05/S06.
2. Winning variant is correctly described.
3. Caveats are appropriately stated.

**Agent suitability**: AI agent -- suitable for drafting. **Human review required** for publication-facing methodology accuracy and tone.

---

### IMP-22: DEVELOPMENT_TRACKER.md Update

**Description**: Update the development tracker to reflect PP3-S08 completion, mark PP-003 status as "in_progress" (or "complete" after IMP-19 sign-off), and move Phase 2+ expansion items to the Deferred section.

**Files to modify**:
- `DEVELOPMENT_TRACKER.md`

**Dependencies**: IMP-19 (validation complete).

**Tests required**: None (documentation).

**Acceptance criteria**:
1. PP3-S08 status updated from "pending" to "completed" with date.
2. PP-003 status reflects current state.

**Agent suitability**: AI agent -- suitable.

---

## Dependency Graph

```
Phase 1: Data Assembly
    IMP-01 (crosswalk)
        |
        v
    IMP-02 (place pop history) -----> IMP-04 (tier assignments)
        |                                 |
        v                                 |
    IMP-03 (historical shares)            |
        |                                 |
        +------ Phase 1 complete ---------+

Phase 2: Model Implementation (depends on Phase 1 complete)
    IMP-07 (config) ----+
                        |
    IMP-05 (core model)-+----> IMP-06 (orchestrator)
                        |
                        +----> IMP-08 (backtest module)

Phase 3: Backtesting (depends on IMP-05, IMP-08, Phase 1)
    IMP-08 -----> IMP-09 (backtest execution)
                      |
                      v
                  IMP-10 (outlier narrative) [HUMAN]

Phase 4: Pipeline Integration (depends on IMP-06, IMP-09)
    IMP-11 (pipeline stage) -----> IMP-12 (QA artifacts)
                                       |
                            IMP-13 (constraints)

Phase 5: Export (depends on IMP-11)
    IMP-14 (place workbook)
    IMP-15 (provisional workbook update)
    IMP-16 (export flag wiring)
    IMP-17 (methodology text) [needs IMP-09 for variant name]

Phase 6: Validation and Documentation (depends on all above)
    IMP-18 (integration tests) -----> IMP-19 (E2E validation) [HUMAN sign-off]
                                          |
                                          v
                                      IMP-20 (ADR update) [HUMAN review]
                                      IMP-21 (methodology docs) [HUMAN review]
                                      IMP-22 (tracker update)
```

**Critical path**: IMP-01 -> IMP-02 -> IMP-03 -> IMP-05 -> IMP-08 -> IMP-09 -> IMP-11 -> IMP-19

**Parallelizable within phases**:
- Phase 1: IMP-04 can run in parallel with IMP-03 (both depend on IMP-02).
- Phase 2: IMP-07 can run in parallel with IMP-05.
- Phase 4: IMP-12 and IMP-13 can run in parallel.
- Phase 5: IMP-14, IMP-15, IMP-16 can run in parallel.
- Phase 6: IMP-20, IMP-21, IMP-22 can run in parallel after IMP-19.

---

## Agent Suitability Assessment

| Task | Agent Suitable | Human Review | Human Execution |
|------|---------------|-------------|-----------------|
| IMP-01: Build crosswalk | Yes | Review output | -- |
| IMP-02: Assemble history | Yes | Confirm data paths | -- |
| IMP-03: Compute shares | Yes | -- | -- |
| IMP-04: Assign tiers | Yes | -- | -- |
| IMP-05: Core model | Yes | -- | -- |
| IMP-06: Orchestrator | Yes | -- | -- |
| IMP-07: Config additions | Yes | -- | -- |
| IMP-08: Backtest module | Yes | -- | -- |
| IMP-09: Backtest execution | Yes | Review results, approve variant | -- |
| IMP-10: Outlier narrative | Draft by agent | **Required**: exclusion decisions | Structural-break judgment |
| IMP-11: Pipeline stage | Yes | -- | -- |
| IMP-12: QA artifacts | Yes | -- | -- |
| IMP-13: Constraints | Yes | -- | -- |
| IMP-14: Place workbook | Yes | Review formatting | -- |
| IMP-15: Provisional update | Yes | -- | -- |
| IMP-16: Export flag wiring | Yes | -- | -- |
| IMP-17: Methodology text | Draft by agent | **Required**: accuracy/tone | -- |
| IMP-18: Integration tests | Yes | -- | -- |
| IMP-19: E2E validation | Yes | **Required**: sign-off | -- |
| IMP-20: ADR update | Draft by agent | **Required**: status change approval | -- |
| IMP-21: Methodology docs | Draft by agent | **Required**: publication review | -- |
| IMP-22: Tracker update | Yes | -- | -- |

**Summary**: 17 of 22 tasks are fully agent-suitable. 5 tasks require human review or judgment (IMP-09 results review, IMP-10 exclusion decisions, IMP-17/IMP-21 methodology review, IMP-19 validation sign-off, IMP-20 ADR approval).

---

## Estimated Task Count

| Phase | Tasks | Agent-Only | Human Review Required |
|-------|-------|-----------|----------------------|
| Phase 1: Data Assembly | 4 | 4 | 0 |
| Phase 2: Model Implementation | 3 | 3 | 0 |
| Phase 3: Backtesting | 3 | 1 | 2 |
| Phase 4: Pipeline Integration | 3 | 3 | 0 |
| Phase 5: Export/Workbook | 4 | 3 | 1 |
| Phase 6: Validation/Docs | 5 | 2 | 3 |
| **Total** | **22** | **16** | **6** |

---

## Scoping Reference Documents

| Document | Path | Content |
|----------|------|---------|
| S01 Scope Envelope | `DEVELOPMENT_TRACKER.md` (PP-003 section) | Place universe, tiers, horizon |
| S02 Data Readiness | `docs/reviews/2026-02-28-place-data-readiness-note.md` | PEP vintage files, coverage gaps |
| S03 Mapping Strategy | `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md` | Crosswalk spec, assignment rules |
| S04 Modeling Spec | `docs/reviews/2026-02-28-pp3-s04-modeling-spec.md` | Logit-linear model, 2x2 matrix, balance-of-county |
| S05 Backtesting Design | `docs/reviews/2026-02-28-pp3-s05-backtesting-design.md` | Windows, metrics, thresholds |
| S06 Output Contract | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` | Schemas, files, workbook structure |
| S07 Approval Gate | `docs/reviews/2026-02-28-pp3-s07-approval-gate.md` | Go decision, all decisions summary |
| ADR-033 | `docs/governance/adrs/033-city-level-projection-methodology.md` | City-level projection methodology |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
