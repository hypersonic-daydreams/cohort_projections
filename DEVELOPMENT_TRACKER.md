# Development Tracker

Canonical, current-state tracker for the North Dakota cohort projections repository.

**Last Updated:** 2026-03-01  
**Projection Horizon:** 2025-2055  
**Status:** Publication preparation with repo-hygiene closeout complete; this file is the single source of truth for remaining projection-development work.

## Purpose

Use this file for active status only. Historical session detail is archived to:

- `docs/archive/DEVELOPMENT_TRACKER_2026-02-26.md`

## Source Of Truth Rule

1. When asked "what work remains for population projections," answer from this file first.
2. Treat these sections as canonical: `Projection Development Backlog (Canonical)`, `Documentation Consistency Queue`, and `Deferred / Later Work`.
3. Do not treat archived docs/reviews as open work unless the item is explicitly listed here.
4. If new open work is discovered, add it here with a source link before ending the session.

## Current Snapshot

| Area | Status | Notes |
|------|--------|-------|
| Core projection engine | complete | County/state production path is operational for all scenarios. |
| Data processing pipeline | complete | Inputs and transforms are in place; no active blocker. |
| Documentation alignment | complete | B01 documentation harmonization complete. |
| Repo-hygiene program | complete | B00-B06 implemented; full adjudicated replay `27/27` passing; RB-003 and RB-004 remediated and closed. |
| Test health baseline | stable | Latest recorded baseline: `1247 passed, 5 skipped` (excl. integration). ADR-056 accepted; PP4-01 through PP4-05 closed — 127 new tests added covering GQ separation, pipeline orchestrators, and base population loaders. |
| Claim replay health | stable | `27/27` adjudicated claims passing (latest full replay: 2026-02-27T18:14:03Z). |

## Projection Development Backlog (Canonical)

| ID | Work Item | Status | Next Milestone | Source |
|------|------|------|------|------|
| PP-001 | Publication-facing output QA and dissemination packaging | active | Collect stakeholder sign-off on 2026-02-28 package set and keep release checklist current for final publication handoff | `docs/reviews/2026-02-28-publication-output-qa-packaging-checklist.md`, `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md` |
| PP-002 | Non-regression validation cadence during publication work | active | Run and record the next cadence cycle after the next material projection/config change (same gate set) | `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`, `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md` |
| PP-003 | City/place projection workstream reactivation (ADR-033) | active | IMP-13A reconciliation-magnitude QA follow-up completed (2026-03-01); next integration milestones are IMP-14 place workbook builder and IMP-15 provisional workbook `Places` sheet wiring | `docs/plans/pp3-s08-implementation-kickoff.md`, `docs/reviews/2026-02-28-pp3-s07-approval-gate.md`, `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md`, `docs/reviews/pp3-backtest-outlier-narrative.md`, `docs/reviews/2026-03-01-pp3-imp11-pipeline-stage-results.md`, `docs/reviews/2026-03-01-pp3-imp12-imp13-results.md`, `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md`, `docs/reviews/2026-03-01-pp3-imp13a-results.md` |
| PP-004 | Test coverage gap closure (ADR-056) | active | Close priority coverage gaps identified in ADR-056 Decision 6; align with PP-002 periodic review cadence | `docs/governance/adrs/056-testing-strategy-maturation.md`, `docs/guides/test-maintenance-practices.md` |

## PP-003 Phase 1 Scoping Checklist (Canonical)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| PP3-S01 | Scope envelope | completed_2026-02-28 | Confirm place universe, projection horizon, and output granularity tiers (HIGH/MODERATE/LOWER/EXCLUDED) for the first release | Scoped statement below (this file) + ADR-033 cross-reference |
| PP3-S02 | Historical place data readiness | completed_2026-02-28 | Verify place-level annual history coverage needed for backtesting (target window: 2000-2024) and identify any gaps requiring imputation or exclusion rules | `docs/reviews/2026-02-28-place-data-readiness-note.md` |
| PP3-S03 | Place-to-county boundary mapping strategy | completed_2026-02-28 | Define authoritative mapping and handling rules for boundary/vintage changes so place shares can be compared consistently over time | `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md` |
| PP3-S04 | Modeling spec for share-trending | completed_2026-02-28 | Select candidate trend specifications and constraint approach (share sum <= 100%, balance-of-county handling) for Phase 1 testing | `docs/reviews/2026-02-28-pp3-s04-modeling-spec.md` |
| PP3-S05 | Backtesting design and metrics | completed_2026-02-28 | Lock backtest design (train/test windows) and acceptance metrics by confidence tier | `docs/reviews/2026-02-28-pp3-s05-backtesting-design.md` |
| PP3-S06 | Output contract | completed_2026-02-28 | Define Phase 1 deliverables: projection files, metadata fields, QA summary tables, and workbook impacts | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` |
| PP3-S07 | Approval gate (required) | approved_2026-02-28 | Human approval recorded to proceed from scoping to implementation (Tier 3 methodology-change control) | `docs/reviews/2026-02-28-pp3-s07-approval-gate.md` |
| PP3-S08 | Implementation kickoff packet | completed_2026-02-28 | Publish execution-ready task list (files, tests, validation gates, and ADR touchpoints) | `docs/plans/pp3-s08-implementation-kickoff.md` |

**Go/No-Go Rule:** PP-003 implementation starts only after `PP3-S01` through `PP3-S07` are complete and explicitly marked `go`.

## PP-003 Phase 1 Implementation Checklist (IMP-01 to IMP-04)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| IMP-01 | Build place-county crosswalk | completed_2026-02-28 | Production run complete: primary crosswalk `357` rows (`355` active + `2` historical_only), multicounty detail `14` rows (`7` places), major-place spot checks PASS | `scripts/data/build_place_county_crosswalk.py`, `data/processed/geographic/place_county_crosswalk_2020.csv`, `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv`, `tests/test_data/test_place_county_crosswalk.py` |
| IMP-02 | Assemble place population history (2000-2024) | completed_2026-02-28 | Production run complete: `8,915` rows, year coverage `2000-2024` contiguous, no null populations, county join complete | `scripts/data/assemble_place_population_history.py`, `data/processed/place_population_history_2000_2024.parquet`, `tests/test_data/test_place_population_history.py` |
| IMP-03 | Compute historical place shares | completed_2026-02-28 | Production run complete: `10,240` rows (`8,915` place + `1,325` balance), county-year share identity holds (`max abs error = 0.0`), epsilon clamping bounds enforced | `cohort_projections/data/process/place_shares.py`, `data/processed/place_shares_2000_2024.parquet`, `tests/test_data/test_place_shares.py` |
| IMP-04 | Assign confidence tiers | completed_2026-02-28 | Tier enrichment written to crosswalk; active-place counts exactly `HIGH=9`, `MODERATE=9`, `LOWER=72`, `EXCLUDED=265` | `scripts/data/build_place_county_crosswalk.py`, `scripts/data/assemble_place_population_history.py`, `data/processed/geographic/place_county_crosswalk_2020.csv`, `tests/test_data/test_place_county_crosswalk.py`, `tests/test_data/test_place_population_history.py` |

**Execution note:** Runtime artifact generation is complete (2026-02-28). Human plausibility review remains recommended for multicounty primary assignments before publication.

## PP-003 Phase 2 Implementation Checklist (IMP-05 onward)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| IMP-05 | Core share-trending module | completed_2026-02-28 | Implemented S04 logit-linear engine (`OLS`/`WLS`, midpoint centering, proportional + cap-and-redistribute constraints, balance reconciliation, county orchestration) with required edge-case handling and full kickoff test matrix coverage | `cohort_projections/data/process/place_share_trending.py`, `cohort_projections/data/process/__init__.py`, `tests/test_data/test_place_share_trending.py` |
| IMP-06 | Place projection orchestrator | completed_2026-02-28 | Implemented county-driven place projection orchestration (`run_place_projections`), tier-specific age/sex allocation (`HIGH` 18x2, `MODERATE` 6x2, `LOWER` total-only), and S06 output writing contract (per-place parquet/metadata/summary + `places_summary.csv` + `places_metadata.json` with required parquet footer metadata keys) with synthetic end-to-end contract tests | `cohort_projections/data/process/place_projection_orchestrator.py`, `cohort_projections/data/process/__init__.py`, `tests/test_data/test_place_projection_orchestrator.py` |
| IMP-07 | Configuration additions | completed_2026-02-28 | Added `place_projections` block to canonical config (paths, model, tiers, backtest windows, output years/key years) and added tests validating defaults plus configuration consumption by both IMP-05 (`trend_all_places_in_county`) and IMP-06 (`run_place_projections`) | `config/projection_config.yaml`, `tests/test_config/test_place_projection_config.py` |
| IMP-08 | Backtest runner script + module | completed_2026-02-28 | Implemented backtest computation module (`run_single_variant`, per-place metrics, tier aggregates, weighted scoring, tie-break winner selection) plus standalone runner executing A-I/A-II/B-I/B-II across windows with output artifacts and synthetic matrix coverage tests including EXCLUDED informational-tier handling | `cohort_projections/data/process/place_backtest.py`, `scripts/backtesting/run_place_backtest.py`, `tests/test_data/test_place_backtest.py`, `cohort_projections/data/process/__init__.py` |
| IMP-09 | Backtest execution + variant selection | completed_2026-02-28 | Ran ND production backtest matrix and produced S05 artifacts; winner selected `B-II` (`wls` + `cap_and_redistribute`, score `3.0757840903894844`). Primary scored tiers all PASS; secondary window HIGH tier flagged FAIL (diagnostic) | `scripts/backtesting/run_place_backtest.py`, `data/backtesting/place_backtest_results/backtest_summary_primary.csv`, `data/backtesting/place_backtest_results/backtest_summary_secondary.csv`, `data/backtesting/place_backtest_results/backtest_per_place_detail.csv`, `data/backtesting/place_backtest_results/backtest_variant_scores.csv`, `data/backtesting/place_backtest_results/backtest_prediction_intervals.csv`, `data/backtesting/place_backtest_results/backtest_winner.json`, `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md` |
| IMP-10 | Outlier narrative + structural-break documentation | completed_2026-03-01 | Human review completed: accepted winner adoption, approved Horace/Williston structural-break interpretations (annexation and oil-economy fluctuation), approved progression to pipeline integration with no diagnostic exclusions applied | `docs/reviews/pp3-backtest-outlier-narrative.md` |
| IMP-11 | Place projection pipeline stage integration | completed_2026-03-01 | Added `02a_run_place_projections.py` stage, wired into `run_complete_pipeline.sh`, validated via integration tests and executed full stage run across active scenarios (`baseline`, `restricted_growth`, `high_growth`) with per-scenario outputs produced | `scripts/pipeline/02a_run_place_projections.py`, `scripts/pipeline/run_complete_pipeline.sh`, `tests/test_integration/test_place_pipeline_stage.py`, `docs/reviews/2026-03-01-pp3-imp11-pipeline-stage-results.md` |
| IMP-12 | QA artifact generation | completed_2026-03-01 | Implemented S06 Section 5 QA artifact pipeline at `data/projections/{scenario}/place/qa/` (`qa_tier_summary.csv`, `qa_share_sum_validation.csv`, `qa_outlier_flags.csv`, `qa_balance_of_county.csv`) with schema checks and outlier flag-type validation; human review gate approved with notes | `cohort_projections/data/process/place_projection_orchestrator.py`, `tests/test_data/test_place_qa_artifacts.py`, `docs/reviews/2026-03-01-pp3-imp12-imp13-results.md`, `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md` |
| IMP-13 | Consistency constraint enforcement | completed_2026-03-01 | Enforced S06 hard constraints (share bounds, county share sum, place<=county totals, non-negative populations, exact output universe, state-level scenario ordering when all scenario county outputs are available) and soft QA signaling (balance warning, share-stability flags, tier-band extreme growth flags); human review gate approved with notes | `cohort_projections/data/process/place_projection_orchestrator.py`, `tests/test_data/test_place_consistency_constraints.py`, `docs/reviews/2026-03-01-pp3-imp12-imp13-results.md`, `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md` |
| IMP-13A | Reconciliation magnitude QA (Gate 4) | completed_2026-03-01 | Added `qa_reconciliation_magnitude.csv` county-year artifact (`total_before_adjustment`, `total_after_adjustment`, `reconciliation_adjustment`, threshold flag fields), integrated Gate 4 magnitude distribution + top-adjustment table in human review package, and added invariant-style tests | `cohort_projections/data/process/place_projection_orchestrator.py`, `scripts/reviews/build_pp3_human_review_package.py`, `tests/test_data/test_place_qa_artifacts.py`, `docs/reviews/2026-03-01-pp3-imp13a-results.md`, `docs/reviews/2026-03-01-pp3-gate4-rescaling-review-note.md` |

### IMP-05 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_share_trending.py` -> `14 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_share_trending.py tests/test_data/test_place_share_trending.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_share_trending.py` -> `Success: no issues found in 1 source file`.

### IMP-06 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_projection_orchestrator.py` -> `4 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_projection_orchestrator.py tests/test_data/test_place_projection_orchestrator.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py` -> `Success: no issues found in 1 source file`.

### IMP-07 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_config/test_place_projection_config.py tests/test_data/test_place_projection_orchestrator.py` -> `7 passed`.
- `source .venv/bin/activate && ruff check tests/test_config/test_place_projection_config.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py cohort_projections/data/process/place_share_trending.py` -> `Success: no issues found in 2 source files`.

### IMP-08 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_backtest.py tests/test_config/test_place_projection_config.py tests/test_data/test_place_projection_orchestrator.py` -> `15 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_backtest.py scripts/backtesting/run_place_backtest.py tests/test_data/test_place_backtest.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_backtest.py` -> `Success: no issues found in 1 source file`.

### IMP-09 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_backtest.py` -> `8 passed`.
- `source .venv/bin/activate && ruff check scripts/backtesting/run_place_backtest.py cohort_projections/data/process/place_backtest.py tests/test_data/test_place_backtest.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy scripts/backtesting/run_place_backtest.py cohort_projections/data/process/place_backtest.py` -> `Success: no issues found in 2 source files`.
- `source .venv/bin/activate && python scripts/backtesting/run_place_backtest.py` -> `Backtest complete: 8 score rows, winner B-II (score=3.0758)`.
- Output artifacts written to `data/backtesting/place_backtest_results/` with required IMP-09 files.
- Primary acceptance gate (S05): `HIGH PASS`, `MODERATE PASS`, `LOWER PASS` (`all_scored_tiers_pass_primary=true` in `backtest_winner.json`).

### IMP-10 Verification Evidence (2026-03-01)

- Human review approvals recorded in `docs/reviews/pp3-backtest-outlier-narrative.md`:
  - Winner `B-II` approved for production integration.
  - Horace and Williston structural-break interpretations approved (annexation / oil-economy fluctuation).
  - No structural-break exclusions applied for diagnostic reporting.
- IMP-09 review checklist updated to completed in `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md`.

### IMP-11 Verification Evidence (2026-03-01)

- `source .venv/bin/activate && ruff check scripts/pipeline/02a_run_place_projections.py tests/test_integration/test_place_pipeline_stage.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy scripts/pipeline/02a_run_place_projections.py` -> `Success: no issues found in 1 source file`.
- `source .venv/bin/activate && pytest tests/test_integration/test_place_pipeline_stage.py` -> `3 passed`.
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py --dry-run` -> dependency checks passed for winner payload + all active scenario county inputs.
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py` -> successful runs for `baseline`, `restricted_growth`, `high_growth`; each wrote place outputs with `90` places and `46` county-balance rows.

### IMP-12 / IMP-13 Verification Evidence (2026-03-01)

- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_projection_orchestrator.py cohort_projections/data/process/__init__.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> `Success: no issues found in 3 source files`.
- `source .venv/bin/activate && pytest tests/test_data/test_place_projection_orchestrator.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> `8 passed`.
- `source .venv/bin/activate && pytest tests/test_integration/test_place_pipeline_stage.py` -> `3 passed`.
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py --scenarios baseline` -> successful baseline stage execution with QA artifacts emitted and hard-constraint checks passing (`90` places, `46` county balances, `31` QA outlier flags).
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py` -> successful full stage execution for `baseline`, `restricted_growth`, and `high_growth`; each scenario wrote place outputs (`90` places, `46` county balances) plus QA artifacts and passed hard constraints.
- Human gate approvals recorded in `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md`: overall **Approved with notes**; Gates 1/2/3/7/8 approved; Gates 4/5/6 approved with notes.

### IMP-13A Verification Evidence (2026-03-01)

- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_projection_orchestrator.py scripts/reviews/build_pp3_human_review_package.py tests/test_data/test_place_qa_artifacts.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py scripts/reviews/build_pp3_human_review_package.py tests/test_data/test_place_qa_artifacts.py` -> `Success: no issues found in 3 source files`.
- `source .venv/bin/activate && pytest tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py tests/test_data/test_place_projection_orchestrator.py tests/test_integration/test_place_pipeline_stage.py` -> `11 passed`.
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py` -> successful full stage execution for `baseline`, `restricted_growth`, and `high_growth` with `qa_reconciliation_magnitude.csv` emitted per scenario (`1,426` rows each).
- `source .venv/bin/activate && python scripts/reviews/build_pp3_human_review_package.py --refresh-latest` -> review package regenerated with Gate 4 reconciliation-magnitude diagnostics (`distribution` + `top county-years by adjustment` table).

## PP-003 Scope Envelope (S01 Result)

Scope statement confirmed on 2026-02-28 (ADR-033 alignment):

- **Projection universe (first release):** 355 active ND incorporated places (2024 PEP place file in `data/raw/geographic/nd_places.csv`), with historical-only handling for dissolved places discovered in S02.
- **Projection horizon:** 2025-2055 annual, aligned to county/state production horizon.
- **Output granularity tiers (ADR-033 thresholds, based on 2024 population):**
  - `HIGH` (>10,000): 9 places
  - `MODERATE` (2,500-10,000): 9 places
  - `LOWER` (500-2,500): 72 places
  - `EXCLUDED` (<500): 265 places
- **Constraint frame:** place outputs remain county-constrained with explicit balance-of-county remainder (full constraint specification continues in `PP3-S04`).

## PP-004 Test Coverage Gap Closure Checklist (Canonical)

Priority coverage gaps from ADR-056 Decision 6 and `docs/guides/test-maintenance-practices.md`. Items are ranked by risk to projection accuracy.

| Step ID | Task | Status | Definition of Done | Reference |
|------|------|------|------|------|
| PP4-01 | GQ separation function tests (ADR-055) | completed_2026-02-28 | 42 tests in `test_gq_separation.py` covering all 5 GQ functions; `base_population_loader.py` 36.4%→68.2%, `residual_migration.py` 54.0%→85.4% | `tests/test_data/test_gq_separation.py` |
| PP4-02 | Pipeline orchestrator tests | completed_2026-02-28 | 38 tests in `test_pipeline_orchestrators.py` covering `run_residual_migration_pipeline` and `run_convergence_pipeline`; `convergence_interpolation.py` 44.3%→88.1% | `tests/test_data/test_pipeline_orchestrators.py` |
| PP4-03 | Base population loader tests | completed_2026-02-28 | 47 tests in `test_base_population_loader.py` covering all 3 loader entry points plus GQ separation; ADR-054 invariant verified | `tests/test_data/test_base_population_loader.py` |
| PP4-04 | Pre-commit hook blind spot fix | completed_2026-02-28 | `files` pattern updated to `^(cohort_projections|tests)/.*\.py$` so test-only changes trigger pytest | `.pre-commit-config.yaml` |
| PP4-05 | Silent skip audit | completed_2026-02-28 | Skip count: 5 (1 PyMC + 4 upstream bug); 222 silent-skip-capable tests all running; all IMPORTS_AVAILABLE gates resolve; baseline established | See PP4-05 audit results below |
| PP4-06 | First PP-002-aligned periodic coverage review | completed_2026-02-28 | All 5 checklist items PASS: 1,247 tests passed (5 skipped, unchanged); coverage stable at baselines; 194 IMPORTS_AVAILABLE gates unchanged; 137 integration tests pass; skip count steady at 5 | `docs/reviews/2026-02-28-pp4-06-periodic-coverage-review.md` |

**Approach:** Gaps are closed incrementally as modules are touched (ADR-056 Decision 6), not as a standalone sprint. PP4-01 through PP4-06 are complete. The PP-004 workstream is closed.

### PP4-05 Audit Results (2026-02-28)

- **Skip count baseline: 5** (1 PyMC slow test + 4 upstream `sigma_u.tolist()` bug in `test_bayesian_var.py`)
- **Alert threshold:** If skip count rises above 5, investigate immediately
- **Silent-skip-capable tests: 222** (194 IMPORTS_AVAILABLE-gated + 28 data-file-gated) — all currently running, none skipping
- **All IMPORTS_AVAILABLE gates resolve to True** in dev environment (matplotlib, seaborn, openpyxl, geopandas, pymc all installed)
- **Coverage deltas from PP4-01/02/03:** `base_population_loader.py` 36.4%→68.2%, `residual_migration.py` 54.0%→85.4%, `convergence_interpolation.py` 44.3%→88.1%

## Documentation Consistency Queue

| ID | Item | Status | Action Needed | Source |
|------|------|------|------|------|
| DOC-001 | ADR-033 deferral rationale still cites unresolved state-county discrepancy as a blocker | closed_2026-02-28 | Completed: ADR-033 deferral section and revision history now reflect ADR-054 accepted/implemented state and sequencing-based deferral rationale | `docs/governance/adrs/033-city-level-projection-methodology.md`, `docs/governance/adrs/054-state-county-aggregation-reconciliation.md` |
| DOC-002 | 2026-02-23 projection output review "minor issues" may be stale | closed_2026-02-28 | Completed: revalidation recorded in-place; both minor issues are now explicitly marked resolved/stale | `docs/reviews/2026-02-23-projection-output-review.md` |

## Repo-Hygiene Batch State

| Batch | Name | Status | Notes |
|------|------|--------|-------|
| B00 | baseline_metrics_and_safety_harness | complete | Baseline metrics and guardrails pinned. |
| B01 | documentation_harmonization | complete | ADR index, navigation, horizon, and oversized docs harmonized. |
| B02 | pipeline_entrypoint_and_order_consistency | complete | Canonical runner and stage wiring remediated. |
| B03 | config_path_and_version_hygiene | complete | Version/config/path hygiene remediated. |
| B04 | code_structure_and_test_scope_alignment | complete | Import boundary, helper dedupe, test scope realignment, orphan wiring complete. |
| B05 | repository_footprint_and_data_hygiene | complete | Wave 1 and Wave 2 (Step 1 + Step 2) complete; extraction and cleanup actions executed. |
| B06 | final_harmonization_and_claim_revalidation | complete | Full claim replay passing; residual risks closed in follow-up remediation wave. |

## Current Working Agreements

- Run commands in project venv: `source .venv/bin/activate`.
- Do not modify raw inputs in `data/raw/`.
- Use dry-run and claim-replay gates before and after batch edits.
- Keep audit narrative files (`docs/reviews/repo-hygiene-audit/00-07*.md`) unchanged unless explicitly requested.

## Near-Term Next Actions

1. Keep the documentation consistency queue current; add and resolve new cross-document drift items as they appear.
2. Keep repo-hygiene evidence current while executing publication tasks (`PP-001`, `PP-002`) and refresh package QA records as new export vintages are generated.
3. Continue PP-003 Phase 2 integration: implement `IMP-14` place workbook export and `IMP-15` provisional workbook place-sheet wiring.
4. ~~Execute `PP4-06`~~ Completed 2026-02-28. PP-004 workstream closed. Future periodic reviews follow the PP-002 cadence.

## Deferred / Later Work

- City/place projection Phase 2+ expansion under ADR-033 (rolling-origin backtests, multi-county place splitting, housing-unit method).
- Optional TIGER integration and geospatial export enhancements.
- Additional publication-facing formatting and package cleanup.

## References

- `AGENTS.md`
- `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`
- `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md`
- `docs/reviews/2026-02-28-publication-output-qa-packaging-checklist.md`
- `docs/reviews/2026-02-28-place-data-readiness-note.md`
- `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md`
- `docs/reviews/2026-02-28-pp3-s04-modeling-spec.md`
- `docs/reviews/2026-02-28-pp3-s05-backtesting-design.md`
- `docs/reviews/2026-02-28-pp3-s06-output-contract.md`
- `docs/reviews/2026-02-28-pp3-s07-approval-gate.md`
- `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md`
- `docs/reviews/pp3-backtest-outlier-narrative.md`
- `docs/reviews/2026-03-01-pp3-imp11-pipeline-stage-results.md`
- `docs/reviews/2026-03-01-pp3-imp12-imp13-results.md`
- `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md`
- `docs/reviews/2026-03-01-pp3-imp13a-results.md`
- `docs/plans/pp3-s08-implementation-kickoff.md`
- `docs/reviews/repo-hygiene-audit/implementation/17-open-risks-blockers-register.md`
- `docs/reviews/repo-hygiene-audit/implementation/02-action-batches.yaml`
- `docs/reviews/repo-hygiene-audit/verification/progress.md`
- `docs/governance/adrs/056-testing-strategy-maturation.md`
- `docs/guides/test-maintenance-practices.md`
- `docs/guides/test-suite-reference.md`
- `docs/guides/testing-workflow.md`
