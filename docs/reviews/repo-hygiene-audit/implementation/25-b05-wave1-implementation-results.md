# B05 Wave 1 Implementation Results

## Batch

- Batch ID: `B05`
- Batch Name: `repository_footprint_and_data_hygiene`
- Date (UTC): 2026-02-27
- Decision: `GO` for Wave 1 (non-destructive compatibility + initial tracked cleanup)

## Scope Implemented

1. Added SDC boundary-compatible path resolution utilities:
   - `cohort_projections/utils/sdc_paths.py`
2. Updated runtime code to stop hardcoding in-repo SDC rate path:
   - `cohort_projections/data/process/residual_migration.py`
3. Exported resolver utilities for shared usage:
   - `cohort_projections/utils/__init__.py`
4. Added test helper for SDC repo discovery and module-level skip behavior:
   - `tests/_sdc_paths.py`
5. Rewired SDC-dependent tests to the shared resolver helper (no in-repo hardcoded paths).
6. Removed tracked root clutter sentinel:
   - deleted `RCLONE_TEST`
   - added `.gitignore` rule for `RCLONE_TEST`

## Validation

### Targeted Lint Gate

Command:
```bash
source .venv/bin/activate
ruff check --select F401,F841 <wave1_file_set>
```
Result: `PASS`  
Evidence: `dry-lint-b05-wave1-targeted.txt`

### Focused Regression Tests

Command:
```bash
source .venv/bin/activate
pytest tests/test_data/test_residual_migration.py \
  tests/test_statistical/test_bayesian_var.py \
  tests/test_statistical/test_multistate_placebo.py \
  tests/test_statistical/test_regime_aware.py \
  tests/test_tools/test_citation_audit.py \
  tests/unit/test_build_dhs_lpr_panel_variants.py \
  tests/unit/test_duration_figure_table_consistency.py \
  tests/unit/test_journal_article_derived_stats.py \
  tests/unit/test_journal_article_versioning.py \
  tests/unit/test_module_7_causal_inference.py \
  tests/unit/test_module_8_duration_analysis.py \
  tests/unit/test_sdc_data_loader.py \
  tests/test_integration/test_adr021_modules.py -q
```
Result: `PASS`  
Summary: `317 passed, 8 skipped, 1 warning`  
Evidence: `dry-tests-b05-wave1-focused.txt`

### B05 Claim Replay (Wave 1)

Command:
```bash
source .venv/bin/activate
python scripts/reviews/run_claim_checks.py run \
  --claim-id RHA-011 --claim-id RHA-012 --claim-id RHA-015 \
  --claim-id RHA-017 --claim-id RHA-018 --claim-id RHA-026
```
Result: `PASS` (`1/1` for all six claims)  
Evidence: `check-replay-b05-wave1.txt`

## Follow-on Work (Wave 2)

1. Execute physical extraction of `sdc_2024_replication/` to sibling repo location.
2. Apply approved root/stale/placeholder cleanup actions from `24-b05-delete-archive-proposal.md`.
3. Re-run B05 claim checks (`RHA-011/012/015/017/018/026`) and redesign to resolved-state assertions.

## Status

- B05 overall: `in_progress`
- Wave 1: `complete`
- Wave 2 (destructive cleanup + extraction): `pending approval/execution`
