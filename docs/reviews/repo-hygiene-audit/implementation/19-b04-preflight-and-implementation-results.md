# B04 Preflight and Implementation Results

## Batch

- Batch ID: `B04`
- Batch Name: `code_structure_and_test_scope_alignment`
- Date (UTC): 2026-02-26
- Decision: `GO` (preflight) and `COMPLETE` (implementation)

## Required Preflight Profiles

### DRY-TESTS

Command:
```bash
source .venv/bin/activate
pytest tests/ -q
```
Result: `PASS`
Evidence: `dry-tests-b04-preflight.log`
Key summary: `1258 passed, 5 skipped, 33 warnings`

### DRY-LINT-TYPE

Commands:
```bash
source .venv/bin/activate
ruff check .
mypy cohort_projections
ruff check cohort_projections/version.py cohort_projections/output/__init__.py cohort_projections/utils/__init__.py cohort_projections/utils/config_loader.py libs/project_utils/project_utils/config.py scripts/data/ingest_stcoreview.py scripts/data_processing/extract_pep_county_migration.py scripts/data_processing/extract_pep_county_migration_with_metadata.py
mypy cohort_projections/version.py cohort_projections/utils/__init__.py cohort_projections/utils/config_loader.py
```
Result: `PASS (non-regression policy)`
Evidence: `dry-lint-type-b04-preflight.log`
Key summary:
- Full repo lint/type baseline still fails outside B04 scope (`ruff` + `mypy` debt unchanged from prior baseline).
- Targeted lint/type gate commands pass.

### DRY-CHECK-REPLAY

Commands:
```bash
source .venv/bin/activate
python scripts/reviews/run_claim_checks.py run --status adjudicated
python scripts/reviews/run_claim_checks.py progress
```
Result: `PASS (command-level)`
Evidence: `dry-check-replay-b04-preflight.log`
Key summary: Replay executed and evidence regenerated; known resolved-state drift remained tracked.

## Implementation Changes Applied

1. Removed direct dynamic import-spec usage in `scripts/projections/run_pep_projections.py` by switching to module import (`importlib.import_module("scripts.pipeline.02_run_projections")`).
2. Consolidated median/dependency summary helpers in `cohort_projections/core/cohort_component.py` to utility-backed staticmethod aliases.
3. Consolidated report median helper in `cohort_projections/output/reports.py` to utility-backed alias.
4. Added package version wiring in `cohort_projections/__init__.py` (`__version__` import/export).
5. Added lazy example-module loader in `cohort_projections/data/process/__init__.py` for `example_usage` ownership boundary.
6. Reduced `cohort_projections/data/process/residual_migration.py` to 1299 lines (from 1309).
7. Moved sibling-repo-focused statistical test module from `tests/unit/` to `tests/test_integration/`.

## Post-Edit Validation

### DRY-LINT-TYPE

Evidence: `dry-lint-type-b04-implementation-postedit.log`
Result: `PASS (non-regression policy)`
- Full baseline remains: `ruff` debt + `mypy` 3 errors in `base_population_loader.py`.
- Targeted gate commands pass.

### DRY-TESTS

Evidence: `dry-tests-b04-implementation-postedit.log`
Result: `PASS`
- `1258 passed, 5 skipped, 33 warnings`

### Affected Claim Replay

Command set:
```bash
source .venv/bin/activate
python scripts/reviews/run_claim_checks.py run --claim-id RHA-008
python scripts/reviews/run_claim_checks.py run --claim-id RHA-010
python scripts/reviews/run_claim_checks.py run --claim-id RHA-014
python scripts/reviews/run_claim_checks.py run --claim-id RHA-024
python scripts/reviews/run_claim_checks.py run --claim-id RHA-025
python scripts/reviews/run_claim_checks.py progress
```
Evidence: `check-replay-b04-affected-postedit.log`
Result: `FAIL_EXPECTED_POST_REMEDIATION` for all 5 claims (`0/1` each), confirming pre-remediation predicates no longer hold.

Evidence artifacts:
- `20260226T210708Z_RHA-008.json`
- `20260226T210708Z_RHA-010.json`
- `20260226T210708Z_RHA-014.json`
- `20260226T210709Z_RHA-024.json`
- `20260226T210709Z_RHA-025.json`
