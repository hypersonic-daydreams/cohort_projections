# B03 Rerun + Implementation Results

**Batch:** `B03` (`config_path_and_version_hygiene`)  
**Date:** 2026-02-26  
**Operator:** codex

## Rerun Gate Results

1. `DRY-CONFIG`  
Status: **pass**  
Evidence: `./dry-config-b03-implementation-postedit.log`

2. `DRY-TESTS`  
Status: **pass**  
Evidence: `./dry-tests-b03-implementation-postedit.log`  
Summary: `1258 passed, 5 skipped, 33 warnings in 256.39s`.

3. `DRY-LINT-TYPE`  
Status: **pass (B03 non-regression policy)**  
Evidence: `./dry-lint-type-b03-implementation-postedit.log`  
Summary:
- Baseline full repo: `RC_RUFF_ALL=1`, `RC_MYPY_ALL=1` (existing debt captured).
- B03-targeted scope: `RC_RUFF_B03=0`, `RC_MYPY_B03=0`, `ASSERTION=PASS`.

4. `DRY-CHECK-REPLAY`  
Status: **command-pass**  
Evidence: `./dry-check-replay-b03-implementation-postedit.log`

## B03 Implementation Scope Applied

- RHA-005: version hygiene remediation
  - removed stale `__version__` from `cohort_projections/output/__init__.py`
  - made `cohort_projections/version.py` derive from package metadata with `pyproject.toml` fallback
- RHA-006: config loader unification path
  - re-exported `ConfigLoader` from `project_utils` in `cohort_projections/utils/config_loader.py`
  - added compatibility methods to `project_utils.ConfigLoader`
  - removed dual-loader shim path in `cohort_projections/utils/__init__.py`
- RHA-007: hardcoded path remediation
  - replaced user-specific absolute path in `scripts/data/ingest_stcoreview.py`
  - replaced literal shared-data path strings in both PEP extraction scripts with path-component construction
- RHA-016: data documentation completion
  - added `DATA_SOURCE_NOTES.md` to five required `data/raw/*` subdirectories

## Affected Claim Replay (Post-Edit)

Evidence log: `./check-replay-b03-affected-postedit.log`

- `RHA-005`: `0/1` (expected drift after remediation)
- `RHA-006`: `0/1` (expected drift after remediation)
- `RHA-007`: `0/1` (expected drift after remediation)
- `RHA-016`: `0/1` (expected drift after remediation)

## Decision

- B03 rerun/implementation outcome: **GO + COMPLETED**.
- Follow-up needed: redesign resolved-state checks for remediated claims before final program harmonization.
