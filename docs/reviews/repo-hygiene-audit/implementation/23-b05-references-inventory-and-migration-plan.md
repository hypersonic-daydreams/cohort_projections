# B05 References Inventory + SDC Extraction Migration Plan

**Date (UTC):** 2026-02-27  
**Batch:** `B05` (`repository_footprint_and_data_hygiene`)  
**Operator:** codex

## Purpose

Satisfy RB-005 action item 2 by documenting:
1. the in-repo references that depend on `sdc_2024_replication/`, and
2. an execution-ready migration plan (with validation + rollback) for extracting
   `sdc_2024_replication/` to a sibling repository under `~/workspace/demography/`.

## Current Footprint Snapshot

- `TOTAL_FILES=11151`
- `SDC_FILES=7842`
- `SDC_SHARE=0.7033`
- Tracked files under `sdc_2024_replication/`: `271`

Commands used:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
exclude = {'.git', '.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.uv_cache', 'htmlcov'}
all_files = [p for p in Path('.').rglob('*') if p.is_file() and not any(part in exclude for part in p.parts)]
sdc_files = [p for p in all_files if p.parts and p.parts[0] == 'sdc_2024_replication']
print(f"TOTAL_FILES={len(all_files)}")
print(f"SDC_FILES={len(sdc_files)}")
print(f"SDC_SHARE={len(sdc_files)/len(all_files):.4f}")
PY
git ls-files sdc_2024_replication | wc -l
```

## Reference Inventory (Actionable)

### Runtime and test path dependencies

- Implemented in this wave:
  - `cohort_projections/data/process/residual_migration.py`
  - `tests/_sdc_paths.py`
  - SDC-dependent tests now resolve path through shared helper (env/sibling/in-repo fallback).

New resolver order:
1. `SDC_2024_REPLICATION_ROOT` env var
2. sibling path: `../sdc_2024_replication`
3. in-repo fallback: `./sdc_2024_replication`

### Remaining references requiring migration updates

`rg` summary (line-hit counts by file):
- `docs/INDEX.md`: 160
- `docs/reviews/methodology_comparison/02_fertility.md`: 9
- `docs/reviews/methodology_comparison/03_mortality.md`: 9
- `docs/governance/sops/SOP-001-external-ai-analysis-integration.md`: 6
- `docs/governance/plans/PACKAGE_EXTRACTION_PLAN.md`: 4
- `AGENTS.md`: 3
- `scripts/setup_rclone_bisync.sh`: 2

Operationally important items to update during extraction:
- `AGENTS.md` (journal article location guidance)
- `docs/INDEX.md` (generated links into `sdc_2024_replication/`)
- `scripts/setup_rclone_bisync.sh` (sync filters with in-repo SDC paths)
- Root symlinks (untracked):
  - `journal_article_output -> sdc_2024_replication/.../output`
  - `journal_article_versions -> sdc_2024_replication/.../output/versions`

## Migration Plan (Execution-Ready)

### Phase 1: Compatibility guardrail (completed)

- Added path resolver module: `cohort_projections/utils/sdc_paths.py`
- Updated runtime + tests to use resolver instead of hardcoded in-repo path.
- Validation:
  - `dry-lint-b05-wave1-targeted.txt`
  - `dry-tests-b05-wave1-focused.txt`

### Phase 2: Extract SDC repository boundary

1. Ensure sibling target exists and is backed by intended git remote:
   - `~/workspace/demography/sdc_2024_replication/`
2. Copy/move current subtree with metadata preserved.
3. Verify the five canonical rate CSVs remain in extracted repo.
4. Remove in-repo tracked `sdc_2024_replication/**` content only after Phase 3 references are updated.

### Phase 3: Reference rewiring

1. Regenerate and prune docs index links:
   - `python scripts/intelligence/generate_docs_index.py`
2. Update AGENTS and SOP references to sibling-repo wording.
3. Update bisync rules so this repo no longer assumes in-repo SDC payloads.
4. Retarget/replace root journal symlinks.

### Phase 4: Validation gates

1. `pytest tests/ -q`
2. `python scripts/reviews/run_claim_checks.py run --status adjudicated`
3. `python scripts/reviews/run_claim_checks.py progress`
4. Re-run B05 affected claims: `RHA-011/012/015/017/018/026`

## Rollback Plan

Trigger rollback if any of these occur:
- runtime cannot resolve required SDC rate files,
- statistical/module test imports fail due boundary change,
- claim replay introduces unexpected non-B05 regressions.

Rollback steps:
1. Restore this repo changes:
   - `git restore cohort_projections/data/process/residual_migration.py cohort_projections/utils/__init__.py cohort_projections/utils/sdc_paths.py tests/_sdc_paths.py tests/`
2. Restore subtree placement (if moved):
   - move `../sdc_2024_replication` back under repo root.
3. Re-run focused validation and full claim replay.

## Status

- RB-005 action 2: **complete** (inventory + migration plan documented)
- B05: **in progress** (Wave 1 compatibility completed; extraction + destructive cleanup pending Wave 2 execution)
