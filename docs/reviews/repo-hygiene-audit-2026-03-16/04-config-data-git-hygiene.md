# Repository Hygiene Audit: Configuration, Data Management, and Git Hygiene

**Date:** 2026-03-16
**Auditor:** Claude Opus 4.6 (automated)
**Scope:** `.gitignore`, tracked data files, configuration management, dependency hygiene, environment setup
**Repository:** `/home/nhaarstad/workspace/demography/cohort_projections`

---

## Summary of Findings

| ID | Category | Severity | Finding |
|----|----------|----------|---------|
| GH-001 | Git Hygiene | WARNING | 488 files tracked despite matching `.gitignore` patterns (committed before rules added) |
| GH-002 | Git Hygiene | WARNING | 2 PDF files (726 KB) tracked in `docs/` with no gitignore coverage for `docs/**/*.pdf` |
| GH-003 | Git Hygiene | INFO | Duplicate `.gitignore` entry: `data/analysis/**` appears twice (lines 74 and 113) |
| GH-004 | Git Hygiene | INFO | `!data/projections/.gitkeep` appears twice (lines 87 and 110) |
| GH-005 | Git Hygiene | WARNING | 205 SDC figure objects (PNG/PDF, ~10 MB) remain in git pack from prior commits; never cleaned with `git filter-repo` or BFG |
| GH-006 | Git Hygiene | INFO | `uv.lock` has 15 revisions in history totaling 11.09 MB of pack objects; inherent to lockfile management |
| GH-007 | Git Hygiene | INFO | `.git/` directory is 67 MB total; moderate for a repo of this age |
| GH-008 | Git Hygiene | INFO | `check-added-large-files` threshold is 5 MB; appropriate for this project |
| GH-009 | Git Hygiene | INFO | Missing `.gitkeep` files for `data/interim/`, `data/backtesting/`, `data/analysis/`, `data/exports/`, `data/output/` (directories exist on disk only via rclone sync) |
| CM-001 | Config | INFO | `config/projection_config.yaml` is well-structured with ADR cross-references and inline comments |
| CM-002 | Config | INFO | No hard-coded absolute paths found in `cohort_projections/` or `scripts/` Python files |
| CM-003 | Config | INFO | GCP project ID `antigravity-sandbox` appears in config YAML and `.env.example` but not in Python source code; acceptable |
| CM-004 | Config | INFO | BigQuery credentials path uses `~` expansion (`~/.config/gcloud/cohort-projections-key.json`); not a hard-coded absolute path |
| CM-005 | Config | WARNING | Magic numbers in `survival_rates.py` (fallback mortality rates: 0.006, 0.0005, 0.001, 0.003, etc.) lack named constants or config references |
| CM-006 | Config | WARNING | Magic numbers in `migration_rates.py` (Rogers-Castro model parameters: 0.02, 0.08, 0.06, 0.001) lack named constants |
| CM-007 | Config | INFO | `fertility.py` magic number `max_plausible_rate = 0.35` is named and self-documenting |
| CM-008 | Config | INFO | `residual_migration.py` cap value `0.20` and dampening factor `0.80` match config values; minor inline duplication |
| DM-001 | Data Mgmt | INFO | Data directory structure follows standard `raw/processed/interim/projections/analysis/exports` layout |
| DM-002 | Data Mgmt | INFO | `data/raw/` subdirectories each have `DATA_SOURCE_NOTES.md` tracked by git; good provenance practice |
| DM-003 | Data Mgmt | INFO | `data/processed/` has `.gitkeep` and all data files are properly gitignored; only `METHODOLOGY_NOTES.md` is tracked |
| DM-004 | Data Mgmt | INFO | `data/interim/` and `data/analysis/` directories are fully gitignored with `**` patterns |
| DM-005 | Data Mgmt | INFO | 11 raw PDFs exist on disk in `data/raw/` (VES reports, etc.) but are properly gitignored via `data/raw/**/*.pdf` |
| DP-001 | Dependencies | INFO | All dependencies use `>=` lower-bound pinning with `uv.lock` providing exact resolution; standard for Python projects |
| DP-002 | Dependencies | WARNING | `pathspec>=0.12.1` is a direct dependency but is never imported in project code; it is only used transitively by mypy |
| DP-003 | Dependencies | INFO | `psycopg2-binary` and `sqlalchemy` are in core dependencies but only used by `reproducibility.py` and `scripts/`; could be optional |
| DP-004 | Dependencies | INFO | `pdfplumber` and `ocrmypdf` are in core dependencies but only used by 3 scripts in `scripts/data/`; could be optional |
| DP-005 | Dependencies | INFO | `db-dtypes` is a transitive requirement of `google-cloud-bigquery`; correct to list explicitly |
| DP-006 | Dependencies | INFO | `matplotlib`, `scipy`, `statsmodels` are duplicated between core and optional `[viz]`/`[stats]` groups; intentional per comment |
| EN-001 | Environment | INFO | `.envrc` is minimal and correct: loads `.env`, sets `VIRTUAL_ENV`, adds to `PATH` |
| EN-002 | Environment | INFO | `.env` exists on disk and is properly gitignored; `.env.example` is tracked with placeholder values |
| EN-003 | Environment | INFO | `.python-version` pins Python 3.12; `pyproject.toml` requires `>=3.11`; consistent |
| EN-004 | Environment | INFO | `uv.lock` is tracked and up-to-date; environment is reproducible |
| EN-005 | Environment | INFO | `bisync.sh` uses rclone for data sync between machines; well-structured with `--dry-run` and `--resync` options |
| EN-006 | Config | INFO | `pre-commit-config.yaml` includes rclone reminder, methodology doc check, pytest, data manifest, and code inventory hooks |

---

## 1. Git Hygiene

### 1.1 `.gitignore` Completeness

The `.gitignore` is comprehensive at 192 lines, covering:

- Python artifacts, virtual environments, IDE files, Jupyter checkpoints
- All data directories (`raw/`, `processed/`, `projections/`, `interim/`, `analysis/`, `backtesting/`, `exports/`, `output/`)
- Generated outputs (HTML reports, JSON evidence, SVG/PNG assets)
- Environment files (`.env`, `.env.local`)
- SDC replication outputs
- Claude Code config (`.claude/*` with `!.claude/commands/` exception)
- OS files (`.DS_Store`, `Thumbs.db`)

**Missing patterns:**

- `docs/**/*.pdf` -- Two PDFs are tracked in `docs/research/` and `docs/reviews/`. No gitignore rule covers PDFs outside of `data/` and `sdc_2024_replication/`. If PDFs in `docs/` are intended as permanent documentation, this is acceptable; if they are generated/downloaded artifacts, a gitignore rule is needed.

**Duplicate entries:**

- `data/analysis/**` appears on lines 74 and 113.
- `!data/projections/.gitkeep` appears on lines 87 and 110.

These duplicates are harmless but add clutter.

### 1.2 Files Tracked Despite `.gitignore` Rules

**488 files** are currently tracked that match gitignore patterns. They were committed before the gitignore rules were added. Git does not retroactively untrack files when a new gitignore rule is created.

| Path Pattern | Count | Size | Notes |
|-------------|-------|------|-------|
| `docs/reviews/repo-hygiene-audit/verification/evidence/*.json` | 455 | 876 KB | Evidence files from prior audit |
| `docs/reviews/assets/pp3-human-review/**/*.svg` | 21 | 938 KB | Generated SVG charts |
| `data/backtesting/place_backtest_results/*` | 8 | 468 KB | Backtest CSVs and JSON |
| `docs/reviews/*.html` | 3 | 282 KB | Generated HTML reports |
| `docs/reviews/pp3-human-review-package_latest.json` | 1 | 1.4 KB | Generated JSON |

**Remediation:** Run `git rm --cached <path>` for each set of files, then commit. This removes them from tracking without deleting from disk.

### 1.3 Large Files in Git History

The git pack (52.47 MB) contains historical blobs that inflate repo size:

| Category | Approx. Size | Status |
|----------|-------------|--------|
| SDC replication figures (PNG/PDF) | ~10 MB (205 objects) | Removed from HEAD but in pack |
| `uv.lock` revisions | 11.09 MB (15 revisions) | Expected; lockfile churn |
| Raw data PDFs (fertility NVSR reports) | ~2.7 MB | Removed from HEAD but in pack |
| Walk-forward HTML reports | ~2.4 MB | Removed from HEAD but in pack |

The SDC figures were committed, later removed (commit `124682f`), but the objects remain in the pack. A `git filter-repo` or BFG cleanup could reclaim ~15 MB, but this is low priority given the 67 MB total `.git/` size.

### 1.4 Sensitive Files

- `.env` exists on disk and is properly gitignored (verified: `.gitignore:105`).
- `.env.example` is tracked with empty credential placeholders -- correct practice.
- `~/.config/gcloud/cohort-projections-key.json` is referenced in config but is outside the repo; never at risk of being committed.
- No API keys, tokens, or credentials found in tracked files.
- The GCP project ID `antigravity-sandbox` appears in config and docs but is not a secret (project IDs are not credentials).

---

## 2. Configuration Management

### 2.1 `config/projection_config.yaml`

The main configuration file is **504 lines**, well-structured with:

- Clear section headers (`=== BASE POPULATION ===`, `=== DEMOGRAPHIC RATES ===`, etc.)
- ADR cross-references on nearly every significant parameter (e.g., "ADR-047", "ADR-055 Phase 2")
- Inline comments explaining purpose and options
- Geographic, demographic, rate, scenario, output, and pipeline sections

**Strengths:**
- No hard-coded file paths; all paths are relative to project root
- Scenarios are declarative with clear descriptions
- ADR traceability is excellent

**Minor observations:**
- `aggregation_tolerance: 0.01` in the config is labeled "1% tolerance" but ADR-062 widened this to 2.0 persons. The config value may be stale or represent a different tolerance (percentage vs. absolute). Worth verifying.

### 2.2 Hard-Coded Paths

No hard-coded absolute paths (`/home/`, `/Users/`, `C:\`) were found in any Python source file in `cohort_projections/` or `scripts/`. All file references use relative paths from the project root or config-driven paths. This is excellent adherence to the "NEVER hard-code file paths" rule.

### 2.3 Magic Numbers

Several files contain numeric literals that function as domain-specific constants. These are partially documented but could benefit from named constants or config extraction:

**`cohort_projections/data/process/survival_rates.py`** (lines 984-998):
Fallback mortality rates for synthetic life tables use age-specific `qx` values (0.006, 0.0005, 0.001, 0.003, 0.02, 0.10) with sex-specific multipliers (0.85, 0.90). These are standard demographic values but are embedded inline without named constants or literature citations.

**`cohort_projections/data/process/migration_rates.py`** (lines 370-378):
Rogers-Castro model parameters (`a1=0.02`, `alpha1=0.08`, `a2=0.06`, `c=0.001`) and age-specific proportions (0.75, 0.45, 0.25, 0.20, 0.10) are inline. These are well-known demographic model parameters but could benefit from a `ROGERS_CASTRO_PARAMS` dict or config reference.

**`cohort_projections/core/fertility.py`** (line 169):
`max_plausible_rate = 0.35` is a named variable with clear intent -- good practice.

**`cohort_projections/data/process/residual_migration.py`** (lines 516, 535):
`cap_value = 0.20` and `male_dampening_factor: float = 0.80` duplicate values from config. The config is the source of truth, so these defaults should ideally reference it or be documented as fallbacks.

### 2.4 Config Directory

The `config/` directory contains 12 files:

```
benchmark_evaluation_policy.yaml
data_sources.yaml
evaluation_config.yaml
experiment_log_schema.yaml
experiment_spec_schema.yaml
method_profiles/
nd_brand.yaml
observatory_config.yaml
observatory_recipes.yaml
observatory_search_policy.yaml
observatory_variants.yaml
projection_config.yaml
```

This is a clean separation of concerns. No stale or orphaned config files were identified.

---

## 3. Data Directory Structure

### 3.1 Layout

```
data/
  raw/           # Source data (synced via rclone, gitignored)
    census/
    census_bureau_methodology/
    enrollment/
    fertility/
    geographic/
    housing/
    immigration/
    migration/
    mortality/
    nd_sdc_2024_projections/
    population/
  processed/     # Derived data (gitignored)
  interim/       # Intermediate processing (gitignored)
    geographic/tiger2020/
    immigration/ocr/
  projections/   # Projection outputs (gitignored)
  analysis/      # Experiment/benchmark outputs (gitignored)
  backtesting/   # Backtest results (gitignored, but 8 files pre-date rule)
  exports/       # Dissemination packages (gitignored)
  output/        # Legacy output directory (gitignored)
  metadata/      # Database backups (gitignored)
```

The structure follows a clean `raw -> processed -> projections -> exports` pipeline pattern. Each `raw/` subdirectory has a `DATA_SOURCE_NOTES.md` tracked by git for provenance.

### 3.2 `.gitkeep` Coverage

`.gitkeep` files are tracked for core directories:
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/projections/.gitkeep`
- `data/raw/{fertility,geographic,immigration,migration,mortality,population}/.gitkeep`
- `data/processed/immigration/{.gitkeep,analysis/.gitkeep,rates/.gitkeep}`
- `data/analysis/experiments/{pending,completed}/.gitkeep`

**Missing `.gitkeep`:** `data/interim/`, `data/backtesting/`, `data/exports/`, `data/output/`, `data/metadata/` -- these directories exist on disk via rclone sync but have no `.gitkeep` to ensure structure on fresh clone. A fresh `git clone` would not create these directories.

### 3.3 Stale/Orphaned Data

- `data/output/` contains `reports/` and `visualizations/` subdirectories but is fully gitignored. The config references `data/exports/` and `data/projections/` as output targets. `data/output/` may be a legacy directory.
- `data/processed/integration_test_results.csv` (216 KB) exists on disk -- appears to be a test artifact, not a pipeline output.
- `data/processed/reports/` contains 10 timestamped `data_processing_report_*.json` files -- pipeline artifacts that accumulate over time.
- `data/processed/city_population_exploration_2020_2024.csv` -- appears to be an exploratory analysis artifact rather than a pipeline output.
- `data/processed/pep_county_components_2000_2024.*` -- superseded by `_2000_2025` versions; stale data.

---

## 4. Dependencies

### 4.1 Pinning Strategy

All dependencies use `>=` lower-bound-only constraints (e.g., `pandas>=2.0.0`). Exact resolution is provided by the tracked `uv.lock` file. This is the standard and recommended approach for application projects using `uv`.

### 4.2 Dependency Review

**Potentially unnecessary core dependencies:**

| Package | Usage | Recommendation |
|---------|-------|----------------|
| `pathspec>=0.12.1` | Not imported anywhere in project code. Only used transitively by mypy. | **Remove from core dependencies.** Mypy will pull it as its own dependency. |
| `psycopg2-binary>=2.9.11` | Used only in `cohort_projections/utils/reproducibility.py` and 7 scripts. | Consider moving to an optional `[db]` group. |
| `sqlalchemy>=2.0.45` | Used only in `cohort_projections/utils/reproducibility.py` and 7 scripts. | Consider moving to an optional `[db]` group. |
| `pdfplumber>=0.11.8` | Used only in 3 scripts (`scripts/data/`). | Consider moving to an optional `[pdf]` group. |
| `ocrmypdf>=16.13.0` | Used only in 3 scripts (`scripts/data/`). | Consider moving to an optional `[pdf]` group (already have `[pdf_export]` but different scope). |

**Duplicate specifications:**
- `matplotlib>=3.10.0` appears in both core dependencies and `[viz]` optional group.
- `scipy>=1.13.0` and `statsmodels>=0.14.6` appear in both core and `[stats]`.
- This is documented as intentional ("Core analysis (canonical versions - also in optional stats group)").

### 4.3 Internal Packages

Three internal packages are specified as editable `uv` sources:
```
project-utils -> libs/project_utils
evidence-review -> libs/evidence_review
codebase-catalog -> libs/codebase_catalog
```

These are properly configured in `[tool.uv.sources]` with `editable = true`.

---

## 5. Environment Setup

### 5.1 Reproducibility

The development environment is well-specified:

- `.python-version` pins Python 3.12
- `pyproject.toml` sets `requires-python = ">=3.11"` (compatible)
- `uv.lock` provides exact dependency resolution (tracked in git)
- `.envrc` sets up `VIRTUAL_ENV` and `PATH` via direnv
- `.env.example` documents all environment variables with descriptions
- `bisync.sh` provides data synchronization between machines

A fresh setup requires:
1. `git clone` (code)
2. `direnv allow` (environment)
3. `uv sync` (dependencies)
4. `cp .env.example .env` and fill in credentials
5. `./scripts/bisync.sh --resync` (data files)

This is well-documented in `CLAUDE.md` under "Session Workflow".

### 5.2 Pre-commit Hooks

The `.pre-commit-config.yaml` includes:

- Standard hooks (trailing whitespace, end-of-file, YAML/JSON/TOML checks, large file check at 5 MB, merge conflict detection)
- Ruff linting and formatting (excluding `sdc_2024_replication/`)
- MyPy type checking (excluding tests, examples, SDC replication, legacy ADR scripts)
- Pytest (fast tests only, triggered by code changes in `cohort_projections/` or `tests/`)
- Custom hooks: rclone reminder, methodology doc check, data manifest enforcement, code inventory update

The `methodology-doc-check` hook triggers only when core engine or data processing files change -- well-targeted.

---

## 6. Recommendations

### Priority 1 (Should fix)

1. **Untrack files matching gitignore rules (GH-001):** Run the following to remove 488 files from git tracking without deleting from disk:
   ```bash
   git rm --cached -r data/backtesting/place_backtest_results/
   git rm --cached -r docs/reviews/repo-hygiene-audit/verification/evidence/
   git rm --cached -r docs/reviews/assets/
   git rm --cached docs/reviews/pp3-human-review-package_latest.json
   git commit -m "chore: untrack files now covered by .gitignore"
   ```

2. **Remove `pathspec` from core dependencies (DP-002):** It is a transitive dependency of mypy and not used by project code.

### Priority 2 (Consider)

3. **Add `docs/**/*.pdf` to `.gitignore` (GH-002):** If the two tracked PDFs are downloaded/generated artifacts rather than permanent documentation. Then `git rm --cached` them.

4. **Clean duplicate `.gitignore` entries (GH-003, GH-004):** Remove the second `data/analysis/**` (line 113) and second `!data/projections/.gitkeep` (line 110).

5. **Add `.gitkeep` files for `data/interim/`, `data/backtesting/`, `data/exports/` (GH-009):** Ensures directory structure is preserved on fresh clone.

6. **Extract magic numbers to named constants (CM-005, CM-006):** Particularly the Rogers-Castro parameters in `migration_rates.py` and fallback mortality rates in `survival_rates.py`.

7. **Move heavy optional dependencies to optional groups (DP-003, DP-004):** `psycopg2-binary`, `sqlalchemy`, `pdfplumber`, `ocrmypdf` are only used by specific scripts/utilities and could be in `[db]` and `[pdf]` optional groups.

### Priority 3 (Low priority / informational)

8. **Consider `git filter-repo` to prune historical blobs (GH-005):** ~15 MB of deleted SDC figures and data files remain in the pack. Only worth doing if repo cloning speed is a concern.

9. **Clean up stale processed data files:** `data/processed/pep_county_components_2000_2024.*` (superseded by `_2000_2025`), `data/processed/city_population_exploration_2020_2024.csv`, `data/processed/integration_test_results.csv`.

10. **Verify `aggregation_tolerance` config value:** `0.01` in config vs. ADR-062's "2.0 persons" widening -- confirm these represent different tolerance types (percentage vs. absolute).

---

## 7. Overall Assessment

The repository demonstrates **strong hygiene practices** overall:

- Comprehensive `.gitignore` with clear categorization and ADR references
- No hard-coded paths in source code
- No credentials or secrets in tracked files
- Well-structured configuration with ADR traceability
- Clean data directory hierarchy with provenance documentation
- Reproducible environment via `uv.lock` + `.envrc` + `.env.example`
- Effective pre-commit hook suite

The primary area for improvement is **retroactive untracking** of ~488 files that were committed before their corresponding `.gitignore` rules were added. This is a common Git workflow issue and straightforward to remediate. The dependency list has minor bloat from packages that could be optional, and a few files of magic numbers in demographic model code would benefit from named constants.

---

| Attribute | Value |
|-----------|-------|
| **Audited** | 2026-03-16 |
| **Auditor** | Claude Opus 4.6 (automated) |
| **Git SHA** | `0b133d4` (HEAD at audit time) |
| **Previous Audit** | `docs/reviews/repo-hygiene-audit/05-config-data-management.md` (2026-02-26) |
