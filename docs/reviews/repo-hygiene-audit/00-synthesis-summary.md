# Repository Hygiene Audit — Synthesis Summary

**Date:** 2026-02-26
**Scope:** Full repository evaluation for organization, AI navigability, and maintainability

---

## Audit Reports

| # | Report | Focus |
|---|--------|-------|
| [01](01-directory-structure.md) | Directory Structure | Tree layout, file counts, nesting, proposed reorganization |
| [02](02-scripts-workflow.md) | Scripts & Workflow | Script inventory, pipeline completeness, execution order |
| [03](03-documentation.md) | Documentation | Doc inventory, redundancy, staleness, AI navigability |
| [04](04-code-organization.md) | Code Organization | Python package structure, module sizes, imports, tests |
| [05](05-config-data-management.md) | Config & Data | Configuration files, data paths, hardcoded values |
| [06](06-ai-navigability-and-embeddings.md) | AI Navigability & Embeddings | Agent usability, embeddings feasibility, tooling |
| [07](07-pipeline-claim-validation.md) | Pipeline Claim Validation | Evidence check of "fragmented and incomplete pipeline" finding |

---

## Top-Level Findings

### The Good

- **Clean Python architecture**: No circular imports, clear layer separation (config → utils → core → data → geographic → output)
- **Strong test coverage**: 1,257 tests, 1.41x test-to-code ratio
- **Excellent AI entry points**: CLAUDE.md → AGENTS.md layered system works well
- **Well-designed config**: YAML-driven with proper `raw → processed → projections → exports` data pipeline
- **Good ADR discipline**: 97 architectural decisions documented with cross-references
- **Core codebase is manageable**: Only 39 files / 17,759 lines of library code

### The Bad

- **`sdc_2024_replication/` dominates everything**: 7,797 files (75% of repo), 702 directories, effectively a separate research project embedded inside this one
- **Documentation outweighs code**: 78,000 lines of markdown vs 63,000 lines of Python — docs are both the greatest strength and the greatest AI navigability risk
- **Pipeline is fragmented and incomplete**: `run_complete_pipeline.sh` only calls 3 of 7 steps; numbering collisions; ghost references to nonexistent files
- **3 oversized modules**: `migration_rates.py` (1,963 lines), `base_population_loader.py` (1,465 lines), `residual_migration.py` (1,309 lines)
- **Root-level clutter**: Stray data files, PDFs, and test directories tracked in git

### The Ugly

- **4 conflicting version strings**: `pyproject.toml` (0.1.0), `version.py` (0.1.0), `output/__init__.py` (1.0.0), `CLAUDE.md` (2.3.0)
- **Ghost script reference**: `scripts/projections/run_all_projections.py` referenced in 10+ docs but doesn't exist
- **Duplicate ConfigLoader implementations** with different behaviors
- **~500 lines of reusable library code trapped in a 2,084-line pipeline script** (`02_run_projections.py`)
- **~3,500 lines of tests testing sibling repos**, not this project

---

## Cross-Cutting Themes

### 1. The Repo Contains Multiple Projects
The single biggest issue. `sdc_2024_replication/` is an independent research project (validation of SDC's 2024 projections) that has grown to dominate the repository. It should be extracted to its own repo or at minimum isolated so it doesn't pollute glob/grep operations.

### 2. Documentation Volume Is the AI Navigability Bottleneck
Embeddings are **not needed** — the core codebase is small enough for grep/glob. The real problem is documentation sprawl: 1,183 markdown files, 3 competing navigation documents, a 925-line changelog masquerading as a tracker, and a 291KB auto-generated inventory. Pruning and focusing documentation will do more than any search infrastructure.

### 3. The Pipeline Concept Is Half-Built
The `scripts/pipeline/` directory has the right idea (numbered steps) but the implementation is incomplete: numbering collisions, missing steps in the runner, and reusable code locked inside scripts. This is the area that would benefit most from the numbered-directory approach the user mentioned.

### 4. Dead Code and Stale References Accumulate
Orphaned modules, unused utility functions, empty placeholder directories, stale doc references, and triplicated data files all suggest the repo has grown organically without periodic cleanup sweeps.

---

## Embeddings Verdict

**Not recommended at this time.** The core library (17,759 lines) fits comfortably in a single context window. Grep/glob find anything in milliseconds. The investment should go into:
1. A concise `CODEBASE_MAP.md` (~200 lines) replacing the 291KB inventory
2. An ADR summary index (one-row-per-ADR table)
3. Documentation pruning

If the repo grows significantly or absorbs additional sub-projects, revisit MCP-based search or Sourcegraph.

---

## Prioritized Action Plan

### Phase 1: Quick Wins (1-2 hours)

| # | Action | Impact |
|---|--------|--------|
| 1.1 | Remove/archive root-level clutter (`2025_popest_data/`, `journal_article_pdfs/`, stray files) | Reduces noise |
| 1.2 | Fix ghost reference: remove `run_all_projections.py` from all docs or create it | Prevents confusion |
| 1.3 | Unify version string to single source of truth in `pyproject.toml` | Eliminates contradictions |
| 1.4 | Delete dead code: `example_usage.py`, orphaned `version.py`, stub `vital_stats.py` | Reduces clutter |
| 1.5 | Update ADR README index (fix 10 wrong statuses, add 2 missing ADRs) | Fixes bad agent info |
| 1.6 | Split DEVELOPMENT_TRACKER.md: active status (~100 lines) + archived history | Reduces context waste |

### Phase 2: Structural Improvements (1-2 days)

| # | Action | Impact |
|---|--------|--------|
| 2.1 | Extract `sdc_2024_replication/` to separate repo (or add to .gitignore + separate sync) | Biggest single improvement |
| 2.2 | Renumber `scripts/pipeline/` sequentially (00-06) and fix `run_complete_pipeline.sh` | Clear workflow |
| 2.3 | Consolidate `scripts/data/` and `scripts/data_processing/` | Reduces confusion |
| 2.4 | Create `CODEBASE_MAP.md` (~200 lines, machine-readable) | AI navigation aid |
| 2.5 | Consolidate 3 navigation docs into 1 | Single source of truth |
| 2.6 | Unify duplicate ConfigLoader implementations | Maintenance reduction |
| 2.7 | Archive 12 stale documents (~4,345 lines) | Context window savings |

### Phase 3: Code Quality (1-2 weeks, incremental)

| # | Action | Impact |
|---|--------|--------|
| 3.1 | Split `migration_rates.py` (1,963 lines) into 3-4 focused modules | Maintainability |
| 3.2 | Extract GQ logic from `base_population_loader.py` | Single responsibility |
| 3.3 | Move reusable functions from `02_run_projections.py` into package | Proper code organization |
| 3.4 | Deduplicate `calculate_median_age()` (3 copies) and `calculate_dependency_ratio()` (2 copies) | DRY |
| 3.5 | Standardize import style (absolute everywhere) | Consistency |
| 3.6 | Move/remove tests that test sibling repos (~3,500 lines) | Focus |
| 3.7 | Add missing DATA_SOURCE_NOTES.md to 5 raw subdirectories | Completeness |

### Phase 4: Ongoing Hygiene (continuous)

| # | Action | Impact |
|---|--------|--------|
| 4.1 | Adopt numbered directory convention for new workflow directories | Clarity |
| 4.2 | Periodic cleanup sweeps (quarterly: dead code, empty dirs, stale docs) | Prevents drift |
| 4.3 | Pre-commit check for hardcoded paths | Prevents regression |
| 4.4 | ADR lifecycle management (archive superseded ADRs) | Keeps index useful |

---

## Proposed Numbered Directory Convention

For `scripts/pipeline/` (the primary workflow):

```
scripts/pipeline/
├── 00_prepare_processed_data.py
├── 01_compute_residual_migration.py
├── 02_compute_convergence.py
├── 03_compute_mortality_improvement.py
├── 04_process_demographic_data.py
├── 05_run_projections.py
├── 06_export_results.py
└── run_complete_pipeline.sh      # calls 00-06 in order
```

For top-level directory structure, consider prefixing key workflow directories:

```
cohort_projections/    # Python package (no number — it's a library)
config/                # Configuration (no number — it's infrastructure)
data/                  # Data directory (no number — it's storage)
scripts/
├── 1_data_fetch/      # Fetch raw data from external sources
├── 2_data_process/    # Process raw → intermediate data
├── 3_pipeline/        # Run projection pipeline (numbered steps)
├── 4_exports/         # Build output workbooks and CSVs
├── 5_analysis/        # Ad-hoc analysis and validation
└── utils/             # Shared script utilities
tests/
docs/
```

---

## Summary Grades

| Area | Grade | Notes |
|------|-------|-------|
| Python package architecture | **A** | Clean layers, no circular imports |
| Test coverage | **A** | 1,257 tests, 1.41x ratio |
| AI entry points (CLAUDE.md) | **A-** | Excellent layered system |
| Configuration management | **B+** | Solid but has duplicate loaders |
| Documentation quality | **B+** | Thorough but overwhelming volume |
| Directory organization | **B-** | Reasonable core, dominated by sdc_replication |
| Script/pipeline organization | **C+** | Good concept, incomplete implementation |
| Repo hygiene (dead code, clutter) | **C** | Organic growth without cleanup sweeps |
| Version management | **D** | Four conflicting version strings |

**Overall: B — Strong foundation with significant organizational debt from organic growth.**

---

*This summary synthesizes findings from 6 detailed audit reports in this directory. See individual reports for full details, file lists, and specific recommendations.*
