# Package Extraction Plan

**Created**: 2026-01-02
**Parent ADR**: [ADR-023: Package Extraction Strategy](../adrs/023-package-extraction-strategy.md)
**Tracker**: [PACKAGE_EXTRACTION_TRACKER.md](./PACKAGE_EXTRACTION_TRACKER.md)

---

## Overview

This document defines the strategy, agent assignments, and validation criteria for extracting three packages from `cohort_projections` into standalone repositories in `~/workspace/libs/`.

### Packages

| Package | Repository | Module | Purpose |
|---------|------------|--------|---------|
| `project-utils` | `project_utils` | `project_utils` | Config loading + logging utilities |
| `evidence-review` | `evidence_review` | `evidence_review` | Citation audit, claims, argumentation |
| `codebase-catalog` | `codebase_catalog` | `codebase_catalog` | Codebase scanning, inventory, hooks |

### Naming Convention

Following workspace patterns (from `commerce_dataverse_reference`):
- **Repository name**: underscores (`project_utils`)
- **Package name** (pyproject.toml): hyphens (`project-utils`)
- **Python module**: underscores (`project_utils`)
- **CLI commands**: hyphens (`check-citations`)

---

## Phase Structure

```
Phase 1a: project_utils
    └── Integration Verification ──┐
                                   │
Phase 1b: evidence_review          │
    └── Integration Verification ──┼── (parallel possible after 1a complete)
                                   │
Phase 2: codebase_catalog          │
    └── Integration Verification ──┘

Phase 3: Workspace Documentation
```

---

## Phase 1a: project_utils

**Duration**: ~1.5 hours
**Agents**: 4 (sequential with integration at end)

### Agent 1a-1: Scaffold Repository

**Task**: Create the repository structure following REPOSITORY_STANDARDS.md

**Actions**:
1. Create directory: `~/workspace/libs/project_utils`
2. Run `uv init` to initialize
3. Create `.envrc`:
   ```bash
   dotenv_if_exists .env
   export VIRTUAL_ENV="$PWD/.venv"
   PATH_add "$VIRTUAL_ENV/bin"
   export PROJECT_ROOT="$PWD"
   ```
4. Create `.gitignore` (include `.venv/`, `.direnv/`, `.env`)
5. Initialize git repository
6. Create GitHub repository and push
7. Run `uv sync` to verify

**Outputs**: Initialized repository with pyproject.toml, .envrc, .gitignore

### Agent 1a-2: Migrate Code

**Task**: Move config_loader.py and logger.py to new package

**Source Files**:
- `cohort_projections/utils/config_loader.py`
- `cohort_projections/utils/logger.py`

**Actions**:
1. Create `project_utils/` module directory
2. Copy `config_loader.py` → `project_utils/config.py`
3. Copy `logger.py` → `project_utils/logging.py`
4. Create `project_utils/__init__.py` with exports:
   ```python
   from project_utils.config import ConfigLoader
   from project_utils.logging import setup_logger, get_logger

   __all__ = ["ConfigLoader", "setup_logger", "get_logger"]
   ```
5. Create `project_utils/py.typed` (empty file for PEP 561)
6. Update pyproject.toml with `pyyaml` dependency
7. Verify imports work: `uv run python -c "from project_utils import ConfigLoader"`

**Outputs**: Working Python module

### Agent 1a-3: Tests and Documentation

**Task**: Create tests and documentation

**Actions**:
1. Create `tests/` directory structure
2. Create `tests/fixtures/sample_config.yaml`
3. Write `tests/test_config.py`
4. Write `tests/test_logging.py`
5. Run `uv run pytest` to verify
6. Create `README.md` with usage examples
7. Create `AGENTS.md` for AI agent guidance

**Outputs**: Passing tests, documentation

### Agent 1a-INT: Integration Verification

**Task**: Wire cohort_projections to use new package

**Actions**:
1. Add editable dependency to cohort_projections:
   ```bash
   cd ~/workspace/demography/cohort_projections
   uv add --editable ../../libs/project_utils
   ```
2. Find all imports of old modules:
   ```bash
   grep -r "from cohort_projections.utils.config_loader" .
   grep -r "from cohort_projections.utils.logger" .
   ```
3. Update all imports to use new package
4. Find all documentation references to old paths and update
5. Run `uv run pytest` to verify nothing broke
6. Run `pre-commit run --all-files` to verify hooks pass
7. **DO NOT** remove old source files yet (defer to Phase 3)

**Outputs**: cohort_projections using new package, all tests passing

---

## Phase 1b: evidence_review

**Duration**: ~3.5 hours
**Agents**: 6 (scaffold first, then 3 parallel migrations, then integration)

### Agent 1b-1: Scaffold Repository

**Task**: Create repository structure

**Actions**: Same as 1a-1, but for `~/workspace/libs/evidence_review`

### Agent 1b-2: Migrate Citations Module

**Task**: Move citation audit code

**Source Files** (from `sdc_2024_replication/.../claim_review/v3_phase3/citation_audit/`):
- `check_citations.py`
- `citation_entry_schema.json`
- `citation_fixes_schema.json`

**Actions**:
1. Create `evidence_review/citations/` directory
2. Refactor `check_citations.py` into modular structure:
   - `auditor.py` - Core audit engine
   - `parsers.py` - LaTeX/BibTeX parsing
   - `apa7.py` - APA 7th Edition rules
   - `reporters.py` - Multi-format reporting
3. Move schemas to `evidence_review/citations/schemas/`
4. Create `__init__.py` with exports
5. Verify imports work

### Agent 1b-3: Migrate Claims Module

**Task**: Move claims extraction code

**Source Files** (from `sdc_2024_replication/.../claim_review/v3_phase3/claims/`):
- `extract_claims.py`
- `build_section_claims.py`
- `qa_claims.py`
- `claim_schema.json`
- `claim_guidelines.md`

**Actions**:
1. Create `evidence_review/claims/` directory
2. Move and organize source files
3. Move schema to `evidence_review/claims/schemas/`
4. Create `__init__.py` with exports
5. Verify imports work

### Agent 1b-4: Migrate Argumentation Module

**Task**: Move argument mapping code

**Source Files** (from `sdc_2024_replication/.../claim_review/v3_phase3/argument_map/`):
- `generate_argument_graphs.py`
- `map_section_arguments.py`
- `build_viewer.py`
- `argument_schema.json`
- `argument_guidelines.md`
- `ARGUMENTATION_METHOD.md`

**Actions**:
1. Create `evidence_review/argumentation/` directory
2. Move and organize source files
3. Move schema to `evidence_review/argumentation/schemas/`
4. Create `__init__.py` with exports
5. Verify imports work

### Agent 1b-5: Integration and Documentation

**Task**: Create CLI, tests, and documentation

**Actions**:
1. Create `evidence_review/cli/` with entry points
2. Update pyproject.toml with CLI scripts
3. Port existing tests from `tests/test_tools/test_citation_audit.py`
4. Create additional unit tests
5. Create integration tests
6. Create `README.md` and `AGENTS.md`
7. Run all tests

### Agent 1b-INT: Integration Verification

**Task**: Wire cohort_projections to use new package

**Actions**:
1. Add editable dependency:
   ```bash
   uv add --editable ../../libs/evidence_review
   ```
2. Find all imports/references to old paths
3. Update imports in any scripts that use these modules
4. Update documentation references
5. Run tests and pre-commit
6. **DO NOT** remove old source files yet

---

## Phase 2: codebase_catalog

**Duration**: ~2.5 hours
**Agents**: 5 (scaffold, then 2 parallel migrations, then integration)

### Agent 2-1: Scaffold Repository

**Task**: Create repository structure

**Actions**: Same as 1a-1, but for `~/workspace/libs/codebase_catalog`

### Agent 2-2: Migrate Scanner and Inventory

**Task**: Move scanning and inventory generation code

**Source Files**:
- `scripts/intelligence/scan_repository.py`
- `scripts/intelligence/generate_inventory_docs.py`
- `scripts/intelligence/archive_manager.py`

**Actions**:
1. Create `codebase_catalog/scanner/` directory
2. Create `codebase_catalog/inventory/` directory
3. Create `codebase_catalog/archive/` directory
4. Refactor into modular structure
5. Create `__init__.py` files with exports

### Agent 2-3: Migrate Hooks

**Task**: Move pre-commit hooks

**Source Files**:
- `scripts/hooks/update_code_inventory.py`
- `scripts/hooks/check_data_manifest.py`

**Actions**:
1. Create `codebase_catalog/hooks/` directory
2. Move and organize hook code
3. Extract governance parsing into `codebase_catalog/governance/`
4. Create `__init__.py` files

### Agent 2-4: Integration and Documentation

**Task**: Create CLI, tests, and documentation

**Actions**:
1. Create `codebase_catalog/cli/` with entry points
2. Update pyproject.toml with CLI scripts
3. Create tests
4. Create `README.md` and `AGENTS.md`
5. Run all tests

### Agent 2-INT: Integration Verification

**Task**: Wire cohort_projections to use new package

**Actions**:
1. Add editable dependency
2. Update `.pre-commit-config.yaml` to use new CLI commands
3. Find and update all imports/references
4. Run tests and pre-commit hooks
5. **DO NOT** remove old source files yet

---

## Phase 3: Workspace Documentation and Cleanup

**Duration**: ~30 minutes
**Agents**: 2

### Agent 3-1: Update REPOSITORY_INVENTORY.md

**Task**: Add new packages to workspace inventory

**Actions**:
1. Add entries for all three packages to REPOSITORY_INVENTORY.md
2. Update summary statistics
3. Update category listings

### Agent 3-2: Final Cleanup

**Task**: Remove extracted source files from cohort_projections

**Actions**:
1. Remove `cohort_projections/utils/config_loader.py`
2. Remove `cohort_projections/utils/logger.py`
3. Remove extracted files from `sdc_2024_replication/.../claim_review/v3_phase3/`
4. Remove extracted files from `scripts/intelligence/` and `scripts/hooks/`
5. Run full test suite one final time
6. Run pre-commit hooks
7. Commit all changes

---

## Validation Gates

### Per-Phase Validation

Before proceeding to next phase, verify:

- [ ] `uv sync` completes without errors
- [ ] `uv run pytest` passes all tests
- [ ] `uv run ruff check .` passes
- [ ] Package is importable: `python -c "import module_name"`
- [ ] CLI commands work (if applicable)
- [ ] cohort_projections tests still pass
- [ ] cohort_projections pre-commit hooks pass

### Final Validation

- [ ] All three packages installed in cohort_projections
- [ ] Full test suite passes
- [ ] Pre-commit hooks pass
- [ ] No references to old file paths remain
- [ ] REPOSITORY_INVENTORY.md updated
- [ ] All GitHub repositories created and pushed

---

## Agent Prompt Templates

### Scaffold Agent

```
You are creating a new Python package repository at ~/workspace/libs/[name].

Follow REPOSITORY_STANDARDS.md from ~/workspace/:
1. Create directory and initialize with uv init
2. Create .envrc with standard template
3. Create .gitignore (include .venv/, .direnv/, .env)
4. Initialize git and create GitHub repository
5. Run uv sync to verify

Package details:
- Repository: [name]
- Package name: [package-name] (in pyproject.toml)
- Module: [module_name]
- Description: [description]

Do NOT create README.md or tests yet.
```

### Migration Agent

```
You are migrating code from cohort_projections to a new package.

Source: [source_path]
Target: ~/workspace/libs/[package]/[module]/[submodule]/

Tasks:
1. Copy source files to target
2. Refactor into clean module structure
3. Update internal imports
4. Create __init__.py with public exports
5. Verify module imports correctly

Preserve all functionality. This is a move, not a rewrite.
Do not add new features or change behavior.
```

### Integration Verification Agent

```
You are verifying cohort_projections works with the extracted [package] package.

Tasks:
1. Add editable dependency: uv add --editable ../../libs/[package]
2. Search for ALL imports of old module paths
3. Update all imports to use new package
4. Search for ALL documentation references to old paths
5. Update documentation references
6. Run: uv run pytest
7. Run: pre-commit run --all-files
8. Report any failures

CRITICAL: Do NOT remove old source files. That happens in Phase 3.
CRITICAL: All tests must pass before marking complete.
```

---

## Coordination Notes

### Parallel Execution

- **Phase 1a**: Sequential (1a-1 → 1a-2 → 1a-3 → 1a-INT)
- **Phase 1b**: 1b-1 first, then 1b-2/1b-3/1b-4 in parallel, then 1b-5 → 1b-INT
- **Phase 2**: 2-1 first, then 2-2/2-3 in parallel, then 2-4 → 2-INT
- **Phase 3**: Sequential (3-1 → 3-2)

### Dependencies

- Phase 1b can start after Phase 1a-INT completes
- Phase 2 can start after Phase 1b-INT completes
- Phase 3 requires all prior phases complete

### GitHub Repository Creation

For each package, create GitHub repository:
```bash
gh repo create hypersonic-daydreams/[repo_name] --private --source=. --push
```

---

## References

- [REPOSITORY_STANDARDS.md](file:///home/nhaarstad/workspace/REPOSITORY_STANDARDS.md)
- [REPOSITORY_INVENTORY.md](file:///home/nhaarstad/workspace/REPOSITORY_INVENTORY.md)
- [ADR-023](../adrs/023-package-extraction-strategy.md)
- [ADR-023a](../adrs/023a-evidence-review-package.md)
- [ADR-023b](../adrs/023b-project-utils-package.md)
- [ADR-023c](../adrs/023c-codebase-catalog-package.md)
