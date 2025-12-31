# Repository Hygiene Implementation Plan

---
| Attribute | Value |
|-----------|-------|
| **Created** | 2025-12-31 |
| **Status** | Approved for Implementation |
| **Version** | 1.0.0 |
| **Author** | Claude Code (Opus 4.5) |
| **Related ADRs** | [ADR-016](./adr/016-raw-data-management-strategy.md) |
---

## Executive Summary

This plan addresses repository hygiene and architecture concerns that have emerged as the project expanded from a focused production tool into a multi-purpose research platform. The plan prioritizes **auditability** by archiving rather than deleting deprecated documents, and ensures all file references and links are updated systematically.

### Goals

1. **Declutter root directory** - Archive 9 session/summary files to `docs/archive/`
2. **Consolidate documentation** - Merge CLAUDE.md and AGENTS.md into single authoritative source
3. **Standardize metadata** - Implement consistent metadata headers across all markdown files
4. **Fix broken references** - Address notebooks/ directory and path inconsistencies
5. **Improve discoverability** - Add metadata to module READMEs that currently lack it

### Key Principles

- **Archive, don't delete** - All deprecated files move to `docs/archive/` for auditability
- **Update all references** - Every file move includes systematic reference updates
- **Metadata headers** - Standardized format at top/bottom of all documentation files
- **Sub-agent parallelization** - Work divided into independent phases for parallel execution

---

## Phase Overview

| Phase | Description | Estimated Effort | Dependencies |
|-------|-------------|------------------|--------------|
| **Phase 1** | Create archive infrastructure | 15 min | None |
| **Phase 2** | Archive session/summary files | 30 min | Phase 1 |
| **Phase 3** | Consolidate CLAUDE.md + AGENTS.md | 45 min | Phase 2 |
| **Phase 4** | Add metadata to module READMEs | 30 min | None (parallel) |
| **Phase 5** | Fix notebooks/ reference and paths | 15 min | None (parallel) |
| **Phase 6** | Update ADR metadata consistency | 20 min | None (parallel) |
| **Phase 7** | Verification and commit | 15 min | Phases 1-6 |

**Total Estimated Time:** 2-3 hours

---

## Standardized Metadata Schema

### Schema A: Root-Level Documentation (README.md, AGENTS.md, etc.)

**Location:** Footer of document (after `---` separator)

```markdown
---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | YYYY-MM-DD |
| **Version** | X.Y.Z |
| **Status** | Current / Archived / In Progress / Needs Review |
| **Maintained By** | [Role/Name] |
| **Related ADRs** | [ADR-XXX](./docs/adr/XXX-title.md) |
```

### Schema B: ADR Files (docs/adr/NNN-*.md)

**Location:** Top of document (after title, before Context)

```markdown
# ADR-NNN: Title

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Date
YYYY-MM-DD

## Last Reviewed
YYYY-MM-DD

## Supersedes
[ADR-XXX if applicable, otherwise remove section]

## Context
...
```

### Schema C: Module READMEs (cohort_projections/*/README.md)

**Location:** After title, single line

```markdown
# Module Name

**Last Updated:** YYYY-MM-DD | **Status:** Current | **Related ADR:** [ADR-XXX](../../docs/adr/XXX-title.md)

[Content...]
```

### Schema D: Archived Files

**Location:** Top of document, before original content

```markdown
---
**ARCHIVED:** YYYY-MM-DD
**Reason:** [Brief explanation]
**Original Location:** /path/to/original/file.md
**Superseded By:** [New file if applicable]
---

[Original content below unchanged]
```

---

## Phase 1: Create Archive Infrastructure

### Objective
Create the archive directory structure with README explaining its purpose.

### Tasks

#### 1.1 Create Archive Directory
```bash
mkdir -p docs/archive
```

#### 1.2 Create Archive README
Create `docs/archive/README.md`:

```markdown
# Archived Documentation

---
| Attribute | Value |
|-----------|-------|
| **Created** | 2025-12-31 |
| **Status** | Active Archive |
| **Purpose** | Preserve deprecated documentation for auditability |
---

## Purpose

This directory contains archived documentation that is no longer actively used but preserved for:
- Historical reference and auditability
- Understanding project evolution
- Retrieving context from previous development phases

## Contents

Files are organized by archive date with metadata headers indicating:
- Original location
- Archive date and reason
- Superseding document (if applicable)

## Retrieval

All archived files retain their original content below the archive header.
To restore a file, copy it back to its original location and remove the archive header.

## Index

| File | Archived | Reason | Original Location |
|------|----------|--------|-------------------|
| SESSION_SUMMARY.md | 2025-12-31 | Session-specific notes | / |
| SESSION_CONTINUATION_SUMMARY.md | 2025-12-31 | Session-specific notes | / |
| BIGQUERY_DATA_SUMMARY.md | 2025-12-31 | Session-specific notes | / |
| IMPLEMENTATION_SUMMARY.md | 2025-12-31 | Implementation complete | / |
| FERTILITY_IMPLEMENTATION_SUMMARY.md | 2025-12-31 | Implementation complete | / |
| GEOGRAPHIC_MODULE_SUMMARY.md | 2025-12-31 | Implementation complete | / |
| OUTPUT_MODULE_IMPLEMENTATION.md | 2025-12-31 | Implementation complete | / |
| PARALLEL_IMPLEMENTATION_SUMMARY.md | 2025-12-31 | Implementation complete | / |
| PIPELINE_IMPLEMENTATION_SUMMARY.md | 2025-12-31 | Implementation complete | / |
```

### Verification
- [ ] `docs/archive/` directory exists
- [ ] `docs/archive/README.md` created with index table

---

## Phase 2: Archive Session/Summary Files

### Objective
Move 9 session/summary files to archive with proper metadata headers.

### Files to Archive

| File | Size | Reason |
|------|------|--------|
| SESSION_SUMMARY.md | 15.5 KB | Session-specific notes, no external refs |
| SESSION_CONTINUATION_SUMMARY.md | 19.6 KB | Session-specific notes, refs archived together |
| BIGQUERY_DATA_SUMMARY.md | 4.3 KB | Session-specific data findings |
| IMPLEMENTATION_SUMMARY.md | 10.2 KB | Implementation complete |
| FERTILITY_IMPLEMENTATION_SUMMARY.md | 13.1 KB | Implementation complete |
| GEOGRAPHIC_MODULE_SUMMARY.md | 14.8 KB | Implementation complete, refs SESSION_CONTINUATION |
| OUTPUT_MODULE_IMPLEMENTATION.md | 9.7 KB | Implementation complete |
| PARALLEL_IMPLEMENTATION_SUMMARY.md | 25.2 KB | Implementation complete |
| PIPELINE_IMPLEMENTATION_SUMMARY.md | 19.2 KB | Implementation complete |

### Tasks

#### 2.1 Add Archive Headers
For each file, prepend archive metadata:

```markdown
---
**ARCHIVED:** 2025-12-31
**Reason:** Session-specific development notes; implementation complete
**Original Location:** /SESSION_SUMMARY.md
**Superseded By:** DEVELOPMENT_TRACKER.md (for ongoing status)
---

[Original content unchanged below]
```

#### 2.2 Move Files
```bash
# For each file:
mv SESSION_SUMMARY.md docs/archive/
mv SESSION_CONTINUATION_SUMMARY.md docs/archive/
mv BIGQUERY_DATA_SUMMARY.md docs/archive/
mv IMPLEMENTATION_SUMMARY.md docs/archive/
mv FERTILITY_IMPLEMENTATION_SUMMARY.md docs/archive/
mv GEOGRAPHIC_MODULE_SUMMARY.md docs/archive/
mv OUTPUT_MODULE_IMPLEMENTATION.md docs/archive/
mv PARALLEL_IMPLEMENTATION_SUMMARY.md docs/archive/
mv PIPELINE_IMPLEMENTATION_SUMMARY.md docs/archive/
```

#### 2.3 Update Cross-References
Based on cross-reference analysis, update these files if they reference archived content:

| Source File | Reference | Action |
|-------------|-----------|--------|
| (None found) | - | No updates needed |

**Finding:** No external files reference the archived documents. They only reference each other, and all are archived together.

#### 2.4 Update DEVELOPMENT_TRACKER.md
Add note that it is now the canonical status document:

```markdown
## Note

This is the canonical project status tracker. Historical session notes have been
archived to [docs/archive/](./docs/archive/) for reference.
```

### Verification
- [ ] All 9 files have archive headers prepended
- [ ] All 9 files moved to `docs/archive/`
- [ ] `docs/archive/README.md` index updated
- [ ] DEVELOPMENT_TRACKER.md updated with canonical note
- [ ] `git status` shows expected changes

---

## Phase 3: Consolidate CLAUDE.md and AGENTS.md

### Objective
Make AGENTS.md the single authoritative document while preserving CLAUDE.md as a thin quick-reference wrapper.

### Analysis Summary

| Aspect | CLAUDE.md | AGENTS.md |
|--------|-----------|-----------|
| Version | 1.1.0 | 1.0.0 |
| Last Updated | 2025-12-29 | 2025-12-28 |
| Purpose | Quick reference | Canonical instructions |
| Unique Content | Test workflow, BigQuery | Agent roles, Quality standards, Demographics |
| Overlap | ~70% | ~70% |

### Tasks

#### 3.1 Merge Unique CLAUDE.md Content into AGENTS.md

**Add new Section 5.4: Test Workflow for AI Agents**

Insert after Section 5 (Quality Standards), before Section 6 (Data Conventions):

```markdown
### 5.4 Test Workflow for AI Agents

#### When Modifying Production Code

1. **Before changing code**: Run `pytest tests/ -v` to establish baseline
2. **After changing code**: Run tests again - failures indicate breaking changes
3. **If tests fail**: Either fix the code OR update the tests (if behavior change is intentional)
4. **Pre-commit enforces this**: Tests run automatically when committing changes to `cohort_projections/`

#### When to Update Tests

| Change Type | Test Action |
| ----------- | ----------- |
| Bug fix | Add test that reproduces the bug, then fix |
| New function | Add tests for the new function |
| Changed signature | Update all tests that call the function |
| Changed behavior | Update tests to expect new behavior |
| Removed function | Remove tests for that function |

#### Test Commands

```bash
pytest tests/ -v                           # All tests
pytest tests/test_core/ -v                 # Just core module tests
pytest tests/ -k "test_fertility" -v       # Tests matching pattern
pytest tests/ -x                           # Stop on first failure
pytest tests/ --tb=long                    # Detailed tracebacks
```

#### Finding Related Tests

```bash
# Find tests for a specific function
grep -r "function_name" tests/

# Find tests for a module
ls tests/test_core/test_fertility.py      # Tests for core/fertility.py
```

#### Test File Mapping

| Production Module | Test File |
| ----------------- | --------- |
| `cohort_projections/core/cohort_component.py` | `tests/test_core/test_cohort_component.py` |
| `cohort_projections/core/fertility.py` | `tests/test_core/test_fertility.py` |
| `cohort_projections/data/process/base_population.py` | `tests/test_data/test_base_population.py` |
| `cohort_projections/output/writers.py` | `tests/test_output/test_writers.py` |
```

**Add Section 10.5: BigQuery Integration**

Insert after Section 10 (Environment Setup):

```markdown
### 10.5 BigQuery Integration (Optional)

The project can optionally use Google BigQuery for Census/demographic data access.

```yaml
# In projection_config.yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  dataset_id: "demographic_data"
```

**Setup:** See [docs/BIGQUERY_SETUP.md](./docs/BIGQUERY_SETUP.md)
```

#### 3.2 Update AGENTS.md Metadata

Update footer:
```markdown
---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2025-12-31 |
| **Version** | 1.2.0 |
| **Status** | Current |
| **Maintained By** | Project Team |
| **Applies To** | All AI Agents |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial version |
| 1.1.0 | 2025-12-29 | Updated for uv package management |
| 1.2.0 | 2025-12-31 | Consolidated test workflow and BigQuery content from CLAUDE.md |
```

#### 3.3 Rewrite CLAUDE.md as Thin Wrapper

Replace entire CLAUDE.md content with:

```markdown
# CLAUDE.md

Quick reference for Claude Code. **For complete guidance, see [AGENTS.md](./AGENTS.md).**

---

## Quick Commands

### Testing
```bash
pytest                          # Run all tests
pytest --cov                    # With coverage
pytest tests/unit/              # Unit tests only
```

### Code Quality
```bash
pre-commit run --all-files      # All quality checks
ruff check cohort_projections/  # Linting
ruff check --fix cohort_projections/  # Auto-fix
mypy cohort_projections/        # Type checking
```

### Data Sync
```bash
./scripts/bisync.sh             # Sync data between computers
python scripts/fetch_data.py    # Fetch from sibling repos
```

### Run Projections
```bash
python scripts/projections/run_all_projections.py
```

---

## Session Workflow

### Starting
```bash
cd ~/workspace/demography/cohort_projections
direnv allow                     # First time only
git pull
./scripts/bisync.sh
uv sync
```

### After Changes
```bash
pytest
pre-commit run --all-files
git add . && git commit -m "..."
./scripts/bisync.sh
```

---

## Key Rules (Summary)

- **NEVER** hard-code file paths (use config)
- **NEVER** skip pre-commit hooks (`--no-verify`)
- **NEVER** commit data files to git
- **ALWAYS** activate virtual environment
- **ALWAYS** run tests before committing
- **ALWAYS** run bisync before switching computers

**Complete rules and workflow:** [AGENTS.md](./AGENTS.md)

---

## Documentation Quick Links

| Document | Purpose |
|----------|---------|
| [AGENTS.md](./AGENTS.md) | Complete AI agent guidance |
| [docs/adr/](./docs/adr/) | Architecture decisions |
| [config/projection_config.yaml](./config/projection_config.yaml) | Main configuration |
| [DEVELOPMENT_TRACKER.md](./DEVELOPMENT_TRACKER.md) | Current project status |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2025-12-31 |
| **Version** | 2.0.0 |
| **Status** | Current |
| **Note** | This is a quick-reference wrapper. See AGENTS.md for complete guidance. |
```

#### 3.4 Update References

| File | Current Reference | New Reference | Action |
|------|-------------------|---------------|--------|
| README.md | (none found) | Add link to both | Update project overview |
| DEVELOPMENT_TRACKER.md | CLAUDE.md only | CLAUDE.md and AGENTS.md | Update line mentioning repo hygiene |

### Verification
- [ ] AGENTS.md contains Section 5.4 (Test Workflow)
- [ ] AGENTS.md contains Section 10.5 (BigQuery)
- [ ] AGENTS.md version updated to 1.2.0
- [ ] CLAUDE.md rewritten as thin wrapper
- [ ] CLAUDE.md version set to 2.0.0 (breaking change)
- [ ] All cross-references updated
- [ ] Both files have standardized metadata footers

---

## Phase 4: Add Metadata to Module READMEs

### Objective
Add standardized metadata headers to module READMEs that currently lack them.

### Files to Update

| File | Current State | Related ADR |
|------|---------------|-------------|
| cohort_projections/core/README.md | No metadata | ADR-004 |
| cohort_projections/geographic/README.md | No metadata | ADR-010, ADR-013 |
| cohort_projections/output/README.md | No metadata | ADR-012, ADR-015 |
| cohort_projections/data/fetch/README.md | No metadata | ADR-006 |
| cohort_projections/data/process/README.md | No metadata | ADR-001, ADR-002, ADR-003 |

### Tasks

#### 4.1 Add Metadata Line After Title

For each file, insert after the `# Title` line:

**cohort_projections/core/README.md:**
```markdown
# Cohort Component Projection Engine

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADR:** [ADR-004](../../docs/adr/004-core-projection-engine-architecture.md)

[existing content...]
```

**cohort_projections/geographic/README.md:**
```markdown
# Geographic Module

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADRs:** [ADR-010](../../docs/adr/010-geographic-scope-and-fips-codes.md), [ADR-013](../../docs/adr/013-multi-geography-projection-design.md)

[existing content...]
```

**cohort_projections/output/README.md:**
```markdown
# Output Module

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADRs:** [ADR-012](../../docs/adr/012-output-export-formats.md), [ADR-015](../../docs/adr/015-output-visualization-design.md)

[existing content...]
```

**cohort_projections/data/fetch/README.md:**
```markdown
# Data Fetch Module

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADR:** [ADR-006](../../../docs/adr/006-data-pipeline-architecture.md)

[existing content...]
```

**cohort_projections/data/process/README.md:**
```markdown
# Data Processing Module

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADRs:** [ADR-001](../../../docs/adr/001-fertility-rate-processing.md), [ADR-002](../../../docs/adr/002-survival-rate-processing.md), [ADR-003](../../../docs/adr/003-migration-rate-processing.md)

[existing content...]
```

### Verification
- [ ] All 5 module READMEs have metadata line after title
- [ ] All ADR links are correct relative paths
- [ ] All links resolve to existing files

---

## Phase 5: Fix Notebooks Reference and Path Inconsistencies

### Objective
Address the non-existent notebooks/ directory reference and standardize path formats.

### Tasks

#### 5.1 Remove Notebooks Directory References

The notebooks/ directory is referenced in project structure diagrams but doesn't exist. Since this is a script-based project by design, remove the references.

**README.md** - Find and update project structure:
```markdown
# Change from:
├── notebooks/              # Jupyter notebooks

# To: (remove line entirely)
```

**CLAUDE.md** - Already being rewritten in Phase 3, ensure no notebooks reference.

#### 5.2 Standardize Path Format

Standardize all markdown links to use `./` prefix for consistency:

| Pattern | Standardize To |
|---------|----------------|
| `docs/adr/` | `./docs/adr/` |
| `AGENTS.md` | `./AGENTS.md` |
| `config/` | `./config/` |

**Files to check and update:**
- README.md
- AGENTS.md (after Phase 3 updates)
- docs/adr/README.md

### Verification
- [ ] No references to notebooks/ directory remain
- [ ] All root-level markdown links use `./` prefix
- [ ] `grep -r "notebooks/" *.md` returns no results

---

## Phase 6: Update ADR Metadata Consistency

### Objective
Ensure all ADRs follow consistent metadata format.

### Current State

| ADR | Format | Issue |
|-----|--------|-------|
| ADR-001 to 016 | `## Status` headers | Consistent |
| ADR-017 | `## Status` headers | Consistent |
| ADR-018 | `**Status:**` inline | **Inconsistent** |
| ADR-019 | Unknown | Check needed |

### Tasks

#### 6.1 Standardize to Header Format

Update ADR-018 to use header format:

```markdown
# Before:
**Status:** Proposed
**Date:** 2025-12-28

# After:
## Status
Proposed

## Date
2025-12-28
```

#### 6.2 Add Last Reviewed Field

Add to all ADRs after Date section:

```markdown
## Last Reviewed
2025-12-31
```

#### 6.3 Finalize ADR-018

ADR-018 has been "Proposed" since 2025-12-28. Either:
- Accept it (change Status to Accepted)
- Or document why it remains Proposed

### Verification
- [ ] All ADRs use `## Status` header format
- [ ] All ADRs have `## Last Reviewed` section
- [ ] ADR-018 status reviewed and updated

---

## Phase 7: Verification and Commit

### Objective
Verify all changes and commit with comprehensive message.

### Tasks

#### 7.1 Run Verification Checks

```bash
# Check for broken markdown links
grep -r "\[.*\](.*\.md)" *.md docs/*.md cohort_projections/*/README.md | \
  while read line; do
    file=$(echo "$line" | sed 's/:.*//')
    link=$(echo "$line" | grep -oP '\]\(\K[^)]+')
    dir=$(dirname "$file")
    if [[ ! -f "$dir/$link" ]] && [[ ! -f "$link" ]]; then
      echo "BROKEN: $file -> $link"
    fi
  done

# Check archived files have headers
for f in docs/archive/*.md; do
  if ! head -1 "$f" | grep -q "ARCHIVED"; then
    echo "MISSING HEADER: $f"
  fi
done

# Verify no notebooks references remain
grep -r "notebooks/" --include="*.md" . | grep -v docs/archive
```

#### 7.2 Run Tests

```bash
pytest tests/ -v
pre-commit run --all-files
```

#### 7.3 Commit Changes

```bash
git add -A

git commit -m "$(cat <<'EOF'
Implement repository hygiene improvements (Option A cleanup)

## Summary
- Archive 9 session/summary files to docs/archive/
- Consolidate CLAUDE.md and AGENTS.md documentation
- Add standardized metadata headers across all markdown files
- Fix broken notebooks/ directory reference
- Standardize path formats in markdown links

## Changes by Phase

### Phase 1: Archive Infrastructure
- Create docs/archive/ directory with README

### Phase 2: Archive Session Files
- Move 9 files: SESSION_SUMMARY.md, SESSION_CONTINUATION_SUMMARY.md,
  BIGQUERY_DATA_SUMMARY.md, IMPLEMENTATION_SUMMARY.md,
  FERTILITY_IMPLEMENTATION_SUMMARY.md, GEOGRAPHIC_MODULE_SUMMARY.md,
  OUTPUT_MODULE_IMPLEMENTATION.md, PARALLEL_IMPLEMENTATION_SUMMARY.md,
  PIPELINE_IMPLEMENTATION_SUMMARY.md
- Add archive headers to each file
- Update DEVELOPMENT_TRACKER.md as canonical status doc

### Phase 3: Documentation Consolidation
- Merge CLAUDE.md unique content (test workflow, BigQuery) into AGENTS.md
- Rewrite CLAUDE.md as thin quick-reference wrapper
- Update version: AGENTS.md 1.2.0, CLAUDE.md 2.0.0

### Phase 4: Module README Metadata
- Add metadata headers to 5 module READMEs
- Link each to relevant ADRs

### Phase 5: Reference Fixes
- Remove non-existent notebooks/ directory references
- Standardize path format to use ./ prefix

### Phase 6: ADR Consistency
- Standardize ADR-018 header format
- Add Last Reviewed field to all ADRs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### Verification
- [ ] All verification checks pass
- [ ] Tests pass
- [ ] Pre-commit hooks pass
- [ ] Commit successful

---

## Sub-Agent Assignment

For parallel execution, work can be distributed to sub-agents as follows:

### Agent 1: Archive Operations (Phases 1-2)
- Create archive directory structure
- Add archive headers to 9 files
- Move files to archive
- Update archive README index

### Agent 2: Documentation Consolidation (Phase 3)
- Merge unique CLAUDE.md content into AGENTS.md
- Update AGENTS.md metadata and version history
- Rewrite CLAUDE.md as thin wrapper
- Update cross-references

### Agent 3: Metadata Additions (Phase 4)
- Add metadata headers to 5 module READMEs
- Verify ADR link paths

### Agent 4: Cleanup and ADR Updates (Phases 5-6)
- Remove notebooks/ references
- Standardize path formats
- Update ADR-018 format
- Add Last Reviewed fields to ADRs

### Agent 5: Verification (Phase 7)
- Run all verification checks
- Run tests
- Prepare commit

---

## Rollback Plan

If issues are discovered after implementation:

### Restore Archived Files
```bash
# Restore a single file
cp docs/archive/SESSION_SUMMARY.md ./
# Remove archive header manually
```

### Restore CLAUDE.md
```bash
git checkout HEAD~1 -- CLAUDE.md
```

### Full Rollback
```bash
git revert HEAD
```

---

## Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Root directory decluttered | 5 markdown files remain (README, CLAUDE, AGENTS, DEVELOPMENT_TRACKER, PROJECT_STATUS) |
| Archive accessible | All 9 archived files in docs/archive/ with headers |
| Single source of truth | AGENTS.md is comprehensive, CLAUDE.md is thin wrapper |
| Metadata coverage | 100% of module READMEs have metadata |
| No broken links | All markdown links resolve |
| Tests pass | `pytest` exits 0 |
| Pre-commit passes | `pre-commit run --all-files` exits 0 |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2025-12-31 |
| **Version** | 1.0.0 |
| **Status** | Ready for Implementation |
