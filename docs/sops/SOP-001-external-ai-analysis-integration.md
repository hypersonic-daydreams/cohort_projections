# SOP-001: External AI Analysis Integration

## Document Information

| Field | Value |
|-------|-------|
| SOP ID | 001 |
| Status | Active |
| Created | 2026-01-01 |
| Last Updated | 2026-01-01 |
| Owner | Project Lead |

---

## 1. Purpose

This SOP defines the workflow for incorporating external AI analysis (e.g., from Google Gemini, ChatGPT, or other AI systems) into the project. It ensures that external critiques, methodology suggestions, and research insights are systematically evaluated, documented, and implemented in a traceable manner.

**Origin**: This SOP was derived from the ADR-019/020 workflow where external Gemini analysis of Census Bureau vintage transitions led to methodology enhancements in the demographic projection system.

---

## 2. Scope

### In Scope
- Receiving and evaluating external AI analysis/critique
- Creating Architecture Decision Records for methodology changes
- Exploratory analysis (Phase A) with one-time research scripts
- Implementation planning with phased agent breakdown
- Production implementation (Phase B) with modular code
- Test suite creation and validation
- Journal article or documentation updates
- Version control and artifact management

### Out of Scope
- Initial prompting/requesting external AI analysis (assumed to be done already)
- Production deployment beyond repository (CI/CD, servers)
- Peer review processes external to the project

---

## 3. Prerequisites

### Required Access
- [ ] Write access to the repository
- [ ] Ability to run pre-commit hooks
- [ ] LaTeX compilation tools (if updating journal article)

### Required Knowledge
- Familiarity with project structure (see [AGENTS.md](../../AGENTS.md))
- Understanding of ADR format (see [docs/adr/README.md](../adr/README.md))
- Python module development patterns used in this project

### Required Tools
- Python environment with project dependencies (`uv sync`)
- Git with pre-commit configured
- Claude Code or equivalent AI assistant for implementation

---

## 4. Procedure

### Phase 0: Intake and Triage

**Objective**: Evaluate external AI analysis and determine scope of integration

**Inputs**:
- External AI analysis document (markdown, PDF, or text)
- Context on what question/problem the analysis addresses

**Steps**:

1. **Review External Analysis**
   - Read the complete external AI output
   - Identify key claims, recommendations, and critiques
   - Note any data requirements or methodology changes suggested

2. **Assess Scope and Impact**
   - Determine if changes are:
     - **Minor**: Documentation-only updates
     - **Moderate**: Code changes within existing modules
     - **Major**: New modules, methodology changes, or architectural decisions
   - For **Major** scope, proceed with full SOP. For minor/moderate, adapt as needed.

3. **Create ADR Stub**
   - Assign next ADR number
   - Create initial ADR with Status: Proposed
   - Document the context and initial decision framing

**Outputs**:
- Initial ADR document (`docs/adr/0XX-title.md`)
- Scope assessment (Major/Moderate/Minor)

**Checkpoint**: ADR exists with Status: Proposed

---

### Phase A: Exploratory Analysis

**Objective**: Validate external AI claims through independent analysis

**Inputs**:
- External AI analysis
- Relevant project data
- Initial ADR

**Steps**:

1. **Create ADR Reports Directory**
   ```
   docs/adr/0XX-reports/
   ├── README.md           # Overview and agent coordination
   ├── agent1_*.py         # First analysis script (if needed)
   ├── agent2_*.py         # Second analysis script (if needed)
   ├── agent3_*.py         # Third analysis script (if needed)
   ├── figures/            # Generated visualizations
   └── data/               # Intermediate data outputs
   ```
   Use template: [templates/adr-report-structure.md](./templates/adr-report-structure.md)

2. **Develop Exploratory Scripts**
   - Create one-time analysis scripts to validate claims
   - Scripts should be self-contained and reproducible
   - Output JSON results and figures for later reference
   - **Important**: Mark scripts with deprecation notice in docstring:
     ```python
     """
     .. deprecated:: YYYY-MM-DD
         This is a **legacy Phase A research script** from the ADR-0XX investigation.
         Retained for reproducibility only. NOT production code.
     """
     ```

3. **Run Analysis and Document Findings**
   - Execute scripts and capture outputs
   - Document key findings in ADR reports directory
   - Update ADR with findings summary

4. **Add Linting Exemptions for Legacy Scripts**
   - Add file-specific exemptions to `pyproject.toml`:
     ```toml
     [tool.ruff.lint.per-file-ignores]
     "docs/adr/0XX-reports/agent*_analysis.py" = [
         "T201",    # Allow print statements
         "D",       # Ignore docstring requirements
         # Add other exemptions as needed
     ]
     ```

**Outputs**:
- Exploratory analysis scripts (deprecated after use)
- Analysis results (JSON, figures)
- Updated ADR with findings

**Checkpoint**: Phase A findings validate or refute external AI claims

---

### Phase B: Planning

**Objective**: Create implementation plan for validated methodology changes

**Inputs**:
- Phase A findings
- Updated ADR

**Steps**:

1. **Create Planning Synthesis Document**
   - Location: `docs/adr/0XX-reports/PLANNING_SYNTHESIS.md`
   - Use template: [templates/planning-synthesis.md](./templates/planning-synthesis.md)
   - Define phased implementation with agent breakdown:
     - **B0**: Infrastructure (dependencies, file structure)
     - **B1-B4**: Core implementation modules
     - **B5**: Documentation updates
     - **B6**: Test suite

2. **Create Individual Agent Plans**
   - Location: `docs/adr/0XX-reports/phase_b_plans/AGENT_BN_PLAN.md`
   - Each plan should specify:
     - Current state assessment
     - Files to create/modify
     - Function signatures
     - Dependencies on other agents
     - Estimated complexity

3. **Review and Approve Plans**
   - Human review of planning documents
   - Resolve any ambiguities or scope questions
   - Mark ADR Status: Accepted

**Outputs**:
- PLANNING_SYNTHESIS.md
- Individual AGENT_BN_PLAN.md files
- ADR with Status: Accepted

**Checkpoint**: Plans approved by human reviewer

---

### Phase B: Implementation

**Objective**: Implement planned changes as production-quality code

**Inputs**:
- Approved planning documents
- Phase A exploratory scripts (for reference)

**Steps**:

1. **Create Module Packages**
   - Location: `sdc_2024_replication/scripts/statistical_analysis/module_BN_*/`
   - Use template: [templates/module-package.md](./templates/module-package.md)
   - Structure:
     ```
     module_BN_name/
     ├── __init__.py          # Public API exports
     ├── component1.py        # Core functionality
     ├── component2.py        # Supporting functionality
     └── conftest.py          # pytest collection exclusion (if needed)
     ```

2. **Implement Core Logic**
   - Follow existing code patterns in the project
   - Use type hints throughout
   - Write docstrings for public functions
   - Avoid `test_*` function names in source code (pytest collection conflict)

3. **Create Runner Script**
   - Location: `sdc_2024_replication/scripts/statistical_analysis/module_BN_*.py`
   - Main entry point that uses the package
   - Outputs results to `results/` directory

4. **Update Journal Article (if applicable)**
   - Location: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/`
   - Add new section or update existing section
   - Reference B1/B2 results with specific numbers

5. **Run Pre-commit and Fix Issues**
   ```bash
   pre-commit run --all-files
   ```
   - Fix any linting errors
   - Add exemptions only for legacy Phase A scripts

**Outputs**:
- Production module packages
- Runner scripts
- Updated journal article sections

**Checkpoint**: `pre-commit run --all-files` passes

---

### Phase B: Testing (B6)

**Objective**: Create comprehensive test suite for new modules

**Inputs**:
- Implemented modules
- Planning documents with test specifications

**Steps**:

1. **Create Test Directory Structure**
   ```
   tests/test_statistical/
   ├── __init__.py
   ├── conftest.py              # Shared fixtures
   ├── test_module_b1.py        # Tests for B1
   ├── test_module_b2.py        # Tests for B2
   └── test_module_b4.py        # Tests for B4 (if applicable)
   ```

2. **Implement Test Fixtures**
   - Add shared fixtures to `tests/conftest.py`
   - Create synthetic data with known properties
   - Use `@pytest.fixture` decorator

3. **Write Unit Tests**
   - Test all public functions
   - Include edge cases
   - Use `@pytest.mark.parametrize` for variations

4. **Add Test-Specific Linting Exemptions**
   ```toml
   "tests/**/*.py" = [
       "S101",    # Allow assert statements
       "PLR2004", # Allow magic values
       "D",       # Ignore docstring requirements
   ]
   ```

5. **Run Test Suite**
   ```bash
   pytest tests/test_statistical/ -v
   ```

**Outputs**:
- Test files with passing tests
- Updated `pyproject.toml` with test exemptions

**Checkpoint**: All tests pass

---

### Phase C: Finalization

**Objective**: Generate final artifacts and commit changes

**Inputs**:
- All Phase B outputs
- Passing test suite

**Steps**:

1. **Compile Journal Article PDF**
   ```bash
   cd sdc_2024_replication/scripts/statistical_analysis/journal_article
   ./compile.sh --clean
   ```

2. **Save Versioned PDF**
   - Naming convention: `article-{MAJOR}.{MINOR}.{PATCH}-{STATUS}_{TIMESTAMP}.pdf`
   - Update `output/CURRENT_VERSION.txt`
   - Update `output/VERSIONS.md`

3. **Run Final Validation**
   ```bash
   pre-commit run --all-files
   pytest
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "ADR-0XX: [Brief description of changes]"
   ```

5. **Push and Sync**
   ```bash
   git push
   ./scripts/bisync.sh  # If using rclone bisync
   ```

**Outputs**:
- Versioned PDF
- Git commit with all changes
- Synced to remote

**Checkpoint**: Changes pushed and synced

---

## 5. Artifacts

### Created Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| ADR | `docs/adr/0XX-*.md` | Document decision and rationale |
| ADR Reports | `docs/adr/0XX-reports/` | Exploratory analysis and planning |
| Production Modules | `sdc_2024_replication/scripts/statistical_analysis/module_BN_*/` | Reusable code |
| Test Suite | `tests/test_statistical/` | Validation |
| Journal Article | `sdc_2024_replication/.../journal_article/output/` | Publication |

### Templates Used

| Template | Location | When Used |
|----------|----------|-----------|
| ADR Report Structure | `docs/sops/templates/adr-report-structure.md` | Phase A, Step 1 |
| Planning Synthesis | `docs/sops/templates/planning-synthesis.md` | Phase B Planning, Step 1 |
| Module Package | `docs/sops/templates/module-package.md` | Phase B Implementation, Step 1 |

---

## 6. Quality Gates

| Gate | Criteria | Responsible |
|------|----------|-------------|
| Phase A Complete | Findings documented, claims validated | AI Agent |
| Plans Approved | Human review of PLANNING_SYNTHESIS.md | Human |
| Pre-commit Pass | All hooks pass | AI Agent |
| Tests Pass | pytest returns 0 | AI Agent |
| PDF Generated | Article compiles without errors | AI Agent |

---

## 7. Troubleshooting

### Pytest Collects Source Functions as Tests

**Symptom**: pytest discovers `test_*` functions from source code

**Cause**: Function names starting with `test_` in non-test files

**Resolution**:
1. Rename functions to `run_*_test` or similar
2. Add `conftest.py` with `collect_ignore_glob = ["*.py"]` to source directory
3. Update all imports and usages

### Pre-commit Fails on Legacy Scripts

**Symptom**: Ruff/mypy errors on Phase A exploratory scripts

**Cause**: Legacy scripts don't follow strict linting rules

**Resolution**: Add file-specific exemptions to `pyproject.toml`:
```toml
"docs/adr/0XX-reports/agent*_analysis.py" = ["T201", "D", ...]
```

### LaTeX Compilation Fails

**Symptom**: `./compile.sh` returns non-zero

**Cause**: Missing references, undefined citations, or syntax errors

**Resolution**:
1. Check `main.log` for specific errors
2. Ensure bibliography entries exist in `references.bib`
3. Run compilation multiple times for cross-references

---

## 8. Related Documentation

- [ADR-020](../adr/020-extended-time-series-methodology-analysis.md) - Example of this SOP in action
- [AGENTS.md](../../AGENTS.md) - AI agent governance rules
- [docs/adr/README.md](../adr/README.md) - ADR creation guide

---

## 9. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-01 | 1.0 | Claude Opus 4.5 | Initial version derived from ADR-020 workflow |

---

*SOP Version: 1.0*
