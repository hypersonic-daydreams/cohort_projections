# 06 - AI Navigability and Embeddings Assessment

**Date:** 2026-02-26
**Reviewer:** Claude Code (Opus 4.6)
**Scope:** AI agent navigability, codebase search strategy, embeddings evaluation

---

## Executive Summary

This repository is **exceptionally well-structured for AI agent navigation** -- among the best-instrumented projects I have analyzed. The layered documentation system (CLAUDE.md -> AGENTS.md -> guides -> ADRs) provides a clear on-ramp. However, the sheer volume of documentation (78,000 lines of markdown, 97 ADRs, 29 review docs) creates a secondary problem: **the repo has more documentation than code**, and an AI agent reading broadly would consume context window budget on historical records rather than current code.

The core codebase (39 Python files, ~18,000 lines) is **small enough that embeddings are not necessary**. Better organization and selective pruning of documentation would yield greater returns than any vector search infrastructure.

**Bottom line:** This is a documentation-rich, code-lean project. The investment should go into pruning and focusing documentation, not adding search infrastructure.

---

## Part 1: AI Agent Navigability Assessment

### 1.1 Entry Point Quality (CLAUDE.md + AGENTS.md)

**CLAUDE.md: Grade A**

The CLAUDE.md file is an exemplary quick-reference card at 92 lines / 2,459 characters (~600 tokens). It provides:
- Exact shell commands for testing, linting, data sync, and running projections
- A clear session workflow (start and end)
- A concise "Key Rules" section with NEVER/ALWAYS constraints
- A documentation index table pointing to deeper resources
- An explicit pointer to AGENTS.md for complete guidance

The one-page design is ideal for the system prompt injection that Claude Code performs automatically. An AI agent can absorb this entire file in seconds and immediately know how to run tests, check code quality, and follow the workflow.

**AGENTS.md: Grade A-**

At 342 lines / 14,174 characters (~3,500 tokens), AGENTS.md provides a thorough but manageable deep reference. Strong points:
- Clear project identity ("North Dakota Population Projection System")
- Explicit autonomy tiers (Tier 1: just do it, Tier 2: do + document, Tier 3: stop and ask)
- Demographic domain context (race categories, geographic hierarchy, age cohorts)
- Shared Census data archive documentation with exact paths
- Documentation index linking to all relevant guides

Minor weaknesses:
- Section 11 ("Repository Intelligence") references a PostgreSQL-backed system that may confuse agents into thinking they need DB access for routine tasks
- The shared Census data archive section (Section 6) is detailed but could cause confusion for agents that do not need to interact with raw Census data
- The version history table at the bottom is noise for an AI agent

**MEMORY.md: Grade A**

The 57-line persistent memory file is well-curated with genuinely useful operational knowledge:
- Critical warning about pre-commit taking 3+ minutes (avoids a common time sink)
- Pre-existing test failures to ignore
- Current projection results for validation
- ADR status summary
- Data vintages

This is exactly the kind of "learned lessons" file that prevents an agent from rediscovering known issues.

### 1.2 Time to Understanding

**How many files must an AI read before being productive?**

| Level | Files to Read | Estimated Tokens | Time to Productive |
|-------|--------------|------------------|--------------------|
| Run tests/lint | 1 (CLAUDE.md) | ~600 | Immediate |
| Make code changes | 2 (CLAUDE.md + AGENTS.md) | ~4,100 | 1 minute |
| Understand architecture | +3 (core/ARCHITECTURE.md + core/README.md + config) | ~7,500 | 3 minutes |
| Modify projection logic | +2 (relevant module + its test) | ~10,000 | 5 minutes |
| Understand a specific ADR | +1 (the specific ADR) | ~11,000 | 6 minutes |

**Verdict: Excellent.** An AI agent can be productive in 1-2 file reads. The layered documentation structure means agents only go deeper when needed, rather than being forced to read everything up front.

### 1.3 Codebase Maps and Architecture Documentation

The repository has strong architectural documentation at multiple levels:

| Document | Content | Quality |
|----------|---------|---------|
| `cohort_projections/core/ARCHITECTURE.md` | ASCII flow diagrams, data flow, component APIs | Excellent |
| `cohort_projections/core/README.md` | 420 lines with math formulas, usage examples, data formats | Excellent |
| `cohort_projections/core/QUICKSTART.md` | Minimal quick-start | Good |
| `cohort_projections/data/process/README.md` | 2,247-line comprehensive module guide | Excessive |
| `cohort_projections/geographic/README.md` | Geographic module overview | Good |
| `cohort_projections/output/README.md` | Output module overview | Good |
| `cohort_projections/data/fetch/README.md` | Data fetching modules | Good |

The `ARCHITECTURE.md` file in `core/` is particularly effective -- its ASCII diagrams are parseable by AI agents and give immediate visual understanding of the projection pipeline.

**Gap identified:** There is no single top-level `CODEBASE_MAP.md` that maps the entire project. The README.md at the project root is a generic project description (3,419 chars) rather than a structural guide. An agent navigating from AGENTS.md must follow multiple links to piece together the full picture.

### 1.4 Docstring Quality

**Core library docstrings: Grade A**

Every public function in the core library has Google-style docstrings with:
- Purpose description
- `Args:` section with column-level DataFrame schema documentation
- `Returns:` section with format details
- `Raises:` section where applicable
- `Notes:` with domain context (e.g., "Domestic migration is typically from IRS county-to-county flows")

Example from `migration.py`:
```python
def apply_migration(
    population: pd.DataFrame,
    migration_rates: pd.DataFrame,
    year: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Apply net migration to population cohorts.

    Net migration is the balance of in-migration and out-migration.
    ...
    Args:
        population: DataFrame with columns [year, age, sex, race, population]
                   Should be post-survival population
        migration_rates: DataFrame with columns [age, sex, race, net_migration]
                        or [age, sex, race, migration_rate]
    ...
    Notes:
        - Net migration can be specified as:
          1. Absolute numbers (net_migration column)
          2. Rates (migration_rate column, applied to population)
        - Domestic migration is typically from IRS county-to-county flows
    """
```

**Data processing scripts: Grade A+**

The data processing scripts (e.g., `build_nd_fertility_rates.py`) follow SOP-002 and contain extraordinary docstrings with:
- Created date, ADR reference, author
- Multi-paragraph purpose section
- Numbered methodology steps
- Key design decisions with trade-offs
- Validation results with actual numbers
- Full input/output file paths with provenance

These are arguably the best-documented scripts I have encountered in any project. They function as self-contained methodology documents.

**Utility modules: Grade B+**

The utility modules (`config_loader.py`, `demographic_utils.py`) have adequate but less detailed docstrings. This is appropriate -- they are less domain-specific.

### 1.5 File Name Self-Documentation

**Grade: A**

File names are descriptive and follow consistent conventions:

| Pattern | Example | Assessment |
|---------|---------|------------|
| Core modules | `cohort_component.py`, `fertility.py`, `mortality.py`, `migration.py` | Self-explanatory |
| Data processing | `residual_migration.py`, `convergence_interpolation.py`, `fertility_rates.py` | Clear purpose |
| Pipeline scripts | `00_prepare_processed_data.py`, `01_compute_residual_migration.py` | Numbered sequence |
| Data scripts | `build_nd_fertility_rates.py`, `fetch_census_gq_data.py` | Verb-noun pattern |
| Tests | `test_residual_migration.py`, `test_convergence_interpolation.py` | Mirror source files |

The `scripts/pipeline/` directory uses numbered prefixes (00, 01, 01b, 01c, 02, 03) that clearly communicate execution order. The slight inconsistency (01, 01b, 01c) suggests organic growth but remains navigable.

### 1.6 Documentation Volume: Context Window Overflow Risk

This is the repository's most significant navigability concern. The numbers:

| Category | Files | Lines | Est. Tokens |
|----------|-------|-------|-------------|
| ADRs | 97 | 32,308 | ~40,000 |
| Review documents | 29 | 10,949 | ~14,000 |
| Module READMEs | 8 | ~5,500 | ~7,000 |
| Guides | 6 | ~2,500 | ~3,000 |
| Root-level MD | 7 | ~7,000 | ~9,000 |
| Data READMEs | 27 | ~4,000 | ~5,000 |
| DEVELOPMENT_TRACKER.md | 1 | 925 | ~4,500 |
| REPOSITORY_INVENTORY.md | 1 | 1,960 | ~25,000 |
| **Total documentation** | **~235** | **~78,000** | **~100,000** |
| | | | |
| Core library Python | 39 | 17,759 | ~39,000 |
| Scripts Python | 51 | 20,531 | ~45,000 |
| Tests Python | 56 | 24,976 | ~57,000 |
| **Total project code** | **146** | **63,266** | **~141,000** |

**The documentation-to-code ratio is approximately 0.7:1 by token count (not counting the SDC replication).** This is unusually high. For comparison, most projects have a ratio of 0.05:1 to 0.15:1.

Specific overflow risks:
- **REPOSITORY_INVENTORY.md** (291 KB, ~25,000 tokens): Auto-generated from PostgreSQL, includes 1,014 entries for `.ruff_cache`, `.mypy_cache`, and `.venv` files. If an agent reads this file, it wastes enormous context on cache file listings.
- **DEVELOPMENT_TRACKER.md** (45 KB, ~4,500 tokens): Contains extensive historical sprint notes. The current status is in the first ~80 lines; the remaining 845 lines are historical.
- **97 ADRs** (32,308 lines total): Most are historical records of completed decisions. Only 3-5 are "live" at any time. An agent searching for relevant context could easily consume 20,000+ tokens reading through ADR history.
- **data/process/README.md** (2,247 lines): This single module README is longer than most of the module's actual source files. It contains detailed processing documentation that overlaps significantly with the SOP-002-compliant docstrings already in the source files.

**Auto-loaded context is well-managed.** The CLAUDE.md + AGENTS.md + MEMORY.md total is only ~5,200 tokens, which is modest and appropriate. The risk is in on-demand reads, not system prompt bloat.

### 1.7 Summary Assessment

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Entry points (CLAUDE.md/AGENTS.md) | A | Layered, concise, actionable |
| Architecture docs | A | ASCII diagrams, clear data flow |
| Docstring quality | A | Google-style with domain context |
| File naming | A | Self-documenting, consistent |
| Time to understanding | A | Productive in 1-2 file reads |
| Documentation volume management | C+ | Too much history, redundancy, noise |
| Cross-referencing | B+ | Good links but some broken refs |

---

## Part 2: Embeddings and Search Strategy

### 2.1 Does This Repository Need Embeddings?

**No.** Here is why.

#### Codebase Size Analysis

The core project (excluding SDC replication, `.venv`, and libs) consists of:

- **39 core library files** totaling 17,759 lines
- **51 script files** totaling 20,531 lines
- **56 test files** totaling 24,976 lines
- **~235 markdown documentation files** totaling ~78,000 lines

The total project code is **~63,000 lines of Python**. This is a **small-to-medium** project by industry standards:

| Category | Line Count | Classification |
|----------|-----------|---------------|
| Tiny | < 5,000 | Single-module tools |
| Small | 5,000 - 50,000 | Typical application |
| **This project** | **~63,000** | **Small-medium** |
| Medium | 50,000 - 200,000 | Mature application |
| Large | 200,000 - 1,000,000 | Framework / platform |
| Very Large | > 1,000,000 | OS kernel, browser engine |

Embeddings become valuable when:
1. The codebase is too large to search with grep/glob in reasonable time (>200K lines)
2. There are non-obvious semantic relationships that keyword search misses
3. The codebase spans many languages or paradigms
4. File names and directory structure are poor

This repository has **none** of these characteristics. Its file naming is excellent, its structure is clean, and Claude Code's built-in Grep/Glob tools can search the entire codebase in milliseconds.

#### What Embeddings Would Add (and Cost)

| Tool | Setup Cost | Maintenance | Benefit for This Repo |
|------|-----------|-------------|----------------------|
| **Cursor indexing** | Low (automatic) | None | Marginal -- Cursor already indexes repos this size trivially |
| **Sourcegraph** | Medium (self-hosted or cloud) | Ongoing | Overkill -- designed for multi-repo, multi-million-line codebases |
| **Greptile** | Medium (API integration) | Per-query cost | Marginal -- the repo is small enough that grep works fine |
| **Custom vector DB** (e.g., ChromaDB + OpenAI embeddings) | High (build + maintain) | Significant | Not justified -- 39 core files are easily navigable |
| **Codebase RAG** (e.g., llama-index, LangChain) | High | Ongoing | Not justified at this scale |

**The maintenance burden of any embedding system would exceed the navigability gains.** When files change (which they do frequently in an active projection system), embeddings must be re-indexed. For 39 core files, this overhead is pure waste.

### 2.2 Claude Code MCP (Model Context Protocol) Evaluation

Claude Code's MCP allows adding external tools and data sources via server plugins. Relevant options:

#### MCP Servers That Could Help

1. **Filesystem MCP** (built-in): Already available. Claude Code can read/search any file.

2. **PostgreSQL MCP** (`@modelcontextprotocol/server-postgres`): The repository already has a PostgreSQL-backed intelligence system (Section 11 of AGENTS.md). An MCP server could expose the `code_inventory` table directly, avoiding the need to read the 291KB `REPOSITORY_INVENTORY.md` file.
   - **Verdict:** Useful if the PostgreSQL system is actively maintained. Currently the inventory includes cache files, suggesting the indexer needs filtering improvements before this would add value.

3. **GitHub MCP** (`@modelcontextprotocol/server-github`): Could allow querying issues and PRs without leaving the Claude Code session.
   - **Verdict:** Low priority -- the project does not appear to use GitHub Issues heavily.

4. **Memory MCP** (`@modelcontextprotocol/server-memory`): Persistent key-value memory across sessions.
   - **Verdict:** Redundant -- the existing MEMORY.md + DEVELOPMENT_TRACKER.md system already serves this purpose and is more transparent.

5. **Custom ADR Search MCP**: A lightweight MCP server that indexes ADR titles, statuses, and summaries for quick lookup without reading full ADR files.
   - **Verdict:** This is the highest-value MCP addition for this specific repository. See Section 3 for details.

#### MCP Verdict

MCP is not a priority. The existing tooling (Grep, Glob, Read) is sufficient for a project this size. The one high-value MCP addition would be an ADR index server, but a simpler solution exists (see Section 3.1).

### 2.3 CODEBASE_MAP.md vs. Embeddings

A structured `CODEBASE_MAP.md` would be **significantly more practical** than embeddings for this repository.

**What it would contain:**

```
# Codebase Map (for AI agents)

## Core Engine (cohort_projections/core/)
cohort_component.py  - Main projection engine class (~800 lines)
fertility.py         - Birth calculation from ASFR (~400 lines)
mortality.py         - Survival/aging application (~350 lines)
migration.py         - Net migration application (~350 lines)

## Data Pipeline (cohort_projections/data/)
data/load/base_population_loader.py  - Load base pop from Census PEP
data/process/residual_migration.py   - Compute migration from Census residual
data/process/convergence_interpolation.py - Time-varying rate convergence
data/process/fertility_rates.py      - Process SEER/NVSS fertility
data/process/survival_rates.py       - Process life tables
...

## Key Files (read these first for any task)
config/projection_config.yaml        - All configurable parameters
scripts/pipeline/02_run_projections.py - Full pipeline orchestration
```

**Why this is better than embeddings:**
1. Zero maintenance infrastructure
2. Fits in <500 tokens of context
3. An AI agent can read it in one shot and know exactly where to look
4. It can include information embeddings cannot capture (e.g., "read this first", "this is deprecated")
5. It is version-controlled and human-reviewable

### 2.4 The Real Problem: Documentation Surface Area, Not Search

The challenge for AI agents navigating this repository is not "finding the right file" -- file names and directory structure make that easy. The challenge is **not drowning in documentation** when trying to understand context.

Consider an agent asked to modify the migration rate computation:
1. It finds `residual_migration.py` trivially via file name
2. It reads the file (~1,300 lines) -- the docstring alone gives full context
3. But then it may also read: the module README (2,247 lines), ADR-003, ADR-040, ADR-045, ADR-049, ADR-051, ADR-055, the convergence interpolation module, the migration rates module, multiple review documents...
4. Total potential context consumption: 20,000+ tokens of documentation before writing a single line of code

**The solution is not better search -- it is documentation discipline.** See Part 3 for specific recommendations.

---

## Part 3: Practical Recommendations

### 3.1 Quick Wins (Today)

#### QW-1: Create CODEBASE_MAP.md

Create a concise (<200 line) machine-readable map of the entire codebase. Include only actively-used files with one-line descriptions and line counts. This replaces the need for agents to read REPOSITORY_INVENTORY.md (which is 1,960 lines of auto-generated noise including cache files).

**Effort:** 30 minutes. **Impact:** High -- saves agents 25,000+ tokens per session.

#### QW-2: Fix REPOSITORY_INVENTORY.md Generation

The auto-generated inventory includes 1,014 entries for `.ruff_cache`, `.mypy_cache`, and `.venv` files. The PostgreSQL indexer should exclude these patterns. If an agent reads this file today, over half the content is cache file listings.

**Effort:** 15 minutes (update SQL query or indexer filter). **Impact:** Medium.

#### QW-3: Archive Root-Level Clutter

These files at the repository root are noise for AI navigation:
- `chatgpt_feedback_on_v0.9.md` (17 KB) -- historical feedback, should be in `docs/archive/`
- `formula_audit_article-0.9-production_20260112_205726.md` (12 KB) -- timestamped audit, should be in `docs/archive/`
- `ward_county_nd_population_2008_2024.xlsx` (15 KB) -- raw data file, should be in `data/raw/`
- `2025_popest_data/` directory -- analysis workspace, should be in `scratch/` or `docs/analysis/`

Every file at the repo root competes for an AI agent's attention when it runs `ls` to orient itself.

**Effort:** 15 minutes. **Impact:** Low-medium -- reduces cognitive load at initial orientation.

#### QW-4: Add "Agent Hint" Comments to DEVELOPMENT_TRACKER.md

The tracker is 925 lines but only the first ~80 lines contain current status. Add a clear marker:

```markdown
<!-- AI AGENTS: Stop reading here. Everything below is historical archive. -->
```

Or better yet, split into `DEVELOPMENT_TRACKER.md` (current only, <100 lines) and `docs/archive/development-history.md`.

**Effort:** 10 minutes. **Impact:** Medium -- prevents agents from consuming 4,000+ tokens of history.

#### QW-5: Trim data/process/README.md

At 2,247 lines, this module README is longer than all the Python files in the module combined. Much of its content duplicates the SOP-002-compliant docstrings already in the source files. Reduce to ~200 lines covering:
- Module overview and data flow
- Quick-start usage examples
- Links to individual file docstrings for details

**Effort:** 30 minutes. **Impact:** Medium -- reduces a common over-read by ~6,000 tokens.

### 3.2 Medium-Term Improvements (1-2 Weeks)

#### MT-1: ADR Summary Index

Create `docs/governance/adrs/SUMMARY_INDEX.md` -- a machine-readable table with one row per ADR:

```markdown
| ADR | Title | Status | Module Affected | One-Line Summary |
|-----|-------|--------|-----------------|------------------|
| 001 | Fertility Rate Processing | Accepted | data/process | SEER ASFR processing pipeline |
| 040 | Bakken Boom Dampening | Accepted | data/process | Dampen oil-county migration 2010-2020 |
| 051 | Oil County Dampening | Rejected | -- | Rejected: 20yr calibration within 2pp |
| 055 | Group Quarters Separation | Accepted | data/load, core | Separate GQ from household pop |
```

This allows an agent to scan all 55 active ADRs in ~2,000 tokens instead of reading 32,000 lines.

**Effort:** 2 hours. **Impact:** High for agents working on methodology changes.

#### MT-2: Introduce "Staleness Markers" in Documentation

Many markdown files have `Last Updated` dates. Add a convention where documents older than 6 months get a warning header:

```markdown
> **Staleness Warning:** This document was last updated on 2025-12-31.
> Some information may be outdated. Check source code for current behavior.
```

**Effort:** 1 hour + ongoing discipline. **Impact:** Medium -- prevents agents from trusting stale docs.

#### MT-3: Create a `/docs/for-agents/` Directory

Consolidate the minimum viable documentation set an AI agent needs:
- `CODEBASE_MAP.md` (from QW-1)
- `ADR_SUMMARY_INDEX.md` (from MT-1)
- `CURRENT_STATUS.md` (extracted from DEVELOPMENT_TRACKER.md, kept to <100 lines)
- `DATA_PIPELINE.md` (the data flow diagram, currently buried in module READMEs)

This gives agents a single directory to read for full orientation without navigating the broader docs hierarchy.

**Effort:** 3-4 hours. **Impact:** High.

#### MT-4: Deduplicate Module READMEs vs. Docstrings

The SOP-002 standard mandates comprehensive docstrings in data processing scripts. This is good. But the module README files duplicate much of this content. Establish a clear rule:
- **Docstrings:** Authoritative source for methodology, inputs, outputs, validation
- **Module READMEs:** Overview only -- list of modules, quick-start usage, and links to docstrings

**Effort:** 3-4 hours of editing. **Impact:** Reduces documentation maintenance burden and avoids conflicting information.

### 3.3 Long-Term Strategy (1-3 Months)

#### LT-1: Do NOT Invest in Embeddings

Based on this assessment:
- The codebase is 63,000 lines of Python (small-medium)
- File naming is excellent and self-documenting
- Directory structure is logical and consistent
- Grep/Glob search finds any relevant file in milliseconds
- The core library is only 39 files -- an agent can hold the entire architecture in context

**Embeddings would add maintenance burden without meaningful navigability improvement.** Revisit this decision only if the codebase grows past 200,000 lines or spans multiple languages.

#### LT-2: Consider a Lightweight ADR MCP Server (Optional)

If ADR volume continues to grow (currently 55 active + 42 in various other states = 97 total), a simple MCP server that exposes ADR search by:
- Status (Accepted, Rejected, Proposed, Superseded)
- Module affected
- Keyword in title

would be a natural extension of the existing `SUMMARY_INDEX.md`. This could be built as a 50-line Python script using the `mcp` SDK.

**Effort:** 4-6 hours. **Impact:** Low-medium -- only justified if ADR count exceeds ~100.

#### LT-3: Automate Documentation Pruning

Set up a CI check that:
1. Flags markdown files not updated in 12+ months
2. Checks DEVELOPMENT_TRACKER.md stays under 200 lines (archive the rest)
3. Validates that REPOSITORY_INVENTORY.md excludes cache/venv entries
4. Checks that module README line counts do not exceed 3x the module's Python line count

**Effort:** 4-6 hours. **Impact:** Prevents documentation bloat from recurring.

#### LT-4: SDC Replication Boundary

The `sdc_2024_replication/` directory contains 108 Python files / 69,568 lines and 40,997 lines of markdown -- more code and documentation than the main project. AGENTS.md correctly identifies this as "Reference Only," but its presence in the repo inflates any automated analysis.

Consider whether this should remain as a subdirectory or be extracted to a separate repository. An AI agent exploring the repo may inadvertently read SDC files when searching broadly.

**Effort:** Variable. **Impact:** Medium for search clarity.

### 3.4 Specific CLAUDE.md / AGENTS.md Improvements

#### CLAUDE.md

The file is nearly optimal. Two minor suggestions:

1. **Add a "Do NOT" section** referencing the MEMORY.md learnings:
   ```markdown
   ## Known Pitfalls
   - Do NOT run `pre-commit run --all-files` -- it takes 3+ minutes. Run `pytest` directly.
   - Exclude `test_residual_computation_single_period` (known 60s+ timeout on large Excel file).
   ```
   Currently this information is only in MEMORY.md, which is session-level context and not visible to all AI tools.

2. **Add a one-line project description** at the very top:
   ```markdown
   # CLAUDE.md
   North Dakota cohort-component population projection system (Python/pandas, 2025-2055 horizon).
   ```
   Currently the file jumps straight into commands without stating what the project does.

#### AGENTS.md

1. **Section 11 (Repository Intelligence):** Either remove this section or add a note that it is optional infrastructure. The instruction to "Read REPOSITORY_INVENTORY.md first" is actively harmful -- that file is 291 KB of auto-generated content including cache files. An agent following this instruction would waste enormous context.

2. **Section 6 (Shared Census Data Archive):** This detailed section about `~/workspace/shared-data/census/` is only relevant for agents adding new data sources. Consider moving it to `docs/guides/data-sources-workflow.md` and replacing it with a one-line pointer.

3. **ADR index in Section 10:** The partial ADR listing (only showing ADR-001 through ADR-003, ADR-007, ADR-016) gives an incomplete picture. Either list all active ADRs or replace with a pointer to the summary index (MT-1 above).

4. **Add an explicit "Architecture Overview" section** with the ASCII diagram currently in `core/ARCHITECTURE.md`. An agent reading AGENTS.md should get the full mental model without needing to read a separate file:
   ```
   CohortComponentProjection (engine)
       -> fertility.py (births)
       -> mortality.py (survival/aging)
       -> migration.py (net migration)

   Data Pipeline:
       raw/ -> scripts/data/ -> processed/ -> scripts/pipeline/ -> projections/
   ```

---

## Appendix A: Codebase Metrics Summary

| Metric | Value |
|--------|-------|
| **Core library files** | 39 (.py) |
| **Core library lines** | 17,759 |
| **Script files** | 51 (.py) |
| **Script lines** | 20,531 |
| **Test files** | 56 (.py) |
| **Test lines** | 24,976 |
| **Total project Python** | 63,266 lines |
| **Documentation files** | ~235 (.md) |
| **Documentation lines** | ~78,000 |
| **ADR count** | 97 (55 active) |
| **Config files** | 3 (.yaml) |
| **Config lines** | 427 (projection_config.yaml) |
| **SDC replication Python** | 69,568 lines (separate concern) |
| **SDC replication markdown** | 40,997 lines (separate concern) |
| **Documentation-to-code ratio** | ~1.2:1 (by lines) |
| **Auto-loaded context budget** | ~5,200 tokens (CLAUDE.md + AGENTS.md + MEMORY.md) |

## Appendix B: Context Window Budget Estimates

Based on the ~4 chars/token approximation for English text and code:

| Content | Characters | Est. Tokens | % of 200K Window |
|---------|-----------|-------------|-------------------|
| CLAUDE.md | 2,459 | ~600 | 0.3% |
| AGENTS.md | 14,174 | ~3,500 | 1.8% |
| MEMORY.md | 4,094 | ~1,000 | 0.5% |
| Core library (all) | 627,999 | ~157,000 | 78% |
| Scripts (all) | 715,579 | ~179,000 | 89% |
| All documentation | 3,847,899 | ~962,000 | 481% |

The entire core library fits comfortably in a single context window (78% of 200K). The entire documentation corpus does not (481%). This confirms that documentation pruning, not search infrastructure, is the correct investment.

## Appendix C: Recommendation Priority Matrix

| ID | Recommendation | Effort | Impact | Priority |
|----|---------------|--------|--------|----------|
| QW-1 | Create CODEBASE_MAP.md | 30 min | High | **1** |
| QW-4 | Split DEVELOPMENT_TRACKER.md | 10 min | Medium | **2** |
| MT-1 | ADR Summary Index | 2 hours | High | **3** |
| MT-3 | Create /docs/for-agents/ | 3-4 hours | High | **4** |
| QW-3 | Archive root-level clutter | 15 min | Low-Med | **5** |
| QW-5 | Trim data/process/README.md | 30 min | Medium | **6** |
| QW-2 | Fix REPOSITORY_INVENTORY.md | 15 min | Medium | **7** |
| MT-4 | Deduplicate READMEs vs docstrings | 3-4 hours | Medium | **8** |
| MT-2 | Staleness markers | 1 hour | Medium | **9** |
| LT-3 | Automate documentation pruning | 4-6 hours | Medium | **10** |
| LT-4 | SDC replication boundary | Variable | Medium | **11** |
| LT-2 | ADR MCP server (optional) | 4-6 hours | Low-Med | **12** |
| LT-1 | Do NOT invest in embeddings | 0 | N/A | -- |

---

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-26 |
| **Reviewer** | Claude Code (Opus 4.6) |
| **Status** | Complete |
| **Classification** | Review / Repo Hygiene Audit |
