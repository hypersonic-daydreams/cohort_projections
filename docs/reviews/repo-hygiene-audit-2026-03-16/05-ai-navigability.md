# 05 - AI Navigability Audit

**Date:** 2026-03-16
**Reviewer:** Claude Code (Opus 4.6, 1M context)
**Scope:** AI agent navigability, documentation discoverability, memory system quality, cross-reference integrity, onboarding path assessment

---

## Findings Summary Table

| # | File/Area | Finding | Severity |
|---|-----------|---------|----------|
| 1 | `AGENTS.md` Section 1 | "Current Focus" header says PP-005, which has been completed since 2026-03-01 | WARNING |
| 2 | `AGENTS.md` Section 11 | PostgreSQL "Repository Intelligence" section may mislead agents into thinking DB access is required for routine work | WARNING |
| 3 | `CLAUDE.md` preamble | Status says "PP-001 through PP-004 is closed"; does not mention PP-005 through PP-009 completion or CF-001 as active work | WARNING |
| 4 | `CLAUDE.md` workflow | "After Changes" section recommends `pre-commit run --all-files` but MEMORY.md explicitly says NOT to do this (3+ min runtime) | WARNING |
| 5 | `MEMORY.md` | "Next available: ADR-063" is stale; ADR-063 already exists; next is ADR-064 | INFO |
| 6 | `MEMORY.md` | Test counts inconsistent: 1,570 (main) vs 1,575 (CF-001 branch) without clear dating | INFO |
| 7 | `MEMORY.md` | References `feature/cf-001-college-fix-revision` branch which does not exist on the local repo (no local branches besides master) | INFO |
| 8 | `REPOSITORY_INVENTORY.md` | 1,960 lines, >256KB; too large for an AI agent to read. Lists .ruff_cache and .mypy_cache files as "active codebase" entries | WARNING |
| 9 | `DEVELOPMENT_TRACKER.md` | PP-007 and PP-008 description cells are extremely long (150+ words each in a table cell), degrading readability | INFO |
| 10 | Cross-references | All checked markdown links in `AGENTS.md`, `CLAUDE.md`, `DEVELOPMENT_TRACKER.md` resolve correctly; zero broken links found | INFO |
| 11 | `AGENTS.md` Section 10 | Documentation index lists only ADR-001 through ADR-003, ADR-007, and ADR-016 by name; 63 ADRs (001-063) are undiscoverable without reading the ADR README separately | INFO |
| 12 | Memory system | `method-config-refactor.md` refers to a completed task on a feature branch; not stale per se but occupies context budget on a finished topic | INFO |
| 13 | Onboarding | No explicit "architecture overview" or "how the code is organized" section in AGENTS.md; the README.md has a skeletal tree but no module-level descriptions | INFO |
| 14 | `CLAUDE.md` Section "Code Quality" | Lists `pre-commit run --all-files` as the first code quality command; contradicts MEMORY.md operational advice | WARNING |

**Severity counts:** 0 CRITICAL, 5 WARNING, 9 INFO

---

## 1. AGENTS.md Quality Assessment

**Grade: A-** (375 lines, ~3,800 tokens)

### Strengths

- **Well-structured layered design.** Twelve numbered sections cover identity, constraints, autonomy, procedures, quality, data, demographics, workflow, environment, docs index, intelligence, and version history. An agent can jump directly to the section it needs.
- **Explicit autonomy tiers.** Tier 1/2/3 is a standout feature that other repositories almost never have. This prevents an AI agent from making methodology changes (Tier 3) without human approval while allowing free rein on bug fixes (Tier 1).
- **Demographic domain encoding.** Race/ethnicity categories, geographic hierarchy, and age cohort definitions are inline, not buried in config files. This prevents an agent from inventing new categories.
- **Shared Census data archive.** Section 6 documents the external archive at `~/workspace/shared-data/census/` with dataset IDs, format notes, and loading instructions. Excellent for an agent that needs to trace data provenance.
- **NEVER/ALWAYS constraints.** Seven NEVER items and nine ALWAYS items create clear guardrails.
- **Observatory naming section.** Prevents agents from confusing "Observatory" with unrelated tools.

### Issues

- **Stale "Current Focus" header (WARNING).** Section 1 says the current focus is "PP-005 Phase 2+ Place Projection Enhancements." The DEVELOPMENT_TRACKER shows PP-005 was completed on 2026-03-01. PP-006 through PP-009 are also complete. The only active work item is CF-001 (College Fix model revision). An agent reading AGENTS.md would believe PP-005 is in progress and might start working on it.
  - **File:** `AGENTS.md`, line 13
  - **Fix:** Update to reflect current state (maintenance mode + CF-001 pending human review).

- **PostgreSQL Section 11 (WARNING).** References `cohort_projections_meta` database, `REPOSITORY_INVENTORY.md`, and `log_execution` context manager. While the infrastructure exists, this section can mislead an agent into thinking it needs PostgreSQL access for routine development tasks. The previous audit (2026-02-26) flagged this same issue.
  - **File:** `AGENTS.md`, lines 332-349
  - **Fix:** Add a brief note that this is optional/background infrastructure and is not needed for most development tasks.

- **Documentation index is selective.** Section 10 names ADR-001 through ADR-003, ADR-007, and ADR-016 but leaves 58 other ADRs undiscoverable from AGENTS.md alone. An agent would need to separately open `docs/governance/adrs/README.md` to find the full list. Not a blocking issue since the README is linked, but a note like "63 ADRs total; see README for full index" would improve orientation.

- **No code architecture overview.** AGENTS.md tells an agent *what* the project is and *how to behave* but does not describe *how the code is organized*. Module responsibilities (`cohort_projections/core/`, `data/process/`, `data/load/`, `geographic/`, `analysis/`, `output/`, `utils/`) are not listed. An agent must independently discover these by reading README files scattered across subpackages or the root `README.md`. The root README has only a 5-line tree.

- **Version history table (lines 355-367)** consumes 12 lines with no operational value for an agent. It was flagged in the previous audit as well.

---

## 2. CLAUDE.md Quality Assessment

**Grade: A-** (122 lines, ~2,800 tokens)

### Strengths

- **True quick-reference format.** Copy-pasteable shell commands for testing, linting, data sync, projections, benchmarking, and Observatory operations.
- **Session workflow.** Start and end procedures are clear and actionable.
- **Key rules summary.** NEVER/ALWAYS items are concise and match AGENTS.md without verbose repetition.
- **Documentation table.** Points to AGENTS.md, SOPs, guides, ADRs, and the tracker with no broken links.

### Issues

- **Stale status preamble (WARNING).** Line 4 says "PP-001 through PP-004 is closed as of 2026-03-01." This understates the current state: PP-005 through PP-009 are also complete, and CF-001 is the sole active work item. An agent skimming this line would not understand the project is in maintenance mode.
  - **File:** `CLAUDE.md`, line 4
  - **Fix:** Update to reflect all completed PPs and active CF-001.

- **Pre-commit contradiction (WARNING).** The "Code Quality" section (line 20) and "After Changes" workflow (line 82) both recommend `pre-commit run --all-files`. However, MEMORY.md explicitly advises against this because it triggers ruff, mypy, AND pytest as hooks, taking 3+ minutes. The pre-commit config confirms this: it includes ruff, mypy, pytest, data-manifest-check, and code-inventory-update hooks.
  - **File:** `CLAUDE.md`, lines 20 and 82
  - **Fix:** Either (a) change the workflow to recommend `pytest` directly (matching MEMORY.md advice), or (b) add a time warning next to the `pre-commit` command. The commands are not wrong per se, but they conflict with operational advice that has been learned through experience.

- **Observatory command block is verbose (INFO).** 14 lines of Observatory CLI commands in CLAUDE.md duplicate the same block that appears in AGENTS.md Section 4. This is acceptable for quick-reference convenience but adds ~30% to the file length.

---

## 3. Memory System Assessment

**Grade: B+**

### Structure

The memory directory at `~/.claude/projects/-home-nhaarstad-workspace-demography-cohort-projections/memory/` contains three files:

| File | Lines | Purpose |
|------|-------|---------|
| `MEMORY.md` | ~110 | Index of operational knowledge, test baselines, project structure, data vintages, experiment results |
| `method-config-refactor.md` | 72 | Detailed plan for a completed refactoring task on the CF-001 branch |
| `observatory-system.md` | 91 | Architecture reference for the Projection Observatory system |

### Strengths

- **MEMORY.md is well-organized.** Sections cover testing/pre-commit, project structure, ADR status, current projection results, group quarters, data vintages, ADR-061, Observatory, experiment sweeps, and reservation counties. Each section has specific file paths and numeric values.
- **Operational advice is valuable.** The "Do NOT run `pre-commit run --all-files`" note and the `test_residual_computation_single_period` timeout warning save real agent time.
- **Observatory-system.md is well-structured.** YAML frontmatter, execution chain diagram, parallel execution guidance, and remaining gaps are all useful for an agent resuming Observatory work.

### Issues

- **Stale ADR counter (INFO).** MEMORY.md says "next available: ADR-063" but ADR-063 (`063-evaluation-framework.md`) already exists. Next available is ADR-064.
  - **File:** `MEMORY.md`, ADR Status Summary section

- **Test count inconsistency (INFO).** "Full suite: 1,570 passed, 5 skipped" (main branch, 2026-03-01) appears alongside "Test count: 1,575 passed, 5 skipped" (CF-001 branch). Both are likely correct for their respective contexts but reading them together without context is confusing. Adding explicit branch context would help.

- **Feature branch reference (INFO).** MEMORY.md and `method-config-refactor.md` reference `feature/cf-001-college-fix-revision` which does not exist as a local branch (only `master` exists locally). The branch may exist on a remote or have been merged/deleted. An agent trying to check it out would fail.

- **method-config-refactor.md is done.** The file's own header says "Status: COMPLETE." It provides historical context for the refactoring but consumes ~72 lines of context budget on a finished task. Consider archiving or summarizing.

- **No staleness markers.** Memory files do not include "last verified" dates. The system-reminder warns that memories may be outdated, but the files themselves do not help an agent assess freshness.

---

## 4. Discoverability Assessment

| Task | Discoverable? | Path |
|------|---------------|------|
| Run tests | Yes (immediate) | `CLAUDE.md` line 12 |
| Find configuration | Yes (immediate) | `AGENTS.md` Section 9, `config/projection_config.yaml` |
| Run projections | Yes (immediate) | `CLAUDE.md` line 34 |
| Data processing logic | Moderate | Not directly linked from AGENTS.md; requires navigating to `cohort_projections/data/process/` and reading its README |
| Current project status | Yes (immediate) | `DEVELOPMENT_TRACKER.md`, explicitly designated source of truth |
| Observatory usage | Yes (immediate) | `CLAUDE.md` Observatory section + `docs/guides/observatory-start-here.md` |
| Code structure | Weak | Root `README.md` has a 5-line tree; no module descriptions in AGENTS.md; must discover subpackage READMEs independently |
| Data pipeline flow | Weak | Requires reading `docs/guides/data-sources-workflow.md`; not surfaced in top-level docs |
| ADR decisions | Moderate | AGENTS.md links to `docs/governance/adrs/README.md` which has the full index; but only 5 ADRs are named in AGENTS.md |
| Benchmarking workflow | Yes | Linked from both AGENTS.md Section 4 and the guides index |

### Key Gaps

1. **No "how the code is organized" section.** An agent arriving at this repo can quickly learn what the project does (AGENTS.md Section 1), what rules to follow (Section 2-3), and how to run things (CLAUDE.md). But understanding the code architecture requires independently reading `README.md`, then the core `ARCHITECTURE.md`, then individual subpackage READMEs. A two-paragraph summary in AGENTS.md linking to the architecture doc would close this gap.

2. **Data pipeline is not surfaced.** The flow from raw data through processing to projections is documented in `docs/guides/data-sources-workflow.md` but is not referenced in either AGENTS.md Section 4 (Workflow Scripts) or CLAUDE.md. An agent asked to "trace where fertility rates come from" would need to grep for it.

---

## 5. Cross-Reference Integrity

### Automated Link Verification

All relative markdown links (`[text](./path)`) in the three primary files (`AGENTS.md`, `CLAUDE.md`, `DEVELOPMENT_TRACKER.md`) were verified programmatically. **Zero broken links found.**

Specifically verified (all present):
- All 8 guide files in `docs/guides/`
- All 4 SOP files in `docs/governance/sops/`
- ADR README and ADR-016
- `config/projection_config.yaml`, `config/observatory_config.yaml`, `config/observatory_variants.yaml`, `config/benchmark_evaluation_policy.yaml`
- All referenced scripts (`run_all_projections.py`, `fetch_data.py`, `bisync.sh`, `run_experiment.py`, `build_experiment_dashboard.py`, `run_experiment_sweep.py`, `observatory.py`, `run_benchmark_suite.py`)
- `docs/methodology_comparison_sdc_2024.md`, `docs/methodology.md`
- `DEVELOPMENT_TRACKER.md`, `docs/archive/DEVELOPMENT_TRACKER_2026-02-26.md`
- `docs/plans/experiment-catalog.md`, `docs/plans/evaluation-blueprint.md`, `docs/plans/benchmarking-process-improvement-roadmap.md`, `docs/plans/observatory-ui-ux-backlog.md`
- All referenced review files in the tracker

### Verified from observatory-start-here.md

All six linked documents in the "Read In This Order" list exist:
1. `DEVELOPMENT_TRACKER.md`
2. `docs/guides/observatory-search-loop.md`
3. `docs/guides/observatory-autonomous-search.md`
4. `docs/guides/benchmarking-workflow.md`
5. `docs/plans/benchmarking-process-improvement-roadmap.md`
6. `docs/guides/configuration-reference.md`

**Assessment: Cross-reference integrity is excellent.** This is a notable achievement given the repository's 279 markdown files and 72 ADR entries.

---

## 6. Onboarding Path Assessment

### For a New AI Agent

The intended reading path is:

1. `CLAUDE.md` (auto-injected by Claude Code as system prompt) -- 122 lines
2. `AGENTS.md` (linked from CLAUDE.md) -- 375 lines
3. `DEVELOPMENT_TRACKER.md` (linked from both) -- 525 lines

This is a reasonable 1,022-line onboarding sequence (~5,000 tokens). An agent can absorb project identity, constraints, commands, and current status in a single pass.

**Strengths:**
- The three-file layered design (quick-ref, full-ref, status) is well-executed.
- Explicit "Source Of Truth Rule" in the tracker prevents agents from treating archived reviews as open work.
- The autonomy framework (Tier 1/2/3) gives agents clear decision boundaries.
- The NEVER/ALWAYS constraints are concrete and actionable.

**Weaknesses:**
- After reading these three files, an agent still does not know the code architecture. It knows the project's purpose, rules, and status but not which modules do what.
- The DEVELOPMENT_TRACKER is operational-state focused (what is done, what remains) rather than architecture-focused. It assumes the reader already knows the codebase.
- There is no "start here for understanding the code" pointer. The root `README.md` has a Quick Start section for running the code but no "if you want to understand the internals, read X" pointer.

### For a New Human Contributor

The root `README.md` provides:
- Project overview and methodology summary
- Project structure (skeletal tree)
- Quick start (environment setup, running projections)
- Testing and configuration pointers

This is adequate as a GitHub landing page but does not link to `AGENTS.md` or the guides directory. A human contributor would need to independently discover these files.

---

## 7. Information Architecture Assessment

### Document Counts

| Category | Count | Notes |
|----------|-------|-------|
| Root instruction files | 3 | `CLAUDE.md`, `AGENTS.md`, `DEVELOPMENT_TRACKER.md` |
| Guides | 12 | `docs/guides/*.md` |
| SOPs | 4 | `docs/governance/sops/SOP-001` through `SOP-004` |
| ADRs | 63+ | `docs/governance/adrs/001-*` through `063-*` |
| Review documents | 50+ | `docs/reviews/*.md` and `*.html` |
| Plans | 6+ | `docs/plans/*.md` |
| All markdown files in docs/ | 279 | Includes all subdirectories |
| Subpackage READMEs | 5 | `core/`, `data/fetch/`, `data/process/`, `geographic/`, `output/` |
| Core architecture docs | 2 | `core/ARCHITECTURE.md`, `core/QUICKSTART.md` |
| Auto-generated files | 2 | `REPOSITORY_INVENTORY.md` (1,960 lines), `docs/INDEX.md` |

### Centralization

Documentation sources of truth are well-defined:

| Topic | Source of Truth | Stated Where |
|-------|----------------|-------------|
| Agent behavior | `AGENTS.md` | `CLAUDE.md` line 2, workspace `CLAUDE.md` |
| Current status | `DEVELOPMENT_TRACKER.md` | `AGENTS.md` Section 8, tracker's own "Source Of Truth Rule" |
| Methodology | `docs/methodology.md` | `AGENTS.md` Section 2 (ALWAYS #9) |
| Data sources | `DATA_SOURCE_NOTES.md` per directory | `AGENTS.md` Section 2 (ALWAYS #7) |
| Architecture decisions | Individual ADRs | `AGENTS.md` Section 3 (Tier 2) |

### Redundancy

- **Observatory CLI commands** appear identically in `CLAUDE.md`, `AGENTS.md` Section 4, and `observatory-system.md` (memory). Three copies. Low risk of divergence since all three must be updated together, but it is three places to maintain.
- **NEVER/ALWAYS rules** appear in both `CLAUDE.md` (summary) and `AGENTS.md` (full). This is intentional and well-executed -- the summary in CLAUDE.md is a strict subset.
- **Benchmarking commands** appear in `CLAUDE.md` and `AGENTS.md`. Same pattern, acceptable.

### Potential Conflict Points

- **Pre-commit advice.** `CLAUDE.md` recommends `pre-commit run --all-files` in two places. `MEMORY.md` says not to do this. `AGENTS.md` says pre-commit hooks should not be bypassed but does not recommend explicitly running them manually. The actual pre-commit config triggers ruff + mypy + pytest + two custom hooks, confirming MEMORY.md's 3+ minute estimate is reasonable.

- **REPOSITORY_INVENTORY.md is unreviewable.** At 1,960 lines and >256KB, it exceeds Claude Code's single-read limit. The file includes cache directory entries (`.ruff_cache`, `.mypy_cache`) as "Active Files," inflating the count to 1,919. For an agent directed to "read REPOSITORY_INVENTORY.md first" (AGENTS.md Section 11), this is a dead end.

---

## 8. Comparison to Previous Audit (2026-02-26)

The previous navigability audit (`docs/reviews/repo-hygiene-audit/06-ai-navigability-and-embeddings.md`) graded CLAUDE.md at A, AGENTS.md at A-, and MEMORY.md at A. It identified the same PostgreSQL Section 11 concern and version-history noise.

### What Has Improved Since Then

- **Observatory documentation is comprehensive.** The `observatory-start-here.md` guide with a numbered reading order is a best practice. Three observatory-specific guides now exist.
- **DEVELOPMENT_TRACKER has explicit Source Of Truth Rule.** This was not present in the previous audit.
- **Cross-reference integrity is perfect.** Zero broken links across 279 markdown files.

### What Remains From Previous Audit

- PostgreSQL Section 11 concern is unchanged.
- Version history table in AGENTS.md is unchanged.
- No code architecture section has been added to AGENTS.md.

### New Issues Since Previous Audit

- AGENTS.md "Current Focus" has become stale (PP-005 completed three weeks ago).
- CLAUDE.md status preamble has become stale (PP-005 through PP-009 all completed).
- MEMORY.md ADR counter is stale.
- Pre-commit advice conflict has emerged between CLAUDE.md and MEMORY.md.

---

## 9. Recommendations

### Priority 1 (Fix This Week)

1. **Update AGENTS.md "Current Focus" header** to reflect maintenance mode + CF-001 as sole active work. This is the most visible source of misinformation for a new agent session.

2. **Update CLAUDE.md status preamble** to mention PP-005 through PP-009 completion and CF-001 as active.

3. **Resolve pre-commit advice conflict.** Either update CLAUDE.md to recommend `pytest` directly (matching learned experience in MEMORY.md), or add a parenthetical time estimate next to `pre-commit run --all-files` so agents can make informed choices.

### Priority 2 (Fix This Month)

4. **Add a code architecture paragraph** to AGENTS.md (or a new Section 4.5) listing the main modules and their responsibilities. Even five lines would close the biggest discoverability gap:
   ```
   cohort_projections/core/     - Projection engine (cohort_component, fertility, mortality, migration)
   cohort_projections/data/     - Data loading (load/) and processing (process/)
   cohort_projections/analysis/ - Evaluation framework and Observatory
   cohort_projections/geographic/ - Multi-geography handling
   cohort_projections/output/   - Report generation and export
   ```

5. **Update MEMORY.md** to fix the stale ADR counter (next available is ADR-064), clarify test count branch context, and add "last verified" dates to key sections.

6. **Consider archiving `method-config-refactor.md`** from the memory directory, or condensing it to a 5-line summary. It is marked COMPLETE and consumes 72 lines of context on a finished task.

### Priority 3 (Nice to Have)

7. **Add a note to AGENTS.md Section 11** clarifying that PostgreSQL access is not required for routine development.

8. **Trim or collapse AGENTS.md version history** (lines 355-367). It has no operational value for agents.

9. **Exclude cache directories from REPOSITORY_INVENTORY.md** to bring it under the 256KB readability threshold. Alternatively, add a note in AGENTS.md that agents should not attempt to read this file in full.

10. **Condense long table cells in DEVELOPMENT_TRACKER.md** for PP-007 and PP-008. These cells are 150+ words each, making the table nearly unreadable in a terminal or narrow viewer.

---

## 10. Overall Assessment

**AI Navigability Grade: A-**

This repository is among the best-instrumented projects for AI agent navigation. The three-file layered documentation system (`CLAUDE.md` -> `AGENTS.md` -> `DEVELOPMENT_TRACKER.md`) is a strong pattern. The autonomy framework, explicit constraints, domain encoding, and zero-broken-link cross-reference integrity are all exemplary.

The primary weaknesses are freshness maintenance (stale "Current Focus" and status lines that fall behind the actual project state) and a missing code architecture overview. Both are straightforward to fix.

The documentation volume (279 markdown files) remains high relative to the codebase, but the source-of-truth designations and reading-order guides effectively prevent agents from drowning in historical material. The Observatory start-here guide is a particularly good example of structured navigation.

---

| Attribute | Value |
|-----------|-------|
| **Audit Date** | 2026-03-16 |
| **Reviewer** | Claude Code (Opus 4.6, 1M context) |
| **Previous Audit** | 2026-02-26 (`docs/reviews/repo-hygiene-audit/06-ai-navigability-and-embeddings.md`) |
| **Files Checked** | 50+ (all referenced paths in AGENTS.md, CLAUDE.md, DEVELOPMENT_TRACKER.md, memory files) |
| **Broken Links Found** | 0 |
| **Findings** | 0 CRITICAL, 5 WARNING, 9 INFO |
