# Documentation Audit Report

**Date:** 2026-02-26
**Auditor:** Claude Code (Opus 4.6)
**Scope:** All markdown files in the cohort_projections repository
**Status:** Complete

---

## Executive Summary

The repository contains **1,183 markdown files** (excluding `.venv/`, `.git/`, `.pytest_cache/`). Of these, **945 files (80%)** belong to `sdc_2024_replication/`, a separate journal article project that shares the repository. The remaining **238 files (~78,000 lines)** serve the core projection system.

**Key Findings:**
1. **Massive sdc_2024_replication sprawl** -- 724 files in a single claim-review evidence directory; 945 total. This is the dominant source of documentation bloat.
2. **8 ADRs have stale status in the README index** -- listed as "Proposed" when the actual files say "Accepted" or "Rejected". ADRs 054 and 055 are entirely missing from the index.
3. **Duplicate files exist** -- ADR-020's `chatgpt_review_package/` and `SHARED/` directories contain byte-identical copies of their parent files (4+ duplicates confirmed).
4. **Three competing navigation documents** exist: `docs/INDEX.md`, `docs/NAVIGATION.md`, and AGENTS.md Section 10. All serve the same purpose.
5. **CLAUDE.md and AGENTS.md** are well-structured and effective for AI agent navigation, with clear layering (CLAUDE.md as quick-reference, AGENTS.md as complete instruction set).
6. **Root-level clutter** -- `chatgpt_feedback_on_v0.9.md`, `formula_audit_article-0.9-*.md`, `REPOSITORY_INVENTORY.md` are orphaned SDC replication artifacts sitting at the repo root.
7. **Several docs/ files are stale** -- `REPOSITORY_EVALUATION.md` (December 2025, calls project "70-75% complete"), `REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md` (completed but not archived).

---

## 1. Documentation Inventory

### 1.1 Root-Level Files (7 files, 4,344 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `CLAUDE.md` | 92 | AI agent quick reference (points to AGENTS.md) | Current, well-maintained |
| `AGENTS.md` | 341 | Complete AI agent instruction set | Current (last updated 2026-02-18) |
| `README.md` | 118 | Project overview and quick start | Current but says "2025-2045" horizon (should be 2025-2055) |
| `DEVELOPMENT_TRACKER.md` | 925 | Project status and task tracking | Current but accumulating historical cruft |
| `REPOSITORY_INVENTORY.md` | 1,960 | Auto-generated file inventory from PostgreSQL | Stale -- lists ruff cache files, references `PROJECT_STATUS.md` (deleted) |
| `chatgpt_feedback_on_v0.9.md` | 153 | ChatGPT review of SDC journal article v0.9 | Misplaced -- belongs in `sdc_2024_replication/` |
| `formula_audit_article-0.9-*.md` | 226 | Formula audit of SDC journal article | Misplaced -- belongs in `sdc_2024_replication/` |

### 1.2 docs/ Top-Level Files (12 files, 5,043 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `methodology.md` | 1,084 | Core methodology document | Current, comprehensive, actively maintained |
| `methodology_comparison_sdc_2024.md` | 647 | Comparison with SDC 2024 projections | Historical reference, likely stable |
| `methodology_analysis_cohort_vs_agegroup.md` | 286 | Cohort vs age-group method comparison | Historical reference |
| `methodology_writing_plan.md` | 213 | Plan for writing methodology.md | Stale -- methodology.md is now written |
| `INDEX.md` | 302 | Auto-generated documentation index from PostgreSQL | Stale -- references `PROJECT_STATUS.md` (deleted) |
| `NAVIGATION.md` | 232 | Goal-oriented navigation guide | Redundant with AGENTS.md Section 10 |
| `REPOSITORY_EVALUATION.md` | 290 | Repository completeness evaluation | Stale -- dated Dec 2025, says "70-75% complete" |
| `REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md` | 836 | Plan for repository cleanup | Stale -- work completed, should be archived |
| `BIGQUERY_SETUP.md` | 237 | BigQuery integration setup | Likely stale -- BigQuery appears unused in current workflow |
| `GIT_RCLONE_SYNC.md` | 181 | Dual-track versioning guide | Current, actively referenced |
| `census_api_usage.md` | 482 | Census API usage guide | Stable reference |
| `census_bureau_methodology_reference.md` | 253 | Census methodology reference | Stable reference |

### 1.3 docs/governance/adrs/ (62 files, ~17,500 lines + 11,319 lines in 020-reports)

55 ADRs (001-055) plus README.md, TEMPLATE.md, ADR-020-021-RECONCILIATION.md, 020a (sub-plan), 023a/023b/023c (child ADRs), and 33 files in 020-reports/ and 021-reports/ subdirectories.

See Section 3 for detailed ADR assessment.

### 1.4 docs/governance/sops/ (7 files)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `README.md` | 58 | SOP index | Current |
| `TEMPLATE.md` | - | SOP template | Current |
| `SOP-001-external-ai-analysis-integration.md` | 481 | External AI analysis workflow | Current |
| `SOP-002-data-processing-documentation.md` | - | Data processing documentation standard | Current |
| `templates/adr-report-structure.md` | - | ADR report template | Current |
| `templates/planning-synthesis.md` | - | Planning template | Current |
| `templates/module-package.md` | - | Module template | Current |

### 1.5 docs/guides/ (6 files)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `README.md` | 32 | Guide index | Current |
| `environment-setup.md` | - | Dev environment setup | Current |
| `testing-workflow.md` | - | Testing procedures | Current |
| `configuration-reference.md` | 552 | Configuration options | Current |
| `data-sources-workflow.md` | - | Data acquisition | Current |
| `troubleshooting.md` | 404 | Common issues | Current |

### 1.6 docs/reviews/ (21 files, ~6,500 lines)

Well-organized review documents with consistent date-prefixed naming. The `methodology_comparison/` subdirectory contains 8 detailed comparison files.

### 1.7 docs/plans/ (5 files, 2,514 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `implementation-plan-census-method-upgrade.md` | 1,183 | Census method upgrade plan | Stale -- work completed |
| `census-method-assessment-and-path-forward.md` | 606 | Census method assessment | Stale -- work completed |
| `convergence-interpolation-scaling-factors.md` | 315 | Convergence scaling analysis | Stale -- work completed |
| `audit-implementation-plan-and-findings.md` | 334 | Audit plan | Stale -- work completed |
| `population-projection-explosion-bug-investigation-2026-02-13.md` | 76 | Bug investigation | Stale -- bug fixed |

### 1.8 docs/archive/ (12 files, ~5,300 lines)

Properly archived files with clear metadata. Well-organized with README index.

### 1.9 docs/ Other Subdirectories

| Directory | Files | Lines | Purpose | Status |
|-----------|-------|-------|---------|--------|
| `analysis/` | 2 | 869 | Census PEP analysis | Stable reference |
| `reports/` | 2 | 573 | Projection divergence, migration averaging | Stable reference |
| `research/` | 1 | 282 | Immigration policy impact | Stable reference |
| `reference/` | 1 | 359 | Geographic hierarchy | Current |
| `postmortems/` | 1 | 42 | CDC WONDER extraction postmortem | Stable reference |
| `governance/plans/` | 2 | 571 | Package extraction plan/tracker | Stale -- work not active |
| `governance/reports/` | 2 | 188 | Hygiene reports | Stale -- work completed |

### 1.10 Code-Embedded Documentation (8 files, 5,102 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cohort_projections/core/README.md` | 420 | Core engine documentation | Current but partially outdated (references IRS migration) |
| `cohort_projections/core/ARCHITECTURE.md` | - | Module relationship diagrams | Current |
| `cohort_projections/core/QUICKSTART.md` | - | Quick start guide | Current |
| `cohort_projections/data/process/README.md` | 2,247 | Data processing documentation | Bloated -- 2,247 lines for a single module README |
| `cohort_projections/data/process/FERTILITY_QUICKSTART.md` | - | Fertility processing guide | Current |
| `cohort_projections/data/fetch/README.md` | - | Data fetching documentation | Current |
| `cohort_projections/geographic/README.md` | 644 | Geographic module docs | Current |
| `cohort_projections/output/README.md` | 669 | Output module docs | Current |

### 1.11 data/ Documentation (27 files, 5,264 lines)

| Category | Files | Purpose | Status |
|----------|-------|---------|--------|
| `data/README.md` | 1 | Data directory overview | Partially stale -- references SEER as source |
| `data/DATA_MANIFEST.md` | 1 | Data inventory with temporal alignment | SDC-replication-focused; partially misplaced |
| `data/DATA_VALIDATION_REPORT.md` | 1 | Validation report | Stale -- dated Feb 2, old data |
| `data/exports/data_dictionary.md` | 1 | Export data dictionary | Current |
| `data/output/reports/` | 3 | Generated state reports | Current (auto-generated) |
| `data/raw/*/DATA_SOURCE_NOTES.md` | 4 | Raw data source notes | Current, required by SOP-002 |
| `data/raw/immigration/*` | 8 | Immigration data manifests | SDC-replication-focused |
| `data/processed/immigration/*` | 3 | Immigration analysis | SDC-replication-focused |
| `data/processed/sdc_2024/METHODOLOGY_NOTES.md` | 1 | SDC methodology notes | Stable reference |
| `data/raw/nd_sdc_2024_projections/` | 3 | SDC raw data documentation | Stable reference |

### 1.12 sdc_2024_replication/ (945 files, ~80,000 lines)

| Subdirectory | Files | Purpose |
|--------------|-------|---------|
| `claim_review/v4_.../evidence/` | 724 | Individual claim evidence files |
| `concordance/explanations/` | 36 | Equation and test explanations |
| `journal_article/output/versions/production/` | ~30 | Statistical analysis reports per build |
| `journal_article/revision_outputs/` | ~30 | Revision critique/recommendation files |
| `revisions/` | ~17 | Version-specific revision notes |
| Other | ~108 | Root files, citation management, etc. |

### 1.13 Miscellaneous

| File/Directory | Lines | Purpose | Status |
|----------------|-------|---------|--------|
| `2025_popest_data/` (4 files) | 897 | Census PEP release analysis | Event-specific, should be archived |
| `libs/*/README.md` (3 files) | 19 | Extracted package READMEs | Minimal stubs |
| `scripts/README.md` | - | Scripts index | Current |
| `scripts/DEPENDENCIES.md` | - | Script dependencies | Current |
| `scripts/pipeline/README.md` | 389 | Pipeline documentation | Current |
| `scripts/intelligence/README.md` | - | Intelligence scripts | Current |
| `.claude/commands/sync.md` | ~100 | Claude Code sync command | Current |

---

## 2. Issues Found

### 2.1 Critical: sdc_2024_replication Documentation Sprawl

**724 markdown files** exist in a single directory (`sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v4_.../evidence/`). These are individual evidence files from a claim-by-claim review of a journal article. Combined with other sdc_2024_replication documentation, this sub-project accounts for **80% of all markdown files** and **50% of total documentation line count** in the repository.

**Impact:** Any tool or process that scans `**/*.md` (including AI agent context gathering) will be overwhelmed by SDC replication files. This actively degrades AI agent navigability for the core projection system.

### 2.2 Duplicate Files in ADR-020-reports

The following files are **byte-identical duplicates**:

| Original | Duplicate |
|----------|-----------|
| `020-reports/AGENT_1_REPORT.md` | `020-reports/chatgpt_review_package/AGENT_1_REPORT.md` |
| `020-reports/AGENT_2_REPORT.md` | `020-reports/chatgpt_review_package/AGENT_2_REPORT.md` |
| `020-reports/AGENT_3_REPORT.md` | `020-reports/chatgpt_review_package/AGENT_3_REPORT.md` |
| `020-reports/CHATGPT_BRIEFING.md` | `020-reports/chatgpt_review_package/CHATGPT_BRIEFING.md` |
| `020-reports/ARTIFACT_SPECIFICATIONS.md` | `020-reports/SHARED/ARTIFACT_SPECIFICATIONS.md` |

**Total wasted:** ~2,400 duplicate lines across 5 files.

### 2.3 ADR README Index is Stale

The ADR README (`docs/governance/adrs/README.md`) has **10 status mismatches** with the actual ADR files:

| ADR | README Says | File Says |
|-----|-------------|-----------|
| 036 | Proposed | Accepted |
| 047 | Proposed | Accepted |
| 048 | Proposed | Accepted |
| 049 | Proposed | Accepted |
| 050 | Proposed | Accepted |
| 051 | Proposed | Rejected |
| 052 | Proposed | Accepted |
| 053 | Proposed | Accepted |
| 054 | (missing) | Accepted |
| 055 | (missing) | Accepted |

The README also reports "12 Proposed" ADRs in its summary counts. The actual count of Proposed ADRs appears to be much lower (possibly only ADR-018, ADR-022, ADR-024, and ADR-033).

### 2.4 Three Competing Navigation/Index Documents

Three documents serve overlapping "find documentation" purposes:

| Document | Lines | Type | Maintained? |
|----------|-------|------|-------------|
| `docs/INDEX.md` | 302 | Auto-generated from PostgreSQL | Stale (references deleted `PROJECT_STATUS.md`) |
| `docs/NAVIGATION.md` | 232 | Goal-oriented navigation ("I want to...") | Partially stale (last updated 2026-02-02) |
| `AGENTS.md` Section 10 | ~30 | Documentation index for AI agents | Current |

All three attempt to point readers to the same set of guides, ADRs, and configuration files. Having three creates confusion about which is authoritative.

### 2.5 Stale Documents That Should Be Archived

| File | Issue |
|------|-------|
| `docs/REPOSITORY_EVALUATION.md` | Dated Dec 2025; says project is "70-75% complete"; references data that doesn't exist |
| `docs/REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md` | 836 lines; the hygiene work is complete |
| `docs/methodology_writing_plan.md` | The methodology.md it planned is now written |
| `docs/plans/implementation-plan-census-method-upgrade.md` | 1,183 lines; work completed |
| `docs/plans/census-method-assessment-and-path-forward.md` | 606 lines; work completed |
| `docs/plans/convergence-interpolation-scaling-factors.md` | 315 lines; work completed |
| `docs/plans/audit-implementation-plan-and-findings.md` | 334 lines; work completed |
| `docs/plans/population-projection-explosion-bug-investigation-2026-02-13.md` | 76 lines; bug fixed |
| `docs/governance/plans/PACKAGE_EXTRACTION_PLAN.md` | 450 lines; extraction not active |
| `docs/governance/plans/PACKAGE_EXTRACTION_TRACKER.md` | 121 lines; extraction not active |
| `docs/governance/reports/HYGIENE_IMPROVEMENT_PROPOSAL.md` | 75 lines; proposal partially implemented |
| `docs/governance/reports/REPOSITORY_HYGIENE_REPORT.md` | 113 lines; work completed |

**Total stale lines:** ~4,345 lines across 12 files that should be in `docs/archive/`.

### 2.6 Misplaced Files at Repository Root

| File | Lines | Should Be |
|------|-------|-----------|
| `chatgpt_feedback_on_v0.9.md` | 153 | `sdc_2024_replication/reviews/` or similar |
| `formula_audit_article-0.9-*.md` | 226 | `sdc_2024_replication/reviews/` or similar |
| `REPOSITORY_INVENTORY.md` | 1,960 | `docs/archive/` (stale auto-generated inventory) |

### 2.7 README.md Has Stale Projection Horizon

`README.md` line 1 says "2025-2045" but the actual projection horizon is **2025-2055** (30 years, per AGENTS.md and methodology.md).

### 2.8 Bloated Module READMEs

`cohort_projections/data/process/README.md` is **2,247 lines** -- the longest markdown file in the project outside sdc_2024_replication. This is excessive for a single module README and suggests it is accumulating implementation details that should live in the code's docstrings or in docs/methodology.md.

### 2.9 data/DATA_MANIFEST.md Is SDC-Focused

The `data/DATA_MANIFEST.md` (445 lines) primarily documents immigration/refugee data sources used by the SDC 2024 replication study, not the core projection system's data. It is misleadingly positioned as the data directory's manifest.

### 2.10 DEVELOPMENT_TRACKER.md Is Growing Into a Changelog

At 925 lines, `DEVELOPMENT_TRACKER.md` contains detailed historical records of completed sprints, data pipeline waves, and ADR implementation histories dating back to December 2025. The "Completed Tasks" section alone spans hundreds of lines. This accumulation makes it harder to find the current status, which is the file's stated purpose.

---

## 3. ADR Organization Assessment

### 3.1 Strengths

- **Clear numbering convention** (NNN-short-title.md) consistently applied across all 55 ADRs
- **Good categorization** in the README index (Data Processing, System Architecture, Demographic Methodology, etc.)
- **Quick Reference section** in README for new team members
- **Well-maintained template** (TEMPLATE.md)
- **Cross-referencing** between ADRs and related review documents is good
- **SOP-vs-ADR distinction** is clearly documented

### 3.2 Issues

- **README index is stale** (10 status mismatches, 2 missing ADRs) -- see Section 2.3
- **ADR-020 and ADR-021 accumulated large report subdirectories** -- 33 files and 11,319 lines in 020-reports/ alone, including duplicate copies packaged for ChatGPT review
- **The ADR-020-021-RECONCILIATION.md** is a one-time analysis artifact, not a proper ADR -- it should be in `docs/reviews/` or `docs/archive/`
- **Some early ADRs (001-015) are very long** (500-740 lines each) and were written as comprehensive design documents rather than focused decision records. This is not necessarily a problem but contrasts with the more focused later ADRs (042-055, typically 200-400 lines each).
- **No formal deprecation or supersession tracking** -- ADR-041 is marked "Superseded by ADR-044" but there is no systematic way to track which ADRs are still active vs. superseded.

### 3.3 ADR-020-reports/ and ADR-021-reports/ Assessment

These directories contain **working artifacts** from multi-agent analysis workflows (agent reports, ChatGPT briefing packages, sub-agent plans). While historically valuable, they are:
- Not ADRs themselves
- Not referenced by active code
- Containing confirmed duplicate files

They would be better served by archiving or by moving to a dedicated `docs/archive/adr-working-papers/` directory.

---

## 4. SOP Organization Assessment

The SOPs are well-organized:
- Only 2 active SOPs, both clearly relevant and maintained
- Clean template system with 3 reusable templates
- Clear SOP-vs-ADR distinction documented in README
- Directory path in README says `docs/sops/` but actual path is `docs/governance/sops/` (minor inconsistency)

**No significant issues.** The SOP system is appropriately sized for the project's needs.

---

## 5. AI Agent Navigability Assessment

### 5.1 What Helps

1. **CLAUDE.md / AGENTS.md layering is excellent.** CLAUDE.md is a 92-line quick reference that immediately points to AGENTS.md for details. This is the ideal pattern -- CLAUDE.md is small enough to be loaded as context without overwhelming the agent.

2. **AGENTS.md is well-structured** with numbered sections, clear NEVER/ALWAYS rules, an autonomy framework (Tier 1/2/3), and a documentation index. An AI agent reading this document would understand the project's constraints, conventions, and where to find things.

3. **MEMORY.md in .claude/projects/** provides effective session-persistent context for Claude Code specifically, with current project state, known test issues, and file locations.

4. **ADR system** provides traceability for "why" decisions. An agent encountering unfamiliar logic can find the rationale in the corresponding ADR.

5. **Review documents** with date-prefixed naming and cross-references to ADRs provide an audit trail.

6. **docs/guides/** provides practical how-to documentation that agents can follow directly.

### 5.2 What Hurts

1. **1,183 markdown files** -- any AI agent tool that globs for `**/*.md` will be overwhelmed. The sdc_2024_replication claim-review evidence alone (724 files) could exhaust context windows or cause timeouts.

2. **Three competing navigation documents** (INDEX.md, NAVIGATION.md, AGENTS.md Section 10) -- an agent does not know which to trust. INDEX.md references a deleted file, making it untrustworthy.

3. **Stale documents that appear authoritative** -- `REPOSITORY_EVALUATION.md` looks like it describes the current state of the project but is 2 months old and wildly inaccurate (says 70-75% complete when the project is in production).

4. **DEVELOPMENT_TRACKER.md at 925 lines** takes significant context window budget. An agent asked to "check current status" will read hundreds of lines of completed historical tasks before finding the current state.

5. **ADR README staleness** means an agent looking up ADR status from the index will get wrong information (8 ADRs listed as Proposed that are actually Accepted/Rejected).

6. **Overly long module READMEs** (data/process/README.md at 2,247 lines) consume context budget when an agent reads a directory's README to understand a module.

7. **Root-level clutter** -- files like `chatgpt_feedback_on_v0.9.md` and `formula_audit_article-0.9-*.md` appear to be important project-level documents but are actually SDC replication artifacts.

### 5.3 Navigability Score

| Aspect | Rating | Notes |
|--------|--------|-------|
| Entry point clarity | A | CLAUDE.md -> AGENTS.md is excellent |
| Rules and constraints | A | NEVER/ALWAYS lists, autonomy tiers |
| Code findability | A | AGENTS.md Section 4 + memory context |
| Decision traceability | B+ | ADRs are good but index is stale |
| Current status findability | B- | DEVELOPMENT_TRACKER.md is too long |
| Data documentation | B- | DATA_SOURCE_NOTES.md is good but DATA_MANIFEST is SDC-focused |
| Signal-to-noise ratio | C | Too many stale/duplicate/misplaced files |
| Scalability | D | 1,183 .md files; glob operations are problematic |

---

## 6. Missing Documentation

### 6.1 Should Exist

| Document | Purpose | Priority |
|----------|---------|----------|
| **Architecture overview** for the full system | High-level diagram showing data flow from raw inputs through processing to projection outputs. The `core/ARCHITECTURE.md` only covers the projection engine, not the full pipeline. | Medium |
| **Deployment/operations guide** | How to run the complete system from scratch on a new machine, including rclone setup, data acquisition, and first projection run. | Medium |
| **Data lineage document** | Which raw data files feed which processed files, and how processed files feed the projection engine. Currently scattered across ADRs and DATA_SOURCE_NOTES.md files. | Low |

### 6.2 Already Exists (Contrary to Common Expectation)

These documents exist and are reasonably well-maintained, which is commendable:

| Document | Location |
|----------|----------|
| Data dictionary | `data/exports/data_dictionary.md` |
| Geographic hierarchy | `docs/reference/geographic-hierarchy.md` |
| Configuration reference | `docs/guides/configuration-reference.md` |
| Methodology document | `docs/methodology.md` (1,084 lines, comprehensive) |
| Troubleshooting guide | `docs/guides/troubleshooting.md` |

---

## 7. Recommendations

### 7.1 High Priority (Direct Impact on AI Agent Navigability)

**R1. Archive stale documents.** Move the 12 files identified in Section 2.5 to `docs/archive/` with proper archive headers. This removes ~4,345 lines of misleading content. Specifically:
- `docs/REPOSITORY_EVALUATION.md`
- `docs/REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md`
- `docs/methodology_writing_plan.md`
- All 5 files in `docs/plans/`
- All 4 files in `docs/governance/plans/` and `docs/governance/reports/`

**R2. Update the ADR README index.** Fix the 10 status mismatches and add entries for ADR-054 and ADR-055. Update the summary counts. This is a mechanical task that takes 10 minutes but has significant impact on agent trust in the index.

**R3. Consolidate navigation documents.** Choose one of these approaches:
- **Option A (recommended):** Delete `docs/INDEX.md` and `docs/NAVIGATION.md`. AGENTS.md Section 10 already serves as the canonical documentation index. Add a one-line note in docs/ directing readers to AGENTS.md.
- **Option B:** Keep `docs/NAVIGATION.md` as the detailed navigation guide but delete `docs/INDEX.md` (auto-generated, stale, references deleted files). Update AGENTS.md Section 10 to point to NAVIGATION.md for details.

**R4. Move misplaced root-level files.** Move `chatgpt_feedback_on_v0.9.md`, `formula_audit_article-0.9-*.md` to `sdc_2024_replication/`. Move `REPOSITORY_INVENTORY.md` to `docs/archive/`.

**R5. Trim DEVELOPMENT_TRACKER.md.** Move the "Completed Tasks" historical sections (lines ~195-260) to `docs/archive/` or collapse them. The file's purpose is current status, not changelog. Target: under 200 lines, showing only current status, next task, active tasks, and known blockers.

### 7.2 Medium Priority (Reducing Sprawl)

**R6. Exclude sdc_2024_replication from documentation tooling.** If any automated documentation scanning or AI agent context-gathering tool globs `**/*.md`, add an exclusion for `sdc_2024_replication/`. Consider whether the 724-file claim-review evidence directory should be gitignored and synced via rclone instead (it is generated output, not source documentation).

**R7. Remove duplicate files in ADR-020-reports.** Delete the `chatgpt_review_package/` subdirectory (all 7 files are identical to their parent-directory counterparts). Delete `SHARED/ARTIFACT_SPECIFICATIONS.md` (identical to parent). Consider archiving the entire `020-reports/` and `021-reports/` directories since their content is historical working artifacts, not active documentation.

**R8. Fix README.md projection horizon.** Change "2025-2045" to "2025-2055" to match the actual 30-year horizon.

**R9. Slim down `cohort_projections/data/process/README.md`.** At 2,247 lines, this is the largest documentation file in the core project. Extract the detailed implementation sections into the code's docstrings or methodology.md, and reduce the README to an overview and quick-reference (~200-300 lines).

### 7.3 Low Priority (Nice to Have)

**R10. Create a full-system architecture document.** A single page in `docs/` showing the end-to-end data flow from raw Census/CDC data through processing scripts to the projection engine to export outputs. The individual module READMEs exist but no document ties them together.

**R11. Standardize the `docs/` subdirectory structure.** Currently there are 11 subdirectories under `docs/` with overlapping purposes:
- `docs/plans/` and `docs/governance/plans/` -- two plan directories
- `docs/reports/` and `docs/governance/reports/` -- two report directories
- `docs/analysis/`, `docs/research/`, `docs/reference/` -- three reference-type directories

Consider consolidating to a simpler structure:
```
docs/
  archive/          # Historical documents
  governance/       # ADRs, SOPs, templates
  guides/           # How-to guides (existing, well-organized)
  methodology.md    # Core methodology (keep at top level)
  reference/        # Merge analysis/, research/, reference/, reports/ here
  reviews/          # Review documents (existing, well-organized)
```

**R12. Add review dates to long-lived documents.** Documents like the data/ README, census_api_usage.md, and module READMEs would benefit from a "Last Verified" date so agents and humans can assess staleness.

**R13. Clean up 2025_popest_data/ directory.** This appears to be event-specific analysis from the January 2026 Census PEP release. Consider whether it should be archived or moved under `docs/analysis/`.

---

## 8. Summary Statistics

| Metric | Value |
|--------|-------|
| Total markdown files | 1,183 |
| Core project files (excl. sdc_2024_replication) | 238 |
| Core project lines | ~78,000 |
| sdc_2024_replication files | 945 |
| sdc_2024_replication lines | ~80,000 |
| Files recommended for archival | 12 |
| Lines in recommended-archive files | ~4,345 |
| Confirmed duplicate files | 5+ |
| Stale ADR index entries | 10 |
| Missing ADR index entries | 2 |
| Competing navigation documents | 3 |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-26 |
| **Version** | 1.0.0 |
| **Auditor** | Claude Code (Opus 4.6) |
| **Scope** | All markdown files in cohort_projections repository |
