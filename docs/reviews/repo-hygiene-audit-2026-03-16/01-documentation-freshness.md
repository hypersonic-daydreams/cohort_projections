# Documentation Freshness Audit

**Date:** 2026-03-16
**Auditor:** Claude Code (automated)
**Scope:** All documentation files in the cohort_projections repository
**Method:** Cross-referencing document claims against file system state, ADR statuses, version stamps, and internal consistency

---

## Summary Table of Findings

| # | Severity | File | Issue | Suggested Fix |
|---|----------|------|-------|---------------|
| 1 | WARNING | `AGENTS.md` | "Current Focus" section says PP-005 Phase 2+ is active; all PP-005 workstreams are completed | Update Section 1 to reflect current focus (CF-001 college fix in progress, Observatory maintenance) |
| 2 | WARNING | `AGENTS.md` | Version 1.8.2 (2026-03-13) does not reflect PP-008/PP-009 or observatory dashboard work completed 2026-03-15 | Bump version and add changelog entry for post-1.8.2 work |
| 3 | INFO | `CLAUDE.md` | Version 2.5.0 dated 2026-03-12; AGENTS.md is 1.8.2 dated 2026-03-13; independent version numbering may confuse | Consider noting that CLAUDE.md and AGENTS.md version numbers are independent, or unifying them |
| 4 | WARNING | `AGENTS.md` | Section 4 references `scripts/analysis/run_benchmark_suite.py` for benchmarking; the current primary entry point is `run_experiment.py` | Clarify that `run_benchmark_suite.py` is the low-level suite runner; `run_experiment.py` is the standard workflow |
| 5 | CRITICAL | ADR README | Status summary says "57 Accepted, 5 Proposed, 1 Rejected, 0 Deprecated, 1 Superseded" = 64 total; actual file count is 67; actual statuses: ~57 Accepted, 2 Proposed, 1 Rejected, 3 Superseded, 4 child ADRs with no status | Fix counts: Accepted=57, Proposed=2, Rejected=1, Superseded=3, No Status=4; total=67 |
| 6 | WARNING | ADR README | ADR-036 listed as "Proposed" in the index; actual file status is "Accepted" | Update ADR-036 row in index from "Proposed" to "Accepted" |
| 7 | WARNING | ADR README | ADR-018 listed as "Proposed" in the index; actual file status is "Superseded by ADR-037" | Update ADR-018 row in index from "Proposed" to "Superseded" |
| 8 | WARNING | ADR README | ADR-024 listed as "Proposed" in the index; actual file status is "Accepted" | Update ADR-024 row in index from "Proposed" to "Accepted" |
| 9 | WARNING | ADR README | ADR-011 listed as "Accepted" in the index; actual file status is "Superseded by ADR-056" | Update ADR-011 row in index from "Accepted" to "Superseded" |
| 10 | WARNING | ADR README | ADR-041 listed with status "Superseded by ADR-044" in index (correct), but total "Superseded" count in summary is 1; should be 3 (011, 018, 041) | Fix superseded count from 1 to 3 |
| 11 | INFO | ADR README | ADR-062 (Aggregation Tolerance Widening) and ADR-063 (Evaluation Framework) are missing from the index tables | Add entries for ADR-062 and ADR-063 to appropriate sections |
| 12 | INFO | ADR README | Child ADRs 020a, 023a, 023b, 023c have no formal `## Status` field | Add status fields to child ADRs (all should be "Accepted" per parent ADR status) |
| 13 | INFO | ADR README | "Last Updated" is 2026-03-04; ADR-063 was added later | Update "Last Updated" to current date |
| 14 | WARNING | `data/raw/housing/` | No `DATA_SOURCE_NOTES.md` file; directory contains `nd_place_housing_units.csv` | Create DATA_SOURCE_NOTES.md for housing data per AGENTS.md ALWAYS rule 7 |
| 15 | INFO | `docs/methodology.md` | Version "1.0" with date "February 2026"; content includes sections added in March 2026 (evaluation framework, rolling-origin, housing-unit) | Bump version to 1.1 or 2.0 and update date to March 2026 |
| 16 | INFO | `docs/methodology.md` | ADR-061 (college fix) referenced in methodology text (Sections 5b, 5f, 5i) but ADR-061 status is still "Proposed" and not listed in the Section 9.3 ADR reference table | Add ADR-061 to Section 9.3 table when it is accepted; note in text that ADR-061 changes are proposed |
| 17 | INFO | `README.md` | References `python scripts/analysis/observatory_dashboard.py --no-open` which exists but is not the main CLI entry point documented elsewhere | Consider aligning with `python scripts/analysis/observatory.py` as the primary reference |
| 18 | INFO | `README.md` | "License" section says "To be determined" | Assign a license or note it is proprietary/internal |
| 19 | WARNING | `AGENTS.md` | Section 11 references `REPOSITORY_INVENTORY.md` and PostgreSQL `cohort_projections_meta` DB; these appear to be from early project setup and may not be actively maintained | Verify if the DB and inventory system are still in active use; if not, add a note or remove the section |
| 20 | INFO | `AGENTS.md` | Section 11 references `scripts/intelligence/generate_docs_index.py` which exists, and `docs/INDEX.md` which exists | No action needed; reference is valid |
| 21 | WARNING | `DEVELOPMENT_TRACKER.md` | Test baseline cited as "1570 passed, 5 skipped" from 2026-03-01; PP-006 added 154 tests and BM-001 added 42 tests; current baseline should be higher | Update test baseline count to reflect post-PP-006/BM-001 counts |
| 22 | INFO | `DEVELOPMENT_TRACKER.md` | Last updated 2026-03-15; current date is 2026-03-16; this is fresh | No action needed |
| 23 | INFO | `docs/methodology.md` | ADR-058 (Multi-County Splitting) is referenced in Section 7.5 implicitly but not explicitly listed in the Section 9.3 ADR table | Add ADR-058 to Section 9.3 for completeness |
| 24 | INFO | `CLAUDE.md` | References `python scripts/fetch_data.py` for data fetching; this script exists | No action needed |
| 25 | INFO | Top-level | No top-level `DATA_SOURCE_NOTES.md` exists; the AGENTS.md ALWAYS rule 7 says to update it "when adding files to data/raw/"; individual subdirectories have their own notes files | Clarify in AGENTS.md whether the rule refers to per-subdirectory notes (which exist) or a single top-level file (which does not exist). Current practice (per-subdirectory) is fine. |
| 26 | INFO | `docs/plans/` | Contains 11 plan files; most are implementation artifacts from completed work | Consider archiving completed plan files or adding a status header to each |
| 27 | WARNING | `AGENTS.md` | Section 1 says "Geographic Scope: Places (~400)"; actual place universe per PP-003 is 355 active + 265 EXCLUDED = 355 projected | Update to "Places (355 active)" or "Places (~355)" for accuracy |

---

## Detailed Findings

### 1. AGENTS.md and CLAUDE.md Consistency

**Files:** `/home/nhaarstad/workspace/demography/cohort_projections/AGENTS.md`, `/home/nhaarstad/workspace/demography/cohort_projections/CLAUDE.md`

**Overall assessment:** The two files are well-aligned. CLAUDE.md explicitly defers to AGENTS.md for complete guidance, and the quick commands in CLAUDE.md are a strict subset of what AGENTS.md describes. No contradictions found between the two files.

**Issues found:**

- **AGENTS.md Current Focus is stale (WARNING).** Section 1 reads: "Current Focus: PP-005 Phase 2+ Place Projection Enhancements." Per DEVELOPMENT_TRACKER.md, PP-005 was completed on 2026-03-01. The current active work item is CF-001 (College Fix model revision, in progress on `feature/cf-001-college-fix-revision` branch). PP-006 through PP-009 have also been completed since the focus line was last updated. The focus line should read something like: "Current Focus: CF-001 College Fix model revision (ADR-061); Projection Observatory maintenance."

- **AGENTS.md version is 1.8.2 dated 2026-03-13 (WARNING).** Significant work has landed since then: PP-008 (deterministic autonomous search sandbox, completed 2026-03-15), PP-009 (longitudinal benchmark-history dashboard, completed 2026-03-15), and multiple observatory dashboard UX follow-ons. The AGENTS.md version changelog does not reflect this.

- **Independent version numbering (INFO).** CLAUDE.md is at version 2.5.0 and AGENTS.md is at version 1.8.2. These are independent numbering schemes, which is fine but could confuse a reader. The relationship could be documented more explicitly.

- **Place count (WARNING).** AGENTS.md Section 1 table says "Places (~400)". The actual production place universe per PP-003 is 355 active places (9 HIGH + 9 MODERATE + 72 LOWER + 265 EXCLUDED). Since EXCLUDED places do not receive projections, the working count is 90 projected places (355 minus 265 EXCLUDED). Consider updating to "Places (355 active, 90 projected)" or similar.

**All file path references in AGENTS.md verified as existing:**
- `config/projection_config.yaml` -- EXISTS
- `scripts/projections/run_all_projections.py` -- EXISTS (via `scripts/` listing)
- `scripts/bisync.sh` -- EXISTS
- `scripts/analysis/run_benchmark_suite.py` -- EXISTS
- `scripts/analysis/run_experiment.py` -- EXISTS
- `scripts/analysis/build_experiment_dashboard.py` -- EXISTS
- `scripts/analysis/run_experiment_sweep.py` -- EXISTS
- `scripts/analysis/observatory.py` -- EXISTS
- `config/observatory_config.yaml` -- EXISTS
- `config/observatory_variants.yaml` -- EXISTS
- `config/benchmark_evaluation_policy.yaml` -- EXISTS
- `docs/guides/benchmarking-workflow.md` -- EXISTS
- `docs/guides/observatory-start-here.md` -- EXISTS
- `docs/guides/observatory-search-loop.md` -- EXISTS
- All SOP files referenced in Section 10 -- EXISTS
- All guide files referenced in Section 10 -- EXISTS
- `REPOSITORY_INVENTORY.md` -- EXISTS
- `scripts/intelligence/generate_docs_index.py` -- EXISTS
- `docs/INDEX.md` -- EXISTS

---

### 2. DEVELOPMENT_TRACKER.md

**File:** `/home/nhaarstad/workspace/demography/cohort_projections/DEVELOPMENT_TRACKER.md`

**Overall assessment:** Well-maintained and current. Last updated 2026-03-15, one day before this audit. All completed items are properly marked. The tracker is comprehensive and serves its stated purpose as the canonical status document.

**Issues found:**

- **Test baseline count may be stale (WARNING).** The "Test health baseline" row in the Current Snapshot table cites "1570 passed, 5 skipped" from the 2026-03-01 PP-005 validation. However, PP-006 added 154 tests and BM-001 added 42 tests after that date. The BM-001 section itself reports "1624 passed, 5 skipped." The Current Snapshot table should reflect the most recent known count.

- **All referenced files verified:** Spot-checked approximately 20 referenced evidence artifacts (review docs, ADRs, config files) and all exist.

- **Backlog status is accurate:** PP-001 through PP-009 are all marked completed with appropriate dates. CF-001 is correctly marked `in_progress`. Near-Term Next Actions are reasonable and current.

---

### 3. ADRs (docs/governance/adrs/)

**Directory:** `/home/nhaarstad/workspace/demography/cohort_projections/docs/governance/adrs/`

**Overall assessment:** The ADR system is mature and well-used. 67 ADR files exist. However, the ADR README index has several status discrepancies with the actual file contents.

#### Status Count Discrepancies (CRITICAL)

The ADR README claims:

| Status | README Count | Actual Count |
|--------|:---:|:---:|
| Accepted | 57 | 57 |
| Proposed | 5 | 2 (022, 061) |
| Rejected | 1 | 1 (051) |
| Deprecated | 0 | 0 |
| Superseded | 1 | 3 (011, 018, 041) |
| No formal status | -- | 4 (020a, 023a, 023b, 023c) |
| **Total** | **64** | **67** |

The README says "Total ADRs: 64" but there are 67 files (excluding README.md, TEMPLATE.md, and RECONCILIATION.md). The count and status breakdown are both wrong.

#### Index-to-File Status Mismatches (WARNING)

These ADRs have different statuses in the README index table versus their actual file content:

| ADR | Index Says | File Says |
|-----|-----------|-----------|
| ADR-018 | Proposed | Superseded by ADR-037 |
| ADR-036 | Proposed | Accepted |
| ADR-024 | Proposed | Accepted |
| ADR-011 | Accepted | Superseded by ADR-056 |

These mismatches explain the inflated "Proposed" count (5 vs 2) and deflated "Superseded" count (1 vs 3).

#### Missing from Index (INFO)

- **ADR-062** (Aggregation Tolerance Widening, Accepted) -- not in any index table
- **ADR-063** (Evaluation Framework, Accepted) -- not in any index table

#### Child ADRs Missing Status Fields (INFO)

Four child ADR files have no formal `## Status` or `Status:` field:
- `020a-vintage-methodology-investigation-plan.md`
- `023a-evidence-review-package.md`
- `023b-project-utils-package.md`
- `023c-codebase-catalog-package.md`

Their parent ADRs (020, 023) are Accepted, and these child ADRs are listed as Accepted in the README index. Adding a status field to each would improve consistency.

#### ADR-022 Staleness Check (INFO)

ADR-022 (Unified Documentation Strategy) has been "Proposed" since 2026-01-01 (75 days). It proposes a PostgreSQL-backed documentation system. AGENTS.md Section 11 references this system as existing. If it is implemented, the ADR should be Accepted. If it was abandoned or descoped, it should be Deprecated.

#### ADR-061 Status (INFO)

ADR-061 (College Fix Model Revision) is correctly marked "Proposed." Per DEVELOPMENT_TRACKER.md, CF-001 is `in_progress` on a feature branch. The Proposed status is appropriate until the owner decides whether to promote the method revision.

---

### 4. docs/methodology.md

**File:** `/home/nhaarstad/workspace/demography/cohort_projections/docs/methodology.md`

**Overall assessment:** Excellent. This is a 1,241-line comprehensive methodology document with detailed mathematical notation, ADR cross-references, and clear explanations of every component. It appears to be the most thorough and carefully written document in the repository.

**Issues found:**

- **Version/date header is stale (INFO).** The header reads "Version: 1.0, Date: February 2026." Content includes sections added in March 2026 (Section 8 Evaluation Framework referencing ADR-063 and PP-006; Section 7.5.1 Rolling-Origin referencing PP-005 WS-A; Section 7.5.2 Housing-Unit referencing PP-005 WS-D). The version should be bumped.

- **ADR-061 references in body (INFO).** Sections 5b (GQ fraction), 5f (college counties list, 25-29 extension), and 5i (rate cap 25-29) reference ADR-061 changes. Since ADR-061 is still "Proposed" and on a feature branch, these references may describe changes that are not yet in production. This should be reviewed: either (a) the methodology doc is describing the proposed/in-progress method, which is fine but should be noted, or (b) some ADR-061 decisions have already been merged to master (the expanded college county list and 25-29 smoothing appear to be in the main branch based on the methodology text).

- **Section 9.3 ADR table is incomplete (INFO).** Missing from the ADR reference table at the end:
  - ADR-058 (Multi-County Place Splitting) -- referenced implicitly in Section 7.5
  - ADR-061 (College Fix) -- referenced in Sections 5b, 5f, 5i
  - ADR-062 (Aggregation Tolerance Widening)
  - ADR-063 (Evaluation Framework) -- referenced in Section 8

- **Formulas and data sources verified:** Spot-checked the base population (799,358), projection horizon (2025-2055), race categories (6), age groups (91), county count (53), and all are consistent with config and other documentation.

---

### 5. DATA_SOURCE_NOTES.md Files

**Overall assessment:** Good coverage. 10 of 11 `data/raw/` subdirectories have `DATA_SOURCE_NOTES.md` files. The population notes file is particularly thorough (469 lines with column definitions, validation tables, and historical notes).

**Issues found:**

- **`data/raw/housing/` has no DATA_SOURCE_NOTES.md (WARNING).** The directory contains `nd_place_housing_units.csv` but no documentation file. Per AGENTS.md ALWAYS rule 7 ("Update DATA_SOURCE_NOTES.md when adding files to data/raw/"), this should have been created when the housing data was added (during PP-005 WS-D, 2026-03-01).

- **Top-level DATA_SOURCE_NOTES.md does not exist (INFO).** AGENTS.md ALWAYS rule 7 references "DATA_SOURCE_NOTES.md" without specifying whether it means per-subdirectory or a single top-level file. The current practice (per-subdirectory) is better for maintainability, but the rule could be clarified.

- **Top-level PDF files in `data/raw/` are undocumented (INFO).** The root of `data/raw/` contains 11 PDF files (2016VES.pdf through 2025 Provisional Data.pdf, plus "ND Population Projections.pdf") that are not covered by any DATA_SOURCE_NOTES.md. These appear to be ND vital event summary PDFs. While they may be reference-only, they should either be moved to a subdirectory with notes or documented in place.

---

### 6. README.md

**File:** `/home/nhaarstad/workspace/demography/cohort_projections/README.md`

**Overall assessment:** Functional but minimal. Provides adequate quick-start instructions. The Observatory section is well-maintained.

**Issues found:**

- **License section (INFO).** Says "To be determined." This has been the case since the README was created. If the project is internal/proprietary, state so explicitly.

- **Observatory dashboard reference (INFO).** The README references `python scripts/analysis/observatory_dashboard.py --no-open` which exists and works. Other documentation primarily references `python scripts/analysis/observatory.py` as the CLI entry point. Both are valid but the README could note the relationship (CLI vs dashboard).

- **Contact section is generic (INFO).** Says "North Dakota demographic projections project" with no actual contact information. Consider adding a responsible party or organization.

---

### 7. Other Documentation Files

#### docs/guides/ (12 files)

All 12 guide files referenced in AGENTS.md Section 10 exist:
- `benchmarking-workflow.md` -- EXISTS
- `configuration-reference.md` -- EXISTS
- `data-sources-workflow.md` -- EXISTS
- `environment-setup.md` -- EXISTS
- `observatory-autonomous-search.md` -- EXISTS
- `observatory-search-loop.md` -- EXISTS
- `observatory-start-here.md` -- EXISTS
- `test-maintenance-practices.md` -- EXISTS
- `test-suite-reference.md` -- EXISTS
- `testing-workflow.md` -- EXISTS
- `troubleshooting.md` -- EXISTS
- `README.md` -- EXISTS

No freshness issues detected from file path verification.

#### docs/governance/sops/ (4 SOPs + README + TEMPLATE)

All 4 SOPs referenced in AGENTS.md exist:
- `SOP-001-external-ai-analysis-integration.md` -- EXISTS
- `SOP-002-data-processing-documentation.md` -- EXISTS
- `SOP-003-method-benchmarking-versioning-promotion.md` -- EXISTS
- `SOP-004-experimental-methodology-branches.md` -- EXISTS

#### docs/plans/ (11 files)

Contains implementation plans and roadmaps. Most reference completed work (PP-003, benchmarking P0, evaluation blueprint). These are historical artifacts and do not need updating, but could benefit from a "Status: Complete" header for clarity.

#### docs/GIT_RCLONE_SYNC.md -- EXISTS

Referenced in README.md. Verified present.

#### docs/methodology_comparison_sdc_2024.md -- EXISTS

Referenced in AGENTS.md Section 10. Verified present.

---

## Recommendations by Priority

### Immediate (CRITICAL)

1. **Fix ADR README status counts and index mismatches.** The status summary table, individual index entries for ADR-011/018/024/036, and the total count are all wrong. This is the most misleading documentation issue found.

### Short-Term (WARNING)

2. **Update AGENTS.md Current Focus** from PP-005 to reflect CF-001 and maintenance mode.
3. **Create `data/raw/housing/DATA_SOURCE_NOTES.md`** documenting `nd_place_housing_units.csv`.
4. **Add ADR-062 and ADR-063 to the ADR index.**
5. **Bump AGENTS.md version** to reflect PP-008/PP-009 work completed 2026-03-15.
6. **Update DEVELOPMENT_TRACKER.md test baseline** from 1570 to the current count (at minimum 1624 per BM-001 section, likely higher with PP-006 additions).
7. **Update AGENTS.md place count** from "~400" to "355 active" or similar accurate number.

### Low Priority (INFO)

8. Bump `docs/methodology.md` version from 1.0 to 1.1+ and update the date.
9. Add status fields to child ADRs (020a, 023a, 023b, 023c).
10. Review ADR-022 (Proposed since 2026-01-01) for staleness.
11. Add ADR-058, ADR-061, ADR-062, ADR-063 to `docs/methodology.md` Section 9.3 table.
12. Clarify AGENTS.md ALWAYS rule 7 to specify per-subdirectory notes.
13. Resolve README.md "License" section.
14. Consider documenting the top-level PDF files in `data/raw/`.

---

## Audit Statistics

| Metric | Value |
|--------|-------|
| Files examined | 85+ |
| File path references verified | 50+ |
| ADR files audited | 67 |
| CRITICAL findings | 1 |
| WARNING findings | 9 |
| INFO findings | 17 |
| Total findings | 27 |

---

| Attribute | Value |
|-----------|-------|
| **Audit Date** | 2026-03-16 |
| **Auditor** | Claude Code (automated) |
| **Repository** | cohort_projections @ master (commit 0b133d4) |
