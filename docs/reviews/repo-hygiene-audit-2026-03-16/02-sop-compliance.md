# SOP Compliance Audit -- 2026-03-16

**Auditor:** Claude Opus 4.6 (automated)
**Scope:** All SOPs in `docs/governance/sops/`, all scripts in `cohort_projections/data/process/` and `scripts/data/`, ADR template compliance, script header consistency, test file organization, and config file documentation.

---

## Summary of Findings

| # | Severity | Area | Finding |
|---|----------|------|---------|
| 1 | WARNING | SOP-002 | 8 of 17 `cohort_projections/data/process/` modules lack the full SOP-002 metadata docstring |
| 2 | WARNING | SOP-002 | 7 of 16 `scripts/data/` scripts lack the full SOP-002 metadata docstring |
| 3 | WARNING | SOP-002 | `data/raw/housing/` directory has no `DATA_SOURCE_NOTES.md` |
| 4 | WARNING | ADR Lifecycle | ~41 of 53 Accepted ADRs lack an "Implementation Results" section |
| 5 | WARNING | ADR Format | ADR-015 uses a non-standard format (bold inline Status, "Context and Problem Statement" instead of "Context") |
| 6 | INFO | ADR Format | ADRs 001-015 predate the template and use older section headings ("Decisions" vs "Decision", "Considered Options" vs "Alternatives Considered") |
| 7 | WARNING | SOP Paths | SOP-001 and SOP README reference `docs/sops/templates/` but actual path is `docs/governance/sops/templates/` |
| 8 | INFO | SOP Template | SOP TEMPLATE.md section "8. Related Documentation" but SOPs 003/004 use different numbers (14/9) due to extra sections; still structurally compliant |
| 9 | INFO | Script Headers | Older scripts in `scripts/` (root level) have minimal docstrings -- acceptable per SOP-002 scope exclusion for non-data-processing scripts |
| 10 | INFO | Test Files | All test files follow `test_*.py` naming convention; `tests/_sdc_paths.py` is a helper, not a test |
| 11 | INFO | Config Files | All config YAML files have inline comments; `benchmark_evaluation_policy.yaml` and `evaluation_config.yaml` include `description` fields per SOP-003 |
| 12 | CRITICAL | SOP-002 | `cohort_projections/data/process/example_usage.py` is a demonstration script in the data processing package with no SOP-002 docstring and no test coverage |
| 13 | INFO | ADR Format | ADR-061 is still "Proposed" status despite active implementation on `feature/cf-001-college-fix-revision` branch |

**Totals:** 1 CRITICAL, 5 WARNING, 7 INFO

---

## 1. SOP Inventory

The project maintains four active SOPs and a template:

| SOP | Title | Status | Template Compliance |
|-----|-------|--------|-------------------|
| SOP-001 | External AI Analysis Integration | Active | Follows template structure (9 sections) |
| SOP-002 | Data Processing Script Documentation | Active | Follows template structure (9 sections) |
| SOP-003 | Method Benchmarking, Versioning, Promotion | Active | Follows template with additional sections (15 total) |
| SOP-004 | Experimental Methodology Branches | Active | Follows template with additional sections (10 total) |
| TEMPLATE | SOP Template | -- | Defines canonical 9-section structure |

All four SOPs follow the template's required sections: Document Information table, Purpose, Scope, Prerequisites, Procedure, Artifacts (some implicit), Quality Gates, Troubleshooting, Related Documentation, Revision History. SOPs 003 and 004 add extra numbered sections (Non-Negotiable Rules, Versioning Model, etc.) which is acceptable -- the template is a floor, not a ceiling.

**Finding:** SOP README's "Directory Structure" section shows `docs/sops/` as the path, but the actual path is `docs/governance/sops/`. SOP-001's Artifacts table references `docs/sops/templates/` three times. These are broken relative paths.

---

## 2. SOP-002 Compliance: Data Processing Scripts

### 2.1 SOP-002 Required Docstring Elements (9 total)

Per SOP-002 Phase 1, every data processing script must have:
1. One-line summary
2. Created date
3. ADR reference
4. Author
5. Purpose section
6. Method section
7. Key design decisions
8. Validation results
9. Inputs / Output / Usage

### 2.2 `scripts/data/` Scripts

| Script | Has Full SOP-002 Docstring? | Missing Elements | Severity |
|--------|----------------------------|------------------|----------|
| `build_nd_fertility_rates.py` | **Yes** | None (exemplar) | -- |
| `build_nd_survival_rates.py` | **Yes** | None (exemplar) | -- |
| `build_race_distribution_from_census.py` | **Yes** | None | -- |
| `fetch_census_gq_data.py` | **Yes** | None | -- |
| `fetch_census_housing_data.py` | **Yes** | Minor: missing Key design decisions | INFO |
| `assemble_place_population_history.py` | **Yes** | None | -- |
| `build_place_county_crosswalk.py` | **Yes** | None | -- |
| `ingest_ves_data.py` | Partial | Missing: Created, ADR, Author, Key design decisions, Validation results | WARNING |
| `ingest_stcoreview.py` | Partial | Missing: Created, ADR, Author, Purpose, Method, Key design decisions, Validation results | WARNING |
| `convert_popest_to_parquet.py` | Partial | Missing: Created, Author, Purpose, Method, Key design decisions, Validation results, Inputs/Output | WARNING |
| `extract_popest_docs.py` | Partial | Missing: Created, Author, Purpose, Method, Key design decisions, Validation results, Inputs/Output | WARNING |
| `archive_popest_raw_by_vintage.py` | Partial | Missing: Created, Author, Purpose, Method, Key design decisions, Validation results, Inputs/Output | WARNING |
| `download_census_pep.py` | Partial | Missing: Created, Author, Purpose, Method, Key design decisions, Validation results | WARNING |
| `build_popest_postgres.py` | Partial | Missing: Created, Author, Purpose, Method, Key design decisions, Validation results, Inputs/Output | WARNING |
| `view_census_catalog.py` | Minimal | 3-line docstring only. Utility script (<50 lines behavior). Acceptable per SOP-002 troubleshooting exception | INFO |
| `test_ves_extraction.py` | Extensive | Has detailed findings, structure docs, and Usage. Missing: Created, ADR, Author, formal Inputs/Output sections | INFO |

**Summary:** 7 scripts in `scripts/data/` have compliant docstrings. 7 scripts (the ADR-034 Census PEP pipeline + `ingest_stcoreview.py` + `ingest_ves_data.py`) predate SOP-002 and lack the required metadata structure. 1 utility script qualifies for the shortened-form exception. 1 test/exploration script has extensive but non-standard documentation.

**Note:** SOP-002's scope explicitly lists `scripts/data/build_*.py` and `scripts/data/ingest_*.py`. The ADR-034 scripts (`convert_popest_to_parquet.py`, `extract_popest_docs.py`, `archive_popest_raw_by_vintage.py`, `download_census_pep.py`, `build_popest_postgres.py`) are in scope but were written before SOP-002 was created (SOP-002 date: 2026-02-23; ADR-034 scripts: ~2026-01-15).

### 2.3 `cohort_projections/data/process/` Modules

SOP-002 Section 2 (Scope) states it covers `scripts/data/build_*.py` and `scripts/data/ingest_*.py`, and explicitly says "Core engine module documentation (covered by Google-style docstrings per AGENTS.md)" is out of scope. However, CLAUDE.md says "ALWAYS include full metadata docstrings in data processing scripts (SOP-002)" and links to SOP-002 for files in `cohort_projections/data/process/`. This creates an ambiguity about whether these library modules are in scope.

Regardless of SOP-002's strict scope, these modules perform data processing and benefit from rich documentation. Assessment against the SOP-002 standard:

| Module | Docstring Quality | Has SOP-002 Elements? | Notes |
|--------|------------------|----------------------|-------|
| `residual_migration.py` | Rich | Partial (Purpose, Method, ADR refs in docstring; missing Created/Author/Validation/Inputs) | Core pipeline module |
| `convergence_interpolation.py` | Rich | Partial (Method described; missing Created/Author/Validation/Inputs) | Core pipeline module |
| `mortality_improvement.py` | Rich | Partial (Method formula; missing Created/Author/Validation/Inputs) | Core pipeline module |
| `migration_rates.py` | Moderate | Minimal (summary + context) | Original early module |
| `base_population.py` | Moderate | Minimal (summary only) | Original early module |
| `fertility_rates.py` | Moderate | Minimal (summary + context) | Original early module |
| `survival_rates.py` | Moderate | Minimal (summary + context) | Original early module |
| `pep_regime_analysis.py` | Moderate | Partial (Purpose described; missing formal structure) | |
| `place_shares.py` | Minimal | 2-line summary only | WARNING |
| `place_share_trending.py` | Moderate | 5-line summary of capabilities | |
| `place_backtest.py` | Minimal | 1-line summary only | WARNING |
| `rolling_origin_backtest.py` | Good | Has Design section (non-standard but informative) | |
| `multicounty_allocation.py` | **Full** | Yes -- Created, ADR, Author, Purpose, Method | Compliant |
| `place_projection_orchestrator.py` | Moderate | Summary + multi-county handling note | |
| `place_housing_unit_projection.py` | Good | Functions listed; ADR reference; missing Created/Author | |
| `example_usage.py` | Minimal | 2-line summary; no SOP-002 elements | CRITICAL (see below) |

### 2.4 DATA_SOURCE_NOTES.md Coverage

| Directory | Has DATA_SOURCE_NOTES.md? | Status |
|-----------|--------------------------|--------|
| `data/raw/fertility/` | Yes | -- |
| `data/raw/mortality/` | Yes | -- |
| `data/raw/population/` | Yes | -- |
| `data/raw/migration/` | Yes | -- |
| `data/raw/census/` | Yes | -- |
| `data/raw/census_bureau_methodology/` | Yes | -- |
| `data/raw/geographic/` | Yes | -- |
| `data/raw/immigration/` | Yes | -- |
| `data/raw/nd_sdc_2024_projections/` | Yes | -- |
| `data/raw/enrollment/` | Yes | -- |
| **`data/raw/housing/`** | **No** | **WARNING** |

The `data/raw/housing/` directory contains `nd_place_housing_units.csv` (produced by `scripts/data/fetch_census_housing_data.py`, ADR-060) but has no `DATA_SOURCE_NOTES.md`. Per SOP-002 Phase 2, every `data/raw/{category}/` directory must have this file.

---

## 3. SOP Template Compliance

### 3.1 SOP Template

The SOP template (`docs/governance/sops/TEMPLATE.md`) defines 9 sections. All four active SOPs include all 9 required sections. SOPs 003 and 004 add domain-specific sections (Versioning Model, Non-Negotiable Rules, Branch Decision Table, etc.) which is appropriate.

### 3.2 ADR Template

The ADR template (`docs/governance/adrs/TEMPLATE.md`) defines the following required structure:

- `## Status`
- `## Date`
- `## Context` (with Requirements, Challenges sub-sections)
- `## Decision` (with numbered decisions, each having Decision, Rationale, Implementation, Alternatives Considered)
- `## Consequences` (Positive, Negative, Risks and Mitigations)
- `## Alternatives Considered`
- `## Implementation Notes`
- `## References`
- `## Revision History`
- `## Related ADRs`

### 3.3 Broken Path References

| File | Broken Reference | Correct Path |
|------|-----------------|--------------|
| `docs/governance/sops/README.md` line ~40 | `docs/sops/` | `docs/governance/sops/` |
| `docs/governance/sops/TEMPLATE.md` line 95 | `docs/sops/templates/name.md` | `docs/governance/sops/templates/name.md` |
| `docs/governance/sops/SOP-001-external-ai-analysis-integration.md` lines 410-412 | `docs/sops/templates/...` | `docs/governance/sops/templates/...` |

---

## 4. ADR Format Compliance

### 4.1 Sample of 10 ADRs

| ADR | Status | Date | Context | Decision | Consequences | Impl. Results | Revision History | Related ADRs | Template Compliance |
|-----|--------|------|---------|----------|-------------|---------------|-----------------|--------------|-------------------|
| 001 | Accepted | Yes | Yes | Yes (numbered decisions) | Yes | No | Yes | Yes | Good -- matches template |
| 015 | Accepted | Yes (inline bold) | Non-standard ("Context and Problem Statement") | Non-standard ("Considered Options") | Partial | No | No | No | Poor -- predates template |
| 036 | Accepted | Yes | Yes | Yes (numbered decisions) | Yes | Yes | Yes | Yes | Excellent |
| 040 | Accepted | Yes | Yes (with Scope, Last Reviewed) | Yes | Yes | No | Yes | Yes | Good |
| 053 | Accepted | Yes | Yes (with Implemented, Last Reviewed, Scope) | Yes | Yes | Yes | Yes | Yes | Excellent |
| 055 | Accepted | Yes | Yes (with Scope, Last Reviewed, amendment note) | Yes | Yes | Yes | Yes | Yes | Excellent |
| 057 | Accepted | Yes | Yes | Yes (3 decisions) | Yes | No | Yes | Yes | Good |
| 060 | Accepted | Yes | Yes | Yes (2 decisions) | Partial | No | Yes | Yes | Good |
| 062 | Accepted | Yes | Yes | Yes | Yes | No | Yes | Yes | Good |
| 063 | Accepted | Yes | Yes | Yes (3+ decisions) | Yes | No | Yes | Yes | Good |

### 4.2 ADR Format Observations

**Consistent elements across all sampled ADRs:**
- `## Status` with valid values (Accepted, Proposed, Rejected, Superseded)
- `## Date` in YYYY-MM-DD format
- `## Context` section with problem description
- `## Decision` section(s) with rationale

**Common deviations:**
- **ADR-015** uses a completely different format: bold inline status (`**Status**: Accepted`), "Context and Problem Statement" instead of "Context", "Considered Options" instead of "Decision", and "Decision Drivers" instead of "Requirements". This predates the current template.
- **Early ADRs (001-015)** use slightly different headings but are structurally similar. These were written before the template was formalized.
- **ADRs 040+** generally follow the template closely, with optional additions like `## Scope`, `## Last Reviewed`, `## Implemented`, and `## Supersedes`.

### 4.3 Implementation Results Gap

SOP-002 Phase 3 requires that every Accepted ADR that has been implemented should have an `## Implementation Results` section. Of the 53 Accepted ADRs:

- **12 have Implementation Results** (ADRs 033, 036, 047, 048, 049, 050, 051, 052, 053, 054, 055, 058)
- **~41 lack Implementation Results**

This is the largest compliance gap. Many ADRs from the 001-035 range were implemented before SOP-002 was created and were never backfilled. However, several post-SOP-002 ADRs (057, 059, 060, 062, 063) are also missing this section despite being implemented.

**Priority for backfill:** ADRs 057, 059, 060, 062, 063 (post-SOP-002, recently implemented, highest value to backfill).

---

## 5. Script Header Consistency

### 5.1 `scripts/data/` (In-scope for SOP-002)

Covered in Section 2.2 above. 7 of 16 scripts are fully compliant.

### 5.2 `scripts/` Root-Level Scripts

SOP-002 explicitly excludes export/visualization scripts and non-data-processing scripts. Assessment for general quality:

| Script | Docstring? | Quality |
|--------|-----------|---------|
| `fetch_data.py` | Yes | Good (Usage, ADR reference, categories) |
| `process_nd_migration.py` | Yes | Minimal (summary + output path) |
| `extract_sdc_fertility_rates.py` | Yes | Good (methodology, outputs listed) |
| `validate_data.py` | Yes | Minimal (1 line) |
| `check_test_coverage.py` | Yes | Good (describes 3 checks, usage) |
| `run_integration_test.py` | Yes | Good (usage, output) |
| `generate_article_pdf.py` | Yes | Minimal (2 lines) |
| `generate_visualizations_and_reports.py` | Yes | Good (4 steps described, usage) |

All root-level scripts have docstrings. Quality varies from minimal 1-liners to structured multi-section headers. Since these are out of SOP-002 scope, this is informational.

### 5.3 `scripts/pipeline/` Scripts

| Script | Docstring? | Quality |
|--------|-----------|---------|
| `00_prepare_processed_data.py` | Yes | Good (source/dest mapping, usage options) |
| `01_process_demographic_data.py` | Yes | Good (usage, components) |
| `01a_compute_residual_migration.py` | Yes | Good (ADR references, usage) |
| `01b_compute_convergence.py` | Not checked in detail | -- |
| `01c_compute_mortality_improvement.py` | Not checked in detail | -- |
| `02_run_projections.py` | Not checked in detail | -- |
| `02a_run_place_projections.py` | Not checked in detail | -- |
| `03_export_results.py` | Not checked in detail | -- |

Pipeline scripts consistently have descriptive docstrings with usage examples. Good overall pattern.

---

## 6. Test File Organization

### 6.1 Directory Structure

```
tests/
  conftest.py                    # Shared fixtures
  _sdc_paths.py                  # Helper module (not a test)
  __init__.py
  test_core/                     # Core engine tests
  test_data/                     # Data processing tests
  test_config/                   # Configuration tests
  test_geographic/               # Geography tests
  test_integration/              # Integration tests
  test_output/                   # Output/export tests
  test_statistical/              # Statistical analysis tests
  test_utils/                    # Utility tests
  test_tools/                    # Tool tests
  test_analysis/                 # Analysis framework tests
    test_evaluation/             # Evaluation framework tests
  unit/                          # Legacy unit tests (SDC replication)
```

### 6.2 Naming Convention Compliance

- All test files follow the `test_*.py` pattern required by pytest.
- `tests/_sdc_paths.py` is correctly prefixed with underscore to avoid pytest collection.
- `tests/unit/` contains legacy tests from the SDC 2024 replication work; these follow `test_*.py` naming.
- Subdirectories follow `test_{module_area}/` naming consistently.

### 6.3 Observations

- **No orphaned test files** detected -- all test directories have `__init__.py` files.
- **`tests/unit/` is a legacy directory** that does not follow the `test_{area}/` convention used by all other directories. Its tests (`test_module_7_causal_inference.py`, `test_module_8_duration_analysis.py`, etc.) reference SDC replication modules. This is acceptable given the historical context documented in ADR-056.
- Test count per MEMORY.md: 1,570 passed, 5 skipped. This is healthy.

---

## 7. Config File Standards

### 7.1 Config File Inventory

| File | Purpose | Has Inline Comments? | Has ADR/Doc Reference? |
|------|---------|---------------------|----------------------|
| `projection_config.yaml` | Main projection configuration | Yes (extensive) | Yes (ADR-010, ADR-059) |
| `data_sources.yaml` | Data source manifest | Yes (extensive) | Yes (ADR-016) |
| `evaluation_config.yaml` | Evaluation framework config | Yes | Yes (evaluation-blueprint.md) |
| `benchmark_evaluation_policy.yaml` | Benchmark gates/thresholds | Yes (descriptions per gate) | Yes (ADR-062, benchmarking roadmap) |
| `observatory_config.yaml` | Observatory system config | Yes | Minimal |
| `observatory_variants.yaml` | Observatory variant catalog | Not checked | -- |
| `observatory_recipes.yaml` | Observatory search recipes | Not checked | -- |
| `observatory_search_policy.yaml` | Search policy | Not checked | -- |
| `experiment_spec_schema.yaml` | Experiment spec contract | Not checked | -- |
| `experiment_log_schema.yaml` | Experiment log contract | Not checked | -- |
| `nd_brand.yaml` | ND brand colors | Yes (source noted) | N/A |
| `method_profiles/*.yaml` | Immutable method profiles | Per SOP-003 contract | Yes |
| `method_profiles/aliases.yaml` | Mutable alias map | Per SOP-003 contract | Yes |

### 7.2 Observations

- **`projection_config.yaml`** is well-documented with inline ADR references (e.g., `# ADR-010`, `# ADR-059`). This is the gold standard for config documentation.
- **`benchmark_evaluation_policy.yaml`** includes machine-readable `description` fields for every gate and threshold, making it self-documenting.
- **Method profiles** follow the SOP-003 contract (Section 6.1) with all required fields.
- **Observatory config files** have minimal documentation compared to the projection config. This is acceptable since they are newer and the observatory system is still maturing.

---

## 8. CRITICAL Finding Detail

### F-12: `example_usage.py` in Production Package

**File:** `cohort_projections/data/process/example_usage.py`
**SOP:** SOP-002 (data processing documentation)
**Severity:** CRITICAL

This file is a demonstration/example script that lives inside the production `cohort_projections/data/process/` package. It:
- Has only a 2-line docstring ("Example usage script for base_population.py")
- Contains no SOP-002 metadata
- Creates synthetic random data (`np.random.seed(42)`)
- Is imported via lazy loading in `__init__.py` (`load_example_usage_module()`)
- Has no test coverage

**Risk:** The file's presence in the production package is confusing. It could be mistaken for production code. Its synthetic data generation could be accidentally invoked.

**Recommendation:** Move to `docs/examples/` or `scripts/examples/`, or remove it entirely since the module's public API is documented in `__init__.py`.

---

## 9. Recommendations (Priority Order)

### High Priority

1. **Backfill Implementation Results for post-SOP-002 ADRs** (057, 059, 060, 062, 063). These are recently implemented ADRs that should have had Implementation Results added when work completed. ~1 hour effort.

2. **Add `DATA_SOURCE_NOTES.md` to `data/raw/housing/`**. The file `nd_place_housing_units.csv` was fetched by `fetch_census_housing_data.py` (ADR-060) and needs provenance documentation per SOP-002 Phase 2. ~15 minutes.

3. **Fix broken template path references** in SOP-001 and SOP README. Change `docs/sops/templates/` to `docs/governance/sops/templates/` in 4 locations. ~5 minutes.

### Medium Priority

4. **Add SOP-002 docstrings to `ingest_stcoreview.py` and `ingest_ves_data.py`**. These are active data processing scripts that produce files consumed by the pipeline. ~30 minutes each.

5. **Move or remove `example_usage.py`** from the production package. ~10 minutes.

6. **Retroactively document ADR-034 pipeline scripts** (`convert_popest_to_parquet.py`, `extract_popest_docs.py`, `archive_popest_raw_by_vintage.py`, `download_census_pep.py`, `build_popest_postgres.py`). These are complex data processing scripts that would benefit from SOP-002 documentation. ~2 hours total.

### Low Priority

7. **Normalize ADR-015 format** to match the current template, or add a note that it uses the pre-template format. This is cosmetic.

8. **Backfill Implementation Results for pre-SOP-002 ADRs** (001-035). This is a large effort (~30+ ADRs) with diminishing returns for older decisions. Consider doing this incrementally when each ADR is revisited.

9. **Add `## Scope` and `## Last Reviewed`** fields to ADRs that lack them. These are optional template additions used by newer ADRs (040+) that improve traceability.

---

## Appendix: SOP-002 Required Elements Checklist

For reference, the 9 required docstring elements from SOP-002 Phase 1:

```
1. One-line summary
2. Created: YYYY-MM-DD
3. ADR: NNN (Part X)
4. Author: who created it
5. Purpose (2-4 sentences, WHY)
6. Method (numbered steps, HOW)
7. Key design decisions (with rationale)
8. Validation results (actual numbers)
9. Inputs / Output / Usage
```

Scripts under 50 lines with a single function and no external data may use a shortened form (Purpose, Inputs, Output, Usage only).

---

*Audit completed: 2026-03-16*
*Auditor: Claude Opus 4.6 (1M context)*
