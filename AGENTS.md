# AGENTS.md

Canonical instruction set for all AI agents working on this codebase.

**Last Updated:** 2026-02-18 | **Version:** 1.5.0 | **Applies To:** Claude Code, GitHub Copilot, Cursor, all AI assistants

---

## 1. Project Identity

**North Dakota Population Projection System** — Official cohort-component population projections for North Dakota at state, county, and place levels.

### Current Focus: 2026 Cohort Projections

The primary work is developing the **2026 official population projections** for North Dakota. This includes refining methodology, validating against Census Bureau benchmarks, and producing publication-ready outputs.

| Attribute | Value |
|-----------|-------|
| Stack | Python 3.12, uv, pandas, polars, YAML configs, pytest, Ruff, MyPy |
| Methodology | Cohort-component (age × sex × race/ethnicity cohorts) |
| Geographic Scope | State (1), Counties (53), Places (~400) |
| Projection Horizon | 2025 to 2055 (annual) |

**Philosophy:** Rigor over cleverness. Reproducibility. Linear pipeline design. Every assumption in ADRs.

### Related Work

- **SDC 2024 Replication** (`sdc_2024_replication/`): Journal article analyzing immigration methodology. See Section 4 for file locations if needed.

---

## 2. Core Constraints

### NEVER
1. Run code outside virtual environment
2. Hard-code file paths, URLs, or credentials (use `config/projection_config.yaml`)
3. Commit secrets (`.env`, API keys, service account files)
4. Bypass pre-commit hooks with `--no-verify`
5. Modify raw data files in `data/raw/`
6. Use `print()` for logging (use `logging` module)
7. Commit code that breaks tests

### ALWAYS
1. Activate virtual environment before running Python
2. Run tests before committing
3. Use type hints for public functions
4. Document methodology changes in ADRs
5. Follow race/ethnicity categories from config

---

## 3. Autonomy Framework

### Tier 1: Full Autonomy (Just Do It)
Bug fixes, code style, test additions, documentation clarifications, logging improvements.

### Tier 2: Autonomous with Documentation (Do + Document)
New dependencies, config changes, algorithm modifications, data source changes, new validation rules.
→ Implement + create/update ADR

**ADR Process:**
- Registry: See [docs/governance/adrs/README.md](./docs/governance/adrs/README.md) for index, template, and naming conventions
- Before creating a new ADR, search existing ADRs to check if the topic is already covered
- If your ADR amends or extends an existing ADR, add an "Amended by" back-reference to the original
- Cross-reference related review documents in `docs/reviews/` when applicable

### Tier 3: Stop and Ask
Data deletion, security changes, breaking output formats, **methodology changes affecting results**, race/ethnicity category changes, geographic hierarchy modifications.
→ Document issue, wait for human approval

---

## 4. System Procedures

### Workflow Scripts
**ALWAYS check `scripts/` before inventing new commands.**
- **Projections**: Use `scripts/projections/run_all_projections.py` for full pipeline runs.
- **Backups**: Use `scripts/bisync.sh`. NEVER run raw `rclone` commands for syncing.
- **Maintenance**: Use provided scripts in `scripts/maintenance/`.

### SDC 2024 Journal Article (Reference Only)
If asked to work with the SDC 2024 journal article:
- **Latest PDF**: Check `journal_article_pdfs/` or `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/CURRENT_VERSION.txt`
- **Source files**: `sdc_2024_replication/scripts/statistical_analysis/journal_article/`

---

## 5. Quality Standards

| Requirement | Standard |
|-------------|----------|
| Type hints | Required for public functions |
| Docstrings | Google style, required for modules/classes/public functions |
| Test coverage | New functionality must have tests |
| Linting | Ruff must pass |
| Type checking | MyPy must pass |

**Demographic Quality Thresholds:**
- Negative population: Error (never allowed)
- Sex ratio: Warning if < 0.90 or > 1.10
- Annual growth rate: Warning if < -5% or > 5%

**Testing:** See [docs/guides/testing-workflow.md](./docs/guides/testing-workflow.md)

---

## 6. Data Conventions

### Directory Structure
```
data/
  raw/           # Source data (NEVER modify)
  processed/     # Cleaned, harmonized inputs
  interim/       # Intermediate calculations
  projections/   # Final outputs
  exports/       # Formatted outputs (CSV, Excel)
```

### Data Immutability
- Files in `data/raw/` are **read-only**
- Re-fetch data rather than modifying raw files
- All transformations output to `data/processed/` or `data/interim/`

### Shared Census Data Archive

Census Bureau population estimates are stored in a shared archive at
`~/workspace/shared-data/census/` (outside this repo, not in git). This is
the canonical source for Census PEP data used across projects.

**Structure:**
```
shared-data/census/popest/
├── catalog.yaml           # Master registry — check here FIRST
├── metadata/              # JSON per dataset (schema, MD5, row counts)
├── parquet/{vintage}/{level}/{file}.parquet   # Processed data
├── raw-archives/{vintage}-raw.zip            # Original CSV downloads
├── derived/docs/          # Extracted PDF documentation
└── docs/
    ├── census-ftp-structure.md      # FTP directory guide
    ├── vintage_differences.md       # Vintage comparison
    ├── ftp-key-reference.md         # Comprehensive FTP site reference
    ├── ftp-key-datasets.csv         # Full dataset index (8,700+ entries)
    └── ftp-key-index.json           # Filtered JSON index for projections
```

**How to find data:**
1. Read `catalog.yaml` for the dataset inventory (IDs, vintages, paths)
2. Read `metadata/{dataset-id}.json` for column schemas and row counts
3. Load parquet from `parquet/{vintage}/{level}/{file}.parquet`

**Key datasets for projections:**

| Dataset ID | Vintage | Content | Format Note |
|------------|---------|---------|-------------|
| `cc-est2024-agesex-all` | 2020-2024 | County age/sex (wide format) | Age cols: `AGE04_TOT`...`AGE85PLUS_FEM` |
| `cc-est2020int-alldata` | 2010-2020 | County age/sex/race (long format) | `AGEGRP` column, filter >0 |
| `co-est2024-alldata` | 2020-2024 | County totals + components | No age/sex detail |
| `co-est2009-alldata` | 2000-2009 | County totals + components | No age/sex detail |

**Important notes:**
- The 2024 age-sex file uses **wide format** (age columns), while the 2020
  intercensal file uses **long format** (AGEGRP rows). Data loaders must
  handle both.
- Some CSV files require `latin1` encoding (noted in catalog.yaml as
  `csv_read_encoding`).
- Census naming: `co-` = county totals, `cc-` = county characteristics
  (with age/sex detail), `sc-` = state characteristics, `sub-` = places.

### Data Sync
- Data syncs via rclone bisync (not git)
- Use `./scripts/bisync.sh` (never run rclone directly)
- See [ADR-016](./docs/governance/adrs/016-raw-data-management-strategy.md)

---

## 7. Demographic Guidelines

### Race/Ethnicity Categories (Mandatory)
Use exactly these 6 categories from `projection_config.yaml`:

| Code | Category |
|------|----------|
| 1 | White alone, Non-Hispanic |
| 2 | Black alone, Non-Hispanic |
| 3 | AIAN alone, Non-Hispanic |
| 4 | Asian/PI alone, Non-Hispanic |
| 5 | Two or more races, Non-Hispanic |
| 6 | Hispanic (any race) |

**Never** create new categories. Map source data to these 6.

### Geographic Hierarchy
State → County (53) → Place → Balance (unincorporated)

Validate: County sums must equal state total (within 1% tolerance).

### Age Cohorts
- Single-year ages: 0-89, 90+
- Fertility ages: 15-49
- Working ages: 18-64
- Elderly: 65+

---

## 8. Session Workflow

### At Session Start
1. Check `DEVELOPMENT_TRACKER.md` for current status
2. Review recent commits: `git log --oneline -10`
3. Run tests: `pytest tests/ -q`

### At Session End
1. Update `DEVELOPMENT_TRACKER.md`
2. Commit with code changes
3. Document blocking issues

---

## 9. Environment Setup

See [docs/guides/environment-setup.md](./docs/guides/environment-setup.md) for detailed setup.

**Quick start:**
```bash
direnv allow          # Auto-activates venv
uv sync               # Install dependencies
```

**Configuration:** `config/projection_config.yaml`

---

## 10. Documentation Index

### Guides (How-To)
| Guide | Purpose |
|-------|---------|
| [docs/guides/testing-workflow.md](./docs/guides/testing-workflow.md) | Test commands and patterns |
| [docs/guides/environment-setup.md](./docs/guides/environment-setup.md) | Environment and tooling setup |
| [docs/guides/configuration-reference.md](./docs/guides/configuration-reference.md) | Configuration options |
| [docs/guides/data-sources-workflow.md](./docs/guides/data-sources-workflow.md) | Data acquisition and processing |
| [docs/guides/troubleshooting.md](./docs/guides/troubleshooting.md) | Common issues and solutions |

### SOPs (Complex Workflows)
| SOP | Purpose |
|-----|---------|
| [docs/governance/sops/](./docs/governance/sops/) | Standard Operating Procedures index |
| [SOP-001](./docs/governance/sops/SOP-001-external-ai-analysis-integration.md) | External AI analysis integration workflow |

### ADRs (Why Decisions)
| ADR | Topic |
|-----|-------|
| [docs/governance/adrs/README.md](./docs/governance/adrs/README.md) | ADR index, template, and naming conventions |
| ADR-001 through ADR-003 | Fertility, survival, migration processing |
| ADR-007 | Race/ethnicity categorization |
| ADR-016 | Raw data management strategy |

### Reference
| Document | Purpose |
|----------|---------|
| [CLAUDE.md](./CLAUDE.md) | Quick commands for Claude Code |
| [DEVELOPMENT_TRACKER.md](./DEVELOPMENT_TRACKER.md) | Current project status |
| [docs/methodology_comparison_sdc_2024.md](./docs/methodology_comparison_sdc_2024.md) | SDC 2024 comparison |
| [docs/reviews/](./docs/reviews/) | Review and QA documents |

---

## 11. Repository Intelligence (Day 2 Operations)

This repository uses a PostgreSQL-backed intelligence system to track code status, documentation links, and execution history.

### For AI Agents:
1.  **Context**: Read [REPOSITORY_INVENTORY.md](REPOSITORY_INVENTORY.md) first to understand the codebase structure and identify active/deprecated files.
2.  **Status Check**: Before modifying a file, check `code_inventory` in `cohort_projections_meta` DB or the inventory file to ensure it is not deprecated.
3.  **Reproducibility**: When creating scripts that generate results for papers, wrap the logic in the `log_execution` context manager.
    ```python
    from cohort_projections.utils.reproducibility import log_execution
    with log_execution(__file__, parameters={...}):
        main()
    ```

### Automation:
-   **Pre-commit Hook**: Automatically updates the database when you commit changes. You do not need to manually update the inventory.
-   **Documentation**: Run `python scripts/intelligence/generate_docs_index.py` to refresh `docs/INDEX.md`.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.5.0 | 2026-02-18 | Added ADR process guidance; updated horizon to 2055; added reviews reference |
| 1.4.0 | 2026-02-02 | Refocused on 2026 cohort projections; de-emphasized SDC 2024; fixed section numbering; updated guides index |
| 1.3.0 | 2026-01-01 | Refactored to ~200 lines; extracted guides and added SOP references |
| 1.2.0 | 2025-12-31 | Consolidated test workflow and BigQuery content |
| 1.1.0 | 2025-12-29 | Updated for uv package management |
| 1.0.0 | 2025-12-28 | Initial AGENTS.md |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-18 |
| **Version** | 1.5.0 |
| **Status** | Current |
| **Applies To** | All AI Agents |
