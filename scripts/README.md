# Scripts Directory

Index of all scripts in the `scripts/` directory for the North Dakota Population Projection System.

---

## Quick Start

Run the complete projection pipeline:
```bash
./scripts/pipeline/run_complete_pipeline.sh
```

Or step-by-step:
```bash
python scripts/pipeline/00_prepare_processed_data.py       # CSV to Parquet
python scripts/pipeline/01_process_demographic_data.py --all  # Process rates
python scripts/pipeline/02_run_projections.py --all          # Run projections
python scripts/pipeline/03_export_results.py --all           # Export results
```

---

## Pipeline Scripts (`pipeline/`)

The main projection pipeline. See [pipeline/README.md](./pipeline/README.md) for detailed documentation.

| Script | Purpose | `--help` |
|--------|---------|----------|
| `00_prepare_processed_data.py` | Convert raw CSV files to Parquet format | Yes |
| `01_process_demographic_data.py` | Process fertility, survival, and migration rates | Yes |
| `02_run_projections.py` | Run cohort-component projections for all geographies | Yes |
| `03_export_results.py` | Export results to CSV/Excel and create distribution packages | Yes |
| `run_complete_pipeline.sh` | Run all pipeline steps in sequence | N/A |

---

## Data Management Scripts

| Script | Purpose | `--help` |
|--------|---------|----------|
| `fetch_data.py` | Fetch data from sibling repositories based on `data_sources.yaml` | Yes |
| `validate_data.py` | Validate all raw data files and generate validation report | No |
| `bisync.sh` | Sync data files with Google Drive via rclone bisync | Yes (`--help`) |
| `setup_rclone_bisync.sh` | Initial setup for rclone bisync (run once per machine) | Yes (`--help`) |

---

## Visualization and Reporting Scripts

| Script | Purpose | `--help` |
|--------|---------|----------|
| `generate_visualizations_and_reports.py` | Generate population pyramids, trend charts, and summary reports | No |
| `generate_article_pdf.py` | Generate PDF from research article markdown | No |

---

## Analysis and Comparison Scripts (`projections/`)

| Script | Purpose | `--help` |
|--------|---------|----------|
| `projections/run_sdc_2024_comparison.py` | Compare our projections with SDC 2024 methodology | No |

---

## Setup and Initialization Scripts (`setup/`)

| Script | Purpose | `--help` |
|--------|---------|----------|
| `setup/01_initialize_project.py` | Create project directories and validate config | No |
| `setup/02_test_bigquery_connection.py` | Test BigQuery API connection | No |
| `setup/03_explore_census_data.py` | Explore Census data via BigQuery | No |

---

## Testing Scripts

| Script | Purpose | `--help` |
|--------|---------|----------|
| `run_integration_test.py` | Run end-to-end integration test with Cass County data | No |
| `check_test_coverage.py` | Find orphaned tests and untested production modules | No |

---

## Data Processing Scripts

| Script | Purpose | `--help` |
|--------|---------|----------|
| `extract_sdc_fertility_rates.py` | Extract fertility rates from SDC 2024 source files | No |
| `process_nd_migration.py` | Process IRS migration data for ND counties | No |

---

## Database Scripts (`db/`)

| Script | Purpose | `--help` |
|--------|---------|----------|
| `db/generate_manifest_docs.py` | Generate DATA_MANIFEST.md from PostgreSQL database | Yes |
| `db/migrate_from_markdown.py` | Migrate manifest data from markdown to PostgreSQL | No |
| `db/backup_manifest_db.sh` | Backup the manifest PostgreSQL database | N/A |

---

## Intelligence Scripts (`intelligence/`)

Scripts for documentation management and code-to-documentation linking.

| Script | Purpose | `--help` |
|--------|---------|----------|
| `intelligence/generate_docs_index.py` | Generate docs/INDEX.md from repository database | No |
| `intelligence/link_documentation.py` | Link code inventory to documentation files | No |

---

## Maintenance Scripts (`maintenance/`)

| Script | Purpose | `--help` |
|--------|---------|----------|
| `maintenance/update_links.py` | Update markdown links after directory reorganization | No |
| `maintenance/apply_migration.py` | Apply database migrations | No |

---

## Git Hooks (`hooks/`)

| Script | Purpose |
|--------|---------|
| `hooks/rclone_reminder.sh` | Post-commit hook reminding to run bisync |

---

## `--help` Support Audit

Scripts with **Yes** in the `--help` column support `--help` and `--dry-run` flags. Scripts marked **No** should be updated to add argparse support for better discoverability.

**Well-documented scripts (argparse with --help):**
- `fetch_data.py`
- `pipeline/00_prepare_processed_data.py`
- `pipeline/01_process_demographic_data.py`
- `pipeline/02_run_projections.py`
- `pipeline/03_export_results.py`
- `db/generate_manifest_docs.py`

**Scripts needing improvement:**
- `validate_data.py` - runs immediately on import, no CLI
- `generate_visualizations_and_reports.py` - no CLI arguments
- `run_integration_test.py` - no CLI arguments
- `check_test_coverage.py` - no CLI arguments
- `extract_sdc_fertility_rates.py` - no CLI arguments
- `process_nd_migration.py` - no CLI arguments

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [pipeline/README.md](./pipeline/README.md) | Detailed pipeline documentation |
| [DEPENDENCIES.md](./DEPENDENCIES.md) | Data flow and script dependencies |
| [../CLAUDE.md](../CLAUDE.md) | Quick reference for AI agents |
| [../AGENTS.md](../AGENTS.md) | Complete AI agent guidance |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-02 |
| **Maintainer** | Project Team |
