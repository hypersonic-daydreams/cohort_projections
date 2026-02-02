# Navigation Guide

Quick reference for finding information in this repository.

**Last Updated**: 2026-02-02

---

## I want to...

### Get Started

| Goal | Location |
|------|----------|
| Set up my development environment | [docs/guides/environment-setup.md](./guides/environment-setup.md) |
| Understand the project | [AGENTS.md](../AGENTS.md) (Section 1) |
| See what's currently being worked on | [DEVELOPMENT_TRACKER.md](../DEVELOPMENT_TRACKER.md) |
| Run the projections | `python scripts/projections/run_all_projections.py` |

### Configure the System

| Goal | Location |
|------|----------|
| Change projection parameters | [config/projection_config.yaml](../config/projection_config.yaml) |
| Set up environment variables | [.env.example](../.env.example) |
| Understand all configuration options | [docs/guides/configuration-reference.md](./guides/configuration-reference.md) |
| Set up BigQuery | [docs/BIGQUERY_SETUP.md](./BIGQUERY_SETUP.md) |

### Work with Data

| Goal | Location |
|------|----------|
| Find where raw data goes | `data/raw/` |
| Understand data sources | [config/data_sources.yaml](../config/data_sources.yaml) |
| Sync data between computers | `./scripts/bisync.sh` |
| Fetch data from Census API | [docs/census_api_usage.md](./census_api_usage.md) |

### Run Tests

| Goal | Location |
|------|----------|
| Run all tests | `pytest` |
| Run with coverage | `pytest --cov` |
| Understand testing workflow | [docs/guides/testing-workflow.md](./guides/testing-workflow.md) |

### Work on the Journal Article

| Goal | Location |
|------|----------|
| Find latest article PDF | `journal_article_pdfs/` or check `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/CURRENT_VERSION.txt` |
| Understand SDC 2024 methodology | [docs/methodology_comparison_sdc_2024.md](./methodology_comparison_sdc_2024.md) |
| Run statistical analysis | `sdc_2024_replication/scripts/statistical_analysis/` |

### Understand Decisions

| Goal | Location |
|------|----------|
| Why was X designed this way? | [docs/governance/adrs/](./governance/adrs/) |
| Race/ethnicity categories | [ADR-007](./governance/adrs/007-race-ethnicity-categorization.md) |
| Data management strategy | [ADR-016](./governance/adrs/016-raw-data-management-strategy.md) |
| Configuration management | [ADR-005](./governance/adrs/005-configuration-management-strategy.md) |

### Modify the Codebase

| Goal | Location |
|------|----------|
| Add a new feature | Check [AGENTS.md](../AGENTS.md) (Section 3: Autonomy Framework) first |
| Run quality checks | `pre-commit run --all-files` |
| Fix linting errors | `ruff check --fix cohort_projections/` |
| Check types | `mypy cohort_projections/` |

---

## Directory Structure

### Root Level

| Directory | Purpose |
|-----------|---------|
| `cohort_projections/` | Main Python package |
| `config/` | Configuration files (YAML) |
| `data/` | All data files (not in git) |
| `docs/` | Documentation |
| `examples/` | Usage examples |
| `libs/` | Extracted utility packages |
| `scripts/` | Runnable scripts |
| `sdc_2024_replication/` | SDC 2024 journal article project |
| `tests/` | Test suite |

### Data Directory

| Directory | Purpose | Modifiable? |
|-----------|---------|-------------|
| `data/raw/` | Original source data | Never |
| `data/processed/` | Cleaned, harmonized inputs | Yes |
| `data/interim/` | Intermediate calculations | Yes |
| `data/projections/` | Final projection outputs | Yes |
| `data/exports/` | Formatted outputs (CSV, Excel) | Yes |

### Documentation

| Directory | Purpose |
|-----------|---------|
| `docs/guides/` | How-to guides |
| `docs/governance/adrs/` | Architecture Decision Records |
| `docs/governance/sops/` | Standard Operating Procedures |
| `docs/archive/` | Historical/deprecated docs |
| `docs/research/` | Research notes |

### Scripts

| Directory | Purpose |
|-----------|---------|
| `scripts/projections/` | Run projection pipeline |
| `scripts/setup/` | Environment setup scripts |
| `scripts/maintenance/` | Repository maintenance |
| `scripts/intelligence/` | Code inventory/documentation |

---

## Key Files

### Configuration

| File | Purpose |
|------|---------|
| `config/projection_config.yaml` | Main configuration |
| `config/data_sources.yaml` | Data source manifest |
| `config/nd_brand.yaml` | Visualization colors |
| `.env` | Environment variables (not in git) |
| `.env.example` | Template for .env |
| `.envrc` | direnv configuration |

### Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Complete AI agent instructions |
| `CLAUDE.md` | Quick reference for Claude Code |
| `DEVELOPMENT_TRACKER.md` | Current project status |
| `README.md` | Project overview |
| `REPOSITORY_INVENTORY.md` | File inventory and status |

### Development

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package and dependency definitions |
| `uv.lock` | Locked dependency versions |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `pytest.ini` | Pytest configuration |
| `ruff.toml` | Linter configuration |

---

## Common Workflows

### Starting a Session

```bash
cd ~/workspace/demography/cohort_projections
direnv allow          # First time only
git pull
./scripts/bisync.sh   # Sync data
uv sync               # Install dependencies
```

### Before Committing

```bash
pytest                          # Run tests
pre-commit run --all-files      # Quality checks
```

### Syncing Between Computers

```bash
./scripts/bisync.sh             # Before switching
# ... switch computers ...
./scripts/bisync.sh             # After switching
```

---

## Finding Specific Information

### Module Documentation

Each module in `cohort_projections/` has docstrings. Use:

```python
from cohort_projections.core import ProjectionEngine
help(ProjectionEngine)
```

### ADR Index

All Architecture Decision Records: [docs/governance/adrs/](./governance/adrs/)

Key ADRs by topic:
- Fertility processing: ADR-001
- Survival processing: ADR-002
- Migration processing: ADR-003
- Projection engine: ADR-004
- Configuration: ADR-005
- Data pipeline: ADR-006
- Race/ethnicity: ADR-007
- BigQuery: ADR-008
- Testing: ADR-011

### API Documentation

Function signatures and docstrings in source code:
- Core engine: `cohort_projections/core/`
- Data loading: `cohort_projections/data/`
- Geographic: `cohort_projections/geographic/`
- Output: `cohort_projections/output/`
- Utilities: `cohort_projections/utils/`

---

## Getting Help

1. **Check this navigation guide** - You're here
2. **Read AGENTS.md** - Complete context for AI assistants
3. **Search ADRs** - Design decisions are documented
4. **Check DEVELOPMENT_TRACKER.md** - Current status and blockers
5. **Look at tests** - Examples of how to use the code

---

*See also: [docs/INDEX.md](./INDEX.md) for auto-generated documentation index*
