# AGENTS.md

Canonical instruction set for all AI agents working on this codebase.

**Last Updated:** 2025-12-28 | **Version:** 1.0.0 | **Applies To:** Claude Code, GitHub Copilot, Cursor, all AI assistants

---

## 1. Project Identity

**North Dakota Population Projection System**

Cohort-component population projections for North Dakota at state, county, and place levels (2025-2045).

| Attribute | Value |
|-----------|-------|
| **Methodology** | Cohort-component (standard demographic projection method) |
| **Geographic Scope** | State (1), Counties (53), Places (~400) |
| **Projection Horizon** | 2025 (base) to 2045 (annual intervals) |
| **Demographic Detail** | Age x Sex x Race/Ethnicity cohorts |
| **Output Use** | Official reports, planning documents |

**Stack:** Python 3.11+, pandas, polars, YAML configs, pytest, Ruff, MyPy

**Philosophy:**
1. Demographic methodology must be rigorous and well-documented
2. Reproducibility over cleverness
3. Linear pipeline design (simpler than multi-source ingestion systems)
4. Every assumption documented in ADRs

---

## 2. Core Constraints (Non-Negotiable)

### NEVER:
1. Run code outside virtual environment (use `source .venv/bin/activate` or micromamba)
2. Hard-code file paths, URLs, or credentials (use `config/projection_config.yaml`)
3. Commit secrets or credentials (`.env`, API keys, service account files)
4. Bypass pre-commit hooks with `--no-verify`
5. Modify raw data files in `data/raw/` (read-only by convention)
6. Use `print()` for logging (use `logging` module)
7. Commit code that breaks tests

### ALWAYS:
1. Activate virtual environment before running Python
2. Run tests before committing: `pytest tests/`
3. Use type hints for all public functions
4. Document decisions in ADRs for methodology changes
5. Follow race/ethnicity categories from `projection_config.yaml`
6. Validate outputs against demographic plausibility thresholds

---

## 3. Agent Roles

This project uses a **linear pipeline** model. Unlike multi-source ingestion systems, most work is sequential.

### 3.1 Data Fetcher Agent
- Retrieves data from sibling repositories (`popest`, `ndx-econareas`, `maps`)
- Validates data structure matches expected schema
- Places raw files in `data/raw/{category}/`
- **Key script:** `scripts/fetch_data.py`
- **Output:** Raw CSV/Parquet files with provenance metadata

### 3.2 Processor Agent
- Transforms raw data into projection inputs
- Calculates fertility, mortality, and migration rates
- Harmonizes race/ethnicity codes to standard categories
- **Key modules:** `cohort_projections/data/process/`
- **Output:** `data/processed/*.parquet`

### 3.3 Projector Agent
- Runs cohort-component calculations
- Applies demographic components: fertility, mortality, migration, aging
- Projects forward year-by-year to horizon
- **Key modules:** `cohort_projections/core/`
- **Output:** `data/projections/*.parquet`

### 3.4 Documenter Agent
- Updates ADRs when methodology changes
- Maintains API documentation and docstrings
- Updates README and methodology docs
- **Key directory:** `docs/adr/`

---

## 4. Autonomy Framework

AI agents have bounded autonomy with clear decision tiers.

### Tier 1: Full Autonomy (Just Do It)
- Bug fixes that do not change methodology
- Code style improvements (formatting, naming)
- Test additions for existing functionality
- Documentation clarifications
- Logging improvements

**Action:** Implement and commit

### Tier 2: Autonomous with Documentation (Do + Document)
- New dependencies in `requirements.txt`
- Configuration schema changes in `projection_config.yaml`
- Algorithm modifications within existing methodology
- Data source changes (new file formats, column mappings)
- New validation rules or thresholds

**Action:** Implement, create/update ADR in `docs/adr/`, commit both

### Tier 3: Stop and Ask
- Data deletion or destructive operations
- Security-related changes (credentials, API keys)
- Breaking changes to output formats
- **Methodology changes affecting projection results**
- Changes to race/ethnicity categories
- Geographic hierarchy modifications
- Multi-year averaging period changes

**Action:** Document the issue, wait for human approval

---

## 5. Quality Standards

### 5.1 Code Quality
| Requirement | Standard |
|-------------|----------|
| Type hints | Required for all public functions |
| Docstrings | Google style, required for modules/classes/public functions |
| Test coverage | New functionality must have tests |
| Linting | Ruff must pass |
| Type checking | MyPy must pass |

### 5.2 Demographic Quality
| Check | Threshold |
|-------|-----------|
| Negative population | Error (never allowed) |
| Sex ratio | Warning if < 0.90 or > 1.10 |
| Age distribution | Warning if any 5-year group > 15% of total |
| Annual growth rate | Warning if < -5% or > 5% |
| TFR (Total Fertility Rate) | Warning if < 1.0 or > 3.0 |

### 5.3 ADR Requirements

Create an ADR when:
1. Methodology decisions affect projection results
2. New data sources are integrated
3. Validation thresholds are changed
4. Output format changes occur

ADR location: `docs/adr/{NNN}-{title}.md`

---

## 6. Data Conventions

### 6.1 Directory Structure
```
data/
  raw/           # Source data (NEVER modify)
    fertility/   # SEER fertility data
    mortality/   # Life tables, death rates
    migration/   # IRS flows, ACS mobility
    geographic/  # County/place reference files
  processed/     # Cleaned, harmonized inputs
  interim/       # Intermediate calculations
  projections/   # Final projection outputs
  exports/       # Formatted outputs (CSV, Excel)
```

### 6.2 File Naming
- Raw: `{source}_{data_type}_{year_range}.{ext}`
- Processed: `{component}_rates.parquet`
- Projections: `projection_{geography}_{scenario}_{timestamp}.parquet`

### 6.3 Data Immutability
- Files in `data/raw/` are **read-only**
- Re-fetch data rather than modifying raw files
- All transformations output to `data/processed/` or `data/interim/`

---

## 7. Demographic-Specific Guidelines

### 7.1 Race/Ethnicity Categories (Mandatory)
Use exactly these 6 categories (from `projection_config.yaml`):

| Code | Category |
|------|----------|
| 1 | White alone, Non-Hispanic |
| 2 | Black alone, Non-Hispanic |
| 3 | AIAN alone, Non-Hispanic |
| 4 | Asian/PI alone, Non-Hispanic |
| 5 | Two or more races, Non-Hispanic |
| 6 | Hispanic (any race) |

**Never** create new categories. Map source data to these 6.

### 7.2 Geographic Hierarchy
```
State (FIPS 38)
  County (53 counties, FIPS 38001-38105)
    Place (incorporated cities, CDPs)
      Balance (unincorporated remainder)
```

Validate: County sums must equal state total (within 1% tolerance).

### 7.3 Age Cohorts
- Single-year ages: 0, 1, 2, ... 89, 90+
- Standard 5-year groups: 0-4, 5-9, ... 80-84, 85+
- Fertility ages: 15-49
- Working ages: 18-64
- Elderly: 65+

### 7.4 Projection Parameters
| Parameter | Value |
|-----------|-------|
| Base year | 2025 |
| Horizon | 2045 |
| Interval | Annual |
| Scenarios | Baseline (primary), High, Low, Zero-migration |

---

## 8. Pre-commit Hook Policy

### Rules
1. **NEVER** use `--no-verify` to bypass hooks
2. **NEVER** disable hooks in configuration
3. If hooks fail, **fix the issues**

### When Hooks Fail

**Small number of issues (< 10):**
- Fix issues directly
- Re-run commit

**Large number of issues (10+):**
- Launch sub-agent with task:
  ```
  Fix all pre-commit hook failures:
  - [List specific issues: ruff errors, type errors, etc.]
  Run pre-commit after fixes to verify.
  Return summary of changes.
  ```

### Only Bypass If
User explicitly requests it for an urgent situation (document why).

---

## 9. Session Workflow

### At Session Start
1. Check `DEVELOPMENT_TRACKER.md` for current status
2. Review recent commits: `git log --oneline -10`
3. Verify environment: `source .venv/bin/activate`
4. Run tests to confirm working state: `pytest tests/ -q`

### During Session
1. Update `DEVELOPMENT_TRACKER.md` as tasks complete
2. Commit logical units of work (not one giant commit)
3. Run tests before each commit

### At Session End
1. Update `DEVELOPMENT_TRACKER.md` with session summary
2. Commit tracker changes with code changes
3. Document any blocking issues for next session

---

## 10. Environment Setup

### Virtual Environment
```bash
# Using micromamba (preferred)
micromamba create -n cohort_proj python=3.11
micromamba activate cohort_proj
pip install -r requirements.txt

# Or using venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration
- Primary config: `config/projection_config.yaml`
- Load with: `cohort_projections.utils.config_loader`
- Never hard-code values that belong in config

### Testing
```bash
pytest tests/                    # All tests
pytest tests/ -q                 # Quiet mode
pytest tests/test_core.py        # Specific module
pytest -k "fertility"            # By keyword
```

---

## 11. Essential Documentation

### Quick Reference
| Document | Purpose |
|----------|---------|
| `README.md` | Project overview, quick start |
| `CLAUDE.md` | Quick commands for Claude Code |
| `AGENTS.md` | This file - agent operating instructions |

### Technical Documentation
| Document | Purpose |
|----------|---------|
| `docs/adr/` | Architecture Decision Records |
| `cohort_projections/core/README.md` | Core engine documentation |
| `cohort_projections/data/process/README.md` | Data processing documentation |

### Key ADRs
| ADR | Topic |
|-----|-------|
| ADR-001 | Fertility rate processing |
| ADR-002 | Survival rate processing |
| ADR-003 | Migration rate processing |
| ADR-004 | Core projection engine architecture |
| ADR-007 | Race/ethnicity categorization |
| ADR-010 | Geographic scope and granularity |

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial AGENTS.md |

---

**Questions?**
- Commands: See `CLAUDE.md`
- Methodology: See `docs/adr/`
- Configuration: See `config/projection_config.yaml`
