# CLAUDE.md

Quick reference for Claude Code. **For complete guidance, see [AGENTS.md](./AGENTS.md).**

Current status: 2026 projection development backlog (`PP-001` through `PP-004`) is closed as of 2026-03-01; use `DEVELOPMENT_TRACKER.md` for maintenance-state tasks and any newly opened work.

---

## Quick Commands

### Testing
```bash
pytest                          # Run all tests
pytest --cov                    # With coverage
pytest tests/unit/              # Unit tests only
```

### Code Quality
```bash
pre-commit run --all-files      # All quality checks
ruff check cohort_projections/  # Linting
ruff check --fix cohort_projections/  # Auto-fix
mypy cohort_projections/        # Type checking
```

### Data Sync
```bash
./scripts/bisync.sh             # Sync data between computers
python scripts/fetch_data.py    # Fetch from sibling repos
```

### Run Projections
```bash
python scripts/projections/run_all_projections.py
```

### Benchmarking

```bash
python scripts/analysis/run_experiment.py --spec <path>  # Full experiment pipeline
python scripts/analysis/run_experiment.py --spec <path> --dry-run  # Validate only
python scripts/analysis/build_experiment_dashboard.py     # Interactive results dashboard
python scripts/analysis/run_experiment_sweep.py --specs s1.yaml s2.yaml  # Batch sweep
python scripts/analysis/run_experiment_sweep.py --grid grid.yaml         # Parameter grid sweep
python scripts/analysis/run_experiment_sweep.py --pending                # Run all pending specs
```

### Projection Observatory

```bash
python scripts/analysis/observatory.py status          # Run inventory & catalog status
python scripts/analysis/observatory.py compare         # Full N-way comparison report
python scripts/analysis/observatory.py rank <metric>   # Rank by specific metric
python scripts/analysis/observatory.py recommend       # Next-experiment suggestions
python scripts/analysis/observatory.py run-pending     # Run all untested variants
python scripts/analysis/observatory.py run-pending --dry-run  # Preview what would run
python scripts/analysis/observatory.py diff <id1> <id2>  # Head-to-head run comparison
python scripts/analysis/observatory.py history         # Chronological experiment progression
python scripts/analysis/observatory.py report          # Generate HTML observatory report
python scripts/analysis/observatory.py refresh         # Rebuild results cache
python scripts/analysis/observatory.py --format json status  # Machine-readable output (json/csv)
```

---

## Session Workflow

### Starting
```bash
cd ~/workspace/demography/cohort_projections
direnv allow                     # First time only
git pull
./scripts/bisync.sh
uv sync
```

### After Changes
```bash
pytest
pre-commit run --all-files
git add . && git commit -m "..."
./scripts/bisync.sh
```

---

## Key Rules (Summary)

- **NEVER** hard-code file paths (use config)
- **NEVER** skip pre-commit hooks (`--no-verify`)
- **NEVER** commit data files to git
- **ALWAYS** activate virtual environment
- **ALWAYS** run tests before committing
- **ALWAYS** run bisync before switching computers
- **ALWAYS** include full metadata docstrings in data processing scripts ([SOP-002](./docs/governance/sops/SOP-002-data-processing-documentation.md))
- **ALWAYS** update `DATA_SOURCE_NOTES.md` when adding files to `data/raw/`
- **ALWAYS** update ADR status and add Implementation Results when work is complete
- **ALWAYS** update `docs/methodology.md` when changing formulas, rates, data sources, or projection logic

**Complete rules and workflow:** [AGENTS.md](./AGENTS.md)

---

## Documentation

| Document | Purpose |
|----------|---------|
| [AGENTS.md](./AGENTS.md) | Complete AI agent guidance |
| [docs/governance/sops/](./docs/governance/sops/) | Standard Operating Procedures |
| [docs/guides/](./docs/guides/) | Detailed how-to guides |
| [docs/governance/adrs/](./docs/governance/adrs/) | Architecture decisions |
| [DEVELOPMENT_TRACKER.md](./DEVELOPMENT_TRACKER.md) | Current project status |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-03-12 |
| **Version** | 2.5.0 |
| **Note** | Quick-reference wrapper for Claude Code. See AGENTS.md for complete guidance. |
