# CLAUDE.md

Quick reference for Claude Code. **For complete guidance, see [AGENTS.md](./AGENTS.md).**

Current status: All projection development packages (`PP-001` through `PP-009`) are complete. Active work item: `CF-001` College Fix Model Revision (ADR-061). Use `DEVELOPMENT_TRACKER.md` for maintenance-state tasks and any newly opened work.

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
pre-commit run --all-files      # All quality checks (~3 min, runs ruff + mypy + pytest)
ruff check cohort_projections/  # Linting
ruff check --fix cohort_projections/  # Auto-fix
mypy cohort_projections/        # Type checking
```

> **Tip:** For quick verification, run `pytest` directly instead of `pre-commit run --all-files`. Pre-commit triggers ruff, mypy, and the full test suite, taking 3+ minutes.

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

**Live Dashboard (preferred starting point):**

```bash
python scripts/analysis/observatory_dashboard.py       # Launch live Panel dashboard (localhost:5006)
python scripts/analysis/observatory_dashboard.py --port 8080  # Custom port
```

The live dashboard is the primary interface for the Observatory. It provides 7 interactive tabs (Command Center, Experiments, History, Scorecards, Projections, Horizon & Bias, Sensitivity) with auto-refreshing progress tracking for autonomous search sessions. The Command Center uses progressive disclosure: a one-click "Start Exploring" button launches autonomous search with smart defaults, and a live Search Progress card shows results — advanced controls are collapsed by default. Use it to monitor `search-auto` runs in real time, compare variants, and review results.

**CLI (alternative / scripting):**

```bash
python scripts/analysis/observatory.py status          # Run inventory & catalog status
python scripts/analysis/observatory.py compare         # Full N-way comparison report
python scripts/analysis/observatory.py rank <metric>   # Rank by specific metric
python scripts/analysis/observatory.py recommend       # Next-experiment suggestions
python scripts/analysis/observatory.py run-pending     # Run all untested variants
python scripts/analysis/observatory.py run-pending --dry-run  # Preview what would run
python scripts/analysis/observatory.py run-recommended     # Run config-only recommendations
python scripts/analysis/observatory.py run-recommended --dry-run  # Preview recommendations
python scripts/analysis/observatory.py search-auto     # Unattended: plan → run → report
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
direnv allow                     # First time only — activates .venv
git pull
./scripts/bisync.sh
uv sync --extra dev --extra dashboard  # Install dev + dashboard deps
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
| [docs/guides/observatory-start-here.md](./docs/guides/observatory-start-here.md) | Observatory entry point and reading order |
| [docs/governance/sops/](./docs/governance/sops/) | Standard Operating Procedures |
| [docs/guides/](./docs/guides/) | Detailed how-to guides |
| [docs/governance/adrs/](./docs/governance/adrs/) | Architecture decisions |
| [DEVELOPMENT_TRACKER.md](./DEVELOPMENT_TRACKER.md) | Current project status |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-03-16 |
| **Version** | 2.6.0 |
| **Note** | Quick-reference wrapper for Claude Code. See AGENTS.md for complete guidance. |
