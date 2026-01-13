# CLAUDE.md

Quick reference for Claude Code. **For complete guidance, see [AGENTS.md](./AGENTS.md).**

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

### Journal Article (SDC 2024 Replication)
```bash
cat sdc_2024_replication/scripts/statistical_analysis/journal_article/output/CURRENT_VERSION.txt
ls -lt journal_article_pdfs/*.pdf | head   # if `journal_article_pdfs/` exists locally
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
| **Last Updated** | 2026-01-13 |
| **Version** | 2.1.1 |
| **Note** | Quick-reference wrapper for Claude Code. See AGENTS.md for complete guidance. |
