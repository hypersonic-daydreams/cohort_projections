# CLAUDE.md

Quick reference for Claude Code. **For complete guidance, see [AGENTS.md](./AGENTS.md).**

Current status: All projection development packages (`PP-001` through `PP-009`) are complete. Active work items: `CF-001` College Fix Model Revision (ADR-061) and `PUB-2026` public projection release handoff. Marketing intake begins 2026-05-27; final public numbers are targeted for the week of 2026-06-01 after CF-001 disposition and any approved production rerun. Use `DEVELOPMENT_TRACKER.md` for maintenance-state tasks and newly opened work.

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

### Public Release Handoff

```bash
ls docs/plans/2026-public-projection-release-handoff/
```

Current PUB-2026 scope: standalone public PDF, baseline-led scenario framing, state/region/county only, no city/place publication, and one consolidated public Excel plus one consolidated CSV.

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

The live dashboard is the primary interface for the Observatory. It provides 7 interactive tabs (Command Center, Decision Brief, Scorecards, Projections, Horizon & Bias, Sensitivity, Experiment History) with auto-refreshing progress tracking for autonomous search sessions. The Command Center uses progressive disclosure: a one-click "Start Exploring" button launches autonomous search with smart defaults, and a live Search Progress card shows results — advanced controls are collapsed by default. Use it to monitor `search-auto` runs in real time, compare variants, and review results.

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

## Brand and visual identity — declared override, migration pending

Workspace authority: `/home/nhaarstad/workspace/libs/nd_brand` governs brand
(start at `processed/agent_reference.md`), **except as declared here**. See the brand bullet
in `/home/nhaarstad/workspace/CLAUDE.md` > "Permanent Workspace References".

- **Exception:** the interactive HTML report, the Panel observatory dashboard, and the
  experiment dashboard render in an **SDC palette**, not the ND Commerce palette — Navy
  `#1F3864`, Blue `#0563C1`, Teal `#00B0F0`, Red `#C00000`, with `SCENARIO_COLORS`,
  `TIER_COLORS`, growth/decline and sex colors, and a registered Plotly template. Defined in
  `scripts/exports/_report_theme.py`, imported by 12 modules; `cohort_projections/analysis/observatory/dashboard/theme.py`
  re-exports it. Font stack is Aptos/Segoe UI/Helvetica Neue/Arial.
- **Scope:** the HTML/Plotly report surfaces above. It does **not** extend to report prose,
  citations, Word or PowerPoint deliverables, logo use, or the Commerce-facing marketing
  handoff, all of which follow nd_brand.
- **Reason:** this repo's output is published by the **North Dakota State Data Center**
  (`README.md`: "Proprietary -- North Dakota State Data Center"), and the palette matches
  SDC's own deck. Whether an SDC publication should carry SDC or ND Commerce identity is an
  institutional question, not a styling one.
- **Decided:** 2026-09-09 (N. Haarstad) — override stands for now, recording a choice
  already in the artifacts.

**Migration is needed and not yet done.** Two things have to happen, in this order:

1. **Settle whose brand this publication carries** — SDC or ND Commerce. Until that is
   answered, do not "fix" `_report_theme.py` toward nd_brand tokens; you would be making an
   institutional decision by editing a constant. Ask Nigel.
2. **If the answer is ND Commerce,** migrate `scripts/exports/_report_theme.py` to import
   from nd_brand rather than hardcode, and expect real gaps: nd_brand has no sequential
   ramp, no dark mode, and only seven CVD-distinguishable categorical colors. The
   growth/decline pair here is red/green, which the workspace accessibility SOP forbids as a
   default signed encoding and `demography/demography_scratchpad/scripts/visual_accessibility_preflight.py`
   flags as an error — so that part needs fixing under **either** answer.

Two smaller items in the same area:

- `config/nd_brand.yaml` is an orphan: it carries the ND Commerce palette with the
  pre-correction names and **no code loads it** (only `config/README.md` and
  `docs/guides/configuration-reference.md` mention it). Delete it rather than migrate it.
- `docs/archive/observatory_wireframe/vendor/nd-brand.css` is a frozen pre-correction copy
  of nd_brand's Claude Design stylesheet. Leave it; it is an archived artifact.

Hex is the identifier — never match a brand color by name across a repo boundary. Six
friendly names in older workspace code now name a different swatch than the guidelines do.

---

## Key Rules (Summary)

- **NEVER** hard-code file paths (use config)
- **NEVER** skip pre-commit hooks (`--no-verify`)
- **NEVER** commit data files to git
- **ALWAYS** activate virtual environment
- **ALWAYS** run tests before committing
- **ALWAYS** run bisync before switching computers
- **ALWAYS** include full metadata docstrings in data processing scripts ([SOP-002](./docs/governance/sops/SOP-002-data-processing-documentation.md))
- **ALWAYS** write public-facing narrative prose (public PDF, explainers, FAQs) in the reserved trade-book voice ([SOP-005](./docs/governance/sops/SOP-005-public-facing-prose-voice.md)) — not for `methodology.md`/ADRs/code
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
| [SOP-005](./docs/governance/sops/SOP-005-public-facing-prose-voice.md) | Public-facing prose voice (reserved trade-book register) |
| [docs/guides/](./docs/guides/) | Detailed how-to guides |
| [docs/governance/adrs/](./docs/governance/adrs/) | Architecture decisions |
| [DEVELOPMENT_TRACKER.md](./DEVELOPMENT_TRACKER.md) | Current project status |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-05-27 |
| **Version** | 2.6.1 |
| **Note** | Quick-reference wrapper for Claude Code. See AGENTS.md for complete guidance. |
