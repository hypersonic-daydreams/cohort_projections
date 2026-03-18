# Environment Setup Guide

Complete setup instructions for the development environment.

**Related**: [AGENTS.md](../../AGENTS.md) (Section 10)

---

## Quick Start

```bash
cd ~/workspace/demography/cohort_projections
direnv allow                      # Auto-activates .venv on cd (first time only)
uv sync --extra dev --extra dashboard   # Install all dev + dashboard dependencies
```

> **Important:** Plain `uv sync` only installs core dependencies. You need
> `--extra dev` for pytest/ruff/mypy and `--extra dashboard` for the
> Observatory Panel dashboard.  See [Dependency Groups](#dependency-groups)
> for the full list.

---

## Virtual Environment

### With direnv (Preferred)

The project uses direnv for automatic virtual environment activation:

```bash
# First time setup
direnv allow

# Environment activates automatically when you cd into the project
cd ~/workspace/demography/cohort_projections
# (venv activates automatically)
```

### Manual Activation

```bash
uv sync                          # Creates .venv and installs dependencies
source .venv/bin/activate        # Activate the environment
```

---

## Package Management (uv)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Adding Dependencies

```bash
uv add <package>                 # Add to pyproject.toml and install
uv add --dev <package>           # Add as dev dependency
uv add "package>=1.0,<2.0"       # With version constraints
```

### Dependency Groups

The project defines optional dependency groups in `pyproject.toml`.
Install the ones you need with `--extra`:

| Group | Contents | When needed |
|-------|----------|-------------|
| `dev` | pytest, ruff, mypy, ipython, jupyter | Development and testing |
| `dashboard` | panel, plotly | Projection Observatory UI |
| `viz` | matplotlib, seaborn, plotly | Visualization scripts |
| `stats` | statsmodels, scipy, arch, scikit-learn | Statistical analysis |
| `bayesian` | pymc, arviz | Bayesian inference (Phase B) |
| `geo` | geopandas, shapely | Geospatial exports |
| `pdf_export` | weasyprint, markdown | PDF report generation |
| `excel_io` | xlrd | Legacy .xls file support |
| `all` | Everything above | Full environment |

```bash
# Typical development setup
uv sync --extra dev --extra dashboard

# Full environment with all optional packages
uv sync --extra all
```

### Syncing Environment

```bash
uv sync --extra dev --extra dashboard  # Recommended for development
uv sync --frozen                       # Use exact versions from uv.lock
```

### Updating Dependencies

```bash
uv lock --upgrade                # Update all packages
uv lock --upgrade-package pandas # Update specific package
```

---

## Configuration

### Primary Configuration File

- Location: `config/projection_config.yaml`
- Loading: Use `cohort_projections.utils`

```python
from cohort_projections.utils import load_projection_config

config = load_projection_config()
project_id = config["bigquery"]["project_id"]
```

### Never Hard-code Values

Values that belong in config:
- File paths
- API endpoints
- Thresholds and parameters
- Geographic codes
- Race/ethnicity categories

---

## Pre-commit Hooks

### Installation

```bash
pre-commit install               # Install hooks
```

### Running Manually

```bash
pre-commit run --all-files       # Run all hooks on all files
pre-commit run ruff              # Run specific hook
```

### Hooks Configured

| Hook | Purpose |
|------|---------|
| trailing-whitespace | Remove trailing whitespace |
| end-of-file-fixer | Ensure files end with newline |
| check-yaml | Validate YAML syntax |
| check-json | Validate JSON syntax |
| check-toml | Validate TOML syntax |
| ruff | Linting |
| ruff-format | Code formatting |
| mypy | Type checking |
| pytest | Run fast tests |

---

## BigQuery Integration (Optional)

The project can optionally use Google BigQuery for Census/demographic data access.

### Configuration

```yaml
# In projection_config.yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  dataset_id: "demographic_data"
```

### Setup

See [docs/BIGQUERY_SETUP.md](./BIGQUERY_SETUP.md) for detailed instructions.

---

## Data Synchronization

Data files are synced via rclone bisync (not git).

### Sync Command

```bash
./scripts/bisync.sh              # Wrapper script (always use this)
```

### When to Sync

- Before starting work on a different machine
- After making changes to data files
- Before switching computers

See [ADR-016](../adr/016-raw-data-management-strategy.md) for data management details.

---

## IDE Configuration

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- Even Better TOML

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

### PyCharm

1. Set Project Interpreter to `.venv/bin/python`
2. Enable Ruff plugin for linting
3. Configure pytest as test runner

---

## Troubleshooting

### "Command not found: uv"

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "direnv: error .envrc is blocked"

Allow direnv for this directory:
```bash
direnv allow
```

### Wrong Python on PATH (conda conflict)

If `which python` shows a conda path (e.g. `/home/user/miniconda3/bin/python`)
instead of `.venv/bin/python`, your shell is using conda's Python rather than
the project's uv-managed venv.  This causes `ModuleNotFoundError` for packages
that are installed in `.venv` but not in conda.

Fix:
```bash
direnv allow                     # Preferred — auto-activates .venv
# or
source .venv/bin/activate        # Manual activation
# or use the venv Python directly
.venv/bin/python <script>
```

### Tests fail with import errors

Ensure virtual environment is activated **and** dev dependencies are installed:
```bash
direnv allow                     # or: source .venv/bin/activate
uv sync --extra dev              # Installs pytest, ruff, mypy, etc.
```

### Observatory dashboard won't start (ModuleNotFoundError: panel)

The dashboard requires the `dashboard` dependency group:
```bash
uv sync --extra dashboard
```

### Pre-commit hooks fail

Run hooks manually to see detailed errors:
```bash
pre-commit run --all-files
```

The pre-commit mypy hook runs in its own isolated environment with
`pandas-stubs`, which is stricter than running `mypy` directly.  If mypy
passes locally but fails in the hook, check for pandas type-stub issues
(e.g. `pd.isna` on `object` types, `Series.rename` with a dict argument).

### Observatory search fails with "dirty checkout"

The autonomous search refuses to run if `git status` shows uncommitted
changes.  Commit or stash your changes first, then retry.

---

## WSL-Specific Notes

When running on Windows Subsystem for Linux:

- **Browser opening**: The Observatory dashboard launcher detects WSL and
  opens the browser via `cmd.exe /c start` to route to Windows Chrome
  instead of WSL's Chromium.
- **direnv**: May need to be installed separately in WSL
  (`sudo apt install direnv`) and hooked into your shell
  (`eval "$(direnv hook bash)"` in `.bashrc`).
- **Git environment variables**: Pre-commit hooks set `GIT_DIR`,
  `GIT_INDEX_FILE`, etc.  The sandbox manager strips these to prevent
  interference with mirror/worktree operations.

---

*Last Updated: 2026-03-18*
