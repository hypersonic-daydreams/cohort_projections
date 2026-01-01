# Environment Setup Guide

Complete setup instructions for the development environment.

**Related**: [AGENTS.md](../../AGENTS.md) (Section 10)

---

## Quick Start

```bash
cd ~/workspace/demography/cohort_projections
direnv allow          # Auto-activates .venv on cd (first time only)
uv sync               # Install all dependencies
```

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

### Syncing Environment

```bash
uv sync                          # Install from pyproject.toml
uv sync --frozen                 # Use exact versions from uv.lock
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
- Loading: Use `cohort_projections.utils.config_loader`

```python
from cohort_projections.utils.config_loader import load_config

config = load_config()
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

### Tests fail with import errors

Ensure virtual environment is activated:
```bash
source .venv/bin/activate
# or
direnv allow
```

### Pre-commit hooks fail

Run hooks manually to see detailed errors:
```bash
pre-commit run --all-files
```

---

*Last Updated: 2026-01-01*
