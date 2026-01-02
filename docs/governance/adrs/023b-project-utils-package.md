# ADR-023b: Project Utils Package

## Purpose

This document specifies the extraction of configuration loading and logging setup utilities into a standalone `project-utils` Python package.

**Parent ADR**: [ADR-023: Package Extraction Strategy](023-package-extraction-strategy.md)

## Package Overview

| Attribute | Value |
|-----------|-------|
| **Package Name** | `project-utils` |
| **Repository** | `~/workspace/libs/project_utils` |
| **Python Module** | `project_utils` |
| **Primary Purpose** | Common boilerplate utilities for Python projects: YAML configuration loading and logging setup |

## Scope

### Feature Modules

#### 1. Configuration Loader (`project_utils.config`)

**Source**: `cohort_projections/utils/config_loader.py` (129 lines)

**Capabilities**:
- Load YAML configuration files with type hints
- Resolve relative paths within configuration
- Support for nested configuration parameter access
- Default value handling
- Configuration validation
- Environment variable interpolation

**Current Implementation**:
```python
# From cohort_projections/utils/config_loader.py
class ConfigLoader:
    """Loads and manages YAML configuration files."""

    def __init__(self, config_path: Path | str):
        """Load configuration from YAML file."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""

    def get_path(self, key: str) -> Path:
        """Get configuration value as resolved Path."""

    @property
    def config(self) -> dict:
        """Return full configuration dictionary."""
```

**Dependencies**: `pyyaml`

#### 2. Logger Setup (`project_utils.logging`)

**Source**: `cohort_projections/utils/logger.py` (94 lines)

**Capabilities**:
- Configure Python standard logging with sensible defaults
- File and console handler setup
- Log level configuration
- Rotating file handler support
- Structured log formatting
- Module-aware logger creation

**Current Implementation**:
```python
# From cohort_projections/utils/logger.py
def setup_logger(
    name: str,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console: bool = True,
    format_string: str | None = None
) -> logging.Logger:
    """Set up a logger with file and/or console handlers."""

def get_logger(name: str) -> logging.Logger:
    """Get a logger by name, creating if necessary."""
```

**Dependencies**: Python stdlib only

## Package Structure

```
project_utils/
├── pyproject.toml
├── uv.lock
├── .envrc
├── .gitignore
├── README.md
├── project_utils/
│   ├── __init__.py
│   ├── config.py                    # Configuration loading
│   ├── logging.py                   # Logger setup
│   └── py.typed                     # PEP 561 marker
├── docs/
│   ├── configuration.md
│   └── logging.md
└── tests/
    ├── test_config.py
    ├── test_logging.py
    └── fixtures/
        ├── sample_config.yaml
        └── nested_config.yaml
```

## API Design

### Configuration API

```python
from project_utils import ConfigLoader

# Load configuration
config = ConfigLoader("config/settings.yaml")

# Access values with dot notation
db_host = config.get("database.host", default="localhost")
db_port = config.get("database.port", default=5432)

# Get paths (resolved relative to config file location)
data_dir = config.get_path("paths.data_directory")
output_dir = config.get_path("paths.output_directory")

# Access full config
all_settings = config.config
```

### Logging API

```python
from project_utils import setup_logger, get_logger

# Set up a logger with file and console output
logger = setup_logger(
    name="my_application",
    log_file="logs/app.log",
    level=logging.DEBUG,
    console=True
)

# Use the logger
logger.info("Application started")
logger.error("Something went wrong", exc_info=True)

# Get existing logger elsewhere in code
logger = get_logger("my_application")
```

### Combined Usage Pattern

```python
from project_utils import ConfigLoader, setup_logger

# Typical application initialization
config = ConfigLoader("config/settings.yaml")

logger = setup_logger(
    name=config.get("app.name", "application"),
    log_file=config.get_path("logging.file"),
    level=getattr(logging, config.get("logging.level", "INFO"))
)

logger.info(f"Loaded configuration from {config.config_path}")
```

## pyproject.toml

```toml
[project]
name = "project-utils"
version = "0.1.0"
description = "Common utilities for Python projects: configuration and logging"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
```

## Migration Plan

### Phase 1: Package Scaffolding (15 min)

1. Create repository at `~/workspace/libs/project_utils`
2. Initialize with `uv init`
3. Create `.envrc` per REPOSITORY_STANDARDS.md
4. Add `pyyaml` dependency

### Phase 2: Code Migration (30 min)

1. Copy `config_loader.py` to `project_utils/config.py`
2. Copy `logger.py` to `project_utils/logging.py`
3. Remove cohort_projections-specific references (if any)
4. Add `__init__.py` with clean exports
5. Add `py.typed` marker for type checking

### Phase 3: Testing (30 min)

1. Create test fixtures (sample YAML files)
2. Write tests for ConfigLoader
3. Write tests for logger setup
4. Verify 100% coverage on core functionality

### Phase 4: Documentation (15 min)

1. Create README.md with quick start
2. Document configuration format expectations
3. Document logging options

### Phase 5: Integration (15 min)

1. Add package to cohort_projections as editable dependency
2. Update imports in cohort_projections:
   ```python
   # Before
   from cohort_projections.utils.config_loader import ConfigLoader
   from cohort_projections.utils.logger import setup_logger

   # After
   from project_utils import ConfigLoader, setup_logger
   ```
3. Verify cohort_projections tests pass
4. Update REPOSITORY_INVENTORY.md

## Dependencies

### Required

- `pyyaml>=6.0` - YAML parsing

### Development Only

- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting
- `mypy` - Type checking

## Why This Package First?

This package is extracted first (Phase 1a) because:

1. **Smallest scope**: Only 2 files, ~220 lines total
2. **Zero coupling**: No project-specific logic whatsoever
3. **Highest reuse**: Every Python project needs config and logging
4. **Workflow validation**: Proves the extraction process before larger packages
5. **Immediate value**: Can be used by other projects immediately

## Success Criteria

The extraction is complete when:

1. [ ] Package is installable via `uv add --editable ../libs/project_utils`
2. [ ] ConfigLoader works identically to current implementation
3. [ ] Logger setup works identically to current implementation
4. [ ] All tests pass with >95% coverage
5. [ ] cohort_projections imports updated and tests pass
6. [ ] Package is listed in REPOSITORY_INVENTORY.md
7. [ ] Type hints are complete (mypy passes in strict mode)

## Future Enhancements

After initial extraction, consider:

1. **Environment variable interpolation**: `${VAR_NAME}` syntax in YAML
2. **Multiple config file merging**: Base + environment-specific configs
3. **JSON config support**: Alternative to YAML
4. **Structured logging**: JSON log output option
5. **Log rotation**: Built-in rotating file handler configuration
6. **Secrets handling**: Integration with `.env` files

## Comparison to Alternatives

| Feature | project-utils | python-dotenv | hydra | dynaconf |
|---------|---------------------|---------------|-------|----------|
| YAML config | Yes | No | Yes | Yes |
| Logging setup | Yes | No | No | No |
| Complexity | Minimal | Minimal | High | Medium |
| Dependencies | 1 (pyyaml) | 0 | Many | Several |
| Learning curve | None | None | Steep | Medium |

This package fills the gap between "raw stdlib" and "full framework" - it provides just enough convenience without framework lock-in.

## Related Documents

- [ADR-023](023-package-extraction-strategy.md): Parent extraction strategy
- [ADR-005](005-configuration-management-strategy.md): Original configuration design
- [ADR-009](009-logging-error-handling-strategy.md): Original logging design
- [config_loader.py](../../cohort_projections/utils/config_loader.py): Source code
- [logger.py](../../cohort_projections/utils/logger.py): Source code
