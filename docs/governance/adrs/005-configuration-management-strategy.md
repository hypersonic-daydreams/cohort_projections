# ADR-005: Configuration Management Strategy

## Status
Accepted

## Date
2025-12-18

## Context

A population projection system has numerous configurable parameters: geographic scope, demographic categories, data sources, rate assumptions, scenario definitions, output formats, and validation thresholds. These parameters need to be managed consistently across all modules while remaining accessible and modifiable by users.

### Requirements

1. **Centralization**: Single source of truth for configuration
2. **Human-Readable**: Non-programmers should be able to modify settings
3. **Type Safety**: Prevent invalid configuration values
4. **Documentation**: Self-documenting format with comments
5. **Modularity**: Different modules access only relevant configuration sections
6. **Versioning**: Configuration should be version-controllable
7. **Environment-Specific**: Support different settings for dev/test/prod
8. **Default Values**: Sensible defaults when configuration is missing

### Challenges

1. **Multiple Consumers**: Core engine, data processors, output modules all need config
2. **Nested Structure**: Hierarchical settings (e.g., `scenarios.baseline.fertility`)
3. **Validation**: Need to catch configuration errors early
4. **Updates**: Configuration may change between projection runs
5. **Documentation**: Users need to understand what each setting does

## Decision

### Decision 1: YAML as Configuration Format

**Decision**: Use YAML for all configuration files, not JSON, TOML, INI, or Python code.

**Example**:
```yaml
project:
  name: "ND Population Projections 2025-2045"
  base_year: 2025
  projection_horizon: 20

demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90

  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"
```

**Rationale**:

**Why YAML**:
- **Human-Readable**: Minimal syntax, natural for hierarchical data
- **Comments**: Support `# comments` for documentation
- **Type-Rich**: Supports strings, numbers, booleans, lists, dictionaries
- **Python Integration**: Excellent library support (`PyYAML`)
- **Industry Standard**: Used by Kubernetes, Ansible, GitHub Actions

**Why Not JSON**:
- No comments (can't document settings inline)
- Verbose (requires quotes around all keys)
- Harder for humans to edit

**Why Not TOML**:
- Less readable for deeply nested structures
- Less widely adopted in data science
- Harder to learn for non-programmers

**Why Not Python Code**:
- Requires programming knowledge to modify
- Risk of code execution vulnerabilities
- Harder to validate statically

**Why Not INI**:
- Limited nesting support
- No standard for lists/arrays
- Less expressive

### Decision 2: Single Centralized Configuration File

**Decision**: Use one main configuration file (`config/projection_config.yaml`) rather than distributed configuration.

**Structure**:
```
config/
  projection_config.yaml          # Main configuration (ALL settings)
  # Optional auxiliary files:
  # fertility_schedules.yaml      # Pre-computed fertility schedules
  # mortality_schedules.yaml      # Pre-computed life tables
  # migration_assumptions.yaml    # Pre-computed migration patterns
```

**Rationale**:

**Single File Advantages**:
- **Simplicity**: One place to look for all settings
- **Consistency**: All modules see same configuration state
- **Versioning**: One file to track in git
- **Deployment**: One file to distribute/modify
- **Documentation**: Can document entire system in one place

**When Auxiliary Files Are Used**:
- Large pre-computed data (schedules, lookup tables)
- Optional overrides
- Environment-specific variations

**Why Not Distributed Configuration**:
- Harder to maintain consistency
- Settings scattered across multiple files
- Unclear precedence when values conflict
- More complex to document

### Decision 3: Nested Sections by Functional Area

**Decision**: Organize configuration into logical sections matching system modules.

**Section Structure**:
```yaml
# Project metadata
project: {...}

# Geographic scope
geography: {...}

# BigQuery integration
bigquery: {...}

# Demographic categories
demographics: {...}

# Rate sources and assumptions
rates:
  fertility: {...}
  mortality: {...}
  migration: {...}

# Scenario definitions
scenarios:
  baseline: {...}
  high_growth: {...}
  low_growth: {...}
  zero_migration: {...}

# Output configuration
output: {...}

# Validation rules
validation: {...}

# Logging configuration
logging: {...}
```

**Rationale**:
- **Modularity**: Each section consumed by specific modules
- **Clarity**: Settings grouped by purpose
- **Scalability**: Easy to add new sections
- **Documentation**: Can document sections separately
- **Discoverability**: Users can find relevant settings easily

**Mapping to Code Modules**:
- `demographics` → Core projection engine
- `rates` → Data processors (fertility, survival, migration)
- `bigquery` → BigQuery client utilities
- `output` → Output/export modules
- `logging` → Logger configuration

### Decision 4: ConfigLoader Class for Programmatic Access

**Decision**: Provide a `ConfigLoader` class for consistent configuration access across all modules.

**Implementation** (`cohort_projections/utils/config_loader.py`):
```python
class ConfigLoader:
    def __init__(self, config_dir=None):
        # Default to project_root/config/
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self._configs = {}  # Cache loaded configs

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration file (with caching)."""
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def get_projection_config(self) -> Dict[str, Any]:
        """Load main projection configuration."""
        return self.load_config("projection_config")

    def get_parameter(self, *keys: str, default=None) -> Any:
        """Get nested parameter with dot notation."""
        config = self.get_projection_config()
        value = config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else default
        return value if value is not None else default
```

**Usage**:
```python
from cohort_projections.utils.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.get_projection_config()

# Access nested values
base_year = loader.get_parameter('project', 'base_year', default=2025)
races = loader.get_parameter('demographics', 'race_ethnicity', 'categories')
```

**Rationale**:
- **Caching**: Load config once, reuse across modules
- **Consistent API**: All modules access config the same way
- **Default Handling**: Graceful fallback when keys missing
- **Path Resolution**: Automatic location of config directory
- **Type Safety**: Returns Python objects (not strings)

### Decision 5: Sensible Default Values Throughout Codebase

**Decision**: All configuration-dependent code should have sensible defaults when config is missing or incomplete.

**Pattern**:
```python
def process_fertility_rates(config=None):
    if config is None:
        config = load_projection_config()

    # Extract with defaults
    averaging_period = config.get('rates', {}).get('fertility', {}).get('averaging_period', 5)
    apply_to_ages = config.get('rates', {}).get('fertility', {}).get('apply_to_ages', [15, 49])

    # Use defaults based on demographic standards
    ...
```

**Common Defaults**:
- Base year: 2025 (current/recent)
- Projection horizon: 20 years (standard medium-term)
- Averaging period: 5 years (Census/ACS standard)
- Age range: 0-90 (demographic convention)
- Mortality improvement: 0.5% annual (historical average)

**Rationale**:
- **Robustness**: Code works even if config incomplete
- **Testing**: Can test without full configuration
- **Documentation**: Defaults communicate expected values
- **Fail-Safe**: System doesn't crash on missing config

**Error vs. Default Strategy**:
- **Critical Parameters**: Raise error if missing (e.g., data file paths)
- **Tuning Parameters**: Use defaults (e.g., averaging period, decimal places)

### Decision 6: No Environment-Specific Configuration Files

**Decision**: Use single configuration file for all environments, not separate dev/test/prod files.

**Rationale**:
- **Simplicity**: One configuration to maintain
- **Consistency**: Dev/prod use same settings (fewer surprises)
- **Demographic Projections**: Not like web apps with different DB connections
- **When Needed**: Can override specific values via environment variables or command-line args

**How to Handle Environment Differences**:
- **File Paths**: Use relative paths or `~` expansion
- **Credentials**: Store separately (not in config), reference path only
- **Data Sources**: Same sources for dev/prod (Census data is Census data)

**If Environment-Specific Config Needed** (future):
```python
# Potential approach (not currently implemented)
config_file = os.getenv('PROJECTION_CONFIG', 'projection_config.yaml')
loader = ConfigLoader()
config = loader.load_config(config_file.replace('.yaml', ''))
```

### Decision 7: Configuration Validation at Load Time

**Decision**: Validate configuration immediately when loaded, not when first used.

**Validation Checks**:
```python
def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration structure and values."""
    issues = []

    # Check required sections
    required_sections = ['project', 'demographics', 'rates', 'scenarios']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")

    # Check value ranges
    base_year = config.get('project', {}).get('base_year')
    if base_year and (base_year < 1900 or base_year > 2100):
        issues.append(f"base_year out of range: {base_year}")

    # Check demographic categories
    races = config.get('demographics', {}).get('race_ethnicity', {}).get('categories', [])
    if len(races) == 0:
        issues.append("No race/ethnicity categories defined")

    return issues
```

**When Validation Occurs**:
- When ConfigLoader loads file
- When user updates configuration
- When running data processors

**Rationale**:
- **Fail Fast**: Catch errors before running projection
- **Clear Messages**: Tell user exactly what's wrong
- **Prevent Garbage**: Invalid config → invalid results

### Decision 8: Inline Documentation via YAML Comments

**Decision**: Document configuration options directly in YAML file using comments.

**Example**:
```yaml
rates:
  fertility:
    source: "SEER"  # or "NVSS", "custom"
    averaging_period: 5  # years to average (2018-2022)
    assumption: "constant"  # "constant", "trending", "scenario"
    apply_to_ages: [15, 49]  # reproductive age range

  mortality:
    source: "SEER"  # or "CDC_life_tables"
    life_table_year: 2020  # base year for mortality rates
    improvement_factor: 0.005  # annual mortality improvement (0.5%)
    cap_survival_at: 1.0  # no survival rate > 100%
```

**Rationale**:
- **Self-Documenting**: Users understand settings without external docs
- **Context**: Comments explain valid values and defaults
- **Examples**: Show typical values inline
- **Maintenance**: Comments updated with code changes

**Documentation Standards**:
- Explain purpose of each setting
- Note valid values/ranges
- Provide examples
- Cite sources when relevant (e.g., "Census standard")

## Consequences

### Positive

1. **Simplicity**: Single YAML file is easy to understand and modify
2. **Human-Friendly**: Non-programmers can adjust settings
3. **Self-Documenting**: Comments explain all options
4. **Consistent Access**: ConfigLoader provides uniform API
5. **Version Control**: Configuration tracked in git
6. **Modularity**: Each section consumed by relevant modules
7. **Robustness**: Defaults prevent crashes on missing values
8. **Validation**: Early error detection prevents garbage results
9. **Flexibility**: Easy to add new parameters without code changes
10. **Transparency**: All settings visible in one place

### Negative

1. **Single Point of Failure**: Invalid config affects entire system
2. **Manual Editing**: Risk of YAML syntax errors
3. **No Type Checking**: YAML doesn't enforce types (mitigated by validation)
4. **Large File**: All settings in one file can become unwieldy
5. **No Secrets Management**: Credentials shouldn't be in config (addressed via separate credential files)
6. **Replication**: Some settings used by multiple modules (DRY violation)

### Risks and Mitigations

**Risk**: User introduces YAML syntax error, system fails to load
- **Mitigation**: YAML parser provides clear error messages with line numbers
- **Mitigation**: Provide example configuration file
- **Mitigation**: Include validation script to check config before running

**Risk**: Configuration file becomes too large and complex
- **Mitigation**: Can split into auxiliary files if needed (fertility_schedules.yaml, etc.)
- **Mitigation**: Current size (~125 lines) is manageable

**Risk**: Sensitive information (API keys, credentials) stored in config
- **Mitigation**: Store only paths to credential files, not credentials themselves
- **Mitigation**: Document security best practices
- **Mitigation**: Add config validation to warn if credentials detected

**Risk**: Configuration changes break existing projections/outputs
- **Mitigation**: Version control tracks all changes
- **Mitigation**: Save config snapshot with each projection run
- **Mitigation**: Document breaking changes in comments

## Alternatives Considered

### Alternative 1: Multiple Configuration Files by Module

**Description**: Separate config files for each module.

```
config/
  core_projection.yaml
  fertility_processing.yaml
  survival_processing.yaml
  migration_processing.yaml
  bigquery.yaml
  output.yaml
```

**Pros**:
- Smaller, more focused files
- Modules only load relevant config
- Can version modules independently

**Cons**:
- Settings scattered across files
- Unclear dependencies between configs
- More files to manage
- Harder to see full system configuration
- Potential inconsistencies between files

**Why Rejected**:
- Single file is sufficient for current complexity (~125 lines)
- Easier to maintain consistency
- Simpler mental model for users

### Alternative 2: JSON Configuration

**Description**: Use JSON instead of YAML.

```json
{
  "project": {
    "name": "ND Population Projections 2025-2045",
    "base_year": 2025,
    "projection_horizon": 20
  }
}
```

**Pros**:
- Strict syntax (fewer parse errors)
- Native to JavaScript (if building web UI)
- Built into Python (no dependencies)

**Cons**:
- No comments (can't document inline)
- Verbose (quotes around all keys)
- Harder for humans to read/edit
- Trailing comma errors frustrating

**Why Rejected**:
- Inline documentation critical for usability
- YAML more human-friendly
- PyYAML is stable and mature

### Alternative 3: Python Configuration Module

**Description**: Configuration as Python code.

```python
# config/projection_config.py
PROJECT_NAME = "ND Population Projections 2025-2045"
BASE_YEAR = 2025
PROJECTION_HORIZON = 20

DEMOGRAPHICS = {
    'age_groups': {
        'type': 'single_year',
        'min_age': 0,
        'max_age': 90
    }
}
```

**Pros**:
- Type checking via IDE
- Can compute values dynamically
- Familiar to Python developers

**Cons**:
- Requires Python knowledge to modify
- Security risk (code execution)
- Harder to validate statically
- Not suitable for non-programmers

**Why Rejected**:
- Need non-programmers to modify config
- Static data doesn't need code
- Security concerns

### Alternative 4: Database-Stored Configuration

**Description**: Store configuration in SQLite or PostgreSQL.

**Pros**:
- Centralized for multi-user systems
- Can query configuration
- Version history via database features

**Cons**:
- Adds database dependency
- Harder to edit (need SQL or UI)
- Overkill for single-user desktop application
- Harder to version control

**Why Rejected**:
- File-based config sufficient
- Version control (git) preferred
- No multi-user requirement

### Alternative 5: TOML Configuration

**Description**: Use TOML format.

```toml
[project]
name = "ND Population Projections 2025-2045"
base_year = 2025
projection_horizon = 20

[demographics.age_groups]
type = "single_year"
min_age = 0
max_age = 90
```

**Pros**:
- Strict syntax
- Comments supported
- Popular in Rust/Python communities

**Cons**:
- Less readable for deeply nested structures
- Arrays of tables syntax complex
- Less widely adopted than YAML

**Why Rejected**:
- YAML handles nested structures better
- More familiar to data scientists
- Better Python tooling

## Implementation Notes

### File Locations

**Main Configuration**:
```
/home/nigel/cohort_projections/config/projection_config.yaml
```

**ConfigLoader Module**:
```
/home/nigel/cohort_projections/cohort_projections/utils/config_loader.py
```

### Usage Patterns

**In Core Projection Engine**:
```python
from cohort_projections.utils.config_loader import ConfigLoader

class CohortComponentProjection:
    def __init__(self, ..., config=None):
        if config is None:
            loader = ConfigLoader()
            config = loader.get_projection_config()

        self.base_year = config.get('project', {}).get('base_year', 2025)
        self.max_age = config.get('demographics', {}).get('age_groups', {}).get('max_age', 90)
```

**In Data Processors**:
```python
from cohort_projections.utils.config_loader import load_projection_config

def process_fertility_rates(config=None):
    if config is None:
        config = load_projection_config()

    averaging_period = config['rates']['fertility']['averaging_period']
```

**In Scripts**:
```python
from cohort_projections.utils.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.get_projection_config()

# Override specific values if needed
config['project']['base_year'] = 2030  # Use updated base year
```

### Configuration Schema

**Full Schema** (documented in projection_config.yaml):
- `project`: Metadata and basic parameters
- `geography`: Geographic scope (state, counties, places)
- `bigquery`: BigQuery integration settings
- `demographics`: Age, sex, race/ethnicity categories
- `rates`: Data sources and assumptions for fertility/mortality/migration
- `scenarios`: Scenario definitions (baseline, high/low growth, etc.)
- `output`: Export formats and options
- `validation`: Validation rules and thresholds
- `logging`: Logging configuration

### Extending Configuration

**To Add New Settings**:
1. Add to appropriate section in `projection_config.yaml`
2. Document with inline comments
3. Update validation if needed
4. Update code to read new setting with sensible default
5. Update this ADR if architecturally significant

**Example**:
```yaml
# Add new setting to rates.migration section
rates:
  migration:
    domestic:
      method: "IRS_county_flows"
      averaging_period: 5
      smooth_extreme_outliers: true
      # NEW SETTING:
      outlier_threshold: 3.0  # standard deviations for outlier detection
```

## References

1. **YAML Specification**: https://yaml.org/spec/1.2/spec.html
2. **PyYAML Documentation**: https://pyyaml.org/wiki/PyYAMLDocumentation
3. **12-Factor App**: https://12factor.net/config - Configuration management best practices
4. **Configuration Management Patterns**: Martin Fowler - https://martinfowler.com/bliki/ConfigurationSynchronization.html

## Revision History

- **2025-12-18**: Initial version (ADR-005) - Configuration management strategy

## Related ADRs

- ADR-004: Core projection engine architecture (uses configuration)
- ADR-008: BigQuery integration design (configuration section)
- ADR-009: Logging and error handling (configuration section)
- All other ADRs reference configuration for their parameters

## Appendix: Complete Configuration Example

See `/home/nigel/cohort_projections/config/projection_config.yaml` for the full configuration file with all settings and inline documentation.

**Key Sections**:

```yaml
# Project metadata
project:
  name: "ND Population Projections 2025-2045"
  base_year: 2025
  projection_horizon: 20

# Geographic scope
geography:
  state: "38"  # North Dakota FIPS
  counties: "all"
  places: "all"

# Demographic structure
demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90
  sex: ["Male", "Female"]
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"

# Scenarios
scenarios:
  baseline:
    fertility: "constant"
    mortality: "improving"
    migration: "recent_average"
    active: true

# Output configuration
output:
  formats: ["parquet", "csv"]
  compression: "gzip"
  decimal_places: 2
```

This configuration structure supports the entire projection system while remaining accessible to users with varying technical backgrounds.
