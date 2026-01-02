# ADR-009: Logging and Error Handling Strategy

## Status
Accepted

## Date
2025-12-18

## Context

A population projection system involves complex data processing, mathematical calculations, and multi-stage pipelines. Comprehensive logging and error handling are essential for debugging, monitoring, validation, and user support.

### Requirements

1. **Debugging**: Developers need detailed information to diagnose issues
2. **Monitoring**: Users need to track projection progress
3. **Audit Trail**: Document data transformations and decisions
4. **Error Diagnosis**: Clear messages when something goes wrong
5. **Validation Tracking**: Record data quality warnings
6. **Performance Monitoring**: Track processing time for optimization
7. **User-Friendly**: Non-technical users can understand errors

### Challenges

1. **Multiple Modules**: Consistent logging across 10+ modules
2. **Log Levels**: Appropriate use of DEBUG, INFO, WARNING, ERROR
3. **File vs. Console**: Where to write logs
4. **Performance**: Logging shouldn't slow down processing significantly
5. **Error Types**: Distinguish between fatal errors and warnings
6. **User Messages**: Technical precision vs. user-friendly explanations

## Decision

### Decision 1: Python Standard Logging Module (Not Print Statements)

**Decision**: Use Python's built-in `logging` module for all output, not print statements.

**Pattern**:
```python
import logging
logger = logging.getLogger(__name__)

# ❌ AVOID: Print statements
print("Processing fertility data...")
print(f"Error: {error}")

# ✅ USE: Logging
logger.info("Processing fertility data...")
logger.error(f"Failed to process: {error}")
```

**Rationale**:
- **Log Levels**: Can filter by severity (DEBUG, INFO, WARNING, ERROR)
- **Configurability**: Can redirect output without code changes
- **Timestamps**: Automatic timestamps on all messages
- **Module Context**: Knows which module logged message
- **Flexibility**: Can log to file, console, or both
- **Standard Practice**: Industry standard for Python applications

**Why Not Print**:
- Can't filter by level
- No timestamps
- Can't redirect to file easily
- Mixes output with actual results
- Unprofessional

### Decision 2: Centralized Logger Configuration

**Decision**: Configure all loggers through centralized `setup_logger()` function with configuration file control.

**Implementation** (`cohort_projections/utils/logger.py`):
```python
def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Args:
        name: Logger name (typically __name__)
        log_level: Level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_to_file: Whether to write to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Format: timestamp - module - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

**Configuration** (in `projection_config.yaml`):
```yaml
logging:
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_to_file: true               # Write to file in addition to console
  log_directory: "logs"           # Directory for log files
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**Usage in Modules**:
```python
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)

logger.info("Starting projection...")
logger.debug(f"Configuration: {config}")
logger.warning("Missing data filled with defaults")
logger.error("Failed to load file")
```

**Rationale**:
- **Consistency**: All modules log same way
- **Configurability**: Change log level without code changes
- **Dual Output**: Console (immediate) + file (archive)
- **Per-Module Files**: Each module gets own log file
- **Timestamp**: All messages timestamped
- **Easy to Use**: One-line setup per module

### Decision 3: Hierarchical Log Levels Philosophy

**Decision**: Use log levels according to clear semantic definitions.

**Log Level Definitions**:

**DEBUG** (detailed diagnostic information):
- Variable values during processing
- Intermediate calculation results
- Configuration values loaded
- Data structure sizes
- **When to use**: Diagnosing specific bugs
- **Audience**: Developers

**INFO** (informational progress messages):
- Processing started/completed
- Files loaded/saved
- Summary statistics
- Major milestones
- **When to use**: Normal operation
- **Audience**: Users, operators

**WARNING** (something unusual but not fatal):
- Missing data filled with defaults
- Rates outside typical range (but plausible)
- Validation issues that don't prevent processing
- Deprecated features used
- **When to use**: Recoverable issues requiring attention
- **Audience**: Users, analysts

**ERROR** (something failed):
- File not found
- Invalid data detected
- Required columns missing
- Calculation failed
- **When to use**: Operation failed, results invalid
- **Audience**: Users, developers

**CRITICAL** (system failure):
- Fatal errors preventing execution
- Unrecoverable data corruption
- **When to use**: Rarely (system-level failures)
- **Audience**: Developers, system administrators

**Examples**:
```python
logger.debug(f"Loaded {len(df)} rows from {filepath}")
logger.info("Processing fertility rates for 2018-2022")
logger.warning("TFR for Hispanic group (2.45) above typical range (1.3-2.3)")
logger.error("Required column 'age' not found in fertility data")
logger.critical("Cannot connect to required data source")
```

**Rationale**:
- **Clarity**: Clear guidelines prevent confusion
- **Filterability**: Can set level to see only relevant messages
- **Debugging**: DEBUG for development, INFO for production
- **Alerting**: ERROR/CRITICAL trigger intervention

### Decision 4: Log to Both File and Console

**Decision**: Write logs to both console (stdout) and file simultaneously.

**Benefits by Output**:

**Console (stdout)**:
- **Immediate Feedback**: See progress in real-time
- **User-Friendly**: Users running scripts see status
- **CI/CD Integration**: Captured by build systems
- **Interactive**: Can monitor long-running processes

**File (logs/)**:
- **Archive**: Permanent record of runs
- **Debugging**: Review after completion
- **Comparison**: Compare logs from different runs
- **Automation**: Parse logs programmatically

**File Organization**:
```
logs/
  cohort_projections_core_cohort_component.log
  cohort_projections_data_process_fertility_rates.log
  cohort_projections_data_process_survival_rates.log
  cohort_projections_data_fetch_census_api.log
  ...
```

**One log file per module** named: `{module_path}.log`

**Rationale**:
- **Best of Both**: Immediate visibility + permanent archive
- **Debugging**: File logs persist after script ends
- **Module Isolation**: Each module's logs in separate file
- **Easy Debugging**: Know which module had issues

**Disk Space**: Log files typically small (< 1 MB per run), not a concern.

### Decision 5: Defensive Programming with Explicit Validation

**Decision**: Validate inputs explicitly and fail fast with clear error messages, not implicit assumptions.

**Pattern**:
```python
def process_fertility_rates(df, config):
    """Process fertility rates."""
    logger.info("Validating input data...")

    # ✅ EXPLICIT VALIDATION with clear errors
    required_columns = ['age', 'race', 'fertility_rate']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        error_msg = f"Missing required columns: {missing}. Found: {df.columns.tolist()}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if df.empty:
        error_msg = "Input DataFrame is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if (df['fertility_rate'] < 0).any():
        error_msg = "Negative fertility rates detected"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Input validation passed")

    # ❌ AVOID: Implicit assumptions
    # Process without checking → cryptic errors later
```

**Error Message Quality**:
```python
# ❌ POOR: Vague error
raise ValueError("Invalid data")

# ✅ GOOD: Specific, actionable error
raise ValueError(
    f"Fertility rate out of range: {rate} at age {age}. "
    f"Expected range: 0.0-0.15. "
    f"Check source data at: {source_file}"
)
```

**Rationale**:
- **Fail Fast**: Catch errors at source, not downstream
- **Clear Messages**: Users know exactly what's wrong
- **Actionable**: Error messages guide fixes
- **Debugging**: Logs show exact failure point
- **Data Quality**: Forces attention to data issues

### Decision 6: Error vs. Warning Philosophy

**Decision**: Use errors for fatal issues (can't continue), warnings for issues that can be worked around.

**ERROR** (raise exception):
- Missing required data
- Invalid data types
- Calculation failures
- File not found
- Configuration errors

**WARNING** (log but continue):
- Missing optional data (filled with defaults)
- Values outside typical range (but plausible)
- Deprecated feature usage
- Performance concerns
- Validation issues that don't prevent processing

**Example**:
```python
# WARNING: Can work around
if 'population' not in df.columns:
    logger.warning("'population' column missing, using unweighted average")
    weighted_avg = False
else:
    weighted_avg = True

# ERROR: Cannot continue
if 'fertility_rate' not in df.columns:
    error_msg = "Required column 'fertility_rate' missing"
    logger.error(error_msg)
    raise ValueError(error_msg)

# WARNING: Unusual but possible
if tfr > 3.0:
    logger.warning(f"TFR ({tfr:.2f}) unusually high, verify source data")

# ERROR: Impossible
if (survival_rate > 1.0).any():
    logger.error("Survival rate > 1.0 detected (impossible)")
    raise ValueError("Invalid survival rates")
```

**Rationale**:
- **User Experience**: Don't fail unnecessarily
- **Robustness**: Handle recoverable issues gracefully
- **Transparency**: Warnings document assumptions/workarounds
- **Safety**: Errors prevent invalid results

### Decision 7: Structured Logging for Key Events

**Decision**: Use structured, consistent log messages for key events to enable parsing and monitoring.

**Structured Patterns**:

**Processing Start/End**:
```python
logger.info(f"Processing {data_type} started: {input_file}")
logger.info(f"Processing {data_type} complete: {output_file} ({len(df)} rows, {time:.2f}s)")
```

**Data Loading**:
```python
logger.info(f"Loading data from: {filepath}")
logger.debug(f"Loaded {len(df)} rows, {len(df.columns)} columns")
logger.debug(f"Columns: {df.columns.tolist()}")
logger.debug(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
```

**Validation**:
```python
logger.info("Validation started")
if issues:
    for issue in issues:
        logger.warning(f"Validation issue: {issue}")
logger.info(f"Validation {'passed' if is_valid else 'failed'} ({len(issues)} issues)")
```

**Data Transformations**:
```python
logger.info(f"Averaging over {years} years ({year_range[0]}-{year_range[1]})")
logger.info(f"Mapped {len(unmapped)} unmapped race codes")
logger.info(f"Filled {missing_count} missing cohorts with defaults")
```

**Rationale**:
- **Consistency**: Same events logged same way
- **Parseable**: Can extract metrics programmatically
- **Monitoring**: Track processing time, record counts
- **Audit Trail**: Document all transformations

## Consequences

### Positive

1. **Debugging**: Clear trail of what happened, when, where
2. **Monitoring**: Users can track progress of long-running projections
3. **Audit Trail**: Full record of data transformations
4. **User Support**: Error messages guide users to solutions
5. **Performance Tracking**: Log timings identify bottlenecks
6. **Consistency**: All modules log same way
7. **Configurability**: Change log level without code changes
8. **Archive**: Permanent record in log files
9. **Module Isolation**: Per-module log files aid debugging
10. **Professional**: Industry-standard logging practices

### Negative

1. **Log File Size**: Files can accumulate over time (mitigation: rotation)
2. **Performance Overhead**: Logging adds ~5-10% to runtime (acceptable)
3. **Verbosity**: DEBUG mode can produce excessive output
4. **Learning Curve**: Developers must learn logging conventions
5. **File Management**: Need to clean up old log files periodically

### Risks and Mitigations

**Risk**: Log files fill disk space
- **Mitigation**: Implement log rotation (keep last N runs)
- **Mitigation**: Compress old logs
- **Mitigation**: Document cleanup procedures

**Risk**: Sensitive data logged (PII, credentials)
- **Mitigation**: Never log individual-level data
- **Mitigation**: Mask credentials in logs
- **Mitigation**: Aggregate statistics only

**Risk**: Excessive logging slows processing
- **Mitigation**: Use INFO level for production (DEBUG for development)
- **Mitigation**: Avoid logging in tight loops
- **Mitigation**: Use f-strings only when message will be logged

**Risk**: Error messages too technical for users
- **Mitigation**: Include actionable guidance in error messages
- **Mitigation**: Provide examples of valid values
- **Mitigation**: Reference documentation for complex issues

## Alternatives Considered

### Alternative 1: Print Statements Only

**Description**: Use `print()` for all output.

**Pros**:
- Simple, no setup required
- Familiar to beginners

**Cons**:
- No log levels
- Can't filter by severity
- No timestamps
- Can't redirect to file easily
- Unprofessional

**Why Rejected**:
- Logging module provides essential features
- Standard practice in production systems

### Alternative 2: No File Logging (Console Only)

**Description**: Log to console only, no file output.

**Pros**:
- Simpler setup
- No disk space used

**Cons**:
- Logs lost after script ends
- Can't review after completion
- Harder to debug intermittent issues

**Why Rejected**:
- File logs essential for debugging
- Disk space cost minimal

### Alternative 3: Separate Logging Library (loguru, structlog)

**Description**: Use third-party logging library instead of standard `logging`.

**Pros** (loguru):
- Simpler API
- Better default formatting
- Automatic exception catching

**Pros** (structlog):
- Structured logging (JSON)
- Better for log aggregation

**Cons**:
- Additional dependency
- Less familiar to Python developers
- Standard logging is sufficient

**Why Rejected**:
- Standard `logging` meets all needs
- Don't need advanced features
- Avoid unnecessary dependencies

### Alternative 4: Single Monolithic Log File

**Description**: All modules write to one log file.

**Pros**:
- Single file to review
- Chronological order across modules

**Cons**:
- Hard to isolate module issues
- File becomes very large
- Harder to parse programmatically

**Why Rejected**:
- Per-module files aid debugging
- Can aggregate if needed

### Alternative 5: External Logging Service (Sentry, Datadog)

**Description**: Send logs to cloud logging service.

**Pros**:
- Centralized logging
- Search and analysis tools
- Alerting

**Cons**:
- Requires internet
- Cost
- Overkill for desktop application
- Privacy concerns (sending logs externally)

**Why Rejected**:
- Not needed for single-user desktop app
- File-based logging sufficient
- Can add later if multi-user deployment

## Implementation Notes

### File Locations

**Logger Module**: `/home/nigel/cohort_projections/cohort_projections/utils/logger.py`

**Log Files**: `/home/nigel/cohort_projections/logs/`

### Usage in All Modules

**Pattern**:
```python
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)

def some_function():
    logger.info("Starting function")
    try:
        # Processing
        logger.debug(f"Intermediate result: {value}")
        logger.info("Function complete")
    except Exception as e:
        logger.error(f"Function failed: {e}", exc_info=True)
        raise
```

### Log Rotation (Future Enhancement)

**Using Python's RotatingFileHandler**:
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5           # Keep 5 old files
)
```

**Not currently implemented** but can be added if log files grow large.

### Exception Logging

**Pattern**:
```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise
```

**`exc_info=True`** includes full stack trace in log.

## References

1. **Python Logging HOWTO**: https://docs.python.org/3/howto/logging.html
2. **Logging Best Practices**: https://docs.python-guide.org/writing/logging/
3. **12-Factor App Logging**: https://12factor.net/logs
4. **Google's Logging Best Practices**: https://cloud.google.com/logging/docs/best-practices

## Revision History

- **2025-12-18**: Initial version (ADR-009) - Logging and error handling strategy

## Related ADRs

- ADR-005: Configuration management (logging configuration section)
- ADR-006: Data pipeline architecture (logging throughout pipeline)
- All ADRs: Logging used in all components
