# ADR 014: Pipeline Orchestration Design

**Status**: Accepted

**Date**: 2025-12-18

**Decision Makers**: Project Team

**Related ADRs**:
- ADR 007: Geographic Module Architecture
- ADR 008: Parallel Processing Strategy
- ADR 011: Configuration Management Approach

## Context

The North Dakota Population Projection System requires an automated pipeline to orchestrate the end-to-end workflow from raw demographic data to dissemination-ready outputs. The system must:

1. Process multiple types of demographic data (fertility, survival, migration)
2. Run projections for hundreds of geographies across multiple scenarios
3. Export results in multiple formats for different audiences
4. Handle failures gracefully and support resume capability
5. Provide clear progress tracking and error reporting
6. Be maintainable and extensible for future enhancements

Key challenges:
- **Complexity**: Multiple data sources, processing steps, and output formats
- **Scale**: 500+ geographies × 20 years × multiple scenarios
- **Reliability**: Long-running operations that may fail partway through
- **Usability**: Must be accessible to both technical and non-technical users
- **Flexibility**: Support partial runs, specific geographies, and custom configurations

## Decision

We will implement a **three-script pipeline architecture** with the following design:

### 1. Script Separation (3 Scripts vs Monolithic)

**Decision**: Separate pipeline into three independent scripts, one per major stage.

**Rationale**:
- **Modularity**: Each script has a single, well-defined responsibility
- **Flexibility**: Run individual stages independently (e.g., re-export without re-projecting)
- **Development**: Teams can work on different stages in parallel
- **Testing**: Easier to test individual components in isolation
- **Debugging**: Simpler to troubleshoot specific pipeline stages
- **Reusability**: Export script can be used with externally-generated projections

**Scripts**:
1. `01_process_demographic_data.py` - Data processing orchestration
2. `02_run_projections.py` - Projection execution orchestration
3. `03_export_results.py` - Export and dissemination orchestration

**Alternatives Considered**:
- **Single monolithic script**: Rejected due to complexity and inflexibility
- **Workflow orchestration tool (Airflow/Prefect)**: Rejected as over-engineering for current needs, but could be future enhancement
- **Make/Makefile**: Rejected due to limited Python integration and error handling
- **Five+ micro-scripts**: Rejected as too granular, would increase coordination overhead

### 2. Configuration-Driven vs Code-Driven Approach

**Decision**: Configuration-driven with sensible defaults and command-line overrides.

**Rationale**:
- **Separation of Concerns**: Data/settings separate from logic
- **Accessibility**: Non-programmers can modify behavior without code changes
- **Version Control**: Configuration changes tracked separately from code
- **Environment Support**: Easy to maintain dev/test/prod configurations
- **Validation**: Single source of truth for all pipeline parameters

**Implementation**:
```yaml
pipeline:
  data_processing:
    validate_outputs: true
    generate_report: true
    fail_fast: false

  projection:
    scenarios: ["baseline"]
    resume_on_restart: true

  export:
    formats: ["csv", "excel"]
    create_packages: true
```

**Command-line overrides** allowed for common operations:
```bash
python 02_run_projections.py --all --scenarios baseline high_growth
```

**Alternatives Considered**:
- **Code-driven (hardcoded parameters)**: Rejected due to inflexibility
- **Environment variables only**: Rejected due to discoverability issues
- **Separate config per script**: Rejected to avoid configuration duplication

### 3. Resume Capability Design

**Decision**: File-based resume using output file existence checks.

**Rationale**:
- **Simplicity**: No database or state management required
- **Reliability**: Filesystem is source of truth
- **Transparency**: Users can see exactly what has been completed
- **Flexibility**: Can manually delete files to force re-processing

**Implementation**:
```python
def get_completed_geographies(output_dir: Path, scenario: str) -> Set[str]:
    """Check for existing projection files to skip."""
    completed = set()
    for file in output_dir.glob(f"{scenario}/*/*.parquet"):
        fips = file.stem.split("_")[0]
        completed.add(fips)
    return completed
```

**Usage**:
```bash
# Run projections, skipping already-completed geographies
python 02_run_projections.py --all --resume
```

**Alternatives Considered**:
- **Database-based state tracking**: Rejected as over-engineering
- **Lock files**: Rejected due to complexity with parallel processing
- **Checkpoint files**: Rejected as redundant with output file existence
- **No resume capability**: Rejected due to long runtimes and reliability needs

### 4. Error Handling Strategy

**Decision**: Dual-mode error handling with fail-fast and continue-on-error modes.

**Modes**:

1. **Continue-on-error (default)**: Log errors, track failures, continue processing
   - Use case: Production runs where partial results are valuable
   - Behavior: Process all items, report failures at end
   - Exit code: 1 if any failures occurred

2. **Fail-fast mode**: Stop on first error
   - Use case: Development, debugging, data validation
   - Behavior: Immediately exit on first error with detailed traceback
   - Exit code: 1 on first error

**Implementation**:
```python
for component in components:
    result = process_component(component)
    report.add_result(result)

    if fail_fast and not result.success:
        logger.error(f"Fail-fast enabled: Stopping due to {component} error")
        return report
```

**Error Tracking**:
- All errors logged to file with full traceback
- Failed items tracked in processing reports
- Summary statistics (success/failure counts) in report
- Exit codes indicate overall success/failure

**Rationale**:
- **Production needs**: Partial results often valuable (e.g., 100 of 103 counties completed)
- **Development needs**: Fast feedback loop for debugging
- **Transparency**: Always know what failed and why
- **Automation-friendly**: Exit codes enable integration with CI/CD

**Alternatives Considered**:
- **Always fail-fast**: Rejected as too restrictive for production
- **Always continue**: Rejected as makes debugging harder
- **Retry logic**: Considered for future enhancement but not initial implementation

### 5. Command-Line Interface Design

**Decision**: Argparse-based CLI with consistent patterns across all scripts.

**Design Principles**:
1. **Consistency**: Similar flags across all three scripts
2. **Clarity**: Self-documenting with help text and examples
3. **Safety**: Dry-run mode for all operations
4. **Flexibility**: Support common use cases without complex syntax

**Common Flags** (all scripts):
- `--all`: Process all items
- `--dry-run`: Show what would be done without doing it
- `--config PATH`: Use custom configuration file

**Script-Specific Flags**:

**Data Processing**:
```bash
--fertility  --survival  --migration  # Select components
--fail-fast                            # Stop on first error
--no-report                            # Skip report generation
```

**Projection Runner**:
```bash
--state  --counties  --places          # Select levels
--fips FIPS [FIPS ...]                 # Specific geographies
--scenarios SCENARIO [SCENARIO ...]    # Select scenarios
--resume                               # Skip completed
```

**Export**:
```bash
--state  --counties  --places          # Select levels
--scenarios SCENARIO [SCENARIO ...]    # Select scenarios
--formats FORMAT [FORMAT ...]          # Select formats
--package / --no-package               # Control packaging
```

**Help and Examples**:
All scripts include extensive help text with examples:
```bash
python 01_process_demographic_data.py --help
```

**Rationale**:
- **User-friendly**: Clear options for common operations
- **Discoverable**: Built-in help and examples
- **Composable**: Flags can be combined logically
- **Future-proof**: Easy to add new options

**Alternatives Considered**:
- **Click library**: Rejected to minimize dependencies
- **Positional arguments**: Rejected as less clear
- **Interactive prompts**: Rejected as not automation-friendly
- **Config file only**: Rejected as less convenient for ad-hoc runs

### 6. Progress Tracking and Reporting

**Decision**: Multi-level reporting with console output, log files, and JSON reports.

**Reporting Levels**:

1. **Console Output** (real-time):
   - Progress messages
   - Summary statistics
   - Error notifications
   - Formatted tables

2. **Log Files** (detailed):
   - All operations logged
   - Full error tracebacks
   - Timestamp on every entry
   - Separate file per module

3. **JSON Reports** (structured):
   - Processing statistics
   - Failed items
   - Performance metrics
   - Machine-readable for automation

**Example Report Structure**:
```json
{
  "start_time": "2025-12-18T10:00:00",
  "end_time": "2025-12-18T10:45:00",
  "duration_seconds": 2700,
  "geographies": {
    "total": 103,
    "completed": 100,
    "failed": 3,
    "skipped": 0
  },
  "failed_geographies": [
    {"fips": "38101", "error": "Missing migration data"}
  ]
}
```

**Rationale**:
- **User experience**: Real-time feedback on progress
- **Debugging**: Detailed logs for troubleshooting
- **Automation**: Structured reports for downstream processing
- **Audit trail**: Complete record of what was processed

**Alternatives Considered**:
- **Progress bars only**: Rejected as insufficient detail
- **Database logging**: Rejected as over-engineering
- **External monitoring service**: Considered for future enhancement

### 7. Parallelization Strategy

**Decision**: Leverage existing multi_geography module; no additional parallelization in pipeline scripts.

**Rationale**:
- **Separation of concerns**: Parallelization logic in domain modules, not orchestration scripts
- **Already implemented**: multi_geography module has robust parallel processing
- **Consistency**: Same parallelization strategy throughout project
- **Configuration**: Control parallelization via config, not script flags

**Configuration**:
```yaml
geographic:
  parallel_processing:
    enabled: true
    max_workers: null  # Auto-detect
    chunk_size: 10
```

**No pipeline-level parallelization** of scripts themselves because:
- Sequential dependencies (must process data before running projections)
- Each script already parallelizes internally where appropriate
- Complexity not justified by benefits

**Alternatives Considered**:
- **Parallel scenario execution**: Considered for future enhancement
- **Pipeline-level parallel execution**: Rejected due to data dependencies
- **Distributed processing (Dask/Ray)**: Over-engineering for current scale

## Consequences

### Positive

1. **Modularity**: Clear separation of concerns makes system maintainable
2. **Flexibility**: Can run individual stages or complete pipeline
3. **Reliability**: Resume capability and error handling make long runs feasible
4. **Usability**: Clear CLI and dry-run mode make scripts accessible
5. **Observability**: Multi-level reporting provides transparency
6. **Testability**: Independent scripts easier to test
7. **Extensibility**: Easy to add new scenarios, formats, or processing steps

### Negative

1. **Coordination**: Users must understand script sequence (mitigated by documentation)
2. **State management**: File-based resume is simple but could miss edge cases
3. **Validation**: No automatic validation that all stages have completed (mitigated by reports)
4. **Shell script needed**: Running complete pipeline requires shell script or manual execution

### Neutral

1. **Configuration complexity**: More options = more configuration (mitigated by defaults)
2. **Learning curve**: Three scripts to understand vs one (mitigated by consistent patterns)

## Implementation Notes

### Script Structure

All scripts follow consistent pattern:
```python
def main():
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()

    config = load_projection_config(args.config)

    # Main processing
    result = process_all_items(...)

    # Reporting
    generate_report(result, config)

    # Exit code
    return 0 if success else 1
```

### Testing Strategy

1. **Unit tests**: Test individual functions (data loading, validation, etc.)
2. **Integration tests**: Test script execution with test data
3. **End-to-end tests**: Full pipeline with small dataset
4. **Dry-run tests**: Verify dry-run mode for all scripts

### Future Enhancements

Potential improvements (not in initial implementation):

1. **Workflow orchestration**: Integrate with Airflow/Prefect for complex workflows
2. **Parallel scenarios**: Run multiple scenarios in parallel
3. **Incremental updates**: Process only changed data
4. **Cloud execution**: Support for cloud-based processing (AWS Batch, GCP Dataflow)
5. **Interactive mode**: Guided CLI for less technical users
6. **Comparison reports**: Automated scenario comparison in export script
7. **Notification system**: Email/Slack notifications on completion
8. **Resource monitoring**: Track CPU/memory usage
9. **Retry logic**: Automatic retry of failed geographies
10. **Validation framework**: Pre-flight checks before processing

## References

- [Pipeline README](../../scripts/pipeline/README.md)
- [Multi-Geography Module](../../cohort_projections/geographic/multi_geography.py)
- [Configuration Management](../../config/projection_config.yaml)
- [Data Processing Modules](../../cohort_projections/data/process/)

## Revision History

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-18 | 1.0 | Initial decision record |
