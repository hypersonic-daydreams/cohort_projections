# Pipeline Orchestration Scripts

This directory contains the three main pipeline scripts that automate the end-to-end workflow for the North Dakota Population Projection System.

## Overview

The pipeline is divided into three sequential stages:

1. **Data Processing** (`01_process_demographic_data.py`) - Process raw demographic data
2. **Projection Execution** (`02_run_projections.py`) - Run cohort-component projections
3. **Export/Dissemination** (`03_export_results.py`) - Export results in various formats

Each script can be run independently or as part of a complete pipeline workflow.

## Pipeline Architecture

```
Raw Data → [01] Process Data → [02] Run Projections → [03] Export Results → Distribution
           └─ Fertility        └─ State             └─ CSV
           └─ Survival         └─ Counties          └─ Excel
           └─ Migration        └─ Places            └─ Packages
```

## Scripts

### 01_process_demographic_data.py

Orchestrates the processing of raw demographic data into standardized formats required by the projection engine.

**Features:**
- Processes fertility rates (SEER → cohort fertility table)
- Processes survival rates (life tables → cohort survival table)
- Processes migration rates (IRS flows → cohort migration table)
- Validates all outputs
- Generates processing reports
- Supports selective processing (individual components or all)
- Error handling modes (continue on error or fail-fast)

**Usage Examples:**

```bash
# Process all demographic data
python 01_process_demographic_data.py --all

# Process specific components
python 01_process_demographic_data.py --fertility
python 01_process_demographic_data.py --survival
python 01_process_demographic_data.py --migration

# Process multiple components
python 01_process_demographic_data.py --fertility --survival

# Dry run (validate without processing)
python 01_process_demographic_data.py --all --dry-run

# Fail-fast mode (stop on first error)
python 01_process_demographic_data.py --all --fail-fast

# Use custom configuration
python 01_process_demographic_data.py --all --config /path/to/config.yaml

# Skip report generation
python 01_process_demographic_data.py --all --no-report
```

**Outputs:**
- `data/processed/fertility_rates.parquet`
- `data/processed/survival_rates.parquet`
- `data/processed/migration_rates.parquet`
- `data/processed/reports/data_processing_report_TIMESTAMP.json`

### 02_run_projections.py

Orchestrates the execution of cohort-component projections for all configured geographies with support for multiple scenarios and parallel processing.

**Features:**
- Runs projections for state, counties, and places
- Supports multiple scenarios (baseline, high growth, low growth, etc.)
- Parallel processing for multiple geographies
- Geography filtering (all, specific levels, or specific FIPS codes)
- Resume capability (skip already-completed geographies)
- Hierarchical aggregation and validation
- Progress tracking with detailed logging

**Usage Examples:**

```bash
# Run all projections
python 02_run_projections.py --all

# Run state-level only
python 02_run_projections.py --state

# Run county-level projections
python 02_run_projections.py --counties

# Run place-level projections
python 02_run_projections.py --places

# Run specific geographies by FIPS
python 02_run_projections.py --fips 38101 38015 38035

# Combine multiple levels
python 02_run_projections.py --state --counties

# Run multiple scenarios
python 02_run_projections.py --all --scenarios baseline high_growth

# Resume from previous run (skip completed geographies)
python 02_run_projections.py --all --resume

# Dry run mode
python 02_run_projections.py --all --dry-run

# Use custom configuration
python 02_run_projections.py --all --config /path/to/config.yaml
```

**Outputs:**
- `data/projections/{scenario}/state/{fips}_projection.parquet`
- `data/projections/{scenario}/county/{fips}_projection.parquet`
- `data/projections/{scenario}/place/{fips}_projection.parquet`
- `data/projections/{scenario}/metadata/projection_run_TIMESTAMP.json`

### 03_export_results.py

Exports projection results to various formats for dissemination and creates distribution packages.

**Features:**
- Converts Parquet → CSV/Excel
- Creates summary statistics tables
- Generates comparison reports
- Packages results for distribution (ZIP archives)
- Creates data dictionaries and documentation
- Supports selective export (specific geographies, formats, or scenarios)

**Usage Examples:**

```bash
# Export all results
python 03_export_results.py --all

# Export specific levels
python 03_export_results.py --state --counties
python 03_export_results.py --places

# Export specific scenarios
python 03_export_results.py --all --scenarios baseline high_growth

# Export only specific formats
python 03_export_results.py --all --formats csv
python 03_export_results.py --all --formats csv excel

# Create distribution packages
python 03_export_results.py --all --package

# Skip package creation
python 03_export_results.py --all --no-package

# Dry run mode
python 03_export_results.py --all --dry-run

# Use custom configuration
python 03_export_results.py --all --config /path/to/config.yaml
```

**Outputs:**
- `data/exports/{scenario}/{level}/csv/*.csv.gz`
- `data/exports/{scenario}/{level}/excel/*.xlsx`
- `data/exports/{scenario}/summaries/*.csv`
- `data/exports/data_dictionary.json`
- `data/exports/data_dictionary.md`
- `data/exports/packages/nd_projections_{level}_YYYYMMDD.zip`
- `data/exports/export_report_TIMESTAMP.json`

## Complete Pipeline Workflow

To run the complete end-to-end pipeline:

```bash
# Step 1: Process demographic data
python scripts/pipeline/01_process_demographic_data.py --all

# Step 2: Run projections (baseline scenario)
python scripts/pipeline/02_run_projections.py --all

# Step 3: Export results
python scripts/pipeline/03_export_results.py --all
```

Or create a shell script to run all three:

```bash
#!/bin/bash
# run_complete_pipeline.sh

set -e  # Exit on error

echo "Starting complete projection pipeline..."

echo "Step 1/3: Processing demographic data..."
python scripts/pipeline/01_process_demographic_data.py --all

echo "Step 2/3: Running projections..."
python scripts/pipeline/02_run_projections.py --all

echo "Step 3/3: Exporting results..."
python scripts/pipeline/03_export_results.py --all

echo "Pipeline complete!"
```

## Configuration

All pipeline scripts use `config/projection_config.yaml` for configuration. Key sections:

```yaml
pipeline:
  data_processing:
    input_dir: "data/raw"
    output_dir: "data/processed"
    validate_outputs: true
    generate_report: true
    fail_fast: false

  projection:
    output_dir: "data/projections"
    scenarios: ["baseline"]
    resume_on_restart: true
    save_intermediate: false

  export:
    output_dir: "data/exports"
    formats: ["csv", "excel"]
    create_packages: true
    package_by: "level"
```

## Error Handling

All scripts include comprehensive error handling:

- **Continue on Error** (default): Log errors and continue processing remaining items
- **Fail-Fast Mode**: Stop immediately on first error (use `--fail-fast` flag)
- **Resume Capability**: Skip already-completed items (use `--resume` flag in projection script)
- **Detailed Logging**: All operations logged to console and log files
- **Exit Codes**:
  - `0` = Success
  - `1` = Error occurred

## Dry Run Mode

All scripts support dry-run mode (`--dry-run` flag) which:
- Validates all inputs
- Shows what would be processed
- Does not modify any files
- Useful for testing and verification

## Logging

Logs are written to:
- Console (stdout) - Real-time progress
- Log files in `logs/` directory - Detailed operation logs
- Report files in respective output directories - Summary statistics

Log level can be configured in `config/projection_config.yaml`:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_directory: "logs"
```

## Performance Considerations

### Parallel Processing

The projection runner (`02_run_projections.py`) supports parallel processing:

```yaml
geographic:
  parallel_processing:
    enabled: true
    max_workers: null  # null = auto-detect CPU count
    chunk_size: 10
```

### Memory Management

For large-scale processing:
- Use chunk_size to limit concurrent operations
- Enable resume capability for long-running jobs
- Monitor memory usage with system tools

### Typical Runtime

Approximate runtimes (will vary by system):
- **Data Processing**: 5-15 minutes
- **Projections** (all geographies, baseline): 30-60 minutes
- **Export**: 10-20 minutes
- **Complete Pipeline**: 45-95 minutes

## Troubleshooting

### Common Issues

**Issue**: "Input file not found"
- Solution: Ensure raw data files are in correct locations specified in config
- Run with `--dry-run` to check file paths

**Issue**: "No geographies to process"
- Solution: Check geography configuration in `projection_config.yaml`
- Verify FIPS codes are correct

**Issue**: "Validation failed"
- Solution: Check processing reports for specific validation errors
- Review data quality in raw input files

**Issue**: "Memory error during parallel processing"
- Solution: Reduce `max_workers` or `chunk_size` in configuration
- Process geographies in batches using FIPS filtering

### Debug Mode

For detailed debugging, set log level to DEBUG:

```yaml
logging:
  level: "DEBUG"
```

Or use Python's debugger:

```bash
python -m pdb scripts/pipeline/01_process_demographic_data.py --all
```

## Testing

Test individual components:

```bash
# Test data processing with dry run
python 01_process_demographic_data.py --all --dry-run

# Test projection for single county
python 02_run_projections.py --fips 38101 --dry-run

# Test export for state level only
python 03_export_results.py --state --dry-run
```

## Dependencies

Required Python packages:
- pandas >= 1.5.0
- numpy >= 1.23.0
- pyarrow >= 10.0.0 (for Parquet support)
- openpyxl >= 3.0.0 (for Excel export)
- pyyaml >= 6.0
- tqdm >= 4.65.0 (optional, for progress bars)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Contributing

When modifying pipeline scripts:
1. Maintain consistent error handling patterns
2. Update documentation and examples
3. Add tests for new functionality
4. Follow project coding standards (Google docstring style, type hints)
5. Update ADR if making architectural changes

## References

- [ADR 014: Pipeline Orchestration Design](../../docs/adr/014-pipeline-orchestration-design.md)
- [Project Configuration Guide](../../config/README.md)
- [Data Processing Documentation](../../cohort_projections/data/process/README.md)
- [Geographic Processing Documentation](../../cohort_projections/geographic/README.md)

## License

North Dakota Population Projection System
Copyright (c) 2025
