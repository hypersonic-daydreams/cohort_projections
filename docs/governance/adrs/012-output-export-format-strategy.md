# ADR-012: Output and Export Format Strategy

## Status
Accepted

## Date
2025-12-18

## Context

Population projection results need to be exported in formats suitable for different uses: analysis (statistical software), reporting (Excel), archiving (compressed files), and sharing (human-readable formats). The choice of output formats affects usability, performance, storage requirements, and interoperability.

### Use Cases

1. **Analysis**: Data scientists need efficient formats for further analysis (Python, R)
2. **Reporting**: Stakeholders need Excel workbooks with formatted tables and charts
3. **Archiving**: Long-term storage requires space-efficient, stable formats
4. **Sharing**: Collaborators need accessible, standard formats
5. **Web/BI**: Integration with dashboards and web applications
6. **Validation**: Demographers need to inspect detailed results

### Requirements

1. **Multiple Formats**: Support different use cases
2. **Performance**: Fast read/write for large projections
3. **Compression**: Minimize storage for large result sets
4. **Precision**: Preserve decimal precision appropriately
5. **Metadata**: Include provenance and quality information
6. **Compatibility**: Work across platforms and tools
7. **Human-Readable**: At least one format accessible to non-programmers

## Decision

### Decision 1: Dual Format Strategy (Parquet + CSV)

**Decision**: Export all projection results in both Parquet (primary, optimized) and CSV (secondary, human-readable).

**Format Roles**:

**Parquet** (Primary for computation):
- **Use**: Re-loading into Python/R for analysis
- **Benefits**: Fast, compressed, preserves types
- **Audience**: Data scientists, automated pipelines

**CSV** (Secondary for accessibility):
- **Use**: Opening in Excel, text editors, other tools
- **Benefits**: Universal, human-readable, simple
- **Audience**: Non-programmers, quick inspection

**Export Pattern**:
```python
# Save projection results
projection.export_results(
    output_path='output/nd_state_projection_2025_2045',
    formats=['parquet', 'csv']
)

# Creates:
# - nd_state_projection_2025_2045.parquet    (primary)
# - nd_state_projection_2025_2045.csv        (backup/readable)
# - nd_state_projection_2025_2045_metadata.json  (provenance)
```

**Rationale**:

**Why Parquet**:
- **Performance**: 5-10x faster to read/write than CSV
- **Size**: 50-80% smaller than CSV (with compression)
- **Types**: Preserves int/float/datetime types
- **Analytics**: Native support in pandas, Spark, R (arrow)
- **Industry Standard**: Becoming standard for data pipelines

**Why CSV**:
- **Universal**: Opens in Excel, Google Sheets, any text editor
- **Simple**: No special tools needed
- **Portable**: Works everywhere
- **Debugging**: Easy to inspect visually
- **Backup**: If Parquet tools unavailable

**Why Both**:
- Negligible cost (Parquet compression saves space)
- Covers all use cases
- Professional practice

### Decision 2: Gzip Compression for Parquet

**Decision**: Use gzip compression for Parquet files (not snappy, zstd, or uncompressed).

**Compression Options**:
```python
# Parquet with gzip compression
df.to_parquet(
    'output/projection.parquet',
    compression='gzip',
    engine='pyarrow'
)
```

**Compression Methods Compared**:

| Method | Speed | Ratio | Compatibility |
|--------|-------|-------|--------------|
| None | Fastest write | 1.0x | Universal |
| Snappy | Fast | 2-3x | Good |
| Gzip | Medium | 5-8x | Excellent |
| Zstd | Medium | 6-9x | Modern tools only |

**Rationale**:

**Why Gzip**:
- **Good Compression**: 5-8x reduction (projection files typically 10-50 MB → 1-5 MB)
- **Universal**: Works everywhere (Python, R, Spark, cloud tools)
- **Balanced**: Not too slow, not too large
- **Standard**: Most common Parquet compression

**Why Not Snappy**:
- Lower compression ratio
- Marginally faster (not critical for projections)

**Why Not Zstd**:
- Newer, less universal
- Minimal additional compression benefit
- May not work in older tools

**Why Not Uncompressed**:
- Wastes disk space
- Slower to read over network
- No benefit

### Decision 3: Include Zero Cells in Output

**Decision**: Include cohorts with zero population in outputs rather than dropping them.

**Example**:
```
age,sex,race,population
0,Male,White NH,100
0,Male,Black NH,5
0,Male,AIAN NH,0      ← Include zero
0,Female,White NH,95
...
```

**Rationale**:

**Why Include Zeros**:
- **Complete Matrix**: All age × sex × race combinations present
- **Easier Analysis**: No need to check for missing combinations
- **Filtering Easy**: `df[df['population'] > 0]` if needed
- **Time Series**: Zeros → positives over time (population growth)
- **File Size**: Minimal increase with compression

**Why Not Drop Zeros**:
- **Incomplete Data**: Missing combinations ambiguous (zero or absent?)
- **Complexity**: Merging incomplete matrices error-prone
- **Inconsistency**: Different years have different rows

**Configuration**:
```yaml
output:
  include_zero_cells: true  # Include cohorts with 0 population
```

Can be disabled if space-critical.

### Decision 4: Two Decimal Places for Population Values

**Decision**: Round population values to 2 decimal places in outputs (not integers, not full precision).

**Rounding**:
```python
# Round to 2 decimals
df['population'] = df['population'].round(2)
```

**Examples**:
- 1234.56789 → 1234.57
- 100.001 → 100.00
- 0.123 → 0.12

**Rationale**:

**Why 2 Decimals**:
- **Precision Sufficient**: Populations are estimates, not exact counts
- **File Size**: Smaller than full precision (1234.5678901234)
- **Readability**: Easier to read than many decimals
- **Standard Practice**: Census uses whole numbers; 2 decimals gives flexibility

**Why Not Integers**:
- Small populations become 0 (loses information)
- Aggregation errors accumulate
- Flexibility for rates and proportions

**Why Not More Decimals**:
- False precision (projections uncertain to ±5-10% anyway)
- Larger files
- Harder to read

**Application**:
- Population counts: 2 decimals
- Rates (fertility, survival): 4-5 decimals (different context)
- Percentages: 1-2 decimals

### Decision 5: Separate Files by Geography and Year Range

**Decision**: Create separate output files for each geography and projection period, not one massive file.

**File Organization**:
```
output/
  projections/
    state/
      nd_state_2025_2045.parquet
      nd_state_2025_2045.csv
      nd_state_2025_2045_metadata.json

    counties/
      nd_county_38101_2025_2045.parquet     # Cass County
      nd_county_38101_2025_2045.csv
      nd_county_38015_2025_2045.parquet     # Burleigh County
      ...

    places/
      nd_place_3825700_2025_2045.parquet    # Fargo city
      nd_place_3807200_2025_2045.parquet    # Bismarck city
      ...

  summaries/
    nd_all_counties_summary_2025_2045.csv   # Summary table
    nd_all_places_summary_2025_2045.csv
```

**Rationale**:

**Why Separate Files**:
- **Modularity**: Load only needed geography
- **Parallel Processing**: Can generate simultaneously
- **Smaller Files**: Each file manageable size
- **Organization**: Clear structure
- **Incremental Updates**: Update one geography without reprocessing all

**Why Not Single File**:
- Large file (400+ places × 20 years × 1,092 cohorts = millions of rows)
- Slow to load entire file for one geography
- Memory intensive
- Single point of failure

**Trade-off**: More files to manage, but worth it for modularity.

### Decision 6: JSON Metadata Files with Comprehensive Provenance

**Decision**: Accompany each projection output with JSON metadata file documenting full provenance and quality metrics.

**Metadata Schema**:
```json
{
  "projection_metadata": {
    "created_date": "2025-12-18T14:30:00",
    "projection_version": "1.0.0",
    "geography": {
      "type": "state",
      "fips": "38",
      "name": "North Dakota"
    },
    "time_period": {
      "start_year": 2025,
      "end_year": 2045,
      "projection_years": 20
    },
    "scenario": "baseline"
  },

  "input_data_sources": {
    "base_population": {
      "source": "Census PEP 2025",
      "file": "data/processed/population/base_population_2025.parquet",
      "total_population": 779094
    },
    "fertility_rates": {
      "source": "SEER 2018-2022",
      "file": "data/processed/fertility/fertility_rates.parquet",
      "averaging_period": 5,
      "tfr_overall": 1.68
    },
    "survival_rates": {
      "source": "SEER Life Tables 2020",
      "file": "data/processed/mortality/survival_rates.parquet",
      "life_expectancy_m": 76.5,
      "life_expectancy_f": 81.2
    },
    "migration_rates": {
      "source": "IRS County Flows 2018-2022",
      "file": "data/processed/migration/migration_rates.parquet",
      "net_migration_annual": 5200
    }
  },

  "projection_results": {
    "total_records": 21840,
    "year_range": [2025, 2045],
    "years_projected": 20,
    "cohorts_per_year": 1092,
    "final_year_population": 891235,
    "total_growth_pct": 14.4,
    "annual_growth_rate_pct": 0.68
  },

  "quality_metrics": {
    "validation_status": "passed",
    "warnings": [],
    "plausibility_checks": {
      "tfr_implied": 1.72,
      "life_expectancy_implied": 78.9,
      "growth_rate_plausible": true
    }
  },

  "configuration_used": {
    "demographics": {...},
    "rates": {...},
    "scenarios": {...}
  }
}
```

**Rationale**:
- **Reproducibility**: Can recreate projection from metadata
- **Validation**: Quality metrics for quick assessment
- **Provenance**: Full audit trail from sources to outputs
- **Comparison**: Compare different projection runs
- **Documentation**: Self-documenting outputs

### Decision 7: Optional Excel Export for Stakeholder Reports

**Decision**: Provide optional Excel export with formatted worksheets for stakeholder presentations.

**Excel Export** (optional, not default):
```python
projection.export_to_excel(
    output_path='output/reports/nd_projection_2025_2045.xlsx',
    include_charts=True,
    format_tables=True
)
```

**Excel Workbook Structure**:
- **Summary Sheet**: Key statistics, totals by year
- **By Age Sheet**: Population pyramids data
- **By Race Sheet**: Racial composition over time
- **By Sex Sheet**: Sex ratios, male/female trends
- **Charts Sheet**: Pre-built charts (age pyramids, growth trends)
- **Metadata Sheet**: Data sources, assumptions

**Rationale**:

**Why Excel**:
- **Stakeholders**: Decision-makers prefer Excel
- **Formatting**: Can include formatted tables, charts
- **Accessibility**: No special tools needed
- **Reporting**: Ready for presentations

**Why Optional**:
- Slower to generate
- Large files
- Not needed for all use cases
- Parquet/CSV sufficient for most

**Implementation**: Use `openpyxl` or `xlsxwriter` library.

### Decision 8: Summary Tables in Addition to Detailed Outputs

**Decision**: Generate both detailed (cohort-level) and summary (aggregated) outputs.

**Detailed Output** (full cohort detail):
```
year,age,sex,race,population
2025,0,Male,White NH,5234.12
2025,0,Female,White NH,4987.34
2025,1,Male,White NH,5190.45
...
```
Size: ~1M rows for state 20-year projection

**Summary Output** (aggregated):
```
year,total_population,median_age,sex_ratio,pct_under_18,pct_65_plus
2025,779094,38.2,0.98,22.3,16.8
2026,784312,38.4,0.98,22.1,17.1
2027,789456,38.6,0.98,21.9,17.4
...
```
Size: ~20 rows for 20-year projection

**Summary Metrics**:
- Total population
- Population by sex
- Population by race
- Age structure (median age, dependency ratios)
- Growth rates (annual, cumulative)

**Rationale**:
- **Detailed**: For analysis, detailed breakdowns
- **Summary**: For quick review, presentations
- **Efficiency**: Summary files tiny, fast to work with
- **Use Cases**: Different audiences need different detail levels

## Consequences

### Positive

1. **Versatility**: Multiple formats support different use cases
2. **Performance**: Parquet provides fast analysis
3. **Accessibility**: CSV opens anywhere
4. **Compression**: Efficient storage with gzip
5. **Completeness**: Include zero cells prevents gaps
6. **Appropriate Precision**: 2 decimals balances accuracy and readability
7. **Organization**: Separate files by geography aids management
8. **Provenance**: Metadata documents full lineage
9. **Reporting**: Optional Excel for stakeholders
10. **Flexibility**: Detailed and summary outputs

### Negative

1. **Multiple Files**: More files to manage per projection
2. **Disk Space**: Dual formats use more space (mitigated by compression)
3. **Complexity**: More export options to maintain
4. **Redundancy**: Some information duplicated across formats
5. **Learning Curve**: Users need to understand which format for which use

### Risks and Mitigations

**Risk**: Parquet format becomes obsolete
- **Mitigation**: CSV backup ensures long-term accessibility
- **Mitigation**: Parquet is industry standard, unlikely to disappear
- **Mitigation**: Can convert Parquet to other formats

**Risk**: Large number of output files becomes unwieldy
- **Mitigation**: Clear directory structure
- **Mitigation**: Naming conventions (FIPS codes)
- **Mitigation**: Summary files for overview

**Risk**: Precision loss from rounding causes issues
- **Mitigation**: 2 decimals sufficient for demographic work
- **Mitigation**: Can export full precision if needed (configuration)
- **Mitigation**: Document rounding in metadata

**Risk**: Metadata and data become out of sync
- **Mitigation**: Generate metadata programmatically (not manual)
- **Mitigation**: Include checksums in metadata
- **Mitigation**: Validate metadata against data

## Alternatives Considered

### Alternative 1: Parquet Only (No CSV)

**Description**: Export only Parquet, skip CSV.

**Pros**:
- Simpler (one format)
- Smaller total size
- Faster exports

**Cons**:
- Not accessible to non-programmers
- Requires special tools
- Harder to inspect quickly

**Why Rejected**:
- CSV accessibility important
- Disk space not critical (compression helps)
- Dual format is best practice

### Alternative 2: Excel as Primary Format

**Description**: Export directly to Excel workbooks.

**Pros**:
- Stakeholder-friendly
- Includes formatting and charts
- Familiar to everyone

**Cons**:
- Slow for large datasets
- Size limits (1M rows in Excel)
- Not suitable for programmatic analysis
- Proprietary format

**Why Rejected**:
- Excel optional, not primary
- Parquet better for analysis
- Excel as reporting format only

### Alternative 3: SQLite Database

**Description**: Export to SQLite database file.

**Pros**:
- Queryable
- Compact
- Relational structure

**Cons**:
- Requires SQL knowledge
- Not human-readable
- Slower than Parquet for analytics
- Less standard for data science

**Why Rejected**:
- Parquet is standard for analytical data
- SQLite adds complexity without clear benefit

### Alternative 4: HDF5 Format

**Description**: Use HDF5 for scientific data storage.

**Pros**:
- Hierarchical structure
- Compression
- Metadata support

**Cons**:
- Less common in data science
- Requires h5py library
- Not human-readable
- Less tooling than Parquet

**Why Rejected**:
- Parquet more standard in data engineering
- Better ecosystem support

### Alternative 5: Single Combined File for All Geographies

**Description**: One file with all counties/places.

**Pros**:
- Single file to manage
- Easier to share

**Cons**:
- Very large (millions of rows)
- Slow to load for one geography
- Memory intensive
- Harder to parallelize generation

**Why Rejected**:
- Modularity outweighs convenience
- Can combine files if needed

## Implementation Notes

### Export Function

**In CohortComponentProjection class**:
```python
def export_results(
    self,
    output_path: str,
    formats: List[str] = ['parquet', 'csv'],
    include_metadata: bool = True,
    compression: str = 'gzip'
):
    """
    Export projection results in multiple formats.

    Args:
        output_path: Base path (without extension)
        formats: List of formats to export ['parquet', 'csv', 'excel']
        include_metadata: Whether to generate metadata JSON
        compression: Compression method for Parquet
    """
    results = self.get_full_results()

    # Round populations
    results['population'] = results['population'].round(2)

    # Export each format
    if 'parquet' in formats:
        results.to_parquet(
            f"{output_path}.parquet",
            compression=compression,
            index=False
        )

    if 'csv' in formats:
        results.to_csv(
            f"{output_path}.csv",
            index=False,
            float_format='%.2f'
        )

    if 'excel' in formats:
        self._export_excel(output_path, results)

    if include_metadata:
        metadata = self._generate_metadata()
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
```

### File Naming Convention

**Pattern**: `{geography}_{type}_{fips}_{start_year}_{end_year}.{ext}`

**Examples**:
- `nd_state_2025_2045.parquet`
- `nd_county_38101_2025_2045.csv`
- `nd_place_3825700_2025_2045_metadata.json`
- `nd_all_counties_summary_2025_2045.csv`

### Configuration

**In `projection_config.yaml`**:
```yaml
output:
  formats:
    - "parquet"          # Primary (fast, compressed)
    - "csv"              # Secondary (readable)
  compression: "gzip"    # Parquet compression
  include_zero_cells: true
  decimal_places: 2
  aggregation_levels:
    - "state"
    - "county"
    - "place"
  generate_summaries: true
  excel_reports: false   # Optional, slower
```

## References

1. **Apache Parquet**: https://parquet.apache.org/
2. **Pandas I/O**: https://pandas.pydata.org/docs/user_guide/io.html
3. **Data Storage Best Practices**: "Data Pipelines with Apache Airflow" (2021)
4. **Demographic Output Standards**: Census Bureau data dissemination guidelines

## Revision History

- **2025-12-18**: Initial version (ADR-012) - Output and export format strategy

## Related ADRs

- ADR-004: Core projection engine (generates outputs)
- ADR-006: Data pipeline (similar format choices for processed data)
- ADR-010: Geographic scope (file organization by geography)
- ADR-011: Testing (validation of outputs)
