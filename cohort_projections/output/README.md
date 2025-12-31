# Output Module

**Last Updated:** 2025-12-31 | **Status:** Current | **Related ADRs:** [ADR-012](../../docs/adr/012-output-export-format-strategy.md), [ADR-015](../../docs/adr/015-output-format-visualization-design.md)

Enhanced output capabilities for population projection results, providing formatted exports, comprehensive reports, and publication-ready visualizations.

## Overview

The output module provides three main components:

1. **Writers**: Export projection data to multiple formats with rich formatting
2. **Reports**: Generate summary statistics and formatted reports
3. **Visualizations**: Create charts and graphs for analysis and presentation

## Installation Requirements

### Core Dependencies (Required)
```bash
pip install pandas numpy
```

### Excel Support (Recommended)
```bash
pip install openpyxl
```

### Visualization Support (Recommended)
```bash
pip install matplotlib
pip install seaborn  # Optional, for enhanced styling
```

### Geospatial Support (Optional)
```bash
pip install geopandas  # For shapefile/GeoJSON exports
```

## Quick Start

```python
from cohort_projections.core import CohortComponentProjection
from cohort_projections.output import (
    write_projection_excel,
    generate_summary_statistics,
    generate_html_report,
    save_all_visualizations
)

# Run your projection
projection = CohortComponentProjection(...)
results = projection.run_projection()

# Export to Excel with formatting
write_projection_excel(
    results,
    output_path='output/projection_2025_2045.xlsx',
    include_charts=True
)

# Generate summary statistics
stats = generate_summary_statistics(results)
print(f"Population growth: {stats['demographic_indicators']['percent_growth']:.1f}%")

# Create HTML report
generate_html_report(
    results,
    output_path='output/report.html',
    title='North Dakota Population Projections 2025-2045'
)

# Generate all visualizations
save_all_visualizations(
    results,
    output_dir='output/charts',
    base_filename='nd_state_2025_2045'
)
```

## Module Components

### 1. Writers (`writers.py`)

Export projection data to various formats with rich formatting and customization.

#### Excel Export

```python
from cohort_projections.output import write_projection_excel

write_projection_excel(
    projection_df=results,
    output_path='output/results.xlsx',
    summary_df=summary,           # Optional pre-computed summary
    metadata=metadata,             # Optional metadata dict
    include_charts=True,           # Embed population charts
    include_formatting=True,       # Apply colors, borders, etc.
    title='ND Population Projections'
)
```

**Features**:
- Multiple sheets: Summary, By Age, By Sex, By Race, Detail, Metadata
- Professional formatting: Bold headers, colored fills, borders
- Number formatting with thousands separators
- Auto-width columns and freeze panes
- Embedded charts (population trends)
- Comprehensive metadata tracking

#### Enhanced CSV Export

```python
from cohort_projections.output import write_projection_csv

# Long format (years as rows)
write_projection_csv(
    projection_df=results,
    output_path='output/projection.csv.gz',
    format_type='long',
    compression='gzip'
)

# Wide format (years as columns)
write_projection_csv(
    projection_df=results,
    output_path='output/projection_wide.csv',
    format_type='wide',
    age_ranges=[(0, 17), (18, 64), (65, 90)],  # Filter to age groups
    sexes=['Female'],                           # Filter to females only
    columns_order=['age', 'sex', 'race']       # Custom column order
)
```

**Features**:
- Wide or long format
- Age range filtering
- Sex/race filtering
- Custom column ordering
- Gzip compression support

#### Multiple Format Export

```python
from cohort_projections.output import write_projection_formats

paths = write_projection_formats(
    projection_df=results,
    output_dir='output/projections',
    base_filename='nd_state_2025_2045',
    formats=['csv', 'excel', 'parquet', 'json'],
    summary_df=summary,
    metadata=metadata
)

# Access output paths
print(paths['excel'])    # PosixPath('output/projections/nd_state_2025_2045.xlsx')
print(paths['parquet'])  # PosixPath('output/projections/nd_state_2025_2045.parquet')
```

**Features**:
- Single function to export to multiple formats
- Consistent naming convention
- Metadata tracking across all formats
- Efficient storage (Parquet for large datasets)

### 2. Reports (`reports.py`)

Generate comprehensive summary statistics and formatted reports.

#### Summary Statistics

```python
from cohort_projections.output import generate_summary_statistics

stats = generate_summary_statistics(
    projection_df=results,
    base_year=2025,
    include_diversity_metrics=True
)

# Access statistics
print(stats['demographic_indicators']['final_population'])  # Final year population
print(stats['demographic_indicators']['growth_rate'])       # Percent growth
print(stats['by_year'][0]['dependency_ratio'])             # Base year dependency ratio

# Age structure
print(stats['age_structure']['year_2045']['65-74']['percent'])  # Percent 65-74 in 2045

# Diversity metrics
print(stats['diversity']['year_2045']['diversity_index'])  # Simpson's diversity index
```

**Statistics Computed**:
- Annual totals (population, by sex, by age group)
- Dependency ratios (total, youth, elderly)
- Sex ratios
- Median age
- Age structure distribution
- Diversity indices
- Growth rates (annual and period)

#### Scenario Comparison

```python
from cohort_projections.output import compare_scenarios

comparison = compare_scenarios(
    baseline_df=baseline_results,
    scenario_df=high_growth_results,
    baseline_name='Baseline',
    scenario_name='High Growth',
    years_to_compare=[2030, 2035, 2040, 2045]
)

# Compare populations
print(comparison[['year', 'Baseline_total', 'High Growth_total', 'difference', 'percent_difference']])
```

#### HTML Report

```python
from cohort_projections.output import generate_html_report

generate_html_report(
    projection_df=results,
    output_path='output/report.html',
    title='North Dakota Population Projections 2025-2045',
    summary_stats=stats,           # Optional pre-computed stats
    metadata=metadata,              # Optional metadata
    include_methodology=True        # Include methodology section
)
```

**Report Includes**:
- Executive summary with key statistics
- Demographic indicators table
- Population trends table
- Methodology section
- Professional styling with CSS

#### Text/Markdown Report

```python
from cohort_projections.output import generate_text_report

# Plain text report
generate_text_report(
    projection_df=results,
    output_path='output/report.txt',
    format_type='text',
    include_tables=True
)

# Markdown report
generate_text_report(
    projection_df=results,
    output_path='output/report.md',
    format_type='markdown',
    include_tables=True
)
```

**Use Cases**:
- Console output
- Email summaries
- Documentation
- README files

### 3. Visualizations (`visualizations.py`)

Create publication-ready charts and graphs.

#### Population Pyramid

```python
from cohort_projections.output import plot_population_pyramid

# Simple pyramid (by sex)
plot_population_pyramid(
    projection_df=results,
    year=2045,
    output_path='output/pyramid_2045.png',
    age_group_size=5,      # 5-year age groups
    dpi=300                # High resolution
)

# Pyramid with race breakdown
plot_population_pyramid(
    projection_df=results,
    year=2045,
    output_path='output/pyramid_2045_race.png',
    by_race=True,          # Stacked by race/ethnicity
    age_group_size=5
)
```

**Features**:
- Side-by-side male/female bars
- Optional race/ethnicity stacking
- Customizable age grouping (1, 5, or 10 years)
- Professional styling
- High-resolution export (PNG, SVG, PDF)

#### Population Trends

```python
from cohort_projections.output import plot_population_trends

# Total population trend
plot_population_trends(
    projection_df=results,
    output_path='output/trends_total.png',
    by='total'
)

# Trends by age group
plot_population_trends(
    projection_df=results,
    output_path='output/trends_age.png',
    by='age_group',
    age_groups={
        'Youth (0-17)': (0, 17),
        'Working Age (18-64)': (18, 64),
        'Elderly (65+)': (65, 90)
    }
)

# Trends by sex
plot_population_trends(
    projection_df=results,
    output_path='output/trends_sex.png',
    by='sex'
)

# Trends by race
plot_population_trends(
    projection_df=results,
    output_path='output/trends_race.png',
    by='race'
)
```

**Features**:
- Multiple grouping options (total, sex, age_group, race)
- Line charts with markers
- Customizable age group definitions
- Formatted axes with thousands separators
- Legend and grid

#### Growth Rates

```python
from cohort_projections.output import plot_growth_rates

# Annual growth rates
plot_growth_rates(
    projection_df=results,
    output_path='output/growth_annual.png',
    period='annual'
)

# 5-year period growth rates
plot_growth_rates(
    projection_df=results,
    output_path='output/growth_5year.png',
    period='5year'
)
```

**Features**:
- Bar chart with color coding (green=growth, red=decline)
- Annual, 5-year, or 10-year periods
- Zero reference line

#### Scenario Comparison

```python
from cohort_projections.output import plot_scenario_comparison

scenarios = {
    'Baseline': baseline_results,
    'High Growth': high_growth_results,
    'Low Growth': low_growth_results
}

plot_scenario_comparison(
    scenario_projections=scenarios,
    output_path='output/scenario_comparison.png',
    title='Population Projection Scenarios'
)
```

**Features**:
- Multiple scenarios on one chart
- Clear color differentiation
- Line chart with markers

#### Generate All Visualizations

```python
from cohort_projections.output import save_all_visualizations

paths = save_all_visualizations(
    projection_df=results,
    output_dir='output/charts',
    base_filename='nd_state_2025_2045',
    years_for_pyramids=[2025, 2035, 2045],  # Optional specific years
    image_format='png',                      # 'png', 'svg', or 'pdf'
    dpi=300,
    style='seaborn-v0_8-darkgrid'
)

# Access generated charts
print(paths['pyramid_2025'])        # Population pyramid for 2025
print(paths['trends_total'])        # Total population trends
print(paths['trends_age_groups'])   # Age group trends
print(paths['growth_rates'])        # Growth rates chart
```

**Generated Charts**:
- Population pyramids (base year, mid-point, final year)
- Total population trends
- Population trends by age group
- Population trends by sex
- Population trends by race
- Annual growth rates

## Configuration

Add output settings to `config/projection_config.yaml`:

```yaml
output:
  # Excel settings
  excel:
    include_charts: true
    include_metadata: true
    format_numbers: true

  # Report settings
  reports:
    generate_html: true
    generate_text: true
    include_methodology: true

  # Visualization settings
  visualizations:
    format: "png"  # png, svg, pdf
    dpi: 300
    style: "seaborn-v0_8-darkgrid"
    color_palette: "Set2"
    figure_size: [10, 6]
```

## File Naming Conventions

The module follows consistent naming conventions:

### Projection Files
- Pattern: `{geography}_{level}_{fips}_projection_{start_year}_{end_year}_{scenario}.{ext}`
- Example: `nd_state_38_projection_2025_2045_baseline.xlsx`

### Visualization Files
- Pattern: `{base_filename}_{chart_type}_{year}.{ext}`
- Example: `nd_state_2025_2045_pyramid_2045.png`

### Report Files
- Pattern: `{geography}_{level}_report_{date}.{ext}`
- Example: `nd_state_report_2025-03-15.html`

## Output Directory Structure

Recommended directory structure:

```
output/
├── projections/          # Projection data files
│   ├── state/
│   │   ├── nd_state_38_projection_2025_2045_baseline.xlsx
│   │   ├── nd_state_38_projection_2025_2045_baseline.parquet
│   │   └── nd_state_38_projection_2025_2045_baseline_metadata.json
│   ├── counties/
│   └── places/
├── charts/               # Visualizations
│   ├── state/
│   │   ├── nd_state_2025_2045_pyramid_2025.png
│   │   ├── nd_state_2025_2045_trends_total.png
│   │   └── nd_state_2025_2045_growth_rates.png
│   ├── counties/
│   └── places/
└── reports/              # HTML/text reports
    ├── nd_state_report.html
    └── nd_state_summary.txt
```

## Best Practices

### 1. Always Include Metadata

```python
metadata = {
    'geography': {'level': 'state', 'fips': '38', 'name': 'North Dakota'},
    'projection': {
        'base_year': 2025,
        'end_year': 2045,
        'scenario': 'baseline',
        'processing_date': datetime.now().isoformat()
    },
    'data_sources': {
        'population': 'Census 2020',
        'fertility': 'SEER 2018-2022',
        'mortality': 'SEER Life Tables 2020',
        'migration': 'IRS County Flows 2018-2022'
    }
}

write_projection_formats(..., metadata=metadata)
```

### 2. Generate Summary Statistics First

```python
# Compute once, use multiple times
stats = generate_summary_statistics(results)

# Use in Excel export
write_projection_excel(results, ..., metadata=stats['demographic_indicators'])

# Use in reports
generate_html_report(results, ..., summary_stats=stats)
```

### 3. Use Appropriate Formats

- **Excel**: Stakeholder presentations, interactive exploration
- **Parquet**: Large datasets, efficient storage, data science workflows
- **CSV**: Data sharing, database import, legacy systems
- **JSON**: Web applications, APIs, metadata

### 4. Customize Visualizations

```python
# Use consistent styling across all charts
style = 'seaborn-v0_8-darkgrid'
dpi = 300

plot_population_pyramid(..., style=style, dpi=dpi)
plot_population_trends(..., style=style, dpi=dpi)
save_all_visualizations(..., style=style, dpi=dpi)
```

### 5. Handle Large Datasets

```python
# For very large projections, filter before export
# Example: Export only 5-year intervals
filtered = results[results['year'] % 5 == 0]

write_projection_excel(filtered, ...)
```

## Common Use Cases

### Stakeholder Presentation Package

```python
# Complete output package for stakeholders
base_name = 'nd_state_2025_2045'

# 1. Export to Excel (for exploration)
write_projection_excel(results, f'output/{base_name}.xlsx', include_charts=True)

# 2. Generate HTML report (for viewing)
generate_html_report(results, f'output/{base_name}_report.html')

# 3. Create all visualizations (for presentations)
save_all_visualizations(results, 'output/charts', base_name, image_format='png', dpi=300)

# 4. Create PDF versions of key charts (for printing)
plot_population_pyramid(results, 2045, f'output/{base_name}_pyramid_2045.pdf')
plot_population_trends(results, f'output/{base_name}_trends.pdf', by='age_group')
```

### Data Science Workflow

```python
# Efficient storage and analysis
write_projection_formats(
    results,
    output_dir='output/data',
    base_filename='projection',
    formats=['parquet', 'csv'],  # Parquet for analysis, CSV for sharing
    compression='gzip'
)

# Generate statistics for analysis
stats = generate_summary_statistics(results, include_diversity_metrics=True)

# Export statistics as JSON
import json
with open('output/statistics.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
```

### Scenario Analysis

```python
# Run multiple scenarios
scenarios = {
    'Baseline': baseline_results,
    'High Growth': high_growth_results,
    'Low Growth': low_growth_results
}

# Export each scenario
for name, results in scenarios.items():
    write_projection_excel(
        results,
        f'output/{name.lower().replace(" ", "_")}_scenario.xlsx'
    )

# Create comparison chart
plot_scenario_comparison(scenarios, 'output/scenario_comparison.png')

# Create comparison table
comparison = compare_scenarios(
    baseline_results,
    high_growth_results,
    'Baseline',
    'High Growth'
)
comparison.to_csv('output/scenario_comparison.csv', index=False)
```

## Troubleshooting

### ImportError: openpyxl not available

```bash
pip install openpyxl
```

### ImportError: matplotlib not available

```bash
pip install matplotlib
```

### Excel file too large (>1M rows)

Use Parquet or CSV instead:
```python
write_projection_formats(results, ..., formats=['parquet', 'csv'])
```

### Charts not displaying properly

Check matplotlib backend:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## See Also

- [Writers API Documentation](writers.py)
- [Reports API Documentation](reports.py)
- [Visualizations API Documentation](visualizations.py)
- [Example: Generate Outputs](../../examples/generate_outputs_example.py)
- [ADR 015: Output Format and Visualization Design](../../docs/adr/015-output-format-visualization-design.md)
