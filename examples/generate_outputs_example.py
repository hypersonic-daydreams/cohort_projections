"""
Comprehensive example demonstrating enhanced output capabilities.

This script shows how to:
1. Run a projection
2. Export to multiple formats (Excel, CSV, Parquet, JSON)
3. Generate summary statistics and reports
4. Create all standard visualizations
5. Compare scenarios
6. Create a complete stakeholder package

Run this script from the project root:
    python examples/generate_outputs_example.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.core import CohortComponentProjection
from cohort_projections.utils.config_loader import ConfigLoader

# Import all output functions
from cohort_projections.output import (
    # Writers
    write_projection_excel,
    write_projection_csv,
    write_projection_formats,

    # Reports
    generate_summary_statistics,
    compare_scenarios,
    generate_html_report,
    generate_text_report,

    # Visualizations
    plot_population_pyramid,
    plot_population_trends,
    plot_growth_rates,
    plot_scenario_comparison,
    save_all_visualizations,
)


def create_sample_data():
    """
    Create sample projection data for demonstration.

    In a real application, this would come from actual data sources.
    """
    print("Creating sample input data...")

    ages = list(range(0, 91))
    sexes = ['Male', 'Female']
    races = [
        'White alone, Non-Hispanic',
        'Black alone, Non-Hispanic',
        'Hispanic (any race)'
    ]

    # Base Population
    base_pop_data = []
    for age in ages:
        for sex in sexes:
            for race in races:
                if age < 18:
                    base_pop = 800
                elif age < 25:
                    base_pop = 1000
                elif age < 65:
                    base_pop = 1200
                else:
                    base_pop = max(100, 1200 - (age - 65) * 40)

                if race == 'White alone, Non-Hispanic':
                    base_pop *= 3.0
                elif race == 'Hispanic (any race)':
                    base_pop *= 1.5

                base_pop_data.append({
                    'year': 2025,
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'population': base_pop
                })

    base_population = pd.DataFrame(base_pop_data)

    # Fertility Rates
    fertility_data = []
    for age in range(15, 50):
        for race in races:
            if age < 20:
                rate = 0.02
            elif age < 25:
                rate = 0.08
            elif age < 30:
                rate = 0.12
            elif age < 35:
                rate = 0.10
            elif age < 40:
                rate = 0.05
            else:
                rate = 0.01

            if race == 'Hispanic (any race)':
                rate *= 1.2

            fertility_data.append({
                'age': age,
                'race': race,
                'fertility_rate': rate
            })

    fertility_rates = pd.DataFrame(fertility_data)

    # Survival Rates
    survival_data = []
    for age in ages:
        for sex in sexes:
            for race in races:
                if age == 0:
                    survival_rate = 0.9935
                elif age < 15:
                    survival_rate = 0.9995
                elif age < 45:
                    survival_rate = 0.9990
                elif age < 65:
                    survival_rate = 0.995
                elif age < 75:
                    survival_rate = 0.98
                elif age < 85:
                    survival_rate = 0.93
                else:
                    survival_rate = 0.80

                if sex == 'Male':
                    survival_rate *= 0.995

                survival_data.append({
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'survival_rate': survival_rate
                })

    survival_rates = pd.DataFrame(survival_data)

    # Migration Rates
    migration_data = []
    for age in ages:
        for sex in sexes:
            for race in races:
                if age < 18:
                    net_mig = 5
                elif age < 25:
                    net_mig = 50
                elif age < 35:
                    net_mig = 100
                elif age < 45:
                    net_mig = 30
                elif age < 65:
                    net_mig = 10
                else:
                    net_mig = -5

                if race == 'Hispanic (any race)':
                    net_mig *= 1.5

                migration_data.append({
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'net_migration': net_mig
                })

    migration_rates = pd.DataFrame(migration_data)

    return base_population, fertility_rates, survival_rates, migration_rates


def run_projection_with_scenario(
    base_population,
    fertility_rates,
    survival_rates,
    migration_rates,
    scenario_name='baseline',
    migration_multiplier=1.0
):
    """Run projection with scenario adjustments."""
    print(f"\nRunning {scenario_name} projection...")

    # Adjust migration for scenario
    adjusted_migration = migration_rates.copy()
    adjusted_migration['net_migration'] *= migration_multiplier

    projection = CohortComponentProjection(
        base_population=base_population,
        fertility_rates=fertility_rates,
        survival_rates=survival_rates,
        migration_rates=adjusted_migration
    )

    results = projection.run_projection(
        start_year=2025,
        end_year=2030,  # 5 years for quick demo
        scenario=scenario_name
    )

    summary = projection.get_projection_summary()

    return results, summary


def main():
    """Main demonstration script."""
    print("=" * 80)
    print("ENHANCED OUTPUT MODULE - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    print()

    # Create output directories
    output_root = project_root / "output" / "examples" / "enhanced_output"
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = output_root / "data"
    charts_dir = output_root / "charts"
    reports_dir = output_root / "reports"

    for directory in [data_dir, charts_dir, reports_dir]:
        directory.mkdir(exist_ok=True)

    # Create sample data
    base_population, fertility_rates, survival_rates, migration_rates = create_sample_data()

    # Run baseline projection
    baseline_results, baseline_summary = run_projection_with_scenario(
        base_population, fertility_rates, survival_rates, migration_rates,
        scenario_name='baseline',
        migration_multiplier=1.0
    )

    # Run high growth scenario
    high_growth_results, high_growth_summary = run_projection_with_scenario(
        base_population, fertility_rates, survival_rates, migration_rates,
        scenario_name='high_growth',
        migration_multiplier=1.5
    )

    # Run low growth scenario
    low_growth_results, low_growth_summary = run_projection_with_scenario(
        base_population, fertility_rates, survival_rates, migration_rates,
        scenario_name='low_growth',
        migration_multiplier=0.5
    )

    print("\n" + "=" * 80)
    print("PART 1: MULTI-FORMAT EXPORTS")
    print("=" * 80)

    # Create metadata
    metadata = {
        'geography': {
            'level': 'example',
            'name': 'Example Projection',
            'fips': 'EX001'
        },
        'projection': {
            'base_year': 2025,
            'end_year': 2030,
            'scenario': 'baseline',
            'processing_date': datetime.now().isoformat()
        },
        'data_sources': {
            'population': 'Sample Data',
            'fertility': 'Sample Rates',
            'mortality': 'Sample Rates',
            'migration': 'Sample Rates'
        }
    }

    # 1. Export to all formats at once
    print("\n1. Exporting to multiple formats...")
    format_paths = write_projection_formats(
        projection_df=baseline_results,
        output_dir=data_dir,
        base_filename='example_projection_baseline',
        formats=['csv', 'excel', 'parquet', 'json'],
        summary_df=baseline_summary,
        metadata=metadata,
        compression='gzip'
    )

    print("   Exported files:")
    for format_type, path in format_paths.items():
        file_size = path.stat().st_size if path.exists() else 0
        print(f"   - {format_type}: {path.name} ({file_size:,} bytes)")

    # 2. Export formatted Excel with charts
    print("\n2. Creating formatted Excel workbook...")
    excel_path = write_projection_excel(
        projection_df=baseline_results,
        output_path=data_dir / 'example_projection_formatted.xlsx',
        summary_df=baseline_summary,
        metadata=metadata,
        include_charts=True,
        include_formatting=True,
        title='Example Population Projection 2025-2030'
    )
    print(f"   Created: {excel_path.name}")

    # 3. Export CSV variants
    print("\n3. Creating CSV variants...")

    # Wide format
    csv_wide = write_projection_csv(
        projection_df=baseline_results,
        output_path=data_dir / 'example_projection_wide.csv',
        format_type='wide'
    )
    print(f"   - Wide format: {csv_wide.name}")

    # Filtered by age groups
    csv_filtered = write_projection_csv(
        projection_df=baseline_results,
        output_path=data_dir / 'example_projection_age_groups.csv.gz',
        format_type='long',
        age_ranges=[(0, 17), (18, 64), (65, 90)],
        compression='gzip'
    )
    print(f"   - Age groups: {csv_filtered.name}")

    print("\n" + "=" * 80)
    print("PART 2: SUMMARY STATISTICS AND REPORTS")
    print("=" * 80)

    # 4. Generate comprehensive statistics
    print("\n4. Generating summary statistics...")
    stats = generate_summary_statistics(
        projection_df=baseline_results,
        base_year=2025,
        include_diversity_metrics=True
    )

    print("\n   Key Statistics:")
    indicators = stats['demographic_indicators']
    print(f"   - Base Population: {indicators['base_population']:,.0f}")
    print(f"   - Final Population: {indicators['final_population']:,.0f}")
    print(f"   - Growth: {indicators['absolute_growth']:+,.0f} ({indicators['percent_growth']:+.1f}%)")
    print(f"   - Median Age Change: {indicators['median_age_base']:.1f} -> {indicators['median_age_final']:.1f}")
    print(f"   - Dependency Ratio Change: {indicators['dependency_ratio_base']:.2f} -> {indicators['dependency_ratio_final']:.2f}")

    # 5. Compare scenarios
    print("\n5. Comparing scenarios...")
    comparison = compare_scenarios(
        baseline_df=baseline_results,
        scenario_df=high_growth_results,
        baseline_name='Baseline',
        scenario_name='High Growth'
    )

    print("\n   Scenario Comparison:")
    print(comparison[['year', 'Baseline_total', 'High Growth_total', 'difference', 'percent_difference']].to_string(index=False))

    # Save comparison
    comparison.to_csv(reports_dir / 'scenario_comparison.csv', index=False)
    print(f"\n   Saved: {reports_dir / 'scenario_comparison.csv'}")

    # 6. Generate HTML report
    print("\n6. Generating HTML report...")
    html_report = generate_html_report(
        projection_df=baseline_results,
        output_path=reports_dir / 'projection_report.html',
        title='Example Population Projection Report 2025-2030',
        summary_stats=stats,
        metadata=metadata,
        include_methodology=True
    )
    print(f"   Created: {html_report.name}")

    # 7. Generate text reports
    print("\n7. Generating text reports...")

    # Plain text
    text_report = generate_text_report(
        projection_df=baseline_results,
        output_path=reports_dir / 'projection_report.txt',
        title='Example Population Projection Report',
        format_type='text',
        summary_stats=stats,
        include_tables=True
    )
    print(f"   - Plain text: {text_report.name}")

    # Markdown
    md_report = generate_text_report(
        projection_df=baseline_results,
        output_path=reports_dir / 'projection_report.md',
        title='Example Population Projection Report',
        format_type='markdown',
        summary_stats=stats,
        include_tables=True
    )
    print(f"   - Markdown: {md_report.name}")

    print("\n" + "=" * 80)
    print("PART 3: VISUALIZATIONS")
    print("=" * 80)

    # 8. Population pyramids
    print("\n8. Creating population pyramids...")

    # Simple pyramid
    pyramid_2030 = plot_population_pyramid(
        projection_df=baseline_results,
        year=2030,
        output_path=charts_dir / 'pyramid_2030_simple.png',
        age_group_size=5,
        dpi=300
    )
    print(f"   - Simple pyramid (2030): {pyramid_2030.name}")

    # Pyramid with race breakdown
    pyramid_2030_race = plot_population_pyramid(
        projection_df=baseline_results,
        year=2030,
        output_path=charts_dir / 'pyramid_2030_by_race.png',
        by_race=True,
        age_group_size=5,
        dpi=300
    )
    print(f"   - Race pyramid (2030): {pyramid_2030_race.name}")

    # 9. Population trends
    print("\n9. Creating population trend charts...")

    # Total population
    trends_total = plot_population_trends(
        projection_df=baseline_results,
        output_path=charts_dir / 'trends_total.png',
        by='total',
        title='Total Population Trends 2025-2030',
        dpi=300
    )
    print(f"   - Total trends: {trends_total.name}")

    # By age group
    trends_age = plot_population_trends(
        projection_df=baseline_results,
        output_path=charts_dir / 'trends_age_groups.png',
        by='age_group',
        age_groups={
            'Youth (0-17)': (0, 17),
            'Working Age (18-64)': (18, 64),
            'Elderly (65+)': (65, 90)
        },
        title='Population Trends by Age Group',
        dpi=300
    )
    print(f"   - Age group trends: {trends_age.name}")

    # By sex
    trends_sex = plot_population_trends(
        projection_df=baseline_results,
        output_path=charts_dir / 'trends_sex.png',
        by='sex',
        title='Population Trends by Sex',
        dpi=300
    )
    print(f"   - Sex trends: {trends_sex.name}")

    # By race
    trends_race = plot_population_trends(
        projection_df=baseline_results,
        output_path=charts_dir / 'trends_race.png',
        by='race',
        title='Population Trends by Race/Ethnicity',
        dpi=300
    )
    print(f"   - Race trends: {trends_race.name}")

    # 10. Growth rates
    print("\n10. Creating growth rate chart...")
    growth_rates = plot_growth_rates(
        projection_df=baseline_results,
        output_path=charts_dir / 'growth_rates_annual.png',
        period='annual',
        title='Annual Population Growth Rates',
        dpi=300
    )
    print(f"    Created: {growth_rates.name}")

    # 11. Scenario comparison
    print("\n11. Creating scenario comparison chart...")
    scenarios = {
        'Baseline': baseline_results,
        'High Growth': high_growth_results,
        'Low Growth': low_growth_results
    }

    scenario_chart = plot_scenario_comparison(
        scenario_projections=scenarios,
        output_path=charts_dir / 'scenario_comparison.png',
        title='Population Projection Scenarios',
        dpi=300
    )
    print(f"    Created: {scenario_chart.name}")

    # 12. Generate all standard visualizations
    print("\n12. Generating all standard visualizations...")
    all_viz_paths = save_all_visualizations(
        projection_df=baseline_results,
        output_dir=charts_dir / 'complete_set',
        base_filename='example_projection',
        years_for_pyramids=[2025, 2030],
        image_format='png',
        dpi=300
    )

    print(f"    Generated {len(all_viz_paths)} charts:")
    for chart_type, path in all_viz_paths.items():
        print(f"    - {chart_type}: {path.name}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nAll outputs saved to: {output_root}")
    print(f"\nDirectory structure:")
    print(f"  {data_dir.relative_to(output_root)}/")
    print(f"    - Excel, CSV, Parquet, JSON exports")
    print(f"  {charts_dir.relative_to(output_root)}/")
    print(f"    - All visualizations (pyramids, trends, growth rates)")
    print(f"  {reports_dir.relative_to(output_root)}/")
    print(f"    - HTML, text, and markdown reports")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Open the Excel file for interactive exploration")
    print("2. View the HTML report in your browser")
    print("3. Review the charts in the charts/ directory")
    print("4. Check the text report for a summary")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
