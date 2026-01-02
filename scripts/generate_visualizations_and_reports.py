#!/usr/bin/env python3
"""
Generate population visualizations and summary reports from projection data.

This script:
1. Aggregates county projections to state level
2. Generates population pyramids for key years (2025, 2035, 2045)
3. Generates trend charts comparing all three scenarios
4. Generates summary statistics reports (HTML and Markdown)

Usage:
    python scripts/generate_visualizations_and_reports.py
"""

import glob
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.output.reports import (  # noqa: E402
    compare_scenarios,
    generate_html_report,
    generate_summary_statistics,
    generate_text_report,
)
from cohort_projections.output.visualizations import (  # noqa: E402
    plot_growth_rates,
    plot_population_pyramid,
    plot_scenario_comparison,
    save_all_visualizations,
)


def load_scenario_data(scenario_name: str, base_path: Path) -> pd.DataFrame:
    """
    Load and aggregate all county projection files for a scenario to state level.

    Args:
        scenario_name: Name of the scenario (baseline, high_growth, low_growth)
        base_path: Base path to projections directory

    Returns:
        DataFrame with aggregated state-level projection data
    """
    scenario_path = base_path / scenario_name / "county"
    parquet_files = glob.glob(str(scenario_path / "*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {scenario_path}")

    print(f"  Loading {len(parquet_files)} county files for {scenario_name}...")

    # Load all county files
    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        dfs.append(df)

    # Concatenate all counties
    combined = pd.concat(dfs, ignore_index=True)

    # Aggregate to state level (sum populations across counties)
    state_df = combined.groupby(["year", "age", "sex", "race"], as_index=False).agg(
        {"population": "sum"}
    )

    print(f"  Aggregated to state level: {len(state_df):,} rows")
    print(f"  Years: {state_df['year'].min()} - {state_df['year'].max()}")
    print(
        f"  Total population ({state_df['year'].min()}): {state_df[state_df['year'] == state_df['year'].min()]['population'].sum():,.0f}"
    )
    print(
        f"  Total population ({state_df['year'].max()}): {state_df[state_df['year'] == state_df['year'].max()]['population'].sum():,.0f}"
    )

    return state_df


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("North Dakota Population Projection Visualizations & Reports")
    print("=" * 70)
    print()

    # Paths
    projections_path = project_root / "data" / "projections"
    viz_output_path = project_root / "data" / "output" / "visualizations"
    reports_output_path = project_root / "data" / "output" / "reports"

    # Create output directories
    viz_output_path.mkdir(parents=True, exist_ok=True)
    reports_output_path.mkdir(parents=True, exist_ok=True)

    scenarios = ["baseline", "high_growth", "low_growth"]
    scenario_display_names = {
        "baseline": "Baseline",
        "high_growth": "High Growth",
        "low_growth": "Low Growth",
    }

    # Load all scenario data
    print("Loading projection data...")
    print("-" * 40)

    scenario_data = {}
    for scenario in scenarios:
        print(f"\n{scenario_display_names[scenario]}:")
        try:
            scenario_data[scenario] = load_scenario_data(scenario, projections_path)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

    if not scenario_data:
        print("\nError: No scenario data found!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    # Key years for pyramids
    key_years = [2025, 2035, 2045]

    # Generate visualizations for baseline scenario
    baseline_df = scenario_data.get("baseline")
    if baseline_df is not None:
        print("\n1. Population Pyramids (State Level)")
        print("-" * 40)

        for year in key_years:
            output_file = viz_output_path / f"nd_state_pyramid_{year}.png"
            print(f"  Generating pyramid for {year}...")
            plot_population_pyramid(
                baseline_df,
                year=year,
                output_path=output_file,
                title=f"North Dakota Population Pyramid - {year}\n(Baseline Scenario)",
                figsize=(12, 10),
            )
            print(f"    Saved: {output_file}")

        # Also generate a version with race breakdown
        print("\n  Generating pyramid with race breakdown for 2045...")
        output_file = viz_output_path / "nd_state_pyramid_2045_by_race.png"
        plot_population_pyramid(
            baseline_df,
            year=2045,
            output_path=output_file,
            by_race=True,
            title="North Dakota Population Pyramid by Race - 2045\n(Baseline Scenario)",
            figsize=(14, 10),
        )
        print(f"    Saved: {output_file}")

        print("\n2. Population Trends (Baseline)")
        print("-" * 40)

        # Save all standard visualizations for baseline
        print("  Generating all standard charts...")
        viz_paths = save_all_visualizations(
            baseline_df,
            output_dir=viz_output_path / "baseline",
            base_filename="nd_state",
            years_for_pyramids=key_years,
        )
        for chart_name, path in viz_paths.items():
            print(f"    {chart_name}: {path}")

    # Scenario comparison
    print("\n3. Scenario Comparison Charts")
    print("-" * 40)

    if len(scenario_data) >= 2:
        # Prepare scenario dict with display names
        scenario_projections = {
            scenario_display_names[name]: df for name, df in scenario_data.items()
        }

        output_file = viz_output_path / "nd_state_scenario_comparison.png"
        print("  Generating scenario comparison...")
        plot_scenario_comparison(
            scenario_projections,
            output_path=output_file,
            title="North Dakota Population Projections: Scenario Comparison\n(2025-2045)",
            figsize=(14, 8),
        )
        print(f"    Saved: {output_file}")

        # Growth rates for each scenario
        for scenario_name, df in scenario_data.items():
            output_file = viz_output_path / f"nd_state_growth_rates_{scenario_name}.png"
            print(f"  Generating growth rates for {scenario_display_names[scenario_name]}...")
            plot_growth_rates(
                df,
                output_path=output_file,
                period="annual",
                title=f"North Dakota Annual Growth Rates - {scenario_display_names[scenario_name]}",
            )
            print(f"    Saved: {output_file}")

    print("\n" + "=" * 70)
    print("Generating Reports")
    print("=" * 70)

    # Generate reports for each scenario
    print("\n4. Summary Statistics Reports")
    print("-" * 40)

    for scenario_name, df in scenario_data.items():
        display_name = scenario_display_names[scenario_name]

        # Generate summary statistics
        print(f"\n  {display_name} Scenario:")
        stats = generate_summary_statistics(df)

        # Save as JSON
        import json

        stats_file = reports_output_path / f"nd_state_{scenario_name}_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"    Statistics JSON: {stats_file}")

        # HTML report
        html_file = reports_output_path / f"nd_state_{scenario_name}_report.html"
        generate_html_report(
            df,
            output_path=html_file,
            title=f"North Dakota Population Projections 2025-2045\n{display_name} Scenario",
            summary_stats=stats,
            metadata={
                "scenario": scenario_name,
                "geographic_level": "State",
                "projection_years": "2025-2045",
            },
        )
        print(f"    HTML Report: {html_file}")

        # Markdown report
        md_file = reports_output_path / f"nd_state_{scenario_name}_report.md"
        generate_text_report(
            df,
            output_path=md_file,
            title=f"North Dakota Population Projections 2025-2045 - {display_name} Scenario",
            format_type="markdown",
            summary_stats=stats,
        )
        print(f"    Markdown Report: {md_file}")

    # Scenario comparison report
    print("\n5. Scenario Comparison Report")
    print("-" * 40)

    if "baseline" in scenario_data and len(scenario_data) > 1:
        baseline = scenario_data["baseline"]

        for scenario_name, df in scenario_data.items():
            if scenario_name == "baseline":
                continue

            display_name = scenario_display_names[scenario_name]
            print(f"  Comparing Baseline vs {display_name}...")

            comparison = compare_scenarios(
                baseline,
                df,
                baseline_name="Baseline",
                scenario_name=display_name,
                years_to_compare=[2025, 2030, 2035, 2040, 2045],
            )

            # Save comparison as CSV
            comparison_file = (
                reports_output_path / f"nd_state_baseline_vs_{scenario_name}_comparison.csv"
            )
            comparison.to_csv(comparison_file, index=False)
            print(f"    Comparison CSV: {comparison_file}")

    # Summary of all outputs
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nVisualizations saved to: {viz_output_path}")
    viz_files = list(viz_output_path.glob("**/*.png"))
    print(f"  Total visualization files: {len(viz_files)}")

    print(f"\nReports saved to: {reports_output_path}")
    report_files = list(reports_output_path.glob("*"))
    print(f"  Total report files: {len(report_files)}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
