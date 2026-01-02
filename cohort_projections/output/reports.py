"""
Report generation for projection results.

This module provides functions to generate summary statistics and formatted
reports from population projection data, including HTML and text reports.

Functions:
    generate_summary_statistics: Calculate comprehensive statistics
    compare_scenarios: Compare baseline vs alternative scenarios
    generate_html_report: Create formatted HTML report
    generate_text_report: Create plain text/markdown report
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..utils import get_logger_from_config

logger = get_logger_from_config(__name__)


def generate_summary_statistics(
    projection_df: pd.DataFrame,
    base_year: int | None = None,
    include_diversity_metrics: bool = True,
) -> dict[str, Any]:
    """
    Calculate comprehensive summary statistics from projection data.

    Computes:
    - Total population by year
    - Growth rates (annual and period)
    - Age distribution (youth, working age, elderly)
    - Dependency ratios (total, youth, elderly)
    - Sex ratios
    - Diversity metrics (if requested)

    Args:
        projection_df: Projection DataFrame with columns [year, age, sex, race, population]
        base_year: Base year (default: minimum year in data)
        include_diversity_metrics: Whether to calculate diversity indices

    Returns:
        Dictionary with comprehensive statistics:
        {
            'by_year': List of annual statistics,
            'age_structure': Age group statistics,
            'demographic_indicators': Dependency ratios, sex ratios, etc.,
            'diversity': Diversity metrics (if requested),
            'growth_analysis': Growth rates and trends
        }

    Example:
        >>> stats = generate_summary_statistics(projection_df)
        >>> stats['demographic_indicators']['dependency_ratio_2045']
        0.68
    """
    logger.info("Generating summary statistics")

    if projection_df.empty:
        logger.warning("Empty projection data - returning empty statistics")
        return {}

    # Validate columns
    required_cols = ["year", "age", "sex", "race", "population"]
    missing_cols = [col for col in required_cols if col not in projection_df.columns]
    if missing_cols:
        raise ValueError(f"projection_df missing required columns: {missing_cols}")

    years = sorted(projection_df["year"].unique())
    if base_year is None:
        base_year = years[0]

    logger.debug(f"Analyzing {len(years)} years from {years[0]} to {years[-1]}")

    # 1. Annual Statistics
    annual_stats = []

    for year in years:
        year_data = projection_df[projection_df["year"] == year]
        total_pop = year_data["population"].sum()

        # By sex
        male_pop = year_data[year_data["sex"] == "Male"]["population"].sum()
        female_pop = year_data[year_data["sex"] == "Female"]["population"].sum()
        sex_ratio = (male_pop / female_pop * 100) if female_pop > 0 else 0

        # By age groups
        youth = year_data[year_data["age"] < 18]["population"].sum()
        working_age = year_data[(year_data["age"] >= 18) & (year_data["age"] < 65)][
            "population"
        ].sum()
        elderly = year_data[year_data["age"] >= 65]["population"].sum()

        # Dependency ratios
        total_dependent = youth + elderly
        dependency_ratio = (total_dependent / working_age) if working_age > 0 else 0
        youth_dependency = (youth / working_age) if working_age > 0 else 0
        elderly_dependency = (elderly / working_age) if working_age > 0 else 0

        # Median age (approximate)
        median_age = _calculate_median_age(year_data)

        annual_stats.append(
            {
                "year": int(year),
                "total_population": float(total_pop),
                "male": float(male_pop),
                "female": float(female_pop),
                "sex_ratio": float(sex_ratio),
                "youth_0_17": float(youth),
                "working_age_18_64": float(working_age),
                "elderly_65_plus": float(elderly),
                "dependency_ratio": float(dependency_ratio),
                "youth_dependency_ratio": float(youth_dependency),
                "elderly_dependency_ratio": float(elderly_dependency),
                "median_age": float(median_age),
            }
        )

    # 2. Age Structure Summary
    age_structure = {}

    for year in [years[0], years[len(years) // 2], years[-1]]:
        year_data = projection_df[projection_df["year"] == year]

        # Standard age groups
        age_groups = {
            "0-4": (0, 4),
            "5-17": (5, 17),
            "18-24": (18, 24),
            "25-44": (25, 44),
            "45-64": (45, 64),
            "65-74": (65, 74),
            "75-84": (75, 84),
            "85+": (85, 150),
        }

        year_structure = {}
        total = year_data["population"].sum()

        for group_name, (min_age, max_age) in age_groups.items():
            group_pop = year_data[(year_data["age"] >= min_age) & (year_data["age"] <= max_age)][
                "population"
            ].sum()
            year_structure[group_name] = {
                "population": float(group_pop),
                "percent": float((group_pop / total * 100) if total > 0 else 0),
            }

        age_structure[f"year_{int(year)}"] = year_structure

    # 3. Demographic Indicators
    base_pop = annual_stats[0]["total_population"]
    final_pop = annual_stats[-1]["total_population"]
    total_years = years[-1] - years[0]

    # Growth metrics
    absolute_growth = final_pop - base_pop
    percent_growth = ((final_pop / base_pop - 1) * 100) if base_pop > 0 else 0
    annual_growth_rate = (
        (((final_pop / base_pop) ** (1 / total_years)) - 1) * 100
        if total_years > 0 and base_pop > 0
        else 0
    )

    demographic_indicators = {
        "base_year": int(base_year),
        "final_year": int(years[-1]),
        "base_population": float(base_pop),
        "final_population": float(final_pop),
        "absolute_growth": float(absolute_growth),
        "percent_growth": float(percent_growth),
        "annual_growth_rate": float(annual_growth_rate),
        "dependency_ratio_base": float(annual_stats[0]["dependency_ratio"]),
        "dependency_ratio_final": float(annual_stats[-1]["dependency_ratio"]),
        "median_age_base": float(annual_stats[0]["median_age"]),
        "median_age_final": float(annual_stats[-1]["median_age"]),
        "sex_ratio_base": float(annual_stats[0]["sex_ratio"]),
        "sex_ratio_final": float(annual_stats[-1]["sex_ratio"]),
    }

    # 4. Diversity Metrics (if requested)
    diversity = {}

    if include_diversity_metrics:
        logger.debug("Calculating diversity metrics")

        for year in [years[0], years[-1]]:
            year_data = projection_df[projection_df["year"] == year]
            total = year_data["population"].sum()

            # Race/ethnicity distribution
            race_dist = {}
            for race in year_data["race"].unique():
                race_pop = year_data[year_data["race"] == race]["population"].sum()
                race_dist[race] = {
                    "population": float(race_pop),
                    "percent": float((race_pop / total * 100) if total > 0 else 0),
                }

            # Diversity index (Simpson's Diversity Index)
            # D = 1 - sum(p_i^2) where p_i is proportion of group i
            proportions = [v["population"] / total for v in race_dist.values() if total > 0]
            diversity_index = 1 - sum(p**2 for p in proportions) if proportions else 0

            diversity[f"year_{int(year)}"] = {
                "race_distribution": race_dist,
                "diversity_index": float(diversity_index),
                "majority_group": max(race_dist.items(), key=lambda x: x[1]["population"])[0]
                if race_dist
                else None,
            }

    # 5. Growth Analysis
    growth_analysis: dict[str, list[dict[str, Any]]] = {
        "annual_growth_rates": [],
        "period_growth_rates": [],
    }

    # Annual growth rates
    for i in range(1, len(annual_stats)):
        prev_pop = annual_stats[i - 1]["total_population"]
        curr_pop = annual_stats[i]["total_population"]
        growth_rate = ((curr_pop / prev_pop - 1) * 100) if prev_pop > 0 else 0

        growth_analysis["annual_growth_rates"].append(
            {
                "year": annual_stats[i]["year"],
                "growth_rate": float(growth_rate),
                "absolute_change": float(curr_pop - prev_pop),
            }
        )

    # 5-year period growth rates
    for i in range(0, len(years), 5):
        if i + 5 < len(years):
            start_year = years[i]
            end_year = years[i + 5]
            start_pop = annual_stats[i]["total_population"]
            end_pop = annual_stats[i + 5]["total_population"]

            period_growth = ((end_pop / start_pop - 1) * 100) if start_pop > 0 else 0
            annual_avg = (((end_pop / start_pop) ** (1 / 5)) - 1) * 100 if start_pop > 0 else 0

            growth_analysis["period_growth_rates"].append(
                {
                    "period": f"{int(start_year)}-{int(end_year)}",
                    "start_population": float(start_pop),
                    "end_population": float(end_pop),
                    "period_growth": float(period_growth),
                    "annual_average": float(annual_avg),
                }
            )

    # Assemble final statistics
    statistics = {
        "by_year": annual_stats,
        "age_structure": age_structure,
        "demographic_indicators": demographic_indicators,
        "diversity": diversity,
        "growth_analysis": growth_analysis,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    logger.info(f"Summary statistics generated for {len(years)} years")

    return statistics


def compare_scenarios(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_name: str = "Baseline",
    scenario_name: str = "Alternative",
    years_to_compare: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compare baseline projection vs alternative scenario.

    Creates side-by-side comparison showing:
    - Population under each scenario
    - Absolute differences
    - Percentage differences
    - Key metrics comparison

    Args:
        baseline_df: Baseline projection DataFrame
        scenario_df: Alternative scenario projection DataFrame
        baseline_name: Name for baseline scenario
        scenario_name: Name for alternative scenario
        years_to_compare: Optional list of specific years (default: all years)

    Returns:
        Comparison DataFrame with columns for both scenarios and differences

    Example:
        >>> comparison = compare_scenarios(
        ...     baseline_df=baseline_results,
        ...     scenario_df=high_growth_results,
        ...     scenario_name="High Growth"
        ... )
    """
    logger.info(f"Comparing scenarios: {baseline_name} vs {scenario_name}")

    # Get years
    baseline_years = set(baseline_df["year"].unique())
    scenario_years = set(scenario_df["year"].unique())
    common_years = sorted(baseline_years & scenario_years)

    if years_to_compare:
        common_years = [y for y in years_to_compare if y in common_years]

    logger.debug(f"Comparing {len(common_years)} years")

    # Calculate totals by year
    comparison_data = []

    for year in common_years:
        baseline_year = baseline_df[baseline_df["year"] == year]
        scenario_year = scenario_df[scenario_df["year"] == year]

        baseline_total = baseline_year["population"].sum()
        scenario_total = scenario_year["population"].sum()

        difference = scenario_total - baseline_total
        pct_difference = ((difference / baseline_total) * 100) if baseline_total > 0 else 0

        # Age group comparisons
        baseline_youth = baseline_year[baseline_year["age"] < 18]["population"].sum()
        scenario_youth = scenario_year[scenario_year["age"] < 18]["population"].sum()

        baseline_working = baseline_year[
            (baseline_year["age"] >= 18) & (baseline_year["age"] < 65)
        ]["population"].sum()
        scenario_working = scenario_year[
            (scenario_year["age"] >= 18) & (scenario_year["age"] < 65)
        ]["population"].sum()

        baseline_elderly = baseline_year[baseline_year["age"] >= 65]["population"].sum()
        scenario_elderly = scenario_year[scenario_year["age"] >= 65]["population"].sum()

        comparison_data.append(
            {
                "year": int(year),
                f"{baseline_name}_total": float(baseline_total),
                f"{scenario_name}_total": float(scenario_total),
                "difference": float(difference),
                "percent_difference": float(pct_difference),
                f"{baseline_name}_youth": float(baseline_youth),
                f"{scenario_name}_youth": float(scenario_youth),
                f"{baseline_name}_working": float(baseline_working),
                f"{scenario_name}_working": float(scenario_working),
                f"{baseline_name}_elderly": float(baseline_elderly),
                f"{scenario_name}_elderly": float(scenario_elderly),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    logger.info(f"Scenario comparison complete: {len(comparison_df)} years analyzed")

    return comparison_df


def generate_html_report(
    projection_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Population Projection Report",
    summary_stats: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    include_methodology: bool = True,
    template_path: Path | None = None,
) -> Path:
    """
    Generate formatted HTML report.

    Creates an HTML report with:
    - Executive summary
    - Key findings
    - Summary statistics tables
    - Methodology section (optional)
    - Metadata and parameters

    Args:
        projection_df: Projection DataFrame
        output_path: Path to output HTML file
        title: Report title
        summary_stats: Pre-computed summary statistics (or will generate)
        metadata: Optional metadata to include
        include_methodology: Whether to include methodology section
        template_path: Optional custom HTML template

    Returns:
        Path to created HTML file

    Example:
        >>> generate_html_report(
        ...     projection_df=results,
        ...     output_path='output/report.html',
        ...     title='North Dakota Population Projections 2025-2045'
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating HTML report: {output_path}")

    # Generate summary statistics if not provided
    if summary_stats is None:
        summary_stats = generate_summary_statistics(projection_df)

    # Build HTML
    html = _build_html_report(
        title=title,
        projection_df=projection_df,
        summary_stats=summary_stats,
        metadata=metadata,
        include_methodology=include_methodology,
    )

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report generated: {output_path}")

    return output_path


def generate_text_report(
    projection_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Population Projection Report",
    format_type: Literal["text", "markdown"] = "text",
    summary_stats: dict[str, Any] | None = None,
    include_tables: bool = True,
) -> Path:
    """
    Generate plain text or Markdown report.

    Creates a text-based report suitable for:
    - Console output
    - Email
    - Documentation
    - README files

    Args:
        projection_df: Projection DataFrame
        output_path: Path to output file
        title: Report title
        format_type: 'text' for plain text or 'markdown' for Markdown
        summary_stats: Pre-computed summary statistics (or will generate)
        include_tables: Whether to include data tables

    Returns:
        Path to created text file

    Example:
        >>> generate_text_report(
        ...     projection_df=results,
        ...     output_path='output/report.md',
        ...     format_type='markdown'
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {format_type} report: {output_path}")

    # Generate summary statistics if not provided
    if summary_stats is None:
        summary_stats = generate_summary_statistics(projection_df)

    lines = []

    # Title
    if format_type == "markdown":
        lines.append(f"# {title}\n")
    else:
        lines.append("=" * len(title))
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    # Generation info
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive Summary
    if format_type == "markdown":
        lines.append("## Executive Summary\n")
    else:
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)

    indicators = summary_stats.get("demographic_indicators", {})

    base_year = indicators.get("base_year", "N/A")
    final_year = indicators.get("final_year", "N/A")
    base_pop = indicators.get("base_population", 0)
    final_pop = indicators.get("final_population", 0)
    growth = indicators.get("absolute_growth", 0)
    pct_growth = indicators.get("percent_growth", 0)

    lines.append(f"Projection Period: {base_year} - {final_year}")
    lines.append(f"Base Year Population: {base_pop:,.0f}")
    lines.append(f"Final Year Population: {final_pop:,.0f}")
    lines.append(f"Projected Growth: {growth:+,.0f} ({pct_growth:+.1f}%)")
    lines.append("")

    # Key Findings
    if format_type == "markdown":
        lines.append("## Key Findings\n")
    else:
        lines.append("KEY FINDINGS")
        lines.append("-" * 40)

    median_age_base = indicators.get("median_age_base", 0)
    median_age_final = indicators.get("median_age_final", 0)
    dep_ratio_base = indicators.get("dependency_ratio_base", 0)
    dep_ratio_final = indicators.get("dependency_ratio_final", 0)

    lines.append(
        f"- Median age increases from {median_age_base:.1f} to {median_age_final:.1f} years"
    )
    lines.append(f"- Dependency ratio changes from {dep_ratio_base:.2f} to {dep_ratio_final:.2f}")

    # Add age structure findings
    by_year = summary_stats.get("by_year", [])
    if len(by_year) >= 2:
        base_youth_pct = (by_year[0]["youth_0_17"] / by_year[0]["total_population"]) * 100
        final_youth_pct = (by_year[-1]["youth_0_17"] / by_year[-1]["total_population"]) * 100
        base_elderly_pct = (by_year[0]["elderly_65_plus"] / by_year[0]["total_population"]) * 100
        final_elderly_pct = (by_year[-1]["elderly_65_plus"] / by_year[-1]["total_population"]) * 100

        lines.append(f"- Youth (0-17) share: {base_youth_pct:.1f}% -> {final_youth_pct:.1f}%")
        lines.append(f"- Elderly (65+) share: {base_elderly_pct:.1f}% -> {final_elderly_pct:.1f}%")

    lines.append("")

    # Population by Year Table
    if include_tables and by_year:
        if format_type == "markdown":
            lines.append("## Population Trends\n")
            lines.append("| Year | Total Population | Growth Rate | Dependency Ratio |")
            lines.append("|------|------------------|-------------|------------------|")

            for i, year_data in enumerate(by_year):
                year = year_data["year"]
                pop = year_data["total_population"]
                dep_ratio = year_data["dependency_ratio"]

                # Calculate growth rate from previous year
                if i > 0:
                    prev_pop = by_year[i - 1]["total_population"]
                    growth_rate = ((pop / prev_pop - 1) * 100) if prev_pop > 0 else 0
                    lines.append(f"| {year} | {pop:,.0f} | {growth_rate:+.2f}% | {dep_ratio:.2f} |")
                else:
                    lines.append(f"| {year} | {pop:,.0f} | -- | {dep_ratio:.2f} |")

            lines.append("")
        else:
            lines.append("POPULATION TRENDS")
            lines.append("-" * 70)
            lines.append(f"{'Year':<8} {'Population':>15} {'Growth Rate':>12} {'Dep. Ratio':>12}")
            lines.append("-" * 70)

            for i, year_data in enumerate(by_year):
                year = year_data["year"]
                pop = year_data["total_population"]
                dep_ratio = year_data["dependency_ratio"]

                if i > 0:
                    prev_pop = by_year[i - 1]["total_population"]
                    growth_rate = ((pop / prev_pop - 1) * 100) if prev_pop > 0 else 0
                    lines.append(
                        f"{year:<8} {pop:>15,.0f} {growth_rate:>11.2f}% {dep_ratio:>12.2f}"
                    )
                else:
                    lines.append(f"{year:<8} {pop:>15,.0f} {'--':>12} {dep_ratio:>12.2f}")

            lines.append("")

    # Methodology
    if format_type == "markdown":
        lines.append("## Methodology\n")
    else:
        lines.append("METHODOLOGY")
        lines.append("-" * 40)

    lines.append("This projection uses the cohort-component method, which projects population by:")
    lines.append("1. Aging existing cohorts and applying survival rates")
    lines.append("2. Adding births based on age-specific fertility rates")
    lines.append("3. Adding net migration by cohort")
    lines.append("")

    # Footer
    lines.append("-" * 70)
    lines.append("North Dakota Cohort Component Projection System")
    lines.append("")

    # Write to file
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"{format_type.title()} report generated: {output_path}")

    return output_path


def _calculate_median_age(population_df: pd.DataFrame) -> float:
    """
    Calculate median age from population data.

    Args:
        population_df: DataFrame with 'age' and 'population' columns

    Returns:
        Median age
    """
    if population_df.empty:
        return 0.0

    # Create age distribution
    age_dist = population_df.groupby("age")["population"].sum().sort_index()

    if age_dist.sum() == 0:
        return 0.0

    # Find median (50th percentile)
    cumsum = age_dist.cumsum()
    total = age_dist.sum()
    median_idx = (cumsum >= total / 2).idxmax()

    return float(median_idx)


def _build_html_report(
    title: str,
    projection_df: pd.DataFrame,
    summary_stats: dict[str, Any],
    metadata: dict[str, Any] | None,
    include_methodology: bool,
) -> str:
    """Build HTML report content."""

    indicators = summary_stats.get("demographic_indicators", {})
    by_year = summary_stats.get("by_year", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="summary-box">
        <p><strong>Generated:</strong> {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Projection Period:</strong> {indicators.get("base_year", "N/A")} - {indicators.get("final_year", "N/A")}</p>
    </div>

    <h2>Executive Summary</h2>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Base Year Population</div>
            <div class="stat-value">{indicators.get("base_population", 0):,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Final Year Population</div>
            <div class="stat-value">{indicators.get("final_population", 0):,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Projected Growth</div>
            <div class="stat-value {"positive" if indicators.get("absolute_growth", 0) >= 0 else "negative"}">{indicators.get("absolute_growth", 0):+,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Growth Rate</div>
            <div class="stat-value {"positive" if indicators.get("percent_growth", 0) >= 0 else "negative"}">{indicators.get("percent_growth", 0):+.1f}%</div>
        </div>
    </div>

    <h2>Key Demographic Indicators</h2>

    <table>
        <tr>
            <th>Indicator</th>
            <th>Base Year</th>
            <th>Final Year</th>
            <th>Change</th>
        </tr>
        <tr>
            <td>Median Age</td>
            <td>{indicators.get("median_age_base", 0):.1f}</td>
            <td>{indicators.get("median_age_final", 0):.1f}</td>
            <td>{indicators.get("median_age_final", 0) - indicators.get("median_age_base", 0):+.1f}</td>
        </tr>
        <tr>
            <td>Dependency Ratio</td>
            <td>{indicators.get("dependency_ratio_base", 0):.2f}</td>
            <td>{indicators.get("dependency_ratio_final", 0):.2f}</td>
            <td>{indicators.get("dependency_ratio_final", 0) - indicators.get("dependency_ratio_base", 0):+.2f}</td>
        </tr>
        <tr>
            <td>Sex Ratio (males per 100 females)</td>
            <td>{indicators.get("sex_ratio_base", 0):.1f}</td>
            <td>{indicators.get("sex_ratio_final", 0):.1f}</td>
            <td>{indicators.get("sex_ratio_final", 0) - indicators.get("sex_ratio_base", 0):+.1f}</td>
        </tr>
    </table>

    <h2>Population Trends</h2>

    <table>
        <tr>
            <th>Year</th>
            <th>Total Population</th>
            <th>Youth (0-17)</th>
            <th>Working Age (18-64)</th>
            <th>Elderly (65+)</th>
            <th>Dependency Ratio</th>
        </tr>
"""

    # Add yearly data
    for year_data in by_year:
        html += f"""        <tr>
            <td>{year_data["year"]}</td>
            <td>{year_data["total_population"]:,.0f}</td>
            <td>{year_data["youth_0_17"]:,.0f}</td>
            <td>{year_data["working_age_18_64"]:,.0f}</td>
            <td>{year_data["elderly_65_plus"]:,.0f}</td>
            <td>{year_data["dependency_ratio"]:.2f}</td>
        </tr>
"""

    html += """    </table>
"""

    # Methodology section
    if include_methodology:
        html += """
    <h2>Methodology</h2>

    <p>This projection employs the <strong>cohort-component method</strong>, the standard demographic technique for population projections.</p>

    <p>The method projects population forward by:</p>
    <ol>
        <li><strong>Survival:</strong> Aging existing cohorts and applying age-sex-race-specific survival rates to account for mortality</li>
        <li><strong>Fertility:</strong> Calculating births by applying age-specific fertility rates to the female population of childbearing age</li>
        <li><strong>Migration:</strong> Adding net migration (domestic and international) by cohort</li>
    </ol>

    <p>The projection is performed annually, with each year building on the previous year's population structure.</p>
"""

    # Footer
    html += f"""
    <div class="footer">
        <p><strong>North Dakota Cohort Component Projection System</strong></p>
        <p>Generated: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

</body>
</html>
"""

    return html
