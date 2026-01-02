"""
Visualization tools for projection data.

This module provides functions to create charts and graphs from population
projection data, including population pyramids, trend charts, and component analysis.

Functions:
    plot_population_pyramid: Create population pyramid for a specific year
    plot_population_trends: Line chart of population over time
    plot_growth_rates: Visualize growth rates
    plot_component_analysis: Visualize demographic components
    plot_scenario_comparison: Compare multiple scenarios
    save_all_visualizations: Generate all standard charts
"""

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Matplotlib for static visualizations
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualizations will not work", stacklevel=2)

# Optional: seaborn for enhanced styling
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from ..utils import get_logger_from_config

logger = get_logger_from_config(__name__)


def plot_population_pyramid(
    projection_df: pd.DataFrame,
    year: int,
    output_path: str | Path,
    by_race: bool = False,
    age_group_size: int = 5,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> Path:
    """
    Create a population pyramid for a specific year.

    Args:
        projection_df: Projection DataFrame
        year: Year to visualize
        output_path: Path to save figure (PNG, SVG, PDF supported)
        by_race: Whether to show race breakdown (stacked)
        age_group_size: Size of age groups (1, 5, or 10 years)
        title: Optional custom title
        figsize: Figure size (width, height) in inches
        dpi: Resolution for raster formats
        style: Matplotlib style to use

    Returns:
        Path to saved figure

    Raises:
        ImportError: If matplotlib is not available
        ValueError: If year not in projection data

    Example:
        >>> plot_population_pyramid(
        ...     projection_df=results,
        ...     year=2045,
        ...     output_path='output/pyramid_2045.png'
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib required for visualizations. Install with: pip install matplotlib"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating population pyramid for year {year}: {output_path}")

    # Filter to specific year
    year_data = projection_df[projection_df["year"] == year].copy()

    if year_data.empty:
        raise ValueError(f"No data for year {year}")

    # Create age groups
    if age_group_size > 1:
        year_data["age_group"] = (year_data["age"] // age_group_size) * age_group_size
    else:
        year_data["age_group"] = year_data["age"]

    # Set style
    if SEABORN_AVAILABLE and style.startswith("seaborn"):
        sns.set_style("darkgrid")
    else:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if by_race:
        # Stacked pyramid by race
        races = sorted(year_data["race"].unique())

        # Get data for males and females by age group and race
        male_data = (
            year_data[year_data["sex"] == "Male"]
            .groupby(["age_group", "race"])["population"]
            .sum()
            .unstack(fill_value=0)
        )
        female_data = (
            year_data[year_data["sex"] == "Female"]
            .groupby(["age_group", "race"])["population"]
            .sum()
            .unstack(fill_value=0)
        )

        # Ensure all races present in both
        for race in races:
            if race not in male_data.columns:
                male_data[race] = 0
            if race not in female_data.columns:
                female_data[race] = 0

        male_data = male_data[races]
        female_data = female_data[races]

        age_groups = sorted(set(male_data.index) | set(female_data.index))

        # Ensure all age groups present
        male_data = male_data.reindex(age_groups, fill_value=0)
        female_data = female_data.reindex(age_groups, fill_value=0)

        # Plot stacked bars
        y_pos = np.arange(len(age_groups))
        bar_height = 0.8

        # Color palette
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("Set2", len(races))
        else:
            colors = plt.colormaps.get_cmap("Set2")(np.linspace(0, 1, len(races)))

        # Males (left side, negative values)
        left_cumsum = np.zeros(len(age_groups))
        for i, race in enumerate(races):
            male_vals = -male_data[race].values  # Negative for left side
            ax.barh(
                y_pos,
                male_vals,
                bar_height,
                left=left_cumsum,
                label=race if i == 0 else "",
                color=colors[i],
            )
            left_cumsum += male_vals

        # Females (right side, positive values)
        right_cumsum = np.zeros(len(age_groups))
        for i, race in enumerate(races):
            female_vals = female_data[race].values
            ax.barh(y_pos, female_vals, bar_height, left=right_cumsum, color=colors[i])
            right_cumsum += female_vals

        # Legend for races
        legend_patches = [
            mpatches.Patch(color=colors[i], label=race) for i, race in enumerate(races)
        ]
        ax.legend(handles=legend_patches, loc="upper right", title="Race/Ethnicity")

    else:
        # Simple pyramid by sex
        male_pop = year_data[year_data["sex"] == "Male"].groupby("age_group")["population"].sum()
        female_pop = (
            year_data[year_data["sex"] == "Female"].groupby("age_group")["population"].sum()
        )

        age_groups = sorted(set(male_pop.index) | set(female_pop.index))

        male_pop = male_pop.reindex(age_groups, fill_value=0)
        female_pop = female_pop.reindex(age_groups, fill_value=0)

        y_pos = np.arange(len(age_groups))
        bar_height = 0.8

        # Males (left, negative)
        ax.barh(y_pos, -np.asarray(male_pop.values), bar_height, label="Male", color="#3498db")

        # Females (right, positive)
        ax.barh(y_pos, female_pop.values, bar_height, label="Female", color="#e74c3c")

        ax.legend(loc="upper right")

    # Age group labels
    if age_group_size > 1:
        age_labels = [f"{age}-{age + age_group_size - 1}" for age in age_groups]
    else:
        age_labels = [str(age) for age in age_groups]

    ax.set_yticks(np.arange(len(age_groups)))
    ax.set_yticklabels(age_labels)
    ax.set_ylabel("Age Group")

    # X-axis formatting
    max_val = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-max_val, max_val)

    # Format x-axis labels as absolute values
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f"{abs(int(x)):,}" for x in x_ticks])
    ax.set_xlabel("Population")

    # Center line
    ax.axvline(0, color="black", linewidth=0.8)

    # Title
    if title is None:
        total_pop = year_data["population"].sum()
        title = f"Population Pyramid - {year}\nTotal Population: {total_pop:,.0f}"

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Population pyramid saved: {output_path}")

    return output_path


def plot_population_trends(
    projection_df: pd.DataFrame,
    output_path: str | Path,
    by: Literal["total", "sex", "age_group", "race"] = "total",
    age_groups: dict[str, tuple[int, int]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> Path:
    """
    Create line chart of population trends over time.

    Args:
        projection_df: Projection DataFrame
        output_path: Path to save figure
        by: Grouping level ('total', 'sex', 'age_group', 'race')
        age_groups: Optional dict of age group definitions (e.g., {'Youth': (0, 17), 'Working': (18, 64)})
        title: Optional custom title
        figsize: Figure size (width, height)
        dpi: Resolution for raster formats
        style: Matplotlib style

    Returns:
        Path to saved figure

    Example:
        >>> plot_population_trends(
        ...     projection_df=results,
        ...     output_path='output/trends.png',
        ...     by='age_group',
        ...     age_groups={'Youth': (0, 17), 'Working': (18, 64), 'Elderly': (65, 90)}
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualizations")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating population trends chart: {output_path}")

    # Set style
    if SEABORN_AVAILABLE and style.startswith("seaborn"):
        sns.set_style("darkgrid")
    else:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if by == "total":
        # Total population over time
        totals = projection_df.groupby("year")["population"].sum()
        ax.plot(
            totals.index,
            totals.values,
            marker="o",
            linewidth=2,
            markersize=4,
            label="Total Population",
        )

    elif by == "sex":
        # By sex
        for sex in ["Male", "Female"]:
            sex_data = (
                projection_df[projection_df["sex"] == sex].groupby("year")["population"].sum()
            )
            ax.plot(
                sex_data.index, sex_data.values, marker="o", linewidth=2, markersize=4, label=sex
            )

    elif by == "age_group":
        # By age groups
        if age_groups is None:
            age_groups = {
                "Youth (0-17)": (0, 17),
                "Working Age (18-64)": (18, 64),
                "Elderly (65+)": (65, 150),
            }

        for group_name, (min_age, max_age) in age_groups.items():
            group_data = (
                projection_df[(projection_df["age"] >= min_age) & (projection_df["age"] <= max_age)]
                .groupby("year")["population"]
                .sum()
            )

            ax.plot(
                group_data.index,
                group_data.values,
                marker="o",
                linewidth=2,
                markersize=4,
                label=group_name,
            )

    elif by == "race":
        # By race
        races = sorted(projection_df["race"].unique())
        for race in races:
            race_data = (
                projection_df[projection_df["race"] == race].groupby("year")["population"].sum()
            )
            ax.plot(
                race_data.index, race_data.values, marker="o", linewidth=2, markersize=4, label=race
            )

    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)

    if title is None:
        title = f"Population Trends by {by.replace('_', ' ').title()}"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Legend
    ax.legend(loc="best", frameon=True)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Population trends chart saved: {output_path}")

    return output_path


def plot_growth_rates(
    projection_df: pd.DataFrame,
    output_path: str | Path,
    period: Literal["annual", "5year", "10year"] = "annual",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> Path:
    """
    Create chart showing population growth rates over time.

    Args:
        projection_df: Projection DataFrame
        output_path: Path to save figure
        period: Growth rate period ('annual', '5year', '10year')
        title: Optional custom title
        figsize: Figure size
        dpi: Resolution
        style: Matplotlib style

    Returns:
        Path to saved figure

    Example:
        >>> plot_growth_rates(
        ...     projection_df=results,
        ...     output_path='output/growth_rates.png',
        ...     period='5year'
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualizations")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating growth rates chart: {output_path}")

    # Calculate population by year
    pop_by_year = projection_df.groupby("year")["population"].sum().sort_index()
    years = pop_by_year.index.values
    populations = pop_by_year.values

    # Calculate growth rates
    growth_years = []
    growth_rates = []

    if period == "annual":
        # Year-over-year growth
        for i in range(1, len(years)):
            prev_pop = populations[i - 1]
            curr_pop = populations[i]
            growth_rate = ((curr_pop / prev_pop - 1) * 100) if prev_pop > 0 else 0

            growth_years.append(years[i])
            growth_rates.append(growth_rate)

    elif period == "5year":
        # 5-year period growth
        for i in range(5, len(years)):
            prev_pop = populations[i - 5]
            curr_pop = populations[i]
            # Annualized growth rate
            growth_rate = (((curr_pop / prev_pop) ** (1 / 5) - 1) * 100) if prev_pop > 0 else 0

            growth_years.append(years[i])
            growth_rates.append(growth_rate)

    elif period == "10year":
        # 10-year period growth
        for i in range(10, len(years)):
            prev_pop = populations[i - 10]
            curr_pop = populations[i]
            # Annualized growth rate
            growth_rate = (((curr_pop / prev_pop) ** (1 / 10) - 1) * 100) if prev_pop > 0 else 0

            growth_years.append(years[i])
            growth_rates.append(growth_rate)

    # Set style
    if SEABORN_AVAILABLE and style.startswith("seaborn"):
        sns.set_style("darkgrid")
    else:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Bar chart with color coding (positive = green, negative = red)
    colors = ["#27ae60" if rate >= 0 else "#e74c3c" for rate in growth_rates]
    ax.bar(growth_years, growth_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Zero line
    ax.axhline(0, color="black", linewidth=1, linestyle="-")

    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Growth Rate (%)", fontsize=12)

    if title is None:
        period_label = {"annual": "Annual", "5year": "5-Year", "10year": "10-Year"}[period]
        title = f"{period_label} Population Growth Rates"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Growth rates chart saved: {output_path}")

    return output_path


def plot_component_analysis(
    births_df: pd.DataFrame | None,
    deaths_df: pd.DataFrame | None,
    migration_df: pd.DataFrame | None,
    output_path: str | Path,
    chart_type: Literal["stacked_area", "grouped_bar"] = "stacked_area",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> Path:
    """
    Visualize demographic components (births, deaths, migration) over time.

    Args:
        births_df: DataFrame with births by year
        deaths_df: DataFrame with deaths by year
        migration_df: DataFrame with net migration by year
        output_path: Path to save figure
        chart_type: Type of chart ('stacked_area' or 'grouped_bar')
        title: Optional custom title
        figsize: Figure size
        dpi: Resolution
        style: Matplotlib style

    Returns:
        Path to saved figure

    Note:
        Component DataFrames should have 'year' and 'count' or 'population' columns.

    Example:
        >>> plot_component_analysis(
        ...     births_df=births,
        ...     deaths_df=deaths,
        ...     migration_df=migration,
        ...     output_path='output/components.png'
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualizations")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating component analysis chart: {output_path}")

    # This is a placeholder - full implementation would require component data
    # tracking during projection runs

    logger.warning("plot_component_analysis requires component tracking during projection")

    # Set style
    if SEABORN_AVAILABLE and style.startswith("seaborn"):
        sns.set_style("darkgrid")
    else:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Placeholder chart
    ax.text(
        0.5,
        0.5,
        "Component analysis requires component data\ntracking during projection runs",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax.transAxes,
    )

    if title is None:
        title = "Demographic Components Analysis"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Component analysis placeholder saved: {output_path}")

    return output_path


def plot_scenario_comparison(
    scenario_projections: dict[str, pd.DataFrame],
    output_path: str | Path,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> Path:
    """
    Compare multiple projection scenarios on one chart.

    Args:
        scenario_projections: Dict mapping scenario name -> projection DataFrame
        output_path: Path to save figure
        title: Optional custom title
        figsize: Figure size
        dpi: Resolution
        style: Matplotlib style

    Returns:
        Path to saved figure

    Example:
        >>> scenarios = {
        ...     'Baseline': baseline_df,
        ...     'High Growth': high_growth_df,
        ...     'Low Growth': low_growth_df
        ... }
        >>> plot_scenario_comparison(
        ...     scenario_projections=scenarios,
        ...     output_path='output/scenario_comparison.png'
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualizations")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating scenario comparison chart: {output_path}")

    # Set style
    if SEABORN_AVAILABLE and style.startswith("seaborn"):
        sns.set_style("darkgrid")
    else:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color palette
    if SEABORN_AVAILABLE:
        colors = sns.color_palette("Set1", len(scenario_projections))
    else:
        colors = plt.colormaps.get_cmap("Set1")(np.linspace(0, 1, len(scenario_projections)))

    # Plot each scenario
    for i, (scenario_name, projection_df) in enumerate(scenario_projections.items()):
        pop_by_year = projection_df.groupby("year")["population"].sum().sort_index()

        ax.plot(
            pop_by_year.index,
            pop_by_year.values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=scenario_name,
            color=colors[i],
        )

    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)

    if title is None:
        title = "Scenario Comparison"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Legend
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Scenario comparison chart saved: {output_path}")

    return output_path


def save_all_visualizations(
    projection_df: pd.DataFrame,
    output_dir: str | Path,
    base_filename: str,
    years_for_pyramids: list[int] | None = None,
    image_format: str = "png",
    dpi: int = 300,
    style: str = "seaborn-v0_8-darkgrid",
) -> dict[str, Path]:
    """
    Generate all standard visualizations.

    Creates:
    - Population pyramids (base year, mid-point, final year)
    - Total population trends
    - Population trends by age group
    - Population trends by sex
    - Population trends by race
    - Growth rates chart

    Args:
        projection_df: Projection DataFrame
        output_dir: Output directory for charts
        base_filename: Base filename prefix for all charts
        years_for_pyramids: Specific years for pyramids (default: first, middle, last)
        image_format: Image format ('png', 'svg', 'pdf')
        dpi: Resolution for raster formats
        style: Matplotlib style

    Returns:
        Dictionary mapping chart type -> output path

    Example:
        >>> paths = save_all_visualizations(
        ...     projection_df=results,
        ...     output_dir='output/charts',
        ...     base_filename='nd_state_2025_2045'
        ... )
        >>> paths['pyramid_base']
        PosixPath('output/charts/nd_state_2025_2045_pyramid_2025.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualizations")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating all standard visualizations in {output_dir}")

    output_paths = {}

    years = sorted(projection_df["year"].unique())

    # Determine years for pyramids
    if years_for_pyramids is None:
        years_for_pyramids = [
            years[0],  # Base year
            years[len(years) // 2],  # Mid-point
            years[-1],  # Final year
        ]

    # 1. Population Pyramids
    for year in years_for_pyramids:
        logger.debug(f"Creating pyramid for year {year}")
        path = plot_population_pyramid(
            projection_df,
            year=year,
            output_path=output_dir / f"{base_filename}_pyramid_{year}.{image_format}",
            dpi=dpi,
            style=style,
        )
        key = f"pyramid_{year}"
        output_paths[key] = path

    # 2. Total population trends
    logger.debug("Creating total population trends")
    path = plot_population_trends(
        projection_df,
        output_path=output_dir / f"{base_filename}_trends_total.{image_format}",
        by="total",
        dpi=dpi,
        style=style,
    )
    output_paths["trends_total"] = path

    # 3. Trends by age group
    logger.debug("Creating age group trends")
    path = plot_population_trends(
        projection_df,
        output_path=output_dir / f"{base_filename}_trends_age_groups.{image_format}",
        by="age_group",
        dpi=dpi,
        style=style,
    )
    output_paths["trends_age_groups"] = path

    # 4. Trends by sex
    logger.debug("Creating sex trends")
    path = plot_population_trends(
        projection_df,
        output_path=output_dir / f"{base_filename}_trends_sex.{image_format}",
        by="sex",
        dpi=dpi,
        style=style,
    )
    output_paths["trends_sex"] = path

    # 5. Trends by race
    logger.debug("Creating race trends")
    path = plot_population_trends(
        projection_df,
        output_path=output_dir / f"{base_filename}_trends_race.{image_format}",
        by="race",
        dpi=dpi,
        style=style,
    )
    output_paths["trends_race"] = path

    # 6. Growth rates
    logger.debug("Creating growth rates chart")
    path = plot_growth_rates(
        projection_df,
        output_path=output_dir / f"{base_filename}_growth_rates.{image_format}",
        period="annual",
        dpi=dpi,
        style=style,
    )
    output_paths["growth_rates"] = path

    logger.info(f"Generated {len(output_paths)} visualizations")

    return output_paths
