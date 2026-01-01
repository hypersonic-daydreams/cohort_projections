"""
Data Loading for Multi-State Placebo Analysis
==============================================

Loads Census Bureau PEP state-level population components data
and adds vintage labels for regime-aware analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Default data locations
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # cohort_projections/
REVISION_OUTPUTS = (
    PROJECT_ROOT
    / "sdc_2024_replication"
    / "scripts"
    / "statistical_analysis"
    / "journal_article"
    / "revision_outputs"
)


def load_state_panel(
    filepath: Optional[Path] = None,
    exclude_territories: bool = True,
) -> pd.DataFrame:
    """
    Load the 50-state population components panel data.

    Parameters
    ----------
    filepath : Path, optional
        Custom path to panel data. If None, uses default location.
    exclude_territories : bool
        If True, excludes DC, Puerto Rico, and other territories.
        Default is True to focus on 50 states.

    Returns
    -------
    pd.DataFrame
        State-level panel with columns:
        - state: State name
        - state_fips: FIPS code
        - year: Calendar year
        - population: Total population
        - intl_migration: International migration count
        - (other columns as available)
    """
    if filepath is None:
        filepath = REVISION_OUTPUTS / "02_G04_causal" / "07_data_panel.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Panel data not found: {filepath}")

    df = pd.read_csv(filepath)

    # Exclude territories if requested
    if exclude_territories:
        # Keep only 50 states (FIPS 1-56, excluding DC=11)
        # Note: FIPS codes skip some numbers
        territories = ["District of Columbia", "Puerto Rico"]
        df = df[~df["state"].isin(territories)]

    # Sort by state and year
    df = df.sort_values(["state", "year"]).reset_index(drop=True)

    return df


def add_vintage_labels(
    df: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Add vintage labels based on year.

    Vintage definitions per ADR-020:
    - Vintage 2009: 2000-2009 (not available in this data)
    - Vintage 2020: 2010-2019
    - Vintage 2024: 2020-2024

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a year column
    year_col : str
        Name of the year column

    Returns
    -------
    pd.DataFrame
        DataFrame with added vintage columns
    """
    df = df.copy()

    def get_vintage(year: int) -> int:
        if year < 2010:
            return 2009
        elif year < 2020:
            return 2020
        else:
            return 2024

    def get_vintage_period(year: int) -> str:
        if year < 2010:
            return "Vintage 2009 (2000-2009)"
        elif year < 2020:
            return "Vintage 2020 (2010-2019)"
        else:
            return "Vintage 2024 (2020-2024)"

    df["vintage"] = df[year_col].apply(get_vintage)
    df["vintage_period"] = df[year_col].apply(get_vintage_period)

    return df


def get_unique_states(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique state names."""
    return sorted(df["state"].unique().tolist())


def filter_to_states(
    df: pd.DataFrame,
    states: list[str],
) -> pd.DataFrame:
    """Filter DataFrame to specific states."""
    return df[df["state"].isin(states)].copy()


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics by state.

    Returns
    -------
    pd.DataFrame
        One row per state with summary statistics
    """
    summary = (
        df.groupby("state")
        .agg(
            n_years=("year", "count"),
            min_year=("year", "min"),
            max_year=("year", "max"),
            mean_intl_migration=("intl_migration", "mean"),
            std_intl_migration=("intl_migration", "std"),
            mean_population=("population", "mean"),
        )
        .reset_index()
    )

    # Add per-capita migration rate (per 1000)
    summary["mean_migration_rate"] = (
        summary["mean_intl_migration"] / summary["mean_population"] * 1000
    )

    return summary
