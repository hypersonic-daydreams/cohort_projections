"""
Vintage Dummy Variable Creation
===============================

Creates indicator variables for vintage/regime periods to capture
level shifts at vintage transition boundaries.

Per ADR-020 Option C, vintage dummies control for potential
methodology-related level shifts without correcting the data.
"""

import pandas as pd


def create_vintage_dummies(
    df: pd.DataFrame,
    year_col: str = "year",
    prefix: str = "vintage",
) -> pd.DataFrame:
    """
    Create vintage/regime dummy variables.

    Reference category is Vintage 2009 (2000s), so coefficients for
    vintage_2010s and vintage_2020s represent level shifts relative
    to the 2000s baseline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a year column
    year_col : str
        Name of the year column
    prefix : str
        Prefix for dummy variable names

    Returns
    -------
    pd.DataFrame
        DataFrame with added vintage dummy columns
    """
    df = df.copy()

    # Vintage indicator dummies (2000s is reference category)
    df[f"{prefix}_2010s"] = ((df[year_col] >= 2010) & (df[year_col] <= 2019)).astype(
        int
    )
    df[f"{prefix}_2020s"] = (df[year_col] >= 2020).astype(int)

    # Also create a categorical vintage variable
    df[f"{prefix}_code"] = df[year_col].apply(_get_vintage_code)

    return df


def _get_vintage_code(year: int) -> int:
    """Return vintage code (2009, 2020, or 2024) for a year."""
    if year < 2010:
        return 2009
    elif year < 2020:
        return 2020
    else:
        return 2024


def create_regime_dummies(
    df: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Create mutually exclusive regime indicator dummies.

    Unlike vintage dummies which omit one category, this creates
    indicators for all three regimes (for use in interactions).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a year column
    year_col : str
        Name of the year column

    Returns
    -------
    pd.DataFrame
        DataFrame with added regime dummy columns
    """
    df = df.copy()

    df["regime_2000s"] = (df[year_col] < 2010).astype(int)
    df["regime_2010s"] = ((df[year_col] >= 2010) & (df[year_col] < 2020)).astype(int)
    df["regime_2020s"] = (df[year_col] >= 2020).astype(int)

    return df


def get_vintage_counts(df: pd.DataFrame, year_col: str = "year") -> dict:
    """
    Count observations by vintage period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a year column
    year_col : str
        Name of the year column

    Returns
    -------
    dict
        Counts per vintage: {"2009": n1, "2020": n2, "2024": n3}
    """
    df = df.copy()
    df["_vintage"] = df[year_col].apply(_get_vintage_code)

    counts = df.groupby("_vintage").size().to_dict()

    return {
        "vintage_2009": counts.get(2009, 0),
        "vintage_2020": counts.get(2020, 0),
        "vintage_2024": counts.get(2024, 0),
        "total": len(df),
    }
