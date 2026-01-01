"""
Regime Definitions for Vintage-Based Analysis
==============================================

Defines vintage boundaries and regime labels for Census Bureau PEP data.
Per ADR-020, three vintages are used:
- Vintage 2009: 2000-2009 (pre-2010 Census methodology)
- Vintage 2020: 2010-2019 (post-2010 Census methodology)
- Vintage 2024: 2020-2024 (post-2020 Census methodology)
"""

from typing import Literal

# Vintage year ranges
VINTAGE_2009_YEARS = range(2000, 2010)  # 2000-2009 inclusive
VINTAGE_2020_YEARS = range(2010, 2020)  # 2010-2019 inclusive
VINTAGE_2024_YEARS = range(2020, 2025)  # 2020-2024 inclusive

# Vintage boundaries (transition years)
VINTAGE_BOUNDARIES = {
    "2009_to_2020": 2010,  # First year of Vintage 2020
    "2020_to_2024": 2020,  # First year of Vintage 2024
}

# Vintage labels for display
VINTAGE_LABELS = {
    2009: "Vintage 2009 (2000-2009)",
    2020: "Vintage 2020 (2010-2019)",
    2024: "Vintage 2024 (2020-2024)",
}

# Regime descriptors
REGIME_NAMES = {
    "2000s": "Pre-2010 Census (Vintage 2009)",
    "2010s": "Post-2010 Census (Vintage 2020)",
    "2020s": "Post-2020 Census (Vintage 2024)",
}

VintageType = Literal[2009, 2020, 2024]
RegimeType = Literal["2000s", "2010s", "2020s"]


def get_vintage_for_year(year: int) -> VintageType:
    """Return the vintage code for a given year."""
    if year in VINTAGE_2009_YEARS:
        return 2009
    elif year in VINTAGE_2020_YEARS:
        return 2020
    elif year in VINTAGE_2024_YEARS:
        return 2024
    else:
        raise ValueError(f"Year {year} is outside the supported range (2000-2024)")


def get_regime_for_year(year: int) -> RegimeType:
    """Return the regime label for a given year."""
    if year in VINTAGE_2009_YEARS:
        return "2000s"
    elif year in VINTAGE_2020_YEARS:
        return "2010s"
    elif year in VINTAGE_2024_YEARS:
        return "2020s"
    else:
        raise ValueError(f"Year {year} is outside the supported range (2000-2024)")


def is_transition_year(year: int) -> bool:
    """Check if year is at a vintage transition boundary."""
    return year in VINTAGE_BOUNDARIES.values()


# Known structural events (for distinguishing from methodology artifacts)
STRUCTURAL_EVENTS = {
    2008: "Financial Crisis",
    2011: "Bakken Boom begins",
    2017: "Travel Ban executive order",
    2020: "COVID-19 pandemic",
}

# COVID-specific constants
COVID_YEAR = 2020
COVID_RECOVERY_YEARS = [2021, 2022, 2023, 2024]
