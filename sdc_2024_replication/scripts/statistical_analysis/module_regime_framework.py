#!/usr/bin/env python3
"""
Module: Policy Regime Framework (ADR-021 Recommendation #4)
============================================================

Formalizes the R_t regime variable that captures distinct policy eras for
North Dakota immigration forecasting, based on structural break analysis
from Module 2.1.2 and validated findings from Phase A exploratory analysis.

Three regimes are defined:
- **Expansion (2010-2016)**: USRAP growth, strong pull factors, 92% refugee share
- **Restriction (2017-2020)**: Ceiling cuts, Travel Ban, processing friction
- **Volatility (2021-2024)**: Parole surge, administrative volatility, 6.7% refugee share

Key Features:
- Explicit R_t variable with regime boundaries
- Policy event mapping with primary source citations
- Regime-specific parameters (baseline growth, volatility, composition)
- Utility functions for regime lookup and parameter retrieval

Usage:
    from module_regime_framework import (
        get_regime,
        get_regime_params,
        get_policy_events_for_regime,
        PolicyRegime,
        REGIME_BOUNDARIES,
    )

    # Get regime for a specific year
    regime = get_regime(2023)  # Returns PolicyRegime.VOLATILITY

    # Get regime parameters
    params = get_regime_params(PolicyRegime.EXPANSION)

References:
- ADR-021: Immigration Status Durability and Policy-Regime Methodology
- Module 2.1.2: Structural Break Detection (Chow tests at 2017, 2020, 2021)
- Phase A Agent 1 Results: Estimand composition analysis
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal


class PolicyRegime(Enum):
    """
    Enumeration of policy regimes for ND international migration.

    Based on structural break analysis (Module 2.1.2) and policy event mapping.
    The regime variable R_t takes one of three values corresponding to
    distinct policy eras with different migration dynamics.
    """

    EXPANSION = "expansion"
    RESTRICTION = "restriction"
    VOLATILITY = "volatility"


# Type alias for regime names
RegimeName = Literal["expansion", "restriction", "volatility"]

# =============================================================================
# REGIME BOUNDARIES
# =============================================================================

REGIME_BOUNDARIES: dict[PolicyRegime, tuple[int, int]] = {
    PolicyRegime.EXPANSION: (2010, 2016),
    PolicyRegime.RESTRICTION: (2017, 2020),
    PolicyRegime.VOLATILITY: (2021, 2024),
}
"""Year ranges (inclusive) for each policy regime."""

REGIME_YEARS: dict[PolicyRegime, range] = {
    PolicyRegime.EXPANSION: range(2010, 2017),  # 2010-2016 inclusive
    PolicyRegime.RESTRICTION: range(2017, 2021),  # 2017-2020 inclusive
    PolicyRegime.VOLATILITY: range(2021, 2025),  # 2021-2024 inclusive
}
"""Python range objects for each regime (for iteration/membership tests)."""


# =============================================================================
# POLICY EVENTS
# =============================================================================


@dataclass(frozen=True)
class PolicyEvent:
    """
    A significant policy event affecting immigration flows.

    Attributes:
        event_id: Unique identifier for the event
        name: Short descriptive name
        date: Date of policy implementation (or announcement)
        regime: Which regime this event belongs to
        mechanism: How the policy affects migration (supply, demand, friction)
        expected_direction: Expected effect on ND international migration
        primary_source: Citation to authoritative source (Federal Register, etc.)
        description: Longer description of the event
    """

    event_id: str
    name: str
    date: date
    regime: PolicyRegime
    mechanism: Literal["supply", "demand", "friction", "capacity", "composition"]
    expected_direction: Literal["positive", "negative", "neutral", "uncertain"]
    primary_source: str
    description: str


# Comprehensive policy event mapping with primary sources
POLICY_EVENTS: dict[str, PolicyEvent] = {
    # ==========================================================================
    # EXPANSION REGIME (2010-2016)
    # ==========================================================================
    "PD_FY2011": PolicyEvent(
        event_id="PD_FY2011",
        name="FY2011 Refugee Ceiling (80,000)",
        date=date(2010, 9, 30),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2010-14, 75 FR 60773 (Oct 1, 2010)",
        description="President Obama sets FY2011 refugee ceiling at 80,000, maintaining USRAP capacity.",
    ),
    "PD_FY2012": PolicyEvent(
        event_id="PD_FY2012",
        name="FY2012 Refugee Ceiling (76,000)",
        date=date(2011, 9, 30),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2011-17, 76 FR 62765 (Oct 11, 2011)",
        description="FY2012 refugee ceiling set at 76,000 with regional allocations.",
    ),
    "PD_FY2013": PolicyEvent(
        event_id="PD_FY2013",
        name="FY2013 Refugee Ceiling (70,000)",
        date=date(2012, 9, 28),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2012-16, 77 FR 60527 (Oct 4, 2012)",
        description="FY2013 refugee ceiling set at 70,000.",
    ),
    "PD_FY2014": PolicyEvent(
        event_id="PD_FY2014",
        name="FY2014 Refugee Ceiling (70,000)",
        date=date(2013, 9, 30),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2013-15, 78 FR 62379 (Oct 11, 2013)",
        description="FY2014 refugee ceiling maintained at 70,000.",
    ),
    "PD_FY2015": PolicyEvent(
        event_id="PD_FY2015",
        name="FY2015 Refugee Ceiling (70,000)",
        date=date(2014, 9, 30),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2014-14, 79 FR 60089 (Oct 6, 2014)",
        description="FY2015 refugee ceiling maintained at 70,000.",
    ),
    "PD_FY2016": PolicyEvent(
        event_id="PD_FY2016",
        name="FY2016 Refugee Ceiling (85,000)",
        date=date(2015, 9, 29),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2015-14, 80 FR 61245 (Oct 9, 2015)",
        description="FY2016 ceiling increased to 85,000 in response to Syrian crisis.",
    ),
    "PD_FY2017": PolicyEvent(
        event_id="PD_FY2017",
        name="FY2017 Refugee Ceiling (110,000)",
        date=date(2016, 9, 28),
        regime=PolicyRegime.EXPANSION,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2016-13, 81 FR 70315 (Oct 12, 2016)",
        description="FY2017 ceiling set at historic high of 110,000 by Obama administration.",
    ),
    # ==========================================================================
    # RESTRICTION REGIME (2017-2020)
    # ==========================================================================
    "EO_13769": PolicyEvent(
        event_id="EO_13769",
        name="Travel Ban Executive Order 13769",
        date=date(2017, 1, 27),
        regime=PolicyRegime.RESTRICTION,
        mechanism="friction",
        expected_direction="negative",
        primary_source="Executive Order 13769, 82 FR 8977 (Jan 27, 2017)",
        description=(
            "Initial Travel Ban: 90-day suspension of entry from 7 countries, "
            "120-day suspension of USRAP, 50,000 cap on FY2017 admissions. "
            "Blocked by courts Feb 3, 2017."
        ),
    ),
    "EO_13780": PolicyEvent(
        event_id="EO_13780",
        name="Revised Travel Ban Executive Order 13780",
        date=date(2017, 3, 6),
        regime=PolicyRegime.RESTRICTION,
        mechanism="friction",
        expected_direction="negative",
        primary_source="Executive Order 13780, 82 FR 13209 (Mar 6, 2017)",
        description=(
            "Revised Travel Ban: 90-day suspension from 6 countries (Iraq removed), "
            "120-day USRAP suspension. Partially blocked by courts, upheld by SCOTUS "
            "June 2018 (Trump v. Hawaii)."
        ),
    ),
    "PD_FY2018": PolicyEvent(
        event_id="PD_FY2018",
        name="FY2018 Refugee Ceiling (45,000)",
        date=date(2017, 9, 29),
        regime=PolicyRegime.RESTRICTION,
        mechanism="supply",
        expected_direction="negative",
        primary_source="Presidential Determination No. 2017-13, 82 FR 46509 (Oct 5, 2017)",
        description="Historic ceiling reduction from 110,000 to 45,000 (59% cut).",
    ),
    "PD_FY2019": PolicyEvent(
        event_id="PD_FY2019",
        name="FY2019 Refugee Ceiling (30,000)",
        date=date(2018, 10, 4),
        regime=PolicyRegime.RESTRICTION,
        mechanism="supply",
        expected_direction="negative",
        primary_source="Presidential Determination No. 2018-11, 83 FR 51299 (Oct 11, 2018)",
        description="Further ceiling reduction to 30,000 (33% cut from FY2018).",
    ),
    "EO_13888": PolicyEvent(
        event_id="EO_13888",
        name="State/Local Consent Requirement",
        date=date(2019, 9, 26),
        regime=PolicyRegime.RESTRICTION,
        mechanism="capacity",
        expected_direction="negative",
        primary_source="Executive Order 13888, 84 FR 52355 (Oct 1, 2019)",
        description=(
            "Required state and local written consent for refugee resettlement. "
            "North Dakota Governor Burgum provided consent Dec 2019. "
            "Order vacated by courts Jan 2020."
        ),
    ),
    "PD_FY2020": PolicyEvent(
        event_id="PD_FY2020",
        name="FY2020 Refugee Ceiling (18,000)",
        date=date(2019, 11, 1),
        regime=PolicyRegime.RESTRICTION,
        mechanism="supply",
        expected_direction="negative",
        primary_source="Presidential Determination No. 2019-15, 84 FR 60223 (Nov 7, 2019)",
        description="Lowest ceiling in USRAP history (18,000).",
    ),
    "COVID_TRAVEL": PolicyEvent(
        event_id="COVID_TRAVEL",
        name="COVID-19 Travel Restrictions",
        date=date(2020, 3, 20),
        regime=PolicyRegime.RESTRICTION,
        mechanism="friction",
        expected_direction="negative",
        primary_source="Proclamation 9984, 85 FR 16717 (Mar 20, 2020); CDC Order Mar 20, 2020",
        description=(
            "COVID-19 pandemic leads to embassy closures, suspended USRAP interviews, "
            "and global travel restrictions. Near-complete halt to refugee arrivals."
        ),
    ),
    "PD_FY2021": PolicyEvent(
        event_id="PD_FY2021",
        name="FY2021 Refugee Ceiling (15,000)",
        date=date(2020, 10, 27),
        regime=PolicyRegime.RESTRICTION,
        mechanism="supply",
        expected_direction="negative",
        primary_source="Presidential Determination No. 2020-12, 85 FR 68705 (Oct 29, 2020)",
        description="FY2021 ceiling set at record low 15,000 by outgoing Trump administration.",
    ),
    # ==========================================================================
    # VOLATILITY REGIME (2021-2024)
    # ==========================================================================
    "LSSND_CLOSURE": PolicyEvent(
        event_id="LSSND_CLOSURE",
        name="LSSND Closure (ND Capacity Shock)",
        date=date(2021, 1, 1),
        regime=PolicyRegime.VOLATILITY,
        mechanism="capacity",
        expected_direction="negative",
        primary_source="Lutheran Social Services of North Dakota closure announcement, Dec 2020",
        description=(
            "Lutheran Social Services of North Dakota closes after 70+ years, ending primary "
            "refugee resettlement capacity. LIRS/Global Refuge assumes ND resettlement "
            "with reduced local infrastructure."
        ),
    ),
    "PD_FY2021_REVISION": PolicyEvent(
        event_id="PD_FY2021_REVISION",
        name="FY2021 Ceiling Raised (62,500)",
        date=date(2021, 5, 3),
        regime=PolicyRegime.VOLATILITY,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2021-05, 86 FR 24475 (May 7, 2021)",
        description=(
            "Biden administration raises FY2021 ceiling from 15,000 to 62,500 and removes "
            "regional allocations. Processing backlog limits actual admissions to ~11,000."
        ),
    ),
    "OAW": PolicyEvent(
        event_id="OAW",
        name="Operation Allies Welcome (Afghan Parole)",
        date=date(2021, 8, 17),
        regime=PolicyRegime.VOLATILITY,
        mechanism="composition",
        expected_direction="positive",
        primary_source="DHS Operation Allies Welcome, Aug 2021; Afghan Adjustment Act pending",
        description=(
            "Emergency Afghan evacuation and parole program following Taliban takeover. "
            "~76,000 Afghans paroled to US with 2-year humanitarian parole status (no automatic "
            "path to permanence). ND estimated to receive 50-100 through FY2023."
        ),
    ),
    "PD_FY2022": PolicyEvent(
        event_id="PD_FY2022",
        name="FY2022 Refugee Ceiling (125,000)",
        date=date(2021, 9, 20),
        regime=PolicyRegime.VOLATILITY,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2021-10, 86 FR 53057 (Sep 24, 2021)",
        description=(
            "Biden sets aspirational ceiling of 125,000 for FY2022. Processing capacity "
            "limitations result in ~25,000 actual admissions."
        ),
    ),
    "U4U": PolicyEvent(
        event_id="U4U",
        name="Uniting for Ukraine (U4U Parole)",
        date=date(2022, 4, 21),
        regime=PolicyRegime.VOLATILITY,
        mechanism="composition",
        expected_direction="positive",
        primary_source="DHS Uniting for Ukraine, 87 FR 24548 (Apr 26, 2022)",
        description=(
            "Humanitarian parole program for Ukrainian nationals following Russian invasion. "
            "Requires US sponsor. ~200,000+ paroled nationally by end 2024. "
            "ND estimated 600-800 arrivals. 2-year parole status with uncertain extension."
        ),
    ),
    "PD_FY2023": PolicyEvent(
        event_id="PD_FY2023",
        name="FY2023 Refugee Ceiling (125,000)",
        date=date(2022, 9, 27),
        regime=PolicyRegime.VOLATILITY,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2022-11, 87 FR 59489 (Sep 30, 2022)",
        description="FY2023 ceiling maintained at 125,000; ~60,000 actual admissions.",
    ),
    "CHNV": PolicyEvent(
        event_id="CHNV",
        name="CHNV Parole Programs",
        date=date(2023, 1, 5),
        regime=PolicyRegime.VOLATILITY,
        mechanism="composition",
        expected_direction="positive",
        primary_source="DHS CHNV Announcement, 88 FR 1266 (Jan 9, 2023)",
        description=(
            "Parole programs for Cuban, Haitian, Nicaraguan, and Venezuelan nationals. "
            "30,000/month cap. Minimal impact on ND due to resettlement patterns."
        ),
    ),
    "WELCOME_CORPS": PolicyEvent(
        event_id="WELCOME_CORPS",
        name="Welcome Corps (Private Sponsorship)",
        date=date(2023, 1, 19),
        regime=PolicyRegime.VOLATILITY,
        mechanism="capacity",
        expected_direction="positive",
        primary_source="State Department Welcome Corps Launch, Jan 2023",
        description=(
            "Private refugee sponsorship program launch (5-person groups can sponsor refugee "
            "families). Adds capacity outside traditional resettlement agencies. "
            "Potential future ND impact unclear."
        ),
    ),
    "PD_FY2024": PolicyEvent(
        event_id="PD_FY2024",
        name="FY2024 Refugee Ceiling (125,000)",
        date=date(2023, 9, 29),
        regime=PolicyRegime.VOLATILITY,
        mechanism="supply",
        expected_direction="positive",
        primary_source="Presidential Determination No. 2023-11, 88 FR 67893 (Oct 3, 2023)",
        description="FY2024 ceiling maintained at 125,000; ~100,000 actual admissions achieved.",
    ),
    "PD_FY2025": PolicyEvent(
        event_id="PD_FY2025",
        name="FY2025 Refugee Ceiling (125,000)",
        date=date(2024, 9, 30),
        regime=PolicyRegime.VOLATILITY,
        mechanism="supply",
        expected_direction="uncertain",
        primary_source="Presidential Determination No. 2024-10, 89 FR 79437 (Oct 3, 2024)",
        description=(
            "FY2025 ceiling set at 125,000 by Biden administration. Incoming Trump "
            "administration expected to reduce ceiling significantly."
        ),
    ),
}


# =============================================================================
# REGIME PARAMETERS
# =============================================================================


@dataclass(frozen=True)
class RegimeParameters:
    """
    Statistical and compositional parameters for a policy regime.

    Derived from Phase A exploratory analysis (Agent 1 results) and
    Module 2.1.2 structural break detection.

    Attributes:
        regime: The policy regime
        start_year: First year of regime (inclusive)
        end_year: Last year of regime (inclusive)
        mean_total_migration: Mean annual ND international migration
        std_total_migration: Standard deviation of annual migration
        mean_refugee_arrivals: Mean annual ND refugee arrivals
        refugee_share_pct: Mean percent of total migration that is refugee
        non_refugee_share_pct: Mean percent that is non-refugee (parole, LPR, other)
        trend_coefficient: Estimated annual trend within regime
        volatility_ratio: Coefficient of variation (std/mean)
        primary_status_composition: Dominant legal status category
        status_durability: Expected retention rate for dominant category
    """

    regime: PolicyRegime
    start_year: int
    end_year: int
    mean_total_migration: float
    std_total_migration: float
    mean_refugee_arrivals: float
    refugee_share_pct: float
    non_refugee_share_pct: float
    trend_coefficient: float
    volatility_ratio: float
    primary_status_composition: str
    status_durability: str


# Regime parameters derived from Agent 1 analysis and Module 2.1.2
REGIME_PARAMETERS: dict[PolicyRegime, RegimeParameters] = {
    PolicyRegime.EXPANSION: RegimeParameters(
        regime=PolicyRegime.EXPANSION,
        start_year=2010,
        end_year=2016,
        mean_total_migration=1289.0,
        std_total_migration=506.7,
        mean_refugee_arrivals=992.3,
        refugee_share_pct=92.4,
        non_refugee_share_pct=7.6,
        trend_coefficient=182.3,  # From Chow test pre-2017 coefficients
        volatility_ratio=0.393,  # CV = 506.7/1289.0
        primary_status_composition="refugee_dominant",
        status_durability="high",  # Refugees have path to LPR/citizenship
    ),
    PolicyRegime.RESTRICTION: RegimeParameters(
        regime=PolicyRegime.RESTRICTION,
        start_year=2017,
        end_year=2020,
        mean_total_migration=1196.5,
        std_total_migration=1221.0,
        mean_refugee_arrivals=378.0,
        refugee_share_pct=102.2,  # Artifact of FY/CY mismatch
        non_refugee_share_pct=-2.2,  # Negative due to FY/CY artifact
        trend_coefficient=467.3,  # High variance skews trend
        volatility_ratio=1.020,  # High volatility CV > 1
        primary_status_composition="refugee_declining",
        status_durability="high",  # Still mostly durable status
    ),
    PolicyRegime.VOLATILITY: RegimeParameters(
        regime=PolicyRegime.VOLATILITY,
        start_year=2021,
        end_year=2024,
        mean_total_migration=3283.8,
        std_total_migration=1759.1,
        mean_refugee_arrivals=218.0,
        refugee_share_pct=6.7,  # Validated by Agent 1
        non_refugee_share_pct=93.3,  # Overwhelmingly non-refugee
        trend_coefficient=1500.1,  # Strong upward trend
        volatility_ratio=0.536,  # CV = 1759.1/3283.8
        primary_status_composition="parole_dominant",
        status_durability="low",  # Parolees face 2-4 year cliff
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_regime(year: int) -> PolicyRegime:
    """
    Get the policy regime for a given year.

    This is the primary function for determining R_t, the regime variable
    used in regime-aware modeling.

    Args:
        year: Calendar year (2010-2024 supported)

    Returns:
        PolicyRegime enum value

    Raises:
        ValueError: If year is outside supported range (2010-2024)

    Example:
        >>> get_regime(2015)
        <PolicyRegime.EXPANSION: 'expansion'>
        >>> get_regime(2023)
        <PolicyRegime.VOLATILITY: 'volatility'>
    """
    for regime, year_range in REGIME_YEARS.items():
        if year in year_range:
            return regime

    raise ValueError(
        f"Year {year} is outside the supported range (2010-2024). "
        f"Regimes are only defined for years in the structural break analysis period."
    )


def get_regime_name(year: int) -> str:
    """
    Get the regime name string for a given year.

    Convenience wrapper around get_regime() for string-based lookups.

    Args:
        year: Calendar year (2010-2024)

    Returns:
        Regime name string ("expansion", "restriction", or "volatility")
    """
    return get_regime(year).value


def get_regime_params(regime: PolicyRegime | str) -> RegimeParameters:
    """
    Get the statistical parameters for a regime.

    Args:
        regime: PolicyRegime enum or string name ("expansion", "restriction", "volatility")

    Returns:
        RegimeParameters dataclass with all regime statistics

    Raises:
        KeyError: If regime is not found
        ValueError: If string is not a valid regime name

    Example:
        >>> params = get_regime_params(PolicyRegime.EXPANSION)
        >>> params.mean_total_migration
        1289.0
        >>> params.refugee_share_pct
        92.4
    """
    if isinstance(regime, str):
        try:
            regime = PolicyRegime(regime.lower())
        except ValueError:
            valid = [r.value for r in PolicyRegime]
            raise ValueError(
                f"Invalid regime name '{regime}'. Valid names: {valid}"
            ) from None

    return REGIME_PARAMETERS[regime]


def get_policy_events_for_regime(regime: PolicyRegime | str) -> list[PolicyEvent]:
    """
    Get all policy events associated with a regime.

    Args:
        regime: PolicyRegime enum or string name

    Returns:
        List of PolicyEvent objects, sorted by date
    """
    if isinstance(regime, str):
        regime = PolicyRegime(regime.lower())

    events = [e for e in POLICY_EVENTS.values() if e.regime == regime]
    return sorted(events, key=lambda e: e.date)


def get_policy_events_for_year(year: int) -> list[PolicyEvent]:
    """
    Get all policy events that occurred in a specific year.

    Args:
        year: Calendar year

    Returns:
        List of PolicyEvent objects from that year, sorted by date
    """
    events = [e for e in POLICY_EVENTS.values() if e.date.year == year]
    return sorted(events, key=lambda e: e.date)


def get_regime_transition_years() -> list[int]:
    """
    Get the years where regime transitions occur.

    Returns:
        List of years that mark the start of a new regime
    """
    return [2017, 2021]  # Expansion->Restriction, Restriction->Volatility


def is_transition_year(year: int) -> bool:
    """
    Check if a year is at a regime boundary.

    Args:
        year: Calendar year

    Returns:
        True if year is a transition year (2017 or 2021)
    """
    return year in get_regime_transition_years()


def get_regime_label(regime: PolicyRegime | str) -> str:
    """
    Get a human-readable label for a regime.

    Args:
        regime: PolicyRegime enum or string name

    Returns:
        Formatted label string
    """
    if isinstance(regime, str):
        regime = PolicyRegime(regime.lower())

    labels = {
        PolicyRegime.EXPANSION: "Expansion (2010-2016): USRAP growth, 92% refugee share",
        PolicyRegime.RESTRICTION: "Restriction (2017-2020): Ceiling cuts, Travel Ban",
        PolicyRegime.VOLATILITY: "Volatility (2021-2024): Parole surge, 7% refugee share",
    }
    return labels[regime]


def create_regime_indicator_series(
    years: list[int],
) -> dict[str, list[int]]:
    """
    Create regime indicator variables for a list of years.

    Returns binary indicator columns suitable for regression analysis.
    The Expansion regime is the reference category (omitted).

    Args:
        years: List of calendar years

    Returns:
        Dictionary with keys 'R_restriction' and 'R_volatility',
        each containing a list of 0/1 indicators

    Example:
        >>> years = [2015, 2016, 2017, 2018, 2021, 2022]
        >>> indicators = create_regime_indicator_series(years)
        >>> indicators['R_restriction']
        [0, 0, 1, 1, 0, 0]
        >>> indicators['R_volatility']
        [0, 0, 0, 0, 1, 1]
    """
    r_restriction = []
    r_volatility = []

    for year in years:
        regime = get_regime(year)
        r_restriction.append(1 if regime == PolicyRegime.RESTRICTION else 0)
        r_volatility.append(1 if regime == PolicyRegime.VOLATILITY else 0)

    return {
        "R_restriction": r_restriction,
        "R_volatility": r_volatility,
    }


def get_regime_for_projection_year(year: int) -> PolicyRegime:
    """
    Get the regime for a projection year (including future years).

    For years beyond 2024, uses the Volatility regime as the base case
    but flags that regime assignment is uncertain.

    Args:
        year: Calendar year (can be future)

    Returns:
        PolicyRegime enum value

    Note:
        For years > 2024, returns VOLATILITY as default but this should
        be overridden by scenario-specific regime assumptions.
    """
    if year <= 2024:
        return get_regime(year)
    # Future years default to Volatility but should be scenario-dependent
    return PolicyRegime.VOLATILITY


# =============================================================================
# EXPORT FUNCTIONS FOR RESULTS
# =============================================================================


def export_regime_framework_summary() -> dict:
    """
    Export the regime framework as a dictionary for JSON serialization.

    Returns:
        Dictionary containing all regime definitions, parameters, and events
    """
    return {
        "framework_version": "1.0.0",
        "adr_reference": "ADR-021",
        "module_reference": "Module 2.1.2 (Structural Break Detection)",
        "regimes": {
            regime.value: {
                "boundaries": {
                    "start_year": REGIME_BOUNDARIES[regime][0],
                    "end_year": REGIME_BOUNDARIES[regime][1],
                },
                "parameters": {
                    "mean_total_migration": REGIME_PARAMETERS[regime].mean_total_migration,
                    "std_total_migration": REGIME_PARAMETERS[regime].std_total_migration,
                    "mean_refugee_arrivals": REGIME_PARAMETERS[regime].mean_refugee_arrivals,
                    "refugee_share_pct": REGIME_PARAMETERS[regime].refugee_share_pct,
                    "non_refugee_share_pct": REGIME_PARAMETERS[regime].non_refugee_share_pct,
                    "trend_coefficient": REGIME_PARAMETERS[regime].trend_coefficient,
                    "volatility_ratio": REGIME_PARAMETERS[regime].volatility_ratio,
                    "primary_status_composition": REGIME_PARAMETERS[
                        regime
                    ].primary_status_composition,
                    "status_durability": REGIME_PARAMETERS[regime].status_durability,
                },
                "events": [
                    {
                        "event_id": e.event_id,
                        "name": e.name,
                        "date": e.date.isoformat(),
                        "mechanism": e.mechanism,
                        "expected_direction": e.expected_direction,
                        "primary_source": e.primary_source,
                        "description": e.description,
                    }
                    for e in get_policy_events_for_regime(regime)
                ],
            }
            for regime in PolicyRegime
        },
        "transition_years": get_regime_transition_years(),
        "structural_break_evidence": {
            "chow_test_2017": {
                "f_statistic": 1.29,
                "p_value": 0.314,
                "significant": False,
                "note": "Not significant at 5% level",
            },
            "chow_test_2020": {
                "f_statistic": 16.01,
                "p_value": 0.0006,
                "significant": True,
                "note": "Highly significant structural break",
            },
            "chow_test_2021": {
                "f_statistic": 10.28,
                "p_value": 0.003,
                "significant": True,
                "note": "Significant structural break",
            },
        },
    }


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    print("=" * 70)
    print("Policy Regime Framework - ADR-021 Recommendation #4")
    print("=" * 70)

    # Print regime summary
    for regime in PolicyRegime:
        print(f"\n{get_regime_label(regime)}")
        params = get_regime_params(regime)
        print(f"  Mean migration: {params.mean_total_migration:.1f}")
        print(f"  Refugee share:  {params.refugee_share_pct:.1f}%")
        print(f"  Durability:     {params.status_durability}")

        events = get_policy_events_for_regime(regime)
        print(f"  Policy events:  {len(events)}")
        for e in events[:3]:  # Show first 3
            print(f"    - {e.name} ({e.date})")
        if len(events) > 3:
            print(f"    ... and {len(events) - 3} more")

    # Export framework summary
    print("\n" + "-" * 70)
    print("Exporting framework summary...")

    output_dir = Path(__file__).parent.parent.parent.parent / "docs" / "governance" / "adrs" / "021-reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rec4_regime_framework.json"

    summary = export_regime_framework_summary()
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to: {output_path}")
    print("\nFramework ready for use in regime-aware modeling.")
