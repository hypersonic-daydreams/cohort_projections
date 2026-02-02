"""
Canonical race/ethnicity category mappings for cohort projections.

This module provides a single source of truth for race/ethnicity mappings
used throughout the projection system. All data processors should import
their mappings from here rather than defining them inline.

The 6 canonical categories follow Census/SEER standards:
    1. White alone, Non-Hispanic
    2. Black alone, Non-Hispanic
    3. AIAN alone, Non-Hispanic (American Indian/Alaska Native)
    4. Asian/PI alone, Non-Hispanic (Asian/Pacific Islander)
    5. Two or more races, Non-Hispanic
    6. Hispanic (any race)

References:
    - AGENTS.md Section 6: Demographic Guidelines
    - ADR-007: Race/ethnicity categorization decision
    - projection_config.yaml: demographics.race_ethnicity.categories
"""

from typing import Literal

# =============================================================================
# CANONICAL DEFINITIONS
# =============================================================================

# The 6 canonical race/ethnicity categories (order matters - matches config)
CANONICAL_RACE_CATEGORIES: tuple[str, ...] = (
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
)

# Numeric codes for canonical categories (1-indexed to match common conventions)
CANONICAL_RACE_CODES: dict[int, str] = {
    1: "White alone, Non-Hispanic",
    2: "Black alone, Non-Hispanic",
    3: "AIAN alone, Non-Hispanic",
    4: "Asian/PI alone, Non-Hispanic",
    5: "Two or more races, Non-Hispanic",
    6: "Hispanic (any race)",
}

# =============================================================================
# CENSUS SOURCE MAPPINGS
# =============================================================================

# Census PEP/ACS race/ethnicity code mappings
# Used by: base_population.py
CENSUS_RACE_MAP: dict[str, str] = {
    # White alone, Non-Hispanic
    "WA_NH": "White alone, Non-Hispanic",
    "NH_WHITE": "White alone, Non-Hispanic",
    "NHWA": "White alone, Non-Hispanic",
    # Black alone, Non-Hispanic
    "BA_NH": "Black alone, Non-Hispanic",
    "NH_BLACK": "Black alone, Non-Hispanic",
    "NHBA": "Black alone, Non-Hispanic",
    # AIAN alone, Non-Hispanic
    "IA_NH": "AIAN alone, Non-Hispanic",
    "NH_AIAN": "AIAN alone, Non-Hispanic",
    "NHIA": "AIAN alone, Non-Hispanic",
    # Asian/PI alone, Non-Hispanic
    "AA_NH": "Asian/PI alone, Non-Hispanic",
    "NH_ASIAN": "Asian/PI alone, Non-Hispanic",
    "NHAA": "Asian/PI alone, Non-Hispanic",
    "NH_API": "Asian/PI alone, Non-Hispanic",
    # Two or more races, Non-Hispanic
    "TOM_NH": "Two or more races, Non-Hispanic",
    "NH_TOM": "Two or more races, Non-Hispanic",
    "NHTOM": "Two or more races, Non-Hispanic",
    "NH_TWO_OR_MORE": "Two or more races, Non-Hispanic",
    # Hispanic (any race)
    "H": "Hispanic (any race)",
    "HISP": "Hispanic (any race)",
    "HISPANIC": "Hispanic (any race)",
}

# =============================================================================
# SEER SOURCE MAPPINGS
# =============================================================================

# SEER/NVSS/CDC race/ethnicity code mappings
# Used by: fertility_rates.py, survival_rates.py
SEER_RACE_MAP: dict[str, str] = {
    # White alone, Non-Hispanic
    "White Non-Hispanic": "White alone, Non-Hispanic",
    "White NH": "White alone, Non-Hispanic",
    "NH White": "White alone, Non-Hispanic",
    "Non-Hispanic White": "White alone, Non-Hispanic",
    "WNH": "White alone, Non-Hispanic",
    "1": "White alone, Non-Hispanic",
    # Black alone, Non-Hispanic
    "Black Non-Hispanic": "Black alone, Non-Hispanic",
    "Black NH": "Black alone, Non-Hispanic",
    "NH Black": "Black alone, Non-Hispanic",
    "Non-Hispanic Black": "Black alone, Non-Hispanic",
    "BNH": "Black alone, Non-Hispanic",
    "2": "Black alone, Non-Hispanic",
    # AIAN alone, Non-Hispanic
    "AIAN Non-Hispanic": "AIAN alone, Non-Hispanic",
    "AIAN NH": "AIAN alone, Non-Hispanic",
    "NH AIAN": "AIAN alone, Non-Hispanic",
    "American Indian/Alaska Native Non-Hispanic": "AIAN alone, Non-Hispanic",
    "AI/AN Non-Hispanic": "AIAN alone, Non-Hispanic",
    "3": "AIAN alone, Non-Hispanic",
    # Asian/PI alone, Non-Hispanic
    "Asian/PI Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "Asian/Pacific Islander Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "Asian NH": "Asian/PI alone, Non-Hispanic",
    "NH Asian": "Asian/PI alone, Non-Hispanic",
    "API Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "4": "Asian/PI alone, Non-Hispanic",
    # Two or more races, Non-Hispanic
    "Two or More Races Non-Hispanic": "Two or more races, Non-Hispanic",
    "Two+ Races NH": "Two or more races, Non-Hispanic",
    "NH Two or More Races": "Two or more races, Non-Hispanic",
    "Multiracial Non-Hispanic": "Two or more races, Non-Hispanic",
    "5": "Two or more races, Non-Hispanic",
    # Hispanic (any race)
    "Hispanic": "Hispanic (any race)",
    "Hispanic (any race)": "Hispanic (any race)",
    "All Hispanic": "Hispanic (any race)",
    "Hisp": "Hispanic (any race)",
    "6": "Hispanic (any race)",
}

# =============================================================================
# MIGRATION SOURCE MAPPINGS
# =============================================================================

# IRS/Census migration data race/ethnicity mappings
# Used by: migration_rates.py
# Note: Migration data often uses the same SEER-style codes
MIGRATION_RACE_MAP: dict[str, str] = SEER_RACE_MAP.copy()

# =============================================================================
# COMBINED MAPPING (ALL SOURCES)
# =============================================================================

# Union of all source mappings for general-purpose use
_ALL_RACE_MAPPINGS: dict[str, str] = {}
_ALL_RACE_MAPPINGS.update(CENSUS_RACE_MAP)
_ALL_RACE_MAPPINGS.update(SEER_RACE_MAP)
_ALL_RACE_MAPPINGS.update(MIGRATION_RACE_MAP)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

SourceType = Literal["census", "seer", "migration", "auto"]


def map_race_to_canonical(
    race_code: str,
    source: SourceType = "auto",
    strict: bool = False,
) -> str | None:
    """
    Map a race/ethnicity code to the canonical category.

    Args:
        race_code: The source race/ethnicity code to map
        source: Data source type for lookup:
            - "census": Use Census PEP/ACS mappings only
            - "seer": Use SEER/NVSS/CDC mappings only
            - "migration": Use IRS/Census migration mappings only
            - "auto": Try all mappings (default)
        strict: If True, raise ValueError for unmapped codes.
                If False (default), return None for unmapped codes.

    Returns:
        Canonical race/ethnicity category string, or None if not found
        and strict=False

    Raises:
        ValueError: If strict=True and race_code cannot be mapped

    Example:
        >>> map_race_to_canonical("White NH")
        'White alone, Non-Hispanic'
        >>> map_race_to_canonical("WA_NH", source="census")
        'White alone, Non-Hispanic'
        >>> map_race_to_canonical("Unknown")
        None
        >>> map_race_to_canonical("Unknown", strict=True)
        ValueError: Cannot map race code 'Unknown' to canonical category
    """
    # Handle common input variations
    race_code_clean = str(race_code).strip()

    # Select the appropriate mapping dictionary
    if source == "census":
        mapping = CENSUS_RACE_MAP
    elif source == "seer":
        mapping = SEER_RACE_MAP
    elif source == "migration":
        mapping = MIGRATION_RACE_MAP
    else:  # "auto"
        mapping = _ALL_RACE_MAPPINGS

    # Check if already canonical
    if race_code_clean in CANONICAL_RACE_CATEGORIES:
        return race_code_clean

    # Try direct lookup
    result = mapping.get(race_code_clean)

    if result is None and strict:
        raise ValueError(
            f"Cannot map race code '{race_code}' to canonical category. "
            f"Valid codes for source='{source}': {list(mapping.keys())}"
        )

    return result


def get_all_valid_aliases(category: str) -> list[str]:
    """
    Get all known aliases for a canonical race/ethnicity category.

    Useful for documentation or validation purposes.

    Args:
        category: A canonical race/ethnicity category

    Returns:
        List of all known aliases that map to this category

    Raises:
        ValueError: If category is not a valid canonical category

    Example:
        >>> aliases = get_all_valid_aliases("White alone, Non-Hispanic")
        >>> "White NH" in aliases
        True
        >>> "WA_NH" in aliases
        True
    """
    if category not in CANONICAL_RACE_CATEGORIES:
        raise ValueError(
            f"'{category}' is not a canonical category. "
            f"Valid categories: {list(CANONICAL_RACE_CATEGORIES)}"
        )

    aliases = []
    for alias, mapped_category in _ALL_RACE_MAPPINGS.items():
        if mapped_category == category:
            aliases.append(alias)

    return sorted(set(aliases))
