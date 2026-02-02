"""
Configuration module for cohort projections.

Provides centralized configuration constants, mappings, and utilities.
This module consolidates settings that are shared across multiple
processing modules.

Exports:
    Race/ethnicity mappings:
        - CANONICAL_RACE_CATEGORIES: The 6 canonical race/ethnicity categories
        - CANONICAL_RACE_CODES: Numeric codes for canonical categories
        - CENSUS_RACE_MAP: Census data source mappings
        - SEER_RACE_MAP: SEER data source mappings (fertility, mortality)
        - MIGRATION_RACE_MAP: IRS/Census migration data mappings
        - map_race_to_canonical: Helper function for mapping

Example:
    >>> from cohort_projections.config import CANONICAL_RACE_CATEGORIES
    >>> from cohort_projections.config import map_race_to_canonical
    >>> map_race_to_canonical("White NH", source="seer")
    'White alone, Non-Hispanic'
"""

from cohort_projections.config.race_mappings import (
    CANONICAL_RACE_CATEGORIES,
    CANONICAL_RACE_CODES,
    CENSUS_RACE_MAP,
    MIGRATION_RACE_MAP,
    SEER_RACE_MAP,
    get_all_valid_aliases,
    map_race_to_canonical,
)

__all__ = [
    # Canonical definitions
    "CANONICAL_RACE_CATEGORIES",
    "CANONICAL_RACE_CODES",
    # Source-specific mappings
    "CENSUS_RACE_MAP",
    "SEER_RACE_MAP",
    "MIGRATION_RACE_MAP",
    # Helper functions
    "map_race_to_canonical",
    "get_all_valid_aliases",
]
