"""
Data fetching modules for cohort projections.

This package contains modules for fetching demographic and geographic data
from various sources including Census Bureau APIs, vital statistics, and
migration data.
"""

from cohort_projections.data.fetch.census_api import CensusDataFetcher

__all__ = ["CensusDataFetcher"]
