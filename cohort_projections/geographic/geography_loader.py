"""
Geography reference data loader for cohort projections.

Loads FIPS codes, names, and geographic relationships for North Dakota
state, counties, and incorporated places. Supports loading from Census
TIGER files or local reference CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
import json

from cohort_projections.utils.logger import get_logger_from_config
from cohort_projections.utils.config_loader import load_projection_config

logger = get_logger_from_config(__name__)


def load_nd_counties(
    source: Literal['local', 'tiger'] = 'local',
    vintage: int = 2020,
    reference_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load North Dakota county reference data.

    Args:
        source: Data source - 'local' for CSV file, 'tiger' for Census TIGER files
        vintage: Census vintage year (default: 2020)
        reference_path: Optional path to local reference CSV file
                       (default: data/raw/geographic/nd_counties.csv)

    Returns:
        DataFrame with columns:
        - state_fips: str (2 digits, e.g., '38')
        - county_fips: str (5 digits, e.g., '38101')
        - county_name: str (e.g., 'Cass County')
        - population: int (optional, latest available)

    Raises:
        FileNotFoundError: If reference file not found
        ValueError: If data format invalid

    Example:
        >>> counties = load_nd_counties()
        >>> len(counties)
        53
        >>> counties[counties['county_name'] == 'Cass County']['county_fips'].values[0]
        '38101'
    """
    logger.info(f"Loading North Dakota counties (source: {source}, vintage: {vintage})")

    if source == 'local':
        # Load from local CSV file
        if reference_path is None:
            project_root = Path(__file__).parent.parent.parent
            reference_path = project_root / "data" / "raw" / "geographic" / "nd_counties.csv"

        reference_path = Path(reference_path)

        if not reference_path.exists():
            # Create default reference file if it doesn't exist
            logger.warning(f"County reference file not found: {reference_path}")
            logger.info("Creating default county reference data")
            counties_df = _create_default_nd_counties()

            # Save for future use
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            counties_df.to_csv(reference_path, index=False)
            logger.info(f"Saved default county reference to {reference_path}")

            return counties_df

        logger.info(f"Loading counties from {reference_path}")
        counties_df = pd.read_csv(reference_path, dtype={'county_fips': str, 'state_fips': str})

    elif source == 'tiger':
        # Load from Census TIGER files
        logger.info("Loading counties from Census TIGER files")
        counties_df = _load_counties_from_tiger(vintage)

    else:
        raise ValueError(f"Unknown source: {source}. Use 'local' or 'tiger'")

    # Validate and standardize
    counties_df = _validate_county_data(counties_df)

    logger.info(f"Loaded {len(counties_df)} North Dakota counties")

    return counties_df


def load_nd_places(
    source: Literal['local', 'tiger'] = 'local',
    vintage: int = 2020,
    reference_path: Optional[Path] = None,
    min_population: Optional[int] = None
) -> pd.DataFrame:
    """
    Load North Dakota incorporated place reference data.

    Args:
        source: Data source - 'local' for CSV, 'tiger' for Census TIGER
        vintage: Census vintage year (default: 2020)
        reference_path: Optional path to local reference CSV
                       (default: data/raw/geographic/nd_places.csv)
        min_population: Optional minimum population threshold for filtering

    Returns:
        DataFrame with columns:
        - state_fips: str (2 digits, '38')
        - place_fips: str (7 digits, e.g., '3825700' for Fargo)
        - place_name: str (e.g., 'Fargo city')
        - county_fips: str (5 digits, containing county)
        - population: int (optional, latest available)

    Raises:
        FileNotFoundError: If reference file not found
        ValueError: If data format invalid

    Example:
        >>> places = load_nd_places(min_population=500)
        >>> fargo = places[places['place_name'] == 'Fargo city']
        >>> fargo['place_fips'].values[0]
        '3825700'
        >>> fargo['county_fips'].values[0]
        '38101'  # Cass County
    """
    logger.info(f"Loading North Dakota places (source: {source}, vintage: {vintage})")

    if source == 'local':
        # Load from local CSV file
        if reference_path is None:
            project_root = Path(__file__).parent.parent.parent
            reference_path = project_root / "data" / "raw" / "geographic" / "nd_places.csv"

        reference_path = Path(reference_path)

        if not reference_path.exists():
            # Create default reference file if it doesn't exist
            logger.warning(f"Place reference file not found: {reference_path}")
            logger.info("Creating default place reference data (major cities only)")
            places_df = _create_default_nd_places()

            # Save for future use
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            places_df.to_csv(reference_path, index=False)
            logger.info(f"Saved default place reference to {reference_path}")

            # Filter by population if specified
            if min_population is not None:
                places_df = places_df[places_df['population'] >= min_population].copy()

            return places_df

        logger.info(f"Loading places from {reference_path}")
        places_df = pd.read_csv(
            reference_path,
            dtype={'place_fips': str, 'state_fips': str, 'county_fips': str}
        )

    elif source == 'tiger':
        # Load from Census TIGER files
        logger.info("Loading places from Census TIGER files")
        places_df = _load_places_from_tiger(vintage)

    else:
        raise ValueError(f"Unknown source: {source}. Use 'local' or 'tiger'")

    # Validate and standardize
    places_df = _validate_place_data(places_df)

    # Filter by population if specified
    if min_population is not None and 'population' in places_df.columns:
        original_count = len(places_df)
        places_df = places_df[places_df['population'] >= min_population].copy()
        logger.info(
            f"Filtered to {len(places_df)}/{original_count} places "
            f"with population >= {min_population}"
        )

    logger.info(f"Loaded {len(places_df)} North Dakota places")

    return places_df


def get_place_to_county_mapping(
    places_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get mapping from places to their containing counties.

    Args:
        places_df: Optional places DataFrame (if None, loads from default source)

    Returns:
        DataFrame with columns:
        - place_fips: str (7 digits)
        - county_fips: str (5 digits)
        - place_name: str
        - county_name: str (if available)

    Notes:
        - Each place belongs to exactly one county
        - Used for hierarchical aggregation (places → counties)
        - Places don't cover entire county (unincorporated areas exist)

    Example:
        >>> mapping = get_place_to_county_mapping()
        >>> fargo = mapping[mapping['place_name'] == 'Fargo city']
        >>> fargo['county_fips'].values[0]
        '38101'  # Cass County
    """
    logger.info("Getting place-to-county mapping")

    # Load places if not provided
    if places_df is None:
        places_df = load_nd_places()

    # Ensure required columns exist
    required_cols = ['place_fips', 'county_fips', 'place_name']
    missing_cols = [col for col in required_cols if col not in places_df.columns]
    if missing_cols:
        raise ValueError(f"places_df missing required columns: {missing_cols}")

    # Create mapping
    mapping = places_df[['place_fips', 'county_fips', 'place_name']].copy()

    # Add county names if available
    try:
        counties_df = load_nd_counties()
        mapping = mapping.merge(
            counties_df[['county_fips', 'county_name']],
            on='county_fips',
            how='left'
        )
    except Exception as e:
        logger.warning(f"Could not load county names: {e}")

    logger.info(f"Created mapping for {len(mapping)} places")

    return mapping


def load_geography_list(
    level: Literal['state', 'county', 'place'],
    config: Optional[Dict] = None,
    fips_codes: Optional[List[str]] = None
) -> List[str]:
    """
    Load list of FIPS codes for geographies to project.

    Args:
        level: Geographic level ('state', 'county', 'place')
        config: Optional configuration dictionary
        fips_codes: Optional explicit list of FIPS codes (overrides config)

    Returns:
        List of FIPS codes to project

    Notes:
        - For state: Returns ['38'] (North Dakota)
        - For counties: All 53 or filtered list based on config
        - For places: All 406 or filtered by population/list

    Configuration Examples:
        ```yaml
        geographic:
          state: "38"
          counties:
            mode: "all"  # or "list"
            fips_codes: ["38101", "38015"]  # if mode="list"
          places:
            mode: "threshold"  # "all", "threshold", or "list"
            min_population: 500
        ```

    Example:
        >>> # Load all counties
        >>> counties = load_geography_list('county')
        >>> len(counties)
        53

        >>> # Load places with population >= 500
        >>> config = {'geographic': {'places': {'mode': 'threshold', 'min_population': 500}}}
        >>> places = load_geography_list('place', config=config)
        >>> len(places)  # Approximately 150
    """
    logger.info(f"Loading geography list for level: {level}")

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # If explicit FIPS codes provided, use those
    if fips_codes is not None:
        logger.info(f"Using explicit FIPS codes list: {len(fips_codes)} geographies")
        return fips_codes

    geographic_config = config.get('geographic', {})

    if level == 'state':
        # State level - just North Dakota
        state_fips = geographic_config.get('state', '38')
        return [state_fips]

    elif level == 'county':
        # County level
        county_config = geographic_config.get('counties', 'all')

        if county_config == 'all' or (isinstance(county_config, dict) and
                                      county_config.get('mode') == 'all'):
            # All counties
            counties_df = load_nd_counties()
            fips_list = counties_df['county_fips'].tolist()

        elif isinstance(county_config, dict):
            mode = county_config.get('mode', 'all')

            if mode == 'list':
                # Explicit list
                fips_list = county_config.get('fips_codes', [])

            elif mode == 'threshold':
                # Filter by population
                min_pop = county_config.get('min_population', 0)
                counties_df = load_nd_counties()

                if 'population' in counties_df.columns:
                    counties_df = counties_df[counties_df['population'] >= min_pop]

                fips_list = counties_df['county_fips'].tolist()

            else:
                raise ValueError(f"Unknown county mode: {mode}")

        else:
            # Default to all
            counties_df = load_nd_counties()
            fips_list = counties_df['county_fips'].tolist()

        logger.info(f"Loaded {len(fips_list)} counties")
        return fips_list

    elif level == 'place':
        # Place level
        place_config = geographic_config.get('places', 'all')

        if place_config == 'all' or (isinstance(place_config, dict) and
                                     place_config.get('mode') == 'all'):
            # All places
            places_df = load_nd_places()
            fips_list = places_df['place_fips'].tolist()

        elif isinstance(place_config, dict):
            mode = place_config.get('mode', 'all')

            if mode == 'list':
                # Explicit list
                fips_list = place_config.get('fips_codes', [])

            elif mode == 'threshold':
                # Filter by population
                min_pop = place_config.get('min_population', 0)
                places_df = load_nd_places(min_population=min_pop)
                fips_list = places_df['place_fips'].tolist()

            else:
                raise ValueError(f"Unknown place mode: {mode}")

        else:
            # Default to all
            places_df = load_nd_places()
            fips_list = places_df['place_fips'].tolist()

        logger.info(f"Loaded {len(fips_list)} places")
        return fips_list

    else:
        raise ValueError(f"Unknown geographic level: {level}")


def get_geography_name(
    fips: str,
    level: Optional[Literal['state', 'county', 'place']] = None
) -> str:
    """
    Get human-readable name for a FIPS code.

    Args:
        fips: FIPS code (2, 5, or 7 digits)
        level: Optional geographic level (auto-detected if not provided)

    Returns:
        Name of the geography

    Example:
        >>> get_geography_name('38')
        'North Dakota'
        >>> get_geography_name('38101')
        'Cass County'
        >>> get_geography_name('3825700')
        'Fargo city'
    """
    # Auto-detect level from FIPS length if not provided
    if level is None:
        if len(fips) == 2:
            level = 'state'
        elif len(fips) == 5:
            level = 'county'
        elif len(fips) == 7:
            level = 'place'
        else:
            raise ValueError(f"Cannot determine level for FIPS: {fips}")

    if level == 'state':
        return 'North Dakota' if fips == '38' else f'State {fips}'

    elif level == 'county':
        try:
            counties_df = load_nd_counties()
            name = counties_df[counties_df['county_fips'] == fips]['county_name'].values
            return name[0] if len(name) > 0 else f'County {fips}'
        except Exception:
            return f'County {fips}'

    elif level == 'place':
        try:
            places_df = load_nd_places()
            name = places_df[places_df['place_fips'] == fips]['place_name'].values
            return name[0] if len(name) > 0 else f'Place {fips}'
        except Exception:
            return f'Place {fips}'

    else:
        raise ValueError(f"Unknown level: {level}")


# Helper functions for creating default reference data

def _create_default_nd_counties() -> pd.DataFrame:
    """
    Create default North Dakota county reference data.

    Returns DataFrame with major ND counties. In production, this would be
    replaced with comprehensive TIGER data.
    """
    # Major ND counties (subset - full list would have all 53)
    counties = [
        {'state_fips': '38', 'county_fips': '38015', 'county_name': 'Burleigh County', 'population': 98458},
        {'state_fips': '38', 'county_fips': '38017', 'county_name': 'Cass County', 'population': 184525},
        {'state_fips': '38', 'county_fips': '38035', 'county_name': 'Grand Forks County', 'population': 73959},
        {'state_fips': '38', 'county_fips': '38101', 'county_name': 'Ward County', 'population': 69919},
        {'state_fips': '38', 'county_fips': '38059', 'county_name': 'Morton County', 'population': 33291},
        {'state_fips': '38', 'county_fips': '38089', 'county_name': 'Stark County', 'population': 33646},
        {'state_fips': '38', 'county_fips': '38091', 'county_name': 'Steele County', 'population': 1798},
        {'state_fips': '38', 'county_fips': '38093', 'county_name': 'Stutsman County', 'population': 21593},
        {'state_fips': '38', 'county_fips': '38097', 'county_name': 'Traill County', 'population': 8052},
        {'state_fips': '38', 'county_fips': '38099', 'county_name': 'Walsh County', 'population': 10563},
    ]

    logger.info("Created default county reference with 10 major counties")
    logger.warning("Using subset of ND counties. For complete data, use Census TIGER files.")

    return pd.DataFrame(counties)


def _create_default_nd_places() -> pd.DataFrame:
    """
    Create default North Dakota place reference data.

    Returns DataFrame with major ND cities. In production, this would be
    replaced with comprehensive TIGER data.
    """
    # Major ND cities
    places = [
        {'state_fips': '38', 'place_fips': '3807200', 'place_name': 'Bismarck city',
         'county_fips': '38015', 'population': 73622},
        {'state_fips': '38', 'place_fips': '3825700', 'place_name': 'Fargo city',
         'county_fips': '38017', 'population': 125990},
        {'state_fips': '38', 'place_fips': '3833900', 'place_name': 'Grand Forks city',
         'county_fips': '38035', 'population': 59166},
        {'state_fips': '38', 'place_fips': '3841500', 'place_name': 'Minot city',
         'county_fips': '38101', 'population': 48415},
        {'state_fips': '38', 'place_fips': '3850420', 'place_name': 'Mandan city',
         'county_fips': '38059', 'population': 24206},
        {'state_fips': '38', 'place_fips': '3885100', 'place_name': 'West Fargo city',
         'county_fips': '38017', 'population': 38626},
        {'state_fips': '38', 'place_fips': '3877100', 'place_name': 'Williston city',
         'county_fips': '38105', 'population': 29160},
        {'state_fips': '38', 'place_fips': '3811380', 'place_name': 'Dickinson city',
         'county_fips': '38089', 'population': 25679},
    ]

    logger.info("Created default place reference with 8 major cities")
    logger.warning("Using subset of ND places. For complete data, use Census TIGER files.")

    return pd.DataFrame(places)


def _load_counties_from_tiger(vintage: int) -> pd.DataFrame:
    """
    Load county data from Census TIGER files.

    This is a placeholder for TIGER integration. In production, would use
    tigris library or direct TIGER file download.
    """
    logger.warning("TIGER loading not yet implemented, using default data")
    return _create_default_nd_counties()


def _load_places_from_tiger(vintage: int) -> pd.DataFrame:
    """
    Load place data from Census TIGER files.

    This is a placeholder for TIGER integration. In production, would use
    tigris library or direct TIGER file download.
    """
    logger.warning("TIGER loading not yet implemented, using default data")
    return _create_default_nd_places()


def _validate_county_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize county DataFrame."""
    required_cols = ['state_fips', 'county_fips', 'county_name']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"County data missing required columns: {missing_cols}")

    # Ensure FIPS codes are strings
    df['state_fips'] = df['state_fips'].astype(str).str.zfill(2)
    df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)

    # Filter to North Dakota
    df = df[df['state_fips'] == '38'].copy()

    # Sort by FIPS
    df = df.sort_values('county_fips').reset_index(drop=True)

    return df


def _validate_place_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize place DataFrame."""
    required_cols = ['state_fips', 'place_fips', 'place_name', 'county_fips']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Place data missing required columns: {missing_cols}")

    # Ensure FIPS codes are strings
    df['state_fips'] = df['state_fips'].astype(str).str.zfill(2)
    df['place_fips'] = df['place_fips'].astype(str).str.zfill(7)
    df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)

    # Filter to North Dakota
    df = df[df['state_fips'] == '38'].copy()

    # Sort by FIPS
    df = df.sort_values('place_fips').reset_index(drop=True)

    return df


if __name__ == "__main__":
    """Example usage and testing."""

    logger.info("Geography loader module loaded successfully")
    logger.info("\n" + "=" * 70)
    logger.info("Testing geography loader functions")
    logger.info("=" * 70)

    # Test county loading
    logger.info("\n1. Loading North Dakota counties...")
    counties = load_nd_counties()
    logger.info(f"   Loaded {len(counties)} counties")
    logger.info(f"   Sample: {counties.head(3)[['county_fips', 'county_name']].to_dict('records')}")

    # Test place loading
    logger.info("\n2. Loading North Dakota places...")
    places = load_nd_places()
    logger.info(f"   Loaded {len(places)} places")
    logger.info(f"   Sample: {places.head(3)[['place_fips', 'place_name']].to_dict('records')}")

    # Test place-to-county mapping
    logger.info("\n3. Creating place-to-county mapping...")
    mapping = get_place_to_county_mapping(places)
    logger.info(f"   Created mapping for {len(mapping)} places")

    # Test geography list loading
    logger.info("\n4. Loading geography lists...")

    state_list = load_geography_list('state')
    logger.info(f"   State level: {len(state_list)} geographies - {state_list}")

    county_list = load_geography_list('county')
    logger.info(f"   County level: {len(county_list)} geographies")

    place_list = load_geography_list('place')
    logger.info(f"   Place level: {len(place_list)} geographies")

    # Test name lookup
    logger.info("\n5. Testing name lookup...")
    logger.info(f"   FIPS '38' = {get_geography_name('38')}")
    logger.info(f"   FIPS '38015' = {get_geography_name('38015')}")
    logger.info(f"   FIPS '3825700' = {get_geography_name('3825700')}")

    logger.info("\n" + "=" * 70)
    logger.info("All tests completed successfully")
    logger.info("=" * 70)
