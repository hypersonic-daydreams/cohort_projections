"""
Census API Data Fetcher for Cohort Projections
===============================================

Fetches demographic data from U.S. Census Bureau APIs for North Dakota:
1. Population Estimates Program (PEP) - Annual estimates by demographics
2. American Community Survey (ACS) - Detailed demographic data for places

This module provides granular demographic data (age, sex, race, Hispanic origin)
needed for cohort-component population projections.

API Documentation:
- PEP: https://www.census.gov/data/developers/data-sets/popest-popproj.html
- ACS: https://www.census.gov/data/developers/data-sets/acs-5year.html
"""

import time
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from cohort_projections.utils.config_loader import ConfigLoader
from cohort_projections.utils.logger import setup_logger

logger = setup_logger(__name__)


class CensusDataFetcher:
    """
    Fetch demographic data from Census Bureau APIs.

    Retrieves Population Estimates Program (PEP) and American Community Survey
    (ACS) data for North Dakota at state, county, and place levels with full
    demographic detail (age, sex, race, Hispanic origin).

    Attributes:
        STATE_FIPS: FIPS code for North Dakota (38)
        STATE_NAME: State name
        cache_dir: Directory for caching raw data files
        config: Configuration loader instance
    """

    STATE_FIPS = "38"
    STATE_NAME = "North Dakota"

    # API endpoints
    PEP_BASE_URL = "https://api.census.gov/data/{vintage}/pep"
    ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

    # Demographic variables mapping
    # PEP variables for detailed characteristics
    PEP_VARS = {
        "POP": "Population estimate",
        "AGE": "Age group",
        "SEX": "Sex",
        "RACE": "Race",
        "HISP": "Hispanic origin",
    }

    # ACS variables for detailed demographics
    ACS_AGE_SEX_VARS = {
        "B01001_001E": "Total population",
        "B01001_002E": "Male total",
        "B01001_026E": "Female total",
    }

    def __init__(
        self,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize Census data fetcher.

        Args:
            cache_dir: Directory for caching downloaded data.
                      Defaults to project_root/data/raw/census
            api_key: Census API key (optional but recommended for higher rate limits)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay in seconds between retry attempts
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir = project_root / "data" / "raw" / "census"

        self.cache_dir = Path(cache_dir)
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config = ConfigLoader()

        # Create cache subdirectories
        self._setup_cache_dirs()

        logger.info(f"Initialized CensusDataFetcher for {self.STATE_NAME}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _setup_cache_dirs(self) -> None:
        """Create cache directory structure."""
        subdirs = ["pep", "acs", "decennial"]
        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _make_request(
        self, url: str, params: dict | None = None, description: str = "API request"
    ) -> requests.Response:
        """
        Make HTTP request with retry logic and error handling.

        Args:
            url: API endpoint URL
            params: Query parameters
            description: Description for logging

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails after all retries
        """
        if params is None:
            params = {}

        # Add API key if available
        if self.api_key:
            params["key"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"{description} - Attempt {attempt + 1}/{self.max_retries}")
                logger.debug(f"URL: {url}")
                logger.debug(f"Params: {params}")

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                logger.debug(f"{description} - Success")
                return response

            except requests.exceptions.RequestException as e:
                logger.warning(f"{description} - Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"{description} - All retry attempts failed")
                    raise

        # Should never reach here, but needed for type checker
        raise requests.RequestException(f"{description} - No attempts made (max_retries=0)")

    def _save_metadata(
        self, file_path: Path, source: str, vintage_or_year: int, api_url: str, record_count: int
    ) -> None:
        """
        Save metadata about downloaded data.

        Args:
            file_path: Path to data file
            source: Data source (e.g., 'PEP', 'ACS')
            vintage_or_year: Vintage year or survey year
            api_url: API endpoint used
            record_count: Number of records downloaded
        """
        metadata = {
            "source": source,
            "vintage_year": vintage_or_year,
            "state_fips": self.STATE_FIPS,
            "state_name": self.STATE_NAME,
            "api_url": api_url,
            "download_timestamp": datetime.now(UTC).isoformat(),
            "record_count": record_count,
            "data_file": file_path.name,
        }

        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved metadata to {metadata_path}")

    def fetch_pep_state_data(
        self, vintage: int = 2024, variables: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Fetch state-level PEP data with demographic characteristics.

        Retrieves population estimates by age, sex, race, and Hispanic origin
        for North Dakota.

        Args:
            vintage: PEP vintage year (e.g., 2024 for 2020-2024 estimates)
            variables: List of variables to fetch. If None, fetches standard
                      demographic variables (AGE, SEX, RACE, HISP)

        Returns:
            DataFrame with state-level population estimates by demographics

        Example:
            >>> fetcher = CensusDataFetcher()
            >>> df = fetcher.fetch_pep_state_data(vintage=2024)
            >>> print(df.columns)
            Index(['state', 'AGE', 'SEX', 'RACE', 'HISP', 'POP', ...])
        """
        logger.info(f"Fetching PEP state data for {self.STATE_NAME}, vintage {vintage}")

        # Build variable list
        if variables is None:
            # Fetch population by age, sex, race, Hispanic origin
            var_list = ["POP", "AGE", "SEX", "RACE", "HISP", "DATE_CODE", "DATE_DESC"]
        else:
            var_list = variables

        # Construct API URL
        url = f"{self.PEP_BASE_URL.format(vintage=vintage)}/charagegroups"

        params = {"get": ",".join(var_list), "for": "state:" + self.STATE_FIPS}

        # Make request
        response = self._make_request(url, params, f"PEP state data (vintage {vintage})")

        # Parse JSON response
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Convert numeric columns
        numeric_cols = ["POP", "AGE", "SEX", "RACE", "HISP", "DATE_CODE"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Retrieved {len(df)} records for state-level PEP data")

        # Cache to file
        cache_file = self.cache_dir / "pep" / f"pep_state_{vintage}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached state data to {cache_file}")

        # Save metadata
        self._save_metadata(cache_file, "PEP", vintage, url, len(df))

        return df

    def fetch_pep_county_data(
        self, vintage: int = 2024, variables: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Fetch county-level PEP data for all North Dakota counties.

        Retrieves population estimates by age, sex, race, and Hispanic origin
        for all 53 counties in North Dakota.

        Args:
            vintage: PEP vintage year
            variables: List of variables to fetch

        Returns:
            DataFrame with county-level population estimates by demographics

        Example:
            >>> fetcher = CensusDataFetcher()
            >>> df = fetcher.fetch_pep_county_data(vintage=2024)
            >>> print(df['county'].nunique())
            53
        """
        logger.info(f"Fetching PEP county data for {self.STATE_NAME}, vintage {vintage}")

        # Build variable list
        if variables is None:
            var_list = ["POP", "AGE", "SEX", "RACE", "HISP", "DATE_CODE", "DATE_DESC"]
        else:
            var_list = variables

        # Construct API URL
        url = f"{self.PEP_BASE_URL.format(vintage=vintage)}/charagegroups"

        params = {"get": ",".join(var_list), "for": "county:*", "in": f"state:{self.STATE_FIPS}"}

        # Make request
        response = self._make_request(url, params, f"PEP county data (vintage {vintage})")

        # Parse JSON response
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Convert numeric columns
        numeric_cols = ["POP", "AGE", "SEX", "RACE", "HISP", "DATE_CODE"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add county names (would need to fetch from separate API or lookup table)
        # For now, we'll just keep the FIPS codes

        logger.info(f"Retrieved {len(df)} records for {df['county'].nunique()} counties")

        # Cache to file
        cache_file = self.cache_dir / "pep" / f"pep_county_{vintage}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached county data to {cache_file}")

        # Save metadata
        self._save_metadata(cache_file, "PEP", vintage, url, len(df))

        return df

    def fetch_acs_place_data(self, year: int = 2023, dataset: str = "acs5") -> pd.DataFrame:
        """
        Fetch ACS place-level data for North Dakota cities and CDPs.

        Retrieves detailed demographic data from the American Community Survey
        for incorporated places and Census-Designated Places (CDPs) where PEP
        may not have complete detail.

        Args:
            year: ACS year (e.g., 2023 for 2019-2023 5-year estimates)
            dataset: ACS dataset ('acs5' for 5-year, 'acs1' for 1-year)

        Returns:
            DataFrame with place-level demographic data

        Example:
            >>> fetcher = CensusDataFetcher()
            >>> df = fetcher.fetch_acs_place_data(year=2023)
            >>> cdps = df[df['NAME'].str.contains('CDP')]
        """
        logger.info(f"Fetching ACS {dataset} place data for {self.STATE_NAME}, year {year}")

        # Build variable list for age/sex distributions
        # B01001: Sex by Age - provides detailed age groups by sex
        var_list = ["NAME"]

        # Add total population
        var_list.append("B01001_001E")  # Total

        # Male population by age groups
        male_vars = [f"B01001_{str(i).zfill(3)}E" for i in range(3, 26)]
        var_list.extend(male_vars)

        # Female population by age groups
        female_vars = [f"B01001_{str(i).zfill(3)}E" for i in range(27, 50)]
        var_list.extend(female_vars)

        # Add race/ethnicity variables
        # B02001: Race
        race_vars = [f"B02001_{str(i).zfill(3)}E" for i in range(1, 11)]
        var_list.extend(race_vars)

        # B03003: Hispanic or Latino origin
        hisp_vars = ["B03003_001E", "B03003_002E", "B03003_003E"]
        var_list.extend(hisp_vars)

        # Construct API URL
        url = self.ACS_BASE_URL.format(year=year)

        params = {"get": ",".join(var_list), "for": "place:*", "in": f"state:{self.STATE_FIPS}"}

        # Make request
        response = self._make_request(url, params, f"ACS {dataset} place data (year {year})")

        # Parse JSON response
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Convert numeric columns (all except NAME, state, place)
        for col in df.columns:
            if col not in ["NAME", "state", "place"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Identify place types (incorporated vs CDP)
        df["is_cdp"] = df["NAME"].str.contains("CDP", case=False)
        df["place_type"] = df["is_cdp"].apply(
            lambda x: "Census-Designated Place" if x else "Incorporated Place"
        )

        logger.info(f"Retrieved {len(df)} places")
        logger.info(f"  - Incorporated places: {(~df['is_cdp']).sum()}")
        logger.info(f"  - CDPs: {df['is_cdp'].sum()}")

        # Cache to file
        cache_file = self.cache_dir / "acs" / f"acs{dataset}_place_{year}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached place data to {cache_file}")

        # Save metadata
        self._save_metadata(cache_file, f"ACS-{dataset.upper()}", year, url, len(df))

        return df

    def fetch_pep_by_file(self, vintage: int = 2024, geography: str = "state") -> pd.DataFrame:
        """
        Fetch PEP data from direct file download.

        Some PEP datasets are more easily accessed via direct CSV file
        downloads rather than API. This method handles file-based retrieval.

        Args:
            vintage: PEP vintage year
            geography: Geographic level ('state', 'county')

        Returns:
            DataFrame with PEP data
        """
        logger.info(f"Fetching PEP data from file for {geography} level, vintage {vintage}")

        # File URL patterns
        if geography == "state":
            file_url = (
                f"https://www2.census.gov/programs-surveys/popest/datasets/"
                f"2020-{vintage}/state/asrh/"
                f"sc-est{vintage}-alldata6-{self.STATE_FIPS}.csv"
            )
        elif geography == "county":
            file_url = (
                f"https://www2.census.gov/programs-surveys/popest/datasets/"
                f"2020-{vintage}/counties/asrh/"
                f"cc-est{vintage}-alldata-{self.STATE_FIPS}.csv"
            )
        else:
            raise ValueError(f"Invalid geography: {geography}. Must be 'state' or 'county'")

        # Download file
        response = self._make_request(
            file_url, description=f"PEP file download ({geography}, vintage {vintage})"
        )

        # Parse CSV
        df = pd.read_csv(StringIO(response.text), encoding="latin1")

        logger.info(f"Downloaded {len(df)} records from PEP file")

        # Cache to file
        cache_file = self.cache_dir / "pep" / f"pep_file_{geography}_{vintage}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached to {cache_file}")

        # Save metadata
        self._save_metadata(cache_file, "PEP-FILE", vintage, file_url, len(df))

        return df

    def fetch_all_pep_data(
        self, vintage: int = 2024, use_file_method: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all PEP data (state and county levels).

        Convenience method to retrieve complete PEP dataset for projections.

        Args:
            vintage: PEP vintage year
            use_file_method: If True, use direct file download method;
                           if False, use API method

        Returns:
            Dictionary with keys 'state' and 'county' containing DataFrames
        """
        logger.info(f"Fetching all PEP data for vintage {vintage}")

        results = {}

        with tqdm(total=2, desc="Fetching PEP data") as pbar:
            # State level
            pbar.set_description("Fetching state-level PEP")
            if use_file_method:
                results["state"] = self.fetch_pep_by_file(vintage, "state")
            else:
                results["state"] = self.fetch_pep_state_data(vintage)
            pbar.update(1)

            # County level
            pbar.set_description("Fetching county-level PEP")
            if use_file_method:
                results["county"] = self.fetch_pep_by_file(vintage, "county")
            else:
                results["county"] = self.fetch_pep_county_data(vintage)
            pbar.update(1)

        logger.info("Completed fetching all PEP data")
        return results

    def fetch_all_acs_data(self, year: int = 2023, dataset: str = "acs5") -> pd.DataFrame:
        """
        Fetch all ACS data for North Dakota.

        Convenience method to retrieve complete ACS dataset.

        Args:
            year: ACS year
            dataset: ACS dataset type ('acs5' or 'acs1')

        Returns:
            DataFrame with place-level ACS data
        """
        logger.info(f"Fetching all ACS {dataset} data for year {year}")

        with tqdm(total=1, desc=f"Fetching ACS {dataset} data") as pbar:
            df = self.fetch_acs_place_data(year, dataset)
            pbar.update(1)

        logger.info("Completed fetching all ACS data")
        return df

    def get_cached_data(
        self, source: str, geography: str, vintage_or_year: int
    ) -> pd.DataFrame | None:
        """
        Retrieve cached data if available.

        Args:
            source: Data source ('pep', 'acs')
            geography: Geographic level ('state', 'county', 'place')
            vintage_or_year: Vintage year or survey year

        Returns:
            DataFrame if cached file exists, None otherwise
        """
        source_lower = source.lower()

        if source_lower == "pep":
            cache_file = self.cache_dir / "pep" / f"pep_{geography}_{vintage_or_year}.parquet"
        elif source_lower.startswith("acs"):
            cache_file = self.cache_dir / "acs" / f"acs5_{geography}_{vintage_or_year}.parquet"
        else:
            logger.warning(f"Unknown source: {source}")
            return None

        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        else:
            logger.debug(f"No cached data found at {cache_file}")
            return None

    def list_cached_files(self) -> dict[str, list[Path]]:
        """
        List all cached data files.

        Returns:
            Dictionary with source types as keys and lists of file paths as values
        """
        cached_files = {
            "pep": list((self.cache_dir / "pep").glob("*.parquet")),
            "acs": list((self.cache_dir / "acs").glob("*.parquet")),
            "decennial": list((self.cache_dir / "decennial").glob("*.parquet")),
        }

        logger.info("Cached files:")
        for source, files in cached_files.items():
            logger.info(f"  {source}: {len(files)} files")
            for file in files:
                logger.debug(f"    - {file.name}")

        return cached_files


def main():
    """
    Example usage of CensusDataFetcher.

    Demonstrates fetching PEP and ACS data for North Dakota.
    """
    print("=" * 70)
    print("Census API Data Fetcher for North Dakota Cohort Projections")
    print("=" * 70)
    print(f"Run timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize fetcher
    fetcher = CensusDataFetcher()

    # Fetch PEP data
    print("\n" + "=" * 70)
    print("Fetching Population Estimates Program (PEP) Data")
    print("=" * 70)

    try:
        pep_data = fetcher.fetch_all_pep_data(vintage=2024, use_file_method=True)

        print(f"\nState-level PEP data: {len(pep_data['state'])} records")
        print(f"County-level PEP data: {len(pep_data['county'])} records")

        if len(pep_data["state"]) > 0:
            print("\nSample state data columns:")
            print(pep_data["state"].columns.tolist()[:10])

        if len(pep_data["county"]) > 0:
            print("\nSample county data columns:")
            print(pep_data["county"].columns.tolist()[:10])
            print(f"\nNumber of counties: {pep_data['county']['COUNTY'].nunique()}")

    except Exception as e:
        logger.error(f"Error fetching PEP data: {e}")
        print(f"\nError fetching PEP data: {e}")

    # Fetch ACS data
    print("\n" + "=" * 70)
    print("Fetching American Community Survey (ACS) Data")
    print("=" * 70)

    try:
        acs_data = fetcher.fetch_all_acs_data(year=2023, dataset="acs5")

        print(f"\nPlace-level ACS data: {len(acs_data)} places")
        print(f"  - Incorporated places: {(~acs_data['is_cdp']).sum()}")
        print(f"  - Census-Designated Places: {acs_data['is_cdp'].sum()}")

        if len(acs_data) > 0:
            print("\nLargest places by population:")
            top_places = acs_data.nlargest(5, "B01001_001E")[["NAME", "B01001_001E", "place_type"]]
            for _idx, row in top_places.iterrows():
                print(f"  {row['NAME']}: {row['B01001_001E']:,} ({row['place_type']})")

    except Exception as e:
        logger.error(f"Error fetching ACS data: {e}")
        print(f"\nError fetching ACS data: {e}")

    # List cached files
    print("\n" + "=" * 70)
    print("Cached Data Files")
    print("=" * 70)
    fetcher.list_cached_files()

    print("\n" + "=" * 70)
    print("Data fetch complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
