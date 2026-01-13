"""
Vital Statistics data fetcher.

Fetches fertility and mortality data from SEER/NVSS sources or generates
mock data if sources are unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...utils import ConfigLoader, get_logger_from_config

logger = get_logger_from_config(__name__)


class VitalStatsFetcher:
    """Fetcher for Vital Statistics data (Fertility/Mortality)."""

    def __init__(self):
        """Initialize the fetcher."""
        self.config = ConfigLoader()
        self.raw_data_dir = (
            Path(__file__).parent.parent.parent.parent / "data" / "raw" / "fertility"
        )
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_fertility_rates(self) -> pd.DataFrame:
        """
        Fetch age-specific fertility rates (ASFR).

        Attempts to load from local file defined in config.
        If file missing, generates plausible mock data.

        Returns:
            DataFrame with fertility rates
        """
        logger.info("Fetching fertility rates...")

        # Get configured file path
        fertility_config = self.config.get_parameter("pipeline", "data_processing", "fertility")
        input_file = fertility_config.get("input_file") if fertility_config else None

        # Resolve path relative to project root
        if input_file:
            project_root = Path(__file__).parent.parent.parent.parent
            file_path = project_root / input_file
        else:
            file_path = self.raw_data_dir / "seer_asfr_2018_2022.csv"

        if file_path.exists():
            logger.info(f"Loading fertility rates from {file_path}")
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Failed to read fertility data: {e}")
                logger.warning("Falling back to mock data generation")
                return self._generate_mock_fertility_data()
        else:
            logger.warning(f"Fertility data file not found at {file_path}")
            logger.warning("Generating mock fertility data for development")
            return self._generate_mock_fertility_data()

    def _generate_mock_fertility_data(self) -> pd.DataFrame:
        """
        Generate plausible mock fertility data.

        Based on typical US patterns:
        - Peak fertility ages 25-34
        - Rates vary by race (calibrated to approx US averages)
        """
        logger.info("Generating mock fertility data")

        # Configuration
        years = range(2018, 2023)  # 5 years of data
        races = [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ]
        ages = range(15, 50)  # 15-49

        records = []

        # Base curve shape (roughly normal centered on 28 with spread)
        def get_base_rate(age):
            # Peak at 28, sigma ~6
            return 0.12 * np.exp(-0.5 * ((age - 28) / 5.5) ** 2)

        # Race multipliers (approximate relative differences)
        race_multipliers = {
            "White alone, Non-Hispanic": 0.95,
            "Black alone, Non-Hispanic": 1.05,
            "AIAN alone, Non-Hispanic": 1.10,
            "Asian/PI alone, Non-Hispanic": 0.90,
            "Two or more races, Non-Hispanic": 1.0,
            "Hispanic (any race)": 1.20,
        }

        for year in years:
            for race in races:
                mult = race_multipliers.get(race, 1.0)
                # Add slight random noise per year
                year_noise = np.random.normal(1.0, 0.02)

                for age in ages:
                    base = get_base_rate(age)
                    # Add small age-specific noise
                    noise = np.random.normal(1.0, 0.05)
                    rate = base * mult * year_noise * noise
                    rate = max(0.001, rate)  # Ensure positive

                    records.append(
                        {
                            "year": year,
                            "race_ethnicity": race,
                            "age": age,
                            "fertility_rate": round(rate, 5),
                        }
                    )

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock fertility records")
        return df

    def fetch_mortality_rates(self) -> pd.DataFrame:
        """Fetch mortality rates (Placeholder)."""
        raise NotImplementedError("Mortality fetching not yet implemented")
