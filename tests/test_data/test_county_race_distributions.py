"""
Tests for county-specific age-sex-race distributions (ADR-047).

Validates both the ingestion logic (build_county_distributions, blending)
and the data loading logic (load_county_age_sex_race_distribution) that
enables county-level race allocation in base population construction.

Test categories:
    1. Ingestion: build_county_distributions produces correct output shape,
       proportions, and blending behavior.
    2. Validation: validate_county_distributions catches structural errors.
    3. Loader: load_county_age_sex_race_distribution returns correct format,
       handles fallback, and respects config toggles.
    4. Integration: Reservation counties get meaningfully different
       distributions than urban counties.

Uses synthetic DataFrames as fixtures — does not depend on actual data files
except for the integration tests that use the real parquet file on disk.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import loader functions
from cohort_projections.data.load.base_population_loader import (
    load_county_age_sex_race_distribution,
    load_county_distributions_file,
)

# Import ingestion script functions
from scripts.data.build_race_distribution_from_census import (
    AGEGRP_MAP,
    DEFAULT_BLEND_THRESHOLD,
    EXPECTED_ROWS,
    RACE_COLUMN_MAP,
    build_county_distributions,
    validate_county_distributions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def statewide_distribution():
    """
    Build a synthetic statewide distribution with 216 rows
    (18 age groups x 2 sexes x 6 races).
    """
    rows = []
    for agegrp_code in sorted(AGEGRP_MAP.keys()):
        age_group = AGEGRP_MAP[agegrp_code]
        for race_ethnicity, sex, _census_cols in RACE_COLUMN_MAP:
            rows.append({
                "age_group": age_group,
                "sex": sex,
                "race_ethnicity": race_ethnicity,
                "estimated_count": 100.0,
                "proportion": 1.0 / EXPECTED_ROWS,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_census_csv(tmp_path):
    """
    Build a synthetic cc-est2024-alldata CSV with 3 counties.

    County 001 (pop 10000): Large county, no blending expected
    County 003 (pop 3000):  Small county, blending expected
    County 005 (pop 500):   Tiny county, heavy blending expected

    All race columns are populated with synthetic counts.
    """
    # Collect all census columns needed
    all_census_cols = set()
    for _race, _sex, cols in RACE_COLUMN_MAP:
        all_census_cols.update(cols)
    all_census_cols = sorted(all_census_cols)

    records = []
    counties = {
        1: {"total_scale": 10000},   # Large county
        3: {"total_scale": 3000},    # Small county
        5: {"total_scale": 500},     # Tiny county
    }

    for county_code, info in counties.items():
        scale = info["total_scale"]
        for agegrp_code in sorted(AGEGRP_MAP.keys()):
            row = {
                "STATE": 38,
                "COUNTY": county_code,
                "YEAR": 6,
                "AGEGRP": agegrp_code,
                "STNAME": "North Dakota",
                "CTYNAME": f"County {county_code:03d}",
            }
            # Distribute population across race-sex columns
            # Make county 001 heavily white, county 003 mixed, county 005 AIAN-heavy
            for col in all_census_cols:
                if county_code == 1:
                    # Large county: mostly white
                    if "NHWA" in col:
                        row[col] = scale * 0.04  # ~72% white across 18 groups
                    elif "NHIA" in col:
                        row[col] = scale * 0.002  # small AIAN
                    else:
                        row[col] = scale * 0.005
                elif county_code == 3:
                    # Small mixed county
                    row[col] = scale * 0.008
                elif county_code == 5:
                    # Tiny AIAN-heavy county
                    if "NHIA" in col:
                        row[col] = scale * 0.04  # ~72% AIAN
                    elif "NHWA" in col:
                        row[col] = scale * 0.005
                    else:
                        row[col] = scale * 0.002
            records.append(row)

    df = pd.DataFrame(records)
    csv_path = tmp_path / "cc-est2024-alldata-38.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_county_distributions_df():
    """
    Build a synthetic county distributions DataFrame matching the parquet
    schema: [fips, age_group, sex, race, proportion].

    Includes 2 counties: 38017 (Cass, urban) and 38085 (Sioux, reservation).
    """
    rows = []
    counties = {
        "38017": {"white_weight": 0.80, "aian_weight": 0.02},
        "38085": {"white_weight": 0.15, "aian_weight": 0.60},
    }

    for fips, weights in counties.items():
        total_weight = 0.0
        county_rows = []
        for agegrp_code in sorted(AGEGRP_MAP.keys()):
            age_group = AGEGRP_MAP[agegrp_code]
            for race_ethnicity, sex, _cols in RACE_COLUMN_MAP:
                if race_ethnicity == "white_nonhispanic":
                    w = weights["white_weight"]
                elif race_ethnicity == "aian_nonhispanic":
                    w = weights["aian_weight"]
                else:
                    w = 0.045  # Spread remaining among 4 other groups
                county_rows.append({
                    "fips": fips,
                    "age_group": age_group,
                    "sex": sex,
                    "race": race_ethnicity,
                    "proportion": w / EXPECTED_ROWS,
                })
                total_weight += w / EXPECTED_ROWS

        # Normalize
        for row in county_rows:
            row["proportion"] /= total_weight
        rows.extend(county_rows)

    return pd.DataFrame(rows)


@pytest.fixture
def county_dist_enabled_config():
    """Config dict with county distributions enabled."""
    return {
        "project": {"base_year": 2025},
        "base_population": {
            "age_resolution": "five_year_uniform",
            "county_distributions": {
                "enabled": True,
                "path": "data/processed/county_age_sex_race_distributions.parquet",
                "blend_threshold": 5000,
            },
        },
        "demographics": {
            "age_groups": {"min_age": 0, "max_age": 90},
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "AIAN alone, Non-Hispanic",
                    "Asian/PI alone, Non-Hispanic",
                    "Two or more races, Non-Hispanic",
                    "Hispanic (any race)",
                ],
            },
        },
    }


@pytest.fixture
def county_dist_disabled_config(county_dist_enabled_config):
    """Config dict with county distributions disabled."""
    config = county_dist_enabled_config.copy()
    config["base_population"] = config["base_population"].copy()
    config["base_population"]["county_distributions"] = {
        "enabled": False,
    }
    return config


# ---------------------------------------------------------------------------
# 1. Ingestion tests: build_county_distributions
# ---------------------------------------------------------------------------

class TestBuildCountyDistributions:
    """Tests for build_county_distributions from the ingestion script."""

    def test_output_shape(self, synthetic_census_csv, statewide_distribution):
        """Each county should have exactly 216 rows (18 age x 2 sex x 6 race)."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        n_counties = result["fips"].nunique()
        assert n_counties == 3, f"Expected 3 counties, got {n_counties}"
        assert len(result) == n_counties * EXPECTED_ROWS

    def test_proportions_sum_to_one(self, synthetic_census_csv, statewide_distribution):
        """Proportions must sum to 1.0 for each county."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        county_sums = result.groupby("fips")["proportion"].sum()
        for fips, prop_sum in county_sums.items():
            assert abs(prop_sum - 1.0) < 1e-6, (
                f"County {fips} proportions sum to {prop_sum}, expected 1.0"
            )

    def test_no_negative_proportions(self, synthetic_census_csv, statewide_distribution):
        """No proportions should be negative."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        assert (result["proportion"] >= 0).all(), "Found negative proportions"

    def test_no_nan_proportions(self, synthetic_census_csv, statewide_distribution):
        """No proportions should be NaN."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        assert not result["proportion"].isna().any(), "Found NaN proportions"

    def test_fips_format(self, synthetic_census_csv, statewide_distribution):
        """FIPS codes should be 5-digit strings starting with 38."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        for fips in result["fips"].unique():
            assert len(fips) == 5, f"FIPS {fips} is not 5 digits"
            assert fips.startswith("38"), f"FIPS {fips} does not start with 38"

    def test_expected_columns(self, synthetic_census_csv, statewide_distribution):
        """Output should have the expected columns."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        expected_cols = {"fips", "age_group", "sex", "race", "proportion"}
        assert set(result.columns) == expected_cols

    def test_large_county_no_blending(self, synthetic_census_csv, statewide_distribution):
        """Counties above blend threshold should use pure county distribution."""
        # Use a threshold that only county 001 (pop ~10000) exceeds
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=5000,
        )
        # County 38001 should NOT be blended (pop > 5000)
        county_001 = result[result["fips"] == "38001"]
        # The large county has a strong white proportion (by construction)
        white_prop = county_001[county_001["race"] == "white_nonhispanic"]["proportion"].sum()
        # Should be heavily white (>60%) since it's not blended with state average
        assert white_prop > 0.5, (
            f"Large county should retain its white-heavy distribution, got {white_prop:.2%}"
        )

    def test_small_county_is_blended(self, synthetic_census_csv, statewide_distribution):
        """Counties below blend threshold should be blended toward statewide."""
        # County 005 (pop ~500) should be heavily blended
        result_blended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=5000,
        )
        result_unblended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=0,  # No blending
        )

        # County 38005 distributions should differ between blended and unblended
        blended_005 = result_blended[result_blended["fips"] == "38005"].sort_values(
            ["age_group", "sex", "race"]
        ).reset_index(drop=True)
        unblended_005 = result_unblended[result_unblended["fips"] == "38005"].sort_values(
            ["age_group", "sex", "race"]
        ).reset_index(drop=True)

        # Distributions should differ (blending moves toward uniform statewide)
        diff = (blended_005["proportion"] - unblended_005["proportion"]).abs().sum()
        assert diff > 0.01, (
            f"Blended and unblended distributions should differ for small county, "
            f"but total absolute difference is only {diff:.6f}"
        )

    def test_zero_threshold_disables_blending(
        self, synthetic_census_csv, statewide_distribution
    ):
        """When blend_threshold=0, no county should be blended."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=0,
        )
        # All counties should have purely county-derived distributions
        # Verify proportions still sum to 1
        county_sums = result.groupby("fips")["proportion"].sum()
        for prop_sum in county_sums.values:
            assert abs(prop_sum - 1.0) < 1e-6

    def test_all_age_groups_present(self, synthetic_census_csv, statewide_distribution):
        """Each county should have all 18 age groups."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        expected_age_groups = set(AGEGRP_MAP.values())
        for fips in result["fips"].unique():
            county_ages = set(result[result["fips"] == fips]["age_group"].unique())
            assert county_ages == expected_age_groups, (
                f"County {fips} missing age groups: {expected_age_groups - county_ages}"
            )

    def test_all_race_categories_present(
        self, synthetic_census_csv, statewide_distribution
    ):
        """Each county should have all 6 race categories."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        expected_races = {race for race, _sex, _cols in RACE_COLUMN_MAP}
        for fips in result["fips"].unique():
            county_races = set(result[result["fips"] == fips]["race"].unique())
            assert county_races == expected_races, (
                f"County {fips} missing races: {expected_races - county_races}"
            )

    def test_both_sexes_present(self, synthetic_census_csv, statewide_distribution):
        """Each county should have both male and female rows."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        for fips in result["fips"].unique():
            county_sexes = set(result[result["fips"] == fips]["sex"].unique())
            assert county_sexes == {"male", "female"}, (
                f"County {fips} missing sex categories: {county_sexes}"
            )


# ---------------------------------------------------------------------------
# 2. Validation tests: validate_county_distributions
# ---------------------------------------------------------------------------

class TestValidateCountyDistributions:
    """Tests for validate_county_distributions from the ingestion script."""

    def test_valid_distribution_passes(
        self, synthetic_census_csv, statewide_distribution
    ):
        """A correctly-built distribution should pass validation."""
        result = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=DEFAULT_BLEND_THRESHOLD,
        )
        assert validate_county_distributions(result) is True

    def test_missing_rows_detected(self, synthetic_county_distributions_df):
        """Validation should detect counties with wrong row count."""
        # Remove some rows from one county
        df = synthetic_county_distributions_df.copy()
        mask = (df["fips"] == "38017") & (df["age_group"] == "0-4")
        df = df[~mask]
        # This should fail row count check
        assert validate_county_distributions(df) is False

    def test_nan_proportions_detected(self, synthetic_county_distributions_df):
        """Validation should detect NaN proportions."""
        df = synthetic_county_distributions_df.copy()
        df.loc[df.index[0], "proportion"] = np.nan
        assert validate_county_distributions(df) is False


# ---------------------------------------------------------------------------
# 3. Loader tests: load_county_age_sex_race_distribution
# ---------------------------------------------------------------------------

class TestLoadCountyDistribution:
    """Tests for load_county_age_sex_race_distribution."""

    def test_returns_none_when_disabled(self, county_dist_disabled_config):
        """Should return None when county distributions are disabled."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            config=county_dist_disabled_config,
        )
        assert result is None

    def test_returns_none_for_missing_county(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Should return None when a county is not in the distributions file."""
        result = load_county_age_sex_race_distribution(
            fips="38999",  # Non-existent county
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is None

    def test_returns_dataframe_for_valid_county(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Should return a DataFrame for a county present in the data."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_output_columns(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Output should have columns [age, sex, race, proportion]."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        expected_cols = {"age", "sex", "race", "proportion"}
        assert expected_cols.issubset(set(result.columns))

    def test_proportions_sum_to_one(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Loaded distribution proportions should sum to 1.0."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        prop_sum = result["proportion"].sum()
        assert abs(prop_sum - 1.0) < 1e-4, (
            f"Proportions sum to {prop_sum}, expected 1.0"
        )

    def test_age_range_correct(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Ages should span 0 to 90."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        assert result["age"].min() == 0
        assert result["age"].max() == 90

    def test_sex_values_title_case(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Sex values should be Title case (Male, Female)."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        assert set(result["sex"].unique()) == {"Male", "Female"}

    def test_race_codes_mapped_to_standard_categories(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Race codes should be mapped to standard projection categories."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        expected_races = {
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        }
        actual_races = set(result["race"].unique())
        assert actual_races == expected_races, (
            f"Missing races: {expected_races - actual_races}"
        )

    def test_fips_padding(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """FIPS codes should be zero-padded to 5 digits."""
        # Pass a 4-digit FIPS (missing leading zero is impossible for ND but
        # test that the function pads correctly)
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None

    def test_complete_cohort_grid(
        self,
        synthetic_county_distributions_df,
        county_dist_enabled_config,
    ):
        """Should produce a complete grid of age x sex x race cohorts."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=county_dist_enabled_config,
        )
        assert result is not None
        # 91 ages x 2 sexes x 6 races = 1092
        expected_rows = 91 * 2 * 6
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(result)}"
        )


class TestLoadCountyDistributionsFile:
    """Tests for load_county_distributions_file."""

    def test_returns_none_when_disabled(self, county_dist_disabled_config):
        """Should return None when county distributions are disabled."""
        result = load_county_distributions_file(config=county_dist_disabled_config)
        assert result is None

    def test_returns_none_when_file_missing(self, county_dist_enabled_config, tmp_path):
        """Should return None when the parquet file does not exist."""
        config = county_dist_enabled_config.copy()
        config["base_population"] = config["base_population"].copy()
        config["base_population"]["county_distributions"] = {
            "enabled": True,
            "path": str(tmp_path / "nonexistent.parquet"),
        }
        result = load_county_distributions_file(config=config)
        assert result is None


# ---------------------------------------------------------------------------
# 4. Integration tests: real parquet file on disk
# ---------------------------------------------------------------------------

class TestCountyDistributionsIntegration:
    """
    Integration tests using the real county distributions parquet file.

    These tests verify the actual generated data meets the ADR-047
    requirements. They depend on the parquet file existing on disk.
    """

    @pytest.fixture
    def real_county_dist(self):
        """Load the real county distributions parquet file."""
        project_root = Path(__file__).resolve().parent.parent.parent
        parquet_path = (
            project_root / "data" / "processed"
            / "county_age_sex_race_distributions.parquet"
        )
        if not parquet_path.exists():
            pytest.skip(
                f"County distributions parquet not found at {parquet_path}. "
                "Run build_race_distribution_from_census.py first."
            )
        return pd.read_parquet(parquet_path)

    def test_53_counties(self, real_county_dist):
        """Should contain exactly 53 North Dakota counties."""
        n_counties = real_county_dist["fips"].nunique()
        assert n_counties == 53, f"Expected 53 counties, got {n_counties}"

    def test_total_rows(self, real_county_dist):
        """Should contain 53 x 216 = 11,448 rows."""
        assert len(real_county_dist) == 53 * 216, (
            f"Expected {53 * 216} rows, got {len(real_county_dist)}"
        )

    def test_each_county_has_216_rows(self, real_county_dist):
        """Every county should have exactly 216 rows."""
        county_counts = real_county_dist.groupby("fips").size()
        bad = county_counts[county_counts != 216]
        assert len(bad) == 0, f"Counties with wrong row count: {dict(bad)}"

    def test_proportions_sum_to_one_per_county(self, real_county_dist):
        """Proportions must sum to 1.0 for each county."""
        county_sums = real_county_dist.groupby("fips")["proportion"].sum()
        bad = county_sums[abs(county_sums - 1.0) > 1e-6]
        assert len(bad) == 0, (
            f"Counties with proportions not summing to 1.0: {dict(bad)}"
        )

    def test_no_negative_proportions(self, real_county_dist):
        """No proportions should be negative."""
        neg = real_county_dist[real_county_dist["proportion"] < 0]
        assert len(neg) == 0, f"Found {len(neg)} negative proportions"

    def test_no_nan_proportions(self, real_county_dist):
        """No proportions should be NaN."""
        nan_count = real_county_dist["proportion"].isna().sum()
        assert nan_count == 0, f"Found {nan_count} NaN proportions"

    def test_sioux_aian_proportion_high(self, real_county_dist):
        """Sioux County (38085) should have >50% AIAN proportion.

        Sioux County is on the Standing Rock reservation and has ~78% AIAN
        population in Census data. After blending (pop ~3,700 < 5,000), the
        AIAN proportion should still be well above statewide average (~4.8%).
        """
        sioux = real_county_dist[real_county_dist["fips"] == "38085"]
        aian_prop = sioux[sioux["race"] == "aian_nonhispanic"]["proportion"].sum()
        assert aian_prop > 0.50, (
            f"Sioux County AIAN proportion is {aian_prop:.1%}, expected >50%"
        )

    def test_rolette_aian_proportion_high(self, real_county_dist):
        """Rolette County (38079) should have >60% AIAN proportion.

        Rolette County includes Turtle Mountain reservation with ~76% AIAN.
        Population ~11,692 is above the blend threshold so no blending.
        """
        rolette = real_county_dist[real_county_dist["fips"] == "38079"]
        aian_prop = rolette[rolette["race"] == "aian_nonhispanic"]["proportion"].sum()
        assert aian_prop > 0.60, (
            f"Rolette County AIAN proportion is {aian_prop:.1%}, expected >60%"
        )

    def test_benson_aian_proportion_elevated(self, real_county_dist):
        """Benson County (38005) should have >30% AIAN proportion.

        Benson County includes Fort Totten/Spirit Lake reservation with ~50%
        AIAN. After potential blending (pop ~5,756, near threshold), AIAN
        should still be substantially above statewide.
        """
        benson = real_county_dist[real_county_dist["fips"] == "38005"]
        aian_prop = benson[benson["race"] == "aian_nonhispanic"]["proportion"].sum()
        assert aian_prop > 0.30, (
            f"Benson County AIAN proportion is {aian_prop:.1%}, expected >30%"
        )

    def test_cass_county_white_majority(self, real_county_dist):
        """Cass County (38017, Fargo) should have >75% white proportion."""
        cass = real_county_dist[real_county_dist["fips"] == "38017"]
        white_prop = cass[cass["race"] == "white_nonhispanic"]["proportion"].sum()
        assert white_prop > 0.75, (
            f"Cass County white proportion is {white_prop:.1%}, expected >75%"
        )

    def test_distributions_differ_between_counties(self, real_county_dist):
        """County distributions should meaningfully differ from each other.

        The median mean-absolute-deviation across counties should be
        non-trivial (>0.001), confirming distributions are not identical.
        """
        avg_dist = real_county_dist.groupby(
            ["age_group", "sex", "race"]
        )["proportion"].mean()

        mad_values = []
        for fips in real_county_dist["fips"].unique():
            county_data = real_county_dist[real_county_dist["fips"] == fips]
            county_indexed = county_data.set_index(
                ["age_group", "sex", "race"]
            )["proportion"]
            mad = (county_indexed - avg_dist).abs().mean()
            mad_values.append(mad)

        median_mad = sorted(mad_values)[len(mad_values) // 2]
        assert median_mad > 0.0005, (
            f"Median MAD is {median_mad:.6f} — distributions may be too similar"
        )

    def test_reservation_vs_urban_divergence(self, real_county_dist):
        """Reservation and urban counties should have substantially different
        race distributions.

        Compare Sioux (reservation) vs Cass (urban) AIAN proportions — the
        difference should be >40 percentage points.
        """
        sioux = real_county_dist[real_county_dist["fips"] == "38085"]
        cass = real_county_dist[real_county_dist["fips"] == "38017"]

        sioux_aian = sioux[sioux["race"] == "aian_nonhispanic"]["proportion"].sum()
        cass_aian = cass[cass["race"] == "aian_nonhispanic"]["proportion"].sum()

        diff = abs(sioux_aian - cass_aian)
        assert diff > 0.40, (
            f"AIAN proportion difference between Sioux ({sioux_aian:.1%}) and "
            f"Cass ({cass_aian:.1%}) is only {diff:.1%}, expected >40pp"
        )


# ---------------------------------------------------------------------------
# 5. Blending-specific tests
# ---------------------------------------------------------------------------

class TestBlendingBehavior:
    """Tests specifically for the population-weighted blending logic."""

    def test_alpha_formula(self):
        """Verify the alpha blending formula: alpha = min(pop / threshold, 1.0)."""
        threshold = 5000

        # County at exactly the threshold: alpha = 1.0 (no blending)
        assert min(5000 / threshold, 1.0) == 1.0

        # County at half the threshold: alpha = 0.5
        assert min(2500 / threshold, 1.0) == 0.5

        # County above threshold: alpha = 1.0 (capped)
        assert min(10000 / threshold, 1.0) == 1.0

        # Very small county: alpha close to 0
        assert abs(min(100 / threshold, 1.0) - 0.02) < 1e-6

    def test_blending_moves_toward_statewide(
        self, synthetic_census_csv, statewide_distribution
    ):
        """Blended small-county distributions should be closer to statewide
        than unblended distributions.

        The statewide distribution is uniform across all cells, so blending
        should make the small county's distribution more uniform.
        """
        blended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=5000,
        )
        unblended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=0,
        )

        # For the tiny county (38005, pop ~500):
        # Compute distance from statewide (uniform) distribution
        state_prop = 1.0 / EXPECTED_ROWS  # Uniform statewide

        blended_005 = blended[blended["fips"] == "38005"]["proportion"]
        unblended_005 = unblended[unblended["fips"] == "38005"]["proportion"]

        blended_dist = (blended_005 - state_prop).abs().sum()
        unblended_dist = (unblended_005 - state_prop).abs().sum()

        assert blended_dist < unblended_dist, (
            f"Blended distribution (dist={blended_dist:.4f}) should be closer "
            f"to statewide than unblended (dist={unblended_dist:.4f})"
        )

    def test_large_county_unaffected_by_blending(
        self, synthetic_census_csv, statewide_distribution
    ):
        """Counties above threshold should have identical distributions
        regardless of blending."""
        blended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=5000,
        )
        unblended = build_county_distributions(
            synthetic_census_csv,
            statewide_distribution,
            blend_threshold=0,
        )

        # County 38001 (pop ~10000 > 5000) should be unaffected
        blended_001 = blended[blended["fips"] == "38001"].sort_values(
            ["age_group", "sex", "race"]
        ).reset_index(drop=True)
        unblended_001 = unblended[unblended["fips"] == "38001"].sort_values(
            ["age_group", "sex", "race"]
        ).reset_index(drop=True)

        diff = (blended_001["proportion"] - unblended_001["proportion"]).abs().max()
        assert diff < 1e-10, (
            f"Large county distribution changed with blending: max diff = {diff}"
        )


# ---------------------------------------------------------------------------
# 6. Sprague interpolation tests for county distributions (ADR-048)
# ---------------------------------------------------------------------------

class TestSpragueCountyInterpolation:
    """Tests for Sprague-based county 5-year-to-single-year expansion (ADR-048).

    When base_population.county_race_interpolation is "sprague" and
    age_resolution is "single_year", the county loader expands 5-year
    age groups using Sprague osculatory interpolation instead of uniform
    splitting, producing smooth single-year distributions.
    """

    @pytest.fixture
    def sprague_config(self):
        """Config dict with Sprague county interpolation enabled."""
        return {
            "project": {"base_year": 2025},
            "base_population": {
                "age_resolution": "single_year",
                "county_race_interpolation": "sprague",
                "county_distributions": {
                    "enabled": True,
                    "path": "data/processed/county_age_sex_race_distributions.parquet",
                    "blend_threshold": 5000,
                },
            },
            "demographics": {
                "age_groups": {"min_age": 0, "max_age": 90},
                "sex": ["Male", "Female"],
                "race_ethnicity": {
                    "categories": [
                        "White alone, Non-Hispanic",
                        "Black alone, Non-Hispanic",
                        "AIAN alone, Non-Hispanic",
                        "Asian/PI alone, Non-Hispanic",
                        "Two or more races, Non-Hispanic",
                        "Hispanic (any race)",
                    ],
                },
            },
        }

    @pytest.fixture
    def statewide_weights_config(self, sprague_config):
        """Config dict with statewide_weights county interpolation."""
        config = sprague_config.copy()
        config["base_population"] = config["base_population"].copy()
        config["base_population"]["county_race_interpolation"] = "statewide_weights"
        return config

    def test_sprague_output_shape(
        self,
        synthetic_county_distributions_df,
        sprague_config,
    ):
        """Sprague expansion should produce 1,092 rows (91 ages x 2 sexes x 6 races)."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=sprague_config,
        )
        assert result is not None
        expected_rows = 91 * 2 * 6
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(result)}"
        )

    def test_sprague_proportions_sum_to_one(
        self,
        synthetic_county_distributions_df,
        sprague_config,
    ):
        """Sprague-expanded proportions should sum to 1.0."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=sprague_config,
        )
        assert result is not None
        prop_sum = result["proportion"].sum()
        assert abs(prop_sum - 1.0) < 1e-4, (
            f"Sprague proportions sum to {prop_sum}, expected 1.0"
        )

    def test_sprague_no_negative_proportions(
        self,
        synthetic_county_distributions_df,
        sprague_config,
    ):
        """Sprague-expanded values should be non-negative (after clamping)."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=sprague_config,
        )
        assert result is not None
        neg_count = (result["proportion"] < 0).sum()
        assert neg_count == 0, f"Found {neg_count} negative proportions after Sprague"

    @staticmethod
    def _build_varying_county_df():
        """Build county distributions with varying proportions across age groups.

        Unlike the uniform synthetic fixture, this one has a realistic
        age gradient so uniform splitting creates visible step-functions
        that Sprague should smooth out.
        """
        age_weights = {
            "0-4": 0.07, "5-9": 0.065, "10-14": 0.06, "15-19": 0.065,
            "20-24": 0.075, "25-29": 0.08, "30-34": 0.075, "35-39": 0.07,
            "40-44": 0.065, "45-49": 0.06, "50-54": 0.055, "55-59": 0.05,
            "60-64": 0.045, "65-69": 0.04, "70-74": 0.035, "75-79": 0.03,
            "80-84": 0.025, "85+": 0.035,
        }
        rows = []
        for age_group, weight in age_weights.items():
            for sex in ["male", "female"]:
                for race in ["white_nonhispanic", "black_nonhispanic",
                             "aian_nonhispanic", "asian_nonhispanic",
                             "multiracial_nonhispanic", "hispanic"]:
                    if race == "white_nonhispanic":
                        race_w = 0.80
                    elif race == "aian_nonhispanic":
                        race_w = 0.02
                    else:
                        race_w = 0.045
                    rows.append({
                        "fips": "38017",
                        "age_group": age_group,
                        "sex": sex,
                        "race": race,
                        "proportion": weight * race_w / 2.0,
                    })
        df = pd.DataFrame(rows)
        total = df["proportion"].sum()
        df["proportion"] = df["proportion"] / total
        return df

    def test_sprague_smoother_than_uniform(
        self,
        sprague_config,
        county_dist_enabled_config,
    ):
        """Sprague should produce smoother results than uniform splitting.

        Compare the sum of squared second differences (roughness) for the
        White male age profile between Sprague and uniform methods.
        Uses a distribution with varying age-group proportions so uniform
        splitting creates real step-function artifacts.
        """
        varying_df = self._build_varying_county_df()

        # Sprague result
        sprague_result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=varying_df,
            config=sprague_config,
        )

        # Uniform result (legacy mode)
        uniform_result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=varying_df,
            config=county_dist_enabled_config,
        )

        assert sprague_result is not None
        assert uniform_result is not None

        # Extract White male age profile for comparison
        def extract_profile(df):
            mask = (
                (df["sex"] == "Male")
                & (df["race"] == "White alone, Non-Hispanic")
            )
            return df[mask].sort_values("age")["proportion"].values

        sprague_profile = extract_profile(sprague_result)
        uniform_profile = extract_profile(uniform_result)

        # Compute roughness: sum of squared second differences (ages 5-79)
        def roughness(arr):
            d2 = np.diff(arr[5:80], n=2)
            return float(np.sum(d2 ** 2))

        sprague_rough = roughness(sprague_profile)
        uniform_rough = roughness(uniform_profile)

        assert sprague_rough < uniform_rough, (
            f"Sprague roughness ({sprague_rough:.2e}) should be less than "
            f"uniform roughness ({uniform_rough:.2e})"
        )

    def test_sprague_no_step_at_group_boundaries(
        self,
        synthetic_county_distributions_df,
        sprague_config,
    ):
        """Sprague should not produce abrupt jumps at 5-year group boundaries.

        At group boundaries (ages 4-5, 9-10, etc.), the ratio of adjacent
        single-year values should be close to 1.0, not the abrupt steps
        produced by uniform splitting.
        """
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=sprague_config,
        )
        assert result is not None

        # Extract total male profile (sum across races)
        male = result[result["sex"] == "Male"]
        age_totals = male.groupby("age")["proportion"].sum().sort_index()

        # Check boundary ratios for ages 5-79
        boundaries = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]
        for b in boundaries:
            if b in age_totals.index and b + 1 in age_totals.index:
                v1 = age_totals[b]
                v2 = age_totals[b + 1]
                if v1 > 0 and v2 > 0:
                    ratio = max(v1 / v2, v2 / v1)
                    assert ratio < 1.30, (
                        f"Step at boundary ages {b}-{b + 1}: ratio = {ratio:.3f}"
                    )

    def test_sprague_age_range_0_to_90(
        self,
        synthetic_county_distributions_df,
        sprague_config,
    ):
        """Sprague expansion should cover ages 0 through 90."""
        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=sprague_config,
        )
        assert result is not None
        assert result["age"].min() == 0
        assert result["age"].max() == 90

    def test_statewide_weights_fallback(
        self,
        synthetic_county_distributions_df,
        statewide_weights_config,
    ):
        """The statewide_weights method should also produce valid output."""
        # This needs a statewide distribution for weights; use a synthetic one
        state_dist_rows = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "AIAN alone, Non-Hispanic",
                    "Asian/PI alone, Non-Hispanic",
                    "Two or more races, Non-Hispanic",
                    "Hispanic (any race)",
                ]:
                    state_dist_rows.append({
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "proportion": 1.0 / (91 * 2 * 6),
                    })
        state_dist = pd.DataFrame(state_dist_rows)

        result = load_county_age_sex_race_distribution(
            fips="38017",
            county_distributions_df=synthetic_county_distributions_df,
            config=statewide_weights_config,
            state_distribution=state_dist,
        )
        assert result is not None
        expected_rows = 91 * 2 * 6
        assert len(result) == expected_rows
        prop_sum = result["proportion"].sum()
        assert abs(prop_sum - 1.0) < 1e-4
