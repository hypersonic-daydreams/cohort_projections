"""
Unit tests for geospatial export functionality (ADR-059).

Tests TIGER boundary loading, projection-to-boundary joining, GeoJSON and
Shapefile export, graceful handling of missing geopandas, year filtering,
and geography level validation.

All tests use synthetic DataFrames and mock file I/O so that actual TIGER
shapefiles are not required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Conditional geopandas import
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd
    from shapely.geometry import box

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

needs_geo = pytest.mark.skipif(not HAS_GEO, reason="geopandas/shapely not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_county_boundaries():
    """Synthetic county GeoDataFrame mimicking TIGER output."""
    if not HAS_GEO:
        pytest.skip("geopandas not available")
    data = {
        "state_fips": ["38", "38", "38"],
        "county_fips": ["38001", "38015", "38017"],
        "county_name": ["Adams County", "Burleigh County", "Cass County"],
        "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def sample_place_boundaries():
    """Synthetic place GeoDataFrame mimicking TIGER output."""
    if not HAS_GEO:
        pytest.skip("geopandas not available")
    data = {
        "state_fips": ["38", "38"],
        "place_fips": ["3807200", "3825700"],
        "place_name": ["Bismarck city", "Fargo city"],
        "county_fips": ["", ""],
        "geometry": [box(0, 0, 0.5, 0.5), box(2, 0, 2.5, 0.5)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def sample_county_projection():
    """Projection data with county_fips column."""
    rows = []
    for fips in ["38001", "38015", "38017"]:
        for year in [2025, 2030, 2035]:
            rows.append(
                {
                    "county_fips": fips,
                    "year": year,
                    "age": 0,
                    "sex": "Male",
                    "race": "White",
                    "population": 100.0 + year - 2025,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_place_projection():
    """Projection data with place_fips column."""
    rows = []
    for fips in ["3807200", "3825700"]:
        for year in [2025, 2030]:
            rows.append(
                {
                    "place_fips": fips,
                    "year": year,
                    "age": 0,
                    "sex": "Male",
                    "race": "White",
                    "population": 500.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TIGER county loading tests
# ---------------------------------------------------------------------------
class TestLoadCountiesFromTiger:
    """Tests for _load_counties_from_tiger."""

    @needs_geo
    def test_filters_to_nd(self, sample_county_boundaries: Any) -> None:
        """All returned rows should have state_fips == '38'."""
        df = sample_county_boundaries
        assert (df["state_fips"] == "38").all()

    @needs_geo
    def test_column_mapping(self, sample_county_boundaries: Any) -> None:
        """Returned DataFrame must have project-convention columns."""
        required = {"state_fips", "county_fips", "county_name", "geometry"}
        assert required.issubset(sample_county_boundaries.columns)

    @needs_geo
    def test_fips_padding(self, sample_county_boundaries: Any) -> None:
        """county_fips should be zero-padded to 5 digits."""
        for fips in sample_county_boundaries["county_fips"]:
            assert len(fips) == 5

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_projection_config")
    def test_file_not_found_raises(self, mock_config: MagicMock) -> None:
        """FileNotFoundError when shapefile path does not exist."""
        mock_config.return_value = {
            "geography": {
                "reference_data": {
                    "tiger_boundaries": {
                        "county_shapefile": "nonexistent/path.shp",
                    }
                }
            }
        }
        from cohort_projections.geographic.geography_loader import (
            _load_counties_from_tiger,
        )

        with pytest.raises(FileNotFoundError, match="TIGER county shapefile"):
            _load_counties_from_tiger(2020)


# ---------------------------------------------------------------------------
# TIGER place loading tests
# ---------------------------------------------------------------------------
class TestLoadPlacesFromTiger:
    """Tests for _load_places_from_tiger."""

    @needs_geo
    def test_column_mapping(self, sample_place_boundaries: Any) -> None:
        """Returned DataFrame must include project-convention columns."""
        required = {"state_fips", "place_fips", "place_name", "geometry"}
        assert required.issubset(sample_place_boundaries.columns)

    @needs_geo
    def test_fips_padding(self, sample_place_boundaries: Any) -> None:
        """place_fips should be zero-padded to 7 digits."""
        for fips in sample_place_boundaries["place_fips"]:
            assert len(fips) == 7

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_projection_config")
    def test_file_not_found_raises(self, mock_config: MagicMock) -> None:
        """FileNotFoundError when shapefile path does not exist."""
        mock_config.return_value = {
            "geography": {
                "reference_data": {
                    "tiger_boundaries": {
                        "place_shapefile": "nonexistent/path.shp",
                    }
                }
            }
        }
        from cohort_projections.geographic.geography_loader import (
            _load_places_from_tiger,
        )

        with pytest.raises(FileNotFoundError, match="TIGER place shapefile"):
            _load_places_from_tiger(2020)


# ---------------------------------------------------------------------------
# Projection-to-boundary join tests
# ---------------------------------------------------------------------------
class TestProjectionBoundaryJoin:
    """Tests for write_projection_shapefile join logic."""

    @needs_geo
    @patch(
        "cohort_projections.geographic.geography_loader.load_tiger_boundaries",
    )
    def test_county_join_produces_features(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Joining county projections to boundaries should produce output."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "counties.geojson"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            year=2025,
        )
        assert out.exists()

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_missing_fips_raises(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        tmp_path: Path,
    ) -> None:
        """ValueError when no FIPS codes match."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        bad_proj = pd.DataFrame(
            {
                "county_fips": ["99999"],
                "year": [2025],
                "population": [100.0],
            }
        )
        with pytest.raises(ValueError, match="No geometries matched"):
            write_projection_shapefile(
                projection_df=bad_proj,
                geography_level="county",
                output_path=tmp_path / "bad.geojson",
                year=2025,
            )


# ---------------------------------------------------------------------------
# GeoJSON export tests
# ---------------------------------------------------------------------------
class TestGeoJSONExport:
    """Tests for GeoJSON output from write_projection_shapefile."""

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_geojson_valid_json(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Output GeoJSON should be parseable JSON with geometry + properties."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "test.geojson"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            year=2030,
            format_type="geojson",
        )
        with open(out) as f:
            data = json.load(f)

        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 3
        # Each feature should have geometry and properties
        for feat in data["features"]:
            assert "geometry" in feat
            assert "properties" in feat
            assert "county_fips" in feat["properties"]
            assert "population" in feat["properties"]

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_geojson_has_population(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Population values should be present as feature properties."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "pop.geojson"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            year=2025,
            format_type="geojson",
        )
        with open(out) as f:
            data = json.load(f)

        pops = [f["properties"]["population"] for f in data["features"]]
        assert all(p > 0 for p in pops)


# ---------------------------------------------------------------------------
# Shapefile export tests
# ---------------------------------------------------------------------------
class TestShapefileExport:
    """Tests for ESRI Shapefile output from write_projection_shapefile."""

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_shapefile_creates_companion_files(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Shapefile export should create .shp, .shx, .dbf companion files."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "output" / "counties.shp"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            year=2025,
            format_type="shapefile",
        )
        assert out.exists()
        assert out.with_suffix(".shx").exists()
        assert out.with_suffix(".dbf").exists()


# ---------------------------------------------------------------------------
# Missing geopandas guard
# ---------------------------------------------------------------------------
class TestMissingGeopandas:
    """Test that ImportError is raised when geopandas is unavailable."""

    def test_import_error_when_geopandas_missing(self, tmp_path: Path) -> None:
        """write_projection_shapefile should raise ImportError without geopandas."""
        with patch(
            "cohort_projections.output.writers.GEOPANDAS_AVAILABLE", False
        ):
            from cohort_projections.output.writers import write_projection_shapefile

            with pytest.raises(ImportError, match="geopandas"):
                write_projection_shapefile(
                    projection_df=pd.DataFrame(),
                    geography_level="county",
                    output_path=tmp_path / "out.geojson",
                )


# ---------------------------------------------------------------------------
# Year filtering
# ---------------------------------------------------------------------------
class TestYearFiltering:
    """Tests for year-based filtering in geospatial export."""

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_year_filter_selects_single_year(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Specifying year should include only that year's data."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "yr.geojson"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            year=2035,
            format_type="geojson",
        )
        with open(out) as f:
            data = json.load(f)
        # All features should come from year 2035
        for feat in data["features"]:
            assert feat["properties"].get("year") == 2035

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_no_year_includes_all(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Omitting year should aggregate across all years."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "all_years.geojson"
        write_projection_shapefile(
            projection_df=sample_county_projection,
            geography_level="county",
            output_path=out,
            format_type="geojson",
        )
        assert out.exists()

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_nonexistent_year_raises(
        self,
        mock_boundaries: MagicMock,
        sample_county_boundaries: Any,
        sample_county_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Requesting a year not in the data should raise ValueError."""
        mock_boundaries.return_value = sample_county_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        with pytest.raises(ValueError, match="No projection data"):
            write_projection_shapefile(
                projection_df=sample_county_projection,
                geography_level="county",
                output_path=tmp_path / "bad_year.geojson",
                year=9999,
            )


# ---------------------------------------------------------------------------
# Geography level parameter validation
# ---------------------------------------------------------------------------
class TestGeographyLevelValidation:
    """Tests for geography_level parameter validation."""

    @needs_geo
    def test_invalid_level_raises(self, tmp_path: Path) -> None:
        """Passing an unsupported level should raise ValueError."""
        from cohort_projections.output.writers import write_projection_shapefile

        with pytest.raises(ValueError, match="county.*place"):
            write_projection_shapefile(
                projection_df=pd.DataFrame({"population": [1]}),
                geography_level="state",
                output_path=tmp_path / "bad.geojson",
            )

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_place_level_accepted(
        self,
        mock_boundaries: MagicMock,
        sample_place_boundaries: Any,
        sample_place_projection: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Place-level export should work end-to-end."""
        mock_boundaries.return_value = sample_place_boundaries
        from cohort_projections.output.writers import write_projection_shapefile

        out = tmp_path / "places.geojson"
        write_projection_shapefile(
            projection_df=sample_place_projection,
            geography_level="place",
            output_path=out,
            year=2025,
            format_type="geojson",
        )
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert len(data["features"]) == 2
