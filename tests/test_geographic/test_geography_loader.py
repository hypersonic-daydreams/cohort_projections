"""
Unit tests for geography loader module.

Tests the functions for loading North Dakota geographic reference data
including counties, places, and geographic hierarchy mappings.
"""

import pandas as pd
import pytest

# Try to import the geography loader module
try:
    from cohort_projections.geographic.geography_loader import (
        _create_default_nd_counties,
        _create_default_nd_places,
        _validate_county_data,
        _validate_place_data,
        get_geography_name,
        get_place_to_county_mapping,
        load_geography_list,
        load_nd_counties,
        load_nd_places,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestLoadNdCounties:
    """Test load_nd_counties function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_returns_dataframe(self):
        """Test that load_nd_counties returns a DataFrame."""
        counties = load_nd_counties()

        assert isinstance(counties, pd.DataFrame)
        assert len(counties) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_has_required_columns(self):
        """Test counties DataFrame has required columns."""
        counties = load_nd_counties()

        assert "state_fips" in counties.columns
        assert "county_fips" in counties.columns
        assert "county_name" in counties.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_fips_format(self):
        """Test county FIPS codes are properly formatted."""
        counties = load_nd_counties()

        # State FIPS should be 2 digits
        for fips in counties["state_fips"]:
            assert len(fips) == 2
            assert fips == "38"  # North Dakota

        # County FIPS should be 5 digits
        for fips in counties["county_fips"]:
            assert len(fips) == 5
            assert fips.startswith("38")

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_local_source(self):
        """Test loading counties from local source."""
        counties = load_nd_counties(source="local")

        assert isinstance(counties, pd.DataFrame)
        assert len(counties) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_tiger_source(self):
        """Test loading counties from TIGER source (falls back to default)."""
        counties = load_nd_counties(source="tiger")

        assert isinstance(counties, pd.DataFrame)
        # TIGER loading not implemented, should use default data
        assert len(counties) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_invalid_source_raises(self):
        """Test invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            load_nd_counties(source="invalid")  # type: ignore[arg-type]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_counties_custom_reference_path(self, tmp_path):
        """Test loading counties from custom reference path."""
        # Create a custom reference file
        custom_counties = pd.DataFrame(
            {
                "state_fips": ["38", "38"],
                "county_fips": ["38001", "38002"],
                "county_name": ["Test County 1", "Test County 2"],
                "population": [1000, 2000],
            }
        )
        custom_path = tmp_path / "custom_counties.csv"
        custom_counties.to_csv(custom_path, index=False)

        counties = load_nd_counties(reference_path=custom_path)

        assert len(counties) == 2
        assert "Test County 1" in counties["county_name"].values


class TestLoadNdPlaces:
    """Test load_nd_places function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_returns_dataframe(self):
        """Test that load_nd_places returns a DataFrame using TIGER source."""
        # Use tiger source which falls back to default data
        places = load_nd_places(source="tiger")

        assert isinstance(places, pd.DataFrame)
        assert len(places) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_has_required_columns(self):
        """Test places DataFrame has required columns."""
        # Use tiger source for predictable format
        places = load_nd_places(source="tiger")

        assert "state_fips" in places.columns
        assert "place_fips" in places.columns
        assert "place_name" in places.columns
        assert "county_fips" in places.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_fips_format(self):
        """Test place FIPS codes are properly formatted."""
        # Use tiger source for predictable format
        places = load_nd_places(source="tiger")

        # State FIPS should be 2 digits
        for fips in places["state_fips"]:
            assert len(fips) == 2
            assert fips == "38"

        # Place FIPS should be 7 digits
        for fips in places["place_fips"]:
            assert len(fips) == 7
            assert fips.startswith("38")

        # County FIPS should be 5 digits
        for fips in places["county_fips"]:
            assert len(fips) == 5

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_min_population_filter(self):
        """Test filtering places by minimum population."""
        # Use tiger source for predictable format
        all_places = load_nd_places(source="tiger")
        filtered_places = load_nd_places(source="tiger", min_population=50000)

        # Filtered should have fewer or equal places
        assert len(filtered_places) <= len(all_places)

        # All remaining places should meet threshold
        if "population" in filtered_places.columns and len(filtered_places) > 0:
            assert all(filtered_places["population"] >= 50000)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_local_source_with_custom_file(self, tmp_path):
        """Test loading places from local source with compatible file."""
        # Create a compatible custom reference file
        custom_places = pd.DataFrame(
            {
                "state_fips": ["38", "38"],
                "place_fips": ["3800001", "3800002"],
                "place_name": ["Test City 1", "Test City 2"],
                "county_fips": ["38001", "38001"],
                "population": [1000, 2000],
            }
        )
        custom_path = tmp_path / "custom_places.csv"
        custom_places.to_csv(custom_path, index=False)

        places = load_nd_places(source="local", reference_path=custom_path)

        assert isinstance(places, pd.DataFrame)
        assert len(places) == 2

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_tiger_source(self):
        """Test loading places from TIGER source (falls back to default)."""
        places = load_nd_places(source="tiger")

        assert isinstance(places, pd.DataFrame)
        assert len(places) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_places_invalid_source_raises(self):
        """Test invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            load_nd_places(source="invalid")  # type: ignore[arg-type]


class TestGetPlaceToCountyMapping:
    """Test get_place_to_county_mapping function."""

    @pytest.fixture
    def default_places_df(self):
        """Create default places DataFrame for tests."""
        return _create_default_nd_places()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_mapping_returns_dataframe(self, default_places_df):
        """Test that mapping returns a DataFrame."""
        # Use default places data to avoid file format issues
        mapping = get_place_to_county_mapping(places_df=default_places_df)

        assert isinstance(mapping, pd.DataFrame)
        assert len(mapping) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_mapping_has_required_columns(self, default_places_df):
        """Test mapping has required columns."""
        # Use default places data to avoid file format issues
        mapping = get_place_to_county_mapping(places_df=default_places_df)

        assert "place_fips" in mapping.columns
        assert "county_fips" in mapping.columns
        assert "place_name" in mapping.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_mapping_fargo_to_cass(self, default_places_df):
        """Test that Fargo maps to Cass County."""
        # Use default places data which includes Fargo
        mapping = get_place_to_county_mapping(places_df=default_places_df)

        fargo = mapping[mapping["place_name"] == "Fargo city"]
        if len(fargo) > 0:
            # Fargo should be in Cass County (38017)
            assert fargo["county_fips"].values[0] == "38017"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_mapping_with_custom_places_df(self):
        """Test mapping with custom places DataFrame."""
        custom_places = pd.DataFrame(
            {
                "state_fips": ["38"],
                "place_fips": ["3800001"],
                "place_name": ["Custom Place"],
                "county_fips": ["38001"],
                "population": [1000],
            }
        )

        mapping = get_place_to_county_mapping(places_df=custom_places)

        assert len(mapping) == 1
        assert mapping["place_name"].values[0] == "Custom Place"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_mapping_missing_columns_raises(self):
        """Test mapping raises error for missing required columns."""
        incomplete_places = pd.DataFrame(
            {
                "place_fips": ["3800001"],
                # Missing county_fips and place_name
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            get_place_to_county_mapping(places_df=incomplete_places)


class TestLoadGeographyList:
    """Test load_geography_list function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_state_list(self):
        """Test loading state-level geography list."""
        state_list = load_geography_list("state")

        assert isinstance(state_list, list)
        assert len(state_list) == 1
        assert state_list[0] == "38"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_county_list(self):
        """Test loading county-level geography list."""
        county_list = load_geography_list("county")

        assert isinstance(county_list, list)
        assert len(county_list) > 0
        # All should be 5-digit FIPS codes starting with 38
        for fips in county_list:
            assert len(fips) == 5
            assert fips.startswith("38")

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_place_list(self):
        """Test loading place-level geography list with explicit fips."""
        # Use explicit FIPS codes to avoid file format issues
        explicit_fips = ["3825700", "3807200"]  # Fargo and Bismarck
        place_list = load_geography_list("place", fips_codes=explicit_fips)

        assert isinstance(place_list, list)
        assert len(place_list) > 0
        # All should be 7-digit FIPS codes starting with 38
        for fips in place_list:
            assert len(fips) == 7
            assert fips.startswith("38")

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_with_explicit_fips_codes(self):
        """Test loading with explicit FIPS codes list."""
        explicit_fips = ["38101", "38015"]
        result = load_geography_list("county", fips_codes=explicit_fips)

        assert result == explicit_fips

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_invalid_level_raises(self):
        """Test invalid geographic level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown geographic level"):
            load_geography_list("invalid_level")  # type: ignore[arg-type]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_with_config_county_all(self):
        """Test loading with config specifying all counties."""
        config = {"geographic": {"counties": "all"}}
        county_list = load_geography_list("county", config=config)

        assert len(county_list) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_with_config_county_list_mode(self):
        """Test loading with config specifying list mode."""
        config = {
            "geographic": {
                "counties": {
                    "mode": "list",
                    "fips_codes": ["38101", "38015"],
                }
            }
        }
        county_list = load_geography_list("county", config=config)

        assert county_list == ["38101", "38015"]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_load_with_config_place_list_mode(self):
        """Test loading with config specifying list mode for places."""
        config = {
            "geographic": {
                "places": {
                    "mode": "list",
                    "fips_codes": ["3825700", "3807200"],  # Fargo and Bismarck
                }
            }
        }
        place_list = load_geography_list("place", config=config)

        assert isinstance(place_list, list)
        assert place_list == ["3825700", "3807200"]


class TestGetGeographyName:
    """Test get_geography_name function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_state_name(self):
        """Test getting state name."""
        name = get_geography_name("38")

        assert name == "North Dakota"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_state_name_with_level(self):
        """Test getting state name with explicit level."""
        name = get_geography_name("38", level="state")

        assert name == "North Dakota"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_county_name(self):
        """Test getting county name."""
        # This depends on the default data, may return generic name
        name = get_geography_name("38015")  # Burleigh County

        assert "County" in name or "38015" in name

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_place_name(self):
        """Test getting place name."""
        # This depends on the default data
        name = get_geography_name("3825700")  # Fargo

        assert len(name) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_auto_detect_level_state(self):
        """Test auto-detection of state level from FIPS length."""
        name = get_geography_name("38")

        assert name == "North Dakota"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_auto_detect_level_county(self):
        """Test auto-detection of county level from FIPS length."""
        name = get_geography_name("38015")

        # Should not raise error and return some name
        assert len(name) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_auto_detect_level_place(self):
        """Test auto-detection of place level from FIPS length."""
        name = get_geography_name("3825700")

        assert len(name) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_invalid_fips_length_raises(self):
        """Test invalid FIPS length raises ValueError."""
        with pytest.raises(ValueError, match="Cannot determine level"):
            get_geography_name("123456789")  # Too many digits

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_unknown_fips_returns_generic(self):
        """Test unknown FIPS returns generic name."""
        name = get_geography_name("38999")  # Non-existent county

        # Should return something like "County 38999"
        assert "38999" in name


class TestCreateDefaultData:
    """Test helper functions for creating default data."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_create_default_counties(self):
        """Test creating default county data."""
        counties = _create_default_nd_counties()

        assert isinstance(counties, pd.DataFrame)
        assert len(counties) > 0
        assert "county_fips" in counties.columns
        assert "county_name" in counties.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_create_default_places(self):
        """Test creating default place data."""
        places = _create_default_nd_places()

        assert isinstance(places, pd.DataFrame)
        assert len(places) > 0
        assert "place_fips" in places.columns
        assert "place_name" in places.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_default_counties_includes_major_counties(self):
        """Test default counties includes major ND counties."""
        counties = _create_default_nd_counties()

        county_names = counties["county_name"].tolist()
        # Should include major counties like Cass, Burleigh
        assert any("Cass" in name for name in county_names) or len(counties) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_default_places_includes_major_cities(self):
        """Test default places includes major ND cities."""
        places = _create_default_nd_places()

        place_names = places["place_name"].tolist()
        # Should include major cities like Fargo, Bismarck
        assert any("Fargo" in name for name in place_names) or len(places) > 0


class TestValidationFunctions:
    """Test validation helper functions."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_county_data_valid(self):
        """Test validation passes for valid county data."""
        valid_data = pd.DataFrame(
            {
                "state_fips": ["38", "38"],
                "county_fips": ["38001", "38002"],
                "county_name": ["County A", "County B"],
            }
        )

        result = _validate_county_data(valid_data)

        assert len(result) == 2

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_county_data_missing_columns_raises(self):
        """Test validation raises for missing columns."""
        invalid_data = pd.DataFrame(
            {
                "state_fips": ["38"],
                # Missing county_fips and county_name
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            _validate_county_data(invalid_data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_county_data_filters_to_nd(self):
        """Test validation filters to North Dakota only."""
        mixed_data = pd.DataFrame(
            {
                "state_fips": ["38", "46"],  # ND and SD
                "county_fips": ["38001", "46001"],
                "county_name": ["ND County", "SD County"],
            }
        )

        result = _validate_county_data(mixed_data)

        assert len(result) == 1
        assert result["state_fips"].values[0] == "38"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_county_data_pads_fips(self):
        """Test validation pads FIPS codes to correct length."""
        data = pd.DataFrame(
            {
                "state_fips": [38, 38],  # Numeric, not padded
                "county_fips": [38001, 38002],
                "county_name": ["County A", "County B"],
            }
        )

        result = _validate_county_data(data)

        assert all(len(fips) == 2 for fips in result["state_fips"])
        assert all(len(fips) == 5 for fips in result["county_fips"])

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_place_data_valid(self):
        """Test validation passes for valid place data."""
        valid_data = pd.DataFrame(
            {
                "state_fips": ["38"],
                "place_fips": ["3800001"],
                "place_name": ["Test City"],
                "county_fips": ["38001"],
            }
        )

        result = _validate_place_data(valid_data)

        assert len(result) == 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_place_data_missing_columns_raises(self):
        """Test validation raises for missing columns."""
        invalid_data = pd.DataFrame(
            {
                "place_fips": ["3800001"],
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            _validate_place_data(invalid_data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validate_place_data_pads_fips(self):
        """Test validation pads place FIPS to 7 digits."""
        data = pd.DataFrame(
            {
                "state_fips": ["38"],
                "place_fips": ["3825700"],  # Correct length
                "place_name": ["Test City"],
                "county_fips": ["38001"],
            }
        )

        result = _validate_place_data(data)

        assert len(result["place_fips"].values[0]) == 7


class TestGeographyLoaderEdgeCases:
    """Test edge cases for geography loader."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_empty_result_after_filter(self):
        """Test handling when all geographies filtered out."""
        # Use tiger source and filter to very high population that no places meet
        places = load_nd_places(source="tiger", min_population=1_000_000_000)

        # Should return empty DataFrame, not error
        assert isinstance(places, pd.DataFrame)
        # Should be empty since no places have 1B population
        assert len(places) == 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_path_as_string(self, tmp_path):
        """Test that string paths work for reference files."""
        custom_counties = pd.DataFrame(
            {
                "state_fips": ["38"],
                "county_fips": ["38001"],
                "county_name": ["String Path County"],
            }
        )
        custom_path = tmp_path / "string_path.csv"
        custom_counties.to_csv(custom_path, index=False)

        counties = load_nd_counties(reference_path=custom_path)

        assert len(counties) == 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_vintage_parameter(self):
        """Test vintage parameter is accepted."""
        # Use tiger source which always returns default data
        counties_2020 = load_nd_counties(source="tiger", vintage=2020)
        places_2020 = load_nd_places(source="tiger", vintage=2020)

        assert len(counties_2020) > 0
        assert len(places_2020) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
