"""
Unit tests for output writers module.

Tests the write_projection_excel, write_projection_csv, write_projection_formats,
and write_projection_shapefile functions for proper file creation and formatting.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# Try to import the writers module
try:
    from cohort_projections.output.writers import (
        GEOPANDAS_AVAILABLE,
        OPENPYXL_AVAILABLE,
        write_projection_csv,
        write_projection_excel,
        write_projection_formats,
        write_projection_shapefile,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    OPENPYXL_AVAILABLE = False
    GEOPANDAS_AVAILABLE = False


class TestWriteProjectionCSV:
    """Test write_projection_csv function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        data = []
        for year in [2025, 2026, 2027]:
            for age in [0, 1, 5, 10, 20, 30, 65, 85]:
                for sex in ["Male", "Female"]:
                    for race in ["White", "Black", "Hispanic"]:
                        data.append(
                            {
                                "year": year,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": 100.0 + year - 2025 + age * 0.5,
                            }
                        )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_basic(self, sample_projection, tmp_path):
        """Test basic CSV export."""
        output_file = tmp_path / "test_output.csv"

        result_path = write_projection_csv(sample_projection, output_file)

        assert output_file.exists()
        assert result_path == output_file
        result_df = pd.read_csv(output_file)
        assert len(result_df) == len(sample_projection)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_long_format(self, sample_projection, tmp_path):
        """Test CSV export in long format (default)."""
        output_file = tmp_path / "test_long.csv"

        write_projection_csv(sample_projection, output_file, format_type="long")

        result_df = pd.read_csv(output_file)
        assert "year" in result_df.columns
        assert "age" in result_df.columns
        assert "sex" in result_df.columns
        assert "race" in result_df.columns
        assert "population" in result_df.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_wide_format(self, sample_projection, tmp_path):
        """Test CSV export in wide format (years as columns)."""
        output_file = tmp_path / "test_wide.csv"

        write_projection_csv(sample_projection, output_file, format_type="wide")

        result_df = pd.read_csv(output_file)
        # Wide format should have year columns like 'year_2025', 'year_2026', etc.
        year_cols = [col for col in result_df.columns if col.startswith("year_")]
        assert len(year_cols) >= 2

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_with_compression(self, sample_projection, tmp_path):
        """Test CSV export with gzip compression."""
        output_file = tmp_path / "test_output.csv.gz"

        result_path = write_projection_csv(sample_projection, output_file, compression="gzip")

        assert output_file.exists()
        assert result_path == output_file
        # Should be able to read the compressed file
        result_df = pd.read_csv(output_file, compression="gzip")
        assert len(result_df) == len(sample_projection)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_filter_age_ranges(self, sample_projection, tmp_path):
        """Test CSV export with age range filtering."""
        output_file = tmp_path / "test_filtered.csv"

        # Filter to ages 0-4 and 65+
        write_projection_csv(
            sample_projection,
            output_file,
            age_ranges=[(0, 4), (65, 90)],
        )

        result_df = pd.read_csv(output_file)
        ages = result_df["age"].unique()
        # Should only have ages in specified ranges
        assert all(a <= 4 or a >= 65 for a in ages)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_filter_sexes(self, sample_projection, tmp_path):
        """Test CSV export with sex filtering."""
        output_file = tmp_path / "test_male_only.csv"

        write_projection_csv(sample_projection, output_file, sexes=["Male"])

        result_df = pd.read_csv(output_file)
        assert result_df["sex"].unique().tolist() == ["Male"]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_filter_races(self, sample_projection, tmp_path):
        """Test CSV export with race filtering."""
        output_file = tmp_path / "test_race_filter.csv"

        write_projection_csv(sample_projection, output_file, races=["White", "Hispanic"])

        result_df = pd.read_csv(output_file)
        assert set(result_df["race"].unique()) == {"White", "Hispanic"}

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_csv_creates_parent_directory(self, sample_projection, tmp_path):
        """Test that CSV export creates parent directories if needed."""
        output_file = tmp_path / "nested" / "dir" / "test_output.csv"

        write_projection_csv(sample_projection, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestWriteProjectionExcel:
    """Test write_projection_excel function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        data = []
        for year in [2025, 2030, 2035]:
            for age in range(0, 91, 5):
                for sex in ["Male", "Female"]:
                    for race in ["White", "Black"]:
                        data.append(
                            {
                                "year": year,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": 1000.0 + year - 2025 + age * 2,
                            }
                        )
        return pd.DataFrame(data)

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_basic(self, sample_projection, tmp_path):
        """Test basic Excel export."""
        output_file = tmp_path / "test_output.xlsx"

        result_path = write_projection_excel(sample_projection, output_file)

        assert output_file.exists()
        assert result_path == output_file

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_has_required_sheets(self, sample_projection, tmp_path):
        """Test that Excel export creates expected sheets."""
        output_file = tmp_path / "test_sheets.xlsx"

        write_projection_excel(sample_projection, output_file)

        # Read the Excel file and check sheets
        with pd.ExcelFile(output_file) as xlsx:
            sheets = xlsx.sheet_names
            assert "Summary" in sheets
            assert "By Age" in sheets
            assert "By Sex" in sheets
            assert "By Race" in sheets
            assert "Detail" in sheets
            assert "Metadata" in sheets

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_with_summary_df(self, sample_projection, tmp_path):
        """Test Excel export with custom summary DataFrame."""
        output_file = tmp_path / "test_summary.xlsx"

        summary_df = pd.DataFrame(
            {
                "Year": [2025, 2030, 2035],
                "Total": [100000, 110000, 120000],
                "Growth": [0.0, 10000.0, 10000.0],
            }
        )

        write_projection_excel(sample_projection, output_file, summary_df=summary_df)

        assert output_file.exists()
        # Read summary sheet
        summary = pd.read_excel(output_file, sheet_name="Summary")
        assert "Year" in summary.columns
        assert "Total" in summary.columns

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_with_metadata(self, sample_projection, tmp_path):
        """Test Excel export with metadata."""
        output_file = tmp_path / "test_metadata.xlsx"

        metadata = {
            "projection_type": "baseline",
            "base_year": 2025,
            "end_year": 2045,
            "parameters": {
                "fertility": "medium",
                "migration": "baseline",
            },
        }

        write_projection_excel(sample_projection, output_file, metadata=metadata)

        assert output_file.exists()

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_no_charts(self, sample_projection, tmp_path):
        """Test Excel export without charts."""
        output_file = tmp_path / "test_no_charts.xlsx"

        write_projection_excel(sample_projection, output_file, include_charts=False)

        assert output_file.exists()

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_no_formatting(self, sample_projection, tmp_path):
        """Test Excel export without formatting."""
        output_file = tmp_path / "test_no_format.xlsx"

        write_projection_excel(sample_projection, output_file, include_formatting=False)

        assert output_file.exists()

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_empty_dataframe_raises(self, tmp_path):
        """Test that Excel export raises error for empty DataFrame."""
        output_file = tmp_path / "test_empty.xlsx"
        empty_df = pd.DataFrame()

        # Empty DataFrame triggers missing columns error first
        with pytest.raises(ValueError, match="missing required columns"):
            write_projection_excel(empty_df, output_file)

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_missing_columns_raises(self, tmp_path):
        """Test that Excel export raises error for missing required columns."""
        output_file = tmp_path / "test_missing_cols.xlsx"
        # DataFrame missing 'race' column
        incomplete_df = pd.DataFrame(
            {
                "year": [2025, 2025],
                "age": [0, 1],
                "sex": ["Male", "Female"],
                "population": [100.0, 100.0],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            write_projection_excel(incomplete_df, output_file)

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_export_excel_with_title(self, sample_projection, tmp_path):
        """Test Excel export with custom title."""
        output_file = tmp_path / "test_title.xlsx"

        write_projection_excel(
            sample_projection,
            output_file,
            title="North Dakota Population Projection 2025-2045",
        )

        assert output_file.exists()


class TestWriteProjectionFormats:
    """Test write_projection_formats function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        data = []
        for year in [2025, 2030]:
            for age in [0, 10, 20, 65]:
                for sex in ["Male", "Female"]:
                    for race in ["White"]:
                        data.append(
                            {
                                "year": year,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": 500.0,
                            }
                        )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_write_csv_format(self, sample_projection, tmp_path):
        """Test writing CSV format only."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["csv"],
        )

        assert "csv" in output_paths
        assert output_paths["csv"].exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_write_parquet_format(self, sample_projection, tmp_path):
        """Test writing Parquet format only."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["parquet"],
        )

        assert "parquet" in output_paths
        assert output_paths["parquet"].exists()
        # Verify parquet is readable
        result_df = pd.read_parquet(output_paths["parquet"])
        assert len(result_df) == len(sample_projection)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_write_json_format(self, sample_projection, tmp_path):
        """Test writing JSON format only."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["json"],
        )

        assert "json" in output_paths
        assert output_paths["json"].exists()
        # Verify JSON is readable
        with open(output_paths["json"]) as f:
            data = json.load(f)
        assert "projection" in data
        assert len(data["projection"]) == len(sample_projection)

    @pytest.mark.skipif(
        not IMPORTS_AVAILABLE or not OPENPYXL_AVAILABLE,
        reason="openpyxl not available",
    )
    def test_write_excel_format(self, sample_projection, tmp_path):
        """Test writing Excel format only."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["excel"],
        )

        assert "excel" in output_paths
        assert output_paths["excel"].exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_write_multiple_formats(self, sample_projection, tmp_path):
        """Test writing multiple formats at once."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["csv", "parquet", "json"],
        )

        assert "csv" in output_paths
        assert "parquet" in output_paths
        assert "json" in output_paths
        for path in output_paths.values():
            assert path.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_write_with_metadata(self, sample_projection, tmp_path):
        """Test writing with metadata file."""
        metadata = {
            "projection_type": "baseline",
            "created_by": "test",
        }

        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["csv"],
            metadata=metadata,
        )

        assert "metadata" in output_paths
        assert output_paths["metadata"].exists()
        with open(output_paths["metadata"]) as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["projection_type"] == "baseline"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_default_formats(self, sample_projection, tmp_path):
        """Test default format selection."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=None,  # Use defaults
        )

        # Default should include csv and parquet
        assert "csv" in output_paths
        assert "parquet" in output_paths

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_compression_applied(self, sample_projection, tmp_path):
        """Test that compression is applied to CSV."""
        output_paths = write_projection_formats(
            sample_projection,
            output_dir=tmp_path,
            base_filename="test_projection",
            formats=["csv"],
            compression="gzip",
        )

        # CSV should be compressed
        assert output_paths["csv"].suffix == ".gz"


class TestWriteProjectionShapefile:
    """Test write_projection_shapefile function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        return pd.DataFrame(
            {
                "year": [2025, 2025, 2025, 2025],
                "age": [0, 1, 2, 3],
                "sex": ["Male", "Male", "Female", "Female"],
                "race": ["White", "White", "White", "White"],
                "population": [100.0, 100.0, 100.0, 100.0],
            }
        )

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_shapefile_not_implemented(self, sample_projection, tmp_path):
        """Test that shapefile export raises NotImplementedError when geopandas available."""
        output_file = tmp_path / "test_output.geojson"

        if GEOPANDAS_AVAILABLE:
            # With geopandas available, raises NotImplementedError
            with pytest.raises(NotImplementedError):
                write_projection_shapefile(
                    sample_projection,
                    geography_level="state",
                    output_path=output_file,
                )
        else:
            # Without geopandas, raises ImportError
            with pytest.raises(ImportError):
                write_projection_shapefile(
                    sample_projection,
                    geography_level="state",
                    output_path=output_file,
                )

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_shapefile_requires_geopandas(self, sample_projection, tmp_path):
        """Test that shapefile export requires geopandas."""
        output_file = tmp_path / "test_output.geojson"

        if not GEOPANDAS_AVAILABLE:
            with pytest.raises(ImportError):
                write_projection_shapefile(
                    sample_projection,
                    geography_level="county",
                    output_path=output_file,
                )


class TestCSVEdgeCases:
    """Test edge cases for CSV export."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_single_row_export(self, tmp_path):
        """Test exporting a single row DataFrame."""
        single_row_df = pd.DataFrame(
            {
                "year": [2025],
                "age": [0],
                "sex": ["Male"],
                "race": ["White"],
                "population": [100.0],
            }
        )
        output_file = tmp_path / "single_row.csv"

        write_projection_csv(single_row_df, output_file)

        result_df = pd.read_csv(output_file)
        assert len(result_df) == 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_with_path_object(self, tmp_path):
        """Test export accepts Path objects."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "age": [0],
                "sex": ["Male"],
                "race": ["White"],
                "population": [100.0],
            }
        )
        output_file = Path(tmp_path) / "path_object.csv"

        result_path = write_projection_csv(df, output_file)

        assert isinstance(result_path, Path)
        assert result_path.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_export_with_string_path(self, tmp_path):
        """Test export accepts string paths."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "age": [0],
                "sex": ["Male"],
                "race": ["White"],
                "population": [100.0],
            }
        )
        output_file = str(tmp_path / "string_path.csv")

        result_path = write_projection_csv(df, output_file)

        assert isinstance(result_path, Path)
        assert result_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
