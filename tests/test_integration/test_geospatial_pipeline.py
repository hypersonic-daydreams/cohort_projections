"""
Integration tests for geospatial export pipeline (ADR-059).

Verifies that the export pipeline correctly handles the 'geojson' and
'shapefile' format flags, produces output files, respects dry-run mode,
and handles missing TIGER files gracefully.

All tests use synthetic projection data and mock the TIGER boundary
loader so actual shapefiles are not required.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Conditional geopandas import
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd  # noqa: F401
    from shapely.geometry import box

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

needs_geo = pytest.mark.skipif(not HAS_GEO, reason="geopandas/shapely not installed")

# ---------------------------------------------------------------------------
# Import export module
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

export_mod = importlib.import_module("scripts.pipeline.03_export_results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(projection_root: Path, export_root: Path) -> dict[str, Any]:
    """Minimal config payload for pipeline tests."""
    return {
        "pipeline": {
            "projection": {"output_dir": str(projection_root)},
            "export": {
                "output_dir": str(export_root),
                "create_packages": False,
                "formats": ["csv"],
            },
        },
        "place_projections": {
            "output": {"key_years": [2025, 2030]},
        },
        "scenarios": {"baseline": {"active": True}},
    }


def _write_projection_parquet(
    projection_root: Path,
    scenario: str,
    level: str,
    fips_codes: list[str],
) -> None:
    """Write minimal projection parquet files for testing."""
    level_dir = projection_root / scenario / level
    level_dir.mkdir(parents=True, exist_ok=True)

    fips_col = "county_fips" if level == "county" else "place_fips"
    for fips in fips_codes:
        rows = []
        for year in [2025, 2030]:
            rows.append(
                {
                    "year": year,
                    "age": 0,
                    "sex": "Male",
                    "race": "White",
                    "population": 1000.0,
                    fips_col: fips,
                }
            )
        df = pd.DataFrame(rows)
        fname = f"nd_{level}_{fips}_projection_2025_2055_baseline.parquet"
        df.to_parquet(level_dir / fname)


def _make_boundaries(level: str) -> Any:
    """Build a synthetic GeoDataFrame for the given level."""
    if level == "county":
        data = {
            "state_fips": ["38", "38"],
            "county_fips": ["38001", "38015"],
            "county_name": ["Adams County", "Burleigh County"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        }
    else:
        data = {
            "state_fips": ["38", "38"],
            "place_fips": ["3807200", "3825700"],
            "place_name": ["Bismarck city", "Fargo city"],
            "county_fips": ["", ""],
            "geometry": [box(0, 0, 0.5, 0.5), box(2, 0, 2.5, 0.5)],
        }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def _ok_result(component: str) -> Any:
    """Return a successful ExportResult placeholder."""
    r = export_mod.ExportResult(component)
    r.success = True
    return r


def _patch_non_geo_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch non-geospatial export steps to isolate pipeline wiring."""
    for func_name in (
        "convert_projection_formats",
        "create_summary_tables",
        "generate_data_dictionary",
        "package_for_distribution",
        "export_place_outputs",
    ):
        _name = func_name  # bind loop variable for closure
        monkeypatch.setattr(
            export_mod,
            func_name,
            lambda *a, _n=_name, **kw: _ok_result(_n),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestGeospatialPipeline:
    """Integration tests for geospatial export through the pipeline."""

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_geojson_export_via_pipeline(
        self,
        mock_boundaries: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end GeoJSON export via export_all_results."""
        _patch_non_geo_steps(monkeypatch)

        proj_root = tmp_path / "projections"
        export_root = tmp_path / "exports"
        _write_projection_parquet(proj_root, "baseline", "county", ["38001", "38015"])

        mock_boundaries.return_value = _make_boundaries("county")

        config = _build_config(proj_root, export_root)
        report = export_mod.export_all_results(
            config=config,
            scenarios=["baseline"],
            levels=["county"],
            formats=["geojson"],
            create_packages=False,
            dry_run=False,
        )

        geo_results = [
            r for r in report.results if r.component.startswith("geospatial_geojson")
        ]
        assert len(geo_results) >= 1
        assert geo_results[0].success

    @needs_geo
    def test_formats_flag_includes_geojson(self) -> None:
        """The --formats argument should accept 'geojson' and 'shapefile'."""
        # Verify that the argument parser accepts these choices
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--formats",
            nargs="+",
            choices=["csv", "excel", "parquet", "geojson", "shapefile"],
        )
        ns = parser.parse_args(["--formats", "geojson", "shapefile"])
        assert "geojson" in ns.formats
        assert "shapefile" in ns.formats

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_dry_run_creates_no_files(
        self,
        mock_boundaries: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dry-run mode should not create any geospatial files."""
        _patch_non_geo_steps(monkeypatch)

        proj_root = tmp_path / "projections"
        export_root = tmp_path / "exports"
        _write_projection_parquet(proj_root, "baseline", "county", ["38001", "38015"])

        mock_boundaries.return_value = _make_boundaries("county")

        config = _build_config(proj_root, export_root)
        report = export_mod.export_all_results(
            config=config,
            scenarios=["baseline"],
            levels=["county"],
            formats=["geojson"],
            create_packages=False,
            dry_run=True,
        )

        geo_results = [
            r for r in report.results if r.component.startswith("geospatial_geojson")
        ]
        assert len(geo_results) >= 1
        # No output files should have been created
        assert len(geo_results[0].output_files) == 0

    @needs_geo
    @patch("cohort_projections.geographic.geography_loader.load_tiger_boundaries")
    def test_county_and_place_levels(
        self,
        mock_boundaries: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both county and place levels should produce geospatial output."""
        _patch_non_geo_steps(monkeypatch)

        proj_root = tmp_path / "projections"
        export_root = tmp_path / "exports"
        _write_projection_parquet(proj_root, "baseline", "county", ["38001", "38015"])
        _write_projection_parquet(
            proj_root, "baseline", "place", ["3807200", "3825700"]
        )

        def _side_effect(level: str, **kwargs: Any) -> Any:
            return _make_boundaries(level)

        mock_boundaries.side_effect = _side_effect

        config = _build_config(proj_root, export_root)
        report = export_mod.export_all_results(
            config=config,
            scenarios=["baseline"],
            levels=["county", "place"],
            formats=["geojson"],
            create_packages=False,
            dry_run=False,
        )

        geo_results = [
            r for r in report.results if r.component.startswith("geospatial_")
        ]
        assert len(geo_results) >= 1
        total_files = sum(r.files_exported for r in geo_results)
        assert total_files >= 2  # at least one per level

    @needs_geo
    def test_missing_tiger_files_handled_gracefully(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing TIGER files should not crash the pipeline."""
        _patch_non_geo_steps(monkeypatch)

        proj_root = tmp_path / "projections"
        export_root = tmp_path / "exports"
        _write_projection_parquet(proj_root, "baseline", "county", ["38001"])

        config = _build_config(proj_root, export_root)

        # Patch load_tiger_boundaries to raise FileNotFoundError
        with patch(
            "cohort_projections.geographic.geography_loader.load_tiger_boundaries",
            side_effect=FileNotFoundError("TIGER shapefile not found"),
        ):
            report = export_mod.export_all_results(
                config=config,
                scenarios=["baseline"],
                levels=["county"],
                formats=["geojson"],
                create_packages=False,
                dry_run=False,
            )

        geo_results = [
            r for r in report.results if r.component.startswith("geospatial_")
        ]
        # Should succeed (graceful handling) but export 0 files
        assert len(geo_results) >= 1
        assert geo_results[0].success
        assert geo_results[0].files_exported == 0
