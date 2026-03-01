"""
Integration tests for the housing-unit method pipeline stage (02c).

Tests cover end-to-end pipeline execution with synthetic data, dry-run mode,
output file creation, config-disabled skipping, and basic export integration.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pipeline_mod = importlib.import_module("scripts.pipeline.02c_run_housing_unit_projections")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_housing_csv(tmp_path: Path, n_places: int = 3, n_years: int = 10) -> Path:
    """Write synthetic housing CSV with *n_places* places and *n_years* vintages."""
    rows = []
    base_fips = 3825700
    for i in range(n_places):
        fips = str(base_fips + i)
        for y_offset in range(n_years):
            year = 2010 + y_offset
            rows.append({
                "place_fips": fips,
                "place_name": f"Place_{i}",
                "year": year,
                "housing_units": 1000 + y_offset * 50 + i * 100,
                "avg_hh_size": 2.40 - y_offset * 0.01,
            })
    csv_path = tmp_path / "housing.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _build_config(
    csv_path: Path,
    output_root: Path,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build minimal config dict for the HU pipeline stage."""
    return {
        "housing_unit_method": {
            "enabled": enabled,
            "housing_data_path": str(csv_path),
            "projection_years": [2025, 2030],
            "trend_method": "log_linear",
            "pph_method": "hold_last",
            "min_history_years": 3,
            "output_dir": str(output_root),
        },
        "scenarios": {"baseline": {"active": True}},
        "pipeline": {
            "projection": {"output_dir": str(output_root)},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHousingUnitPipelineIntegration:
    """Integration tests for the 02c pipeline stage."""

    def test_end_to_end_with_synthetic_data(self, tmp_path: Path) -> None:
        """Full pipeline run produces Parquet and metadata outputs."""
        csv = _write_housing_csv(tmp_path, n_places=3, n_years=10)
        output_root = tmp_path / "projections"
        config = _build_config(csv, output_root)

        exit_code = pipeline_mod.run_stage(
            config=config,
            scenarios=["baseline"],
            dry_run=False,
        )
        assert exit_code == 0

        parquet_path = output_root / "baseline" / "place" / "housing_unit_projections.parquet"
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        assert "place_fips" in df.columns
        assert "population_hu" in df.columns

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        """Dry-run mode validates but does not create output files."""
        csv = _write_housing_csv(tmp_path)
        output_root = tmp_path / "projections"
        config = _build_config(csv, output_root)

        exit_code = pipeline_mod.run_stage(
            config=config,
            scenarios=["baseline"],
            dry_run=True,
        )
        assert exit_code == 0
        parquet_path = output_root / "baseline" / "place" / "housing_unit_projections.parquet"
        assert not parquet_path.exists()

    def test_output_files_and_format(self, tmp_path: Path) -> None:
        """Outputs include correctly structured Parquet and JSON metadata."""
        csv = _write_housing_csv(tmp_path, n_places=2, n_years=8)
        output_root = tmp_path / "projections"
        config = _build_config(csv, output_root)

        pipeline_mod.run_stage(config=config, scenarios=["baseline"], dry_run=False)

        meta_path = output_root / "baseline" / "place" / "housing_unit_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert meta["scenario"] == "baseline"
        assert meta["trend_method"] == "log_linear"
        assert meta["place_count"] > 0

    def test_config_disabled_skips_gracefully(self, tmp_path: Path) -> None:
        """When housing_unit_method.enabled is false, stage exits 0 without writing."""
        csv = _write_housing_csv(tmp_path)
        output_root = tmp_path / "projections"
        config = _build_config(csv, output_root, enabled=False)

        exit_code = pipeline_mod.run_stage(
            config=config,
            scenarios=["baseline"],
            dry_run=False,
        )
        assert exit_code == 0
        parquet_path = output_root / "baseline" / "place" / "housing_unit_projections.parquet"
        assert not parquet_path.exists()

    def test_multiple_scenarios(self, tmp_path: Path) -> None:
        """Stage writes separate outputs per scenario."""
        csv = _write_housing_csv(tmp_path, n_places=2, n_years=6)
        output_root = tmp_path / "projections"
        config = _build_config(csv, output_root)

        exit_code = pipeline_mod.run_stage(
            config=config,
            scenarios=["baseline", "high_growth"],
            dry_run=False,
        )
        assert exit_code == 0
        for scenario in ["baseline", "high_growth"]:
            path = output_root / scenario / "place" / "housing_unit_projections.parquet"
            assert path.exists(), f"Missing output for {scenario}"
