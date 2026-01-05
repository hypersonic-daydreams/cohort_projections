"""
Unit tests for SDC statistical_analysis data loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sdc_scripts_dir() -> Path:
    return Path(__file__).parent.parent.parent / "sdc_2024_replication" / "scripts"


def test_data_loader_files_mode_uses_local_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sdc_scripts_dir: Path
):
    monkeypatch.setenv("SDC_ANALYSIS_DATA_SOURCE", "files")

    sys.path.insert(0, str(sdc_scripts_dir))
    try:
        from statistical_analysis import data_loader  # noqa: E402

        monkeypatch.setattr(data_loader, "_immigration_analysis_dir", lambda: tmp_path)
        monkeypatch.setattr(
            data_loader.db_config,
            "get_db_connection",
            lambda: (_ for _ in ()).throw(AssertionError("DB connection should not be used")),
        )

        pd.DataFrame(
            {
                "year": [2020],
                "nd_intl_migration": [100],
                "us_intl_migration": [10000],
                "nd_share_of_us_intl_pct": [1.0],
                "nd_share_of_us_pop_pct": [0.3],
            }
        ).to_csv(tmp_path / "nd_migration_summary.csv", index=False)

        pd.DataFrame(
            {
                "state": ["Alabama", "North Dakota"],
                "state_fips": [1, 38],
                "year": [2020, 2020],
                "population": [1_000_000, 800_000],
                "pop_change": [10, 5],
                "births": [12, 8],
                "deaths": [9, 7],
                "natural_change": [3, 1],
                "intl_migration": [100, 200],
                "domestic_migration": [-5, 10],
                "net_migration": [95, 210],
            }
        ).to_csv(tmp_path / "combined_components_of_change.csv", index=False)

        pd.DataFrame(
            {
                "fiscal_year": [2020],
                "state": ["North Dakota"],
                "nationality": ["Total"],
                "arrivals": [123],
            }
        ).to_parquet(tmp_path / "refugee_arrivals_by_state_nationality.parquet", index=False)

        migration = data_loader.load_migration_summary()
        assert migration.loc[0, "nd_intl_migration"] == 100

        panel = data_loader.load_panel_data()
        assert set(panel.columns) >= {
            "year",
            "state",
            "population",
            "intl_migration",
            "domestic_migration",
            "net_migration",
        }

        components = data_loader.load_state_components()
        assert set(components.columns) == {
            "year",
            "state",
            "intl_migration",
            "domestic_migration",
            "pop_estimate",
        }

        refugees = data_loader.load_refugee_arrivals()
        assert refugees.loc[0, "arrivals"] == 123
        assert {"fiscal_year", "state", "nationality", "arrivals"} <= set(refugees.columns)
    finally:
        sys.path.remove(str(sdc_scripts_dir))
        monkeypatch.delenv("SDC_ANALYSIS_DATA_SOURCE", raising=False)
