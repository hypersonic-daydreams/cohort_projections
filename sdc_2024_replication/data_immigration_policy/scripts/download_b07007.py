#!/usr/bin/env python3
# mypy: ignore-errors
"""
Download ACS Table B07007 (Geographical Mobility by Citizenship Status).

This script retrieves ACS 5-year group data for all states, saving per-year CSVs,
combined output, and year-specific variable labels. Configuration is sourced from
config/projection_config.yaml (data_sources.acs_*).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution

logger = setup_logger(__name__)


def load_settings() -> dict[str, Any]:
    """Load ACS settings from projection config."""
    config = ConfigLoader().get_projection_config()
    data_sources = config.get("data_sources", {})
    acs_base = data_sources.get("acs_api_base")
    acs_cfg = data_sources.get("acs_moved_from_abroad", {})

    if not acs_base:
        raise ValueError("Missing data_sources.acs_api_base in projection_config.yaml")

    table_id = acs_cfg.get("table_id")
    start_year = acs_cfg.get("start_year")
    end_year = acs_cfg.get("end_year")
    raw_dir = acs_cfg.get("raw_dir")

    if not table_id or start_year is None or end_year is None or not raw_dir:
        raise ValueError("Missing ACS B07007 settings in projection_config.yaml")

    project_root = Path(__file__).resolve().parents[3]
    raw_path = project_root / raw_dir
    raw_path.mkdir(parents=True, exist_ok=True)

    return {
        "acs_base": acs_base,
        "table_id": table_id,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "raw_path": raw_path,
    }


def fetch_variable_labels(acs_base: str, table_id: str, year: int) -> dict[str, str]:
    """Fetch variable labels for a table group."""
    url = f"{acs_base.format(year=year)}/groups/{table_id}.json"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        logger.warning("Group %s not available for %s (HTTP 404).", table_id, year)
        return {}
    response.raise_for_status()
    data = response.json()

    labels: dict[str, str] = {}
    for var_id, var_info in data.get("variables", {}).items():
        if var_id.startswith(f"{table_id}_"):
            labels[var_id] = var_info.get("label", "")

    return labels


def fetch_year_data(acs_base: str, table_id: str, year: int) -> pd.DataFrame | None:
    """Fetch B07007 data for a specific year using group()."""
    url = acs_base.format(year=year)
    params = {"get": f"group({table_id})", "for": "state:*"}

    response = requests.get(url, params=params, timeout=120)
    if response.status_code == 404:
        logger.warning("ACS group %s not available for %s (HTTP 404).", table_id, year)
        return None
    response.raise_for_status()
    data = response.json()

    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)

    drop_cols = [c for c in df.columns if c.endswith(("EA", "MA"))]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


def main() -> list[pd.DataFrame]:
    """Download B07007 data for configured years."""
    settings = load_settings()
    acs_base = settings["acs_base"]
    table_id = settings["table_id"]
    raw_path: Path = settings["raw_path"]
    start_year = settings["start_year"]
    end_year = settings["end_year"]

    years = list(range(start_year, end_year + 1))
    all_years_data: list[pd.DataFrame] = []
    year_labels: dict[int, dict[str, str]] = {}

    for year in years:
        logger.info("Fetching %s data for %s...", table_id, year)
        df = fetch_year_data(acs_base, table_id, year)
        if df is None:
            continue

        df["year"] = year
        all_years_data.append(df)

        year_file = raw_path / f"{table_id.lower()}_states_{year}.csv"
        df.to_csv(year_file, index=False)
        logger.info("Saved %s", year_file)

        labels = fetch_variable_labels(acs_base, table_id, year)
        if labels:
            year_labels[year] = labels
            labels_file = raw_path / f"{table_id.lower()}_variable_labels_{year}.json"
            with labels_file.open("w") as f:
                json.dump(labels, f, indent=2)
            logger.info("Saved labels %s", labels_file)

        time.sleep(1)

    if all_years_data:
        combined = pd.concat(all_years_data, ignore_index=True)
        combined_file = raw_path / f"{table_id.lower()}_states_all_years.csv"
        combined.to_csv(combined_file, index=False)
        logger.info("Saved combined file %s", combined_file)

        most_recent_year = max(year_labels.keys()) if year_labels else None
        if most_recent_year and year_labels[most_recent_year]:
            default_labels_file = raw_path / f"{table_id.lower()}_variable_labels.json"
            with default_labels_file.open("w") as f:
                json.dump(year_labels[most_recent_year], f, indent=2)
            logger.info("Saved default labels %s", default_labels_file)

    return all_years_data


if __name__ == "__main__":
    settings = load_settings()
    with log_execution(
        __file__,
        parameters={"table_id": settings["table_id"], "series": "acs_moved_from_abroad"},
    ):
        main()
