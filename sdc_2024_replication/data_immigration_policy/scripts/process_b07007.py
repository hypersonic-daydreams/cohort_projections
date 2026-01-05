#!/usr/bin/env python3
# mypy: ignore-errors
"""
Process ACS Table B07007 moved-from-abroad series.

Creates a long-format state panel with moved-from-abroad counts by citizenship,
including margins of error, using config-driven paths from projection_config.yaml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution

logger = setup_logger(__name__)

_LABELS_CACHE: dict[int, dict[str, str]] = {}


def load_settings() -> dict[str, Any]:
    """Load ACS settings from projection config."""
    config = ConfigLoader().get_projection_config()
    data_sources = config.get("data_sources", {})
    acs_cfg = data_sources.get("acs_moved_from_abroad", {})

    table_id = acs_cfg.get("table_id")
    raw_dir = acs_cfg.get("raw_dir")
    processed_dir = acs_cfg.get("processed_dir")
    output_prefix = acs_cfg.get("output_prefix")

    if not table_id or not raw_dir or not processed_dir or not output_prefix:
        raise ValueError("Missing ACS B07007 settings in projection_config.yaml")

    project_root = Path(__file__).resolve().parents[3]
    return {
        "table_id": table_id,
        "raw_path": project_root / raw_dir,
        "processed_path": project_root / processed_dir,
        "output_prefix": output_prefix,
    }


def load_labels_for_year(raw_path: Path, table_id: str, year: int) -> dict[str, str]:
    """Load variable labels for a specific year with caching."""
    if year in _LABELS_CACHE:
        return _LABELS_CACHE[year]

    year_labels_file = raw_path / f"{table_id.lower()}_variable_labels_{year}.json"
    if year_labels_file.exists():
        with year_labels_file.open() as f:
            labels = json.load(f)
        _LABELS_CACHE[year] = labels
        return labels

    default_labels_file = raw_path / f"{table_id.lower()}_variable_labels.json"
    if default_labels_file.exists():
        with default_labels_file.open() as f:
            labels = json.load(f)
        _LABELS_CACHE[year] = labels
        logger.warning("Using default labels for year %s (year-specific file not found).", year)
        return labels

    logger.error("No labels found for year %s.", year)
    return {}


def parse_label(label: str) -> dict[str, str | None] | None:
    """Parse a B07007 label into components."""
    if label.startswith("Estimate!!"):
        label = label.replace("Estimate!!", "")
    else:
        return None

    parts = [p.rstrip(":") for p in label.split("!!")]
    if not parts or parts[0] != "Total" or len(parts) < 2:
        return None

    mobility_status = parts[1]
    citizenship = parts[2] if len(parts) >= 3 else None
    citizenship_detail = parts[3] if len(parts) >= 4 else None

    return {
        "mobility_status": mobility_status,
        "citizenship": citizenship,
        "citizenship_detail": citizenship_detail,
    }


def build_validation_log(df: pd.DataFrame, output_path: Path) -> None:
    """Write a short validation summary for the processed series."""
    year_min = int(df["year"].min()) if not df.empty else None
    year_max = int(df["year"].max()) if not df.empty else None
    missing_estimate = int(df["estimate"].isna().sum())
    missing_moe = int(df["margin_of_error"].isna().sum())
    negative_estimate = int((df["estimate"] < 0).sum())

    lines = [
        "# ACS B07007 Moved-From-Abroad Validation Summary",
        "",
        f"- Rows: {len(df):,}",
        f"- Years: {year_min}-{year_max}",
        f"- States: {df['state_fips'].nunique()}",
        f"- Missing estimates: {missing_estimate:,}",
        f"- Missing MOE: {missing_moe:,}",
        f"- Negative estimates: {negative_estimate:,}",
    ]

    output_path.write_text("\n".join(lines) + "\n")


def main() -> pd.DataFrame:
    """Process B07007 moved-from-abroad series into long format."""
    settings = load_settings()
    table_id = settings["table_id"]
    raw_path: Path = settings["raw_path"]
    processed_path: Path = settings["processed_path"]
    output_prefix = settings["output_prefix"]

    processed_path.mkdir(parents=True, exist_ok=True)

    data_file = raw_path / f"{table_id.lower()}_states_all_years.csv"
    df = pd.read_csv(data_file, dtype=str)

    estimate_cols = [c for c in df.columns if re.match(rf"{table_id}_\d{{3}}E$", c)]
    moe_cols = [c for c in df.columns if re.match(rf"{table_id}_\d{{3}}M$", c)]
    id_vars = [c for c in ["NAME", "state", "year", "GEO_ID"] if c in df.columns]

    est_long = df.melt(
        id_vars=id_vars, value_vars=estimate_cols, var_name="variable", value_name="estimate"
    )
    moe_long = df.melt(
        id_vars=id_vars, value_vars=moe_cols, var_name="variable", value_name="margin_of_error"
    )
    moe_long["variable"] = moe_long["variable"].str[:-1] + "E"

    combined = est_long.merge(moe_long, on=id_vars + ["variable"], how="left")

    records: list[pd.DataFrame] = []
    for year_str, group in combined.groupby("year"):
        try:
            year = int(year_str)
        except (TypeError, ValueError):
            logger.warning("Skipping non-numeric year value: %s", year_str)
            continue

        labels = load_labels_for_year(raw_path, table_id, year)
        if not labels:
            logger.warning("No labels found for year %s; skipping.", year)
            continue

        meta_map: dict[str, dict[str, str | None]] = {}
        for var_id, label in labels.items():
            if not var_id.endswith("E"):
                continue
            parsed = parse_label(label)
            if parsed and parsed["mobility_status"] == "Moved from abroad":
                meta_map[var_id] = parsed

        if not meta_map:
            logger.warning("No moved-from-abroad variables found for year %s.", year)
            continue

        subset = group[group["variable"].isin(meta_map)].copy()
        subset["year"] = year
        subset["mobility_status"] = subset["variable"].map(
            lambda var: meta_map[var]["mobility_status"]
        )
        subset["citizenship"] = subset["variable"].map(lambda var: meta_map[var]["citizenship"])
        subset["citizenship_detail"] = subset["variable"].map(
            lambda var: meta_map[var]["citizenship_detail"]
        )

        subset["estimate"] = pd.to_numeric(subset["estimate"], errors="coerce")
        subset["margin_of_error"] = pd.to_numeric(subset["margin_of_error"], errors="coerce")

        records.append(subset)

    if not records:
        raise ValueError("No moved-from-abroad records found in ACS B07007 data.")

    output = pd.concat(records, ignore_index=True)
    output = output.rename(columns={"NAME": "state_name", "state": "state_fips"})
    output["table_id"] = table_id

    output_file = processed_path / f"{output_prefix}.parquet"
    output.to_parquet(output_file, index=False)

    output_csv = processed_path / f"{output_prefix}.csv"
    output.to_csv(output_csv, index=False)

    validation_file = processed_path / f"{output_prefix}_validation.md"
    build_validation_log(output, validation_file)

    logger.info("Saved %s", output_file)
    logger.info("Saved %s", output_csv)
    logger.info("Saved %s", validation_file)

    negative_count = int((output["estimate"] < 0).sum())
    if negative_count:
        logger.warning("Found %s negative estimates in moved-from-abroad series.", negative_count)

    return output


if __name__ == "__main__":
    settings = load_settings()
    with log_execution(
        __file__,
        parameters={"table_id": settings["table_id"], "series": "acs_moved_from_abroad"},
    ):
        main()
