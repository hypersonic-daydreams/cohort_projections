"""
Build DHS LPR multi-year panel variants for modeling.

This script derives modeling-friendly panel variants from the canonical output:
  `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet`

Variants are written back to `data/processed/immigration/analysis/` without
modifying the canonical source file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution

logger = setup_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _immigration_analysis_dir(config: dict) -> Path:
    processed_dir = (
        config.get("data_sources", {})
        .get("acs_moved_from_abroad", {})
        .get("processed_dir", "data/processed/immigration/analysis")
    )
    return _project_root() / processed_dir


def build_state_fips_mapping(components_path: Path) -> pd.DataFrame:
    """Build a 50-states + DC mapping from `combined_components_of_change.csv`.

    Args:
        components_path: Path to `combined_components_of_change.csv`.

    Returns:
        DataFrame with columns: `state`, `state_fips` (int) for FIPS 1-56 (incl. DC=11).
    """
    df = pd.read_csv(components_path, usecols=["state", "state_fips"])
    df = df[df["state_fips"].between(1, 56)]
    df = df.drop_duplicates(subset=["state_fips"]).sort_values("state_fips")
    return df.reset_index(drop=True)


def build_lpr_panel_variants(
    lpr_state_time: pd.DataFrame, state_fips_map: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build modeling-friendly LPR panel variants.

    Args:
        lpr_state_time: Canonical LPR state time series with columns:
            `state_or_territory`, `fiscal_year`, `lpr_count`.
        state_fips_map: Mapping DataFrame with columns `state`, `state_fips`.

    Returns:
        Dictionary of variant DataFrames keyed by output stem name.
    """
    expected_states = set(state_fips_map["state"])

    df = lpr_state_time.rename(columns={"state_or_territory": "state"}).copy()
    df["lpr_count"] = pd.to_numeric(df["lpr_count"], errors="coerce")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")

    df = df[df["state"].isin(expected_states)].copy()
    df = df.merge(state_fips_map, on="state", how="left", validate="many_to_one")

    present_by_year = df.groupby("fiscal_year")["state"].nunique()
    expected_n = len(expected_states)
    incomplete_years = sorted(present_by_year[present_by_year != expected_n].index.dropna().tolist())
    if incomplete_years:
        logger.warning("LPR panel has incomplete state coverage for years: %s", incomplete_years)

    balanced_years = sorted(present_by_year[present_by_year == expected_n].index.dropna().tolist())
    df_balanced = df[df["fiscal_year"].isin(balanced_years)].copy()

    us_totals = (
        df.groupby("fiscal_year", dropna=True)["lpr_count"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"lpr_count": "us_total_lpr_count"})
        .sort_values("fiscal_year")
    )

    nd = df[df["state"] == "North Dakota"][["fiscal_year", "lpr_count"]].rename(
        columns={"lpr_count": "nd_lpr_count"}
    )
    nd_share = nd.merge(us_totals, on="fiscal_year", how="left", validate="one_to_one")
    nd_share["nd_share_pct"] = (nd_share["nd_lpr_count"] / nd_share["us_total_lpr_count"] * 100).round(
        4
    )

    return {
        "dhs_lpr_by_state_time_states_only": df.sort_values(["state_fips", "fiscal_year"]).reset_index(
            drop=True
        ),
        "dhs_lpr_by_state_time_states_only_balanced": df_balanced.sort_values(
            ["state_fips", "fiscal_year"]
        ).reset_index(drop=True),
        "dhs_lpr_us_total_time": us_totals.reset_index(drop=True),
        "dhs_lpr_nd_share_time": nd_share.sort_values("fiscal_year").reset_index(drop=True),
    }


def main() -> None:
    """Build and write panel variants for DHS LPR data."""
    config = ConfigLoader().get_projection_config()
    analysis_dir = _immigration_analysis_dir(config)

    lpr_path = analysis_dir / "dhs_lpr_by_state_time.parquet"
    components_path = analysis_dir / "combined_components_of_change.csv"

    logger.info("Loading LPR state time series: %s", lpr_path)
    lpr_state_time = pd.read_parquet(lpr_path)

    logger.info("Building state FIPS mapping from: %s", components_path)
    state_fips_map = build_state_fips_mapping(components_path)

    variants = build_lpr_panel_variants(lpr_state_time, state_fips_map)

    for stem, df in variants.items():
        parquet_path = analysis_dir / f"{stem}.parquet"
        logger.info("Writing %s (%s rows) -> %s", stem, len(df), parquet_path)
        df.to_parquet(parquet_path, index=False)

        if stem.endswith("_states_only") or stem.endswith("_states_only_balanced"):
            csv_path = analysis_dir / f"{stem}.csv"
            df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    with log_execution(__file__, parameters={"series": "dhs_lpr", "variants": "panel"}):
        main()
