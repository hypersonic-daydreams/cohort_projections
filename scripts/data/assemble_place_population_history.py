#!/usr/bin/env python3
"""
Assemble ND place population history (2000-2024) for PP-003 Phase 1.

Created: 2026-02-28
ADR: 033 (PP3-S02/PP3-S04 data assembly implementation)
Author: Codex / N. Haarstad

Purpose
-------
Build a unified long-format place population history table for North Dakota
using approved S02 handoff rules:
- 2000-2009 from `sub-est00int`
- 2010-2019 from `sub-est2020int`
- 2020-2024 from `sub-est2024`

The output provides one row per place-year with county assignment from the
IMP-01 crosswalk and source-vintage provenance for each record.

Method
------
1. Load the three source PEP place datasets from the shared Census parquet
   archive.
2. Filter to ND place rows (STATE=38, SUMLEV=162 where available).
3. Reshape wide `POPESTIMATEYYYY` columns to long format with explicit year
   windows per S02.
4. Concatenate windows, enforce continuity (2000-2024), and validate missing
   values/duplicate keys.
5. Join place->county mapping from the IMP-01 crosswalk and carry
   `historical_only` flags.
6. Write `data/processed/place_population_history_2000_2024.parquet`.
7. Compute 2024 populations for active places and update confidence tiers
   (`confidence_tier`, `tier_boundary`) in the crosswalk (IMP-04).

Key design decisions
--------------------
- Lock `sub-est2020int` as canonical for 2010-2019 (S02 Gap D rule).
- Retain dissolved places as historical-only records through 2019 to preserve
  backtest/training continuity.
- Store a `vintage_source` field for auditable handoff provenance.

Validation results
------------------
- Validation is runtime-enforced. The script logs:
  - row count and place count,
  - min/max year and gap checks,
  - null-population checks,
  - join coverage to county crosswalk.
- Expected production target from S02 note: 8,915 rows, years 2000-2024.

Inputs
------
- Shared Census parquet archive (defaults):
  - `~/workspace/shared-data/census/popest/parquet/2000-2010/place/sub-est00int.parquet`
  - `~/workspace/shared-data/census/popest/parquet/2010-2020/place/sub-est2020int.parquet`
  - `~/workspace/shared-data/census/popest/parquet/2020-2024/place/sub-est2024.parquet`
- Crosswalk:
  - `data/processed/geographic/place_county_crosswalk_2020.csv`

Outputs
-------
- `data/processed/place_population_history_2000_2024.parquet`
- Updated `data/processed/geographic/place_county_crosswalk_2020.csv`
  with `confidence_tier` and `tier_boundary` (IMP-04).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from scripts.data.build_place_county_crosswalk import (
        add_tiers_to_crosswalk,
        assign_confidence_tiers,
    )
except ModuleNotFoundError:  # pragma: no cover - CLI fallback when run as a file path
    from build_place_county_crosswalk import (  # type: ignore[no-redef]
        add_tiers_to_crosswalk,
        assign_confidence_tiers,
    )

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ND_STATE_FIPS = "38"


def _normalize_fips(value: str | int | float | None, width: int) -> str | None:
    """Normalize numeric/text values to zero-padded FIPS-like codes."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)[-width:]


def _filter_nd_place_rows(df: pd.DataFrame, state_fips: str = ND_STATE_FIPS) -> pd.DataFrame:
    """Filter source DataFrame to ND place-level rows only."""
    required = {"STATE", "PLACE"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for place filtering: {missing_str}")

    out = df.copy()
    out["STATE"] = out["STATE"].map(lambda v: _normalize_fips(v, 2))
    out["PLACE"] = out["PLACE"].map(lambda v: _normalize_fips(v, 5))
    out = out[out["STATE"] == state_fips].copy()

    if "SUMLEV" in out.columns:
        out = out[out["SUMLEV"] == "162"].copy()

    out["place_fips"] = out["STATE"] + out["PLACE"]
    out["place_name"] = out.get("NAME", "").astype(str).str.strip()
    return out


def reshape_place_vintage_to_long(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    vintage_source: str,
) -> pd.DataFrame:
    """Reshape one PEP place vintage from wide POPESTIMATE columns to long format."""
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    filtered = _filter_nd_place_rows(df)
    years = list(range(start_year, end_year + 1))
    pop_cols = [f"POPESTIMATE{year}" for year in years]
    missing_pop_cols = [col for col in pop_cols if col not in filtered.columns]
    if missing_pop_cols:
        missing_str = ", ".join(missing_pop_cols[:5])
        suffix = " ..." if len(missing_pop_cols) > 5 else ""
        raise ValueError(f"{vintage_source} missing required columns: {missing_str}{suffix}")

    long = filtered.melt(
        id_vars=["place_fips", "place_name"],
        value_vars=pop_cols,
        var_name="population_column",
        value_name="population",
    )
    long["year"] = long["population_column"].str.extract(r"(\d{4})").astype(int)
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    long["vintage_source"] = vintage_source

    return long[["place_fips", "place_name", "year", "population", "vintage_source"]]


def assemble_place_population_history(
    sub_est00int: pd.DataFrame,
    sub_est2020int: pd.DataFrame,
    sub_est2024: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble 2000-2024 place population history from approved handoff windows."""
    vintage_2000s = reshape_place_vintage_to_long(
        df=sub_est00int,
        start_year=2000,
        end_year=2009,
        vintage_source="sub-est00int",
    )
    vintage_2010s = reshape_place_vintage_to_long(
        df=sub_est2020int,
        start_year=2010,
        end_year=2019,
        vintage_source="sub-est2020int",
    )
    vintage_2020s = reshape_place_vintage_to_long(
        df=sub_est2024,
        start_year=2020,
        end_year=2024,
        vintage_source="sub-est2024",
    )

    history = pd.concat(
        [vintage_2000s, vintage_2010s, vintage_2020s],
        ignore_index=True,
    )
    history = history.sort_values(["place_fips", "year"]).reset_index(drop=True)
    return history


def attach_crosswalk(
    history: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Attach county_fips and historical-only flags from crosswalk."""
    required_history = {"place_fips", "year", "population"}
    missing_history = required_history - set(history.columns)
    if missing_history:
        raise ValueError(f"History missing required columns: {sorted(missing_history)}")

    required_crosswalk = {"place_fips", "county_fips", "historical_only"}
    missing_crosswalk = required_crosswalk - set(crosswalk.columns)
    if missing_crosswalk:
        raise ValueError(f"Crosswalk missing required columns: {sorted(missing_crosswalk)}")

    mapping = crosswalk[["place_fips", "county_fips", "historical_only"]].drop_duplicates(
        subset=["place_fips"],
    )
    joined = history.merge(mapping, on="place_fips", how="left", validate="many_to_one")
    return joined


def validate_place_history(history: pd.DataFrame) -> None:
    """Validate continuity, nulls, and uniqueness invariants."""
    required = {
        "place_fips",
        "place_name",
        "county_fips",
        "year",
        "population",
        "vintage_source",
        "historical_only",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History output missing required columns: {sorted(missing)}")

    year_min = int(history["year"].min())
    year_max = int(history["year"].max())
    if (year_min, year_max) != (2000, 2024):
        raise ValueError(f"Expected year range 2000-2024, found {year_min}-{year_max}")

    observed_years = set(history["year"].astype(int).unique())
    expected_years = set(range(2000, 2025))
    if observed_years != expected_years:
        missing_years = sorted(expected_years - observed_years)
        raise ValueError(f"Missing years in history output: {missing_years}")

    if history["population"].isna().any():
        raise ValueError("History output contains null population values.")
    if history["county_fips"].isna().any():
        raise ValueError("History output contains null county_fips values.")

    duplicate_keys = history.duplicated(subset=["place_fips", "year"])
    if duplicate_keys.any():
        dup_count = int(duplicate_keys.sum())
        raise ValueError(f"History output has {dup_count} duplicate place_fips/year rows.")


def build_tiers_from_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build 2024 tier assignments from active place populations."""
    required = {"place_fips", "year", "population", "historical_only"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History missing required tier columns: {sorted(missing)}")

    active_2024 = history[
        (history["year"] == 2024) & (~history["historical_only"].fillna(False))
    ].copy()
    active_2024 = active_2024.rename(columns={"population": "population_2024"})
    return assign_confidence_tiers(active_2024[["place_fips", "population_2024"]])


def parse_args() -> argparse.Namespace:
    """Parse command-line args for place history assembly."""
    default_popest = Path.home() / "workspace" / "shared-data" / "census" / "popest" / "parquet"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sub-est00int",
        type=Path,
        default=default_popest / "2000-2010" / "place" / "sub-est00int.parquet",
        help="Path to sub-est00int parquet.",
    )
    parser.add_argument(
        "--sub-est2020int",
        type=Path,
        default=default_popest / "2010-2020" / "place" / "sub-est2020int.parquet",
        help="Path to sub-est2020int parquet.",
    )
    parser.add_argument(
        "--sub-est2024",
        type=Path,
        default=default_popest / "2020-2024" / "place" / "sub-est2024.parquet",
        help="Path to sub-est2024 parquet.",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "geographic"
        / "place_county_crosswalk_2020.csv",
        help="Path to primary place-county crosswalk CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "place_population_history_2000_2024.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--skip-tier-update",
        action="store_true",
        help="Do not update crosswalk with confidence-tier columns.",
    )
    return parser.parse_args()


def main() -> None:
    """Run place-history assembly and optional tier update."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    sub_est00int = pd.read_parquet(args.sub_est00int)
    sub_est2020int = pd.read_parquet(args.sub_est2020int)
    sub_est2024 = pd.read_parquet(args.sub_est2024)

    history = assemble_place_population_history(
        sub_est00int=sub_est00int,
        sub_est2020int=sub_est2020int,
        sub_est2024=sub_est2024,
    )

    crosswalk = pd.read_csv(args.crosswalk, dtype={"place_fips": str, "county_fips": str})
    history = attach_crosswalk(history, crosswalk)
    history["place_fips"] = history["place_fips"].map(lambda v: _normalize_fips(v, 7))
    history["county_fips"] = history["county_fips"].map(lambda v: _normalize_fips(v, 5))
    history["historical_only"] = history["historical_only"].fillna(False).astype(bool)

    validate_place_history(history)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(args.output, index=False)
    logger.info("Wrote place history: %s (%d rows)", args.output, len(history))

    if not args.skip_tier_update:
        tiers = build_tiers_from_history(history)
        updated_crosswalk = add_tiers_to_crosswalk(crosswalk, tiers)
        updated_crosswalk.to_csv(args.crosswalk, index=False)
        tier_counts = updated_crosswalk.loc[
            ~updated_crosswalk["historical_only"].fillna(False),
            "confidence_tier",
        ].value_counts()
        logger.info(
            "Updated crosswalk tiers at %s with counts: %s",
            args.crosswalk,
            dict(sorted(tier_counts.to_dict().items())),
        )


if __name__ == "__main__":
    main()
