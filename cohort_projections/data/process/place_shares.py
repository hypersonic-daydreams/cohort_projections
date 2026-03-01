"""
Historical place-share computation for PP-003 Phase 1.

Computes place share of county population for each place-year, applies epsilon
clamping for logit modeling stability, and derives balance-of-county shares.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

ND_STATE_FIPS = "38"


def _normalize_fips(value: str | int | float | None, width: int) -> str | None:
    """Normalize values to zero-padded FIPS-like codes."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)[-width:]


def normalize_county_population_history(
    county_population_history: pd.DataFrame,
    year_min: int = 2000,
    year_max: int = 2024,
    state_fips: str = ND_STATE_FIPS,
) -> pd.DataFrame:
    """
    Normalize county population history to long format.

    Supports either:
    1. Long format with ``county_fips``, ``year``, and a population column.
    2. Wide PEP format with ``STATE``, ``COUNTY`` and ``POPESTIMATEYYYY`` columns.
    """
    df = county_population_history.copy()

    if {"county_fips", "year"}.issubset(df.columns):
        pop_col = next(
            (
                col
                for col in [
                    "county_population",
                    "population",
                    "total_population",
                    "pop_total",
                    "POPESTIMATE",
                ]
                if col in df.columns
            ),
            None,
        )
        if pop_col is None:
            raise ValueError(
                "Long county-population input must include one of: "
                "county_population, population, total_population, pop_total, POPESTIMATE."
            )

        long = df[["county_fips", "year", pop_col]].copy()
        long["county_fips"] = long["county_fips"].map(lambda v: _normalize_fips(v, 5))
        long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
        long["county_population"] = pd.to_numeric(long[pop_col], errors="coerce")
        long = long.dropna(subset=["county_fips", "year", "county_population"]).copy()
        long["year"] = long["year"].astype(int)

    elif {"STATE", "COUNTY"}.issubset(df.columns):
        df["STATE"] = df["STATE"].map(lambda v: _normalize_fips(v, 2))
        df["COUNTY"] = df["COUNTY"].map(lambda v: _normalize_fips(v, 3))
        df = df[(df["STATE"] == state_fips) & (df["COUNTY"] != "000")].copy()

        pop_cols = [
            col
            for col in df.columns
            if col.startswith("POPESTIMATE")
            and col[-4:].isdigit()
            and year_min <= int(col[-4:]) <= year_max
        ]
        if not pop_cols:
            raise ValueError(
                f"Wide county-population input has no POPESTIMATE columns in range {year_min}-{year_max}."
            )

        long = df.melt(
            id_vars=["STATE", "COUNTY"],
            value_vars=pop_cols,
            var_name="population_column",
            value_name="county_population",
        )
        long["year"] = long["population_column"].str.extract(r"(\d{4})").astype(int)
        long["county_fips"] = long["STATE"] + long["COUNTY"]
        long["county_population"] = pd.to_numeric(long["county_population"], errors="coerce")
        long = long.dropna(subset=["county_fips", "year", "county_population"]).copy()
        long = long[["county_fips", "year", "county_population"]]

    else:
        raise ValueError(
            "Unsupported county population schema. "
            "Expected either long (county_fips/year/population) or wide PEP (STATE/COUNTY/POPESTIMATE*)."
        )

    long = long[(long["year"] >= year_min) & (long["year"] <= year_max)].copy()
    long = long.sort_values(["county_fips", "year"]).reset_index(drop=True)

    if long["county_population"].le(0).any():
        raise ValueError("County population must be strictly positive for share computation.")

    if long.duplicated(subset=["county_fips", "year"]).any():
        raise ValueError("County population history contains duplicate county_fips/year rows.")

    return long


def compute_historical_shares(
    place_population_history: pd.DataFrame,
    county_population_history: pd.DataFrame,
    epsilon: float = 0.001,
    year_min: int = 2000,
    year_max: int = 2024,
) -> pd.DataFrame:
    """
    Compute historical place shares and balance-of-county shares.

    Args:
        place_population_history: Long place history with columns including
            ``place_fips``, ``county_fips``, ``year``, ``population``.
        county_population_history: County history in long or wide schema.
        epsilon: Lower/upper clamp bound in share space for logit stability.
        year_min: First year to include.
        year_max: Last year to include.

    Returns:
        DataFrame with place rows and balance rows. Core columns:
        ``county_fips``, ``place_fips``, ``year``, ``row_type``,
        ``share_raw``, ``share`` (epsilon-clamped), ``county_population``.
    """
    if not (0 < epsilon < 0.5):
        raise ValueError("epsilon must be in (0, 0.5)")

    place = place_population_history.copy()
    required_place_cols = {"place_fips", "county_fips", "year", "population"}
    missing_place_cols = required_place_cols - set(place.columns)
    if missing_place_cols:
        missing = ", ".join(sorted(missing_place_cols))
        raise ValueError(f"Place history missing required columns: {missing}")

    place["place_fips"] = place["place_fips"].map(lambda v: _normalize_fips(v, 7))
    place["county_fips"] = place["county_fips"].map(lambda v: _normalize_fips(v, 5))
    place["year"] = pd.to_numeric(place["year"], errors="coerce").astype("Int64")
    place["population"] = pd.to_numeric(place["population"], errors="coerce")
    place = place.dropna(subset=["place_fips", "county_fips", "year", "population"]).copy()
    place["year"] = place["year"].astype(int)
    place = place[(place["year"] >= year_min) & (place["year"] <= year_max)].copy()

    county = normalize_county_population_history(
        county_population_history=county_population_history,
        year_min=year_min,
        year_max=year_max,
    )

    merged = place.merge(
        county,
        on=["county_fips", "year"],
        how="left",
        validate="many_to_one",
    )
    if merged["county_population"].isna().any():
        missing_pairs = merged.loc[
            merged["county_population"].isna(),
            ["county_fips", "year"],
        ].drop_duplicates()
        raise ValueError(
            "Missing county populations for place-history rows: "
            f"{missing_pairs.to_dict(orient='records')[:5]}"
        )

    merged["share_raw"] = merged["population"] / merged["county_population"]
    if merged["share_raw"].lt(0).any():
        raise ValueError("Negative place share detected; population inputs must be non-negative.")
    if merged["share_raw"].gt(1).any():
        over = merged.loc[merged["share_raw"] > 1, ["place_fips", "county_fips", "year", "share_raw"]]
        raise ValueError(
            "Place share exceeds 1.0 for one or more rows; check county totals. "
            f"Sample: {over.head(5).to_dict(orient='records')}"
        )

    merged["share"] = merged["share_raw"].clip(lower=epsilon, upper=1 - epsilon)
    merged["row_type"] = "place"

    # Carry optional columns where present.
    optional_cols = [
        col
        for col in ["place_name", "historical_only", "vintage_source"]
        if col in merged.columns
    ]
    place_rows = merged[
        [
            "county_fips",
            "place_fips",
            "year",
            "row_type",
            *optional_cols,
            "population",
            "county_population",
            "share_raw",
            "share",
        ]
    ].copy()

    sums = place_rows.groupby(["county_fips", "year"], as_index=False)["share_raw"].sum()
    county_unique = county.drop_duplicates(subset=["county_fips", "year"])
    sums = sums.merge(county_unique, on=["county_fips", "year"], how="left", validate="one_to_one")
    sums["balance_share_raw"] = 1.0 - sums["share_raw"]
    if sums["balance_share_raw"].lt(-1e-9).any():
        bad = sums[sums["balance_share_raw"] < -1e-9][["county_fips", "year", "balance_share_raw"]]
        raise ValueError(
            "Balance share is negative beyond tolerance for one or more county-years. "
            f"Sample: {bad.head(5).to_dict(orient='records')}"
        )

    sums["balance_share_raw"] = sums["balance_share_raw"].clip(lower=0.0)
    sums["balance_share"] = sums["balance_share_raw"].clip(lower=epsilon, upper=1 - epsilon)
    sums["balance_population"] = sums["balance_share_raw"] * sums["county_population"]

    balance_rows = pd.DataFrame(
        {
            "county_fips": sums["county_fips"],
            "place_fips": pd.NA,
            "year": sums["year"],
            "row_type": "balance_of_county",
            "population": sums["balance_population"],
            "county_population": sums["county_population"],
            "share_raw": sums["balance_share_raw"],
            "share": sums["balance_share"],
        }
    )

    if "historical_only" in place_rows.columns:
        balance_rows["historical_only"] = False
    if "vintage_source" in place_rows.columns:
        balance_rows["vintage_source"] = "derived_balance"
    if "place_name" in place_rows.columns:
        balance_rows["place_name"] = pd.NA

    output = pd.concat([place_rows, balance_rows], ignore_index=True, sort=False)
    output = output.sort_values(["county_fips", "year", "row_type", "place_fips"]).reset_index(drop=True)

    logger.info(
        "Computed historical shares: %d place rows, %d balance rows",
        len(place_rows),
        len(balance_rows),
    )
    return output


def load_county_population_history(county_population_path: Path) -> pd.DataFrame:
    """Load county population history from parquet or CSV path."""
    suffix = county_population_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(county_population_path)
    if suffix == ".csv":
        return pd.read_csv(county_population_path, dtype=str)
    raise ValueError(f"Unsupported county population file type: {county_population_path}")

