"""
PEP regime analysis for county-level migration projections.

Classifies North Dakota counties by economic type (oil, metro, rural) and
calculates regime-aware weighted migration averages from Census PEP data.
This implements Phase 2 of ADR-035: using historical migration regimes
(pre-Bakken, boom, bust/COVID, recovery) to produce forward-looking
migration assumptions for the projection engine.

The key insight is that simple period averages are misleading for ND counties
because the Bakken oil boom created extreme, non-repeatable migration patterns.
Regime-weighted averages with dampening produce more defensible projections.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

__all__ = [
    "OIL_COUNTIES",
    "METRO_COUNTIES",
    "MIGRATION_REGIMES",
    "DEFAULT_REGIME_WEIGHTS",
    "DEFAULT_DAMPENING",
    "classify_counties",
    "calculate_regime_averages",
    "calculate_regime_weighted_average",
    "load_pep_preferred_estimates",
    "generate_regime_analysis_report",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Oil-producing counties (Bakken core)
OIL_COUNTIES: dict[str, str] = {
    "38105": "Williams",
    "38053": "McKenzie",
    "38061": "Mountrail",
    "38025": "Dunn",
    "38089": "Stark",
}

# Metropolitan counties
METRO_COUNTIES: dict[str, str] = {
    "38017": "Cass",  # Fargo
    "38015": "Burleigh",  # Bismarck
    "38035": "Grand Forks",
    "38101": "Ward",  # Minot
}

# Migration regime periods
MIGRATION_REGIMES: dict[str, dict[str, int | str]] = {
    "pre_bakken": {"start": 2000, "end": 2010, "label": "Pre-Bakken"},
    "boom": {"start": 2011, "end": 2015, "label": "Bakken Boom"},
    "bust_covid": {"start": 2016, "end": 2021, "label": "Bust + COVID"},
    "recovery": {"start": 2022, "end": 2025, "label": "Recovery"},
}

# Default regime weights (must sum to 1.0)
DEFAULT_REGIME_WEIGHTS: dict[str, float] = {
    "pre_bakken": 0.15,
    "boom": 0.10,
    "bust_covid": 0.25,
    "recovery": 0.50,
}

# Default dampening factors
DEFAULT_DAMPENING: dict[str, float] = {
    "boom": 0.60,  # Matches SDC 2024
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def classify_counties(
    county_fips_list: list[str],
    custom_classifications: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Classify ND counties by economic type.

    Each county is assigned a type based on its presence in the oil-producing
    or metropolitan county lists.  Counties appearing in both are typed as
    ``"oil_metro"``.  All others are ``"rural"``.

    Args:
        county_fips_list: List of 5-digit county FIPS codes (e.g. ``["38017"]``).
        custom_classifications: Optional dict mapping ``"oil"`` and/or
            ``"metro"`` to custom lists of FIPS codes that override the
            module-level constants.

    Returns:
        DataFrame with columns ``[county_fips, county_type, is_oil, is_metro]``.
    """
    logger.info(f"Classifying {len(county_fips_list)} counties")

    oil_set = set(OIL_COUNTIES.keys())
    metro_set = set(METRO_COUNTIES.keys())

    if custom_classifications is not None:
        if "oil" in custom_classifications:
            oil_set = set(custom_classifications["oil"])
        if "metro" in custom_classifications:
            metro_set = set(custom_classifications["metro"])

    records: list[dict[str, str | bool]] = []
    for fips in county_fips_list:
        is_oil = fips in oil_set
        is_metro = fips in metro_set

        if is_oil and is_metro:
            county_type = "oil_metro"
        elif is_oil:
            county_type = "oil"
        elif is_metro:
            county_type = "metro"
        else:
            county_type = "rural"

        records.append(
            {
                "county_fips": fips,
                "county_type": county_type,
                "is_oil": is_oil,
                "is_metro": is_metro,
            }
        )

    result = pd.DataFrame(records)

    n_oil = result["is_oil"].sum()
    n_metro = result["is_metro"].sum()
    n_rural = (result["county_type"] == "rural").sum()
    logger.info(f"Classification results: oil={n_oil}, metro={n_metro}, rural={n_rural}")

    return result


def calculate_regime_averages(
    pep_data: pd.DataFrame,
    regimes: dict[str, dict[str, int | str]] | None = None,
) -> pd.DataFrame:
    """Calculate migration summary statistics by county and regime period.

    Groups ``pep_data`` by county and regime, then computes mean, median,
    standard deviation, count of years, and total net migration for each
    group.

    Args:
        pep_data: PEP data with at least columns
            ``[county_fips, year, netmig]``.  Should contain only preferred
            estimates (see :func:`load_pep_preferred_estimates`).
        regimes: Optional custom regime definitions.  Defaults to
            :data:`MIGRATION_REGIMES`.

    Returns:
        DataFrame with columns ``[county_fips, regime, mean_netmig,
        median_netmig, std_netmig, n_years, total_netmig]``.
    """
    if regimes is None:
        regimes = MIGRATION_REGIMES

    logger.info(
        f"Calculating regime averages for {pep_data['county_fips'].nunique()} "
        f"counties across {len(regimes)} regimes"
    )

    records: list[dict] = []

    for regime_key, regime_def in regimes.items():
        start = int(regime_def["start"])
        end = int(regime_def["end"])

        mask = (pep_data["year"] >= start) & (pep_data["year"] <= end)
        regime_data = pep_data.loc[mask]

        if regime_data.empty:
            logger.warning(f"No data for regime '{regime_key}' ({start}-{end})")
            continue

        grouped = regime_data.groupby("county_fips")["netmig"]

        for county_fips, group in grouped:
            records.append(
                {
                    "county_fips": county_fips,
                    "regime": regime_key,
                    "mean_netmig": group.mean(),
                    "median_netmig": group.median(),
                    "std_netmig": group.std() if len(group) > 1 else 0.0,
                    "n_years": len(group),
                    "total_netmig": group.sum(),
                }
            )

    result = pd.DataFrame(records)
    logger.info(f"Computed {len(result)} county-regime summary records")
    return result


def calculate_regime_weighted_average(
    regime_averages: pd.DataFrame,
    weights: dict[str, float] | None = None,
    dampening: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute regime-weighted average net migration for each county.

    For every county the function:

    1. Looks up the mean net migration for each regime from
       *regime_averages*.
    2. Applies dampening (multiplicative) to specified regimes **before**
       weighting.
    3. Computes the weighted sum using *weights*.

    Args:
        regime_averages: Output of :func:`calculate_regime_averages`.
        weights: Regime weights keyed by regime name.  Must sum to 1.0.
            Defaults to :data:`DEFAULT_REGIME_WEIGHTS`.
        dampening: Multiplicative dampening factors keyed by regime name.
            Values not present default to 1.0 (no dampening).  Defaults to
            :data:`DEFAULT_DAMPENING`.

    Returns:
        DataFrame with columns ``[county_fips, weighted_avg_netmig,
        pre_bakken_mean, boom_mean, bust_covid_mean, recovery_mean]``.

    Raises:
        ValueError: If *weights* do not sum to 1.0 (within tolerance of 0.001).
    """
    if weights is None:
        weights = DEFAULT_REGIME_WEIGHTS
    if dampening is None:
        dampening = DEFAULT_DAMPENING

    # Validate weights
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        raise ValueError(f"Regime weights must sum to 1.0, got {weight_sum:.4f}")

    logger.info(f"Computing regime-weighted averages with weights={weights}, dampening={dampening}")

    counties = regime_averages["county_fips"].unique()
    records: list[dict] = []

    for county in counties:
        county_data = regime_averages[regime_averages["county_fips"] == county]
        regime_means: dict[str, float] = {}
        for _, row in county_data.iterrows():
            regime_means[row["regime"]] = row["mean_netmig"]

        # Apply dampening then weighting
        weighted_sum = 0.0
        for regime_key, weight in weights.items():
            raw_mean = regime_means.get(regime_key, 0.0)
            damp_factor = dampening.get(regime_key, 1.0)
            dampened_mean = raw_mean * damp_factor
            weighted_sum += dampened_mean * weight

        record: dict = {
            "county_fips": county,
            "weighted_avg_netmig": weighted_sum,
        }

        # Store per-regime means for transparency
        for regime_key in weights:
            col_name = f"{regime_key}_mean"
            record[col_name] = regime_means.get(regime_key, np.nan)

        records.append(record)

    result = pd.DataFrame(records)
    logger.info(
        f"Computed weighted averages for {len(result)} counties, "
        f"overall mean={result['weighted_avg_netmig'].mean():.1f}"
    )
    return result


def load_pep_preferred_estimates(pep_path: str | Path) -> pd.DataFrame:
    """Load PEP parquet file and return only preferred estimates.

    Args:
        pep_path: Path to the PEP county components parquet file
            (e.g. ``data/processed/pep_county_components_2000_2024.parquet``).

    Returns:
        DataFrame filtered to rows where ``is_preferred_estimate`` is ``True``.

    Raises:
        FileNotFoundError: If *pep_path* does not exist.
        ValueError: If the required column ``is_preferred_estimate`` is missing.
    """
    pep_path = Path(pep_path)

    if not pep_path.exists():
        raise FileNotFoundError(f"PEP data file not found: {pep_path}")

    logger.info(f"Loading PEP data from {pep_path}")
    df = pd.read_parquet(pep_path)

    if "is_preferred_estimate" not in df.columns:
        raise ValueError(
            "PEP data is missing 'is_preferred_estimate' column. "
            "Ensure the data was produced by the metadata-enhanced extraction."
        )

    original_len = len(df)
    df = df[df["is_preferred_estimate"]].copy()
    logger.info(f"Filtered to preferred estimates: {len(df)}/{original_len} rows")
    return df


def generate_regime_analysis_report(
    pep_data: pd.DataFrame,
    regime_averages: pd.DataFrame,
    classifications: pd.DataFrame,
    weighted_averages: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Generate a Markdown report summarising the regime analysis.

    The report includes:

    * Summary table of all counties with classification and weighted average.
    * Per-regime statistics broken out by county type (oil / metro / rural).
    * Top 10 highest and lowest migration counties.

    Args:
        pep_data: PEP preferred-estimate data.
        regime_averages: Output of :func:`calculate_regime_averages`.
        classifications: Output of :func:`classify_counties`.
        weighted_averages: Output of :func:`calculate_regime_weighted_average`.
        output_path: File path for the Markdown report.

    Returns:
        The resolved :class:`~pathlib.Path` to which the report was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating regime analysis report at {output_path}")

    # Merge classification with weighted averages
    summary = weighted_averages.merge(classifications, on="county_fips", how="left")

    # Try to attach county name from pep_data
    if "county_name" in pep_data.columns:
        name_map = (
            pep_data[["county_fips", "county_name"]]
            .drop_duplicates("county_fips")
            .set_index("county_fips")["county_name"]
        )
        summary["county_name"] = summary["county_fips"].map(name_map)
    elif "geoid" in pep_data.columns:
        # Fall back to geoid-based lookup
        name_lookup = (
            pep_data.drop_duplicates("county_fips")
            if "county_fips" in pep_data.columns
            else pd.DataFrame()
        )
        if not name_lookup.empty and "county_name" in name_lookup.columns:
            name_map = name_lookup.set_index("county_fips")["county_name"]
            summary["county_name"] = summary["county_fips"].map(name_map)

    if "county_name" not in summary.columns:
        summary["county_name"] = summary["county_fips"]

    lines: list[str] = []
    lines.append("# Migration Regime Analysis: North Dakota Counties")
    lines.append("")
    lines.append("Generated by `pep_regime_analysis.py` (ADR-035 Phase 2).")
    lines.append("")

    # ---- Section 1: Summary table ----
    lines.append("## County Summary")
    lines.append("")
    lines.append("| County | FIPS | Type | Weighted Avg Net Mig |")
    lines.append("|--------|------|------|---------------------:|")

    summary_sorted = summary.sort_values("weighted_avg_netmig", ascending=False)
    for _, row in summary_sorted.iterrows():
        lines.append(
            f"| {row['county_name']} | {row['county_fips']} "
            f"| {row['county_type']} | {row['weighted_avg_netmig']:,.1f} |"
        )
    lines.append("")

    # ---- Section 2: Per-regime statistics by county type ----
    lines.append("## Regime Statistics by County Type")
    lines.append("")

    merged_regime = regime_averages.merge(
        classifications[["county_fips", "county_type"]], on="county_fips", how="left"
    )

    for regime_key in MIGRATION_REGIMES:
        regime_label = MIGRATION_REGIMES[regime_key]["label"]
        regime_rows = merged_regime[merged_regime["regime"] == regime_key]
        if regime_rows.empty:
            continue

        lines.append(f"### {regime_label}")
        lines.append("")
        lines.append("| County Type | Mean Net Mig | Median | Std Dev | Counties |")
        lines.append("|-------------|-------------:|-------:|--------:|---------:|")

        for ctype in ["oil", "metro", "rural", "oil_metro"]:
            ctype_rows = regime_rows[regime_rows["county_type"] == ctype]
            if ctype_rows.empty:
                continue
            lines.append(
                f"| {ctype} "
                f"| {ctype_rows['mean_netmig'].mean():,.1f} "
                f"| {ctype_rows['median_netmig'].mean():,.1f} "
                f"| {ctype_rows['std_netmig'].mean():,.1f} "
                f"| {ctype_rows['county_fips'].nunique()} |"
            )
        lines.append("")

    # ---- Section 3: Top 10 highest / lowest ----
    lines.append("## Top 10 Highest Weighted Migration")
    lines.append("")
    lines.append("| Rank | County | FIPS | Type | Weighted Avg |")
    lines.append("|-----:|--------|------|------|-------------:|")
    top10 = summary_sorted.head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['county_name']} | {row['county_fips']} "
            f"| {row['county_type']} | {row['weighted_avg_netmig']:,.1f} |"
        )
    lines.append("")

    lines.append("## Top 10 Lowest Weighted Migration")
    lines.append("")
    lines.append("| Rank | County | FIPS | Type | Weighted Avg |")
    lines.append("|-----:|--------|------|------|-------------:|")
    bottom10 = summary_sorted.tail(10).iloc[::-1]
    for rank, (_, row) in enumerate(bottom10.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['county_name']} | {row['county_fips']} "
            f"| {row['county_type']} | {row['weighted_avg_netmig']:,.1f} |"
        )
    lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    logger.info(f"Report written to {output_path} ({len(lines)} lines)")
    return output_path
