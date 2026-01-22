"""Generate derived LaTeX macros for the journal article.

This script computes a small set of descriptive values directly from the
on-hand Census PEP state totals files and writes them to `derived_stats.tex`.

Intended use: called automatically by `compile.sh` (best-effort) and usable
standalone via `uv run python generate_derived_stats.py`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cohort_projections.utils import ConfigLoader

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW_YEARS = 5


@dataclass(frozen=True)
class PepShareSummary:
    """Computed PEP shares for an average period and latest year."""

    share_start_year: int
    share_end_year: int
    share_n_years: int
    latest_year: int
    nd_share_us_international_migration_pct: float
    nd_share_us_population_pct: float
    nd_share_us_international_migration_pct_mean: float
    nd_share_us_population_pct_mean: float
    context_window_years: int
    natural_increase_mean_prior_window: float
    natural_increase_mean_last_window: float
    domestic_migration_mean_last_double_window: float
    domestic_migration_negative_years_last_double_window: int
    domestic_migration_years_last_double_window: int


def _project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[4]


def _available_years(df: pd.DataFrame) -> list[int]:
    """Return years present in both `POPESTIMATE` and `INTERNATIONALMIG` columns."""
    pop_years: set[int] = set()
    mig_years: set[int] = set()

    pattern_pop = re.compile(r"^POPESTIMATE(?P<year>\d{4})$")
    pattern_mig = re.compile(r"^INTERNATIONALMIG(?P<year>\d{4})$")

    for column in df.columns:
        match = pattern_pop.match(column)
        if match:
            pop_years.add(int(match.group("year")))
            continue
        match = pattern_mig.match(column)
        if match:
            mig_years.add(int(match.group("year")))

    return sorted(pop_years & mig_years)


def _load_vintage(path: Path) -> pd.DataFrame:
    """Load a single PEP vintage file with minimal validation."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "NAME" not in df.columns:
        raise ValueError(f"Missing NAME column: {path}")
    return df


def _extract_year_values(
    df: pd.DataFrame,
    year: int,
) -> tuple[float, float, float, float] | None:
    """Extract (us_pop, nd_pop, us_intmig, nd_intmig) for a given year if present."""
    pop_col = f"POPESTIMATE{year}"
    mig_col = f"INTERNATIONALMIG{year}"
    if pop_col not in df.columns or mig_col not in df.columns:
        return None

    required_names = {"United States", "North Dakota"}
    names = set(df["NAME"].astype(str))
    if not required_names.issubset(names):
        raise ValueError("Missing required rows: United States and/or North Dakota.")

    us = df.loc[df["NAME"].eq("United States")].iloc[0]
    nd = df.loc[df["NAME"].eq("North Dakota")].iloc[0]

    us_pop = float(us[pop_col])
    nd_pop = float(nd[pop_col])
    us_int_mig = float(us[mig_col])
    nd_int_mig = float(nd[mig_col])
    return us_pop, nd_pop, us_int_mig, nd_int_mig


def _extract_nd_context_values(
    df: pd.DataFrame,
    year: int,
) -> tuple[float, float, float] | None:
    """Extract (nd_births, nd_deaths, nd_domesticmig) for a given year if present."""
    births_col = f"BIRTHS{year}"
    deaths_col = f"DEATHS{year}"
    dom_col = f"DOMESTICMIG{year}"
    if births_col not in df.columns or deaths_col not in df.columns or dom_col not in df.columns:
        return None

    if "North Dakota" not in set(df["NAME"].astype(str)):
        raise ValueError("Missing required row: North Dakota.")
    nd = df.loc[df["NAME"].eq("North Dakota")].iloc[0]
    return float(nd[births_col]), float(nd[deaths_col]), float(nd[dom_col])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of empty list.")
    return sum(values) / len(values)


def compute_pep_share_summary(
    vintage_paths: list[Path],
    *,
    share_start_year: int = 2010,
    context_window_years: int = DEFAULT_CONTEXT_WINDOW_YEARS,
) -> PepShareSummary:
    """Compute North Dakota shares using a union of multiple PEP vintage files.

    Strategy: ingest vintages in priority order (newest first) and, for any
    overlapping year, keep the first-seen values (i.e., the newer revision).

    Args:
        vintage_paths: Paths to PEP vintage CSVs in priority order (newest first).
        share_start_year: Start year for the mean-share period.

    Returns:
        Summary including latest-year shares and mean shares over the requested period.

    Raises:
        FileNotFoundError: If any provided vintage path is missing.
        ValueError: If required rows/columns are missing or no usable years exist.
    """
    if not vintage_paths:
        raise ValueError("No vintage paths provided.")

    share_year_values: dict[int, tuple[float, float, float, float]] = {}
    context_year_values: dict[int, tuple[float, float, float]] = {}
    for path in vintage_paths:
        df = _load_vintage(path)
        for year in _available_years(df):
            if year not in share_year_values:
                share_values = _extract_year_values(df, year)
                if share_values is not None:
                    share_year_values[year] = share_values
            if year not in context_year_values:
                context_values = _extract_nd_context_values(df, year)
                if context_values is not None:
                    context_year_values[year] = context_values

    usable_share_years = sorted(share_year_values.keys())
    if not usable_share_years:
        raise ValueError("No usable PEP years found across vintages.")

    latest_year = max(usable_share_years)
    share_years = [year for year in usable_share_years if year >= share_start_year]
    if not share_years:
        raise ValueError(
            f"No usable years found at or after share_start_year={share_start_year}."
        )

    intl_shares: list[float] = []
    pop_shares: list[float] = []

    for year in share_years:
        us_pop, nd_pop, us_int_mig, nd_int_mig = share_year_values[year]
        if us_pop <= 0:
            raise ValueError(f"Invalid U.S. population for {year}: {us_pop}")
        if us_int_mig <= 0:
            raise ValueError(f"Invalid U.S. international migration for {year}: {us_int_mig}")
        intl_shares.append(nd_int_mig / us_int_mig * 100.0)
        pop_shares.append(nd_pop / us_pop * 100.0)

    us_pop_latest, nd_pop_latest, us_int_latest, nd_int_latest = share_year_values[latest_year]

    if context_window_years <= 0:
        raise ValueError("context_window_years must be positive.")

    prior_window_years = list(
        range(latest_year - 2 * context_window_years + 1, latest_year - context_window_years + 1)
    )
    last_window_years = list(range(latest_year - context_window_years + 1, latest_year + 1))
    last_double_window_years = prior_window_years + last_window_years

    missing_context_years = [
        year for year in last_double_window_years if year not in context_year_values
    ]
    if missing_context_years:
        raise ValueError(
            "Missing required BIRTHS/DEATHS/DOMESTICMIG values for years: "
            + ", ".join(str(year) for year in missing_context_years)
        )

    natural_increase_prior = [
        (context_year_values[year][0] - context_year_values[year][1])
        for year in prior_window_years
    ]
    natural_increase_last = [
        (context_year_values[year][0] - context_year_values[year][1])
        for year in last_window_years
    ]
    domestic_migration_last10 = [context_year_values[year][2] for year in last_double_window_years]

    return PepShareSummary(
        share_start_year=min(share_years),
        share_end_year=max(share_years),
        share_n_years=len(share_years),
        latest_year=latest_year,
        nd_share_us_international_migration_pct=nd_int_latest / us_int_latest * 100.0,
        nd_share_us_population_pct=nd_pop_latest / us_pop_latest * 100.0,
        nd_share_us_international_migration_pct_mean=_mean(intl_shares),
        nd_share_us_population_pct_mean=_mean(pop_shares),
        context_window_years=context_window_years,
        natural_increase_mean_prior_window=_mean(natural_increase_prior),
        natural_increase_mean_last_window=_mean(natural_increase_last),
        domestic_migration_mean_last_double_window=_mean(domestic_migration_last10),
        domestic_migration_negative_years_last_double_window=sum(
            1 for value in domestic_migration_last10 if value < 0
        ),
        domestic_migration_years_last_double_window=len(domestic_migration_last10),
    )


def write_derived_stats(summary: PepShareSummary, output_path: Path) -> None:
    """Write derived LaTeX macros to a `.tex` include file.

    Args:
        summary: Computed shares summary.
        output_path: Output `.tex` file path.
    """
    prior_window_start_year = summary.latest_year - 2 * summary.context_window_years + 1
    prior_window_end_year = summary.latest_year - summary.context_window_years
    last_window_start_year = summary.latest_year - summary.context_window_years + 1
    last_window_end_year = summary.latest_year

    last_double_window_start_year = prior_window_start_year
    last_double_window_end_year = last_window_end_year

    def fmt_int(value: float) -> str:
        return f"{int(round(value)):,}"

    def fmt_signed_int(value: float) -> str:
        rounded = int(round(value))
        return f"{rounded:+,}".replace("+", "")

    content = (
        "% Auto-generated by generate_derived_stats.py (do not edit by hand)\n"
        f"\\newcommand{{\\PEPLatestYear}}{{{summary.latest_year}}}\n"
        f"\\newcommand{{\\PEPShareYearStart}}{{{summary.share_start_year}}}\n"
        f"\\newcommand{{\\PEPShareYearEnd}}{{{summary.share_end_year}}}\n"
        f"\\newcommand{{\\PEPShareNYears}}{{{summary.share_n_years}}}\n"
        f"\\newcommand{{\\NDShareUSIntlMigPctLatest}}{{{summary.nd_share_us_international_migration_pct:.2f}}}\n"
        f"\\newcommand{{\\NDShareUSPopPctLatest}}{{{summary.nd_share_us_population_pct:.2f}}}\n"
        f"\\newcommand{{\\NDShareUSIntlMigPctMean}}{{{summary.nd_share_us_international_migration_pct_mean:.2f}}}\n"
        f"\\newcommand{{\\NDShareUSPopPctMean}}{{{summary.nd_share_us_population_pct_mean:.2f}}}\n"
        f"\\newcommand{{\\PEPContextWindowYears}}{{{summary.context_window_years}}}\n"
        f"\\newcommand{{\\PEPPriorFiveYearStart}}{{{prior_window_start_year}}}\n"
        f"\\newcommand{{\\PEPPriorFiveYearEnd}}{{{prior_window_end_year}}}\n"
        f"\\newcommand{{\\PEPLastFiveYearStart}}{{{last_window_start_year}}}\n"
        f"\\newcommand{{\\PEPLastFiveYearEnd}}{{{last_window_end_year}}}\n"
        f"\\newcommand{{\\PEPLastTenYearStart}}{{{last_double_window_start_year}}}\n"
        f"\\newcommand{{\\PEPLastTenYearEnd}}{{{last_double_window_end_year}}}\n"
        f"\\newcommand{{\\NDNaturalIncreaseMeanPriorFive}}{{{fmt_int(summary.natural_increase_mean_prior_window)}}}\n"
        f"\\newcommand{{\\NDNaturalIncreaseMeanLastFive}}{{{fmt_int(summary.natural_increase_mean_last_window)}}}\n"
        f"\\newcommand{{\\NDDomesticMigMeanLastTen}}{{{fmt_signed_int(summary.domestic_migration_mean_last_double_window)}}}\n"
        f"\\newcommand{{\\NDDomesticMigNegYearsLastTen}}{{{summary.domestic_migration_negative_years_last_double_window}}}\n"
        f"\\newcommand{{\\NDDomesticMigYearsLastTen}}{{{summary.domestic_migration_years_last_double_window}}}\n"
    )
    output_path.write_text(content, encoding="utf-8")


def _default_vintage_paths(project_root: Path) -> list[Path]:
    """Return default PEP vintage file paths using projection config."""
    config = ConfigLoader().get_projection_config()
    raw_dir = (
        config.get("data_sources", {})
        .get("census_components", {})
        .get("raw_dir", "data/raw/immigration/census_population_estimates")
    )
    source_dir = project_root / raw_dir
    return [
        source_dir / "NST-EST2024-ALLDATA.csv",
        source_dir / "NST-EST2020-ALLDATA.csv",
        source_dir / "NST-EST2009-ALLDATA.csv",
    ]


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = _project_root()
    vintage_paths = _default_vintage_paths(project_root)
    output_path = Path(__file__).resolve().parent / "derived_stats.tex"

    summary = compute_pep_share_summary(vintage_paths, share_start_year=2010)
    write_derived_stats(summary, output_path)
    LOGGER.info(
        "Wrote %s (PEP %s-%s mean shares: intl_mig=%.3f%%, pop=%.3f%%; latest %s: intl_mig=%.3f%%, pop=%.3f%%)",
        output_path.as_posix(),
        summary.share_start_year,
        summary.share_end_year,
        summary.nd_share_us_international_migration_pct_mean,
        summary.nd_share_us_population_pct_mean,
        summary.latest_year,
        summary.nd_share_us_international_migration_pct,
        summary.nd_share_us_population_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
