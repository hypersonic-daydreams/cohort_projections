from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def derived_stats_module():
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root
        / "sdc_2024_replication"
        / "scripts"
        / "statistical_analysis"
        / "journal_article"
        / "generate_derived_stats.py"
    )
    spec = importlib.util.spec_from_file_location("generate_derived_stats", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_minimal_pep_vintage(
    path: Path,
    *,
    years: list[int],
    us_pop: dict[int, float],
    nd_pop: dict[int, float],
    us_intmig: dict[int, float],
    nd_intmig: dict[int, float],
    nd_births: dict[int, float] | None = None,
    nd_deaths: dict[int, float] | None = None,
    nd_domesticmig: dict[int, float] | None = None,
) -> None:
    header = (
        ["NAME"]
        + [f"POPESTIMATE{y}" for y in years]
        + [f"INTERNATIONALMIG{y}" for y in years]
        + [f"BIRTHS{y}" for y in years]
        + [f"DEATHS{y}" for y in years]
        + [f"DOMESTICMIG{y}" for y in years]
    )
    rows = []
    for name in ["United States", "North Dakota"]:
        values = [name]
        for y in years:
            values.append(us_pop[y] if name == "United States" else nd_pop[y])
        for y in years:
            values.append(us_intmig[y] if name == "United States" else nd_intmig[y])
        for y in years:
            if name == "United States":
                values.append(0)
            else:
                assert nd_births is not None
                values.append(nd_births[y])
        for y in years:
            if name == "United States":
                values.append(0)
            else:
                assert nd_deaths is not None
                values.append(nd_deaths[y])
        for y in years:
            if name == "United States":
                values.append(0)
            else:
                assert nd_domesticmig is not None
                values.append(nd_domesticmig[y])
        rows.append(values)

    # Write CSV without pandas to keep test dependencies minimal.
    content = [",".join(header)]
    for row in rows:
        content.append(",".join(str(x) for x in row))
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def test_compute_pep_share_summary_prefers_newer_overlap(derived_stats_module, tmp_path):
    newer = tmp_path / "NST-EST2024-ALLDATA.csv"
    older = tmp_path / "NST-EST2020-ALLDATA.csv"

    # Overlap year 2020 appears in both; newer should win.
    years_newer = [2020, 2021, 2022, 2023, 2024]
    years_older = [2015, 2016, 2017, 2018, 2019, 2020]

    def make_series(years: list[int], value: float) -> dict[int, float]:
        return dict.fromkeys(years, value)

    _write_minimal_pep_vintage(
        newer,
        years=years_newer,
        us_pop=make_series(years_newer, 1000),
        nd_pop=make_series(years_newer, 10),
        us_intmig={2020: 100, 2021: 100, 2022: 100, 2023: 100, 2024: 200},
        nd_intmig={2020: 2, 2021: 2, 2022: 2, 2023: 2, 2024: 4},
        nd_births=make_series(years_newer, 110),
        nd_deaths=make_series(years_newer, 100),  # NI = 10 for each year in last-5
        nd_domesticmig={2020: -1, 2021: -1, 2022: -1, 2023: 1, 2024: -1},
    )
    _write_minimal_pep_vintage(
        older,
        years=years_older,
        us_pop=make_series(years_older, 1000),
        nd_pop=make_series(years_older, 10),
        us_intmig=make_series(years_older, 100),
        nd_intmig=dict.fromkeys(years_older, 1),  # would imply 1.0% if it were used for 2020
        nd_births=make_series(years_older, 105),
        nd_deaths=make_series(years_older, 100),  # NI = 5 for each year in prior-5
        nd_domesticmig={2015: -1, 2016: -1, 2017: -1, 2018: -1, 2019: -1, 2020: 99},
    )

    summary = derived_stats_module.compute_pep_share_summary(
        [newer, older],
        share_start_year=2015,
    )

    assert summary.latest_year == 2024
    assert summary.share_start_year == 2015
    assert summary.share_end_year == 2024
    assert summary.share_n_years == 10

    # Latest shares (2024): nd_intmig/us_intmig = 4/200 = 2%
    assert summary.nd_share_us_international_migration_pct == pytest.approx(2.0)

    # Check that 2020 uses newer vintage (2/100=2%) rather than older (1/100=1%).
    # Mean shares (2015-2019: 1% each, 2020-2023: 2% each, 2024: 2%) -> (5*1 + 5*2)/10 = 1.5
    assert summary.nd_share_us_international_migration_pct_mean == pytest.approx(1.5)

    # Context stats: prior-5 (2015-2019) NI=5; last-5 (2020-2024) NI=10
    assert summary.natural_increase_mean_prior_window == pytest.approx(5.0)
    assert summary.natural_increase_mean_last_window == pytest.approx(10.0)

    # Domestic migration: 2015-2019 all -1, 2020-2024 = [-1,-1,-1,1,-1] -> 9 negatives of 10
    assert summary.domestic_migration_mean_last_double_window == pytest.approx(
        (-5 + (-3 + 1 - 1)) / 10.0
    )
    assert summary.domestic_migration_negative_years_last_double_window == 9
    assert summary.domestic_migration_years_last_double_window == 10


def test_write_derived_stats_emits_expected_macros(derived_stats_module, tmp_path):
    output = tmp_path / "derived_stats.tex"
    summary = derived_stats_module.PepShareSummary(
        share_start_year=2010,
        share_end_year=2012,
        share_n_years=3,
        latest_year=2012,
        nd_share_us_international_migration_pct=0.25,
        nd_share_us_population_pct=0.30,
        nd_share_us_international_migration_pct_mean=0.20,
        nd_share_us_population_pct_mean=0.29,
        context_window_years=5,
        natural_increase_mean_prior_window=100.0,
        natural_increase_mean_last_window=50.0,
        domestic_migration_mean_last_double_window=-10.0,
        domestic_migration_negative_years_last_double_window=8,
        domestic_migration_years_last_double_window=10,
    )
    derived_stats_module.write_derived_stats(summary, output)
    text = output.read_text(encoding="utf-8")

    assert "\\newcommand{\\PEPLatestYear}{2012}" in text
    assert "\\newcommand{\\PEPShareYearStart}{2010}" in text
    assert "\\newcommand{\\PEPShareYearEnd}{2012}" in text
    assert "\\newcommand{\\PEPShareNYears}{3}" in text
    assert "\\newcommand{\\NDShareUSIntlMigPctLatest}{0.25}" in text
    assert "\\newcommand{\\NDShareUSPopPctLatest}{0.30}" in text
    assert "\\newcommand{\\NDShareUSIntlMigPctMean}{0.20}" in text
    assert "\\newcommand{\\NDShareUSPopPctMean}{0.29}" in text
    assert "\\newcommand{\\PEPContextWindowYears}{5}" in text
    assert "\\newcommand{\\PEPPriorFiveYearStart}{2003}" in text
    assert "\\newcommand{\\PEPPriorFiveYearEnd}{2007}" in text
    assert "\\newcommand{\\PEPLastFiveYearStart}{2008}" in text
    assert "\\newcommand{\\PEPLastFiveYearEnd}{2012}" in text
    assert "\\newcommand{\\PEPLastTenYearStart}{2003}" in text
    assert "\\newcommand{\\PEPLastTenYearEnd}{2012}" in text
    assert "\\newcommand{\\NDNaturalIncreaseMeanPriorFive}{100}" in text
    assert "\\newcommand{\\NDNaturalIncreaseMeanLastFive}{50}" in text
    assert "\\newcommand{\\NDDomesticMigMeanLastTen}{-10}" in text
    assert "\\newcommand{\\NDDomesticMigNegYearsLastTen}{8}" in text
    assert "\\newcommand{\\NDDomesticMigYearsLastTen}{10}" in text
